"""
Object Accumulator — stitch successive RPI reconstructions into one object.

Each RPI frame is a small patch of the sample. If the stage is moving, reading
its position alongside every frame lets the patches be stamped into a large
canvas and averaged where they overlap, turning a stream of disconnected
reconstructions into a real image.

This module is imported lazily by rpi_live_gui.MainWindow._on_open_accumulator,
so none of it — including pyepics — is loaded unless the feature is used.

How it fits together
--------------------
  * The window borrows the main window's validated run inputs
    (snapshot_run_inputs) and starts *its own* worker from WORKER_CLASSES, so
    the main Start button stays out of the picture.
  * It attaches a position_reader to that worker. The worker calls it in its
    own thread right after each diffraction pattern is read, so the position
    is sampled as close in time to the frame as possible.
  * Accumulation happens in the GUI thread on frame_ready — it is a ~200x200
    masked add and costs nothing.
  * Normalisation (obj_sum / contrib over the whole canvas) runs in a separate
    thread and coalesces bursts, so a slow redraw of a large canvas never
    throttles acquisition. Only the final setImage happens on the GUI thread,
    because Qt requires it.

Debugging aids, since this is expected to need shaking out on the beamline:
  * Four position sources: EPICS motor PVs, the quDIS interferometer over the
    qds_bridge (matched to each frame by detector timestamp), and two that
    touch no PVs at all — an H5 replay reading "ppX"/"ppY" (or a ptychi
    result's fractional "positions_px", placed with a subpixel shift)
    alongside an H5 pattern replay, and a fully synthetic raster. The entire pipeline can be
    exercised offline.
  * "Test position read" reports exactly what the configured source returns.
  * Per-frame failures are logged with a traceback and counted; the run
    continues.
  * A timestamped diagnostic log with a "Copy log" button.
  * A one-shot view of the pixel-contribution tracker.
"""

import datetime
import sys
import traceback
from collections import deque
from pathlib import Path

import numpy as np

from PyQt6 import uic
from PyQt6.QtCore import (Qt, QEvent, QMutex, QThread, QTimer, QWaitCondition,
                          pyqtSignal)
from PyQt6.QtGui import QActionGroup
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLayout,
    QLineEdit, QMenuBar, QMessageBox, QPushButton, QSizePolicy, QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg  # after PyQt6, so pyqtgraph binds to it

from pg_image_tools import ImagePlotWidget

from rpi_utils import (
    compute_pixel_sizes,
    read_master_positions,
    scan_tag_from_path,
    illumination_mask,
    process_phase_region,
    save_accumulated_object,
    DEFAULT_PROBE_SHAPE,
    DEFAULT_FILL_FRACTION,
)
from rpi_worker import WORKER_CLASSES
import qds_positions
from layout_utils import compact_group_boxes


_UI_FILE = Path(__file__).parent / 'object_accumulator.ui'

# Position-source combo entries → internal keys (index-aligned).
_SOURCE_LABELS = ['EPICS motor PVs',
                  'QDS interferometer (ch1 = H, ch2 = V)',
                  'H5 file (ppX / ppY, or ptychi positions_px)',
                  'Simulated — no PV reads',
                  'Simulated — all frames at (0, 0)']
_SOURCES       = ['epics', 'qds', 'h5', 'simulated', 'zero']

# The interferometer reports nanometres; positions are handled in microns.
_NM_TO_UM = 1.0e-3

# Position-unit combo entries → multiplier onto microns (index-aligned).
_UNIT_LABELS = ['µm', 'mm', 'm']
_UNITS       = ['um', 'mm', 'm']
_UNIT_TO_UM  = {'um': 1.0, 'mm': 1.0e3, 'm': 1.0e6}
_DEFAULT_UNITS = 'm'

# Where the first accepted frame lands (index-aligned). A corner choice insets
# by half a frame so that frame sits flush against the edge and the scan grows
# inward, rather than half of it falling off the canvas.
_ANCHOR_LABELS = ['Centre', 'Top-left', 'Top-right', 'Bottom-left', 'Bottom-right']
_ANCHORS       = ['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right']

# Display downsample factors offered in the Options menu. None = automatic.
_DOWNSAMPLE_CHOICES = (None, 1, 2, 4, 8, 16)

# Automatic downsampling targets roughly this many pixels on screen. Handing
# pyqtgraph a 20-megapixel array every frame is the single biggest source of
# GUI lag, and nothing is visible at that density anyway.
_TARGET_DISPLAY_PX = 1_500_000

# Position alignment: a phase-correlation correction larger than this is
# treated as a failed correlation rather than a real offset, and discarded.
_DEFAULT_ALIGN_LIMIT_NM = 150.0
# Below this many overlapping pixels there is nothing to correlate against,
# whatever the fraction says.
_MIN_ALIGN_OVERLAP_PX = 64

# ...and below this fraction of the frame's illuminated area, alignment is
# skipped rather than attempted: a sliver of overlap gives a confident-looking
# but meaningless shift. Such a frame goes on its nominal position and is not
# counted as a failure, so the failure policy does not discard it.
_DEFAULT_MIN_OVERLAP_FRAC = 0.30

# Columns of the per-frame record table saved alongside the object. Written
# into the file as an attribute so the table is self-describing.
_RECORD_COLUMNS = [
    'frame_idx', 'source_idx', 'timestamp', 'loss',
    'pos_x_um', 'pos_y_um',
    'nominal_dx_px', 'nominal_dy_px',
    'final_dx_px', 'final_dy_px',
    'align_dx_px', 'align_dy_px',
    'norm_offset_rad',
]

# What to do with a frame whose alignment failed — the correction exceeded the
# limit, or the correlation peak fell below the gate (index-aligned).
_ALIGN_FAIL_LABELS = ['Use nominal position', 'Discard the frame']
_ALIGN_FAIL_MODES  = ['nominal', 'discard']

# Default correlation-peak gate. The metric is the normalised cross-correlation
# of the two images over their overlap, evaluated at the fitted shift: 1.0 is a
# perfect match, 0.0 is noise. Off by default.
_DEFAULT_ALIGN_PEAK = 0.30

# Text lines reserved for each status label. Their height is pinned to this so
# changing status text can never resize the window.
# Hard cap on a diagnostic log entry. Longer detail goes to stdout.
_LOG_MAX_CHARS = 84

# Start is green while idle and Stop red while running, as in the main window.
_START_BTN_STYLE = 'QPushButton { background-color: #2e9e44; color: white; font-weight: bold; }'
_STOP_BTN_STYLE  = 'QPushButton { background-color: #c0392b; color: white; font-weight: bold; }'

_RESERVED_LINES = {
    'canvas_info_label':  2,
    'monitor_label':      6,
    'accum_cursor_label': 1,
}

# Illuminated area of each frame is stamped at this multiple of the probe's
# filling fraction. Tune here rather than in the UI.
_ACCUM_STAMP_FUDGE = 1.0

# How a new frame combines with what is already on the canvas (index-aligned).
# Every mode keeps the invariant that the displayed/saved image is
# obj_sum / pixel_contribution wherever the tracker is non-zero.
_MODE_LABELS = [
    'Average overlapping images',
    'Replace with newest',
    'Fill empty pixels only',
    'Decaying average (newest weighted most)',
]
_MODES = ['average', 'replace', 'fill_empty', 'decay']
_DECAY_MODE_INDEX = _MODES.index('decay')

# Decaying average: once a pixel's weight has fallen this low it stops being
# scaled, so a patch that is never revisited holds its value indefinitely
# instead of decaying into float underflow. Both arrays are frozen together,
# so the ratio — the actual image — is untouched either way.
_DECAY_FLOOR = 1e-3

# Guard rails on the canvas allocation, so a mis-typed pixel size can't try to
# allocate a terabyte. Bytes per canvas pixel = float32 object + the tracker.
_MAX_CANVAS_SIDE  = 12000
_BYTES_PER_PIXEL  = {'float32': 4 + 4, 'uint8': 4 + 1}
_MAX_CANVAS_BYTES = 2 * 1024 ** 3

_PV_CONNECT_TIMEOUT_S = 3.0

# Position file written alongside each scan; looked for next to the H5 being
# replayed when the window opens.
_POS_H5_GLOB = '*_para.hdf5'

# Positions taken from a ptychi reconstruction are fractional pixels. They are
# rounded to this many decimals and the remainder applied as a Fourier shift.
_SUBPIXEL_DECIMALS = 2

# Accumulated-position overlay on the object image.
_POS_MARKER_SIZE  = 3
_POS_MARKER_COLOR = 'r'


# ══════════════════════════════════════════════════════════════════════════
# Position sources
# ══════════════════════════════════════════════════════════════════════════

def _subpixel_shift(arr, dy, dx):
    """arr moved by (dy, dx) pixels (positive = toward higher indices), by
    the Fourier shift theorem. Wraps at the edges, which the illumination
    mask keeps out of the stamp."""
    ny, nx = arr.shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    ramp = np.exp(-2j * np.pi * (ky * dy + kx * dx))
    return np.fft.ifft2(np.fft.fft2(arr) * ramp).astype(arr.dtype, copy=False)


class PositionSourceError(RuntimeError):
    """Raised with a message intended to be shown to the user verbatim."""


class _PositionSource:
    """Reads one (horizontal, vertical) stage position per frame, in microns.

    The three backends share the unit scaling / inversion / axis-swap stage so
    that a sign or axis mix-up can be fixed in the UI rather than in code.
    """

    def __init__(self, kind, h_pv='', v_pv='', h5_path='', units='um',
                 invert_h=False, invert_v=False, swap=False,
                 sim_step_um=1.0, sim_extent_um=(30.0, 30.0),
                 qds_host=qds_positions.DEFAULT_HOST,
                 qds_port=qds_positions.DEFAULT_PORT,
                 qds_maxlen=qds_positions.DEFAULT_MAXLEN,
                 qds_every_n=qds_positions.DEFAULT_EVERY_N,
                 qds_offset_s=0.0):
        self.kind      = kind
        self.h_pv      = h_pv.strip()
        self.v_pv      = v_pv.strip()
        self.h5_path   = h5_path.strip()
        # The interferometer always reports nm, whatever the units combo says.
        self.scale     = (_NM_TO_UM if kind == 'qds'
                          else _UNIT_TO_UM.get(units, 1.0))
        self.invert_h  = invert_h
        self.invert_v  = invert_v
        self.swap      = swap

        self._sim_step   = max(float(sim_step_um), 1e-3)
        self._sim_extent = sim_extent_um
        self._sim_i      = 0

        self._pvs      = None
        self._h5_fh    = None
        self._ppx      = None
        self._ppy      = None
        self._counter  = 0
        self._last_h5_index = None
        # 'ppx' (ppX / ppY in the units combo's units), 'ptychi'
        # (positions_px × obj_pixel_size_m) or 'master' (a raw master file's
        # /entry/sample/positions in its own units), decided at open.
        self.h5_format = ''
        self._ptychi_pixel_m = None
        self._master_units   = ''
        # Fractional pixel positions should be placed with a subpixel shift
        # rather than rounded. True for ptychi positions.
        self.subpixel  = False

        self.qds_host     = qds_host
        self.qds_port     = int(qds_port)
        self.qds_maxlen   = int(qds_maxlen)
        self.qds_every_n  = int(qds_every_n)
        # Seconds added to the detector timestamp before matching: the sample
        # used is the one nearest to t_detector + offset.
        self.qds_offset_s = float(qds_offset_s)
        self._qds         = None
        self._qds_last_dt = None    # t_match − target of the last match, s

    # -- lifecycle ----------------------------------------------------------

    def open(self):
        """Connect / open. Raises PositionSourceError with a usable message."""
        if self.kind == 'epics':
            self._open_epics()
        elif self.kind == 'qds':
            self._open_qds()
        elif self.kind == 'h5':
            self._open_h5()
        # 'simulated' and 'zero' need nothing

    def _open_epics(self):
        if not self.h_pv or not self.v_pv:
            raise PositionSourceError(
                'Both the horizontal and the vertical motor PV must be set.')
        try:
            from epics import PV
        except ImportError as exc:
            raise PositionSourceError(
                f'pyepics is not available, so EPICS positions cannot be '
                f'read ({exc}). Use the H5 or simulated source instead.')
        pvs = {}
        unreachable = []
        for axis, name in (('h', self.h_pv), ('v', self.v_pv)):
            pv = PV(name)
            if not pv.wait_for_connection(timeout=_PV_CONNECT_TIMEOUT_S):
                unreachable.append(name)
            pvs[axis] = pv
        if unreachable:
            raise PositionSourceError(
                'Could not connect to position PV(s) within '
                f'{_PV_CONNECT_TIMEOUT_S:g} s: ' + ', '.join(unreachable))
        self._pvs = pvs

    def _open_qds(self):
        if not qds_positions.QDS_AVAILABLE:
            raise PositionSourceError(qds_positions.QDS_UNAVAILABLE_REASON)
        qds = qds_positions.QDSHistory(self.qds_host, self.qds_port,
                                       self.qds_maxlen, self.qds_every_n)
        qds.start()
        if not qds.wait_first(_PV_CONNECT_TIMEOUT_S):
            qds.stop()
            if qds.connect_error:
                why = f'connect failed: {qds.connect_error}'
            elif qds.untimed:
                why = (f'{qds.untimed} sample(s) arrived without timestamps — '
                       f'the publisher predates them; update and restart it')
            else:
                why = (f'no samples within {_PV_CONNECT_TIMEOUT_S:g} s. '
                       f'{qds_positions.PUBLISHER_HINT}')
            raise PositionSourceError(f'QDS bridge {qds.endpoint}: {why}')
        self._qds = qds

    def set_qds_maxlen(self, n):
        self.qds_maxlen = int(n)
        if self._qds is not None:
            self._qds.set_maxlen(n)

    def set_qds_every_n(self, n):
        self.qds_every_n = int(n)
        if self._qds is not None:
            self._qds.set_every_n(n)

    def _open_h5(self):
        if not self.h5_path:
            raise PositionSourceError(
                'Select an H5 file containing "ppX" and "ppY", '
                '"positions_px" and "obj_pixel_size_m" (a ptychi result), or '
                'a raw master file with /entry/sample/positions.')
        if not Path(self.h5_path).is_file():
            raise PositionSourceError(f'"{self.h5_path}" is not a file.')
        import h5py
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if 'ppX' in f and 'ppY' in f:
                    ppx = np.asarray(f['ppX'][()], dtype=np.float64).ravel()
                    ppy = np.asarray(f['ppY'][()], dtype=np.float64).ravel()
                    self.h5_format = 'ppx'
                elif 'positions_px' in f and 'obj_pixel_size_m' in f:
                    ppx, ppy = self._read_ptychi_positions(f)
                else:
                    master = read_master_positions(f)
                    if master is None:
                        raise PositionSourceError(
                            f'{self.h5_path} has neither "ppX" + "ppY", '
                            f'"positions_px" + "obj_pixel_size_m", nor a raw '
                            f'master file\'s /entry/sample/positions. '
                            f'Available keys: {list(f.keys())}')
                    ppx, ppy = self._use_master_positions(*master)
        except PositionSourceError:
            raise
        except Exception as exc:
            raise PositionSourceError(f'Could not read {self.h5_path}: {exc}')
        if ppx.size == 0 or ppy.size == 0:
            raise PositionSourceError(f'{self.h5_path}: ppX/ppY are empty.')
        if ppx.size != ppy.size:
            raise PositionSourceError(
                f'{self.h5_path}: ppX has {ppx.size} entries but ppY has '
                f'{ppy.size}. They must match.')
        self._ppx, self._ppy = ppx, ppy

    def _read_ptychi_positions(self, f):
        """(x_um, y_um) from a ptychi reconstruction's positions_px, an
        (N, 2) array of (row, col) = (y, x) in object pixels. Converted to
        microns here, so the units combo does not apply; invert / swap do."""
        pos = np.asarray(f['positions_px'][()], dtype=np.float64)
        pix_m = float(np.asarray(f['obj_pixel_size_m'][()]).ravel()[0])
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise PositionSourceError(
                f'{self.h5_path}: positions_px has shape {pos.shape}; '
                f'expected (N, 2).')
        if not pix_m > 0:
            raise PositionSourceError(
                f'{self.h5_path}: obj_pixel_size_m is {pix_m}.')
        self.h5_format       = 'ptychi'
        self._ptychi_pixel_m = pix_m
        self.scale           = 1.0
        self.subpixel        = True
        return pos[:, 1] * pix_m * 1e6, pos[:, 0] * pix_m * 1e6

    def _use_master_positions(self, h, v, units):
        """Positions from a raw master file. Its 'units' attribute sets the
        scale, so the units combo does not apply; invert / swap do."""
        if units not in _UNIT_TO_UM:
            raise PositionSourceError(
                f'{self.h5_path}: /entry/sample/positions has units '
                f'"{units}"; expected one of {", ".join(_UNITS)}.')
        self.h5_format     = 'master'
        self._master_units = units
        self.scale         = _UNIT_TO_UM[units]
        return h, v

    def close(self):
        self._pvs = None
        self._ppx = self._ppy = None
        if self._qds is not None:
            self._qds.stop()
            self._qds = None

    # -- reading ------------------------------------------------------------

    def read(self, index=None, frame_ts=None):
        """Return (x_um, y_um). `index` is the H5 pattern index, or None when
        the images come from a PV. `frame_ts` is the detector timestamp in
        Unix-epoch seconds, or None; only the QDS source uses it. Raises
        PositionSourceError on failure."""
        if self.kind == 'epics':
            h, v = self._read_epics()
        elif self.kind == 'qds':
            h, v = self._read_qds(frame_ts)
        elif self.kind == 'h5':
            h, v = self._read_h5(index)
        elif self.kind == 'zero':
            # Every frame on the same spot; no transform, so it stays (0, 0).
            return 0.0, 0.0
        else:
            h, v = self._read_simulated(index)
        h, v = self._transform(h, v)
        return float(h), float(v)

    def read_indexed(self, index=None, frame_ts=None):
        """read(), plus the ppX/ppY index used (None unless an H5 source).

        Used as the worker's position hook so the index travels with the
        frame — read back later from the source, it could already belong to
        the next frame."""
        x, y = self.read(index, frame_ts)
        return x, y, (self._last_h5_index if self.kind == 'h5' else None)

    def read_all(self):
        """Every position in an open H5 source, as (x_um, y_um) arrays with
        the same scaling / inversion / swap as read()."""
        if self.kind != 'h5' or self._ppx is None:
            raise PositionSourceError('Only an open H5 source has a full '
                                      'position list.')
        return self._transform(self._ppx.copy(), self._ppy.copy())

    def _transform(self, h, v):
        """Raw units → microns, then inversion and axis swap. Works on
        scalars and arrays alike."""
        h = h * self.scale
        v = v * self.scale
        if self.invert_h:
            h = -h
        if self.invert_v:
            v = -v
        if self.swap:
            h, v = v, h
        return h, v

    def _read_epics(self):
        if self._pvs is None:
            raise PositionSourceError('Position PVs are not connected.')
        # pyepics' documented recipe for using PVs created in another thread.
        try:
            import epics
            epics.ca.use_initial_context()
        except Exception:
            pass
        out = {}
        for axis, pv in self._pvs.items():
            value = pv.get(timeout=_PV_CONNECT_TIMEOUT_S)
            if value is None:
                raise PositionSourceError(
                    f'{pv.pvname} returned no value (connected='
                    f'{pv.connected}).')
            out[axis] = float(value)
        return out['h'], out['v']

    def _read_qds(self, frame_ts):
        """The interferometer sample nearest frame_ts + offset, in nm. With no
        frame_ts (H5 replay, a test read) there is nothing to match, so the
        newest sample is used."""
        if self._qds is None:
            raise PositionSourceError('QDS bridge is not connected.')
        if frame_ts is None:
            newest = self._qds.newest()
            if newest is None:
                raise PositionSourceError('QDS history is empty.')
            self._qds_last_dt = None
            return newest[1], newest[2]
        target = frame_ts + self.qds_offset_s
        match = self._qds.closest(target)
        if match is None:
            raise PositionSourceError('QDS history is empty.')
        t, h, v, is_oldest, n = match
        self._qds_last_dt = t - target
        if is_oldest:
            n_stored, oldest, newest = self._qds.stats()
            span = (f'{oldest - target:+.3f} .. {newest - target:+.3f} s'
                    if n_stored else 'empty')
            print(f'WARNING: QDS position match is the oldest of {n} stored '
                  f'samples (history spans {span} around the frame), so the '
                  f'right sample may already have been discarded. Lengthen '
                  f'the QDS history, keep fewer samples, or check the '
                  f'detector-position timestamp offset '
                  f'({self.qds_offset_s:+g} s).', flush=True)
        return h, v

    def _read_h5(self, index):
        if self._ppx is None:
            raise PositionSourceError('Position H5 is not open.')
        if index is None:
            # Live images with H5 positions: fall back to our own counter so
            # the two at least advance together.
            index = self._counter
            self._counter += 1
        i = int(index) % self._ppx.size
        self._last_h5_index = i
        return self._ppx[i], self._ppy[i]

    def _read_simulated(self, index):
        """Deterministic snake raster across the canvas; no PVs touched."""
        if index is None:
            index = self._sim_i
            self._sim_i += 1
        sx, sy = self._sim_extent
        # Stay inside ~80% of the canvas so edge clipping isn't the first
        # thing a test run runs into.
        ncols = max(1, int((0.8 * sx) // self._sim_step))
        row, col = divmod(int(index), ncols)
        if row % 2:
            col = ncols - 1 - col          # serpentine
        nrows = max(1, int((0.8 * sy) // self._sim_step))
        row %= nrows
        h = (col - (ncols - 1) / 2.0) * self._sim_step
        v = (row - (nrows - 1) / 2.0) * self._sim_step
        # Undo the transform the caller is about to apply, so simulated
        # positions land where they claim regardless of the invert/swap boxes.
        if self.swap:
            h, v = v, h
        if self.invert_h:
            h = -h
        if self.invert_v:
            v = -v
        return h / self.scale, v / self.scale

    def describe(self):
        """One short line naming the source."""
        if self.kind == 'epics':
            return f'EPICS {self.h_pv} / {self.v_pv}'
        if self.kind == 'qds':
            return f'QDS {self.qds_host}:{self.qds_port} ch1/ch2'
        if self.kind == 'h5':
            n = 0 if self._ppx is None else self._ppx.size
            if self.h5_format == 'ptychi':
                return f'H5 ptychi positions_px n={n}'
            if self.h5_format == 'master':
                return f'H5 master /entry/sample/positions n={n}'
            return f'H5 ppX/ppY n={n}'
        if self.kind == 'zero':
            return 'simulated, all frames at (0, 0)'
        return f'simulated raster, step {self._sim_step:.3f} um'

    def detail(self):
        """Extra lines for the Test-read readout. Must be called while open."""
        if self.kind == 'h5':
            if self._ppx is None:
                return ['not open']
            if self.h5_format == 'ptychi':
                return [f'file {Path(self.h5_path).name}',
                        f'ptychi positions_px n={self._ppx.size}, '
                        f'{self._ptychi_pixel_m * 1e9:.3f} nm/px',
                        f'x  [{self._ppx.min():+.4g}, {self._ppx.max():+.4g}] um',
                        f'y  [{self._ppy.min():+.4g}, {self._ppy.max():+.4g}] um',
                        f'(units combo ignored; subpixel placement on)']
            if self.h5_format == 'master':
                return [f'file {Path(self.h5_path).name}',
                        f'/entry/sample/positions n={self._ppx.size}, '
                        f'units {self._master_units}',
                        f'X  [{self._ppx.min():.4g}, {self._ppx.max():.4g}]',
                        f'Z  [{self._ppy.min():.4g}, {self._ppy.max():.4g}]',
                        '(ranges in file units; units combo ignored)']
            return [f'file {Path(self.h5_path).name}',
                    f'ppX  {self._ppx.shape} {self._ppx.dtype}'
                    f'  [{self._ppx.min():.4g}, {self._ppx.max():.4g}]',
                    f'ppY  {self._ppy.shape} {self._ppy.dtype}'
                    f'  [{self._ppy.min():.4g}, {self._ppy.max():.4g}]',
                    '(ranges in raw file units)']
        if self.kind == 'epics':
            return [f'H  {self.h_pv}', f'V  {self.v_pv}']
        if self.kind == 'qds':
            if self._qds is None:
                return ['not open']
            n, _oldest, newest = self._qds.stats()
            lines = [f'history {n}/{self._qds.maxlen}, every '
                     f'{self._qds.every_n}, offset {self.qds_offset_s:+g} s']
            if newest is not None:
                lines.append(f'newest sample age '
                             f'{qds_positions.age_s(newest) * 1e3:+.1f} ms')
            if self._qds_last_dt is not None:
                lines.append(f'last match dt {self._qds_last_dt * 1e3:+.1f} ms')
            return lines
        if self.kind == 'zero':
            return ['every frame at (0, 0) um, no PVs read']
        return [f'step {self._sim_step:.3f} um, no PVs read']


# ══════════════════════════════════════════════════════════════════════════
# Background normalisation
# ══════════════════════════════════════════════════════════════════════════

class _RenderWorker(QThread):
    """Prepares everything the display needs, off the GUI thread.

    Not just the obj_sum / contrib division: also the colour levels, the
    coverage statistics and (when its window is open) the contribution map.
    All of those are full-canvas passes, and on a 4000x4000 object each one
    costs tens of milliseconds — done in the GUI thread they show up directly
    as lag. What is left for the GUI thread is the one thing Qt insists on
    doing there: the setImage raster.

    Requests coalesce: if several frames land while a pass is in flight, only
    one more redraw follows. Acquisition therefore stays bottlenecked by
    EPICS and the reconstruction, never by the display.
    """

    # dict: image, levels, tracker, tracker_levels, coverage, max_weight
    rendered = pyqtSignal(object)

    def __init__(self, data_mutex, parent=None):
        super().__init__(parent)
        self._data_mutex = data_mutex      # guards obj_sum / contrib
        self._wait_mutex = QMutex()
        self._wait_cond  = QWaitCondition()
        self._dirty      = False
        self._abort      = False
        self._downsample = 1
        self._want_tracker = False
        self._obj_sum    = None
        self._contrib    = None
        self._buffer     = None            # reused output, keyed by shape

    def set_arrays(self, obj_sum, contrib):
        self._wait_mutex.lock()
        self._obj_sum = obj_sum
        self._contrib = contrib
        self._buffer  = None
        self._wait_mutex.unlock()

    def set_want_tracker(self, want):
        self._wait_mutex.lock()
        self._want_tracker = bool(want)
        self._wait_mutex.unlock()

    def request(self, downsample=1):
        self._wait_mutex.lock()
        self._downsample = max(1, int(downsample))
        self._dirty = True
        self._wait_cond.wakeOne()
        self._wait_mutex.unlock()

    def stop(self):
        self._wait_mutex.lock()
        self._abort = True
        self._wait_cond.wakeAll()
        self._wait_mutex.unlock()
        self.wait(3000)

    def run(self):
        while True:
            self._wait_mutex.lock()
            while not self._dirty and not self._abort:
                self._wait_cond.wait(self._wait_mutex)
            if self._abort:
                self._wait_mutex.unlock()
                return
            self._dirty = False
            ds      = self._downsample
            want_tr = self._want_tracker
            obj_sum = self._obj_sum
            contrib = self._contrib
            self._wait_mutex.unlock()

            if obj_sum is None or contrib is None:
                continue

            payload = None
            self._data_mutex.lock()
            try:
                o = obj_sum[::ds, ::ds] if ds > 1 else obj_sum
                c = contrib[::ds, ::ds] if ds > 1 else contrib
                if self._buffer is None or self._buffer.shape != o.shape:
                    self._buffer = np.zeros(o.shape, dtype=np.float32)
                out = self._buffer
                out.fill(0.0)
                np.divide(o, c, out=out, where=c > 0)
                img = out.copy()      # the GUI thread gets its own snapshot
                tracker = (np.asarray(c, dtype=np.float32).copy()
                           if want_tr else None)
                # Full-canvas statistics, deliberately computed here rather
                # than in _update_monitor where they would block the GUI.
                covered   = int(np.count_nonzero(contrib))
                total     = int(contrib.size)
                max_w     = float(contrib.max()) if total else 0.0
            except Exception:
                self._data_mutex.unlock()
                continue
            self._data_mutex.unlock()

            lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                lo, hi = 0.0, max(hi if np.isfinite(hi) else 1.0, lo + 1e-9)
            tr_levels = None
            if tracker is not None:
                tr_hi = float(tracker.max())
                tr_levels = (0.0, tr_hi if tr_hi > 0 else 1.0)

            payload = {
                'image': img, 'levels': (lo, hi),
                'tracker': tracker, 'tracker_levels': tr_levels,
                'coverage': (100.0 * covered / total) if total else 0.0,
                'max_weight': max_w, 'downsample': ds,
            }
            self.rendered.emit(payload)


# ══════════════════════════════════════════════════════════════════════════
# Main window
# ══════════════════════════════════════════════════════════════════════════

class ObjectAccumulatorWindow(QDialog):

    def __init__(self, main_window):
        super().__init__(main_window)
        uic.loadUi(str(_UI_FILE), self)
        self.setObjectName('ObjectAccumulatorWindow')
        compact_group_boxes(self)
        self.start_btn.setStyleSheet(_START_BTN_STYLE)

        # pg_image_tools viewer (lineout, measure, filters, compare, ... in the
        # right-click menu). self.accum_image_view stays the bare pg.ImageView.
        self.accum_plot = ImagePlotWidget(self.accum_plot_container, keep_points=True)
        self.accum_image_view = self.accum_plot.image_view
        self.accum_image_view.setObjectName('accum_image_view')
        self.accum_plot.action_no_zoom_tracking.setChecked(True)
        # A real top-level window: non-modal, with its own minimise/maximise.
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setModal(False)

        self._main = main_window

        # Run state
        self._worker        = None
        self._source        = None
        self._obj_sum       = None
        self._contrib       = None
        self._contrib_max   = None    # saturation ceiling for an integer tracker
        self._anchor_rc     = None    # canvas (row, col) of the first frame
        self._tracker_win   = None    # live contribution map, when open
        self._displayed_ds  = 1       # downsample factor of what is on screen
        self._decay_mask    = None    # reused buffer, avoids a per-frame alloc
        self._align_limit_nm = _DEFAULT_ALIGN_LIMIT_NM
        self._align_peak_min = _DEFAULT_ALIGN_PEAK
        self._align_min_overlap = _DEFAULT_MIN_OVERLAP_FRAC
        self._last_align_peak = float('nan')
        self._last_align_overlap = float('nan')
        self._n_align_ok = self._n_align_rejected = self._n_skip_align = 0
        self._n_align_nooverlap = 0
        self._n_skip_nothing = 0
        self._warned_align = False
        self._warned_nothing = False
        self._warned_decay  = False
        self._warned_saturate = False
        self._origin_um     = None    # position of the first accepted frame
        self._records       = []      # one row per accepted frame
        self._scan_tag      = None    # e.g. 'S0082', from the H5 being replayed
        self._run_started   = None
        self._res_rpi_m     = None
        self._stamp_mask    = None
        self._probe_shape   = DEFAULT_PROBE_SHAPE
        self._fill_fraction = DEFAULT_FILL_FRACTION
        self._run_params    = None
        self._closing       = False
        self._qds_maxlen    = qds_positions.DEFAULT_MAXLEN
        self._qds_every_n   = qds_positions.DEFAULT_EVERY_N
        self._qds_offset_s  = 0.0

        # Counters shown on the monitor label
        self._n_in = self._n_accepted = 0
        self._n_skip_loss = self._n_skip_pos = self._n_skip_bounds = 0
        self._n_errors = 0
        self._n_align_ok = self._n_align_rejected = self._n_skip_align = 0
        self._n_align_nooverlap = 0
        self._n_skip_nothing = 0
        self._last_align_peak = float('nan')
        self._last_align_overlap = float('nan')
        self._warned_align = False
        self._warned_nothing = False
        self._decay_mask = None
        self._last_loss = float('nan')
        self._last_pos_um = None
        self._last_pos_px = None
        self._last_pos_idx = None     # ppX/ppY index of the last position
        self._coverage = 'cover —'
        self._worker_status = ''
        self._frames_since_redraw = 0

        # obj_sum / contrib are touched by the GUI thread (stamping) and the
        # render thread (normalising). The stamp is ~200x200 so the render
        # thread is effectively never held up.
        self._data_mutex = QMutex()
        self._render = _RenderWorker(self._data_mutex, self)
        self._render.rendered.connect(self._on_rendered)
        self._render.start()

        # The stage-position controls live in their own dialog. Its widgets
        # are aliased here so the rest of the window addresses them exactly
        # as it did when they sat on the panel.
        self._pos_dlg = _StagePositionsDialog(self)
        for name in _StagePositionsDialog.WIDGETS:
            setattr(self, name, getattr(self._pos_dlg, name))
        self._pos_scatter_win = None   # Test-read plot of every H5 position
        self._pos_autofound   = None   # *_para.hdf5 picked up at open, if any

        self.pos_source_combo.addItems(_SOURCE_LABELS)
        if not qds_positions.QDS_AVAILABLE:
            item = self.pos_source_combo.model().item(_SOURCES.index('qds'))
            item.setEnabled(False)
            item.setToolTip(qds_positions.QDS_UNAVAILABLE_REASON)
        self.motor_units_combo.addItems(_UNIT_LABELS)
        self.accum_mode_combo.addItems(_MODE_LABELS)
        self.first_frame_combo.addItems(_ANCHOR_LABELS)

        self._build_menu()
        self._freeze_layout()
        self._load_settings()
        self._connect_signals()
        self._on_source_changed()
        self._on_tracker_dtype_changed()   # may force the mode off 'decay'
        self._on_mode_changed()
        self._update_canvas_info()

        # The menu's "Auto colorscale" and the auto-levels checkbox are one
        # setting. setChecked() doesn't emit triggered, so this can't loop.
        auto = self.accum_plot.action_auto_colorscale
        auto.setChecked(self.accum_autolevel_check.isChecked())
        auto.triggered.connect(self.accum_autolevel_check.setChecked)
        self.accum_autolevel_check.toggled.connect(auto.setChecked)

        vb = self.accum_image_view.getView()
        self._accum_vb = vb
        self._first_render = True
        self._cursor_proxy = pg.SignalProxy(
            vb.scene().sigMouseMoved, rateLimit=30, slot=self._on_cursor_moved)
        self._pos_scatter = pg.ScatterPlotItem(
            size=_POS_MARKER_SIZE, pen=None, brush=pg.mkBrush(_POS_MARKER_COLOR))
        self._pos_scatter.setZValue(10)
        self._pos_scatter.setVisible(self.show_positions_check.isChecked())
        vb.addItem(self._pos_scatter)

        self._log('Ready. Tracebacks print to stdout.')
        if self._pos_autofound:
            self._log(f'Position H5: found {Path(self._pos_autofound).name}.')

    # ------------------------------------------------------------------ #
    # Layout / menu                                                       #
    # ------------------------------------------------------------------ #

    def _freeze_layout(self):
        """Stop the window resizing itself as status text changes length.

        Three things caused that: a dialog layout's default size constraint
        pulls the window out to its sizeHint; the monospace status labels
        report a width proportional to their current text; and their height
        changes with the number of lines. So each one is ignored horizontally
        and pinned to a reserved line count vertically. The image gets
        everything that is left.
        """
        self.layout().setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        for name in _RESERVED_LINES:
            lb = getattr(self, name, None)
            if lb is None:        # label dropped from the .ui
                continue
            lb.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            lb.setWordWrap(False)
            lb.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._apply_label_heights()
        # The controls column keeps its width; everything else goes to the image.
        self.accum_splitter.setStretchFactor(0, 0)
        self.accum_splitter.setStretchFactor(1, 1)
        # Diagnostics start small and can be dragged right down to nothing.
        self.right_splitter.setStretchFactor(0, 1)
        self.right_splitter.setStretchFactor(1, 0)
        self._sized_splitters = False

    def showEvent(self, event):
        # setSizes() before the first show has nothing to divide up — the
        # splitter has no width yet — so the proportions have to be applied
        # once the real geometry exists.
        super().showEvent(event)
        if not self._sized_splitters:
            self._sized_splitters = True
            # Controls column as wide as its contents need, so it never
            # scrolls sideways.
            w = max(self.accum_splitter.width(), 600)
            left = (self.accum_scroll_contents.minimumSizeHint().width()
                    + self.accum_scroll.verticalScrollBar().sizeHint().width()
                    + 2 * self.accum_scroll.frameWidth())
            self.accum_splitter.setSizes([left, w - left])
            h = max(self.right_splitter.height(), 400)
            self.right_splitter.setSizes([int(h * 0.82), h - int(h * 0.82)])
            self._apply_label_heights()

    def _apply_label_heights(self):
        """Reserve a fixed number of text lines for each status label, so its
        height never changes with its content."""
        for name, n_lines in _RESERVED_LINES.items():
            lb = getattr(self, name, None)
            if lb is None:
                continue
            lb.setFixedHeight(lb.fontMetrics().lineSpacing() * n_lines + 4)

    def changeEvent(self, event):
        # The font size is driven from the main window, and the reserved
        # heights are in pixels — recompute them whenever it changes.
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            QTimer.singleShot(0, self._apply_label_heights)

    def _build_menu(self):
        """Options that are set once and then forgotten live in a menu rather
        than taking up panel space."""
        bar = QMenuBar(self)
        bar.setObjectName('accum_menubar')
        menu = bar.addMenu('&Options')
        menu.setObjectName('accum_options_menu')

        def _toggle(text, tip):
            a = menu.addAction(text)
            a.setCheckable(True)
            a.setToolTip(tip)
            return a

        self._act_invert_h = _toggle(
            'Invert &horizontal',
            'Flip the sign of the horizontal position. Stage motion and '
            'beam-on-sample motion are often opposite.')
        self._act_invert_v = _toggle(
            'Invert &vertical', 'Flip the sign of the vertical position.')
        self._act_swap = _toggle(
            '&Swap H/V', 'Swap horizontal and vertical. Quick fix when the '
                         'stitch comes out transposed.')
        menu.addSeparator()

        qds_menu = menu.addMenu('&QDS interferometer')
        qds_menu.setObjectName('accum_qds_menu')
        qds_menu.setToolTip('Settings for the QDS interferometer position '
                            'source. Changes apply immediately, even mid-run.')
        a = qds_menu.addAction('History &length…')
        a.setToolTip('How many timestamped positions are kept to match '
                     'frames against.')
        a.triggered.connect(self._on_set_qds_maxlen)
        a = qds_menu.addAction('Keep every &Nth position…')
        a.setToolTip('Store only every Nth received position, so the history '
                     'covers N times longer without growing.')
        a.triggered.connect(self._on_set_qds_every_n)
        a = qds_menu.addAction('Detector-position timestamp &offset…')
        a.setToolTip('Seconds added to the detector timestamp before looking '
                     'up the nearest position. Positive if the interferometer '
                     'clock runs ahead of the detector IOC clock.')
        a.triggered.connect(self._on_set_qds_offset)
        menu.addSeparator()
        self._act_uint8 = _toggle(
            '&uint8 contribution tracker',
            'Store the tracker as uint8: 5 bytes per canvas pixel instead of '
            '8. Counts saturate at 255 and the decaying average is '
            'unavailable. Fixed once a run starts.')
        menu.addSeparator()
        self._act_align_limit = menu.addAction('&Alignment limit…')
        self._act_align_limit.setToolTip(
            'Largest position correction the phase-correlation alignment may '
            'apply. Anything bigger is treated as a failed correlation.')
        self._act_align_limit.triggered.connect(self._on_set_align_limit)

        fail_menu = menu.addMenu('Alignment &failure')
        fail_menu.setObjectName('accum_align_fail_menu')
        fail_menu.setToolTip('What to do with a frame whose alignment failed.')
        self._align_fail_group = QActionGroup(self)
        self._align_fail_group.setExclusive(True)
        self._align_fail_actions = {}
        for key, label in zip(_ALIGN_FAIL_MODES, _ALIGN_FAIL_LABELS):
            a = fail_menu.addAction(label)
            a.setCheckable(True)
            a.setData(key)
            self._align_fail_group.addAction(a)
            self._align_fail_actions[key] = a
        self._align_fail_actions['nominal'].setChecked(True)

        self._act_align_overlap = menu.addAction('Minimum &overlap…')
        self._act_align_overlap.setToolTip(
            'Skip alignment (and use the nominal position) when the new '
            'frame overlaps the existing object by less than this fraction '
            'of its illuminated area.')
        self._act_align_overlap.triggered.connect(self._on_set_align_overlap)

        self._act_align_gate = menu.addAction('Gate on correlation &peak')
        self._act_align_gate.setCheckable(True)
        self._act_align_gate.setToolTip(
            'Also require the correlation peak to clear a threshold. The '
            'metric is the normalised cross-correlation over the overlap at '
            'the fitted shift: 1.0 is a perfect match, 0.0 is noise.')
        self._act_align_peak = menu.addAction('Correlation peak &threshold…')
        self._act_align_peak.triggered.connect(self._on_set_align_peak)
        menu.addSeparator()

        ds_menu = menu.addMenu('&Downsample display')
        ds_menu.setObjectName('accum_downsample_menu')
        self._ds_group = QActionGroup(self)
        self._ds_group.setExclusive(True)
        self._ds_actions = {}
        for v in _DOWNSAMPLE_CHOICES:
            a = ds_menu.addAction('Auto' if v is None
                                  else ('None (1×)' if v == 1 else f'{v}×'))
            a.setCheckable(True)
            a.setData(v)
            if v is None:
                a.setToolTip(f'Pick a factor that keeps the displayed image '
                             f'near {_TARGET_DISPLAY_PX // 1000}k pixels.')
            self._ds_group.addAction(a)
            self._ds_actions[v] = a
        self._ds_actions[None].setChecked(True)

        self.layout().setMenuBar(bar)
        self._menubar = bar

    # ------------------------------------------------------------------ #
    # Wiring                                                              #
    # ------------------------------------------------------------------ #

    def _connect_signals(self):
        self.pos_source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.accum_mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._act_uint8.toggled.connect(self._on_tracker_dtype_changed)
        self.setup_positions_btn.clicked.connect(self._on_setup_positions)
        self.pos_h5_browse_btn.clicked.connect(self._on_browse_pos_h5)
        self.test_read_btn.clicked.connect(self._on_test_read)
        self.show_positions_check.toggled.connect(self._on_show_positions_toggled)
        self.size_x_um_spin.valueChanged.connect(self._update_canvas_info)
        self.size_y_um_spin.valueChanged.connect(self._update_canvas_info)
        self.start_btn.clicked.connect(self._on_start_stop)
        self.refresh_btn.clicked.connect(self._on_refresh_object)
        self.save_btn.clicked.connect(self._on_save_object)
        self.show_tracker_btn.clicked.connect(self._on_show_tracker)
        self.close_btn.clicked.connect(self.close)
        self.copy_log_btn.clicked.connect(self._on_copy_log)
        self.clear_log_btn.clicked.connect(self.log_text.clear)

    def _on_source_changed(self):
        kind = self._current_source_kind()
        is_epics = kind == 'epics'
        is_h5    = kind == 'h5'
        for w in (self.label_h_pv, self.h_motor_pv_edit,
                  self.label_v_pv, self.v_motor_pv_edit):
            w.setEnabled(is_epics)
        for w in (self.label_pos_h5, self.pos_h5_file_edit, self.pos_h5_browse_btn):
            w.setEnabled(is_h5)
        # Simulated positions are generated in microns already, and the
        # interferometer always reports nm.
        for w in (self.label_motor_units, self.motor_units_combo):
            w.setEnabled(kind not in ('simulated', 'zero', 'qds'))

    def _on_mode_changed(self):
        decaying = self._current_mode() == 'decay'
        self.label_decay.setEnabled(decaying)
        self.decay_spin.setEnabled(decaying)
        if self._obj_sum is not None:
            self._log(f'Mode: {self._current_mode()}'
                      + (f' {self.decay_spin.value():.3f}' if decaying else '') + '.')

    def _on_tracker_dtype_changed(self):
        """uint8 weights cannot be fractional, so the decaying average is
        greyed out of the combo while that option is ticked."""
        use_uint8 = self._act_uint8.isChecked()
        item = self.accum_mode_combo.model().item(_DECAY_MODE_INDEX)
        if item is not None:
            item.setEnabled(not use_uint8)
        if use_uint8 and self.accum_mode_combo.currentIndex() == _DECAY_MODE_INDEX:
            self.accum_mode_combo.setCurrentIndex(_MODES.index('average'))
            self._log('uint8 tracker: decay unavailable, using average.')
        self._update_canvas_info()

    def _on_set_align_limit(self):
        value, ok = QInputDialog.getDouble(
            self, 'Alignment limit',
            'Reject phase-correlation corrections larger than (nm):',
            self._align_limit_nm, 1.0, 100000.0, 1)
        if not ok:
            return
        self._align_limit_nm = float(value)
        self._log(f'Alignment limit {self._align_limit_nm:g} nm.')

    def _on_set_align_peak(self):
        value, ok = QInputDialog.getDouble(
            self, 'Correlation peak threshold',
            'Reject alignments whose normalised correlation is below:',
            self._align_peak_min, 0.0, 1.0, 2)
        if not ok:
            return
        self._align_peak_min = float(value)
        self._log(f'Correlation gate {self._align_peak_min:.2f}.')

    def _on_set_align_overlap(self):
        value, ok = QInputDialog.getDouble(
            self, 'Minimum overlap',
            'Skip alignment when the overlap is below (% of the frame):',
            self._align_min_overlap * 100.0, 0.0, 100.0, 1)
        if not ok:
            return
        self._align_min_overlap = float(value) / 100.0
        self._log(f'Minimum overlap {value:g}%.')

    def _running_qds_source(self):
        """The open QDS source of the current run, or None."""
        src = self._source
        if src is not None and src.kind == 'qds' and src._qds is not None:
            return src
        return None

    def _on_set_qds_maxlen(self):
        value, ok = QInputDialog.getInt(
            self, 'QDS history length',
            'Number of timestamped positions to keep:',
            self._qds_maxlen, 2, 100000)
        if not ok:
            return
        self._qds_maxlen = int(value)
        src = self._running_qds_source()
        if src is not None:
            src.set_qds_maxlen(value)
        self._log(f'QDS history {value}.')

    def _on_set_qds_every_n(self):
        value, ok = QInputDialog.getInt(
            self, 'QDS position decimation',
            'Keep every Nth received position:',
            self._qds_every_n, 1, 10000)
        if not ok:
            return
        self._qds_every_n = int(value)
        src = self._running_qds_source()
        if src is not None:
            src.set_qds_every_n(value)
        self._log(f'QDS keep every {value}.')

    def _on_set_qds_offset(self):
        value, ok = QInputDialog.getDouble(
            self, 'Detector-position timestamp offset',
            'Use the position nearest to detector time + (s):',
            self._qds_offset_s, -1000.0, 1000.0, 4)
        if not ok:
            return
        self._qds_offset_s = float(value)
        src = self._running_qds_source()
        if src is not None:
            src.qds_offset_s = self._qds_offset_s
        self._log(f'QDS timestamp offset {value:+g} s.')

    def _align_fail_mode(self):
        a = self._align_fail_group.checkedAction()
        return a.data() if a is not None else 'nominal'

    def _current_source_kind(self):
        return _SOURCES[max(0, self.pos_source_combo.currentIndex())]

    def _current_mode(self):
        return _MODES[max(0, self.accum_mode_combo.currentIndex())]

    def _current_anchor(self):
        return _ANCHORS[max(0, self.first_frame_combo.currentIndex())]

    def _tracker_dtype_name(self):
        return 'uint8' if self._act_uint8.isChecked() else 'float32'

    def _downsample_setting(self):
        """The menu choice: an int, or None for automatic."""
        a = self._ds_group.checkedAction()
        if a is None:
            return None
        v = a.data()
        return None if v is None else int(v)

    def _downsample(self):
        """The factor actually used, resolving 'Auto' against the canvas size."""
        ds = self._downsample_setting()
        if ds is not None:
            return max(1, ds)
        if self._obj_sum is None:
            return 1
        ny, nx = self._obj_sum.shape
        if ny * nx <= _TARGET_DISPLAY_PX:
            return 1
        return max(1, int(np.ceil(np.sqrt(ny * nx / _TARGET_DISPLAY_PX))))

    def _anchor_rc_for(self, ny, nx, n):
        """Canvas (row, col) the first accepted frame is centred on."""
        half = n // 2
        return {
            'center':       (ny // 2,     nx // 2),
            'top_left':     (half,        half),
            'top_right':    (half,        nx - half),
            'bottom_left':  (ny - half,   half),
            'bottom_right': (ny - half,   nx - half),
        }.get(self._current_anchor(), (ny // 2, nx // 2))

    def _current_units(self):
        return _UNITS[max(0, self.motor_units_combo.currentIndex())]

    def _on_browse_pos_h5(self):
        dlg = QFileDialog(self)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        dlg.setNameFilter('HDF5 (*.h5 *.hdf5)')
        # The scan being replayed is the best guess at where its positions
        # are; fall back to wherever the current position file lives.
        start_dir = self._main_h5_dir()
        if start_dir is None:
            current = Path(self.pos_h5_file_edit.text().strip() or '.')
            if current.parent.is_dir():
                start_dir = current.parent
        if start_dir is not None:
            dlg.setDirectory(str(start_dir))
        if dlg.exec():
            self.pos_h5_file_edit.setText(dlg.selectedFiles()[0])

    def _main_h5_dir(self):
        """Directory of the H5 the main panel is replaying, or None when it
        is reading images from a PV."""
        if not self._main_is_h5_replay():
            return None
        try:
            text = self._main.h5_file_edit.text().strip()
        except Exception:
            return None
        if not text:
            return None
        d = Path(text).expanduser().parent
        return d if d.is_dir() else None

    def _find_pos_h5(self):
        """The replayed H5 itself when it is a raw master file with
        positions, else the first *_para.hdf5 beside it, or None."""
        d = self._main_h5_dir()
        if d is None:
            return None
        main_h5 = str(Path(self._main.h5_file_edit.text().strip()).expanduser())
        try:
            import h5py
            with h5py.File(main_h5, 'r') as f:
                if read_master_positions(f) is not None:
                    return main_h5
        except Exception:
            pass
        matches = sorted(d.glob(_POS_H5_GLOB))
        return str(matches[0]) if matches else None

    def _on_setup_positions(self):
        self._pos_dlg.show()
        self._pos_dlg.raise_()
        self._pos_dlg.activateWindow()

    # ------------------------------------------------------------------ #
    # Settings (persisted in settings.ini [accumulator] by the main window) #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ini_path():
        """The settings file the main window reads and writes.

        Asked for at call time rather than rebuilt from __file__, so this
        window can never end up reading a different file than the one its
        settings are saved into.
        """
        import rpi_live_gui
        return str(rpi_live_gui._INI_FILE)

    def _load_settings(self):
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(self._ini_path())
        # Settings handed back when an earlier accumulator window closed only
        # reach settings.ini when the main window next saves, so they take
        # precedence over the file — otherwise reopening brings back stale
        # values.
        handed_back = getattr(self._main, '_accum_settings', None) or {}

        def _get(key, fallback):
            if key in handed_back:
                return str(handed_back[key])
            return cfg.get('accumulator', key, fallback=fallback)

        # Images from an H5 replay almost always mean positions from an H5 too,
        # so that is the default pairing unless a source was saved before.
        default_source = 'h5' if self._main_is_h5_replay() else 'simulated'
        source = _get('source', default_source)
        if source == 'qds' and not qds_positions.QDS_AVAILABLE:
            source = default_source
        self.pos_source_combo.setCurrentIndex(
            _SOURCES.index(source) if source in _SOURCES
            else _SOURCES.index(default_source))
        self.h_motor_pv_edit.setText(_get('h_motor_pv', '12ideMCS1:m1.RBV'))
        self.v_motor_pv_edit.setText(_get('v_motor_pv', '12ideMCS1:m2.RBV'))
        # The replayed master file's own positions, or a position file
        # sitting beside the scan, beat any remembered one, which is usually
        # from an earlier scan.
        self._pos_autofound = self._find_pos_h5()
        self.pos_h5_file_edit.setText(self._pos_autofound
                                      or _get('pos_h5_file', ''))
        units = _get('units', _DEFAULT_UNITS)
        self.motor_units_combo.setCurrentIndex(
            _UNITS.index(units) if units in _UNITS else _UNITS.index(_DEFAULT_UNITS))
        anchor = _get('first_frame', 'center')
        self.first_frame_combo.setCurrentIndex(
            _ANCHORS.index(anchor) if anchor in _ANCHORS else 0)
        self._act_invert_h.setChecked(_get('invert_h', 'false').lower() == 'true')
        self._act_invert_v.setChecked(_get('invert_v', 'false').lower() == 'true')
        self._act_swap.setChecked(_get('swap_axes', 'false').lower() == 'true')
        self.size_x_um_spin.setValue(float(_get('size_x_um', '30.0')))
        self.size_y_um_spin.setValue(float(_get('size_y_um', '30.0')))
        self._act_uint8.setChecked(
            _get('uint8_tracker', 'false').lower() == 'true')
        self.loss_thresh_edit.setText(_get('loss_threshold', '0'))
        self.unwrap_check.setChecked(_get('unwrap', 'true').lower() == 'true')
        self.ramp_check.setChecked(_get('ramp', 'true').lower() == 'true')
        self.zero_mean_check.setChecked(_get('zero_mean', 'true').lower() == 'true')
        self.recenter_check.setChecked(_get('recenter', 'false').lower() == 'true')
        self.align_check.setChecked(_get('align', 'false').lower() == 'true')
        self.normalize_check.setChecked(
            _get('normalize', 'false').lower() == 'true')
        self._align_limit_nm = float(_get('align_limit_nm',
                                          str(_DEFAULT_ALIGN_LIMIT_NM)))
        self._align_peak_min = float(_get('align_peak_min',
                                          str(_DEFAULT_ALIGN_PEAK)))
        self._align_min_overlap = float(_get('align_min_overlap',
                                             str(_DEFAULT_MIN_OVERLAP_FRAC)))
        self._act_align_gate.setChecked(
            _get('align_gate', 'false').lower() == 'true')
        fail = _get('align_fail', 'nominal')
        self._align_fail_actions.get(
            fail, self._align_fail_actions['nominal']).setChecked(True)
        mode = _get('accum_mode', 'average')
        self.accum_mode_combo.setCurrentIndex(
            _MODES.index(mode) if mode in _MODES else 0)
        self.decay_spin.setValue(float(_get('decay', '0.95')))
        self.display_every_spin.setValue(int(_get('display_every', '1')))
        ds_raw = _get('downsample', 'auto')
        ds = None if ds_raw.strip().lower() in ('auto', 'none', '') else int(ds_raw)
        self._ds_actions.get(ds, self._ds_actions[None]).setChecked(True)
        self.accum_autolevel_check.setChecked(_get('autolevel', 'true').lower() == 'true')
        self.show_positions_check.setChecked(
            _get('show_positions', 'false').lower() == 'true')
        self._qds_maxlen = max(2, int(_get(
            'qds_history_len', str(qds_positions.DEFAULT_MAXLEN))))
        self._qds_every_n = max(1, int(_get(
            'qds_every_n', str(qds_positions.DEFAULT_EVERY_N))))
        self._qds_offset_s = float(_get('qds_ts_offset_s', '0.0'))

    def _main_is_h5_replay(self):
        try:
            return bool(self._main.h5_check.isChecked())
        except Exception:
            return False

    def get_settings(self):
        """Values for settings.ini's [accumulator] section."""
        return {
            'source':         self._current_source_kind(),
            'h_motor_pv':     self.h_motor_pv_edit.text(),
            'v_motor_pv':     self.v_motor_pv_edit.text(),
            'pos_h5_file':    self.pos_h5_file_edit.text(),
            'units':          self._current_units(),
            'invert_h':       str(self._act_invert_h.isChecked()).lower(),
            'invert_v':       str(self._act_invert_v.isChecked()).lower(),
            'swap_axes':      str(self._act_swap.isChecked()).lower(),
            'size_x_um':      str(self.size_x_um_spin.value()),
            'size_y_um':      str(self.size_y_um_spin.value()),
            'first_frame':    self._current_anchor(),
            'uint8_tracker':  str(self._act_uint8.isChecked()).lower(),
            'loss_threshold': self.loss_thresh_edit.text(),
            'unwrap':         str(self.unwrap_check.isChecked()).lower(),
            'ramp':           str(self.ramp_check.isChecked()).lower(),
            'zero_mean':      str(self.zero_mean_check.isChecked()).lower(),
            'recenter':       str(self.recenter_check.isChecked()).lower(),
            'align':          str(self.align_check.isChecked()).lower(),
            'normalize':      str(self.normalize_check.isChecked()).lower(),
            'align_limit_nm': str(self._align_limit_nm),
            'align_gate':     str(self._act_align_gate.isChecked()).lower(),
            'align_peak_min': str(self._align_peak_min),
            'align_min_overlap': str(self._align_min_overlap),
            'align_fail':     self._align_fail_mode(),
            'accum_mode':     self._current_mode(),
            'decay':          str(self.decay_spin.value()),
            'display_every':  str(self.display_every_spin.value()),
            'downsample':     ('auto' if self._downsample_setting() is None
                               else str(self._downsample_setting())),
            'autolevel':      str(self.accum_autolevel_check.isChecked()).lower(),
            'show_positions': str(self.show_positions_check.isChecked()).lower(),
            'qds_history_len': str(self._qds_maxlen),
            'qds_every_n':    str(self._qds_every_n),
            'qds_ts_offset_s': str(self._qds_offset_s),
        }

    # ------------------------------------------------------------------ #
    # Logging / monitor                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _elide(text):
        text = ' '.join(str(text).split())
        return (text if len(text) <= _LOG_MAX_CHARS
                else text[:_LOG_MAX_CHARS - 1] + '…')

    def _log(self, message, detail=None):
        """Append one short entry to the diagnostic log.

        `detail` adds indented continuation lines under the same timestamp —
        for results like a position test read that are naturally a few short
        facts rather than one sentence. Every line is elided here rather than
        at the call site, so nothing can turn the log into a wall of text.
        Anything that genuinely needs more room — a traceback, a worker
        diagnostic — is printed to stdout instead.
        """
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        lines = [f'[{ts}] {self._elide(message)}']
        for d in (detail or []):
            lines.append(f'{"":>11}{self._elide(d)}')
        self.log_text.append('\n'.join(lines))
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _log_exception(self, context):
        """One short line in the GUI log; the full traceback goes to stdout.

        A traceback in the log box would swamp it, but losing it entirely
        would make this feature much harder to debug — so it is printed to
        the terminal the GUI was launched from.
        """
        self._n_errors += 1
        exc = sys.exc_info()[1]
        where = ''
        tb = traceback.extract_tb(sys.exc_info()[2])
        if tb:
            where = f' @{Path(tb[-1].filename).name}:{tb[-1].lineno}'
        msg = ' '.join(str(exc).split())
        if len(msg) > 44:
            msg = msg[:43] + '…'
        self._log(f'{context}: {type(exc).__name__}: {msg}{where}')
        traceback.print_exc()

    def _on_copy_log(self):
        QApplication.clipboard().setText(self.log_text.toPlainText())
        self._log('Log copied.')

    def _update_monitor(self, full_stats=False):
        """Repaint the monitor block. Always exactly _RESERVED_LINES lines, so
        it cannot change the layout."""
        running = self._worker is not None and self._worker.isRunning()
        thresh = self._loss_threshold()
        lines = [
            f'{"running" if running else "stopped":8s} in {self._n_in}  '
            f'kept {self._n_accepted}',
            f'skip loss {self._n_skip_loss} pos {self._n_skip_pos} '
            f'edge {self._n_skip_bounds} full {self._n_skip_nothing} '
            f'err {self._n_errors}',
            f'loss  {self._last_loss:.4g} / '
            f'{"off" if thresh <= 0 else f"{thresh:.4g}"}',
        ]
        # self._coverage is refreshed by the render thread, never recomputed
        # here — those are full-canvas passes and this runs per frame.
        lines.append(self._coverage)
        # The worker line ('H5 img i/N | stage: report') would be cut off on
        # one line, so it is split after its first field.
        head, sep, rest = self._worker_status.partition(' | ')
        lines.append(head)
        lines.append(rest)

        n = _RESERVED_LINES['monitor_label']
        lines = [ln[:70] for ln in lines[:n]] + [''] * max(0, n - len(lines))
        self.monitor_label.setText('\n'.join(lines))

    def _loss_threshold(self):
        try:
            return float(self.loss_thresh_edit.text().strip())
        except (ValueError, AttributeError):
            return 0.0

    # ------------------------------------------------------------------ #
    # Canvas sizing                                                       #
    # ------------------------------------------------------------------ #

    def _estimate_pixel_size(self):
        """RPI object pixel size for the main window's current settings, or
        None (plus a reason) if it cannot be worked out yet."""
        try:
            params = self._main._collect_params()
        except Exception as exc:
            return None, f'could not read the main window settings ({exc})'
        res_m_probe = params.get('res_m_probe')
        n_probe     = params.get('n_probe')
        if not res_m_probe or not n_probe:
            return None, 'no probe is loaded yet'
        _, res_rpi_m = compute_pixel_sizes(
            res_m_probe, n_probe, params['crop_N'], params['obj_shape'])
        return res_rpi_m, None

    def _canvas_dims(self, res_rpi_m):
        nx = int(round(self.size_x_um_spin.value() * 1e-6 / res_rpi_m))
        ny = int(round(self.size_y_um_spin.value() * 1e-6 / res_rpi_m))
        return max(1, ny), max(1, nx)

    def _update_canvas_info(self):
        res_rpi_m, why = self._estimate_pixel_size()
        if res_rpi_m is None:
            self.canvas_info_label.setText(f'Canvas size unknown — {why}.')
            return
        ny, nx = self._canvas_dims(res_rpi_m)
        dtype_name = self._tracker_dtype_name()
        mb = ny * nx * _BYTES_PER_PIXEL[dtype_name] / 1024 ** 2
        # Two short lines rather than one long one — the controls column is
        # narrow and these labels do not wrap.
        self.canvas_info_label.setText(
            f'{res_rpi_m * 1e9:.2f} nm/px → {ny} × {nx} px\n'
            f'{mb:.0f} MB, {dtype_name} tracker'
        )

    # ------------------------------------------------------------------ #
    # Position source construction / test                                 #
    # ------------------------------------------------------------------ #

    def _build_source(self, res_rpi_m=None, obj_shape=None):
        """Construct (but do not open) a position source from the UI."""
        # Step the simulated raster by half an illuminated diameter so tiles
        # overlap visibly — overlap is the thing worth seeing in a dry run.
        step_um = 1.0
        if res_rpi_m and obj_shape:
            step_um = max(
                0.5 * self._fill_fraction * obj_shape * res_rpi_m * 1e6, 1e-3)
        return _PositionSource(
            kind      = self._current_source_kind(),
            h_pv      = self.h_motor_pv_edit.text(),
            v_pv      = self.v_motor_pv_edit.text(),
            h5_path   = self.pos_h5_file_edit.text(),
            units     = self._current_units(),
            invert_h  = self._act_invert_h.isChecked(),
            invert_v  = self._act_invert_v.isChecked(),
            swap      = self._act_swap.isChecked(),
            sim_step_um   = step_um,
            sim_extent_um = (self.size_x_um_spin.value(),
                             self.size_y_um_spin.value()),
            qds_maxlen    = self._qds_maxlen,
            qds_every_n   = self._qds_every_n,
            qds_offset_s  = self._qds_offset_s,
        )

    def _on_test_read(self):
        res_rpi_m, _ = self._estimate_pixel_size()
        obj_shape = None
        try:
            obj_shape = self._main._collect_params()['obj_shape']
        except Exception:
            pass
        src = self._build_source(res_rpi_m, obj_shape)
        try:
            src.open()
            pos = src.read(0)
            # Interrogate while still open — close() drops the loaded arrays.
            desc, detail = src.describe(), src.detail()
            all_xy = src.read_all() if src.kind == 'h5' else None
        except PositionSourceError as exc:
            self._log(f'Test read FAILED: {exc}')
            return
        except Exception:
            self._log_exception('test read')
            return
        finally:
            src.close()
        self._log(f'Test read OK: {desc}',
                  detail + [f'first x={pos[0]:+.4f} um  y={pos[1]:+.4f} um'])
        if all_xy is not None:
            self._show_position_scatter(*all_xy, Path(src.h5_path).name)

    def _show_position_scatter(self, x_um, y_um, title):
        if self._pos_scatter_win is None:
            self._pos_scatter_win = _PositionScatterWindow(self)
            try:
                from font_utils import apply_font_size_to_tree
                apply_font_size_to_tree(self._pos_scatter_win,
                                        self._main._current_font_size())
            except Exception:
                pass
        self._pos_scatter_win.set_positions(x_um, y_um, title)
        self._pos_scatter_win.show()
        self._pos_scatter_win.raise_()
        self._pos_scatter_win.activateWindow()

    # ------------------------------------------------------------------ #
    # Start / Stop                                                        #
    # ------------------------------------------------------------------ #

    def _on_start_stop(self):
        if self._worker is not None and self._worker.isRunning():
            self._stop_run()
        else:
            self._start_run()

    def _fail_start(self, message):
        self._log(f'Start aborted: {message.split(chr(10))[0]}')
        QMessageBox.warning(self, 'Cannot start', message)

    def _start_run(self):
        # 1. Borrow the main window's validated run inputs.
        try:
            params, probe, pixel_mask = self._main.snapshot_run_inputs()
        except RuntimeError as exc:
            self._fail_start(str(exc))
            return
        except Exception:
            self._log_exception('main settings')
            self._fail_start('Could not read the main window settings — '
                             'see the diagnostic log.')
            return

        # An H5 replay stops after one lap of the file (wrapping round when it
        # starts midway) rather than stamping the same patterns again. PV
        # runs are open-ended and ignore this.
        params['h5_single_pass'] = True

        self._probe_shape   = params.get('probe_shape', DEFAULT_PROBE_SHAPE)
        self._fill_fraction = float(params.get('fill_fraction', DEFAULT_FILL_FRACTION))
        obj_shape = params['obj_shape']

        # 2. Work out the canvas geometry and refuse anything unreasonable.
        if not params.get('res_m_probe') or not params.get('n_probe'):
            self._fail_start('The loaded probe has no pixel size — cannot size '
                             'the canvas.')
            return
        _, res_rpi_m = compute_pixel_sizes(
            params['res_m_probe'], params['n_probe'],
            params['crop_N'], obj_shape)
        if not np.isfinite(res_rpi_m) or res_rpi_m <= 0:
            self._fail_start(f'Computed an impossible pixel size '
                             f'({res_rpi_m!r}). Check the probe and Crop N.')
            return
        ny, nx = self._canvas_dims(res_rpi_m)
        dtype_name = self._tracker_dtype_name()
        n_bytes = ny * nx * _BYTES_PER_PIXEL[dtype_name]
        if max(ny, nx) > _MAX_CANVAS_SIDE:
            self._fail_start(
                f'Canvas would be {ny} × {nx} px at {res_rpi_m * 1e9:.2f} nm/px, '
                f'over the {_MAX_CANVAS_SIDE} px limit. Reduce the canvas size '
                f'or use a coarser object shape.')
            return
        if n_bytes > _MAX_CANVAS_BYTES:
            self._fail_start(
                f'Canvas would need {n_bytes / 1024**3:.2f} GB '
                f'({ny} × {nx} px), over the '
                f'{_MAX_CANVAS_BYTES / 1024**3:.0f} GB limit.')
            return
        if obj_shape > min(ny, nx):
            self._fail_start(
                f'A single frame ({obj_shape} px) is larger than the canvas '
                f'({ny} × {nx} px). Increase the canvas size.')
            return

        # 3. Allocate.
        try:
            obj_sum = np.zeros((ny, nx), dtype=np.float32)
            # float32 by default: the decaying average scales the tracker by a
            # fractional factor, and float32 still counts exactly well past any
            # realistic frame total. uint8 trades the decaying average (and a
            # 255 ceiling on overlaps) for 3 bytes per pixel.
            contrib = np.zeros((ny, nx), dtype=np.dtype(dtype_name))
        except MemoryError:
            self._fail_start(f'Out of memory allocating a {ny} × {nx} canvas '
                             f'({n_bytes / 1024**2:.0f} MB).')
            return

        # 4. Build and open the position source; one test read before committing.
        source = self._build_source(res_rpi_m, obj_shape)
        try:
            source.open()
            probe_pos = source.read(0)
        except PositionSourceError as exc:
            self._fail_start(str(exc))
            return
        except Exception:
            self._log_exception('position source')
            self._fail_start('Could not open the position source — see the '
                             'diagnostic log.')
            return

        # 5. Commit the run state.
        self._data_mutex.lock()
        try:
            self._obj_sum = obj_sum
            self._contrib = contrib
            self._contrib_max = (int(np.iinfo(contrib.dtype).max)
                                 if np.issubdtype(contrib.dtype, np.integer)
                                 else None)
        finally:
            self._data_mutex.unlock()
        self._warned_decay = False
        self._warned_saturate = False
        self._render.set_arrays(obj_sum, contrib)

        self._source        = source
        self._res_rpi_m     = res_rpi_m
        self._run_params    = params
        self._run_started   = datetime.datetime.now()
        self._scan_tag      = (scan_tag_from_path(params.get('h5_file'))
                               if params.get('h5_mode') else None)
        self._stamp_mask    = illumination_mask(
            obj_shape, self._probe_shape, self._fill_fraction, _ACCUM_STAMP_FUDGE)
        self._anchor_rc     = self._anchor_rc_for(ny, nx, obj_shape)
        self._origin_um     = None
        self._records       = []      # one row per accepted frame
        self._n_in = self._n_accepted = 0
        self._n_skip_loss = self._n_skip_pos = self._n_skip_bounds = 0
        self._n_errors = 0
        self._n_align_ok = self._n_align_rejected = self._n_skip_align = 0
        self._n_align_nooverlap = 0
        self._n_skip_nothing = 0
        self._last_align_peak = float('nan')
        self._last_align_overlap = float('nan')
        self._warned_align = False
        self._warned_nothing = False
        self._decay_mask = None
        self._last_loss = float('nan')
        self._last_pos_um = self._last_pos_px = self._last_pos_idx = None
        self._coverage = 'cover —'
        self._worker_status = ''
        self._frames_since_redraw = 0
        self._first_render = True

        self._log(f'Canvas {ny}x{nx} px, {res_rpi_m * 1e9:.2f} nm/px, '
                  f'{n_bytes / 1024**2:.0f} MB, {dtype_name}.')
        self._log(f'Frame {obj_shape} px, {self._probe_shape} '
                  f'{self._fill_fraction:.2f}, anchor {self._current_anchor()} '
                  f'@ {self._anchor_rc}.')
        self._log(f'Pos: {source.describe()}, first '
                  f'({probe_pos[0]:+.3f}, {probe_pos[1]:+.3f}) um.')
        self._log(f'{params["method"]}, '
                  f'{"H5" if params["h5_mode"] else "PV"}, '
                  f'{self._current_mode()}'
                  + (f' {self.decay_spin.value():.3f}'
                     if self._current_mode() == 'decay' else '') + '.')

        # 6. Start the worker with the position hook attached.
        try:
            worker_cls = WORKER_CLASSES[params['method']]
            self._worker = worker_cls(
                params          = params,
                probe_raw       = probe,
                pixel_mask_full = pixel_mask,
                # The accumulator keeps no frame history — results go straight
                # into the canvas. maxlen=1 so a long run cannot pile up object
                # arrays the way an unbounded list would.
                results_deque   = deque(maxlen=1),
                parent          = self,
            )
            self._worker.position_reader = self._source.read_indexed
            self._worker.frame_ready.connect(self._on_frame_ready)
            self._worker.status_update.connect(self._on_worker_status)
            self._worker.error_occurred.connect(self._on_worker_error)
            self._worker.finished.connect(self._on_worker_finished)
            self._worker.start()
        except Exception:
            self._log_exception('worker start')
            self._fail_start('Could not start the reconstruction worker — '
                             'see the diagnostic log.')
            self._worker = None
            return

        self._set_running_ui(True)
        self._update_monitor()
        self._log('Started.')

    def _stop_run(self):
        if self._worker is not None:
            self._worker.stop()
            if not self._worker.wait(10000):
                self._log('Worker did not stop within 10 s.')
        self._log('Stopped.')

    def _on_worker_finished(self):
        if getattr(self._worker, 'finished_h5_pass', False):
            self._log('Read every pattern in the H5 file; stopped.')
        self._set_running_ui(False)
        if self._tracker_win is not None:
            self._tracker_win.close()
            self._tracker_win = None
        if self._source is not None:
            self._source.close()
        self._update_monitor(full_stats=True)

    def _set_running_ui(self, running):
        self.start_btn.setText('Stop' if running else 'Start')
        self.start_btn.setStyleSheet(_STOP_BTN_STYLE if running else _START_BTN_STYLE)
        # Fixed for the duration of a run
        for w in (self.positions_group, self.canvas_group):
            w.setEnabled(not running)
        # The dialog may already be open; its controls freeze with the button.
        self._pos_dlg.set_editable(not running)
        # The tracker dtype is baked into the allocation, so it is frozen too;
        # it lives in the Options menu rather than in canvas_group now.
        self._act_uint8.setEnabled(not running)
        # Available during and after a run, once there is something to act on
        have_data = self._obj_sum is not None
        self.refresh_btn.setEnabled(running)   # "while its running", per spec
        self.save_btn.setEnabled(have_data)
        self.show_tracker_btn.setEnabled(have_data)

    # ------------------------------------------------------------------ #
    # Per-frame accumulation                                              #
    # ------------------------------------------------------------------ #

    def _on_worker_status(self, text):
        # The worker's per-iteration chatter belongs on one monitor line, not
        # in the log and not in the position-source readout.
        self._worker_status = (text or '').strip().splitlines()[-1:][0] \
            if (text or '').strip() else ''
        self._update_monitor()

    def _on_worker_error(self, msg):
        self._log(f'Worker error: {msg.splitlines()[0] if msg else "?"}')
        print(msg, flush=True)
        self._set_running_ui(False)
        QMessageBox.critical(
            self, 'Worker error',
            'The reconstruction worker stopped with an error. The full '
            'diagnostic is in the log at the bottom of this window.')

    def _on_frame_ready(self, entry):
        """Stamp one reconstruction into the canvas.

        Wrapped whole so a single bad frame logs a traceback and is counted,
        rather than taking the run down.
        """
        try:
            self._accumulate(entry)
        except Exception:
            self._log_exception(f'frame {entry.get("frame_idx")}')
            self._update_monitor()

    def _accumulate(self, entry):
        self._n_in += 1

        # -- loss gate --
        loss_hist = entry.get('loss_history')
        loss = (float(loss_hist[-1]) if loss_hist is not None and len(loss_hist)
                else float('nan'))
        self._last_loss = loss
        threshold = self._loss_threshold()
        if threshold > 0:
            if not np.isfinite(loss):
                self._n_skip_loss += 1
                self._log(f'Frame {entry.get("frame_idx")}: no loss, skipped.')
                self._update_monitor()
                return
            if loss > threshold:
                self._n_skip_loss += 1
                self._update_monitor()
                return

        # -- position gate --
        pos = entry.get('position')
        if pos is None:
            self._n_skip_pos += 1
            err = entry.get('position_error') or 'no position was returned'
            self._log(f'Frame {entry.get("frame_idx")}: {err}, skipped.')
            self._update_monitor()
            return
        self._last_pos_um = (float(pos[0]), float(pos[1]))
        self._last_pos_idx = pos[2] if len(pos) > 2 else None

        # The first accepted frame defines the centre of the canvas.
        if self._origin_um is None:
            self._origin_um = self._last_pos_um
            self._log(f'Anchor ({pos[0]:+.3f}, {pos[1]:+.3f}) um '
                      f'-> px {self._anchor_rc}.')

        fx = (self._last_pos_um[0] - self._origin_um[0]) * 1e-6 / self._res_rpi_m
        fy = (self._last_pos_um[1] - self._origin_um[1]) * 1e-6 / self._res_rpi_m
        dx_px, dy_px = int(round(fx)), int(round(fy))
        # Fractional positions (ptychi): place on the nearest whole pixel and
        # shift the frame by the remainder, to _SUBPIXEL_DECIMALS places.
        sub_dy = sub_dx = 0.0
        if self._source is not None and self._source.subpixel:
            fx, fy = round(fx, _SUBPIXEL_DECIMALS), round(fy, _SUBPIXEL_DECIMALS)
            dx_px, dy_px = int(round(fx)), int(round(fy))
            sub_dx = round(fx - dx_px, _SUBPIXEL_DECIMALS)
            sub_dy = round(fy - dy_px, _SUBPIXEL_DECIMALS)
        else:
            fx, fy = float(dx_px), float(dy_px)
        self._last_pos_px = (fx, fy)

        obj = entry['obj']
        n   = obj.shape[-1]
        if obj.shape[0] != obj.shape[1]:
            self._n_errors += 1
            self._log(f'Frame {entry.get("frame_idx")}: {obj.shape} '
                      f'not square, skipped.')
            self._update_monitor()
            return
        # The object shape can only change between runs, but check anyway: a
        # mismatched stamp mask would silently misplace every pixel.
        if self._stamp_mask.shape[0] != n:
            self._stamp_mask = illumination_mask(
                n, self._probe_shape, self._fill_fraction, _ACCUM_STAMP_FUDGE)
            self._log(f'Frame size now {n} px, mask rebuilt.')
        if sub_dy or sub_dx:
            obj = _subpixel_shift(obj, sub_dy, sub_dx)

        ny, nx = self._obj_sum.shape
        ar, ac = self._anchor_rc
        r0 = ar + dy_px - n // 2
        c0 = ac + dx_px - n // 2
        # Where the motors alone would have put it, kept so the saved record
        # shows the nominal position and the correction separately. Includes
        # the subpixel part, which the shift above has already applied.
        nom_dx_px, nom_dy_px = fx, fy

        # If this frame would write nothing — fill-empty mode landing on
        # canvas that is already covered — there is no point unwrapping it or
        # correlating it. Checked at the nominal placement: an alignment
        # correction is bounded by the limit and cannot turn "writes nothing"
        # into "writes a lot".
        if not self._will_add_pixels(r0, c0, n):
            self._n_skip_nothing += 1
            if not self._warned_nothing:
                self._warned_nothing = True
                self._log(f'Frame {entry.get("frame_idx")}: area already '
                          f'filled, skipped before reconstruction work.')
            self._update_monitor()
            return

        # Phase prep, restricted to the illuminated area. Done before
        # placement because the alignment correlates on the prepared phase.
        phase = process_phase_region(
            np.angle(obj).astype(np.float32), self._stamp_mask,
            unwrap    = self.unwrap_check.isChecked(),
            ramp      = self.ramp_check.isChecked(),
            zero_mean = self.zero_mean_check.isChecked(),
        )

        if self.align_check.isChecked():
            status, shift = self._alignment_shift(phase, r0, c0, n,
                                                  entry.get('frame_idx'))
            if status == 'fail' and self._align_fail_mode() == 'discard':
                self._n_skip_align += 1
                self._update_monitor()
                return
            if status == 'ok':
                r0 += shift[0]
                c0 += shift[1]
                dy_px += shift[0]
                dx_px += shift[1]
                fy += shift[0]
                fx += shift[1]
                self._last_pos_px = (fx, fy)

        sr0, sr1 = max(0, r0), min(ny, r0 + n)
        sc0, sc1 = max(0, c0), min(nx, c0 + n)
        if sr0 >= sr1 or sc0 >= sc1:
            self._n_skip_bounds += 1
            self._log(f'Frame {entry.get("frame_idx")}: off-canvas at '
                      f'({dx_px:+d}, {dy_px:+d}) px, skipped.')
            self._update_monitor()
            return
        if (sr1 - sr0) < n or (sc1 - sc0) < n:
            self._log(f'Frame {entry.get("frame_idx")}: clipped to '
                      f'{sr1 - sr0}x{sc1 - sc0} px.')

        # Tile-local window matching the clipped canvas window.
        tr0, tc0 = sr0 - r0, sc0 - c0
        sub_mask  = self._stamp_mask[tr0:tr0 + (sr1 - sr0), tc0:tc0 + (sc1 - sc0)]
        sub_phase = phase[tr0:tr0 + (sr1 - sr0), tc0:tc0 + (sc1 - sc0)]

        norm_offset = 0.0
        self._data_mutex.lock()
        try:
            if self.normalize_check.isChecked():
                norm_offset = self._normalize_offset(
                    sub_phase, sub_mask, sr0, sr1, sc0, sc1)
                if norm_offset:
                    sub_phase = sub_phase + np.float32(norm_offset)
            self._stamp(sub_phase, sub_mask, sr0, sr1, sc0, sc1)
        finally:
            self._data_mutex.unlock()

        self._records.append((
            float(entry.get('frame_idx') if entry.get('frame_idx') is not None else -1),
            float(entry.get('source_idx') if entry.get('source_idx') is not None else -1),
            float(entry.get('timestamp') or 0.0),
            loss,
            self._last_pos_um[0], self._last_pos_um[1],
            float(nom_dx_px), float(nom_dy_px),
            float(fx), float(fy),
            float(fx - nom_dx_px), float(fy - nom_dy_px),
            norm_offset,
        ))
        self._n_accepted += 1

        # Ask for a redraw; the render thread coalesces bursts so this never
        # throttles acquisition.
        self._frames_since_redraw += 1
        if self._frames_since_redraw >= self.display_every_spin.value():
            self._frames_since_redraw = 0
            self._render.request(self._downsample())
            self._update_monitor(full_stats=True)
        else:
            self._update_monitor()

    def _normalize_offset(self, phase, mask, r0, r1, c0, c1):
        """Phase offset that brings a frame's mean onto the accumulator's.

        Both means are taken over the same pixels: illuminated in the new
        frame and already populated on the canvas. Returns 0.0 when there is
        too little overlap to say anything — the first frame, or one landing
        on fresh canvas. Call with _data_mutex held.
        """
        cnt = self._contrib[r0:r1, c0:c1]
        overlap = mask & (cnt > 0)
        if int(np.count_nonzero(overlap)) < _MIN_ALIGN_OVERLAP_PX:
            return 0.0
        acc = (self._obj_sum[r0:r1, c0:c1][overlap].astype(np.float64)
               / cnt[overlap])
        offset = float(acc.mean() - phase[overlap].astype(np.float64).mean())
        return offset if np.isfinite(offset) else 0.0

    def _will_add_pixels(self, r0, c0, n):
        """Would stamping a frame here change anything?

        Only ever False in fill-empty mode, where a frame landing on canvas
        that is already covered contributes nothing: every other mode adds a
        contribution to each illuminated pixel regardless. Returns True for a
        fully off-canvas frame so the caller's own bounds check reports that
        case with its more useful message.
        """
        if self._current_mode() != 'fill_empty' or self._contrib is None:
            return True
        ny, nx = self._contrib.shape
        sr0, sr1 = max(0, r0), min(ny, r0 + n)
        sc0, sc1 = max(0, c0), min(nx, c0 + n)
        if sr0 >= sr1 or sc0 >= sc1:
            return True                       # off-canvas: let the caller say so
        tr0, tc0 = sr0 - r0, sc0 - c0
        sub_mask = self._stamp_mask[tr0:tr0 + (sr1 - sr0), tc0:tc0 + (sc1 - sc0)]
        self._data_mutex.lock()
        try:
            cnt = self._contrib[sr0:sr1, sc0:sc1]
            return bool(np.any(sub_mask & (cnt == 0)))
        finally:
            self._data_mutex.unlock()

    @staticmethod
    def _correlation_peak(ref, mov, overlap, dy, dx):
        """Normalised cross-correlation of the two images at a given shift.

        Evaluated only where both the reference and the shifted frame are
        inside the overlap, and mean-subtracted first, so it is a genuine
        match quality in [-1, 1] rather than a brightness comparison: 1.0 is
        a perfect match, 0.0 is noise. This is what the correlation gate
        thresholds — a raw phase-correlation peak height would depend on the
        array size and the masking, and so would need recalibrating every
        time the object shape changed.
        """
        h, w = ref.shape
        r0, r1 = max(0, dy), min(h, h + dy)
        c0, c1 = max(0, dx), min(w, w + dx)
        if r1 - r0 < 2 or c1 - c0 < 2:
            return 0.0
        a = ref[r0:r1, c0:c1]
        b = mov[r0 - dy:r1 - dy, c0 - dx:c1 - dx]
        m = overlap[r0:r1, c0:c1] & overlap[r0 - dy:r1 - dy, c0 - dx:c1 - dx]
        if int(np.count_nonzero(m)) < _MIN_ALIGN_OVERLAP_PX:
            return 0.0
        av = a[m].astype(np.float64)
        bv = b[m].astype(np.float64)
        av -= av.mean()
        bv -= bv.mean()
        na = float(np.sqrt(np.dot(av, av)))
        nb = float(np.sqrt(np.dot(bv, bv)))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return float(np.dot(av, bv) / (na * nb))

    def _alignment_shift(self, phase, r0, c0, n, frame_idx=None):
        """Phase-correlate a prepared frame against the accumulated object.

        Only pixels that are both illuminated in the new frame and already
        populated on the canvas take part — a masked cross-correlation, so
        the empty canvas around a partially-built object cannot drag the
        result.

        Returns (status, shift):
          ('ok', (dy, dx))  a correction to apply, in canvas pixels
          ('none', None)    nothing to align against — not a failure, the
                            frame simply goes on its motor position
          ('fail', None)    the correlation was rejected; the caller applies
                            the configured failure policy

        A correlation that fails tends to fail wildly rather than subtly,
        which is what the limit and the gate are for.
        """
        try:
            from skimage.registration import phase_cross_correlation
        except ImportError:                      # older scikit-image
            try:
                from skimage.feature import register_translation \
                    as phase_cross_correlation
            except ImportError:
                if not self._warned_align:
                    self._warned_align = True
                    self._log('Alignment needs scikit-image; skipping.')
                return 'none', None

        ny, nx = self._obj_sum.shape
        sr0, sr1 = max(0, r0), min(ny, r0 + n)
        sc0, sc1 = max(0, c0), min(nx, c0 + n)
        if sr0 >= sr1 or sc0 >= sc1:
            return 'none', None                  # off-canvas; caller reports it

        tr0, tc0 = sr0 - r0, sc0 - c0
        h, w = sr1 - sr0, sc1 - sc0
        frame_mask = self._stamp_mask[tr0:tr0 + h, tc0:tc0 + w]
        moving     = np.ascontiguousarray(phase[tr0:tr0 + h, tc0:tc0 + w])

        self._data_mutex.lock()
        try:
            cnt = self._contrib[sr0:sr1, sc0:sc1]
            overlap = frame_mask & (cnt > 0)
            n_overlap = int(np.count_nonzero(overlap))
            n_frame = int(np.count_nonzero(self._stamp_mask))
            frac = n_overlap / n_frame if n_frame else 0.0
            self._last_align_overlap = frac
            if n_overlap == 0:
                # Virgin canvas — nothing to align against at all. Distinct
                # from a thin overlap, and not worth counting.
                return 'none', None
            if (n_overlap < _MIN_ALIGN_OVERLAP_PX
                    or frac < self._align_min_overlap):
                # Too little in common to correlate meaningfully. Not a
                # failure — the frame simply goes on its motor position.
                self._n_align_nooverlap += 1
                return 'none', None
            ref = np.zeros((h, w), dtype=np.float32)
            np.divide(self._obj_sum[sr0:sr1, sc0:sc1], cnt, out=ref, where=cnt > 0)
        finally:
            self._data_mutex.unlock()

        try:
            result = phase_cross_correlation(
                ref, moving, reference_mask=overlap, moving_mask=overlap)
        except Exception:
            if not self._warned_align:
                self._warned_align = True
                self._log_exception('alignment')
            else:
                self._n_errors += 1
            return 'fail', None

        shift = np.asarray(result[0] if isinstance(result, tuple) else result,
                           dtype=float)
        if shift.size < 2 or not np.all(np.isfinite(shift)):
            self._reject_align(frame_idx, 'no finite shift')
            return 'fail', None

        dy, dx = int(round(shift[0])), int(round(shift[1]))
        limit_px = (self._align_limit_nm * 1e-9) / self._res_rpi_m
        magnitude = float(np.hypot(shift[0], shift[1]))
        if magnitude > limit_px:
            self._reject_align(
                frame_idx,
                f'{magnitude * self._res_rpi_m * 1e9:.0f} nm > '
                f'{self._align_limit_nm:g} nm')
            return 'fail', None

        if self._act_align_gate.isChecked():
            peak = self._correlation_peak(ref, moving, overlap, dy, dx)
            self._last_align_peak = peak
            if peak < self._align_peak_min:
                self._reject_align(frame_idx,
                                   f'peak {peak:.2f} < {self._align_peak_min:.2f}')
                return 'fail', None

        if dy == 0 and dx == 0:
            self._n_align_ok += 1
            return 'none', None                  # nothing to move
        self._n_align_ok += 1
        return 'ok', (dy, dx)

    def _reject_align(self, frame_idx, reason):
        """Count a failed alignment, logging only the first one per run."""
        self._n_align_rejected += 1
        if not self._warned_align:
            self._warned_align = True
            self._log(f'Frame {frame_idx}: align rejected ({reason}); '
                      f'further ones counted only.')

    def _stamp(self, phase, mask, r0, r1, c0, c1):
        """Combine one prepared frame into the canvas under the current mode.

        Call with _data_mutex held. Every mode maintains the invariant that
        obj_sum / contrib is the image, so the display, the save and the
        pixel-contribution tracker stay consistent across modes.
        """
        mode = self._current_mode()
        obj  = self._obj_sum[r0:r1, c0:c1]
        cnt  = self._contrib[r0:r1, c0:c1]

        if mode == 'replace':
            # Newest frame wins outright: one contribution, its own value.
            obj[mask] = phase[mask]
            cnt[mask] = 1
            return

        if mode == 'fill_empty':
            # Only ever write where nothing has been written before.
            fresh = mask & (cnt == 0)
            obj[fresh] = phase[fresh]
            cnt[fresh] = 1
            return

        if mode == 'decay' and self._contrib_max is not None and not self._warned_decay:
            # Unreachable through the UI (the combo entry is greyed out for an
            # integer tracker) but worth saying out loud rather than silently
            # behaving like a plain average.
            self._warned_decay = True
            self._log('Decay needs a float tracker; averaging instead.')

        if mode == 'decay' and self._contrib_max is None:
            # Fade the *whole* canvas before adding, so older frames count for
            # less. Numerator and denominator are scaled together, so an
            # untouched pixel's normalised value is unchanged — only its weight
            # relative to newer frames drops.
            #
            # Pixels already below _DECAY_FLOOR are left alone, which stops a
            # never-revisited patch from decaying into float underflow (and
            # disappearing) while keeping its value exact.
            #
            # This is the one mode that costs O(canvas) rather than O(frame)
            # per accepted frame.
            decay = float(self.decay_spin.value())
            if decay < 1.0:
                # Reused buffer: this is canvas-sized and would otherwise be
                # allocated and freed on every single accepted frame.
                if (self._decay_mask is None
                        or self._decay_mask.shape != self._contrib.shape):
                    self._decay_mask = np.empty(self._contrib.shape, dtype=bool)
                active = self._decay_mask
                np.greater(self._contrib, _DECAY_FLOOR, out=active)
                np.multiply(self._obj_sum, decay, out=self._obj_sum, where=active)
                np.multiply(self._contrib, decay, out=self._contrib, where=active)

        # 'average', and the add step of 'decay'
        if self._contrib_max is None:
            obj += np.where(mask, phase, 0.0)
            cnt += mask
            return

        # Integer tracker: stop at the ceiling rather than wrapping, which
        # would roll the count back through zero. Numerator and denominator
        # must freeze *together* — incrementing only the sum would let the
        # normalised value climb without bound once the count sticks.
        add = mask & (cnt < self._contrib_max)
        obj += np.where(add, phase, 0.0)
        cnt[add] += 1
        if not self._warned_saturate and add.sum() != mask.sum():
            self._warned_saturate = True
            self._log(f'uint8 ceiling ({self._contrib_max}) hit; '
                      f'those pixels now ignore new frames.')

    # ------------------------------------------------------------------ #
    # Display                                                             #
    # ------------------------------------------------------------------ #

    def _on_rendered(self, payload):
        """Paint what the render thread prepared.

        Deliberately does no array maths — levels and statistics arrive
        already computed, so the only cost here is the raster itself.
        """
        if not payload:
            return
        img = payload.get('image')
        if img is None or img.size == 0:
            return
        self._displayed_ds = payload.get('downsample', 1)
        # pg_image_tools is col-major (image[x, y]); the canvas is [row, col].
        self.accum_plot.set_image(
            img.T, pixel_size_m=(self._res_rpi_m or 1.0) * self._displayed_ds,
            autoRange=True if self._first_render else None)
        if self.accum_autolevel_check.isChecked():
            self.accum_plot.set_levels(*payload['levels'])
        self._first_render = False
        self._update_pos_scatter()

        self._coverage = (f'cover {payload["coverage"]:.1f}%'
                          f'  max {payload["max_weight"]:.3g}')
        self._update_monitor()

        tracker = payload.get('tracker')
        if tracker is not None and self._tracker_win is not None:
            self._tracker_win.update_image(tracker, payload['tracker_levels'])

    def _on_show_positions_toggled(self, checked):
        self._pos_scatter.setVisible(checked)
        self._update_pos_scatter()

    def _update_pos_scatter(self):
        """Mark the centre of every accumulated frame on the object image.

        Drawn from the frame records' final placement, which already includes
        any phase-correlation correction — with alignment off it is just the
        motor position. Refreshed with each redraw, so it costs nothing while
        hidden.
        """
        if not self.show_positions_check.isChecked():
            return
        if not self._records or self._anchor_rc is None:
            self._pos_scatter.setData([], [])
            return
        rec = np.asarray(self._records, dtype=np.float64)
        ar, ac = self._anchor_rc
        ds = self._displayed_ds or 1
        # Row-major image: x is the column, y the row, both in displayed
        # (possibly downsampled) pixels.
        self._pos_scatter.setData(x=(ac + rec[:, 8]) / ds,
                                  y=(ar + rec[:, 9]) / ds)

    def _on_cursor_moved(self, event):
        pos = event[0]
        if self._obj_sum is None or not self._accum_vb.sceneBoundingRect().contains(pos):
            self.accum_cursor_label.setText('')
            return
        pt = self._accum_vb.mapSceneToView(pos)
        ds = self._displayed_ds
        col, row = int(pt.x()) * ds, int(pt.y()) * ds
        ny, nx = self._obj_sum.shape
        if not (0 <= row < ny and 0 <= col < nx):
            self.accum_cursor_label.setText('')
            return
        weight = float(self._contrib[row, col])
        value = (self._obj_sum[row, col] / weight) if weight else float('nan')
        ar, ac = self._anchor_rc or (ny // 2, nx // 2)
        dx_um = (col - ac) * self._res_rpi_m * 1e6
        dy_um = (row - ar) * self._res_rpi_m * 1e6
        self.accum_cursor_label.setText(
            f'x={col} px  y={row} px     offset: {dx_um:+.3f}, {dy_um:+.3f} µm'
            f'     contributions: {weight:.4g}     phase: '
            + (f'{value:.4g}' if weight else '—'))

    # ------------------------------------------------------------------ #
    # Buttons                                                             #
    # ------------------------------------------------------------------ #

    def _on_refresh_object(self):
        if self._obj_sum is None:
            return
        self._data_mutex.lock()
        try:
            self._obj_sum.fill(0.0)
            self._contrib.fill(0)
        finally:
            self._data_mutex.unlock()
        self._records = []
        self._n_accepted = 0
        self._n_skip_bounds = 0
        if self.recenter_check.isChecked():
            self._origin_um = None
            self._log('Zeroed; next frame re-anchors.')
        else:
            self._log('Zeroed; anchor kept.')
        self._first_render = True
        self._render.request(self._downsample())
        self._update_monitor(full_stats=True)

    def _on_save_object(self):
        if self._obj_sum is None:
            QMessageBox.information(self, 'Nothing to save',
                                    'Start a run first.')
            return
        # Same folder the main panel's Save dir browser opens in.
        start_dir = ''
        try:
            start_dir = self._main.browse_dir('save')
            if start_dir is None:
                start_dir = self._main.save_dir_edit.text().strip()
        except Exception:
            pass
        # Name the file after the scan being replayed when there is one —
        # far easier to match up with the raw data than a wall-clock stamp.
        stem = (f'rpi_object_{self._scan_tag}' if self._scan_tag
                else f'rpi_object_{datetime.datetime.now():%Y%m%d_%H%M%S}')
        default = str(Path(start_dir or '.') / f'{stem}.h5')
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save accumulated object', default, 'HDF5 (*.h5)')
        if not path:
            return
        if not path.lower().endswith(('.h5', '.hdf5')):
            path += '.h5'
        p = self._run_params or {}
        meta = {
            'size_x_um':      self.size_x_um_spin.value(),
            'size_y_um':      self.size_y_um_spin.value(),
            'probe_shape':    self._probe_shape,
            'fill_fraction':  self._fill_fraction,
            'stamp_fudge':    _ACCUM_STAMP_FUDGE,
            'unwrap':         self.unwrap_check.isChecked(),
            'remove_ramp':    self.ramp_check.isChecked(),
            'zero_mean':      self.zero_mean_check.isChecked(),
            'align':          self.align_check.isChecked(),
            'normalize_to_accumulator': self.normalize_check.isChecked(),
            'align_limit_nm': self._align_limit_nm,
            'align_gate':     self._act_align_gate.isChecked(),
            'align_peak_min': self._align_peak_min,
            'align_min_overlap': self._align_min_overlap,
            'align_fail':     self._align_fail_mode(),
            'n_align_applied':  self._n_align_ok,
            'n_align_rejected': self._n_align_rejected,
            'n_align_discarded': self._n_skip_align,
            'n_align_thin_overlap': self._n_align_nooverlap,
            'n_skipped_already_filled': self._n_skip_nothing,
            'accum_mode':     self._current_mode(),
            'decay':          (self.decay_spin.value()
                               if self._current_mode() == 'decay' else 1.0),
            'decay_floor':    _DECAY_FLOOR,
            'tracker_dtype':  (str(self._contrib.dtype)
                               if self._contrib is not None else ''),
            'loss_threshold': self._loss_threshold(),
            'position_source': self._current_source_kind(),
            'h_motor_pv':     self.h_motor_pv_edit.text(),
            'v_motor_pv':     self.v_motor_pv_edit.text(),
            'position_units': self._current_units(),
            'position_h5_format': (self._source.h5_format
                                   if self._source is not None else ''),
            'subpixel_shift': bool(self._source is not None
                                   and self._source.subpixel),
            'subpixel_decimals': _SUBPIXEL_DECIMALS,
            'invert_h':       self._act_invert_h.isChecked(),
            'invert_v':       self._act_invert_v.isChecked(),
            'swap_axes':      self._act_swap.isChecked(),
            'origin_um':      list(self._origin_um or (0.0, 0.0)),
            # Where the first frame was placed — positions_px are offsets from
            # this, so without it the saved coordinates cannot be interpreted.
            'first_image_anchor': self._current_anchor(),
            'anchor_rc':      list(self._anchor_rc or (0, 0)),
            # --- provenance -------------------------------------------
            'saved_at':       datetime.datetime.now().isoformat(timespec='seconds'),
            'run_started_at': (self._run_started.isoformat(timespec='seconds')
                               if self._run_started else ''),
            'scan_tag':       self._scan_tag or '',
            'source_h5_file': p.get('h5_file', '') if p.get('h5_mode') else '',
            'position_h5_file': (self.pos_h5_file_edit.text()
                                 if self._current_source_kind() == 'h5' else ''),
            'h5_start_idx':   p.get('h5_start_idx', 0),
            'h5_stride':      p.get('h5_stride', 1),
            'n_avg':          p.get('n_avg', 1),
            'image_source':   'h5' if p.get('h5_mode') else 'pv',
            'image_pv':       '' if p.get('h5_mode') else p.get('pv_name', ''),
            # --- which probe produced these reconstructions (not the probe) --
            'probe_name':     p.get('probe_name', ''),
            'probe_path':     p.get('probe_path', ''),
            'probe_pixel_size_m': p.get('res_m_probe') or 0.0,
            'probe_array_size':   p.get('n_probe') or 0,
            'probe_modes_used':   p.get('n_probe_modes', 0),
            # --- reconstruction settings -------------------------------
            'method':         p.get('method', ''),
            'res_mode':       p.get('res_mode', ''),
            'obj_shape':      p.get('obj_shape', 0),
            'obj_init':       p.get('obj_init', ''),
            'crop_N':         p.get('crop_N', 0),
            'beam_center_y':  p.get('beam_cy', 0),
            'beam_center_x':  p.get('beam_cx', 0),
            'energy_kev':     p.get('energy_kev', 0.0),
            'detector_distance_m': p.get('detector_distance', 0.0),
            'obj_pixel_size_nm': (self._res_rpi_m or 0.0) * 1e9,
            'canvas_shape':   list(self._obj_sum.shape),
            'n_frames_in':      self._n_in,
            'n_frames_accepted': self._n_accepted,
            'record_columns': _RECORD_COLUMNS,
        }
        try:
            rec = (np.asarray(self._records, dtype=np.float64)
                   if self._records else np.zeros((0, len(_RECORD_COLUMNS))))
            self._data_mutex.lock()
            try:
                written = save_accumulated_object(
                    path, self._obj_sum, self._contrib,
                    rec[:, 4:6], rec[:, 8:10],      # positions_um, final px
                    self._res_rpi_m, meta,
                    extra={
                        'frame_records':        rec,
                        'positions_px_nominal': rec[:, 6:8],
                        'alignment_shift_px':   rec[:, 10:12],
                        'normalization_offset_rad': rec[:, 12],
                        'frame_index':          rec[:, 0],
                        'source_index':         rec[:, 1],
                        'frame_timestamp':      rec[:, 2],
                        'frame_loss':           rec[:, 3],
                    })
            finally:
                self._data_mutex.unlock()
        except Exception as exc:
            self._log_exception('save')
            QMessageBox.critical(self, 'Save failed', str(exc))
            return
        self._log(f'Saved {Path(written).name}.')

    def _on_show_tracker(self):
        if self._contrib is None:
            return
        if self._tracker_win is None:
            self._tracker_win = _TrackerWindow(self)
            self._tracker_win.closed.connect(self._on_tracker_closed)
            # Fed from the same render pass as the object, so a live
            # contribution map costs one extra downsampled copy per redraw
            # and nothing on the GUI thread.
            self._render.set_want_tracker(True)
            self._log('Contribution map opened (live).')
        self._tracker_win.show()
        self._tracker_win.raise_()
        self._tracker_win.activateWindow()
        self._render.request(self._downsample())

    def _on_tracker_closed(self):
        self._tracker_win = None
        self._render.set_want_tracker(False)

    # ------------------------------------------------------------------ #
    # Shutdown                                                            #
    # ------------------------------------------------------------------ #

    def closeEvent(self, event):
        if self._closing:
            event.accept()
            return
        self._closing = True
        try:
            if self._worker is not None and self._worker.isRunning():
                self._worker.stop()
                self._worker.wait(10000)
        except Exception:
            pass
        try:
            self._render.stop()
        except Exception:
            pass
        if self._tracker_win is not None:
            self._tracker_win.close()
            self._tracker_win = None
        if self._pos_scatter_win is not None:
            self._pos_scatter_win.close()
            self._pos_scatter_win = None
        self._pos_dlg.close()
        if self._source is not None:
            self._source.close()
        try:
            # Hand the settings over: the main window may be saved long after
            # this one is gone.
            self._main.on_accumulator_closed(self.get_settings())
        except Exception:
            pass
        super().closeEvent(event)


class _TrackerWindow(QDialog):
    """Live view of the pixel-contribution tracker.

    Updated from the accumulator's render thread on the same cadence as the
    object itself, so watching overlap build up costs no extra full-canvas
    work and no extra GUI-thread time beyond the raster.
    """

    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('TrackerWindow')
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setWindowTitle('Pixel contribution (live)')
        self.resize(760, 800)

        layout = QVBoxLayout(self)
        self._plot = ImagePlotWidget(keep_points=True)
        self._plot.action_no_zoom_tracking.setChecked(True)
        self._plot.action_auto_colorscale.setChecked(False)   # levels come with each frame
        self._iv = self._plot.image_view
        self._iv.setObjectName('tracker_image_view')
        layout.addWidget(self._plot)
        self._info = QLabel('')
        self._info.setObjectName('tracker_info_label')
        self._info.setStyleSheet('font-family: monospace;')
        layout.addWidget(self._info)
        self._first = True

    def update_image(self, tracker, levels=None):
        if tracker is None or tracker.size == 0:
            return
        self._plot.set_image(tracker.T, autoRange=True if self._first else None)
        if not self._plot.action_auto_colorscale.isChecked():
            self._plot.set_levels(*(levels or (0.0, 1.0)))
        self._first = False
        hi = levels[1] if levels else float(tracker.max())
        self._info.setText(
            f'max {hi:.4g}   '
            f'{100.0 * np.count_nonzero(tracker) / tracker.size:.1f}% covered   '
            f'(displayed {tracker.shape[0]}x{tracker.shape[1]})')

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class _StagePositionsDialog(QDialog):
    """Where each frame's stage position comes from.

    Non-modal, so the diagnostic log stays visible while test-reading. The
    widgets are owned here but aliased onto ObjectAccumulatorWindow (see
    WIDGETS), which remains the one place that reads and persists them.
    """

    WIDGETS = (
        'label_pos_source', 'pos_source_combo',
        'label_h_pv', 'h_motor_pv_edit',
        'label_v_pv', 'v_motor_pv_edit',
        'label_pos_h5', 'pos_h5_file_edit', 'pos_h5_browse_btn',
        'label_motor_units', 'motor_units_combo',
        'test_read_btn',
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('StagePositionsDialog')
        self.setWindowTitle('Stage positions')
        self.setModal(False)
        self.setMinimumWidth(460)

        def _named(widget, name):
            widget.setObjectName(name)   # font_utils only reaches named widgets
            setattr(self, name, widget)
            return widget

        self._body = _named(QWidget(), 'pos_dlg_body')
        form = QFormLayout()
        form.addRow(_named(QLabel('Source:'), 'label_pos_source'),
                    _named(QComboBox(), 'pos_source_combo'))
        self.pos_source_combo.setToolTip(
            'Where each frame\'s stage position comes from. "QDS '
            'interferometer" picks, from a short history streamed by the '
            'qds_bridge, the position nearest the detector frame\'s '
            'timestamp (settings under Options ▸ QDS interferometer). '
            '"Simulated" reads no PVs at all and walks a deterministic raster '
            '— use it to exercise the whole pipeline offline. "All frames at '
            '(0, 0)" stacks every frame on one spot.')
        form.addRow(_named(QLabel('Horizontal PV:'), 'label_h_pv'),
                    _named(QLineEdit('12ideMCS1:m1.RBV'), 'h_motor_pv_edit'))
        self.h_motor_pv_edit.setToolTip('Read-back PV for the horizontal stage.')
        form.addRow(_named(QLabel('Vertical PV:'), 'label_v_pv'),
                    _named(QLineEdit('12ideMCS1:m2.RBV'), 'v_motor_pv_edit'))
        self.v_motor_pv_edit.setToolTip(
            'Read-back PV for the vertical stage. The default here is a '
            'placeholder — replace it with the real vertical motor.')

        h5_row = QHBoxLayout()
        h5_row.addWidget(_named(QLineEdit(), 'pos_h5_file_edit'))
        self.pos_h5_file_edit.setPlaceholderText(
            'H5 with ppX / ppY, or a ptychi recon (positions_px)…')
        self.pos_h5_file_edit.setToolTip(
            'H5 file holding the positions for an H5 pattern replay, in '
            'datasets "ppX" (horizontal) and "ppY" (vertical). Indexed by the '
            'same pattern index as the diffraction file. On opening, a '
            f'{_POS_H5_GLOB} beside the replayed H5 is picked automatically\n'
            '(or, for a raw master file, the file itself: '
            '/entry/sample/positions, (X, Z) in its "units" attribute).\n'
            'A ptychi reconstruction also works: "positions_px" (N x 2, '
            'row/col, fractional) times "obj_pixel_size_m". Its positions '
            'ignore the units setting and are placed with a '
            f'{_SUBPIXEL_DECIMALS}-decimal subpixel shift.')
        h5_row.addWidget(_named(QPushButton('...'), 'pos_h5_browse_btn'))
        self.pos_h5_browse_btn.setMaximumWidth(34)
        form.addRow(_named(QLabel('Position H5:'), 'label_pos_h5'), h5_row)

        form.addRow(_named(QLabel('Position units:'), 'label_motor_units'),
                    _named(QComboBox(), 'motor_units_combo'))
        self.motor_units_combo.setToolTip(
            'Units the PVs / ppX / ppY report in (ignored for a ptychi '
            'positions file). Everything downstream works '
            'in microns. Axis inversion and H/V swap live in the Options menu.')

        body_vl = QVBoxLayout(self._body)
        body_vl.setContentsMargins(0, 0, 0, 0)
        body_vl.addLayout(form)
        body_vl.addWidget(_named(QPushButton('Test position read'),
                                 'test_read_btn'))
        self.test_read_btn.setToolTip(
            'Read the configured position source once, right now, and report '
            'what came back. For an H5 source, also plot every position.')

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.setObjectName('pos_dlg_buttons')
        buttons.rejected.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(self._body)
        layout.addWidget(buttons)

    def set_editable(self, editable):
        """Freeze the controls during a run, leaving Close usable."""
        self._body.setEnabled(editable)


class _PositionScatterWindow(QDialog):
    """Every position in a position H5, as the accumulator will use them
    (microns, after the unit / invert / swap options)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('PositionScatterWindow')
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.resize(640, 640)

        layout = QVBoxLayout(self)
        self._plot = pg.PlotWidget()
        self._plot.setObjectName('pos_scatter_plot')
        self._plot.setLabel('bottom', 'x', units='µm')
        self._plot.setLabel('left', 'y (down)', units='µm')
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        # Same orientation as the stitched image, where y grows downwards.
        self._plot.invertY(True)
        self._plot.setAspectLocked(True)
        layout.addWidget(self._plot)

        self._all = pg.ScatterPlotItem(size=4, pen=None,
                                       brush=pg.mkBrush(70, 130, 220, 200))
        # Where the scan starts, so its direction can be read off the plot.
        self._first = pg.ScatterPlotItem(size=10, pen=pg.mkPen('k'),
                                         brush=pg.mkBrush('g'), symbol='s')
        self._plot.addItem(self._all)
        self._plot.addItem(self._first)

        self._info = QLabel('')
        self._info.setObjectName('pos_scatter_info_label')
        self._info.setStyleSheet('font-family: monospace;')
        layout.addWidget(self._info)

    def set_positions(self, x_um, y_um, title=''):
        x = np.asarray(x_um, dtype=np.float64)
        y = np.asarray(y_um, dtype=np.float64)
        self.setWindowTitle(f'Positions — {title}' if title else 'Positions')
        self._all.setData(x=x, y=y)
        self._first.setData(x=x[:1], y=y[:1])
        self._plot.autoRange()
        self._info.setText(
            f'n={x.size}   x [{x.min():+.3f}, {x.max():+.3f}] µm   '
            f'y [{y.min():+.3f}, {y.max():+.3f}] µm   (green = first)')
