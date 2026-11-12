"""
RPIWorker — QThread that runs the RPI event loop.

Three reconstruction backends share one acquisition/display scaffold:

  * CGWorker    — pure-PyTorch conjugate-gradient RPI (from_ALS/conjugate_gradient_rpi.py)
  * LBFGSWorker — cdtools two-stage LBFGS RPI
  * LSQMLWorker — ptychi LSQML on a single pattern at position (0, 0)
  * ChainWorker — a user-built list of steps from the three, run in order on
                  each frame with the object and probe handed step to step

All subclass RPIWorkerBase, which owns the common per-frame loop:
  1. Read a diffraction pattern (EPICS PV or H5 replay)
  2. Crop and apply the pre-cropped pixel mask
  3-4. Reconstruct (algorithm-specific, via self._reconstruct)
  5. Push the result into the shared deque and emit frame_ready

On any unhandled exception the error_occurred signal carries a full
diagnostic string: array shapes, the step where it failed, and the
traceback.
"""

import logging
import os
import sys
import threading
import time
import traceback

import numpy as np
import pvaccess as pva
import torch as t

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'from_ALS'))
from conjugate_gradient_rpi import iterate_CG

import cdtools
from cdtools.datasets import Ptycho2DDataset

import ptychi.api as ptychi_api
from ptychi.api.task import PtychographyTask

from PyQt6.QtCore import QThread, pyqtSignal

from rpi_chain import ALGO_LABELS
from rpi_utils import (
    DPReader,
    build_cxi_inmem,
    interp_probe,
    compute_pixel_sizes,
    energy_kev_to_wavelength,
)


# ptychi warns on every patch extraction when the patch reaches the object
# edge. In RPI the object *is* the probe footprint, so that is always true and
# the warning would flood the terminal once per epoch.
_DROP_LOG_PREFIXES = (
    'Patch extractor has to pad',
    'Patch placer has to pad',
)


class _DropLogMessages(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith(_DROP_LOG_PREFIXES)


# ptychi logs through the root logger; with no handlers configured that goes
# to logging.lastResort, so filter there as well as on any real handlers.
for _h in logging.getLogger().handlers + [logging.lastResort]:
    if _h is not None:
        _h.addFilter(_DropLogMessages())


_DTYPE_MAP = {
    pva.UBYTE:  np.uint8,   pva.BYTE:   np.int8,
    pva.USHORT: np.uint16,  pva.SHORT:  np.int16,
    pva.UINT:   np.uint32,  pva.INT:    np.int32,
    pva.ULONG:  np.uint64,  pva.LONG:   np.int64,
    pva.FLOAT:  np.float32, pva.DOUBLE: np.float64,
}


def _pv_timestamp(pv):
    """The NTNDArray timeStamp as Unix-epoch seconds, or None if absent.

    PVA's secondsPastEpoch counts from 1970, unlike Channel Access's 1990
    epoch, so this compares directly with time.time() on a synced machine.
    """
    try:
        ts = pv['timeStamp']
        return ts['secondsPastEpoch'] + ts['nanoseconds'] * 1e-9
    except Exception:
        return None


def _pv_to_image(pv):
    """Decode a PVA AreaDetector PvObject to a 2D numpy array (rows, cols)."""
    data_key = list(pv['value'][0].keys())[0]
    # pv['dimension'] is ordered [size_x, size_y] with x fastest-varying,
    # so the (rows, cols) shape for a C-order reshape is the reverse.
    shape    = tuple(dim['size'] for dim in pv['dimension'])[::-1]
    codec    = pv['codec']['name']

    if codec == '':
        arr = np.array(pv['value'][0][data_key])
    elif codec == 'lz4':
        import lz4.block
        dtype = _DTYPE_MAP[pv['codec']['parameters'][0]['value']]
        arr = np.frombuffer(
            lz4.block.decompress(pv['value'][0][data_key],
                                 uncompressed_size=pv['uncompressedSize']),
            dtype=dtype,
        )
    elif codec == 'bslz4':
        import bitshuffle
        dtype = _DTYPE_MAP[pv['codec']['parameters'][0]['value']]
        arr = bitshuffle.decompress_lz4(
            pv['value'][0][data_key],
            (pv['uncompressedSize'] // dtype().itemsize,),
            dtype,
        )
    elif codec == 'blosc':
        import blosc2
        dtype = _DTYPE_MAP[pv['codec']['parameters'][0]['value']]
        arr = np.frombuffer(blosc2.decompress(pv['value'][0][data_key]), dtype=dtype)
    else:
        raise ValueError(f'Unsupported codec: {codec!r}')

    return arr.reshape(shape)


# ══════════════════════════════════════════════════════════════════════════
# Base class — shared acquisition/display loop
# ══════════════════════════════════════════════════════════════════════════

class RPIWorkerBase(QThread):
    # Emitted once per completed RPI frame
    frame_ready      = pyqtSignal(dict)
    # Human-readable status updates for the GUI status bar
    status_update    = pyqtSignal(str)
    # Emitted on error; carries a full diagnostic string
    error_occurred   = pyqtSignal(str)
    # Emitted after each successful PV read; carries the full raw 2D image
    raw_image_ready  = pyqtSignal(object)

    def __init__(self, params, probe_raw, pixel_mask_full, results_deque, parent=None):
        """
        Parameters
        ----------
        params : dict
            All GUI parameter values (see _collect_params in rpi_live_gui.py).
        probe_raw : ndarray, shape (N_opr, N_modes, N, N), complex
            Full probe array; worker uses first OPR mode and interpolates if
            needed.
        pixel_mask_full : ndarray, 2-D float32
            Full-size detector pixel mask (1=live, 0=dead).
        results_deque : collections.deque
            Shared deque; worker appends entries, main thread reads them.
        """
        super().__init__(parent)
        self.params           = params
        self.probe_raw        = probe_raw
        self.pixel_mask_full  = pixel_mask_full
        self.results_deque    = results_deque
        self._running         = False
        self._frame_counter   = 0
        self._last_plot_call_id = None
        self._step            = 'not started'
        # 'H5 img i/N' while replaying an H5, else ''. Prefixed onto the
        # status line so the pattern being reconstructed is always visible.
        self._h5_tag          = ''
        # Set when params['h5_single_pass'] ended the run after one lap of
        # the H5 file, so the owner can tell that apart from a user stop.
        self.finished_h5_pass = False
        # True once an H5 replay file turns out to be a raw master file.
        self.h5_raw           = False

        # Optional hook, set by the Object Accumulator after construction:
        #   position_reader(frame_index, frame_ts) -> (x, y) or (x, y, source_index)
        # It is called in this thread right after each diffraction pattern is
        # acquired, so a blocking pyepics .get() is fine here. frame_index is
        # the H5 pattern index in replay mode and None when reading from a PV,
        # which lets an H5-backed position source stay aligned with the
        # patterns. frame_ts is the PV's timeStamp in Unix-epoch seconds, or
        # None in replay mode, so a timestamped source can pick the sample
        # taken with the frame. The result lands in entry['position']; a
        # failure lands in entry['position_error'] and does not stop the run.
        self.position_reader  = None

        # Mid-run probe swap (used by the "Propagate probe" window): the GUI
        # thread drops a new raw probe here and the hot loop picks it up before
        # the next frame. Guarded by a lock because the two threads touch it.
        self._probe_lock      = threading.Lock()
        self._pending_probe   = None    # (probe_raw, label) or None

        # Shared timing / diagnostic state
        self._total_time      = 0.0
        self._last_frame_s    = None
        self._device          = 'cuda'

        # One-time-setup products (also referenced by the diagnostic formatter)
        self._probe_interp    = None
        self._mask_crop       = None
        self._dp              = None
        self._res_rpi_m       = None
        self._wavelength      = None
        self._x_pix_cxi       = None
        self._N_modes         = None
        self._detector_distance = None
        self._crop_N          = None
        self._obj_shape       = None

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def stop(self):
        """Signal the loop to exit after the current frame finishes."""
        self._running = False

    def set_probe(self, probe_raw, label=''):
        """Replace the probe without stopping the run.

        Called from the GUI thread (the "Propagate probe" window). The new
        probe is interpolated and handed to the algorithm before the next
        frame starts; the frame in flight finishes with the old one.
        """
        with self._probe_lock:
            self._pending_probe = (probe_raw, label)

    # ------------------------------------------------------------------
    # Probe preparation
    # ------------------------------------------------------------------

    def _prepare_probe_interp(self):
        """Interpolate self.probe_raw to the crop size and select modes."""
        probe_interp = interp_probe(self.probe_raw, self._crop_N)
        n_sel = self.params['n_probe_modes']
        if n_sel and n_sel < probe_interp.shape[0]:
            probe_interp = probe_interp[:n_sel]
        self._probe_interp = probe_interp
        self._N_modes      = probe_interp.shape[0]

    def _apply_pending_probe(self):
        """Swap in a probe handed over by set_probe(), if there is one."""
        with self._probe_lock:
            pending, self._pending_probe = self._pending_probe, None
        if pending is None:
            return
        probe_raw, label = pending
        self._step = 'applying new probe'
        self.probe_raw = probe_raw
        self._prepare_probe_interp()
        self._on_probe_changed()
        self.status_update.emit(
            f'New probe applied mid-run{f" ({label})" if label else ""}.')

    def _on_probe_changed(self):
        """Hook for backends that cache something derived from the probe."""
        pass

    # ------------------------------------------------------------------
    # Main loop (shared)
    # ------------------------------------------------------------------

    def run(self):
        self._running = True
        self._step    = 'initializing'

        p = self.params
        _h5_fh = None   # DPReader for H5 replay; closed in finally

        try:
            device    = self._device
            crop_N    = p['crop_N']
            obj_shape = p['obj_shape']
            beam_cy   = p['beam_cy']
            beam_cx   = p['beam_cx']
            self._crop_N    = crop_N
            self._obj_shape = obj_shape

            half   = crop_N // 2
            y0, y1 = beam_cy - half, beam_cy + half
            x0, x1 = beam_cx - half, beam_cx + half

            # ---- one-time setup (shared) -------------------------------

            self._step = 'cropping pixel mask'
            self._mask_crop = self.pixel_mask_full[y0:y1, x0:x1].astype(np.float32)

            self._step = 'interpolating probe'
            self.status_update.emit('Preparing probe…')
            self._prepare_probe_interp()
            N_probe = self.probe_raw.shape[-1]

            self._step = 'computing pixel sizes'
            self._wavelength = energy_kev_to_wavelength(p['energy_kev'])
            self._x_pix_cxi, self._res_rpi_m = compute_pixel_sizes(
                p['res_m_probe'], N_probe, crop_N, obj_shape
            )
            self._detector_distance = p['detector_distance']

            # ---- one-time setup (algorithm-specific) -------------------
            self._step = 'algorithm setup'
            self._setup()

            if p['h5_mode']:
                self._step = 'opening H5 replay file'
                self.status_update.emit('H5 mode: opening file…')
                _h5_fh    = DPReader(p['h5_file'])
                self.h5_raw = _h5_fh.is_raw
                _n_h5_dp  = _h5_fh.n
                _h5_N     = _h5_fh.frame_shape[-1]   # spatial size of each stored pattern
                _h5_idx   = p['h5_start_idx']     # local frame counter (wraps around)
                _h5_stride = max(1, int(p.get('h5_stride', 1)))
                self.status_update.emit(
                    f'H5 file ready: {_n_h5_dp} '
                    f'{"raw frames" if self.h5_raw else "patterns"}, size '
                    f'{_h5_fh.frame_shape[0]}×{_h5_fh.frame_shape[1]}, '
                    f'starting at index {_h5_idx}, analyzing every '
                    f'{_h5_stride}{"" if _h5_stride == 1 else "-th"} pattern'
                )
            else:
                self._step = 'connecting to EPICS PV'
                channel = pva.Channel(p['pv_name'], pva.PVA)

            # Patterns averaged into each reconstruction. With 1 the
            # averaging block below is skipped entirely.
            n_avg     = max(1, int(p.get('n_avg', 1)))
            _avg_n    = 0           # patterns summed so far in this group
            _avg_sum  = None
            _avg_raw  = None        # full frames summed, shown in the pattern viewer
            _avg_pos  = []          # position read with each of them

            # ---- hot loop ----------------------------------------------
            while self._running:
                if _avg_n == 0:
                    frame_start = time.time()

                # 0. Pick up a probe handed over mid-run (re-propagation).
                if self._pending_probe is not None:
                    self._apply_pending_probe()

                # 1. Acquire diffraction pattern
                self._step = 'acquiring diffraction pattern'
                pos_index = None        # H5 pattern index, for the position hook
                frame_ts  = None        # detector timestamp, for the position hook
                if p['h5_mode']:
                    # One lap = the index has advanced by the file length,
                    # however far in it started and whatever the stride.
                    if (p.get('h5_single_pass')
                            and _h5_idx - p['h5_start_idx'] >= _n_h5_dp):
                        self.finished_h5_pass = True
                        self.status_update.emit(
                            f'Read all {_n_h5_dp} H5 patterns; stopping.')
                        break
                    pos_index = _h5_idx % _n_h5_dp
                    self._h5_tag = f'H5 img {pos_index}/{_n_h5_dp}'
                    self.status_update.emit(
                        f'{self._status_line1()}\n{self._h5_tag} | reading')
                    raw    = _h5_fh[pos_index].astype(np.float32)
                    _h5_idx += _h5_stride
                    if n_avg == 1:      # else the group average, below
                        self.raw_image_ready.emit(raw)
                    if self.h5_raw:
                        # Full detector frame: crop around the beam centre
                        # exactly as for a PV image.
                        dp = raw[y0:y1, x0:x1] * self._mask_crop
                    else:
                        # Pattern is already centered; crop further around its centre.
                        cy, cx = _h5_N // 2, _h5_N // 2
                        dp = raw[cy - half : cy + half, cx - half : cx + half] * self._mask_crop
                else:
                    try:
                        pv_obj = channel.get('field()')
                    except pva.PvaException as exc:
                        self.status_update.emit(f'PV read failed — retrying… ({exc})')
                        time.sleep(1)
                        continue
                    if pv_obj is None:
                        self.status_update.emit('PV returned None — retrying…')
                        time.sleep(1)
                        continue
                    try:
                        raw = _pv_to_image(pv_obj).astype(np.float32)
                    except Exception as exc:
                        self.status_update.emit(f'Frame decode failed — retrying… ({exc})')
                        time.sleep(1)
                        continue
                    if n_avg == 1:      # else the group average, below
                        self.raw_image_ready.emit(raw)

                    frame_id = pv_obj['uniqueId']
                    if not p['allow_repeat_frames'] and frame_id == self._last_plot_call_id:
                        time.sleep(0.05)
                        continue
                    self._last_plot_call_id = frame_id
                    frame_ts = _pv_timestamp(pv_obj)

                    dp  = raw[y0:y1, x0:x1] * self._mask_crop

                # 1b. Sample the stage position as close in time to the frame
                #     as possible. A failure here is reported per-frame rather
                #     than aborting the run — the accumulator decides whether
                #     a position-less frame is usable.
                position, position_error = None, None
                if self.position_reader is not None:
                    self._step = 'reading stage position'
                    try:
                        position = self.position_reader(pos_index, frame_ts)
                    except Exception as exc:
                        position_error = f'{type(exc).__name__}: {exc}'

                # 1c. Average n_avg patterns into one. The group's position is
                #     the mean of its positions; any failed read fails it.
                #     Frames are float32 already (cast on read), so neither
                #     the sums nor the division are integer arithmetic.
                if n_avg > 1:
                    if _avg_n == 0:
                        _avg_sum = dp.astype(np.float32, copy=True)
                        _avg_raw = raw.astype(np.float32, copy=True)
                    else:
                        _avg_sum += dp
                        _avg_raw += raw
                    _avg_pos.append((position, position_error))
                    _avg_n += 1
                    if _avg_n < n_avg:
                        self.status_update.emit(
                            f'{self._status_line1()}\n'
                            + (f'{self._h5_tag} | ' if self._h5_tag else '')
                            + f'averaging {_avg_n}/{n_avg}')
                        continue
                    dp  = _avg_sum / np.float32(n_avg)
                    # The pattern viewer (and a PV frame's saved raw image)
                    # gets the summed frames, not their average.
                    raw = _avg_raw
                    self.raw_image_ready.emit(raw)
                    errors = [e for _, e in _avg_pos if e is not None]
                    if self.position_reader is None:
                        position, position_error = None, None
                    elif errors:
                        position, position_error = None, errors[0]
                    else:
                        group = [pos for pos, _ in _avg_pos]
                        # (x, y, extra…): average x and y, keep the last
                        # pattern's extras (e.g. its H5 position index).
                        position = (float(np.mean([g[0] for g in group])),
                                    float(np.mean([g[1] for g in group])),
                                    *group[-1][2:])
                        position_error = None
                    _avg_n, _avg_sum, _avg_raw, _avg_pos = 0, None, None, []

                self._dp = dp

                if not self._running:
                    break

                # 2-4. Reconstruct (algorithm-specific). Returns
                #      (obj, loss, probe) or None to signal the loop was
                #      stopped mid-frame. probe is None unless the backend
                #      changed it this frame (refined or rescaled).
                result = self._reconstruct(dp)
                if result is None:
                    return
                obj, loss, probe_out = result

                # 5. Collect results and emit
                self._step = 'collecting results'
                self._last_frame_s  = time.time() - frame_start
                self._total_time   += self._last_frame_s

                entry = {
                    'obj':          obj,
                    'loss_history': loss,
                    'pixel_size_m': self._res_rpi_m,
                    'frame_idx':    self._frame_counter,
                    'source_idx':   pos_index,
                    'timestamp':    time.time(),
                    'detector_timestamp': frame_ts,
                    # The probe that produced obj: the refined one when the
                    # backend updated it, else the one handed in. The next
                    # frame starts from the handed-in probe either way.
                    'probe_interp': (probe_out if probe_out is not None
                                     else self._probe_interp),
                    'raw_image':    raw if not p['h5_mode'] else None,
                    'position':       position,
                    'position_error': position_error,
                    'n_averaged':     n_avg,
                }
                self.results_deque.append(entry)
                self.frame_ready.emit(entry)

                last_loss = loss[-1] if len(loss) else float('nan')
                n_done    = self._frame_counter + 1   # frames completed including this one
                avg       = self._total_time / n_done
                print(f'Frame {self._frame_counter} done | '
                      f'time={self._last_frame_s:.1f}s | avg={avg:.1f}s'
                      f'{self._extra_summary(n_done)} | loss={last_loss:.4g}',
                      flush=True)
                self.status_update.emit(
                    f'Average reconstruction time: {avg:.1f} s, '
                    f'last frame: {self._last_frame_s:.1f} s\n'
                    + (f'{self._h5_tag} | ' if self._h5_tag else '')
                    + f'Frame {self._frame_counter} complete | loss: {last_loss:.4g}'
                )
                self._frame_counter += 1

        except Exception:
            diag = self._format_diagnostic()
            self.error_occurred.emit(diag)
        finally:
            if _h5_fh is not None:
                _h5_fh.close()

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _status(self, stage, report):
        """Emit a two-line status string to the GUI."""
        # The H5 index leads the second line: the accumulator's monitor shows
        # only that line, truncated, so anything after it may be cut off.
        prefix = f'{self._h5_tag} | ' if self._h5_tag else ''
        self.status_update.emit(
            f'{self._status_line1()}\n{prefix}{stage}: {report}')

    def _status_line1(self):
        n = self._frame_counter
        if n == 0:
            return 'Waiting for first frame…'
        avg = self._total_time / n
        return (f'Average reconstruction time: {avg:.1f} s, '
                f'last frame: {self._last_frame_s:.1f} s')

    def _extra_summary(self, n_done):
        """Optional extra fields for the per-frame terminal summary line."""
        return ''

    # ------------------------------------------------------------------
    # Algorithm hooks — implemented by subclasses
    # ------------------------------------------------------------------

    def _setup(self):
        """One-time, algorithm-specific setup (tensors, buffers, params)."""
        raise NotImplementedError

    def _reconstruct(self, dp):
        """Reconstruct one frame.

        Return (obj_np, loss_array, probe_np_or_None), or None if stopped. The
        probe is returned only when the backend changed it this frame
        (refined, or rescaled as LSQML does), so it matches obj.
        """
        raise NotImplementedError

    def _algo_diagnostic_lines(self, p):
        """Algorithm-specific parameter lines for the diagnostic dump."""
        return []

    # ------------------------------------------------------------------
    # Diagnostic formatter (shared)
    # ------------------------------------------------------------------

    @staticmethod
    def _arr(label, a):
        if a is None:
            return f'  {label}: not yet computed'
        return f'  {label}: shape={a.shape}  dtype={a.dtype}'

    def _format_diagnostic(self):
        p = self.params
        lines = [
            f'Step when error occurred:  {self._step}',
            '',
            'Array shapes:',
            self._arr('probe_raw  (full loaded probe)', self.probe_raw),
            self._arr('probe_interp  (interpolated to crop_N)', self._probe_interp),
            self._arr('pixel_mask_full  (full detector mask)', self.pixel_mask_full),
            self._arr('mask_crop  (pre-cropped mask)', self._mask_crop),
            self._arr('dp  (diffraction pattern, cropped+masked)', self._dp),
            '',
            'Key parameters:',
            f'  method={p.get("method", "?")}',
            f'  crop_N={p["crop_N"]}  obj_shape={p["obj_shape"]}  obj_init={p.get("obj_init", "?")}',
            f'  beam_cy={p["beam_cy"]}  beam_cx={p["beam_cx"]}',
            f'  energy_kev={p["energy_kev"]}  detector_distance={p["detector_distance"]} m',
            f'  res_m_probe={p["res_m_probe"]:.4g} m  ({p["res_m_probe"]*1e9:.3f} nm)',
        ]
        lines += self._algo_diagnostic_lines(p)
        lines += ['', 'Traceback:', traceback.format_exc()]
        return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════
# Backend pieces shared with the chain worker
# ══════════════════════════════════════════════════════════════════════════

def _cg_initial_object(obj_init, obj_shape, device):
    """Fresh (1, n, n) CG object: flat, or small complex Gaussian noise."""
    if obj_init == 'random':
        return (
            t.randn(1, obj_shape, obj_shape, device=device)
            + 1j * t.randn(1, obj_shape, obj_shape, device=device)
        ).to(t.complex64) * 0.01
    return t.ones(1, obj_shape, obj_shape, device=device, dtype=t.complex64)


def _lbfgs_probe_out(model, results, probe_update):
    """The probe a cdtools RPI model ended with, or None if it was fixed."""
    if probe_update == 'full':
        return results['probe']
    if probe_update == 'weights':
        # The probe the model actually used: prs[i] = sum_j W[i, j] probe[j]
        weights = model.weights.detach().cpu().numpy()
        return np.einsum('ij,jyx->iyx', weights, results['probe'])
    return None


def _lsqml_options(wavelength, res_rpi_m, x_pix_cxi, step_scaler, momentum,
                   n_epochs, update_probe, probe_update_start,
                   rescale_first_epoch=True, all_mode_step_size=False):
    """ptychi LSQMLOptions for one pattern at position (0, 0)."""
    options = ptychi_api.LSQMLOptions()
    options.data_options.wavelength_m = wavelength

    options.object_options.pixel_size_m = res_rpi_m
    options.object_options.optimal_step_size_scaler = step_scaler
    # Rescales object and probe against each other; with one pattern it
    # only fights the (often fixed) probe.
    options.object_options.remove_object_probe_ambiguity.enabled = False

    # The probe keeps its own (finer) grid; ptychi resamples between them.
    options.probe_options.pixel_size_m = x_pix_cxi
    options.probe_options.optimizable = update_probe
    options.probe_options.optimization_plan.start = probe_update_start
    # Mode orthogonalization runs whether or not the probe is optimizable,
    # so it is tied to probe update rather than left at ptychi's default.
    options.probe_options.orthogonalize_incoherent_modes.enabled = update_probe
    options.probe_options.orthogonalize_incoherent_modes.optimization_plan.start = \
        probe_update_start
    options.probe_options.orthogonalize_opr_modes.enabled = False

    options.probe_position_options.optimizable = False

    r = options.reconstructor_options
    # The joint object/probe step-size solve is singular while the probe
    # is fixed.
    r.single_slice_solve_obj_prb_step_size_jointly = False
    # PATCH sampling diverges when the object is coarser than the probe
    # and covers the whole probe; FULL_OBJECT is ptychi's operator for
    # that case (AUTO only picks it for near-field data).
    r.forward_model_options.shift_operator = ptychi_api.ShiftOperators.FULL_OBJECT
    r.batch_size = 1
    r.batching_mode = ptychi_api.BatchingModes.RANDOM
    r.momentum_acceleration_gain = momentum
    r.num_epochs = n_epochs
    # With the default (True), epoch 0 only rescales the probe to the
    # pattern's intensity, which absorbs the normalization change from
    # interp_probe. A chain turns it off after its first step, whose object
    # has already absorbed that scale.
    r.rescale_probe_intensity_in_first_epoch = rescale_first_epoch
    # ptychi's default (as in PtychoShelves) solves the step sizes from probe
    # mode 0 alone, which is fine while it carries nearly all the power, as
    # in a loaded probe. A probe mixed by another backend (e.g. LBFGS mode
    # weights) can spread it evenly, and the step then overshoots and diverges.
    r.solve_step_sizes_only_using_first_probe_mode = not all_mode_step_size
    return options


def _lsqml_initial_object(obj_init, n):
    """Fresh (1, n, n) LSQML object: flat, or small complex Gaussian noise."""
    if obj_init == 'random':
        rng = np.random.default_rng()
        obj = (rng.standard_normal((1, n, n))
               + 1j * rng.standard_normal((1, n, n))) * 0.01
        return obj.astype(np.complex64)
    return np.ones((1, n, n), dtype=np.complex64)


# ══════════════════════════════════════════════════════════════════════════
# Conjugate-gradient backend
# ══════════════════════════════════════════════════════════════════════════

class CGWorker(RPIWorkerBase):
    """Pure-PyTorch conjugate-gradient RPI."""

    def _setup(self):
        p = self.params
        device = self._device
        # Convert probe and mask to GPU tensors once (the probe again on each
        # mid-run swap, via _on_probe_changed)
        self._step = 'moving probe and mask to GPU'
        self._on_probe_changed()
        self._mask_t  = t.tensor(self._mask_crop, device=device)

        self._n_iter      = p['n_iter']
        self._clear_every = p['clear_every']
        self._loss_target = p['loss_target']
        self._obj_init    = p.get('obj_init', 'flat')

        # Preallocated GPU buffer for the per-iteration loss. iterate_CG writes
        # into it device-side (no host sync); the curve is copied back once per
        # frame. Reused every frame — CG restarts iter at 0.
        self._loss_buf    = t.zeros(self._n_iter, device=device)

        self._cg_total_ms = 0.0   # cumulative CG-only time across completed frames
        self._last_cg_ms  = 0.0

    def _on_probe_changed(self):
        self._probe_t = t.tensor(self._probe_interp, dtype=t.complex64,
                                 device=self._device)

    def _reconstruct(self, dp):
        p          = self.params
        device     = self._device
        obj_shape  = self._obj_shape
        n_iter     = self._n_iter
        loss_buf   = self._loss_buf

        # 2. Prepare sqrtPattern — fftshift moves zero-freq to corner to match
        #    forward()'s FFT convention (no fftshift inside).
        self._step = 'preparing sqrtPattern'
        dp_shifted = np.fft.fftshift(dp)
        sqrt_pat   = np.sqrt(np.maximum(dp_shifted, 0.0))
        sqrt_pat_t = t.tensor(
            sqrt_pat[None], dtype=t.float32, device=device
        )  # (1, crop_N, crop_N)

        # 3. Initialise rec dict with a fresh object each frame
        self._step = 'initializing CG rec dict'
        obj_init = _cg_initial_object(self._obj_init, obj_shape, device)
        obj_init.requires_grad_(True)

        rec = {
            'obj':           obj_init,
            'probe':         self._probe_t,
            'sqrtPattern':   sqrt_pat_t,
            'mask':          self._mask_t,
            'iter':          0,
            'clearEvery':    self._clear_every,
            'last_grad':     None,
            'last_step_dir': None,
            'loss_buf':      loss_buf,
        }

        # 4. CG iteration loop
        self._step = 'CG optimization'
        loss_target = self._loss_target
        target_triggered = False
        cooldown_remaining = 0
        loss_now = float('nan')   # most recently host-synced loss value
        cg_start = time.time()
        for cg_i in range(n_iter):
            iterate_CG(rec)
            if not self._running:
                return None
            # Read the loss back to the host only periodically. Each read forces
            # a host/device stall, so doing it every iteration serialises the
            # pipeline; every 10 lets the GPU run ahead. The loss_target check
            # inherits this cadence, which is fine — the cooldown is also ~10.
            if cg_i == 0 or (cg_i + 1) % 10 == 0:
                loss_now = loss_buf[cg_i].item()
                elapsed_ms = (time.time() - cg_start) * 1000
                n_prev = self._frame_counter
                avg_ms = (self._cg_total_ms / n_prev) if n_prev > 0 else float('nan')
                avg_str = f'{avg_ms:.0f}ms' if n_prev > 0 else '---'
                print(
                    f'CG iter {cg_i + 1}/{n_iter}'
                    f'  loss={loss_now:.4g}'
                    f'  algo={elapsed_ms:.0f}ms'
                    f'  avg={avg_str}',
                    flush=True,
                )
            self._status('CG', f'iter {cg_i + 1}/{n_iter}  loss={loss_now:.4g}')
            if target_triggered:
                cooldown_remaining -= 1
                if cooldown_remaining == 0:
                    break
            elif loss_target > 0 and loss_now < loss_target:
                target_triggered = True
                cooldown_remaining = 10
        if device == 'cuda':
            t.cuda.synchronize()   # accurate algo timing + drains queue
        cg_elapsed_ms = (time.time() - cg_start) * 1000
        self._cg_total_ms += cg_elapsed_ms
        self._last_cg_ms   = cg_elapsed_ms

        # 5. Extract result
        obj  = rec['obj'][0].detach().cpu().numpy()
        n_cg_done = rec['iter']   # iterations actually run this frame
        loss = loss_buf[:n_cg_done].detach().cpu().numpy()
        return obj, loss, None   # CG never updates the probe

    def _extra_summary(self, n_done):
        avg_cg_ms = self._cg_total_ms / n_done
        return f' | cg={self._last_cg_ms:.0f}ms | cg_avg={avg_cg_ms:.0f}ms'

    def _algo_diagnostic_lines(self, p):
        return [
            f'  n_iter={p["n_iter"]}  clear_every={p["clear_every"]}',
            f'  loss_target={p["loss_target"]}',
        ]


# ══════════════════════════════════════════════════════════════════════════
# cdtools two-stage LBFGS backend
# ══════════════════════════════════════════════════════════════════════════

class LBFGSWorker(RPIWorkerBase):
    """cdtools RPI model with two-stage LBFGS optimisation."""

    def _setup(self):
        p = self.params
        self._reg1    = p['reg_factor1']
        self._reg2    = [p['reg_factor2_lo'], p['reg_factor2_hi']]
        self._lr1     = p['lr1']
        self._lr2     = p['lr2']
        self._n_iter1 = p['n_iter1']
        self._n_iter2 = p['n_iter2']
        self._obj_init = p.get('obj_init', 'flat')
        # 'off', 'weights' (cdtools' N_modes x N_modes mode-mixing matrix) or
        # 'full' (every probe pixel)
        self._probe_update       = p.get('lbfgs_probe_update', 'off')
        self._probe_update_start = int(p.get('lbfgs_probe_update_start', 10))

    def _segments(self):
        """The two LBFGS stages as (stage, n_iter, lr, reg, probe_on) runs.

        With probe update on, the stage containing iteration k (counted
        across both stages) is split there: the probe (or its mode weights)
        only becomes trainable in a fresh LBFGS_optimize call, since each call
        builds its optimizer from model.parameters() at that moment.
        """
        stages = [('Stage 1', self._n_iter1, self._lr1, [self._reg1, self._reg1]),
                  ('Stage 2', self._n_iter2, self._lr2, self._reg2)]
        k = self._probe_update_start if self._probe_update != 'off' else None
        segments, done = [], 0
        for stage, n, lr, reg in stages:
            if k is not None and done < k < done + n:
                segments.append((stage, k - done, lr, reg, False))
                segments.append((stage, done + n - k, lr, reg, True))
            elif n > 0:
                segments.append((stage, n, lr, reg, k is not None and done >= k))
            done += n
        return segments

    def _reconstruct(self, dp):
        device   = self._device
        patterns = dp[None]                        # (1, crop_N, crop_N)

        # 2. Build in-memory CXI and load dataset
        self._step = 'building in-memory CXI / loading dataset'
        cxi_mem = build_cxi_inmem(
            patterns, self._wavelength, self._detector_distance, self._x_pix_cxi
        )
        dataset = Ptycho2DDataset.from_cxi(cxi_mem)
        cxi_mem.close()

        if not self._running:
            return None

        self._step = 'moving dataset to GPU'
        dataset.get_as(device=device)

        # 3. Cold-start RPI model
        self._step = 'initializing RPI model (from_dataset)'
        model = cdtools.models.RPI.from_dataset(
            dataset,
            self._probe_interp,
            obj_size=[self._obj_shape, self._obj_shape],
            n_modes=self._N_modes,
            mask=self._mask_crop,
            initialization=('random' if self._obj_init == 'random' else 'uniform'),
        )
        model.to(device=device)

        # 4. Two-stage LBFGS optimisation. cdtools keeps the probe as a fixed
        #    buffer and the mode-mixing matrix (identity) as a frozen
        #    parameter. At iteration k, 'full' promotes the probe to a
        #    parameter (nn.Module.__setattr__ moves it out of the buffers) and
        #    'weights' unfreezes the mixing matrix.
        for stage, n_iter, lr, reg, probe_on in self._segments():
            if probe_on and self._probe_update == 'full' \
                    and not isinstance(model.probe, t.nn.Parameter):
                model.probe = t.nn.Parameter(model.probe.detach().clone())
            if probe_on and self._probe_update == 'weights':
                model.weights.requires_grad_(True)
            tag = f' (probe {self._probe_update})' if probe_on else ''
            self._step = f'LBFGS {stage.lower()} optimization{tag}'
            for _ in model.LBFGS_optimize(
                n_iter, dataset, lr=lr, regularization_factor=reg,
            ):
                self._status(stage + tag, model.report())
                if not self._running:
                    return None
            if not self._running:
                return None

        # 5. Collect results
        self._step = 'collecting results'
        results = model.save_results()
        probe = _lbfgs_probe_out(model, results, self._probe_update)
        return results['obj'], results['loss_history'], probe

    def _status(self, stage, report):
        # LBFGS also mirrors each stage report to the terminal.
        super()._status(stage, report)
        print(f'{stage}: {report}', flush=True)

    def _algo_diagnostic_lines(self, p):
        return [
            f'  n_iter1={p["n_iter1"]}  n_iter2={p["n_iter2"]}',
            f'  lr1={p["lr1"]}  lr2={p["lr2"]}',
            f'  probe_update={p.get("lbfgs_probe_update")}  '
            f'probe_update_start={p.get("lbfgs_probe_update_start")}',
        ]


# ══════════════════════════════════════════════════════════════════════════
# ptychi LSQML backend
# ══════════════════════════════════════════════════════════════════════════

# LSQML early stop: the loss is compared with its value this many epochs back.
_LSQML_EARLY_STOP_WINDOW = 10


class LSQMLWorker(RPIWorkerBase):
    """ptychi LSQML on one diffraction pattern at position (0, 0).

    Same geometry as the RPI backends: the probe sits on the crop_N grid at
    x_pix_cxi per pixel, and the object is obj_shape pixels of res_rpi_m, so
    both span the same field of view. ptychi resamples the object onto the
    probe grid; there is no padding and no position correction.
    """

    def _setup(self):
        p = self.params
        self._n_epochs           = p['lsqml_n_epochs']
        self._step_scaler        = p['lsqml_step_scaler']
        self._momentum           = p['lsqml_momentum']
        self._update_probe       = bool(p.get('lsqml_update_probe', False))
        self._probe_update_start = int(p.get('lsqml_probe_update_start', 100))
        # Stop once the loss changes by no more than this fraction over
        # _LSQML_EARLY_STOP_WINDOW epochs; 0 = always run every epoch.
        self._early_stop_frac    = float(p.get('lsqml_early_stop_pct', 0.0)) / 100.0
        self._obj_init           = p.get('obj_init', 'flat')

        self._lsqml_total_ms = 0.0   # cumulative LSQML-only time across completed frames
        self._last_lsqml_ms  = 0.0
        self._last_epochs    = 0     # epochs actually run in the last frame

    def _build_options(self):
        return _lsqml_options(
            self._wavelength, self._res_rpi_m, self._x_pix_cxi,
            step_scaler=self._step_scaler, momentum=self._momentum,
            n_epochs=self._n_epochs, update_probe=self._update_probe,
            probe_update_start=self._probe_update_start)

    def _initial_object(self):
        return _lsqml_initial_object(self._obj_init, self._obj_shape)

    def _reconstruct(self, dp):
        # ptychi sets torch's default device (and dtype) for the calling
        # thread and leaves it set. Restore it so code that runs later in
        # this thread, e.g. cdtools building CPU tensors, is unaffected.
        prev_device = t.get_default_device()
        try:
            return self._reconstruct_lsqml(dp)
        finally:
            t.set_default_device(None if prev_device.type == 'cpu' else prev_device)

    def _reconstruct_lsqml(self, dp):
        n_epochs = self._n_epochs

        # 2. Build a fresh task every frame (cheap: ~10 ms)
        self._step = 'building ptychi task'
        task = PtychographyTask(
            self._build_options(),
            diffraction_data    = dp[None].astype(np.float32),
            object_data         = self._initial_object(),
            probe_data          = self._probe_interp[None].astype(np.complex64),
            probe_position_x_px = np.zeros(1, dtype=np.float32),
            probe_position_y_px = np.zeros(1, dtype=np.float32),
            valid_pixel_mask    = self._mask_crop > 0,
        )

        # 3. One epoch at a time so Stop is honoured and progress is shown.
        #    Early stop compares the loss with its value W epochs back. Epoch 0
        #    only rescales the probe, so it is never the reference; with probe
        #    update on, the check waits until the probe has been refined for
        #    W epochs, since the loss usually plateaus before that starts.
        self._step = 'LSQML optimization'
        window   = _LSQML_EARLY_STOP_WINDOW
        earliest = window + 1
        if self._update_probe:
            earliest = max(earliest, self._probe_update_start + window)
        loss_hist = []
        lsqml_start = time.time()
        for ep in range(n_epochs):
            task.run(1)
            if not self._running:
                return None
            loss_now = float(task.reconstructor.loss_tracker.table['loss'].iloc[-1])
            loss_hist.append(loss_now)
            stalled = (self._early_stop_frac > 0 and ep >= earliest
                       and abs(loss_now - loss_hist[ep - window])
                       <= self._early_stop_frac * abs(loss_hist[ep - window]))
            if ep == 0 or (ep + 1) % 10 == 0 or stalled:
                elapsed_ms = (time.time() - lsqml_start) * 1000
                print(f'LSQML epoch {ep + 1}/{n_epochs}'
                      f'  loss={loss_now:.4g}  algo={elapsed_ms:.0f}ms', flush=True)
            probe_tag = (' (probe)' if self._update_probe
                         and ep >= self._probe_update_start else '')
            self._status('LSQML' + probe_tag,
                         f'epoch {ep + 1}/{n_epochs}  loss={loss_now:.4g}')
            if stalled:
                print(f'LSQML stopped early at epoch {ep + 1}/{n_epochs}: loss changed '
                      f'<= {self._early_stop_frac * 100:g}% over {window} epochs', flush=True)
                break
        self._last_epochs = len(loss_hist)
        lsqml_elapsed_ms = (time.time() - lsqml_start) * 1000
        self._lsqml_total_ms += lsqml_elapsed_ms
        self._last_lsqml_ms   = lsqml_elapsed_ms

        # 4. Extract result
        self._step = 'collecting results'
        obj  = task.get_data_to_cpu('object', as_numpy=True)[0]
        loss = task.reconstructor.loss_tracker.table['loss'].to_numpy(dtype=np.float64)
        # Always hand back ptychi's probe, even when it was not optimized:
        # epoch 0 rescales its intensity to the pattern, and obj only matches
        # the data together with that rescaled probe.
        probe = task.get_data_to_cpu('probe', as_numpy=True)[0]
        del task
        return obj, loss, probe

    def _extra_summary(self, n_done):
        avg_ms = self._lsqml_total_ms / n_done
        return (f' | epochs={self._last_epochs}/{self._n_epochs}'
                f' | lsqml={self._last_lsqml_ms:.0f}ms | lsqml_avg={avg_ms:.0f}ms')

    def _algo_diagnostic_lines(self, p):
        return [
            f'  lsqml_n_epochs={p.get("lsqml_n_epochs")}  '
            f'step_scaler={p.get("lsqml_step_scaler")}  momentum={p.get("lsqml_momentum")}  '
            f'early_stop_pct={p.get("lsqml_early_stop_pct")}',
            f'  update_probe={p.get("lsqml_update_probe")}  '
            f'probe_update_start={p.get("lsqml_probe_update_start")}',
            f'  probe pixel={self._x_pix_cxi}  object pixel={self._res_rpi_m}',
        ]


# ══════════════════════════════════════════════════════════════════════════
# Custom chain of the backends above
# ══════════════════════════════════════════════════════════════════════════

def _match_obj_modes(obj, n_modes):
    """obj (J, n, n) resized to n_modes incoherent modes.

    The total object power sum_j |O_j|^2 is kept, so the simulated intensity
    (and the loss) carries across the hand-off: growing copies mode 0 into
    the new modes and scales them all down; shrinking keeps the leading
    modes and scales them up. One mode split into J equal copies of O/sqrt(J)
    is the symmetric start cdtools' uniform initialisation also uses.
    """
    j = obj.shape[0]
    if j == n_modes:
        return obj
    if n_modes > j:
        out = np.concatenate([obj, np.repeat(obj[:1], n_modes - j, axis=0)])
    else:
        out = obj[:n_modes]
    p_in, p_out = np.sum(np.abs(obj) ** 2), np.sum(np.abs(out) ** 2)
    scale = np.sqrt(p_in / p_out) if p_out > 0 else 1.0
    return (out * scale).astype(np.complex64)


def _fourier_shift(obj, d):
    """obj (J, n, n) moved by d pixels along both axes (periodic, sub-pixel)."""
    k = np.fft.fftfreq(obj.shape[-1])
    ramp = np.exp(-2j * np.pi * d * (k[:, None] + k[None, :]))
    return np.fft.ifft2(np.fft.fft2(obj, axes=(-2, -1)) * ramp,
                        axes=(-2, -1)).astype(np.complex64)


class ChainWorker(RPIWorkerBase):
    """A custom chain of CG / LSQML / LBFGS steps (see rpi_chain.py).

    Every frame runs params['chain_steps'] in order. The first step starts
    from obj_init exactly as its standalone backend would, so a one-row chain
    reproduces that backend; each later step starts from the object and probe
    the previous one ended with. LBFGS Incoh Obj uses N_modes object modes,
    every other step one (see _match_obj_modes).

    The three backends Fourier-interpolate the object onto the probe grid
    but scale and place it differently, so the same array gives different
    exit waves. Between steps the object is kept in cdtools' convention and
    converted at each step's boundary (_to_backend / _from_backend); the
    frame's result is returned in the last step's convention, matching what
    that backend shows on its own.
    """

    def _setup(self):
        p = self.params
        self._steps    = [dict(s) for s in p['chain_steps']]
        self._obj_init = p.get('obj_init', 'flat')
        self._step_k   = None    # index of the running step, for the diagnostic
        self._last_step_iters = []
        self._dataset  = None    # cdtools dataset, built once per frame
        cg_iters = [s['n_iter'] for s in self._steps if s['algo'] == 'CG']
        if cg_iters:
            self._step = 'moving mask to GPU'
            self._mask_t   = t.tensor(self._mask_crop, device=self._device)
            # Shared by every CG step; each restarts its iteration count at 0.
            self._loss_buf = t.zeros(max(cg_iters), device=self._device)

    # Object conventions, relative to cdtools' (object n x n, probe M x M):
    #   * cdtools upsamples keeping the object's values, then an ortho FFT.
    #   * The vendored CG interaction leaves out cdtools' sqrt(M^2/n^2)
    #     rescale, so its object is M/n larger for the same exit wave.
    #   * ptychi keeps values but uses an unnormalized FFT (M times the ortho
    #     one), so its object is M times smaller. It also centres the object
    #     geometrically, (n-1)/2 on (M-1)/2, while Fourier padding lines up
    #     pixel 0 on pixel 0: its object sits (1 - n/M)/2 object pixels away.
    # Checked by comparing the backends' forward models on one object.

    def _to_backend(self, obj, algo):
        """A chain object (cdtools convention) as `algo` takes it."""
        if algo == 'CG':
            return (obj * (self._crop_N / self._obj_shape)).astype(np.complex64)
        if algo == 'LSQML':
            d = 0.5 * (1.0 - self._obj_shape / self._crop_N)
            return _fourier_shift(obj, -d) / np.float32(self._crop_N)
        return obj

    def _from_backend(self, obj, algo):
        """An object from `algo`, in the chain's (cdtools) convention."""
        if algo == 'CG':
            return (obj * (self._obj_shape / self._crop_N)).astype(np.complex64)
        if algo == 'LSQML':
            d = 0.5 * (1.0 - self._obj_shape / self._crop_N)
            return _fourier_shift(obj * np.float32(self._crop_N), d)
        return obj

    def _reconstruct(self, dp):
        obj   = None                  # (J, n, n) complex64; None before step 0
        probe = self._probe_interp
        losses, iters = [], []
        self._dataset = None
        for k, s in enumerate(self._steps):
            self._step_k = k
            tag = f'Step {k + 1}/{len(self._steps)} {ALGO_LABELS[s["algo"]]}'
            run = {'CG':          self._step_cg,
                   'LSQML':       self._step_lsqml,
                   'LBFGS_INCOH': self._step_lbfgs,
                   'LBFGS':       self._step_lbfgs}[s['algo']]
            out = run(s, k, tag, dp, obj, probe)
            if out is None:
                return None
            obj, probe, loss = out
            loss = np.asarray(loss, dtype=np.float64).ravel()
            losses.append(loss)
            iters.append(len(loss))
        self._step_k = None
        self._dataset = None
        self._last_step_iters = iters
        loss = np.concatenate(losses) if losses else np.zeros(0)
        obj_out = self._to_backend(obj, self._steps[-1]['algo'])[0]
        # The probe only goes back when a step changed it (LSQML always
        # rescales or refines it; LBFGS when its probe update is on).
        return obj_out, loss, (None if probe is self._probe_interp else probe)

    # ------------------------------------------------------------------
    # Steps: each takes and returns the object in cdtools' convention, as
    # (obj (J, n, n), probe, loss), or returns None if stopped
    # ------------------------------------------------------------------

    def _step_cg(self, s, k, tag, dp, obj, probe):
        device   = self._device
        n_iter   = s['n_iter']
        loss_buf = self._loss_buf

        self._step = f'{tag}: preparing sqrtPattern'
        # fftshift moves zero-freq to the corner, matching forward()
        sqrt_pat = np.sqrt(np.maximum(np.fft.fftshift(dp), 0.0))
        sqrt_pat_t = t.tensor(sqrt_pat[None], dtype=t.float32, device=device)

        if obj is None:
            obj_t = _cg_initial_object(self._obj_init, self._obj_shape, device)
        else:
            obj_t = t.tensor(self._to_backend(_match_obj_modes(obj, 1), 'CG'),
                             dtype=t.complex64, device=device)
        obj_t.requires_grad_(True)
        rec = {
            'obj':           obj_t,
            'probe':         t.tensor(probe, dtype=t.complex64, device=device),
            'sqrtPattern':   sqrt_pat_t,
            'mask':          self._mask_t,
            'iter':          0,
            'clearEvery':    s['clear_every'],
            'last_grad':     None,
            'last_step_dir': None,
            'loss_buf':      loss_buf,
        }

        # Same loop as CGWorker: loss read back every 10 iterations, and a
        # loss_target hit ends the step after a 10-iteration cooldown.
        self._step = f'{tag}: CG optimization'
        loss_target = s['loss_target']
        target_triggered = False
        cooldown_remaining = 0
        loss_now = float('nan')
        for cg_i in range(n_iter):
            iterate_CG(rec)
            if not self._running:
                return None
            if cg_i == 0 or (cg_i + 1) % 10 == 0:
                loss_now = loss_buf[cg_i].item()
                print(f'{tag} iter {cg_i + 1}/{n_iter}  loss={loss_now:.4g}',
                      flush=True)
            self._status(tag, f'iter {cg_i + 1}/{n_iter}  loss={loss_now:.4g}')
            if target_triggered:
                cooldown_remaining -= 1
                if cooldown_remaining == 0:
                    break
            elif loss_target > 0 and loss_now < loss_target:
                target_triggered = True
                cooldown_remaining = 10

        obj_out = self._from_backend(rec['obj'].detach().cpu().numpy(), 'CG')
        loss = loss_buf[:rec['iter']].detach().cpu().numpy()
        return obj_out, probe, loss

    def _step_lsqml(self, s, k, tag, dp, obj, probe):
        # ptychi sets torch's default device for this thread and leaves it
        # set; restore it so a cdtools step later in the chain is unaffected.
        prev_device = t.get_default_device()
        try:
            return self._step_lsqml_run(s, k, tag, dp, obj, probe)
        finally:
            t.set_default_device(None if prev_device.type == 'cpu' else prev_device)

    def _step_lsqml_run(self, s, k, tag, dp, obj, probe):
        n_epochs     = s['n_iter']
        update_probe = s['probe_update'] == 'full'
        # Only the first step rescales the probe to the pattern in epoch 0, as
        # the standalone backend does. A later step's object already carries
        # that scale, so rescaling the probe again would double it.
        rescale = (k == 0)
        pu_start = 1 if rescale else 0
        # Step sizes from all probe modes once an earlier step has changed
        # the probe (see _lsqml_options); a loaded probe keeps ptychi's
        # default, so a one-row chain still matches LSQMLWorker.
        probe_changed = probe is not self._probe_interp

        self._step = f'{tag}: building ptychi task'
        task = PtychographyTask(
            _lsqml_options(
                self._wavelength, self._res_rpi_m, self._x_pix_cxi,
                step_scaler=s['step_size'], momentum=s['momentum'],
                n_epochs=n_epochs, update_probe=update_probe,
                probe_update_start=pu_start, rescale_first_epoch=rescale,
                all_mode_step_size=probe_changed),
            diffraction_data    = dp[None].astype(np.float32),
            object_data         = (_lsqml_initial_object(self._obj_init, self._obj_shape)
                                   if obj is None
                                   else self._to_backend(_match_obj_modes(obj, 1), 'LSQML')),
            probe_data          = probe[None].astype(np.complex64),
            probe_position_x_px = np.zeros(1, dtype=np.float32),
            probe_position_y_px = np.zeros(1, dtype=np.float32),
            valid_pixel_mask    = self._mask_crop > 0,
        )

        # Early stop as in LSQMLWorker; epoch 0 is only a usable reference
        # when it did not just rescale the probe.
        self._step = f'{tag}: LSQML optimization'
        early_stop_frac = s['early_stop_pct'] / 100.0
        window   = _LSQML_EARLY_STOP_WINDOW
        earliest = window + 1 if rescale else window
        if update_probe:
            earliest = max(earliest, pu_start + window)
        loss_hist = []
        for ep in range(n_epochs):
            task.run(1)
            if not self._running:
                return None
            loss_now = float(task.reconstructor.loss_tracker.table['loss'].iloc[-1])
            loss_hist.append(loss_now)
            stalled = (early_stop_frac > 0 and ep >= earliest
                       and abs(loss_now - loss_hist[ep - window])
                       <= early_stop_frac * abs(loss_hist[ep - window]))
            if ep == 0 or (ep + 1) % 10 == 0 or stalled:
                print(f'{tag} epoch {ep + 1}/{n_epochs}  loss={loss_now:.4g}',
                      flush=True)
            self._status(tag, f'epoch {ep + 1}/{n_epochs}  loss={loss_now:.4g}')
            if stalled:
                print(f'{tag} stopped early at epoch {ep + 1}/{n_epochs}', flush=True)
                break

        self._step = f'{tag}: collecting results'
        obj_out = self._from_backend(
            task.get_data_to_cpu('object', as_numpy=True)[:1], 'LSQML')
        loss    = task.reconstructor.loss_tracker.table['loss'].to_numpy(dtype=np.float64)
        probe   = task.get_data_to_cpu('probe', as_numpy=True)[0]
        del task
        return obj_out, probe, loss

    def _lbfgs_dataset(self, dp):
        """The frame's cdtools dataset, built on first use and then shared."""
        if self._dataset is None:
            self._step = 'building in-memory CXI / loading dataset'
            cxi_mem = build_cxi_inmem(
                dp[None], self._wavelength, self._detector_distance, self._x_pix_cxi)
            dataset = Ptycho2DDataset.from_cxi(cxi_mem)
            cxi_mem.close()
            dataset.get_as(device=self._device)
            self._dataset = dataset
        return self._dataset

    def _step_lbfgs(self, s, k, tag, dp, obj, probe):
        device  = self._device
        incoh   = s['algo'] == 'LBFGS_INCOH'
        n_modes = self._N_modes if incoh else 1
        # cdtools' regularizer: factors[0] on object mode 0, factors[1] on
        # the others (unused with a single mode).
        reg = list(s['reg']) if incoh else [s['reg'], s['reg']]
        probe_update = s['probe_update']

        dataset = self._lbfgs_dataset(dp)
        if not self._running:
            return None

        self._step = f'{tag}: initializing RPI model (from_dataset)'
        model = cdtools.models.RPI.from_dataset(
            dataset,
            probe,
            obj_size=[self._obj_shape, self._obj_shape],
            n_modes=n_modes,
            mask=self._mask_crop,
            initialization=('random' if self._obj_init == 'random' else 'uniform'),
        )
        if obj is not None:
            model.obj.data = t.as_tensor(_match_obj_modes(obj, n_modes),
                                         dtype=model.obj.dtype)
        model.to(device=device)
        # As in LBFGSWorker: 'full' promotes the probe buffer to a parameter,
        # 'weights' unfreezes the mode-mixing matrix.
        if probe_update == 'full':
            model.probe = t.nn.Parameter(model.probe.detach().clone())
        elif probe_update == 'weights':
            model.weights.requires_grad_(True)

        label = tag + (f' (probe {probe_update})' if probe_update != 'off' else '')
        self._step = f'{label}: LBFGS optimization'
        for _ in model.LBFGS_optimize(
            s['n_iter'], dataset, lr=s['step_size'], regularization_factor=reg,
        ):
            report = model.report()
            self._status(label, report)
            print(f'{label}: {report}', flush=True)
            if not self._running:
                return None

        self._step = f'{tag}: collecting results'
        results = model.save_results()
        # save_results keeps only object mode 0; the next step gets them all.
        obj_out = model.obj.detach().cpu().numpy().astype(np.complex64)
        probe_out = _lbfgs_probe_out(model, results, probe_update)
        return (obj_out, probe if probe_out is None else probe_out,
                results['loss_history'])

    def _extra_summary(self, n_done):
        return f' | step iters={"/".join(str(n) for n in self._last_step_iters)}'

    def _algo_diagnostic_lines(self, p):
        lines = [f'  chain: {len(p.get("chain_steps", []))} steps'
                 + (f', failed in step {self._step_k + 1}'
                    if getattr(self, '_step_k', None) is not None else '')]
        for i, s in enumerate(p.get('chain_steps', [])):
            fields = '  '.join(f'{key}={s[key]}' for key in s if key != 'algo')
            lines.append(f'    {i + 1}. {ALGO_LABELS.get(s["algo"], s["algo"])}  {fields}')
        lines.append(f'  probe pixel={self._x_pix_cxi}  object pixel={self._res_rpi_m}')
        return lines


# Method-string → worker class. Used by the GUI to pick a backend at Start.
WORKER_CLASSES = {
    'CG':    CGWorker,
    'LBFGS': LBFGSWorker,
    'LSQML': LSQMLWorker,
    'CHAIN': ChainWorker,
}
