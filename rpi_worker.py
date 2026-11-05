"""
RPIWorker — QThread that runs the RPI event loop.

Two reconstruction backends share one acquisition/display scaffold:

  * CGWorker    — pure-PyTorch conjugate-gradient RPI (from_ALS/conjugate_gradient_rpi.py)
  * LBFGSWorker — cdtools two-stage LBFGS RPI

Both subclass RPIWorkerBase, which owns the common per-frame loop:
  1. Read a diffraction pattern (EPICS PV or H5 replay)
  2. Crop and apply the pre-cropped pixel mask
  3-4. Reconstruct (algorithm-specific, via self._reconstruct)
  5. Push the result into the shared deque and emit frame_ready

On any unhandled exception the error_occurred signal carries a full
diagnostic string: array shapes, the step where it failed, and the
traceback.
"""

import os
import sys
import threading
import time
import traceback

import h5py
import numpy as np
import pvaccess as pva
import torch as t

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'from_ALS'))
from conjugate_gradient_rpi import iterate_CG

import cdtools
from cdtools.datasets import Ptycho2DDataset

from PyQt5.QtCore import QThread, pyqtSignal

from rpi_utils import (
    build_cxi_inmem,
    interp_probe,
    compute_pixel_sizes,
    energy_kev_to_wavelength,
)


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
        _h5_fh = None   # h5py file handle; closed in finally

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
                _h5_fh    = h5py.File(p['h5_file'], 'r')
                _h5_dset  = _h5_fh['dp']          # shape (N_dp, N, N)
                _n_h5_dp  = _h5_dset.shape[0]
                _h5_N     = _h5_dset.shape[-1]    # spatial size of each stored pattern
                _h5_idx   = p['h5_start_idx']     # local frame counter (wraps around)
                _h5_stride = max(1, int(p.get('h5_stride', 1)))
                self.status_update.emit(
                    f'H5 file ready: {_n_h5_dp} patterns, size {_h5_N}×{_h5_N}, '
                    f'starting at index {_h5_idx}, analyzing every '
                    f'{_h5_stride}{"" if _h5_stride == 1 else "-th"} pattern'
                )
            else:
                self._step = 'connecting to EPICS PV'
                channel = pva.Channel(p['pv_name'], pva.PVA)

            # ---- hot loop ----------------------------------------------
            while self._running:
                frame_start = time.time()

                # 0. Pick up a probe handed over mid-run (re-propagation).
                if self._pending_probe is not None:
                    self._apply_pending_probe()

                # 1. Acquire diffraction pattern
                self._step = 'acquiring diffraction pattern'
                pos_index = None        # H5 pattern index, for the position hook
                frame_ts  = None        # detector timestamp, for the position hook
                if p['h5_mode']:
                    pos_index = _h5_idx % _n_h5_dp
                    self._h5_tag = f'H5 img {pos_index}/{_n_h5_dp}'
                    self.status_update.emit(
                        f'{self._status_line1()}\n{self._h5_tag} | reading')
                    raw    = _h5_dset[pos_index].astype(np.float32)
                    _h5_idx += _h5_stride
                    self.raw_image_ready.emit(raw)
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
                    self.raw_image_ready.emit(raw)

                    frame_id = pv_obj['uniqueId']
                    if not p['allow_repeat_frames'] and frame_id == self._last_plot_call_id:
                        time.sleep(0.05)
                        continue
                    self._last_plot_call_id = frame_id
                    frame_ts = _pv_timestamp(pv_obj)

                    dp  = raw[y0:y1, x0:x1] * self._mask_crop

                self._dp = dp

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

                if not self._running:
                    break

                # 2-4. Reconstruct (algorithm-specific). Returns (obj, loss) or
                #      None to signal the loop was stopped mid-frame.
                result = self._reconstruct(dp)
                if result is None:
                    return
                obj, loss = result

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
                    'probe_interp': self._probe_interp,
                    'raw_image':    raw if not p['h5_mode'] else None,
                    'position':       position,
                    'position_error': position_error,
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
        """Reconstruct one frame. Return (obj_np, loss_array) or None if stopped."""
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
        if self._obj_init == 'random':
            obj_init = (
                t.randn(1, obj_shape, obj_shape, device=device)
                + 1j * t.randn(1, obj_shape, obj_shape, device=device)
            ).to(t.complex64) * 0.01
        else:
            obj_init = t.ones(1, obj_shape, obj_shape, device=device, dtype=t.complex64)
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
        return obj, loss

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

        # 4. Two-stage LBFGS optimisation
        if self._n_iter1 > 0:
            self._step = 'LBFGS stage 1 optimization'
            for _ in model.LBFGS_optimize(
                self._n_iter1, dataset, lr=self._lr1,
                regularization_factor=[self._reg1, self._reg1],
            ):
                self._status('Stage 1', model.report())
                if not self._running:
                    return None

        if not self._running:
            return None

        self._step = 'LBFGS stage 2 optimization'
        for _ in model.LBFGS_optimize(
            self._n_iter2, dataset, lr=self._lr2,
            regularization_factor=self._reg2,
        ):
            self._status('Stage 2', model.report())
            if not self._running:
                return None

        # 5. Collect results — probe NOT stored in deque
        self._step = 'collecting results'
        results = model.save_results()
        return results['obj'], results['loss_history']

    def _status(self, stage, report):
        # LBFGS also mirrors each stage report to the terminal.
        super()._status(stage, report)
        print(f'{stage}: {report}', flush=True)

    def _algo_diagnostic_lines(self, p):
        return [
            f'  n_iter1={p["n_iter1"]}  n_iter2={p["n_iter2"]}',
            f'  lr1={p["lr1"]}  lr2={p["lr2"]}',
        ]


# Method-string → worker class. Used by the GUI to pick a backend at Start.
WORKER_CLASSES = {
    'CG':    CGWorker,
    'LBFGS': LBFGSWorker,
}
