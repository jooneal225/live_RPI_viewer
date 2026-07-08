"""
Utility functions for RPI Live: dataset construction, probe loading, saving.
"""

import os
import sys
import csv
import time
from pathlib import Path

import numpy as np
import h5py
import tifffile
from skimage.restoration import unwrap_phase

# Optional: complex-aware interpolation from beamline library
try:
    sys.path.insert(0, '/home/beams/ONEAL/code/Libraries/misc')
    from interp_complex import interp_complex_nd
    _HAS_INTERP_COMPLEX = True
except ImportError:
    _HAS_INTERP_COMPLEX = False


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def load_probe_file(path):
    """
    Load probe and pixel size from a ptychi HDF5 reconstruction file.

    Returns
    -------
    probe : ndarray, shape (N_opr, N_modes, N, N), complex
    res_m : float
        Object pixel size in metres.
    """
    with h5py.File(path, 'r') as f:
        probe = f['probe'][()]
        res_m = float(f['obj_pixel_size_m'][()])
    return probe, res_m


def interp_probe(probe, target_N):
    """
    Interpolate every mode in the first OPR block to (target_N, target_N).

    Parameters
    ----------
    probe : ndarray, shape (N_opr, N_modes, N, N)
    target_N : int

    Returns
    -------
    ndarray, shape (N_modes, target_N, target_N)
    """
    modes = probe[0]  # (N_modes, N, N)
    N = modes.shape[-1]
    if N == target_N:
        return modes.copy()

    if _HAS_INTERP_COMPLEX:
        return interp_complex_nd(modes, (target_N, target_N))

    # Fallback: scipy zoom on real/imag separately
    from scipy.ndimage import zoom
    factor = target_N / N
    result = np.stack([
        zoom(modes[i].real, factor) + 1j * zoom(modes[i].imag, factor)
        for i in range(modes.shape[0])
    ])
    return result


# ---------------------------------------------------------------------------
# Pixel mask helpers
# ---------------------------------------------------------------------------

def load_pixel_mask(path):
    """
    Load the full-size detector pixel mask from an HDF5 file.

    Tries 'valid_pixel_mask' first, then 'det_pixel_mask'.

    Returns
    -------
    ndarray, 2D float32  (1 = live, 0 = dead)
    """
    _KEYS = ('valid_pixel_mask', 'det_pixel_mask')
    with h5py.File(path, 'r') as f:
        for key in _KEYS:
            if key in f:
                return f[key][()].astype(np.float32)
        raise KeyError(
            f'No pixel mask found in {path}. '
            f'Tried keys: {_KEYS}. Available: {list(f.keys())}'
        )


def crop_mask(mask_full, beam_cy, beam_cx, crop_N):
    """
    Pre-crop the pixel mask once; call whenever crop_N or beam centre changes.
    """
    half = crop_N // 2
    y0, y1 = beam_cy - half, beam_cy + half
    x0, x1 = beam_cx - half, beam_cx + half
    return mask_full[y0:y1, x0:x1].astype(np.float32)


# ---------------------------------------------------------------------------
# Wavelength helper
# ---------------------------------------------------------------------------

def energy_kev_to_wavelength(energy_kev):
    """Convert X-ray energy in keV to wavelength in metres."""
    return 1.239841984e-9 / energy_kev


# ---------------------------------------------------------------------------
# In-memory CXI construction (mirrors make_cxi_wrapper_dynamic)
# ---------------------------------------------------------------------------

def build_cxi_inmem(patterns, wavelength, detector_distance, x_pixel_size_cxi):
    """
    Build an in-memory HDF5 file in CXI format containing a single diffraction
    pattern.  cdtools reads this via Ptycho2DDataset.from_cxi().

    The structure mirrors make_cxi_wrapper_dynamic so that the same cdtools
    code path is used.  No data is ever written to disk.

    Parameters
    ----------
    patterns : ndarray, shape (1, N, N)
    wavelength : float  (metres)
    detector_distance : float  (metres)
    x_pixel_size_cxi : float
        Object-space pixel size (metres) matching cdtools convention.

    Returns
    -------
    h5py.File  (in-memory; caller must close it after from_cxi)
    """
    f = h5py.File('__inmem__', 'w', driver='core', backing_store=False)
    entry = f.create_group('entry_1')
    instrument = entry.create_group('instrument_1')
    det = instrument.create_group('detector_1')
    beam = instrument.create_group('source_1')

    det.create_dataset('data', data=patterns.astype(np.float32))
    det.create_dataset('distance', data=float(detector_distance))
    det.create_dataset('x_pixel_size', data=float(x_pixel_size_cxi))
    # Single position at origin, shape (1, 2) matching make_cxi_wrapper_dynamic
    det.create_dataset('translation', data=np.array([[0., 0.]], dtype=np.float32))
    beam.create_dataset('wavelength', data=float(wavelength))

    return f


# ---------------------------------------------------------------------------
# Pixel size calculations
# ---------------------------------------------------------------------------

def compute_pixel_sizes(res_m_probe, N_probe, crop_N, obj_shape):
    """
    Compute CXI pixel size and RPI object pixel size.

    Parameters
    ----------
    res_m_probe : float   probe pixel size from ptychi reconstruction
    N_probe     : int     probe array size
    crop_N      : int     diffraction pattern size used in this run
    obj_shape   : int     RPI object size (square)

    Returns
    -------
    x_pixel_size_cxi : float   pixel size to put in the CXI file
    res_rpi_m        : float   physical pixel size of the RPI object
    """
    x_pixel_size_cxi = res_m_probe * N_probe / crop_N
    res_rpi_m = res_m_probe * N_probe / obj_shape
    return x_pixel_size_cxi, res_rpi_m


# ---------------------------------------------------------------------------
# Save  (dispatches on params['method']: 'CG' or 'LBFGS')
# ---------------------------------------------------------------------------

def _build_recon_dir_name(params):
    crop_N    = params['crop_N']
    obj_shape = params['obj_shape']
    if params['method'] == 'CG':
        n_iter      = params['n_iter']
        clear_every = params['clear_every']
        return f'Ndp{crop_N}_Nrpi{obj_shape}_CG_Niter{n_iter}_clear{clear_every}'
    # LBFGS
    reg1 = params['reg_factor1']
    reg2 = [params['reg_factor2_lo'], params['reg_factor2_hi']]
    lr1  = params['lr1']
    lr2  = params['lr2']
    return (
        f'Ndp{crop_N}_Nrpi{obj_shape}_LBFGS'
        f'_reg{int(round(np.log10(reg1)))}'
        f'_{int(round(np.log10(reg2[0])))}'
        f'_{int(round(np.log10(reg2[1])))}'
        f'_lr{int(100*lr1)}_{int(100*lr2)}'
    )


def save_rpi_result(entry, probe_interp, params, save_dir):
    """
    Save one deque entry to disk in the same format as cdtools_SF_single.py.

    Parameters
    ----------
    entry : dict
        Keys: obj, loss_history, pixel_size_m, frame_idx, timestamp
    probe_interp : ndarray, shape (N_modes, N, N), complex
        The probe actually used in the reconstruction.
    params : dict
        GUI parameters. Must include 'method' ('CG' or 'LBFGS') and the
        algorithm-specific keys (CG: n_iter, clear_every; LBFGS: reg_factor1,
        reg_factor2_lo, reg_factor2_hi, lr1, lr2, n_iter1, n_iter2).
    save_dir : str or Path
    """
    save_dir = Path(save_dir)
    obj       = entry['obj']
    loss      = entry['loss_history']
    pix_m     = entry['pixel_size_m']
    frame_idx = entry['frame_idx']

    method    = params['method']
    crop_N    = params['crop_N']
    obj_shape = params['obj_shape']

    dir_name  = _build_recon_dir_name(params)
    recon_dir = save_dir / dir_name
    recon_dir.mkdir(parents=True, exist_ok=True)
    (recon_dir / 'object_ph').mkdir(exist_ok=True)
    (recon_dir / 'object_mag').mkdir(exist_ok=True)

    if method == 'CG':
        stem = f'frame{frame_idx}_Niter{params["n_iter"]}'
    else:
        stem = f'frame{frame_idx}_Niter{params["n_iter1"]}_{params["n_iter2"]}'

    # HDF5
    h5_path = recon_dir / f'recon_{stem}.h5'
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('object',           data=obj[None])
        f.create_dataset('probe',            data=probe_interp[None])
        f.create_dataset('loss',             data=loss)
        f.create_dataset('obj_pixel_size_m', data=pix_m)
        f.create_dataset('positions_px',     data=np.array([[0, 0]]))
        f.create_dataset('method',           data=method)
        f.create_dataset('Ndp',              data=crop_N)
        f.create_dataset('Nrpi',             data=obj_shape)
        if method == 'CG':
            f.create_dataset('Niter',        data=params['n_iter'])
            f.create_dataset('clear_every',  data=params['clear_every'])
        else:
            reg1 = params['reg_factor1']
            reg2 = [params['reg_factor2_lo'], params['reg_factor2_hi']]
            f.create_dataset('regularization', data=[[reg1, reg1], reg2])
            f.create_dataset('learning_rate',  data=[params['lr1'], params['lr2']])
            f.create_dataset('Niter',          data=[params['n_iter1'], params['n_iter2']])
        raw = entry.get('raw_image')
        if raw is not None:
            f.create_dataset('raw_image', data=raw)

    # TIFFs
    temp_abs = np.abs(obj).astype(np.float32)
    temp_abs -= temp_abs.min()
    if temp_abs.max() > 0:
        temp_abs *= 65535.0 / temp_abs.max()

    temp_ph = unwrap_phase(np.angle(obj)).astype(np.float32)
    temp_ph -= temp_ph.min()
    if temp_ph.max() > 0:
        temp_ph *= 65535.0 / temp_ph.max()

    _save_tiff(recon_dir / f'object_mag/object_mag_{stem}.tiff',
               temp_abs.astype(np.uint16), pix_m)
    _save_tiff(recon_dir / f'object_ph/object_ph_{stem}.tiff',
               temp_ph.astype(np.uint16), pix_m)

    # CSV log
    csv_path = save_dir / 'recon_completed.csv'
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow([stem, str(h5_path)])

    return str(h5_path)


def remove_phase_ramp(phase, radius=None):
    """
    Remove a best-fit linear phase ramp from a 2D phase array.

    Parameters
    ----------
    phase : (Ny, Nx) ndarray
        2D real-valued phase array, e.g. unwrap(np.angle(...))
    radius : float or None
        If given, only use pixels within a centered disk of this radius
        (in pixels) to fit the ramp.

    Returns
    -------
    phase_corr : ndarray
        Phase with linear ramp removed.
    """
    ny, nx = phase.shape
    y, x = np.indices((ny, nx))
    x = x - (nx - 1) / 2
    y = y - (ny - 1) / 2

    if radius is not None:
        mask = (x**2 + y**2) <= radius**2
    else:
        mask = np.ones_like(phase, dtype=bool)

    A = np.column_stack([x[mask], y[mask], np.ones(mask.sum())])
    coeffs, _, _, _ = np.linalg.lstsq(A, phase[mask], rcond=None)
    a, b, c = coeffs
    ramp = a * x + b * y + c
    return phase - ramp


def autosave_append(path, entry):
    """
    Append one RPI result to a cumulative H5 file.

    Creates the file and resizable datasets on the first call; resizes and
    appends on every subsequent call.  The file can be opened externally
    while recording — h5py flushes after each append.

    Parameters
    ----------
    path : str or Path
        Destination H5 file (created if it does not exist).
    entry : dict
        Keys: obj (2-D complex ndarray), loss_history, pixel_size_m,
        frame_idx, timestamp.
    """
    obj       = entry['obj']
    loss      = np.asarray(entry['loss_history'], dtype=np.float32)
    pix_m     = float(entry['pixel_size_m'])
    frame_idx = int(entry['frame_idx'])
    ts        = float(entry['timestamp'])
    raw       = entry.get('raw_image')
    ny, nx    = obj.shape
    n_loss    = len(loss)

    with h5py.File(path, 'a') as f:
        if 'obj' not in f:
            f.create_dataset('obj',
                             data=obj[None],
                             maxshape=(None, ny, nx),
                             chunks=(1, ny, nx))
            f.create_dataset('loss',
                             data=loss[None],
                             maxshape=(None, n_loss),
                             chunks=(1, n_loss))
            f.create_dataset('frame_idx',
                             data=np.array([frame_idx], dtype=np.int32),
                             maxshape=(None,), chunks=(64,))
            f.create_dataset('timestamp',
                             data=np.array([ts], dtype=np.float64),
                             maxshape=(None,), chunks=(64,))
            f.create_dataset('pixel_size_m', data=pix_m)
            if raw is not None:
                ry, rx = raw.shape
                f.create_dataset('raw_image',
                                 data=raw[None],
                                 maxshape=(None, ry, rx),
                                 chunks=(1, ry, rx))
        else:
            n = f['obj'].shape[0]
            f['obj'].resize(n + 1, axis=0)
            f['obj'][n] = obj
            f['loss'].resize(n + 1, axis=0)
            f['loss'][n] = loss
            f['frame_idx'].resize(n + 1, axis=0)
            f['frame_idx'][n] = frame_idx
            f['timestamp'].resize(n + 1, axis=0)
            f['timestamp'][n] = ts
            if raw is not None and 'raw_image' in f:
                f['raw_image'].resize(n + 1, axis=0)
                f['raw_image'][n] = raw
        f.flush()


def _save_tiff(path, img, pixel_size_m):
    tifffile.imwrite(
        path,
        img,
        imagej=True,
        metadata={
            'unit': 'um',
            'xspacing': 1e6 * pixel_size_m,
            'yspacing': 1e6 * pixel_size_m,
        },
    )
