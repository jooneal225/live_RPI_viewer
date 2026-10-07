"""
Utility functions for RPI Live: dataset construction, probe loading, saving.
"""

import os
import sys
import csv
import json
import re
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
# Illuminated-area geometry
# ---------------------------------------------------------------------------

PROBE_SHAPES = ('circle', 'square')
DEFAULT_PROBE_SHAPE = 'circle'
DEFAULT_FILL_FRACTION = 0.5

# Masks are identical for every frame of a run, so build each one once.
_ILLUM_MASK_CACHE = {}


def illumination_mask(n, shape=DEFAULT_PROBE_SHAPE,
                      fill_fraction=DEFAULT_FILL_FRACTION, fudge=1.0):
    """
    Boolean (n, n) mask of the illuminated area, centred in the array.

    RPI reconstructs a square object array, but the probe only illuminates a
    circle or square inside it. Everything outside that region is noise, so
    phase unwrapping, ramp fitting, display zoom and object stitching all want
    to be restricted to it.

    Parameters
    ----------
    n : int
        Side length of the (square) object array.
    shape : {'circle', 'square'}
        Geometry of the illuminated area.
    fill_fraction : float
        Illuminated diameter (or side) as a fraction of n.
    fudge : float
        Multiplier applied to fill_fraction. Callers use different values for
        different purposes (e.g. 1.2 for the display zoom box, 1.0 for phase
        processing); see the _*_FUDGE constants in rpi_live_gui.py.

    Returns
    -------
    ndarray, (n, n) bool
    """
    n = int(n)
    key = (n, shape, round(float(fill_fraction), 6), round(float(fudge), 6))
    cached = _ILLUM_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    half = fudge * float(fill_fraction) * n / 2.0
    y, x = np.indices((n, n), dtype=np.float32)
    cy = cx = (n - 1) / 2.0
    y -= cy
    x -= cx
    if shape == 'square':
        mask = (np.abs(x) <= half) & (np.abs(y) <= half)
    else:   # default to a circle for anything unrecognised
        mask = (x * x + y * y) <= half * half

    # A degenerate fill_fraction would mask out every pixel and make the ramp
    # fit singular; fall back to the whole array and let the caller see it.
    if not mask.any():
        mask = np.ones((n, n), dtype=bool)

    _ILLUM_MASK_CACHE[key] = mask
    return mask


# ---------------------------------------------------------------------------
# Wavelength helper
# ---------------------------------------------------------------------------

def energy_kev_to_wavelength(energy_kev):
    """Convert X-ray energy in keV to wavelength in metres."""
    return 1.239841984e-9 / energy_kev


def propagate_probe_angular_spectrum(probe, res_m, wavelength_m, distance_um):
    """
    Propagate a probe by distance_um microns using cdtools' angular-spectrum
    near-field propagator. Returns a new array; the input is left untouched.

    Parameters
    ----------
    probe : ndarray, shape (N_opr, N_modes, N, N), complex
    res_m : float
        Probe pixel size in metres.
    wavelength_m : float
        Wavelength of light in metres.
    distance_um : float
        Propagation distance in microns (positive = downstream).

    Returns
    -------
    ndarray, same shape as probe, complex64
    """
    import torch
    from cdtools.tools import propagators

    shape = probe.shape[-2:]
    z_m = distance_um * 1e-6
    wavefront = torch.as_tensor(probe, dtype=torch.complex64)
    prop = propagators.generate_angular_spectrum_propagator(
        shape, (res_m, res_m), wavelength_m, z_m,
    )
    return propagators.near_field(wavefront, prop).numpy()


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


def remove_phase_ramp(phase, radius=None, mask=None):
    """
    Remove a best-fit linear phase ramp from a 2D phase array.

    Parameters
    ----------
    phase : (Ny, Nx) ndarray
        2D real-valued phase array, e.g. unwrap(np.angle(...))
    radius : float or None
        If given, only use pixels within a centered disk of this radius
        (in pixels) to fit the ramp. Ignored when `mask` is given.
    mask : (Ny, Nx) bool ndarray or None
        Region to operate on, typically from illumination_mask(). The plane is
        fitted on these pixels *and subtracted only from them* — pixels outside
        keep their original value, so the dark corners of an RPI object are
        left alone rather than having a meaningless ramp applied to them.

    Returns
    -------
    phase_corr : ndarray
        Phase with the linear ramp removed.
    """
    ny, nx = phase.shape
    y, x = np.indices((ny, nx))
    x = x - (nx - 1) / 2
    y = y - (ny - 1) / 2

    if mask is not None:
        fit_mask = np.asarray(mask, dtype=bool)
        restrict = True
    elif radius is not None:
        fit_mask = (x**2 + y**2) <= radius**2
        restrict = False     # legacy behaviour: fit on a disk, subtract globally
    else:
        fit_mask = np.ones_like(phase, dtype=bool)
        restrict = False

    # Three free parameters need at least three points; bail out unchanged
    # rather than handing a singular system to lstsq.
    if fit_mask.sum() < 3:
        return phase

    A = np.column_stack([x[fit_mask], y[fit_mask], np.ones(fit_mask.sum())])
    coeffs, _, _, _ = np.linalg.lstsq(A, phase[fit_mask], rcond=None)
    a, b, c = coeffs
    ramp = a * x + b * y + c
    if restrict:
        return np.where(fit_mask, phase - ramp, phase)
    return phase - ramp


def process_phase_region(phase, mask, unwrap=False, ramp=False, zero_mean=False):
    """
    Apply phase post-processing **inside `mask` only**, leaving the rest alone.

    An RPI object array is mostly dark outside the illuminated spot. Unwrapping
    or ramp-fitting over the whole array lets that noise drive the result, so
    every caller (live display and object stitching alike) restricts the
    operation to the illuminated region from illumination_mask().

    Parameters
    ----------
    phase : (Ny, Nx) real ndarray
        Wrapped phase, i.e. np.angle(obj).
    mask : (Ny, Nx) bool ndarray
        Region to operate on.
    unwrap : bool
        Unwrap the phase inside the mask.
    ramp : bool
        Fit and subtract a linear ramp over the mask. Ramp removal needs a
        continuous phase, so the phase is unwrapped internally even when
        `unwrap` is False — in that case the result is re-wrapped afterwards.
    zero_mean : bool
        Subtract the mean over the mask (removes the per-frame piston offset
        that otherwise makes stitched tiles visible).

    Returns
    -------
    ndarray, float32, same shape as `phase`
    """
    out = np.asarray(phase, dtype=np.float32)
    if not (unwrap or ramp or zero_mean):
        return out
    out = out.copy()
    mask = np.asarray(mask, dtype=bool)

    if (unwrap or ramp) and mask.sum() >= 4:
        # unwrap_phase honours a masked array: masked pixels take no part in
        # the unwrap and come back masked, so they keep their original value.
        try:
            uw = unwrap_phase(np.ma.masked_array(out, ~mask))
            out = np.where(mask, np.ma.filled(uw, 0.0).astype(np.float32), out)
        except Exception:
            # A pathological mask (disconnected, single-pixel-wide) can make
            # the unwrap fail; the wrapped phase is a usable fallback.
            pass

    if ramp:
        out = remove_phase_ramp(out, mask=mask).astype(np.float32)
        if not unwrap:
            rewrapped = (out + np.pi) % (2 * np.pi) - np.pi
            out = np.where(mask, rewrapped, out).astype(np.float32)

    if zero_mean and mask.any():
        out = np.where(mask, out - float(out[mask].mean()), out).astype(np.float32)

    return out


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


# ---------------------------------------------------------------------------
# Configuration JSON (probe library + beamline + detector geometry)
# ---------------------------------------------------------------------------
#
# settings.ini remains the authoritative per-session store. This JSON is the
# portable export/import format: it carries everything needed to reproduce a
# setup on another machine or after a beamtime, *except* Crop N, which is a
# per-run detector choice rather than a property of the setup.

CONFIG_SCHEMA_VERSION = 1

_BEAMLINE_KEYS = ('energy_kev', 'detector_distance')
_DETECTOR_KEYS = ('beam_center_y', 'beam_center_x')


def normalize_probe_entry(entry):
    """Coerce one probe-library entry into the canonical dict shape.

    Accepts partial dicts (older exports, hand-edited files) and fills in the
    defaults, so every consumer can assume all keys are present.
    """
    shape = str(entry.get('shape', DEFAULT_PROBE_SHAPE)).lower()
    if shape not in PROBE_SHAPES:
        shape = DEFAULT_PROBE_SHAPE
    try:
        fill = float(entry.get('fill_fraction', DEFAULT_FILL_FRACTION))
    except (TypeError, ValueError):
        fill = DEFAULT_FILL_FRACTION
    fill = min(max(fill, 0.01), 1.0)
    return {
        'name':          str(entry.get('name', '')),
        'path':          str(entry.get('path', '')),
        'enabled':       bool(entry.get('enabled', True)),
        'shape':         shape,
        'fill_fraction': fill,
        'params':        dict(entry.get('params') or {}),
    }


def build_config_dict(probes, beamline, detector):
    """Assemble the JSON-serialisable config document."""
    return {
        'schema_version': CONFIG_SCHEMA_VERSION,
        'probes':   [normalize_probe_entry(p) for p in (probes or [])],
        'beamline': {k: float(beamline.get(k, 0.0)) for k in _BEAMLINE_KEYS},
        'detector': {k: int(detector.get(k, 0))     for k in _DETECTOR_KEYS},
    }


def write_config_json(path, cfg):
    """Write a config document (from build_config_dict) to disk."""
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
        f.write('\n')


def read_config_json(path):
    """
    Read and validate a config JSON.

    Returns
    -------
    (probes, beamline, detector)
        probes   : list of normalised entry dicts
        beamline : {'energy_kev', 'detector_distance'}
        detector : {'beam_center_y', 'beam_center_x'}

    Raises
    ------
    ValueError
        With a message naming the specific problem, so the GUI can show it
        verbatim instead of a bare traceback.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except OSError as exc:
        raise ValueError(f'Could not open {path}: {exc}')
    except json.JSONDecodeError as exc:
        raise ValueError(f'{path} is not valid JSON: {exc}')

    if not isinstance(data, dict):
        raise ValueError(
            f'{path}: expected a JSON object at the top level, '
            f'got {type(data).__name__}.'
        )

    version = data.get('schema_version', CONFIG_SCHEMA_VERSION)
    if int(version) > CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f'{path} was written by a newer version of this GUI '
            f'(schema_version {version} > {CONFIG_SCHEMA_VERSION}).'
        )

    raw_probes = data.get('probes', [])
    if not isinstance(raw_probes, list):
        raise ValueError(f'{path}: "probes" must be a list.')
    probes = []
    for i, p in enumerate(raw_probes):
        if not isinstance(p, dict):
            raise ValueError(f'{path}: probes[{i}] must be an object.')
        if not p.get('path'):
            raise ValueError(f'{path}: probes[{i}] has no "path".')
        probes.append(normalize_probe_entry(p))

    def _section(name, keys, cast):
        src = data.get(name, {})
        if not isinstance(src, dict):
            raise ValueError(f'{path}: "{name}" must be an object.')
        out = {}
        for k in keys:
            if k not in src:
                continue
            try:
                out[k] = cast(src[k])
            except (TypeError, ValueError):
                raise ValueError(f'{path}: {name}.{k} is not a number ({src[k]!r}).')
        return out

    beamline = _section('beamline', _BEAMLINE_KEYS, float)
    detector = _section('detector', _DETECTOR_KEYS, int)
    return probes, beamline, detector


# ---------------------------------------------------------------------------
# Accumulated (stitched) object save
# ---------------------------------------------------------------------------

_SCAN_DIR_RE = re.compile(r'^[Ss](\d{1,6})$')
_SCAN_ANY_RE = re.compile(r'[Ss](\d{3,6})')


def scan_tag_from_path(path):
    """Best-effort scan identifier for a data file, e.g. 'S0082'.

    Beamline paths put the scan in its own directory
    (.../preproc/S0082/data_roi0_Ndp1024_dp.hdf5), so the directory components
    are checked deepest-first before falling back to the filename. Returns
    None when nothing looks like a scan number, and the caller should then use
    a timestamp.
    """
    if not path:
        return None
    p = Path(str(path))
    dirs = list(reversed(p.parts[:-1]))
    for part in dirs:                       # a directory that is just "S0082"
        m = _SCAN_DIR_RE.match(part)
        if m:
            return f'S{int(m.group(1)):04d}'
    for part in dirs:                       # ...or contains it, "scan_S0082"
        m = _SCAN_ANY_RE.search(part)
        if m:
            return f'S{int(m.group(1)):04d}'
    m = _SCAN_ANY_RE.search(p.stem)         # ...or only the filename has it
    if m:
        return f'S{int(m.group(1)):04d}'
    return None


def save_accumulated_object(path, obj_sum, contrib, positions_um, positions_px,
                            pixel_size_m, meta=None, extra=None):
    """
    Save a stitched object from the Object Accumulator to HDF5.

    The stitched image (object_normalized = obj_sum / contrib) is stored
    alongside the contribution tracker. The raw summed array is deliberately
    not written — it is only an intermediate, and at canvas size it would
    double the file for nothing.

    Parameters
    ----------
    path : str or Path
    obj_sum : (Ny, Nx) float ndarray
        Summed phase; each contributing frame added once. Used to compute
        object_normalized, not stored itself.
    contrib : (Ny, Nx) uint ndarray
        Per-pixel contribution count (the pixel_contribution_tracker).
    positions_um : sequence of (x, y)
        Motor positions of every accepted frame, in microns.
    positions_px : sequence of (col, row)
        The same positions as canvas pixel offsets from the centre.
    pixel_size_m : float
        Canvas pixel size (equal to the RPI object pixel size).
    meta : dict or None
        Scalar run parameters, written as root attributes.
    extra : dict or None
        Additional {name: array} datasets, e.g. the per-frame record table.
        Empty arrays are skipped.

    Returns
    -------
    str  the path written
    """
    path = str(path)
    contrib = np.asarray(contrib)
    normalized = np.zeros(obj_sum.shape, dtype=np.float32)
    np.divide(obj_sum, contrib, out=normalized, where=contrib > 0)

    with h5py.File(path, 'w') as f:
        f.create_dataset('object_normalized',  data=normalized)
        f.create_dataset('pixel_contribution', data=contrib)
        f.create_dataset('positions_um',
                         data=np.asarray(positions_um, dtype=np.float64).reshape(-1, 2))
        f.create_dataset('positions_px',
                         data=np.asarray(positions_px, dtype=np.int64).reshape(-1, 2))
        f.create_dataset('obj_pixel_size_m', data=float(pixel_size_m))
        for name, arr in (extra or {}).items():
            arr = np.asarray(arr)
            if arr.size:
                f.create_dataset(name, data=arr)
        for k, v in (meta or {}).items():
            try:
                f.attrs[k] = v
            except TypeError:
                f.attrs[k] = str(v)   # h5py rejects e.g. None / nested dicts
    return path
