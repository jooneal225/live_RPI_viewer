"""
Custom RPI chain — the step schema shared by the editor, worker and saver.

A chain is a list of steps run in order on every frame, each one handing its
object (and probe) to the next:

  * CG          — the conjugate-gradient backend
  * LSQML       — the ptychi LSQML backend
  * LBFGS_INCOH — cdtools LBFGS with an incoherent object of 2 modes,
                  regularized as [mode 0, mode 1]
  * LBFGS       — cdtools LBFGS with a single coherent object

The standalone two-stage LBFGS method, which uses one object mode per probe
mode, is LBFGS_INCOH [reg1, reg1] followed by LBFGS_INCOH [reg2_lo, reg2_hi]
when the probe has 2 modes.

Each step is a plain dict holding only the keys its algorithm uses (see
STEP_FIELDS). The editor shows it as one table row (COLUMNS); the JSON file
stores the dicts as they are, plus the chain's name. Nothing here imports Qt.
"""

import json
import re
import zlib


CHAIN_SCHEMA_VERSION = 1

ALGOS       = ['CG', 'LSQML', 'LBFGS_INCOH', 'LBFGS']
ALGO_LABELS = {
    'CG':          'Fast CG',
    'LSQML':       'LSQML (ptychi)',
    'LBFGS_INCOH': 'LBFGS Incoh Obj',
    'LBFGS':       'LBFGS',
}
# Short names used in save-directory names.
_ALGO_TAGS  = {'CG': 'CG', 'LSQML': 'LS', 'LBFGS_INCOH': 'LBi', 'LBFGS': 'LB'}

# Keys each step dict carries, besides 'algo'.
STEP_FIELDS = {
    'CG':          ('n_iter', 'clear_every', 'loss_target'),
    'LSQML':       ('n_iter', 'step_size', 'probe_update', 'momentum', 'early_stop_pct'),
    'LBFGS_INCOH': ('n_iter', 'step_size', 'reg', 'probe_update'),
    'LBFGS':       ('n_iter', 'step_size', 'reg', 'probe_update'),
}

# Probe update choices; it applies from the step's first iteration. To start
# it later, split the step into two rows.
PROBE_UPDATES = {
    'CG':          (),
    'LSQML':       ('off', 'full'),
    'LBFGS_INCOH': ('off', 'weights', 'full'),
    'LBFGS':       ('off', 'weights', 'full'),
}

# Table columns. 'options' holds the rare per-algorithm extras as key=value.
COLUMNS       = ['algo', 'n_iter', 'step_size', 'reg', 'probe_update', 'options']
COLUMN_LABELS = ['Algorithm', 'N iter', 'Step size', 'Regularization',
                 'Probe update', 'Options']
NOT_APPLICABLE = '—'

# Options-column keys → step field, per algorithm.
OPTION_KEYS = {
    'CG':    {'clear': 'clear_every', 'target': 'loss_target'},
    'LSQML': {'momentum': 'momentum', 'stop': 'early_stop_pct'},
}

COLUMN_TOOLTIPS = {
    'algo':         'Fixed per row — remove the row and add another to change it.',
    'n_iter':       'Iterations (LSQML: epochs) in this step.',
    'step_size':    'LBFGS: learning rate. LSQML: object step-size scaler. '
                    'CG computes its own step.',
    'reg':          'LBFGS Incoh Obj: "main, second" — L2 regularization of '
                    'object mode 0 and of object mode 1 (always 2 modes).\n'
                    'LBFGS: one value for the single object mode.',
    'probe_update': 'From the first iteration of this step. '
                    'LBFGS: off / weights / full. LSQML: off / full.\n'
                    'To start a probe update later, split the step into two rows.',
    'options':      'CG: clear=<restart period> target=<loss target, 0=off>\n'
                    'LSQML: momentum=<gain> stop=<early stop %, 0=off>',
}

COMMON_KEYS = ('n_probe_modes', 'crop_N', 'obj_shape', 'obj_init', 'n_avg')
OBJ_INITS   = ('random', 'flat')
# Same ranges as the spin boxes in the RPI Parameters window.
COMMON_RANGES = {
    'n_probe_modes': (0, 32),
    'crop_N':        (64, 2048),
    'obj_shape':     (32, 2048),
    'n_avg':         (1, 1000),
}


# ── Field checks ────────────────────────────────────────────────────────────

def _as_int(v):
    """int(v), accepting 20, '20', 20.0 and '20.0' but not 20.5."""
    f = float(v)
    if not f.is_integer():
        raise ValueError
    return int(f)


# key → (cast, predicate, description used in error messages)
_SCALAR_FIELDS = {
    'n_iter':         (_as_int, lambda v: v >= 1,        'an integer ≥ 1'),
    'step_size':      (float,   lambda v: v > 0,         'a number > 0'),
    'clear_every':    (_as_int, lambda v: v >= 1,        'an integer ≥ 1'),
    'loss_target':    (float,   lambda v: v >= 0,        'a number ≥ 0'),
    'momentum':       (float,   lambda v: v >= 0,        'a number ≥ 0'),
    'early_stop_pct': (float,   lambda v: 0 <= v <= 100, 'a percentage from 0 to 100'),
}


def check_field(algo, key, value):
    """Normalize one step field. Returns (value, None) or (None, error)."""
    if key == 'probe_update':
        choices = PROBE_UPDATES[algo]
        v = str(value).strip().lower()
        # A unique prefix is enough ('w' → 'weights').
        hits = [c for c in choices if c.startswith(v)] if v else []
        if v in choices:
            return v, None
        if len(hits) == 1:
            return hits[0], None
        return None, f'must be one of: {", ".join(choices)}'
    if key == 'reg':
        if algo == 'LBFGS_INCOH':
            if isinstance(value, str):
                parts = [x for x in re.split(r'[\s,;]+', value.strip(' []()'))
                         if x]
            elif isinstance(value, (list, tuple)):
                parts = list(value)
            else:
                parts = [value]
            if len(parts) == 1:
                parts = parts * 2       # one value = same for every mode
            if len(parts) != 2:
                return None, 'needs two values: "main, second"'
            try:
                vals = [float(x) for x in parts]
            except (TypeError, ValueError):
                return None, 'needs two numbers: "main, second"'
            if any(x < 0 for x in vals):
                return None, 'values must be ≥ 0'
            return vals, None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None, 'must be one number ≥ 0'
        if v < 0:
            return None, 'must be ≥ 0'
        return v, None
    cast, ok, desc = _SCALAR_FIELDS[key]
    try:
        v = cast(str(value).strip().rstrip('%') if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return None, f'must be {desc}'
    if not ok(v):
        return None, f'must be {desc}'
    return v, None


def applies(algo, column):
    """True if `column` is editable for a row of `algo`."""
    if column == 'algo':
        return True
    if column == 'options':
        return algo in OPTION_KEYS
    return column in STEP_FIELDS[algo]


# ── Table cells ─────────────────────────────────────────────────────────────

def format_cell(step, column):
    algo = step['algo']
    if column == 'algo':
        return ALGO_LABELS[algo]
    if not applies(algo, column):
        return NOT_APPLICABLE
    if column == 'n_iter':
        return str(step['n_iter'])
    if column == 'step_size':
        return f'{step["step_size"]:g}'
    if column == 'reg':
        if algo == 'LBFGS_INCOH':
            return f'{step["reg"][0]:g}, {step["reg"][1]:g}'
        return f'{step["reg"]:g}'
    if column == 'probe_update':
        return step['probe_update']
    if algo == 'CG':
        return f'clear={step["clear_every"]} target={step["loss_target"]:g}'
    return f'momentum={step["momentum"]:g} stop={step["early_stop_pct"]:g}%'


def parse_cell(algo, column, text):
    """Parse an edited cell. Returns ({field: value, ...}, None) or (None, error).

    The Options cell may set any subset of its keys; keys left out keep
    their current value.
    """
    if column != 'options':
        v, err = check_field(algo, column, text)
        return (None, err) if err else ({column: v}, None)
    keys = OPTION_KEYS[algo]
    out = {}
    for tok in re.split(r'[\s,;]+', text.strip()):
        if not tok:
            continue
        if '=' not in tok:
            return None, f'"{tok}" is not key=value (keys: {", ".join(keys)})'
        k, v = (s.strip() for s in tok.split('=', 1))
        k = k.lower()
        if k not in keys:
            return None, f'unknown key "{k}" (keys: {", ".join(keys)})'
        val, err = check_field(algo, keys[k], v)
        if err:
            return None, f'{k} {err}'
        out[keys[k]] = val
    return out, None


# ── Steps ───────────────────────────────────────────────────────────────────

def _f(p, key, default):
    try:
        return float(str(p[key]).strip())
    except (KeyError, ValueError, TypeError):
        return default


def _i(p, key, default):
    try:
        return _as_int(p[key])
    except (KeyError, ValueError, TypeError):
        return default


def _b(p, key, default):
    if key not in p:
        return default
    return str(p[key]).strip().lower() == 'true'


def step_from_params(algo, p):
    """A new step prefilled from the RPI Parameters window.

    `p` is MainWindow._get_current_rpi_params(). LBFGS Incoh Obj takes stage
    1's values (probe fixed) and LBFGS takes stage 2's, with the window's
    probe update choice.
    """
    if algo == 'CG':
        return {'algo': 'CG',
                'n_iter':      _i(p, 'cg_n_iter', 100),
                'clear_every': _i(p, 'cg_clear_every', 10),
                'loss_target': max(0.0, _f(p, 'cg_loss_target', 0.0))}
    if algo == 'LSQML':
        return {'algo': 'LSQML',
                'n_iter':         _i(p, 'lsqml_n_epochs', 200),
                'step_size':      _f(p, 'lsqml_step_scaler', 0.9),
                'probe_update':   'full' if _b(p, 'lsqml_update_probe', False) else 'off',
                'momentum':       _f(p, 'lsqml_momentum', 0.0),
                'early_stop_pct': _f(p, 'lsqml_early_stop_pct', 1.0)}
    if algo == 'LBFGS_INCOH':
        reg1 = _f(p, 'lbfgs_reg_factor1', 1e-5)
        return {'algo': 'LBFGS_INCOH',
                'n_iter':       max(1, _i(p, 'lbfgs_n_iter1', 20)),
                'step_size':    _f(p, 'lbfgs_lr1', 0.2),
                'reg':          [reg1, reg1],
                'probe_update': 'off'}
    pu = str(p.get('lbfgs_probe_update', 'off'))
    return {'algo': 'LBFGS',
            'n_iter':       max(1, _i(p, 'lbfgs_n_iter2', 60)),
            'step_size':    _f(p, 'lbfgs_lr2', 0.1),
            'reg':          _f(p, 'lbfgs_reg_factor2_lo', 1e-8),
            'probe_update': pu if pu in PROBE_UPDATES['LBFGS'] else 'off'}


def validate_step(step):
    """Normalized copy of a step dict, or raise ValueError naming the problem."""
    if not isinstance(step, dict):
        raise ValueError('a step must be an object')
    algo = step.get('algo')
    if algo not in ALGOS:
        raise ValueError(f'unknown algorithm {algo!r} (one of {", ".join(ALGOS)})')
    out = {'algo': algo}
    for key in STEP_FIELDS[algo]:
        if key not in step:
            raise ValueError(f'{ALGO_LABELS[algo]} step has no "{key}"')
        v, err = check_field(algo, key, step[key])
        if err:
            raise ValueError(f'{ALGO_LABELS[algo]} {key} {err} (got {step[key]!r})')
        out[key] = v
    return out


def validate_common(common):
    """Normalized copy of the common parameters, or raise ValueError."""
    if not isinstance(common, dict):
        raise ValueError('"common" must be an object')
    out = {}
    for key in COMMON_KEYS:
        if key not in common:
            raise ValueError(f'common parameters have no "{key}"')
        if key == 'obj_init':
            if common[key] not in OBJ_INITS:
                raise ValueError(f'obj_init must be one of {", ".join(OBJ_INITS)}')
            out[key] = common[key]
            continue
        lo, hi = COMMON_RANGES[key]
        try:
            v = _as_int(common[key])
        except (TypeError, ValueError):
            raise ValueError(f'{key} must be an integer (got {common[key]!r})')
        if not lo <= v <= hi:
            raise ValueError(f'{key} must be between {lo} and {hi} (got {v})')
        # cdtools' Fourier upsampling (and the hand-off between backends,
        # see ChainWorker) only holds for even array sizes.
        if key in ('crop_N', 'obj_shape') and v % 2:
            raise ValueError(f'{key} must be even (got {v})')
        out[key] = v
    return out


def total_iterations(steps):
    return sum(s['n_iter'] for s in steps)


def recipe_tag(steps):
    """Compact chain description for directory names, e.g.
    'LBi20-LB60w-LS200p-CG100' ('w'/'p': probe weights/probe updated)."""
    parts = []
    for s in steps:
        mark = {'full': 'p', 'weights': 'w'}.get(s.get('probe_update', 'off'), '')
        parts.append(f'{_ALGO_TAGS[s["algo"]]}{s["n_iter"]}{mark}')
    tag = '-'.join(parts)
    if len(tag) > 100:
        # Keep long chains to a sane path length, still unique per recipe.
        tag = (f'{len(steps)}steps_{total_iterations(steps)}it_'
               f'{zlib.crc32(tag.encode()):08x}')
    return tag


# ── JSON ────────────────────────────────────────────────────────────────────

def chain_to_dict(common, steps, name=''):
    d = {'schema_version': CHAIN_SCHEMA_VERSION}
    if name:
        d['name'] = name
    d['common'] = dict(common)
    d['steps'] = [dict(s) for s in steps]
    return d


def chain_to_json(common, steps, name=''):
    return json.dumps(chain_to_dict(common, steps, name))


def write_chain_json(path, common, steps, name=''):
    with open(path, 'w') as f:
        json.dump(chain_to_dict(common, steps, name), f, indent=2)


def read_chain_json(path):
    """Read and validate a chain JSON. Returns (common, steps, name).

    `name` is '' for files written before chains had names.
    Raises ValueError with a message naming the problem, for the GUI to show.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except OSError as exc:
        raise ValueError(f'Could not open {path}: {exc}')
    except json.JSONDecodeError as exc:
        raise ValueError(f'{path} is not valid JSON: {exc}')
    if not isinstance(data, dict) or 'steps' not in data:
        raise ValueError(f'{path} is not a custom RPI chain file.')
    version = data.get('schema_version', CHAIN_SCHEMA_VERSION)
    try:
        newer = int(version) > CHAIN_SCHEMA_VERSION
    except (TypeError, ValueError):
        raise ValueError(f'{path}: schema_version {version!r} is not a number.')
    if newer:
        raise ValueError(f'{path} was written by a newer version of this GUI '
                         f'(schema_version {version} > {CHAIN_SCHEMA_VERSION}).')
    try:
        common = validate_common(data.get('common'))
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}.')
    raw = data['steps']
    if not isinstance(raw, list):
        raise ValueError(f'{path}: "steps" must be a list.')
    steps = []
    for i, s in enumerate(raw):
        try:
            steps.append(validate_step(s))
        except ValueError as exc:
            raise ValueError(f'{path}: step {i + 1}: {exc}.')
    name = data.get('name', '')
    return common, steps, name.strip() if isinstance(name, str) else ''
