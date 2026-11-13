"""
RPI Live Viewer — main entry point.

A single GUI that can run any of three RPI backends, or a chain of them:
  * "Fast conjugate gradient RPI"  (CGWorker)
  * "Moderate LSQML RPI"           (LSQMLWorker, ptychi)
  * "Slow LBFGS RPI"               (LBFGSWorker, cdtools)
  * "Custom RPI Chain"             (ChainWorker) — a list of steps built in
    "Create Custom RPI chain" (chain_editor.py), which holds any number of
    named chains. It is greyed out until a chain has a step, and chains are
    not kept across restarts (only in JSON files saved from that window).

The algorithm is chosen from a combo box in the main window; all RPI parameters
are edited in a separate non-blocking "Setup parameters" window. The currently
selected algorithm is used when Start is pressed, so stopping, switching the
combo, and restarting changes the backend.

A second combo picks between two stored object shapes — a coarse grid that
converges reliably and a fine one that does not — so the resolution/robustness
trade-off is one click rather than a trip into the parameters window. With
"Custom RPI Chain" selected, the same combo picks which chain runs instead
(each chain brings its own crop and object shape).

"Propagate probe…" opens its own non-blocking window holding the probe
propagation distance (10 µm per scroll step). It can be turned while a
reconstruction runs: the raw probe is re-propagated and handed to the running
worker, which picks it up before its next frame.

The probe library, beamline energy/distance and detector beam centre are
edited together in "Probes & Geometry…" and can be exported to
and imported from a JSON file. settings.ini remains the authoritative
per-session store; the JSON is the portable format, offered at startup and at
exit.

Usage:
    python rpi_live_gui.py

CUDA_VISIBLE_DEVICES is set from settings.ini *before* any torch/cdtools
import so that the correct GPU is exposed as device 0.
"""
from __future__ import annotations  # X | Y union syntax back to Python 3.7

import os
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt6"

# ── Set GPU before any torch/cdtools import ─────────────────────────────────
import os
import configparser as _cp

_pre_cfg = _cp.ConfigParser()
_pre_cfg.read(os.path.join(os.path.dirname(__file__), 'settings.ini'))
os.environ['CUDA_VISIBLE_DEVICES'] = _pre_cfg.get('rpi', 'gpu_index', fallback='0')
# keep Triton's current default (FP fusion on) explicitly; silences its
# "default will change in 3.10" warning when ptychi first compiles a kernel
os.environ.setdefault('TRITON_DEFAULT_FP_FUSION', '1')
# ────────────────────────────────────────────────────────────────────────────

import sys
import configparser
import datetime
import traceback
from pathlib import Path
from collections import deque

import numpy as np

import pyqtgraph as pg
from PyQt6 import uic
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QCheckBox, QLabel,
    QComboBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QLineEdit,
    QListWidget, QListWidgetItem, QScrollArea, QFrame, QWidget, QGridLayout,
)
from PyQt6.QtCore import Qt, QByteArray, QTimer
from PyQt6.QtGui import QBrush, QColor

from font_utils import apply_font_size_to_tree, DEFAULT_FONT_SIZE
from layout_utils import compact_group_boxes
from pg_image_tools import ImagePlotWidget
from rpi_utils import (
    load_probe_file,
    load_pixel_mask,
    save_rpi_result,
    autosave_append,
    energy_kev_to_wavelength,
    propagate_probe_angular_spectrum,
    illumination_mask,
    process_phase_region,
    normalize_probe_entry,
    build_config_dict,
    write_config_json,
    read_config_json,
    master_optics_type,
    classify_optic,
    UNKNOWN_OPTIC,
    DEFAULT_PROBE_SHAPE,
    DEFAULT_FILL_FRACTION,
)
from rpi_worker import WORKER_CLASSES
from rpi_chain import chain_to_json
from chain_editor import ChainEditor
from probe_viewer import ProbeViewer, DEFAULT_GEOMETRY


# Path to the .ui file (same directory as this script)
_UI_FILE    = Path(__file__).parent / 'rpi_live_gui.ui'
_INI_FILE   = Path(__file__).parent / 'settings.ini'

# Combo-box entries → internal method strings (index-aligned).
# 'CHAIN' runs the custom chain; it is never saved as the method in
# settings.ini or a probe's parameters, since the chain itself is not kept.
_METHOD_LABELS = ['Fast conjugate gradient RPI', 'Moderate LSQML RPI', 'Slow LBFGS RPI',
                  'Custom RPI Chain']
_METHODS       = ['CG', 'LSQML', 'LBFGS', 'CHAIN']
# Column titles in the RPI Parameters window (full names in their tooltips).
_METHOD_SHORT_LABELS = ['Fast CG RPI', 'Moderate LSQML RPI', 'Slow LBFGS RPI']

# Combo-box entries → internal object-initialisation strings (index-aligned).
# Shared by all algorithms: CG and LSQML use it directly, LBFGS maps 'flat'
# to cdtools' 'uniform' initialisation.
_INIT_LABELS = ['Random', 'Flat']
_INITS       = ['random', 'flat']

# Resolution mode → which of the two stored object shapes is used.
# res_rpi_m = res_m_probe * N_probe / obj_shape, so a bigger obj_shape means
# smaller pixels (higher resolution) on a harder-to-converge problem.
_RES_MODE_LABELS    = ['Low res, good convergence', 'High res, bad convergence']
_RES_MODES          = ['lowres', 'highres']
_DEFAULT_OBJ_SHAPES = {'lowres': 100, 'highres': 200}
# Detector crop (diffraction pattern size) stored per resolution mode too.
_DEFAULT_CROP_NS    = {'lowres': 512, 'highres': 512}

# LBFGS probe update (index-aligned). cdtools' RPI model keeps the probe fixed;
# 'weights' trains its N_modes x N_modes mixing of the loaded probe modes,
# 'full' trains every probe pixel. (ptychi has no single-pattern equivalent of
# the mixing matrix, so LSQML only has off/full, as a checkbox.)
_LBFGS_PROBE_UPDATE_LABELS = ['Off', 'Mode weight matrix', 'Full probe']
_LBFGS_PROBE_UPDATES       = ['off', 'weights', 'full']

# Fudge factors applied to a probe's filling fraction. Tune these here rather
# than exposing them in the UI.
_ZOOM_FUDGE         = 1.2   # "Zoom on illuminated area" box
_PHASE_REGION_FUDGE = 1.0   # unwrap / ramp-removal region

_LOG_FLOOR = 1e-12          # floor for the log colour scale

_START_BTN_STYLE = 'QPushButton { background-color: #2e9e44; color: white; font-weight: bold; }'
_STOP_BTN_STYLE  = 'QPushButton { background-color: #c0392b; color: white; font-weight: bold; }'

_EXIT_BTN_FONT_DELTA = 3    # EXIT is this many points larger than everything else

# "..." buttons on the main panel, sized to their text in _on_font_size_changed.
_BROWSE_BTNS = ('base_path_browse_btn', 'probe_browse_btn', 'pixel_mask_browse_btn',
                'save_dir_browse_btn', 'h5_browse_btn')

# RPI parameters that can be stored per-probe (gpu_index is excluded: it is a
# machine setting, not a probe property).
_PROBE_PARAM_KEYS = (
    'method', 'res_mode', 'obj_shape_lowres', 'obj_shape_highres',
    'crop_N_lowres', 'crop_N_highres', 'obj_init',
    'cg_n_iter', 'cg_clear_every', 'cg_loss_target', 'cg_probe_modes',
    'lbfgs_reg_factor1', 'lbfgs_reg_factor2_lo', 'lbfgs_reg_factor2_hi',
    'lbfgs_lr1', 'lbfgs_lr2', 'lbfgs_n_iter1', 'lbfgs_n_iter2', 'lbfgs_probe_modes',
    'lbfgs_probe_update', 'lbfgs_probe_update_start',
    'lsqml_n_epochs', 'lsqml_probe_modes', 'lsqml_step_scaler', 'lsqml_momentum',
    'lsqml_update_probe', 'lsqml_probe_update_start', 'lsqml_early_stop_pct',
)

# Main-window widgets switched off while the Object Accumulator is open. Left
# alone: the RPI Parameters group, "Probes & Geometry…", the
# EPICS PV field, the font size and EXIT.
_ACCUM_LOCK_GROUPS = ('h5_group', 'saving_group', 'display_group')
_ACCUM_LOCK_WIDGETS = (
    'label_probe', 'probe_file_edit', 'probe_browse_btn',
    'label_probe_list', 'probe_list_widget',
    'label_mask', 'pixel_mask_file_edit', 'pixel_mask_browse_btn',
    'label_savedir', 'save_dir_edit', 'save_dir_browse_btn',
    'start_btn', 'view_pv_btn', 'view_deque_btn', 'allow_repeat_check',
)


def _named(widget, name):
    """setObjectName(...) and return the widget, so font_utils.apply_font_size_to_tree
    (which skips unnamed widgets) reaches everything we build in code."""
    widget.setObjectName(name)
    return widget


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi(str(_UI_FILE), self)
        compact_group_boxes(self.scrollContents)

        # pg_image_tools viewer (lineout, measure, filters, compare, ... in the
        # right-click menu). self.image_view stays the bare pg.ImageView.
        self.image_plot = ImagePlotWidget(self.image_plot_container, keep_points=True)
        self.image_view = _named(self.image_plot.image_view, 'image_view')
        # Zoom is driven here (zoom-to-illuminated-area), not reset every frame.
        self.image_plot.action_no_zoom_tracking.setChecked(True)
        # Display options that live in the image's right-click menu.
        self.autolevel_action  = self.image_plot.action_auto_colorscale
        self.zoom_illum_action = self.image_plot.add_menu_action(
            'Zoom on illuminated area', checkable=True,
            after=self.image_plot.action_auto_colorscale)
        self.zoom_illum_action.setToolTip(
            'Keep the view zoomed to the illuminated area of the active probe. '
            'Unchecking resets to the full object once, then leaves the view alone.')
        # Resizing the window grows the image, not the controls.
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        # Runtime state
        self._worker                   = None
        self._results_deque: deque     = deque(maxlen=3)
        self._probe_raw                = None   # (N_opr, N_modes, N, N) complex
        self._probe_path               = None
        self._probe_choices: list      = []     # [{name, path, …}, …] probe library
        self._probe_data_cache: dict   = {}     # {path: {'probe', 'res_m'}}, shared
                                                 # across ProbeViewer opens so probes
                                                 # already loaded aren't re-read
        self._probe_list_paths: list   = []     # paths index-aligned with probe_list_widget
        self._pixel_mask_full          = None   # 2-D float32
        self._res_m_probe              = None   # float
        self._autosave_h5_path: str | None = None  # set at first frame of each run
        self._h5_optics_checked: str | None = None  # H5 replay file whose optics_type was tried
        self._pv_viewer: _PVImageViewer | None = None
        self._accum_win                = None   # ObjectAccumulatorWindow, lazily imported
        self._prop_win                 = None   # "Propagate Probe" window
        self._chain_win: ChainEditor | None = None  # "Custom RPI Chain" editor, built lazily
        self._plain_method_idx         = 0      # last algo_combo index other than CHAIN
        self._res_mode_idx             = 0      # index into _RES_MODES; res_mode_combo
                                                 # shows it unless CHAIN is selected
        self._chain_sel_id             = None   # id of the chain res_mode_combo picked
        self._fit_left_pane_pending    = True   # no saved splitter position yet
        self._prop_status_label        = None
        self._prop_applied_um          = None   # distance last pushed to a worker
        self._accum_settings: dict     = {}     # survives the window being closed
        self._closing                  = False

        # Beamline + detector geometry, edited in the setup window rather than
        # on the main panel (Crop N lives in the RPI parameters window).
        self._geometry = dict(DEFAULT_GEOMETRY)

        # Reconstruction-image cursor readout state
        self._last_display_img  = None   # processed image currently shown
        self._last_pixel_size_m = None   # metres/pixel of the current frame
        self._mouse_rc          = None   # (row, col) under the cursor, or None

        # Display state
        self._deque_label_items = []     # pg.TextItems overlaying the deque mosaic

        # Build the RPI parameter widgets + the non-blocking parameters window,
        # and populate the combo boxes.
        self._build_rpi_param_widgets()
        self._build_params_window()
        self._build_propagation_window()
        self._build_paths_window()
        self._set_start_btn(running=False)
        self.algo_combo.addItems(_METHOD_LABELS)
        self.algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        self._res_mode_tooltip = self.res_mode_combo.toolTip()
        self.res_mode_combo.currentIndexChanged.connect(self._on_res_combo_changed)
        self._fill_res_combo()
        self._refresh_chain_option()

        self._connect_signals()
        self._load_settings()

        _vb = self.image_view.getView()

        # Track the cursor over the reconstruction image for the readout label.
        self._recon_vb = _vb
        self._recon_mouse_proxy = pg.SignalProxy(
            _vb.scene().sigMouseMoved, rateLimit=60, slot=self._on_recon_mouse_moved
        )

        self._wire_font_size()

        # Offer to replace the settings.ini configuration with a JSON one
        # before anything is read from disk.
        self._prompt_import_config()

        # Probe selection happens through "Probes & Geometry…" —
        # opened here once at startup — rather than an automatic disk read of
        # settings.ini's last-used probe path.
        self._on_change_probe()
        self._try_load_mask()

    # ------------------------------------------------------------------ #
    # RPI parameter widgets + Setup-parameters window                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _spin(minimum, maximum, value, special=None, name=None):
        s = QSpinBox()
        s.setRange(minimum, maximum)
        s.setValue(value)
        if special is not None:
            s.setSpecialValueText(special)
        if name:
            s.setObjectName(name)
        return s

    @staticmethod
    def _dspin(minimum, maximum, value, decimals, step, name=None):
        s = QDoubleSpinBox()
        s.setDecimals(decimals)
        s.setRange(minimum, maximum)
        s.setSingleStep(step)
        s.setValue(value)
        if name:
            s.setObjectName(name)
        return s

    def _build_rpi_param_widgets(self):
        """Create every RPI parameter widget (defaults; overwritten on load)."""
        # Common parameters
        self.obj_shape_lowres_spin = self._spin(
            32, 2048, _DEFAULT_OBJ_SHAPES['lowres'], name='obj_shape_lowres_spin')
        self.obj_shape_lowres_spin.setToolTip(
            'Object grid used by "Low res, good convergence". Coarser grid, '
            'larger pixels, more reliable convergence.'
        )
        self.obj_shape_highres_spin = self._spin(
            32, 2048, _DEFAULT_OBJ_SHAPES['highres'], name='obj_shape_highres_spin')
        self.obj_shape_highres_spin.setToolTip(
            'Object grid used by "High res, bad convergence". Finer grid, '
            'smaller pixels, harder to converge.'
        )
        self.crop_N_lowres_spin = self._spin(
            64, 2048, _DEFAULT_CROP_NS['lowres'], name='crop_N_lowres_spin')
        self.crop_N_lowres_spin.setToolTip(
            'Detector crop size used by "Low res, good convergence". Per-run '
            'choice, not part of the exported JSON config.')
        self.crop_N_highres_spin = self._spin(
            64, 2048, _DEFAULT_CROP_NS['highres'], name='crop_N_highres_spin')
        self.crop_N_highres_spin.setToolTip(
            'Detector crop size used by "High res, bad convergence". Per-run '
            'choice, not part of the exported JSON config.')
        self.n_avg_spin       = self._spin(1, 1000, 1, name='n_avg_spin')
        self.n_avg_spin.setToolTip(
            'Average this many consecutive diffraction patterns (H5 or PV) '
            'into one before each reconstruction. 1 = no averaging. An '
            'acquisition setting, not stored per probe.')
        self.gpu_index_spin   = self._spin(0, 7, 3, name='gpu_index_spin')
        self.gpu_index_spin.setToolTip('Applies at next launch (GPU is selected '
                                       'before torch is imported).')
        self.obj_init_combo   = _named(QComboBox(), 'obj_init_combo')
        self.obj_init_combo.addItems(_INIT_LABELS)
        self.obj_init_combo.setCurrentIndex(_INITS.index('flat'))
        self.obj_init_combo.setToolTip(
            'Initial object guess used by both algorithms each frame.'
        )
        # Lives in its own "Propagate probe" window, not the parameters one,
        # because it is meant to be turned while a reconstruction runs.
        self.probe_prop_um_spin = self._dspin(-1e6, 1e6, 0.0, 2, 10.0,
                                              name='probe_prop_um_spin')
        self.probe_prop_um_spin.setToolTip(
            'Propagate the loaded probe by this many microns (angular-spectrum '
            'method), always starting from the raw loaded probe. Scrolling '
            'steps by 10 µm. |value| < 1 µm is treated as no propagation.'
        )
        # Fast conjugate-gradient parameters
        self.n_iter_spin        = self._spin(1, 9999, 100, name='n_iter_spin')
        self.clear_every_spin   = self._spin(1, 9999, 10, name='clear_every_spin')
        self.loss_target_edit   = _named(QLineEdit('0'), 'loss_target_edit')
        self.cg_probe_modes_spin = self._spin(0, 32, 0, special='All',
                                              name='cg_probe_modes_spin')
        # Slow LBFGS parameters
        self.reg1_edit          = _named(QLineEdit('1e-5'), 'reg1_edit')
        self.reg2_lo_edit       = _named(QLineEdit('1e-8'), 'reg2_lo_edit')
        self.reg2_hi_edit       = _named(QLineEdit('1e-6'), 'reg2_hi_edit')
        self.lr1_spin           = self._dspin(0.001, 2.0, 0.2, 3, 0.01, name='lr1_spin')
        self.lr2_spin           = self._dspin(0.001, 2.0, 0.1, 3, 0.01, name='lr2_spin')
        self.n_iter1_spin       = self._spin(0, 1000, 20, name='n_iter1_spin')
        self.n_iter2_spin       = self._spin(0, 1000, 60, name='n_iter2_spin')
        self.lbfgs_probe_modes_spin = self._spin(0, 32, 0, special='All',
                                                 name='lbfgs_probe_modes_spin')
        self.lbfgs_probe_update_combo = _named(QComboBox(), 'lbfgs_probe_update_combo')
        self.lbfgs_probe_update_combo.addItems(_LBFGS_PROBE_UPDATE_LABELS)
        self.lbfgs_probe_update_combo.setToolTip(
            'From the iteration below on (counted across both stages):\n'
            '  Mode weight matrix: train the N_modes x N_modes mixing of the '
            'loaded probe modes (stays close to the loaded probe).\n'
            '  Full probe: train every probe pixel (can absorb the object on a '
            'single pattern).\n'
            'Each frame still starts from the loaded probe.'
        )
        self.lbfgs_probe_update_start_spin = self._spin(
            0, 2000, 10, name='lbfgs_probe_update_start_spin')
        # Slow LSQML (ptychi) parameters
        self.lsqml_n_epochs_spin    = self._spin(1, 100000, 200, name='lsqml_n_epochs_spin')
        self.lsqml_n_epochs_spin.setToolTip(
            'Epoch 0 only rescales the probe intensity to the pattern; '
            'the object is updated from epoch 1 on.'
        )
        self.lsqml_probe_modes_spin = self._spin(0, 32, 0, special='All',
                                                 name='lsqml_probe_modes_spin')
        self.lsqml_step_scaler_spin = self._dspin(0.001, 2.0, 0.9, 3, 0.05,
                                                  name='lsqml_step_scaler_spin')
        self.lsqml_step_scaler_spin.setToolTip(
            'Scales the optimal object step size LSQML computes (ptychi default 0.9).')
        self.lsqml_momentum_spin    = self._dspin(0.0, 2.0, 0.0, 3, 0.05,
                                                  name='lsqml_momentum_spin')
        self.lsqml_momentum_spin.setToolTip('Momentum acceleration gain (0 = off).')
        self.lsqml_early_stop_spin  = self._dspin(0.0, 100.0, 1.0, 2, 0.5,
                                                  name='lsqml_early_stop_spin')
        self.lsqml_early_stop_spin.setSuffix(' %')
        self.lsqml_early_stop_spin.setSpecialValueText('Off')
        self.lsqml_early_stop_spin.setToolTip(
            'Stop a frame early once the loss changes by no more than this '
            'over 10 epochs. 0 = never stop early. With probe update on, the '
            'check starts 10 epochs after the probe update does.'
        )
        self.lsqml_update_probe_check = _named(QCheckBox('Update probe'),
                                               'lsqml_update_probe_check')
        self.lsqml_update_probe_check.setToolTip(
            'Let LSQML refine the probe from the epoch below on. Each frame '
            'still starts from the loaded probe.'
        )
        self.lsqml_probe_update_start_spin = self._spin(
            0, 100000, 100, name='lsqml_probe_update_start_spin')
        # A start iteration only means something while its probe update is on.
        self.lsqml_probe_update_start_spin.setEnabled(self.lsqml_update_probe_check.isChecked())
        self.lsqml_update_probe_check.toggled.connect(self.lsqml_probe_update_start_spin.setEnabled)
        self.lbfgs_probe_update_start_spin.setEnabled(self._current_lbfgs_probe_update() != 'off')
        self.lbfgs_probe_update_combo.currentIndexChanged.connect(
            lambda _i: self.lbfgs_probe_update_start_spin.setEnabled(
                self._current_lbfgs_probe_update() != 'off'))

    # Per-probe RPI parameter set: widget getters/setters keyed by name.
    # gpu_index is intentionally excluded (a machine setting, not a probe one).
    def _get_current_rpi_params(self):
        return {
            'method':              self._plain_method(),
            'res_mode':            self._current_res_mode(),
            'obj_shape_lowres':    self.obj_shape_lowres_spin.value(),
            'obj_shape_highres':   self.obj_shape_highres_spin.value(),
            'crop_N_lowres':       self.crop_N_lowres_spin.value(),
            'crop_N_highres':      self.crop_N_highres_spin.value(),
            'obj_init':            self._current_obj_init(),
            'cg_n_iter':           self.n_iter_spin.value(),
            'cg_clear_every':      self.clear_every_spin.value(),
            'cg_loss_target':      self.loss_target_edit.text(),
            'cg_probe_modes':      self.cg_probe_modes_spin.value(),
            'lbfgs_reg_factor1':   self.reg1_edit.text(),
            'lbfgs_reg_factor2_lo': self.reg2_lo_edit.text(),
            'lbfgs_reg_factor2_hi': self.reg2_hi_edit.text(),
            'lbfgs_lr1':           self.lr1_spin.value(),
            'lbfgs_lr2':           self.lr2_spin.value(),
            'lbfgs_n_iter1':       self.n_iter1_spin.value(),
            'lbfgs_n_iter2':       self.n_iter2_spin.value(),
            'lbfgs_probe_modes':   self.lbfgs_probe_modes_spin.value(),
            'lbfgs_probe_update':  self._current_lbfgs_probe_update(),
            'lbfgs_probe_update_start': self.lbfgs_probe_update_start_spin.value(),
            'lsqml_n_epochs':      self.lsqml_n_epochs_spin.value(),
            'lsqml_probe_modes':   self.lsqml_probe_modes_spin.value(),
            'lsqml_step_scaler':   self.lsqml_step_scaler_spin.value(),
            'lsqml_momentum':      self.lsqml_momentum_spin.value(),
            'lsqml_update_probe':  self.lsqml_update_probe_check.isChecked(),
            'lsqml_probe_update_start': self.lsqml_probe_update_start_spin.value(),
            'lsqml_early_stop_pct': self.lsqml_early_stop_spin.value(),
        }

    def _apply_rpi_params(self, p):
        """Load a saved per-probe parameter dict into the editable widgets."""
        def _i(key, default):
            try:    return int(p[key])
            except (KeyError, ValueError, TypeError): return default
        def _f(key, default):
            try:    return float(p[key])
            except (KeyError, ValueError, TypeError): return default
        def _b(key, default):
            # Stored as a bool in memory, as 'True'/'False' text in the ini.
            if key not in p: return default
            return str(p[key]).strip().lower() == 'true'
        method = p.get('method')
        if method in _METHODS and method != 'CHAIN':
            self.algo_combo.setCurrentIndex(_METHODS.index(method))
        self._set_res_mode(p.get('res_mode'))
        self.obj_shape_lowres_spin.setValue(
            _i('obj_shape_lowres', self.obj_shape_lowres_spin.value()))
        self.obj_shape_highres_spin.setValue(
            _i('obj_shape_highres', self.obj_shape_highres_spin.value()))
        self.crop_N_lowres_spin.setValue(
            _i('crop_N_lowres', self.crop_N_lowres_spin.value()))
        self.crop_N_highres_spin.setValue(
            _i('crop_N_highres', self.crop_N_highres_spin.value()))
        obj_init = p.get('obj_init')
        if obj_init in _INITS:
            self.obj_init_combo.setCurrentIndex(_INITS.index(obj_init))
        self.n_iter_spin.setValue(_i('cg_n_iter', self.n_iter_spin.value()))
        self.clear_every_spin.setValue(_i('cg_clear_every', self.clear_every_spin.value()))
        if 'cg_loss_target' in p:
            self.loss_target_edit.setText(str(p['cg_loss_target']))
        self.cg_probe_modes_spin.setValue(_i('cg_probe_modes', self.cg_probe_modes_spin.value()))
        if 'lbfgs_reg_factor1' in p:
            self.reg1_edit.setText(str(p['lbfgs_reg_factor1']))
        if 'lbfgs_reg_factor2_lo' in p:
            self.reg2_lo_edit.setText(str(p['lbfgs_reg_factor2_lo']))
        if 'lbfgs_reg_factor2_hi' in p:
            self.reg2_hi_edit.setText(str(p['lbfgs_reg_factor2_hi']))
        self.lr1_spin.setValue(_f('lbfgs_lr1', self.lr1_spin.value()))
        self.lr2_spin.setValue(_f('lbfgs_lr2', self.lr2_spin.value()))
        self.n_iter1_spin.setValue(_i('lbfgs_n_iter1', self.n_iter1_spin.value()))
        self.n_iter2_spin.setValue(_i('lbfgs_n_iter2', self.n_iter2_spin.value()))
        self.lbfgs_probe_modes_spin.setValue(_i('lbfgs_probe_modes', self.lbfgs_probe_modes_spin.value()))
        lbfgs_pu = p.get('lbfgs_probe_update')
        if lbfgs_pu in _LBFGS_PROBE_UPDATES:
            self.lbfgs_probe_update_combo.setCurrentIndex(_LBFGS_PROBE_UPDATES.index(lbfgs_pu))
        self.lbfgs_probe_update_start_spin.setValue(
            _i('lbfgs_probe_update_start', self.lbfgs_probe_update_start_spin.value()))
        self.lsqml_n_epochs_spin.setValue(_i('lsqml_n_epochs', self.lsqml_n_epochs_spin.value()))
        self.lsqml_probe_modes_spin.setValue(_i('lsqml_probe_modes', self.lsqml_probe_modes_spin.value()))
        self.lsqml_step_scaler_spin.setValue(_f('lsqml_step_scaler', self.lsqml_step_scaler_spin.value()))
        self.lsqml_momentum_spin.setValue(_f('lsqml_momentum', self.lsqml_momentum_spin.value()))
        self.lsqml_update_probe_check.setChecked(
            _b('lsqml_update_probe', self.lsqml_update_probe_check.isChecked()))
        self.lsqml_probe_update_start_spin.setValue(
            _i('lsqml_probe_update_start', self.lsqml_probe_update_start_spin.value()))
        self.lsqml_early_stop_spin.setValue(
            _f('lsqml_early_stop_pct', self.lsqml_early_stop_spin.value()))

    @staticmethod
    def _format_rpi_params(p):
        if not p:
            return 'No parameters saved for this probe.'
        return (
            f"method: {p.get('method', '?')}     res mode: {p.get('res_mode', '?')}     "
            f"obj init: {p.get('obj_init', '?')}\n"
            f"obj shape: low={p.get('obj_shape_lowres', '?')}  "
            f"high={p.get('obj_shape_highres', '?')}     "
            f"crop N: low={p.get('crop_N_lowres', '?')}  "
            f"high={p.get('crop_N_highres', '?')}\n"
            f"[CG]    n_iter={p.get('cg_n_iter', '?')}  clear_every={p.get('cg_clear_every', '?')}  "
            f"loss_target={p.get('cg_loss_target', '?')}  probe_modes={p.get('cg_probe_modes', '?')}\n"
            f"[LBFGS] reg={p.get('lbfgs_reg_factor1', '?')}/{p.get('lbfgs_reg_factor2_lo', '?')}/"
            f"{p.get('lbfgs_reg_factor2_hi', '?')}  lr={p.get('lbfgs_lr1', '?')}/{p.get('lbfgs_lr2', '?')}  "
            f"n_iter={p.get('lbfgs_n_iter1', '?')}/{p.get('lbfgs_n_iter2', '?')}  "
            f"probe_modes={p.get('lbfgs_probe_modes', '?')}  "
            f"{MainWindow._probe_update_text(p.get('lbfgs_probe_update', 'off'), p.get('lbfgs_probe_update_start'))}\n"
            f"[LSQML] epochs={p.get('lsqml_n_epochs', '?')}  "
            f"step_scaler={p.get('lsqml_step_scaler', '?')}  "
            f"momentum={p.get('lsqml_momentum', '?')}  "
            f"early_stop={p.get('lsqml_early_stop_pct', '?')}%  "
            f"probe_modes={p.get('lsqml_probe_modes', '?')}  "
            f"{MainWindow._probe_update_text(
                'full' if str(p.get('lsqml_update_probe', False)).strip().lower() == 'true' else 'off',
                p.get('lsqml_probe_update_start'))}"
        )

    @staticmethod
    def _probe_update_text(mode, start):
        if mode == 'off':
            return 'probe fixed'
        what = 'probe mode weights' if mode == 'weights' else 'probe'
        return f"{what} updated from {start if start is not None else '?'}"

    def _build_params_window(self):
        """Non-blocking window: editable params on top, one column per method,
        plus a probe list whose per-probe saved parameters can be viewed and
        copied to / saved from the editable fields. Everything but Close
        scrolls, so the window can be made smaller than its contents."""
        win = QDialog(self)
        win.setObjectName('params_win')
        win.setWindowTitle('RPI Parameters')
        win.setWindowFlags(win.windowFlags() | Qt.WindowType.Window)
        win_layout = QVBoxLayout(win)
        scroll = _named(QScrollArea(), 'params_scroll')
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = _named(QWidget(), 'params_body')
        scroll.setWidget(body)
        win_layout.addWidget(scroll, 1)
        outer = QVBoxLayout(body)
        outer.setContentsMargins(0, 0, 0, 0)

        common_box = _named(QGroupBox('Common'), 'common_box')
        cvl = QVBoxLayout(common_box)
        # Rows: resolution mode; columns: detector crop, object shape.
        res_grid = QGridLayout()
        res_grid.addWidget(_named(QLabel('Det crop'), 'label_crop_head'), 0, 1)
        res_grid.addWidget(_named(QLabel('Obj shape'), 'label_obj_head'), 0, 2)
        for r, (label, crop, obj) in enumerate((
                ('Low res:', self.crop_N_lowres_spin, self.obj_shape_lowres_spin),
                ('High res:', self.crop_N_highres_spin, self.obj_shape_highres_spin))):
            lab = _named(QLabel(label), f'label_res_{_RES_MODES[r]}')
            lab.setToolTip(_RES_MODE_LABELS[r])
            res_grid.addWidget(lab, r + 1, 0)
            res_grid.addWidget(crop, r + 1, 1)
            res_grid.addWidget(obj, r + 1, 2)
        # A column to the right with the remaining common parameters.
        res_grid.setColumnMinimumWidth(3, 16)
        for r, (label, name, tip, w) in enumerate((
                ('Obj init:', 'label_obj_init', 'Initial object', self.obj_init_combo),
                ('Avg N dp:', 'label_n_avg', 'Average this many diffraction '
                 'patterns before each reconstruction', self.n_avg_spin),
                ('GPU idx:', 'label_gpu', 'GPU index', self.gpu_index_spin))):
            lab = _named(QLabel(label), name)
            lab.setToolTip(tip)
            res_grid.addWidget(lab, r, 4)
            res_grid.addWidget(w, r, 5)
        for c in (1, 2, 5):
            res_grid.setColumnStretch(c, 1)
        cvl.addLayout(res_grid)
        chain_btn = _named(QPushButton('Create Custom RPI chain'), 'create_chain_btn')
        chain_btn.setToolTip(
            'Build a list of CG / LSQML / LBFGS steps run in order on each '
            'frame, selectable as "Custom RPI Chain" in the main window.')
        chain_btn.clicked.connect(self._on_open_chain)
        cvl.addWidget(chain_btn)
        outer.addWidget(common_box)

        cols = QHBoxLayout()

        cg_box = _named(QGroupBox(_METHOD_SHORT_LABELS[0]), 'cg_box')
        cgf = QFormLayout(cg_box)
        cgf.addRow(_named(QLabel('N iter:'), 'label_cg_niter'),      self.n_iter_spin)
        cgf.addRow(_named(QLabel('Clear every:'), 'label_cg_clear'), self.clear_every_spin)
        cgf.addRow(_named(QLabel('Loss target:'), 'label_cg_loss'), self.loss_target_edit)
        cgf.labelForField(self.loss_target_edit).setToolTip('Stop at this loss; 0 = off.')
        cgf.addRow(_named(QLabel('Probe modes:'), 'label_cg_modes'), self.cg_probe_modes_spin)

        lsq_box = _named(QGroupBox(_METHOD_SHORT_LABELS[1]), 'lsqml_box')
        lsf = QFormLayout(lsq_box)
        lsf.addRow(_named(QLabel('N epochs:'), 'label_lsq_nep'),          self.lsqml_n_epochs_spin)
        lsf.addRow(_named(QLabel('Obj step scaler:'), 'label_lsq_step'), self.lsqml_step_scaler_spin)
        lsf.addRow(_named(QLabel('Momentum gain:'), 'label_lsq_mom'),     self.lsqml_momentum_spin)
        lsf.addRow(_named(QLabel('Early stop:'), 'label_lsq_es'),
                   self.lsqml_early_stop_spin)
        lsf.labelForField(self.lsqml_early_stop_spin).setToolTip(
            'Stop when the loss changes by less than this over 10 epochs.')
        lsf.addRow(_named(QLabel('Probe modes:'), 'label_lsq_modes'),     self.lsqml_probe_modes_spin)
        lsf.addRow(self.lsqml_update_probe_check)
        lsf.addRow(_named(QLabel('Probe upd. from ep:'), 'label_lsq_pu_start'),
                   self.lsqml_probe_update_start_spin)

        lb_box = _named(QGroupBox(_METHOD_SHORT_LABELS[2]), 'lb_box')
        lbf = QFormLayout(lb_box)
        lbf.addRow(_named(QLabel('Reg factor 1:'), 'label_lb_reg1'),    self.reg1_edit)
        lbf.addRow(_named(QLabel('Reg factor 2 lo:'), 'label_lb_reg2lo'), self.reg2_lo_edit)
        lbf.addRow(_named(QLabel('Reg factor 2 hi:'), 'label_lb_reg2hi'), self.reg2_hi_edit)
        lbf.addRow(_named(QLabel('LR 1:'), 'label_lb_lr1'),            self.lr1_spin)
        lbf.addRow(_named(QLabel('LR 2:'), 'label_lb_lr2'),            self.lr2_spin)
        lbf.addRow(_named(QLabel('N iter 1:'), 'label_lb_n1'),         self.n_iter1_spin)
        lbf.addRow(_named(QLabel('N iter 2:'), 'label_lb_n2'),         self.n_iter2_spin)
        lbf.addRow(_named(QLabel('Probe modes:'), 'label_lb_modes'),   self.lbfgs_probe_modes_spin)
        lbf.addRow(_named(QLabel('Probe update:'), 'label_lb_pu'),
                   self.lbfgs_probe_update_combo)
        lbf.addRow(_named(QLabel('Probe upd. from it:'), 'label_lb_pu_start'),
                   self.lbfgs_probe_update_start_spin)

        for i, box in enumerate((cg_box, lsq_box, lb_box)):
            box.setToolTip(_METHOD_LABELS[i])
            cols.addWidget(box, 0, Qt.AlignmentFlag.AlignTop)

        outer.addLayout(cols)

        # ---- Per-probe parameter presets ----
        preset_box = _named(QGroupBox('Per-probe parameters'), 'preset_box')
        pv = QVBoxLayout(preset_box)

        self._params_probe_list = _named(QListWidget(), 'params_probe_list')
        self._params_probe_list.setToolTip(
            'Probes from "Probes & Geometry…". '
            'Each can store its own RPI parameters.'
        )
        self._params_probe_list.currentRowChanged.connect(self._on_params_probe_selected)
        self._params_probe_list.setMinimumHeight(50)
        self._params_probe_list.setMaximumHeight(120)
        pv.addWidget(self._params_probe_list)

        btn_row = QHBoxLayout()
        self._save_to_probe_btn   = _named(QPushButton('Save to probe'),
                                           'save_to_probe_btn')
        self._copy_from_probe_btn = _named(QPushButton('Copy from probe'),
                                           'copy_from_probe_btn')
        self._save_to_probe_btn.setToolTip(
            'Store the RPI parameters above in the selected probe.')
        self._copy_from_probe_btn.setToolTip(
            "Replace the RPI parameters above with the selected probe's.")
        self._save_to_probe_btn.clicked.connect(self._on_save_params_to_probe)
        self._copy_from_probe_btn.clicked.connect(self._on_copy_params_from_probe)
        btn_row.addWidget(self._save_to_probe_btn)
        btn_row.addWidget(self._copy_from_probe_btn)
        pv.addLayout(btn_row)

        saved_box = _named(QGroupBox("Selected probe's saved parameters"),
                           'saved_box')
        sv = QVBoxLayout(saved_box)
        self._saved_params_label = _named(QLabel('No probe selected.'), 'saved_params_label')
        self._saved_params_label.setWordWrap(True)
        self._saved_params_label.setStyleSheet('font-family: monospace;')
        self._saved_params_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        sv.addWidget(self._saved_params_label)
        pv.addWidget(saved_box)

        outer.addWidget(preset_box)

        outer.addStretch(1)

        close_btn = _named(QPushButton('Close'), 'params_close_btn')
        close_btn.clicked.connect(win.hide)
        win_layout.addWidget(close_btn)

        self._params_scroll = scroll
        self._params_win_sized = False   # sized to its contents on first open
        self._params_win = win

    # ------------------------------------------------------------------ #
    # Setup Paths window                                                  #
    # ------------------------------------------------------------------ #

    def _build_paths_window(self):
        """Non-blocking window for the paths that rarely change: probe file,
        pixel mask and save dir. Their widgets come from the .ui
        (paths_more_widget), so everything that reads them is unchanged."""
        win = QDialog(self)
        win.setObjectName('paths_win')
        win.setWindowTitle('Setup Paths')
        win.setWindowFlags(win.windowFlags() | Qt.WindowType.Window)
        outer = QVBoxLayout(win)
        outer.addWidget(self.paths_more_widget)
        close_btn = _named(QPushButton('Close'), 'paths_close_btn')
        close_btn.clicked.connect(win.hide)
        outer.addWidget(close_btn)
        win.resize(560, win.sizeHint().height())
        self._paths_win = win

    def _on_setup_paths(self):
        apply_font_size_to_tree(self._paths_win, self._current_font_size())
        self._paths_win.show()
        self._paths_win.raise_()
        self._paths_win.activateWindow()

    # ------------------------------------------------------------------ #
    # Probe propagation window                                            #
    # ------------------------------------------------------------------ #

    def _build_propagation_window(self):
        """Non-blocking window holding the probe propagation distance.

        Turning the dial while a reconstruction is running re-propagates the
        raw probe and hands it to the worker, which picks it up before its
        next frame — no stop/start needed.
        """
        win = QDialog(self)
        win.setObjectName('prop_win')
        win.setWindowTitle('Propagate Probe')
        win.setWindowFlags(win.windowFlags() | Qt.WindowType.Window)
        outer = QVBoxLayout(win)

        form = QFormLayout()
        form.addRow(_named(QLabel('Distance (µm):'), 'label_prop'),
                    self.probe_prop_um_spin)
        outer.addLayout(form)

        self.prop_live_check = _named(
            QCheckBox('Apply live to a running reconstruction'),
            'prop_live_check')
        self.prop_live_check.setChecked(True)
        self.prop_live_check.setToolTip(
            'When on, every change is propagated and sent to a running worker '
            'after a short pause. When off, the value is only used at the '
            'next Start (or when "Apply now" is pressed).'
        )
        outer.addWidget(self.prop_live_check)

        self._prop_status_label = _named(QLabel(''), 'prop_status_label')
        self._prop_status_label.setWordWrap(True)
        outer.addWidget(self._prop_status_label)

        btn_row = QHBoxLayout()
        apply_btn = _named(QPushButton('Apply now'), 'prop_apply_btn')
        apply_btn.clicked.connect(lambda: self._apply_propagation(force=True))
        reset_btn = _named(QPushButton('Reset to 0 µm'), 'prop_reset_btn')
        reset_btn.clicked.connect(lambda: self.probe_prop_um_spin.setValue(0.0))
        close_btn = _named(QPushButton('Close'), 'prop_close_btn')
        close_btn.clicked.connect(win.hide)
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(reset_btn)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

        # Debounce: a wheel spin fires valueChanged per step, and each one
        # would otherwise cost a propagation.
        self._prop_apply_timer = QTimer(self)
        self._prop_apply_timer.setSingleShot(True)
        self._prop_apply_timer.setInterval(300)
        self._prop_apply_timer.timeout.connect(self._apply_propagation)
        self.probe_prop_um_spin.valueChanged.connect(self._on_prop_value_changed)

        self._prop_win = win
        self._prop_applied_um = None   # distance last handed to a worker
        self._update_prop_status()

    def _on_prop_value_changed(self, _value):
        self._update_prop_status()
        if self.prop_live_check.isChecked():
            self._prop_apply_timer.start()

    def _live_workers(self):
        """Running workers that can accept a new probe right now.

        The Object Accumulator runs its own worker, so it is included.
        """
        workers = []
        for w in (self._worker,
                  getattr(self._accum_win, '_worker', None)):
            if w is not None and w.isRunning():
                workers.append(w)
        return workers

    def _apply_propagation(self, force=False):
        """Re-propagate the raw probe and hand it to any running worker."""
        self._prop_apply_timer.stop()
        distance_um = self.probe_prop_um_spin.value()
        workers = self._live_workers()
        if not workers:
            self._update_prop_status()
            if force:
                _probe, err = self._propagated_probe()   # validate only
                if err:
                    QMessageBox.warning(self._prop_win, 'Propagation error', err)
                else:
                    self._set_status(
                        f'Nothing is running — {distance_um:.2f} µm will be '
                        f'applied at the next Start.')
            return

        probe, err = self._propagated_probe()
        if err:
            self._prop_status_label.setText(f'Propagation failed: {err}')
            if force:
                QMessageBox.warning(self._prop_win, 'Propagation error', err)
            return
        for w in workers:
            w.set_probe(probe, f'{distance_um:.2f} µm')
        self._prop_applied_um = distance_um
        self._update_prop_status()
        self._set_status(f'Probe propagation set to {distance_um:.2f} µm '
                         f'— applied to the running reconstruction.')

    def _update_prop_status(self):
        if self._prop_status_label is None:
            return
        distance_um = self.probe_prop_um_spin.value()
        if self._live_workers():
            if self._prop_applied_um is None:
                state = (f'Running with the distance set at Start. '
                         f'Current field: {distance_um:.2f} µm.')
            elif abs(self._prop_applied_um - distance_um) < 1e-9:
                state = f'Running with {self._prop_applied_um:.2f} µm applied.'
            else:
                state = (f'Running with {self._prop_applied_um:.2f} µm; '
                         f'{distance_um:.2f} µm not applied yet.')
        else:
            state = (f'Nothing running — {distance_um:.2f} µm will be used '
                     f'at the next Start.')
        self._prop_status_label.setText(state)

    def _on_open_propagation(self):
        self._update_prop_status()
        apply_font_size_to_tree(self._prop_win, self._current_font_size())
        self._prop_win.show()
        self._prop_win.raise_()
        self._prop_win.activateWindow()

    def _current_method(self):
        return _METHODS[self.algo_combo.currentIndex()]

    def _plain_method(self):
        """The selected method, or the last one before CHAIN was picked —
        what settings.ini and per-probe parameters store."""
        return _METHODS[self._plain_method_idx]

    def _current_lbfgs_probe_update(self):
        return _LBFGS_PROBE_UPDATES[max(0, self.lbfgs_probe_update_combo.currentIndex())]

    def _current_obj_init(self):
        return _INITS[self.obj_init_combo.currentIndex()]

    def _current_res_mode(self):
        return _RES_MODES[self._res_mode_idx]

    def _set_res_mode(self, res_mode):
        if res_mode not in _RES_MODES:
            return
        self._res_mode_idx = _RES_MODES.index(res_mode)
        if self._current_method() != 'CHAIN':
            self.res_mode_combo.setCurrentIndex(self._res_mode_idx)

    def _res_mode_crop_N(self):
        """Detector crop size implied by the resolution combo."""
        if self._current_res_mode() == 'highres':
            return self.crop_N_highres_spin.value()
        return self.crop_N_lowres_spin.value()

    def _res_mode_obj_shape(self):
        """Object grid size implied by the resolution combo."""
        if self._current_res_mode() == 'highres':
            return self.obj_shape_highres_spin.value()
        return self.obj_shape_lowres_spin.value()

    def _current_crop_N(self):
        """Detector crop size of the next run: the chain's when it is selected."""
        if self._current_method() == 'CHAIN' and self._chain_is_set(self._chain_sel_id):
            return self._chain_win.common(self._chain_sel_id)['crop_N']
        return self._res_mode_crop_N()

    def _current_obj_shape(self):
        """Object grid size of the next run: the chain's when it is selected."""
        if self._current_method() == 'CHAIN' and self._chain_is_set(self._chain_sel_id):
            return self._chain_win.common(self._chain_sel_id)['obj_shape']
        return self._res_mode_obj_shape()

    def _refresh_params_probe_list(self):
        """Repopulate the Setup-window probe list from the current library."""
        lw = self._params_probe_list
        prev = lw.currentRow()
        lw.blockSignals(True)
        lw.clear()
        for entry in self._probe_choices:
            item = QListWidgetItem(entry['name'])
            item.setToolTip(entry['path'])
            lw.addItem(item)
        lw.blockSignals(False)
        if 0 <= prev < lw.count():
            lw.setCurrentRow(prev)
        elif lw.count():
            lw.setCurrentRow(0)
        else:
            self._on_params_probe_selected(-1)

    def _on_params_probe_selected(self, row):
        has = 0 <= row < len(self._probe_choices)
        self._save_to_probe_btn.setEnabled(has)
        self._copy_from_probe_btn.setEnabled(
            has and bool(self._probe_choices[row].get('params'))
        )
        if has:
            self._saved_params_label.setText(
                self._format_rpi_params(self._probe_choices[row].get('params'))
            )
        else:
            self._saved_params_label.setText('No probe selected.')

    def _on_save_params_to_probe(self):
        row = self._params_probe_list.currentRow()
        if not (0 <= row < len(self._probe_choices)):
            return
        self._probe_choices[row]['params'] = self._get_current_rpi_params()
        self._on_params_probe_selected(row)   # refresh read-only view + buttons
        self._set_status(f'Saved RPI parameters to probe '
                         f'“{self._probe_choices[row]["name"]}”.')

    def _on_copy_params_from_probe(self):
        row = self._params_probe_list.currentRow()
        if not (0 <= row < len(self._probe_choices)):
            return
        params = self._probe_choices[row].get('params')
        if not params:
            QMessageBox.information(self._params_win, 'No parameters',
                                    'This probe has no saved RPI parameters.')
            return
        self._apply_rpi_params(params)
        self._set_status(f'Copied RPI parameters from probe '
                         f'“{self._probe_choices[row]["name"]}”.')

    def _on_setup_params(self):
        self._refresh_params_probe_list()
        apply_font_size_to_tree(self._params_win, self._current_font_size())
        if not self._params_win_sized:
            # Big enough to show everything at the current font, within the
            # screen; it can still be shrunk, and then scrolls.
            self._params_win_sized = True
            win, scroll = self._params_win, self._params_scroll
            want = scroll.widget().sizeHint()
            extra = win.sizeHint() - scroll.sizeHint()
            avail = (win.screen() or QApplication.primaryScreen()).availableGeometry()
            win.resize(min(want.width() + extra.width() + 4, avail.width()),
                       min(want.height() + extra.height() + 4, avail.height() - 60))
        self._params_win.show()
        self._params_win.raise_()
        self._params_win.activateWindow()

    # ------------------------------------------------------------------ #
    # Custom RPI chain                                                     #
    # ------------------------------------------------------------------ #

    def _chain_is_set(self, cid=None):
        """True if chain `cid` (any chain, if None) has a step."""
        return self._chain_win is not None and self._chain_win.is_set(cid)

    def _chain_errors(self):
        """Problems with the chain picked in res_mode_combo."""
        if not self._chain_is_set(self._chain_sel_id):
            return ['No custom RPI chain with steps is selected. Build one with '
                    '"Create Custom RPI chain" in the RPI Parameters window, and '
                    'pick it in the combo under the method.']
        return self._chain_win.errors(self._chain_sel_id)

    def _chain_common_defaults(self):
        """Common chain parameters taken from the RPI Parameters window."""
        return {
            'n_probe_modes': 0,
            'crop_N':        self._res_mode_crop_N(),
            'obj_shape':     self._res_mode_obj_shape(),
            'obj_init':      self._current_obj_init(),
            'n_avg':         self.n_avg_spin.value(),
        }

    def _on_open_chain(self):
        if self._chain_win is None:
            self._chain_win = ChainEditor(self._chain_common_defaults,
                                          self._get_current_rpi_params, self)
            self._chain_win.chain_changed.connect(self._refresh_chain_option)
            self._refresh_chain_option()
        elif self._chain_sel_id is not None:
            self._chain_win.select(self._chain_sel_id)
        apply_font_size_to_tree(self._chain_win, self._current_font_size())
        self._chain_win.show()
        self._chain_win.raise_()
        self._chain_win.activateWindow()

    def _refresh_chain_option(self):
        """Grey out "Custom RPI Chain" while no chain has a step, and keep the
        chain names in res_mode_combo current.

        If the last chain with steps goes while CHAIN is selected, fall back
        to the last other method — unless a run is in progress (the combo is
        disabled then); the run keeps its own copy and this is redone when it
        ends.
        """
        is_set = self._chain_is_set()
        self.algo_combo.model().item(_METHODS.index('CHAIN')).setEnabled(is_set)
        if (not is_set and self._current_method() == 'CHAIN'
                and self.algo_combo.isEnabled()):
            self.algo_combo.setCurrentIndex(self._plain_method_idx)
        elif self._current_method() == 'CHAIN':
            self._fill_res_combo()

    def _fill_res_combo(self):
        """res_mode_combo lists the chains while CHAIN is selected (those
        without steps greyed out), and the two resolution modes otherwise."""
        combo = self.res_mode_combo
        combo.blockSignals(True)
        try:
            combo.clear()
            if self._current_method() == 'CHAIN':
                chains = self._chain_win.chains() if self._chain_win else []
                for cid, name, n in chains:
                    combo.addItem(f'{name} ({n} step{"s" if n != 1 else ""})', cid)
                    combo.model().item(combo.count() - 1).setEnabled(n > 0)
                runnable = [cid for cid, _name, n in chains if n]
                if self._chain_sel_id not in runnable:
                    self._chain_sel_id = runnable[0] if runnable else None
                combo.setCurrentIndex(combo.findData(self._chain_sel_id))
                combo.setToolTip('Custom chain to run. Chains are built and '
                                 'edited in "Create Custom RPI chain" '
                                 '(Setup parameters…).')
            else:
                combo.addItems(_RES_MODE_LABELS)
                combo.setCurrentIndex(self._res_mode_idx)
                combo.setToolTip(self._res_mode_tooltip)
        finally:
            combo.blockSignals(False)

    def _on_res_combo_changed(self, idx):
        if idx < 0:
            return
        if self._current_method() == 'CHAIN':
            self._chain_sel_id = self.res_mode_combo.itemData(idx)
            if self._chain_win is not None and self._chain_win.isVisible():
                self._chain_win.select(self._chain_sel_id)
        else:
            self._res_mode_idx = idx

    def _on_algo_changed(self, idx):
        if _METHODS[idx] != 'CHAIN':
            self._plain_method_idx = idx
        self._fill_res_combo()

    def _set_method_controls_enabled(self, enabled):
        """algo/res-mode combos: off while a run is in progress."""
        self.algo_combo.setEnabled(enabled)
        self.res_mode_combo.setEnabled(enabled)
        if enabled:
            self._refresh_chain_option()

    # ------------------------------------------------------------------ #
    # Signal wiring                                                        #
    # ------------------------------------------------------------------ #

    def _connect_signals(self):
        # Browse buttons
        self.base_path_browse_btn.clicked.connect(self._browse_base_path)
        self.probe_browse_btn.clicked.connect(
            lambda: self._browse_file(self.probe_file_edit, 'HDF5 (*.h5 *.hdf5)',
                                      self.browse_dir('probe')))
        self.pixel_mask_browse_btn.clicked.connect(
            lambda: self._browse_file(self.pixel_mask_file_edit, 'HDF5 (*.h5 *.hdf5)'))
        self.save_dir_browse_btn.clicked.connect(self._browse_save_dir)
        self.h5_browse_btn.clicked.connect(
            lambda: self._browse_file(self.h5_file_edit, 'HDF5 (*.h5 *.hdf5)',
                                      self.browse_dir('h5')))

        # H5 replay toggle
        self.h5_check.toggled.connect(self._on_h5_toggled)

        # Allow-repeat-frames toggle (live while a run is in progress)
        self.allow_repeat_check.toggled.connect(self._on_allow_repeat_toggled)

        # Reload probe/mask when paths change
        self.probe_file_edit.editingFinished.connect(self._try_load_probe)
        self.pixel_mask_file_edit.editingFinished.connect(self._try_load_mask)
        self.h5_file_edit.editingFinished.connect(self._on_h5_file_set)

        # RPI parameters window
        self.setup_params_btn.clicked.connect(self._on_setup_params)

        # Probe file / mask / save dir window
        self.setup_paths_btn.clicked.connect(self._on_setup_paths)

        # Probe propagation window (usable while a run is in progress)
        self.propagate_probe_btn.clicked.connect(self._on_open_propagation)

        # Main controls
        self.view_pv_btn.clicked.connect(self._on_view_pv_image)
        self.view_deque_btn.toggled.connect(self._on_view_deque)
        self.start_btn.clicked.connect(self._on_start_stop)
        self.change_probe_btn.clicked.connect(self._on_change_probe)
        self.accumulate_btn.clicked.connect(self._on_open_accumulator)
        self.probe_list_widget.itemClicked.connect(self._on_probe_list_clicked)
        self.save_btn.clicked.connect(self._on_save)
        self.exit_btn.clicked.connect(self.close)

        # Display toggles rerender the last frame
        self.unwrap_check.toggled.connect(self._rerender_last)
        self.ramp_check.toggled.connect(self._rerender_last)
        self.abs_check.toggled.connect(self._on_abs_toggled)
        self.log_check.toggled.connect(self._rerender_last)
        self.autolevel_action.toggled.connect(self._on_autolevel_toggled)
        self.zoom_illum_action.toggled.connect(self._on_zoom_illum_toggled)

    # ------------------------------------------------------------------ #
    # Settings persistence                                                 #
    # ------------------------------------------------------------------ #

    def _load_settings(self):
        cfg = configparser.ConfigParser()
        cfg.read(str(_INI_FILE))

        def _get(section, key, fallback):
            return cfg.get(section, key, fallback=fallback)

        self.base_path_edit.setText(         _get('paths',    'base_path',        ''))
        self.probe_file_edit.setText(        _get('paths',    'probe_file',       ''))
        self.pixel_mask_file_edit.setText(   _get('paths',    'pixel_mask_file',  ''))
        self.pv_name_edit.setText(           _get('epics',    'pv_name',          ''))
        self.save_dir_edit.setText(          _get('paths',    'save_dir',         ''))
        self.h5_check.setChecked(            _get('h5_replay', 'enabled',          'false').lower() == 'true')
        self.h5_file_edit.setText(           _get('h5_replay', 'h5_file',          ''))
        # The restored probe already goes with the restored file.
        self._h5_optics_checked = self.h5_file_edit.text().strip()
        self.h5_start_spin.setValue(    int(  _get('h5_replay', 'h5_start_idx',     '0')))
        self.h5_stride_spin.setValue(   int(  _get('h5_replay', 'h5_stride',        '1')))
        self._on_h5_toggled(self.h5_check.isChecked())

        # ---- Beamline / detector geometry (edited in the setup window) ----
        self._geometry = {
            'energy_kev':        float(_get('beamline', 'energy_kev',        '12.0')),
            'detector_distance': float(_get('beamline', 'detector_distance', '2.0')),
            'beam_center_y':     int(  _get('detector', 'beam_center_y',     '512')),
            'beam_center_x':     int(  _get('detector', 'beam_center_x',     '512')),
        }

        # ---- RPI: common ----
        # Legacy fallback: obj_shape used to be a single common setting.
        legacy_shape = _get('rpi', 'obj_shape', '')
        self.obj_shape_lowres_spin.setValue(int(
            _get('rpi', 'obj_shape_lowres',
                 legacy_shape or str(_DEFAULT_OBJ_SHAPES['lowres']))))
        self.obj_shape_highres_spin.setValue(int(
            _get('rpi', 'obj_shape_highres', str(_DEFAULT_OBJ_SHAPES['highres']))))
        # Legacy fallback: crop_N used to be one setting in [detector].
        legacy_crop = _get('detector', 'crop_N', '')
        self.crop_N_lowres_spin.setValue(int(
            _get('rpi', 'crop_N_lowres', legacy_crop or str(_DEFAULT_CROP_NS['lowres']))))
        self.crop_N_highres_spin.setValue(int(
            _get('rpi', 'crop_N_highres', legacy_crop or str(_DEFAULT_CROP_NS['highres']))))
        self._set_res_mode(_get('rpi', 'res_mode', 'lowres'))
        self.gpu_index_spin.setValue(   int(  _get('rpi', 'gpu_index',    '3')))
        self.n_avg_spin.setValue(       int(  _get('rpi', 'n_avg',        '1')))
        self.probe_prop_um_spin.setValue(float(_get('rpi', 'probe_propagation_um', '0')))
        obj_init = _get('rpi', 'obj_init', 'flat')
        self.obj_init_combo.setCurrentIndex(
            _INITS.index(obj_init) if obj_init in _INITS else _INITS.index('flat')
        )
        method = _get('rpi', 'method', 'CG')
        self.algo_combo.setCurrentIndex(
            _METHODS.index(method) if method in _METHODS and method != 'CHAIN' else 0)
        # Legacy fallback: probe_modes used to be a single common setting.
        legacy_modes = _get('rpi', 'probe_modes', '0')

        # ---- RPI: conjugate-gradient ----
        self.n_iter_spin.setValue(      int(  _get('rpi_cg', 'n_iter',       '100')))
        self.clear_every_spin.setValue( int(  _get('rpi_cg', 'clear_every',  '10')))
        self.loss_target_edit.setText(        _get('rpi_cg', 'loss_target',  '0'))
        self.cg_probe_modes_spin.setValue(int(_get('rpi_cg', 'probe_modes', legacy_modes)))

        # ---- RPI: LBFGS ----
        self.reg1_edit.setText(               _get('rpi_lbfgs', 'reg_factor1',    '1e-5'))
        self.reg2_lo_edit.setText(            _get('rpi_lbfgs', 'reg_factor2_lo', '1e-8'))
        self.reg2_hi_edit.setText(            _get('rpi_lbfgs', 'reg_factor2_hi', '1e-6'))
        self.lr1_spin.setValue(         float(_get('rpi_lbfgs', 'lr1',            '0.2')))
        self.lr2_spin.setValue(         float(_get('rpi_lbfgs', 'lr2',            '0.1')))
        self.n_iter1_spin.setValue(     int(  _get('rpi_lbfgs', 'n_iter1',        '20')))
        self.n_iter2_spin.setValue(     int(  _get('rpi_lbfgs', 'n_iter2',        '60')))
        self.lbfgs_probe_modes_spin.setValue(int(_get('rpi_lbfgs', 'probe_modes', legacy_modes)))
        lbfgs_pu = _get('rpi_lbfgs', 'probe_update', 'off')
        self.lbfgs_probe_update_combo.setCurrentIndex(
            _LBFGS_PROBE_UPDATES.index(lbfgs_pu) if lbfgs_pu in _LBFGS_PROBE_UPDATES else 0)
        self.lbfgs_probe_update_start_spin.setValue(
            int(_get('rpi_lbfgs', 'probe_update_start', '10')))

        # ---- RPI: LSQML ----
        self.lsqml_n_epochs_spin.setValue(   int(  _get('rpi_lsqml', 'n_epochs',     '200')))
        self.lsqml_probe_modes_spin.setValue(int(  _get('rpi_lsqml', 'probe_modes',  legacy_modes)))
        self.lsqml_step_scaler_spin.setValue(float(_get('rpi_lsqml', 'step_scaler',  '0.9')))
        self.lsqml_momentum_spin.setValue(   float(_get('rpi_lsqml', 'momentum',     '0.0')))
        self.lsqml_update_probe_check.setChecked(
            _get('rpi_lsqml', 'update_probe', 'false').lower() == 'true')
        self.lsqml_probe_update_start_spin.setValue(
            int(_get('rpi_lsqml', 'probe_update_start', '100')))
        self.lsqml_early_stop_spin.setValue(float(_get('rpi_lsqml', 'early_stop_pct', '1.0')))

        # ---- Display ----
        self.unwrap_check.setChecked(         _get('display', 'unwrap',     'false').lower() == 'true')
        self.ramp_check.setChecked(           _get('display', 'ramp',       'false').lower() == 'true')
        self.abs_check.setChecked(            _get('display', 'abs',        'false').lower() == 'true')
        self.zoom_illum_action.setChecked(     _get('display', 'zoom_illum', 'false').lower() == 'true')
        self.autolevel_action.setChecked(      _get('display', 'autolevel',  'true').lower() == 'true')
        self.log_check.setChecked(            _get('display', 'log',        'false').lower() == 'true')
        self._on_abs_toggled(self.abs_check.isChecked())

        # ---- App ----
        self.deque_length_spin.setValue(int(  _get('app', 'deque_length', '3')))
        self.autosave_check.setChecked(       _get('app', 'autosave',     'false').lower() == 'true')

        # ---- Probe-choice library (each entry: name, path, shape, fill, params) ----
        self._probe_choices = []
        n_choices = int(_get('probe_choices', 'count', '0'))
        for i in range(n_choices):
            pth = _get('probe_choices', f'path{i}', '')
            if not pth:
                continue
            params = {}
            sec = f'probe_params_{i}'
            if cfg.has_section(sec):
                for k in _PROBE_PARAM_KEYS:
                    if cfg.has_option(sec, k):
                        params[k] = cfg.get(sec, k)
                # Migrate a legacy single obj_shape into the low-res slot, so
                # values tuned before the two-shape split keep working.
                if 'obj_shape_lowres' not in params and cfg.has_option(sec, 'obj_shape'):
                    params['obj_shape_lowres'] = cfg.get(sec, 'obj_shape')
            self._probe_choices.append(normalize_probe_entry({
                'name':          _get('probe_choices', f'name{i}', ''),
                'path':          pth,
                'enabled':       _get('probe_choices', f'enabled{i}', 'true').lower() == 'true',
                'shape':         _get('probe_choices', f'shape{i}', DEFAULT_PROBE_SHAPE),
                'fill_fraction': _get('probe_choices', f'fill_fraction{i}',
                                      str(DEFAULT_FILL_FRACTION)),
                'params':        params,
            }))

        # ---- Window geometry + splitter position ----
        geo = _get('window', 'geometry', '')
        if geo:
            self.restoreGeometry(QByteArray.fromHex(geo.encode()))
        spl = _get('window', 'main_splitter', '')
        if spl:
            self.splitter.restoreState(QByteArray.fromHex(spl.encode()))
        self._fit_left_pane_pending = not spl

        # Line edits: keep the tooltip in sync with the (possibly truncated)
        # text so hovering always reveals the full value. Restricted to our own
        # fields (pyqtgraph's ImageView has many internal QLineEdits).
        for le in (self.base_path_edit, self.probe_file_edit,
                   self.pixel_mask_file_edit, self.pv_name_edit,
                   self.save_dir_edit, self.h5_file_edit,
                   self.loss_target_edit, self.reg1_edit, self.reg2_lo_edit,
                   self.reg2_hi_edit):
            le.setToolTip(le.text())
            le.textChanged.connect(lambda text, w=le: w.setToolTip(text))

    def _save_settings(self):
        cfg = configparser.ConfigParser()
        cfg['paths']    = {
            'base_path':        self.base_path_edit.text(),
            'probe_file':       self.probe_file_edit.text(),
            'pixel_mask_file':  self.pixel_mask_file_edit.text(),
            'save_dir':         self.save_dir_edit.text(),
        }
        cfg['epics']    = {'pv_name': self.pv_name_edit.text()}
        cfg['h5_replay'] = {
            'enabled':      str(self.h5_check.isChecked()).lower(),
            'h5_file':      self.h5_file_edit.text(),
            'h5_start_idx': str(self.h5_start_spin.value()),
            'h5_stride':    str(self.h5_stride_spin.value()),
        }
        cfg['beamline'] = {
            'energy_kev':        str(self._geometry['energy_kev']),
            'detector_distance': str(self._geometry['detector_distance']),
        }
        cfg['detector'] = {
            'beam_center_y': str(self._geometry['beam_center_y']),
            'beam_center_x': str(self._geometry['beam_center_x']),
        }
        cfg['rpi']      = {
            'obj_shape_lowres':    str(self.obj_shape_lowres_spin.value()),
            'obj_shape_highres':   str(self.obj_shape_highres_spin.value()),
            'crop_N_lowres':       str(self.crop_N_lowres_spin.value()),
            'crop_N_highres':      str(self.crop_N_highres_spin.value()),
            'res_mode':            self._current_res_mode(),
            'obj_init':            self._current_obj_init(),
            'gpu_index':           str(self.gpu_index_spin.value()),
            'n_avg':               str(self.n_avg_spin.value()),
            'method':              self._plain_method(),
            'probe_propagation_um': str(self.probe_prop_um_spin.value()),
        }
        cfg['rpi_cg']   = {
            'n_iter':       str(self.n_iter_spin.value()),
            'clear_every':  str(self.clear_every_spin.value()),
            'loss_target':  self.loss_target_edit.text(),
            'probe_modes':  str(self.cg_probe_modes_spin.value()),
        }
        cfg['rpi_lbfgs'] = {
            'reg_factor1':    self.reg1_edit.text(),
            'reg_factor2_lo': self.reg2_lo_edit.text(),
            'reg_factor2_hi': self.reg2_hi_edit.text(),
            'lr1':            str(self.lr1_spin.value()),
            'lr2':            str(self.lr2_spin.value()),
            'n_iter1':        str(self.n_iter1_spin.value()),
            'n_iter2':        str(self.n_iter2_spin.value()),
            'probe_modes':    str(self.lbfgs_probe_modes_spin.value()),
            'probe_update':   self._current_lbfgs_probe_update(),
            'probe_update_start': str(self.lbfgs_probe_update_start_spin.value()),
        }
        cfg['rpi_lsqml'] = {
            'n_epochs':       str(self.lsqml_n_epochs_spin.value()),
            'probe_modes':    str(self.lsqml_probe_modes_spin.value()),
            'step_scaler':    str(self.lsqml_step_scaler_spin.value()),
            'momentum':       str(self.lsqml_momentum_spin.value()),
            'update_probe':   str(self.lsqml_update_probe_check.isChecked()).lower(),
            'probe_update_start': str(self.lsqml_probe_update_start_spin.value()),
            'early_stop_pct': str(self.lsqml_early_stop_spin.value()),
        }
        cfg['display']  = {
            'unwrap':     str(self.unwrap_check.isChecked()).lower(),
            'ramp':       str(self.ramp_check.isChecked()).lower(),
            'abs':        str(self.abs_check.isChecked()).lower(),
            'zoom_illum': str(self.zoom_illum_action.isChecked()).lower(),
            'autolevel':  str(self.autolevel_action.isChecked()).lower(),
            'log':        str(self.log_check.isChecked()).lower(),
        }
        cfg['app']      = {
            'deque_length': str(self.deque_length_spin.value()),
            'autosave':     str(self.autosave_check.isChecked()).lower(),
            'font_size':    str(self.spinBox_fontSize.value()),
        }
        cfg['window']   = {
            'geometry': bytes(self.saveGeometry().toHex()).decode(),
            'main_splitter': bytes(self.splitter.saveState().toHex()).decode(),
        }
        probe_choices = {'count': str(len(self._probe_choices))}
        for i, entry in enumerate(self._probe_choices):
            probe_choices[f'name{i}'] = entry['name']
            probe_choices[f'path{i}'] = entry['path']
            probe_choices[f'enabled{i}'] = str(entry.get('enabled', True))
            probe_choices[f'shape{i}'] = entry.get('shape', DEFAULT_PROBE_SHAPE)
            probe_choices[f'fill_fraction{i}'] = str(
                entry.get('fill_fraction', DEFAULT_FILL_FRACTION))
        cfg['probe_choices'] = probe_choices
        # Per-probe RPI parameter sets (only for probes that have one).
        for i, entry in enumerate(self._probe_choices):
            params = entry.get('params') or {}
            if params:
                cfg[f'probe_params_{i}'] = {k: str(v) for k, v in params.items()}

        # The accumulator owns its own section. Prefer the live window, fall
        # back to what it handed over when it was closed, and only then to
        # what is already on disk — otherwise settings changed during an
        # accumulator session are lost whenever that window is closed before
        # the main one.
        accum = None
        if self._accum_win is not None:
            try:
                accum = self._accum_win.get_settings()
            except Exception:
                accum = None
        if accum is None and self._accum_settings:
            accum = self._accum_settings
        if accum is not None:
            cfg['accumulator'] = {k: str(v) for k, v in accum.items()}
        else:
            old = configparser.ConfigParser()
            old.read(str(_INI_FILE))
            if old.has_section('accumulator'):
                cfg['accumulator'] = dict(old['accumulator'])

        with open(str(_INI_FILE), 'w') as f:
            cfg.write(f)

    def closeEvent(self, event):
        if self._closing:
            event.accept()
            return
        self._closing = True
        if self._accum_win is not None:
            self._accum_win.close()
        self._save_settings()
        self._prompt_export_config()
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        super().closeEvent(event)

    # ------------------------------------------------------------------ #
    # Font size                                                            #
    # ------------------------------------------------------------------ #

    def _wire_font_size(self):
        cfg = configparser.ConfigParser()
        cfg.read(str(_INI_FILE))
        size = cfg.getint('app', 'font_size', fallback=DEFAULT_FONT_SIZE)
        self.spinBox_fontSize.setValue(size)
        self.spinBox_fontSize.valueChanged.connect(self._on_font_size_changed)
        # Deferred: a window built before the event loop starts has a backlog
        # of show/restoreGeometry events that would otherwise undo the font.
        QTimer.singleShot(0, lambda: (self._on_font_size_changed(size),
                                      self._fit_left_pane()))

    def _fit_left_pane(self):
        """With no saved splitter position, make the left pane only as wide
        as its controls need (at the current font) and give the image the rest."""
        if not self._fit_left_pane_pending:
            return
        self._fit_left_pane_pending = False
        left = (self.scrollContents.minimumSizeHint().width()
                + self.scrollArea.verticalScrollBar().sizeHint().width()
                + 2 * self.scrollArea.frameWidth())
        self.splitter.setSizes([left, max(1, self.splitter.width() - left)])

    def _current_font_size(self):
        return self.spinBox_fontSize.value()

    def _on_font_size_changed(self, size):
        """Apply `size` to every window of the application.

        The application font covers what is built later or has no name
        (message boxes, file dialogs, menus); the tree pass covers the named
        widgets, whose .ui files bake in their own point size.
        """
        app = QApplication.instance()
        f = app.font()
        f.setPointSize(size)
        app.setFont(f)
        for w in app.topLevelWidgets():
            apply_font_size_to_tree(w, size)
        # EXIT is deliberately oversized; this must run *after* the tree pass,
        # which would otherwise flatten it back to `size`.
        f = self.exit_btn.font()
        f.setPointSize(size + _EXIT_BTN_FONT_DELTA)
        f.setBold(True)
        self.exit_btn.setFont(f)
        # QPushButton's style minimum (~80 px) would make each "..." as wide
        # as a word and waste the left pane's width.
        for name in _BROWSE_BTNS:
            b = getattr(self, name)
            b.setFixedWidth(b.fontMetrics().horizontalAdvance(b.text()) + 2 * size)
        # List items carry their own font, so they need rebuilding at the new size.
        self._refresh_probe_list_widget()

    # ------------------------------------------------------------------ #
    # Configuration JSON import / export prompts                          #
    # ------------------------------------------------------------------ #

    def _prompt_import_config(self):
        """Offer, at startup, to replace the settings.ini configuration with a
        JSON one. The dialog shows what the ini currently holds so declining is
        an informed choice."""
        dlg = _ConfigLoadDialog(self._probe_choices, self._geometry, self)
        apply_font_size_to_tree(dlg, self._current_font_size())
        dlg.exec()
        if dlg.choice != _ConfigLoadDialog.IMPORT_JSON:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, 'Import Configuration', '', 'JSON files (*.json);;All files (*)'
        )
        if not path:
            return
        try:
            probes, beamline, detector = read_config_json(path)
        except ValueError as exc:
            QMessageBox.warning(
                self, 'Import failed',
                f'{exc}\n\nKeeping the settings.ini configuration.'
            )
            return
        self._probe_choices = probes
        self._geometry.update(beamline)
        self._geometry.update(detector)
        self._set_status(f'Configuration imported from {path} '
                         f'({len(probes)} probes).')

    def _prompt_export_config(self):
        if QMessageBox.question(
            self, 'Export configuration',
            'Export the current probe list and geometry to a JSON file?\n\n'
            '(settings.ini has already been saved either way.)',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Configuration', '', 'JSON files (*.json)'
        )
        if not path:
            return
        if not path.lower().endswith('.json'):
            path += '.json'
        try:
            write_config_json(path, build_config_dict(
                self._probe_choices, self._geometry, self._geometry))
        except Exception as exc:
            QMessageBox.warning(self, 'Export failed', str(exc))

    # ------------------------------------------------------------------ #
    # File loading                                                         #
    # ------------------------------------------------------------------ #

    def _try_load_probe(self):
        path = self.probe_file_edit.text().strip()
        if not path or not Path(path).is_file():
            return
        try:
            probe, res_m = load_probe_file(path)
            self._probe_raw   = probe
            self._probe_path  = path
            self._res_m_probe = res_m
            self._probe_data_cache[path] = {'probe': probe, 'res_m': res_m}
            self._refresh_probe_list_widget()
            self._set_status(
                f'Probe loaded: shape={probe.shape}, '
                f'pixel size={res_m*1e9:.2f} nm'
            )
        except Exception as exc:
            self._set_status(f'Probe load error: {exc}')

    def _try_load_mask(self):
        path = self.pixel_mask_file_edit.text().strip()
        if not path or not Path(path).is_file():
            return
        try:
            self._pixel_mask_full = load_pixel_mask(path)
            self._set_status(
                f'Mask loaded: shape={self._pixel_mask_full.shape}'
            )
        except Exception as exc:
            self._set_status(f'Mask load error: {exc}')

    def _on_h5_file_set(self):
        """A new raw master file picks the initial probe matching its
        zone plate's optics_type. Preprocessed files, or a master without
        that attribute, are left alone; a failed match is one terminal line."""
        path = self.h5_file_edit.text().strip()
        if path == self._h5_optics_checked:
            return
        self._h5_optics_checked = path
        if not Path(path).is_file():
            return
        try:
            optics_type = master_optics_type(path)
        except Exception:
            return
        if optics_type is None:
            return
        try:
            self._select_probe_for_optic(classify_optic(optics_type), path)
        except Exception as exc:
            print(f'optics_type "{optics_type}": no initial probe set ({exc})',
                  flush=True)

    def _select_probe_for_optic(self, optic, h5_path):
        """Make the first enabled probe whose name classifies as `optic` the
        active one (raises with a short reason otherwise). The current probe
        is kept if it already matches; among the rest, the one whose file
        shares the longest path with the H5 file (same experiment) wins."""
        if optic == UNKNOWN_OPTIC:
            raise ValueError('unrecognised optic')
        if self._worker and self._worker.isRunning():
            raise RuntimeError('reconstruction is running')
        matches = [c for c in self._probe_choices
                   if c.get('enabled', True) and classify_optic(c['name']) == optic]
        if not matches:
            raise LookupError(f'no probe named like {optic}')
        if any(c['path'] == self._probe_path for c in matches):
            return
        h5_parts = Path(h5_path).resolve().parts

        def _shared(c):
            n = 0
            for a, b in zip(Path(c['path']).resolve().parts, h5_parts):
                if a != b:
                    break
                n += 1
            return n
        matches.sort(key=_shared, reverse=True)   # stable: list order breaks ties
        for c in matches:
            data = self._probe_data_cache.get(c['path'])
            if data is None:
                try:
                    probe, res_m = load_probe_file(c['path'])
                except Exception:
                    continue
                data = self._probe_data_cache[c['path']] = {'probe': probe,
                                                            'res_m': res_m}
            self._probe_raw   = data['probe']
            self._probe_path  = c['path']
            self._res_m_probe = data['res_m']
            self.probe_file_edit.setText(c['path'])
            self._refresh_probe_list_widget()
            self._rerender_last()
            self._set_status(f'Probe set from optics_type ({optic}): “{c["name"]}”')
            return
        raise OSError(f'no {optic} probe file could be loaded')

    # ------------------------------------------------------------------ #
    # Browse helpers                                                       #
    # ------------------------------------------------------------------ #

    def browse_dir(self, kind):
        """Starting folder for a file browser, derived from the base path.

        kind: 'probe' -> <base>/ptychi_recons/
              'save'  -> <base>/rpi_recons/rpi_live/, created if missing
              'h5'    -> <base>/preproc/, else <base>/ptycho/
        A folder that does not exist falls back to its nearest existing
        parent. Returns None when no base path is set, so callers keep their
        own default (usually the folder of the current field's value).
        """
        text = self.base_path_edit.text().strip()
        if not text:
            return None
        base = Path(text).expanduser()
        if kind == 'probe':
            target = base / 'ptychi_recons'
        elif kind == 'save':
            target = base / 'rpi_recons' / 'rpi_live'
            # Only under a base that exists — a typo in the base path should
            # not quietly build a new directory tree.
            if base.is_dir() and not target.is_dir():
                try:
                    target.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    self._set_status(f'Could not create {target}: {exc}')
        elif kind == 'h5':
            target = base / 'preproc'
            if not target.is_dir() and (base / 'ptycho').is_dir():
                target = base / 'ptycho'
        else:
            target = base
        return self._existing_ancestor(target)

    @staticmethod
    def _existing_ancestor(p):
        """p itself if it is a directory, else its nearest existing parent
        (as a str), or '' when nothing on the way up exists."""
        for d in (p, *p.parents):
            if d.is_dir():
                return str(d)
        return ''

    def _browse_base_path(self):
        dlg = QFileDialog(self)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dlg.setDirectory(self._start_dir_from_edit(self.base_path_edit))
        if dlg.exec():
            self.base_path_edit.setText(dlg.selectedFiles()[0])

    def _browse_file(self, line_edit, file_filter, start_dir=None):
        # Instantiate directly so setDirectory() is honoured — the static
        # helpers on some Linux/Qt builds ignore the directory argument even
        # with DontUseNativeDialog.
        dlg = QFileDialog(self)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        dlg.setNameFilter(file_filter)
        dlg.setDirectory(start_dir if start_dir is not None
                         else self._start_dir_from_edit(line_edit))
        if dlg.exec():
            path = dlg.selectedFiles()[0]
            line_edit.setText(path)
            line_edit.editingFinished.emit()

    def _browse_save_dir(self):
        dlg = QFileDialog(self)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        start_dir = self.browse_dir('save')
        dlg.setDirectory(start_dir if start_dir is not None
                         else self._start_dir_from_edit(self.save_dir_edit))
        if dlg.exec():
            self.save_dir_edit.setText(dlg.selectedFiles()[0])

    @staticmethod
    def _start_dir_from_edit(line_edit):
        """Return the best starting directory for a file dialog given the
        current text of a line edit.  Works for partial paths too."""
        text = line_edit.text().strip()
        if not text:
            return ''
        p = Path(text)
        if p.is_dir():
            return str(p)
        if p.parent.is_dir():
            return str(p.parent)
        # Walk up until we find an existing ancestor
        for parent in p.parents:
            if parent.is_dir():
                return str(parent)
        return ''

    # ------------------------------------------------------------------ #
    # Debug / display toggles                                              #
    # ------------------------------------------------------------------ #

    def _on_h5_toggled(self, checked):
        self.h5_file_edit.setEnabled(checked)
        self.h5_browse_btn.setEnabled(checked)
        self.label_h5_start.setEnabled(checked)
        self.h5_start_spin.setEnabled(checked)
        self.label_h5_stride.setEnabled(checked)
        self.h5_stride_spin.setEnabled(checked)
        self.pv_name_edit.setEnabled(not checked)

    def _on_allow_repeat_toggled(self, checked):
        if self._worker is not None:
            self._worker.params['allow_repeat_frames'] = checked

    def _on_abs_toggled(self, checked):
        # np.angle() is signed, so a log colour scale only makes sense on |obj|.
        self.log_check.setEnabled(checked)
        # Unwrap / ramp removal are phase operations.
        self.unwrap_check.setEnabled(not checked)
        self.ramp_check.setEnabled(not checked)
        self._rerender_last()

    def _on_autolevel_toggled(self, checked):
        # Turning it off holds whatever is on screen, including any later
        # histogram drags: pg_image_tools leaves the levels alone then.
        self._rerender_last()

    def _on_zoom_illum_toggled(self, checked):
        if checked:
            self._apply_zoom()
        else:
            # Reset to the full object once, then leave the view alone so the
            # user can pan and zoom freely.
            self._recon_vb.autoRange()

    # ------------------------------------------------------------------ #
    # Start / Stop                                                         #
    # ------------------------------------------------------------------ #

    def _on_start_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()
            self._set_start_btn(running=False)
            self._set_method_controls_enabled(True)
            self._set_status('Stopped.')
            self._prop_applied_um = None
            self._update_prop_status()
            return

        # Validate
        if self._probe_raw is None:
            QMessageBox.warning(self, 'No probe', 'Load a probe file first.')
            return
        if self._pixel_mask_full is None:
            QMessageBox.warning(self, 'No mask', 'Load a pixel mask file first.')
            return
        if not self.h5_check.isChecked() and not self.pv_name_edit.text().strip():
            QMessageBox.warning(self, 'No PV', 'Enter an EPICS PV name.')
            return
        if self.h5_check.isChecked() and not Path(self.h5_file_edit.text()).is_file():
            QMessageBox.warning(self, 'No H5 file', 'Select a valid H5 file for replay.')
            return
        if self.autosave_check.isChecked() and not self.save_dir_edit.text().strip():
            QMessageBox.warning(self, 'No save dir',
                                'Set a save directory before enabling auto-save.')
            return
        if self._current_method() == 'CHAIN' and self._chain_errors():
            QMessageBox.warning(self, 'Custom RPI chain',
                                'Fix the custom RPI chain first:\n\n'
                                + '\n'.join(self._chain_errors()))
            return

        probe_for_worker = self.prepare_probe_for_run()
        if probe_for_worker is None:
            return

        self._autosave_h5_path = None   # new file for each run
        params = self._collect_params()

        # Recreate deque with current length
        self._results_deque = deque(maxlen=params['deque_length'])
        self.save_entry_spin.setMaximum(0)

        worker_cls = WORKER_CLASSES[params['method']]
        self._worker = worker_cls(
            params            = params,
            probe_raw         = probe_for_worker,
            pixel_mask_full   = self._pixel_mask_full,
            results_deque     = self._results_deque,
            parent            = self,
        )
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.status_update.connect(self._set_status)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.raw_image_ready.connect(self._on_raw_image_ready)
        self._worker.finished.connect(self._on_worker_finished)

        self._worker.start()
        self._set_start_btn(running=True)
        self._set_method_controls_enabled(False)   # algorithm is fixed while running
        if params['method'] == 'CHAIN':
            n = len(params['chain_steps'])
            detail = f'"{params["chain_name"]}", {n} step{"s" if n != 1 else ""}'
        else:
            detail = _RES_MODE_LABELS[self._res_mode_idx]
        self._set_status(f'Starting… ({_METHOD_LABELS[self.algo_combo.currentIndex()]}, '
                         f'{detail})')
        self._update_prop_status()

    def _set_start_btn(self, running):
        """Green "Start" while idle, red "Stop" while a run is in progress."""
        self.start_btn.setText('Stop' if running else 'Start')
        self.start_btn.setStyleSheet(_STOP_BTN_STYLE if running else _START_BTN_STYLE)

    def _propagated_probe(self):
        """(probe, error_message) for the current propagation distance.

        Propagation always starts from the raw loaded probe, so repeated
        Start/Stop cycles and repeated live changes never compound it.
        """
        if self._probe_raw is None:
            return None, 'No probe is loaded.'
        distance_um = self.probe_prop_um_spin.value()
        if abs(distance_um) < 1.0:
            return self._probe_raw, None
        try:
            probe = propagate_probe_angular_spectrum(
                self._probe_raw, self._res_m_probe,
                energy_kev_to_wavelength(self._geometry['energy_kev']),
                distance_um,
            )
            return probe, None
        except Exception as exc:
            return None, f'Could not propagate probe: {exc}'

    def prepare_probe_for_run(self):
        """Return the probe to hand to a worker, propagated if requested.

        Returns None (after showing a message) if propagation fails. Also used
        by the Object Accumulator.
        """
        probe, err = self._propagated_probe()
        if err:
            QMessageBox.warning(self, 'Propagation error', err)
            return None
        distance_um = self.probe_prop_um_spin.value()
        if abs(distance_um) >= 1.0:
            self._set_status(f'Propagated probe by {distance_um:.2f} µm before starting.')
        self._prop_applied_um = distance_um
        self._update_prop_status()
        return probe

    def _on_worker_finished(self):
        self._set_start_btn(running=False)
        self._set_method_controls_enabled(True)
        self._prop_applied_um = None
        self._update_prop_status()

    def _collect_params(self):
        """Read all relevant GUI fields into a plain dict for the worker.

        Returns a superset containing every algorithm's parameters plus the
        selected 'method'; each backend uses only the keys it needs.
        """
        def _safe_float(text, default):
            try:
                return float(text.strip())
            except (ValueError, AttributeError):
                return default

        method = self._current_method()
        if method == 'CHAIN':
            chain_name = self._chain_win.name(self._chain_sel_id)
            chain_common, chain_steps = self._chain_win.snapshot(self._chain_sel_id)
            n_probe_modes = chain_common['n_probe_modes']
        else:
            # Probe modes is per-algorithm; pick the active algorithm's value.
            n_probe_modes = {
                'CG':    self.cg_probe_modes_spin,
                'LBFGS': self.lbfgs_probe_modes_spin,
                'LSQML': self.lsqml_probe_modes_spin,
            }[method].value()
        probe_entry = self._active_probe_entry()
        probe_shape, fill_fraction = self._active_probe_geometry()

        params = {
            'method':           method,
            'pv_name':          self.pv_name_edit.text().strip(),
            'h5_mode':          self.h5_check.isChecked(),
            'h5_file':          self.h5_file_edit.text().strip(),
            'h5_start_idx':     self.h5_start_spin.value(),
            'h5_stride':        self.h5_stride_spin.value(),
            'beam_cy':          self._geometry['beam_center_y'],
            'beam_cx':          self._geometry['beam_center_x'],
            'crop_N':           self._current_crop_N(),
            'res_mode':         self._current_res_mode(),
            'obj_shape':        self._current_obj_shape(),
            'obj_init':         self._current_obj_init(),
            'gpu_index':        self.gpu_index_spin.value(),
            'n_avg':            self.n_avg_spin.value(),
            'n_probe_modes':    n_probe_modes,  # 0 = use all
            'allow_repeat_frames': self.allow_repeat_check.isChecked(),
            # --- CG params ---
            'n_iter':           self.n_iter_spin.value(),
            'clear_every':      self.clear_every_spin.value(),
            'loss_target':      _safe_float(self.loss_target_edit.text(), 0.0),
            # --- LBFGS params ---
            'reg_factor1':      _safe_float(self.reg1_edit.text(),    1e-5),
            'reg_factor2_lo':   _safe_float(self.reg2_lo_edit.text(), 1e-8),
            'reg_factor2_hi':   _safe_float(self.reg2_hi_edit.text(), 1e-6),
            'lr1':              self.lr1_spin.value(),
            'lr2':              self.lr2_spin.value(),
            'n_iter1':          self.n_iter1_spin.value(),
            'n_iter2':          self.n_iter2_spin.value(),
            'lbfgs_probe_update':       self._current_lbfgs_probe_update(),
            'lbfgs_probe_update_start': self.lbfgs_probe_update_start_spin.value(),
            # --- LSQML params ---
            'lsqml_n_epochs':           self.lsqml_n_epochs_spin.value(),
            'lsqml_step_scaler':        self.lsqml_step_scaler_spin.value(),
            'lsqml_momentum':           self.lsqml_momentum_spin.value(),
            'lsqml_update_probe':       self.lsqml_update_probe_check.isChecked(),
            'lsqml_probe_update_start': self.lsqml_probe_update_start_spin.value(),
            'lsqml_early_stop_pct':     self.lsqml_early_stop_spin.value(),
            # --- shared app / beamline ---
            'deque_length':     self.deque_length_spin.value(),
            'energy_kev':       self._geometry['energy_kev'],
            'detector_distance': self._geometry['detector_distance'],
            # Probe metadata passed through to avoid re-reading inside worker,
            # and so results can record which probe produced them.
            'res_m_probe':      self._res_m_probe,
            'n_probe':          (self._probe_raw.shape[-1]
                                 if self._probe_raw is not None else None),
            'probe_shape':      probe_shape,
            'fill_fraction':    fill_fraction,
            'probe_name':       (probe_entry or {}).get('name', ''),
            'probe_path':       self._probe_path or '',
        }
        if method == 'CHAIN':
            # The chain's common parameters replace the window's. crop_N and
            # obj_shape already come from it (_current_crop_N/_obj_shape).
            params.update({
                'res_mode':    'chain',
                'obj_init':    chain_common['obj_init'],
                'n_avg':       chain_common['n_avg'],
                'chain_name':  chain_name,
                'chain_steps': chain_steps,
                'chain_json':  chain_to_json(chain_common, chain_steps, chain_name),
            })
        return params

    def snapshot_run_inputs(self):
        """Validate and gather everything a worker needs for one run.

        Shared by Start and the Object Accumulator so both enforce the same
        preconditions. Returns (params, probe, pixel_mask_full); raises
        RuntimeError with a message meant to be shown to the user.
        """
        if self._probe_raw is None:
            raise RuntimeError(
                'No probe is loaded. Open "Probes & Geometry…", '
                'press "Load registered probes", and accept one.'
            )
        if self._pixel_mask_full is None:
            raise RuntimeError(
                'No pixel mask is loaded. Set a valid mask file on the main panel.'
            )
        if self.h5_check.isChecked():
            if not Path(self.h5_file_edit.text()).is_file():
                raise RuntimeError(
                    f'H5 replay is enabled but "{self.h5_file_edit.text()}" '
                    f'is not a file.'
                )
        elif not self.pv_name_edit.text().strip():
            raise RuntimeError('No EPICS image PV is set on the main panel.')
        if self._current_method() == 'CHAIN' and self._chain_errors():
            raise RuntimeError('Fix the custom RPI chain first:\n'
                               + '\n'.join(self._chain_errors()))
        probe = self.prepare_probe_for_run()
        if probe is None:
            raise RuntimeError('Probe propagation failed — see the message above.')
        return self._collect_params(), probe, self._pixel_mask_full

    # ------------------------------------------------------------------ #
    # Frame display                                                        #
    # ------------------------------------------------------------------ #

    def _on_frame_ready(self, entry):
        # Update save spin range to reflect how many entries are in the deque
        n = len(self._results_deque)
        self.save_entry_spin.setMaximum(max(0, n - 1))

        if self.view_deque_btn.isChecked():
            self._render_mosaic()
        else:
            self._render_entry(entry)

        if self.autosave_check.isChecked():
            self._autosave_frame(entry)

    def _autosave_frame(self, entry):
        save_dir = self.save_dir_edit.text().strip()
        if not save_dir:
            return
        if self._autosave_h5_path is None:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self._autosave_h5_path = str(Path(save_dir) / f'rpi_live_{ts}.h5')
        try:
            autosave_append(self._autosave_h5_path, entry)
        except Exception as exc:
            self._set_status(f'Auto-save error: {exc}')

    def _active_probe_entry(self):
        """The probe-library entry currently in use, or None."""
        return next((c for c in self._probe_choices
                     if c['path'] == self._probe_path), None)

    def _active_probe_geometry(self):
        """(shape, fill_fraction) of the probe currently in use."""
        entry = self._active_probe_entry()
        if entry is None:
            return DEFAULT_PROBE_SHAPE, DEFAULT_FILL_FRACTION
        return (entry.get('shape', DEFAULT_PROBE_SHAPE),
                float(entry.get('fill_fraction', DEFAULT_FILL_FRACTION)))

    def _illumination_mask_for(self, img, fudge=_PHASE_REGION_FUDGE):
        """Illuminated-area mask matching `img`, or None if it isn't square."""
        if img.ndim != 2 or img.shape[0] != img.shape[1]:
            return None
        shape, ff = self._active_probe_geometry()
        return illumination_mask(img.shape[0], shape, ff, fudge)

    def _process_phase(self, obj):
        """Apply the current Display settings (unwrap / ramp removal) to the
        phase of one complex object array and return the display phase.

        Both operations are confined to the illuminated area of the active
        probe — the dark corners of an RPI object are noise, and letting them
        drive the unwrap or the ramp fit corrupts the useful part.
        """
        phase     = np.angle(obj).astype(np.float32)
        do_unwrap = self.unwrap_check.isChecked()
        do_ramp   = self.ramp_check.isChecked()
        if not (do_unwrap or do_ramp):
            return phase
        mask = self._illumination_mask_for(phase)
        if mask is None:
            mask = np.ones(phase.shape, dtype=bool)
        return process_phase_region(phase, mask, unwrap=do_unwrap, ramp=do_ramp)

    def _display_array(self, entry):
        obj = entry['obj']
        if self.abs_check.isChecked():
            return np.abs(obj).astype(np.float32)
        return self._process_phase(obj)

    def _render_entry(self, entry):
        img = self._display_array(entry)
        self._last_pixel_size_m = entry.get('pixel_size_m')
        self._render_image(img)

    def _render_image(self, img, allow_zoom=True):
        """Push one already-processed array to the main view, honouring the
        log / auto-colorscale / zoom settings."""
        disp = img
        if self.log_check.isChecked() and self.abs_check.isChecked():
            disp = np.log10(np.maximum(img, _LOG_FLOOR))

        # Auto levels are computed here rather than via autoLevels=True:
        # the deque mosaic pads unfilled cells with NaN, which autoLevels
        # propagates into an all-NaN range. With auto off, levels stays None
        # and pg_image_tools holds the current levels and histogram view.
        levels = None
        if self.autolevel_action.isChecked() and np.isfinite(disp).any():
            levels = (float(np.nanmin(disp)), float(np.nanmax(disp)))

        # pg_image_tools works in col-major (image[x, y]) whatever the global
        # imageAxisOrder, so hand it data[row, col] transposed.
        self.image_plot.set_image(disp.T, pixel_size_m=self._last_pixel_size_m or 1.0)
        if levels is not None:
            self.image_plot.set_levels(*levels)

        self._last_display_img = img   # un-logged, so the readout stays honest
        self._update_pixel_info()
        if allow_zoom:
            self._apply_zoom()

    def _apply_zoom(self):
        """Keep the view on the illuminated area while that menu entry is ticked."""
        if not self.zoom_illum_action.isChecked():
            return
        if self.view_deque_btn.isChecked():
            return      # a single tile's illuminated area means nothing on the grid
        img = self._last_display_img
        if img is None or img.ndim != 2:
            return
        ny, nx = img.shape
        _, ff = self._active_probe_geometry()
        half = _ZOOM_FUDGE * ff * min(ny, nx) / 2.0
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        self._recon_vb.setRange(xRange=(cx - half, cx + half),
                                yRange=(cy - half, cy + half), padding=0)

    def _rerender_last(self):
        """Re-display the most recent deque entry when a display toggle changes."""
        if not self._results_deque:
            return
        if self.view_deque_btn.isChecked():
            self._render_mosaic()
        else:
            self._render_entry(self._results_deque[-1])

    # ------------------------------------------------------------------ #
    # Reconstruction-image cursor readout                                 #
    # ------------------------------------------------------------------ #

    def _on_recon_mouse_moved(self, event):
        pos = event[0]
        img = self._last_display_img
        if img is None:
            return
        if not self._recon_vb.sceneBoundingRect().contains(pos):
            self._mouse_rc = None
        else:
            pt = self._recon_vb.mapSceneToView(pos)
            x, y = int(pt.x()), int(pt.y())   # x = column, y = row (row-major)
            ny, nx = img.shape
            self._mouse_rc = (y, x) if (0 <= x < nx and 0 <= y < ny) else None
        self._update_pixel_info()

    def _update_pixel_info(self):
        parts = []
        if self._last_pixel_size_m is not None:
            parts.append(f'pixel size: {self._last_pixel_size_m * 1e9:.1f} nm')
        if self._mouse_rc is not None and self._last_display_img is not None:
            r, c = self._mouse_rc
            parts.append(f'x={c} px  y={r} px')
            label = '|obj|' if self.abs_check.isChecked() else 'phase'
            value = float(self._last_display_img[r, c])
            parts.append(f'{label}: {value:.4g}' if np.isfinite(value)
                         else f'{label}: —')
        self.pixel_info_label.setText('     '.join(parts))

    # ------------------------------------------------------------------ #
    # View deque — a grid mosaic drawn in the main view                   #
    # ------------------------------------------------------------------ #

    def _on_view_deque(self, checked):
        if checked:
            if not self._results_deque:
                QMessageBox.information(self, 'Empty deque',
                                        'Run at least one frame first.')
                self.view_deque_btn.setChecked(False)
                return
            self._render_mosaic()
            self._recon_vb.autoRange()
        else:
            self._clear_deque_labels()
            if self._results_deque:
                self._render_entry(self._results_deque[-1])
            self._recon_vb.autoRange()

    @staticmethod
    def _build_mosaic(imgs):
        """Tile frames into a square grid, oldest top-left to newest bottom-right.

        The grid is square — ceil(sqrt(n)) on a side — so a 3-frame deque gives
        2x2 with the bottom-right cell empty, and a 5-frame deque gives 3x3
        with four empty. Empty cells are NaN so they can be excluded from the
        colour range.
        """
        n     = len(imgs)
        ncols = nrows = int(np.ceil(np.sqrt(n)))
        h, w  = imgs[0].shape
        canvas = np.full((nrows * h, ncols * w), np.nan, dtype=np.float32)
        origins = []
        for i, im in enumerate(imgs):
            r, c = divmod(i, ncols)
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
            origins.append((r * h, c * w))
        return canvas, origins, (h, w)

    def _clear_deque_labels(self):
        for item in self._deque_label_items:
            self._recon_vb.removeItem(item)
        self._deque_label_items = []

    def _render_mosaic(self):
        imgs = [self._display_array(e) for e in self._results_deque]
        if not imgs:
            return
        if len({im.shape for im in imgs}) > 1:
            self._set_status('Deque frames have differing shapes — '
                             'showing the latest frame only.')
            self._clear_deque_labels()
            self._render_entry(self._results_deque[-1])
            return

        canvas, origins, (h, w) = self._build_mosaic(imgs)

        self._clear_deque_labels()
        pad = max(2, int(round(0.02 * min(h, w))))
        for i, (r0, c0) in enumerate(origins):
            # Black on white so the index stays readable over any colour map.
            label = pg.TextItem(str(i), color=(0, 0, 0),
                                fill=pg.mkBrush(255, 255, 255, 230), anchor=(0, 0))
            # anchor=(0,0) is the item's own top-left; with ImageView's inverted
            # y-axis that lands on the tile's visual top-left corner.
            label.setPos(c0 + pad, r0 + pad)
            self._recon_vb.addItem(label)
            self._deque_label_items.append(label)

        self._last_pixel_size_m = self._results_deque[-1].get('pixel_size_m')
        self._render_image(canvas, allow_zoom=False)

    # ------------------------------------------------------------------ #
    # Probe library                                                        #
    # ------------------------------------------------------------------ #

    def _on_change_probe(self):
        if self._worker and self._worker.isRunning():
            QMessageBox.information(
                self, 'Stop first',
                'Stop the reconstruction loop before changing the probe.'
            )
            return

        dlg = ProbeViewer(
            choices      = self._probe_choices,
            geometry     = self._geometry,
            current_path = self._probe_path,
            probe_cache  = self._probe_data_cache,
            browse_dir   = self.browse_dir('probe'),
            parent       = self,
        )
        apply_font_size_to_tree(dlg, self._current_font_size())
        dlg.exec()
        # Keep any add/delete/rename/change-file/shape/geometry edits,
        # regardless of accept/cancel, and reflect them in the Setup window.
        self._probe_choices = dlg.get_choices()
        self._geometry = dlg.get_geometry()
        self._refresh_params_probe_list()

        name, path, probe, res_m = dlg.get_result()
        if path is not None:
            self._probe_raw   = probe
            self._probe_path  = path
            self._res_m_probe = res_m
            self.probe_file_edit.setText(path)
            self._set_status(
                f'Probe set: “{name}” — shape={probe.shape}, '
                f'pixel size={res_m*1e9:.2f} nm'
            )
        self._refresh_probe_list_widget()

    def _refresh_probe_list_widget(self):
        """Repopulate the main-panel probe list: every enabled probe, with the
        active one in bold and not-yet-loaded ones greyed out."""
        lw = self.probe_list_widget
        lw.blockSignals(True)
        lw.clear()
        self._probe_list_paths = []
        for c in self._probe_choices:
            if not c.get('enabled', True):
                continue
            path   = c['path']
            loaded = path in self._probe_data_cache
            item = QListWidgetItem(c['name'] if loaded
                                   else f'{c["name"]}   (not loaded)')
            item.setToolTip(
                f'{path}\nillumination: {c.get("shape", DEFAULT_PROBE_SHAPE)} '
                f'@ {float(c.get("fill_fraction", DEFAULT_FILL_FRACTION)):.2f}'
            )
            if not loaded:
                item.setForeground(QBrush(QColor('gray')))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            if path == self._probe_path:
                # Built from the widget's current font so the point size keeps
                # tracking the font-size spin box (an item font, once set, no
                # longer inherits from the widget).
                f = lw.font()
                f.setBold(True)
                item.setFont(f)
            lw.addItem(item)
            self._probe_list_paths.append(path)
        if self._probe_path in self._probe_list_paths:
            lw.setCurrentRow(self._probe_list_paths.index(self._probe_path))
        lw.blockSignals(False)

    def _on_probe_list_clicked(self, item):
        row = self.probe_list_widget.row(item)
        if not (0 <= row < len(self._probe_list_paths)):
            return
        path = self._probe_list_paths[row]
        if path == self._probe_path:
            return
        data = self._probe_data_cache.get(path)
        if data is None:
            self._set_status(
                'That probe is not loaded — open "Probes & Geometry…" '
                'and press "Load registered probes".'
            )
            self._refresh_probe_list_widget()
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(
                self, 'Stop first',
                'Stop the reconstruction loop before switching probes.'
            )
            self._refresh_probe_list_widget()   # revert visible selection
            return
        entry = next((c for c in self._probe_choices if c['path'] == path), None)
        name = entry['name'] if entry else Path(path).stem
        self._probe_raw   = data['probe']
        self._probe_path  = path
        self._res_m_probe = data['res_m']
        self.probe_file_edit.setText(path)
        self._refresh_probe_list_widget()
        self._rerender_last()   # the illuminated area may have changed
        self._set_status(f'Switched probe: “{name}”')

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #

    def _on_save(self):
        if not self._results_deque:
            QMessageBox.information(self, 'Nothing to save',
                                    'Run at least one frame first.')
            return

        idx = self.save_entry_spin.value()
        # Deque index: 0 = newest = index -1; 1 = second newest = -2, etc.
        deque_list = list(self._results_deque)
        if idx >= len(deque_list):
            QMessageBox.warning(self, 'Invalid entry',
                                f'Only {len(deque_list)} entries in the deque.')
            return
        entry = deque_list[-(idx + 1)]          # copy reference; numpy arrays are immutable here

        save_dir = self.save_dir_edit.text().strip()
        if not save_dir:
            save_dir = QFileDialog.getExistingDirectory(self, 'Select save directory')
            if not save_dir:
                return
            self.save_dir_edit.setText(save_dir)

        params = self._collect_params()

        try:
            h5_path = save_rpi_result(
                entry       = entry,
                probe_interp= entry['probe_interp'],
                params      = params,
                save_dir    = save_dir,
            )
            self._set_status(f'Saved → {h5_path}')
        except Exception as exc:
            QMessageBox.critical(self, 'Save error', str(exc))

    # ------------------------------------------------------------------ #
    # Object Accumulator                                                   #
    # ------------------------------------------------------------------ #

    def _on_open_accumulator(self):
        if self._accum_win is not None:
            self._accum_win.show()
            self._accum_win.raise_()
            self._accum_win.activateWindow()
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(
                self, 'Stop first',
                'Stop the main reconstruction loop first — the Object '
                'Accumulator runs its own.'
            )
            return
        # Imported here, not at module scope, so none of the accumulator's
        # dependencies are loaded unless the feature is actually used.
        try:
            from object_accumulator import ObjectAccumulatorWindow
        except Exception:
            _DiagnosticDialog(
                'Could not import the Object Accumulator module.\n\n'
                + traceback.format_exc(), self).exec()
            return
        try:
            win = ObjectAccumulatorWindow(self)
        except Exception:
            _DiagnosticDialog(
                'Could not create the Object Accumulator window.\n\n'
                + traceback.format_exc(), self).exec()
            return
        self._accum_win = win
        apply_font_size_to_tree(win, self._current_font_size())
        self.set_locked(True)
        win.show()
        win.raise_()

    def on_accumulator_closed(self, settings=None):
        """Called by ObjectAccumulatorWindow.closeEvent.

        `settings` is kept so the accumulator's choices still reach
        settings.ini when that window is closed before the main one.
        """
        if settings:
            self._accum_settings = dict(settings)
        self._accum_win = None
        self.set_locked(False)

    def set_locked(self, locked):
        """Disable the main panel while the Object Accumulator is open.

        Left enabled: the RPI Parameters group, "Probes & Geometry…", the
        EPICS PV field, font size and EXIT.

        Note that paths_group and control_group are *not* disabled wholesale —
        a disabled QGroupBox disables its children regardless of their own
        state, and two of the three exceptions live inside those groups.
        """
        for name in _ACCUM_LOCK_GROUPS:
            getattr(self, name).setEnabled(not locked)
        for name in _ACCUM_LOCK_WIDGETS:
            getattr(self, name).setEnabled(not locked)
        # The PV field is one of the exceptions, but it is also driven by the
        # H5 toggle; re-apply that rule rather than leaving it unconditionally on.
        if not locked:
            self._on_h5_toggled(self.h5_check.isChecked())
        else:
            self.pv_name_edit.setEnabled(True)
            self.label_pv.setEnabled(True)

    # ------------------------------------------------------------------ #
    # Diffraction-pattern viewer / worker errors                           #
    # ------------------------------------------------------------------ #

    def _on_view_pv_image(self):
        if self._pv_viewer is None:
            self._pv_viewer = _PVImageViewer(self)
            apply_font_size_to_tree(self._pv_viewer, self._current_font_size())
        self._pv_viewer.show()
        self._pv_viewer.raise_()

    def _on_raw_image_ready(self, img):
        if self._pv_viewer is not None and self._pv_viewer.isVisible():
            self._pv_viewer.update_image(
                img,
                beam_cy = self._geometry['beam_center_y'],
                beam_cx = self._geometry['beam_center_x'],
                crop_N  = self._current_crop_N(),
                # A raw master frame is a full detector image, so the
                # overlay goes on the beam centre as for a PV image.
                h5_mode = (self.h5_check.isChecked()
                           and not getattr(self._worker, 'h5_raw', False)),
            )

    def _on_worker_error(self, msg):
        self._set_start_btn(running=False)
        self._set_method_controls_enabled(True)
        first_line = msg.split('\n')[0] if msg else 'Worker error'
        self._set_status(f'Error: {first_line}')
        _DiagnosticDialog(msg, self).exec()

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def _set_status(self, text):
        self.status_label.setText(text)


# ─────────────────────────────────────────────────────────────────────────── #

class _ConfigLoadDialog(QDialog):
    """Startup choice: keep the settings.ini configuration, or import a JSON.

    The ini's current contents are shown so that declining the import is an
    informed choice rather than a blind one.
    """

    USE_INI, IMPORT_JSON = 1, 2

    def __init__(self, probes, geometry, parent=None):
        super().__init__(parent)
        self.setObjectName('ConfigLoadDialog')
        self.setWindowTitle('Load configuration')
        self.setModal(True)
        self.resize(560, 420)
        self.choice = self.USE_INI

        layout = QVBoxLayout(self)
        layout.addWidget(_named(QLabel(
            'Start from the configuration stored in settings.ini, or import '
            'one from a JSON file?'), 'intro_label'))
        layout.itemAt(0).widget().setWordWrap(True)

        summary = _named(QTextEdit(), 'summary_text')
        summary.setReadOnly(True)
        summary.setFontFamily('monospace')
        enabled = [p for p in probes if p.get('enabled', True)]
        lines = [f'Probes in settings.ini: {len(probes)} '
                 f'({len(enabled)} enabled)', '']
        for p in probes:
            flag = ' ' if p.get('enabled', True) else '-'
            lines.append(f'  [{flag}] {p["name"]}   '
                         f'[{p.get("shape", DEFAULT_PROBE_SHAPE)} '
                         f'{float(p.get("fill_fraction", DEFAULT_FILL_FRACTION)):.2f}]')
            lines.append(f'        {p["path"]}')
        if not probes:
            lines.append('  (none registered)')
        lines += [
            '',
            f'Energy            : {geometry["energy_kev"]} keV',
            f'Detector distance : {geometry["detector_distance"]} m',
            f'Beam centre       : y={geometry["beam_center_y"]}  '
            f'x={geometry["beam_center_x"]}',
        ]
        summary.setPlainText('\n'.join(lines))
        layout.addWidget(summary, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        use_btn    = _named(QPushButton('Use settings.ini'), 'use_ini_btn')
        import_btn = _named(QPushButton('Import JSON…'), 'import_json_btn')
        use_btn.setDefault(True)
        use_btn.clicked.connect(self._on_use_ini)
        import_btn.clicked.connect(self._on_import)
        btn_row.addWidget(use_btn)
        btn_row.addWidget(import_btn)
        layout.addLayout(btn_row)

    def _on_use_ini(self):
        self.choice = self.USE_INI
        self.accept()

    def _on_import(self):
        self.choice = self.IMPORT_JSON
        self.accept()


class _PVImageViewer(QDialog):
    """Non-modal window showing the latest raw diffraction pattern."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('PVImageViewer')
        self.setWindowTitle('Diffraction Pattern')
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.resize(600, 650)

        self._last_img    = None
        self._beam_cy     = 0
        self._beam_cx     = 0
        self._crop_N      = 0
        self._h5_mode     = False
        self._first_image = True

        layout = QVBoxLayout(self)

        self._plot = ImagePlotWidget(keep_points=True)
        self._plot.action_no_zoom_tracking.setChecked(True)
        self._plot.action_auto_colorscale.setChecked(False)   # levels set below
        self._iv = _named(self._plot.image_view, 'pv_image_view')
        layout.addWidget(self._plot)

        # Bottom bar: log checkbox on the left, cursor position on the right
        bottom = QHBoxLayout()
        self._log_check = _named(QCheckBox('log₁₀'), 'pv_log_check')
        self._log_check.toggled.connect(self._rerender_reset_levels)
        bottom.addWidget(self._log_check)
        bottom.addStretch()
        self._pos_label = _named(QLabel(''), 'pv_pos_label')
        bottom.addWidget(self._pos_label)
        layout.addLayout(bottom)

        vb = self._iv.getView()

        # Overlay items added directly to the ViewBox
        self._dot  = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush('r'), size=8)
        self._rect = pg.PlotDataItem(pen=pg.mkPen('r', width=1))
        vb.addItem(self._dot)
        vb.addItem(self._rect)

        # Mouse tracking — keep reference so it isn't garbage-collected
        self._mouse_proxy = pg.SignalProxy(
            vb.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )
        self._vb = vb

    def update_image(self, img, beam_cy=0, beam_cx=0, crop_N=0, h5_mode=False):
        self._last_img = img
        self._beam_cy  = beam_cy
        self._beam_cx  = beam_cx
        self._crop_N   = crop_N
        self._h5_mode  = h5_mode
        self._do_render(reset_levels=False)

    def _rerender_reset_levels(self):
        if self._last_img is not None:
            self._do_render(reset_levels=True)

    def _do_render(self, reset_levels=False):
        img = self._last_img
        if img is None:
            return
        if self._log_check.isChecked():
            display = np.log10(np.maximum(img, 1.0))
        else:
            display = img
        self._plot.set_image(display.T, autoRange=True if self._first_image else None)
        if self._first_image or reset_levels:
            self._plot.auto_levels()
        self._first_image = False
        self._update_overlays(img.shape)

    def _update_overlays(self, img_shape):
        rows, cols = img_shape
        # In view coords: x = column index, y = row index
        if self._h5_mode:
            cx, cy = cols // 2, rows // 2
        else:
            cx, cy = self._beam_cx, self._beam_cy
        self._dot.setData([cx], [cy])
        if self._crop_N > 0:
            half = self._crop_N // 2
            x0, x1 = cx - half, cx + half
            y0, y1 = cy - half, cy + half
            self._rect.setData(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
            )

    def _on_mouse_moved(self, event):
        pos = event[0]
        if self._vb.sceneBoundingRect().contains(pos):
            pt = self._vb.mapSceneToView(pos)
            self._pos_label.setText(f'x={int(pt.x())}, y={int(pt.y())}')
        else:
            self._pos_label.setText('')


class _DiagnosticDialog(QDialog):
    """Resizable, scrollable dialog for worker error diagnostics."""

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setObjectName('DiagnosticDialog')
        self.setWindowTitle('Worker Error — Diagnostic')
        self.resize(820, 520)

        layout = QVBoxLayout(self)

        te = _named(QTextEdit(), 'diag_text')
        te.setReadOnly(True)
        te.setPlainText(text)
        te.setFontFamily('monospace')
        layout.addWidget(te)

        close_btn = _named(QPushButton('Close'), 'diag_close_btn')
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


def main():
    pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
