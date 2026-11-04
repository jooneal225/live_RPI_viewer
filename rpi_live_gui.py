"""
RPI Live Viewer — main entry point.

A single GUI that can run either RPI backend:
  * "Fast conjugate gradient RPI"  (CGWorker)
  * "Slow LBFGS RPI"               (LBFGSWorker)

The algorithm is chosen from a combo box in the main window; all RPI parameters
are edited in a separate non-blocking "Setup parameters" window. The currently
selected algorithm is used when Start is pressed, so stopping, switching the
combo, and restarting changes the backend.

A second combo picks between two stored object shapes — a coarse grid that
converges reliably and a fine one that does not — so the resolution/robustness
trade-off is one click rather than a trip into the parameters window.

"Propagate probe…" opens its own non-blocking window holding the probe
propagation distance (10 µm per scroll step). It can be turned while a
reconstruction runs: the raw probe is re-propagated and handed to the running
worker, which picks it up before its next frame.

The probe library, beamline energy/distance and detector beam centre are
edited together in "Setup Initial Probes and Geometry" and can be exported to
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
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"

# ── Set GPU before any torch/cdtools import ─────────────────────────────────
import os
import configparser as _cp

_pre_cfg = _cp.ConfigParser()
_pre_cfg.read(os.path.join(os.path.dirname(__file__), 'settings.ini'))
os.environ['CUDA_VISIBLE_DEVICES'] = _pre_cfg.get('rpi', 'gpu_index', fallback='0')
# ────────────────────────────────────────────────────────────────────────────

import sys
import configparser
import datetime
import traceback
from pathlib import Path
from collections import deque

import numpy as np

import pyqtgraph as pg
from PyQt5 import uic
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QCheckBox, QLabel,
    QComboBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QLineEdit,
    QListWidget, QListWidgetItem,
)
from PyQt5.QtCore import Qt, QByteArray, QTimer
from PyQt5.QtGui import QBrush, QColor

from font_utils import apply_font_size_to_tree, DEFAULT_FONT_SIZE
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
    DEFAULT_PROBE_SHAPE,
    DEFAULT_FILL_FRACTION,
)
from rpi_worker import WORKER_CLASSES
from probe_viewer import ProbeViewer, DEFAULT_GEOMETRY


# Path to the .ui file (same directory as this script)
_UI_FILE    = Path(__file__).parent / 'rpi_live_gui.ui'
_INI_FILE   = Path(__file__).parent / 'settings.ini'

# Combo-box entries → internal method strings (index-aligned).
_METHOD_LABELS = ['Fast conjugate gradient RPI', 'Slow LBFGS RPI']
_METHODS       = ['CG', 'LBFGS']

# Combo-box entries → internal object-initialisation strings (index-aligned).
# Shared by both algorithms: CG uses it directly, LBFGS maps 'flat' to
# cdtools' 'uniform' initialisation.
_INIT_LABELS = ['Random', 'Flat']
_INITS       = ['random', 'flat']

# Resolution mode → which of the two stored object shapes is used.
# res_rpi_m = res_m_probe * N_probe / obj_shape, so a bigger obj_shape means
# smaller pixels (higher resolution) on a harder-to-converge problem.
_RES_MODE_LABELS    = ['Low res, good convergence', 'High res, bad convergence']
_RES_MODES          = ['lowres', 'highres']
_DEFAULT_OBJ_SHAPES = {'lowres': 100, 'highres': 200}

# Fudge factors applied to a probe's filling fraction. Tune these here rather
# than exposing them in the UI.
_ZOOM_FUDGE         = 1.2   # "Zoom on illuminated area" box
_PHASE_REGION_FUDGE = 1.0   # unwrap / ramp-removal region

_LOG_FLOOR = 1e-12          # floor for the log colour scale

_EXIT_BTN_FONT_DELTA = 3    # EXIT is this many points larger than everything else

# RPI parameters that can be stored per-probe (gpu_index is excluded: it is a
# machine setting, not a probe property).
_PROBE_PARAM_KEYS = (
    'method', 'res_mode', 'obj_shape_lowres', 'obj_shape_highres', 'obj_init',
    'cg_n_iter', 'cg_clear_every', 'cg_loss_target', 'cg_probe_modes',
    'lbfgs_reg_factor1', 'lbfgs_reg_factor2_lo', 'lbfgs_reg_factor2_hi',
    'lbfgs_lr1', 'lbfgs_lr2', 'lbfgs_n_iter1', 'lbfgs_n_iter2', 'lbfgs_probe_modes',
)

# Main-window widgets switched off while the Object Accumulator is open. Left
# alone: the RPI Parameters group, "Setup Initial Probes and Geometry", the
# EPICS PV field, the font size and EXIT.
_ACCUM_LOCK_GROUPS = ('h5_group', 'detector_group', 'saving_group', 'display_group')
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
        self._pv_viewer: _PVImageViewer | None = None
        self._accum_win                = None   # ObjectAccumulatorWindow, lazily imported
        self._prop_win                 = None   # "Propagate Probe" window
        self._prop_status_label        = None
        self._prop_applied_um          = None   # distance last pushed to a worker
        self._accum_settings: dict     = {}     # survives the window being closed
        self._closing                  = False

        # Beamline + detector geometry, edited in the setup window rather than
        # on the main panel (Crop N stays on the main panel).
        self._geometry = dict(DEFAULT_GEOMETRY)

        # Reconstruction-image cursor readout state
        self._last_display_img  = None   # processed image currently shown
        self._last_pixel_size_m = None   # metres/pixel of the current frame
        self._mouse_rc          = None   # (row, col) under the cursor, or None

        # Display state
        self._frozen_levels     = None   # (lo, hi) while auto-colorscale is off
        self._deque_label_items = []     # pg.TextItems overlaying the deque mosaic

        # Build the RPI parameter widgets + the non-blocking parameters window,
        # and populate the combo boxes.
        self._build_rpi_param_widgets()
        self._build_params_window()
        self._build_propagation_window()
        self.algo_combo.addItems(_METHOD_LABELS)
        self.res_mode_combo.addItems(_RES_MODE_LABELS)

        self._connect_signals()
        self._load_settings()

        # Add Reset Zoom to reconstruction image view context menu
        _vb = self.image_view.getView()
        _vb.menu.addAction('Reset Zoom', _vb.autoRange)

        # Track the cursor over the reconstruction image for the readout label.
        self._recon_vb = _vb
        self._recon_mouse_proxy = pg.SignalProxy(
            _vb.scene().sigMouseMoved, rateLimit=60, slot=self._on_recon_mouse_moved
        )

        self._wire_font_size()

        # Offer to replace the settings.ini configuration with a JSON one
        # before anything is read from disk.
        self._prompt_import_config()

        # Probe selection happens through "Setup Initial Probes and Geometry" —
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

    # Per-probe RPI parameter set: widget getters/setters keyed by name.
    # gpu_index is intentionally excluded (a machine setting, not a probe one).
    def _get_current_rpi_params(self):
        return {
            'method':              self._current_method(),
            'res_mode':            self._current_res_mode(),
            'obj_shape_lowres':    self.obj_shape_lowres_spin.value(),
            'obj_shape_highres':   self.obj_shape_highres_spin.value(),
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
        }

    def _apply_rpi_params(self, p):
        """Load a saved per-probe parameter dict into the editable widgets."""
        def _i(key, default):
            try:    return int(p[key])
            except (KeyError, ValueError, TypeError): return default
        def _f(key, default):
            try:    return float(p[key])
            except (KeyError, ValueError, TypeError): return default
        method = p.get('method')
        if method in _METHODS:
            self.algo_combo.setCurrentIndex(_METHODS.index(method))
        res_mode = p.get('res_mode')
        if res_mode in _RES_MODES:
            self.res_mode_combo.setCurrentIndex(_RES_MODES.index(res_mode))
        self.obj_shape_lowres_spin.setValue(
            _i('obj_shape_lowres', self.obj_shape_lowres_spin.value()))
        self.obj_shape_highres_spin.setValue(
            _i('obj_shape_highres', self.obj_shape_highres_spin.value()))
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

    @staticmethod
    def _format_rpi_params(p):
        if not p:
            return 'No parameters saved for this probe.'
        return (
            f"method: {p.get('method', '?')}     res mode: {p.get('res_mode', '?')}     "
            f"obj init: {p.get('obj_init', '?')}\n"
            f"obj shape: low={p.get('obj_shape_lowres', '?')}  "
            f"high={p.get('obj_shape_highres', '?')}\n"
            f"[CG]    n_iter={p.get('cg_n_iter', '?')}  clear_every={p.get('cg_clear_every', '?')}  "
            f"loss_target={p.get('cg_loss_target', '?')}  probe_modes={p.get('cg_probe_modes', '?')}\n"
            f"[LBFGS] reg={p.get('lbfgs_reg_factor1', '?')}/{p.get('lbfgs_reg_factor2_lo', '?')}/"
            f"{p.get('lbfgs_reg_factor2_hi', '?')}  lr={p.get('lbfgs_lr1', '?')}/{p.get('lbfgs_lr2', '?')}  "
            f"n_iter={p.get('lbfgs_n_iter1', '?')}/{p.get('lbfgs_n_iter2', '?')}  "
            f"probe_modes={p.get('lbfgs_probe_modes', '?')}"
        )

    def _build_params_window(self):
        """Non-blocking window: editable params on top, one column per method,
        plus a probe list whose per-probe saved parameters can be viewed and
        copied to / saved from the editable fields."""
        win = QDialog(self)
        win.setObjectName('params_win')
        win.setWindowTitle('RPI Parameters')
        win.setWindowFlags(win.windowFlags() | Qt.Window)
        outer = QVBoxLayout(win)

        common_box = _named(QGroupBox('Common'), 'common_box')
        cf = QFormLayout(common_box)
        cf.addRow(_named(QLabel(f'Obj shape — {_RES_MODE_LABELS[0]}:'), 'label_obj_lowres'),
                  self.obj_shape_lowres_spin)
        cf.addRow(_named(QLabel(f'Obj shape — {_RES_MODE_LABELS[1]}:'), 'label_obj_highres'),
                  self.obj_shape_highres_spin)
        cf.addRow(_named(QLabel('Obj init:'), 'label_obj_init'), self.obj_init_combo)
        cf.addRow(_named(QLabel('GPU index:'), 'label_gpu'), self.gpu_index_spin)
        outer.addWidget(common_box)

        cols = QHBoxLayout()

        cg_box = _named(QGroupBox(_METHOD_LABELS[0]), 'cg_box')
        cgf = QFormLayout(cg_box)
        cgf.addRow(_named(QLabel('N iter:'), 'label_cg_niter'),      self.n_iter_spin)
        cgf.addRow(_named(QLabel('Clear every:'), 'label_cg_clear'), self.clear_every_spin)
        cgf.addRow(_named(QLabel('Loss target (0=off):'), 'label_cg_loss'), self.loss_target_edit)
        cgf.addRow(_named(QLabel('Probe modes:'), 'label_cg_modes'), self.cg_probe_modes_spin)
        cols.addWidget(cg_box)

        lb_box = _named(QGroupBox(_METHOD_LABELS[1]), 'lb_box')
        lbf = QFormLayout(lb_box)
        lbf.addRow(_named(QLabel('Reg factor 1:'), 'label_lb_reg1'),    self.reg1_edit)
        lbf.addRow(_named(QLabel('Reg factor 2 lo:'), 'label_lb_reg2lo'), self.reg2_lo_edit)
        lbf.addRow(_named(QLabel('Reg factor 2 hi:'), 'label_lb_reg2hi'), self.reg2_hi_edit)
        lbf.addRow(_named(QLabel('LR 1:'), 'label_lb_lr1'),            self.lr1_spin)
        lbf.addRow(_named(QLabel('LR 2:'), 'label_lb_lr2'),            self.lr2_spin)
        lbf.addRow(_named(QLabel('N iter 1:'), 'label_lb_n1'),         self.n_iter1_spin)
        lbf.addRow(_named(QLabel('N iter 2:'), 'label_lb_n2'),         self.n_iter2_spin)
        lbf.addRow(_named(QLabel('Probe modes:'), 'label_lb_modes'),   self.lbfgs_probe_modes_spin)
        cols.addWidget(lb_box)

        outer.addLayout(cols)

        # ---- Per-probe parameter presets ----
        preset_box = _named(QGroupBox('Per-probe parameters'), 'preset_box')
        pv = QVBoxLayout(preset_box)

        self._params_probe_list = _named(QListWidget(), 'params_probe_list')
        self._params_probe_list.setToolTip(
            'Probes from "Setup Initial Probes and Geometry". '
            'Each can store its own RPI parameters.'
        )
        self._params_probe_list.currentRowChanged.connect(self._on_params_probe_selected)
        pv.addWidget(self._params_probe_list)

        btn_row = QHBoxLayout()
        self._save_to_probe_btn   = _named(QPushButton('Save RPI parameters to probe'),
                                           'save_to_probe_btn')
        self._copy_from_probe_btn = _named(QPushButton('Copy RPI parameters from probe'),
                                           'copy_from_probe_btn')
        self._save_to_probe_btn.clicked.connect(self._on_save_params_to_probe)
        self._copy_from_probe_btn.clicked.connect(self._on_copy_params_from_probe)
        btn_row.addWidget(self._save_to_probe_btn)
        btn_row.addWidget(self._copy_from_probe_btn)
        pv.addLayout(btn_row)

        saved_box = _named(QGroupBox('Saved parameters for selected probe (read-only)'),
                           'saved_box')
        sv = QVBoxLayout(saved_box)
        self._saved_params_label = _named(QLabel('No probe selected.'), 'saved_params_label')
        self._saved_params_label.setWordWrap(True)
        self._saved_params_label.setStyleSheet('font-family: monospace;')
        self._saved_params_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sv.addWidget(self._saved_params_label)
        pv.addWidget(saved_box)

        outer.addWidget(preset_box)

        close_btn = _named(QPushButton('Close'), 'params_close_btn')
        close_btn.clicked.connect(win.hide)
        outer.addWidget(close_btn)

        self._params_win = win

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
        win.setWindowFlags(win.windowFlags() | Qt.Window)
        outer = QVBoxLayout(win)

        form = QFormLayout()
        form.addRow(_named(QLabel('Distance (µm):'), 'label_prop'),
                    self.probe_prop_um_spin)
        outer.addLayout(form)

        self.prop_live_check = _named(
            QCheckBox('Apply to the running reconstruction as I change it'),
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

    def _current_obj_init(self):
        return _INITS[self.obj_init_combo.currentIndex()]

    def _current_res_mode(self):
        return _RES_MODES[max(0, self.res_mode_combo.currentIndex())]

    def _current_obj_shape(self):
        """Object grid size implied by the resolution combo."""
        if self._current_res_mode() == 'highres':
            return self.obj_shape_highres_spin.value()
        return self.obj_shape_lowres_spin.value()

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
        self._params_win.show()
        self._params_win.raise_()
        self._params_win.activateWindow()

    # ------------------------------------------------------------------ #
    # Signal wiring                                                        #
    # ------------------------------------------------------------------ #

    def _connect_signals(self):
        # Browse buttons
        self.probe_browse_btn.clicked.connect(
            lambda: self._browse_file(self.probe_file_edit, 'HDF5 (*.h5 *.hdf5)'))
        self.pixel_mask_browse_btn.clicked.connect(
            lambda: self._browse_file(self.pixel_mask_file_edit, 'HDF5 (*.h5 *.hdf5)'))
        self.save_dir_browse_btn.clicked.connect(self._browse_save_dir)
        self.h5_browse_btn.clicked.connect(
            lambda: self._browse_file(self.h5_file_edit, 'HDF5 (*.h5 *.hdf5)'))

        # H5 replay toggle
        self.h5_check.toggled.connect(self._on_h5_toggled)

        # Allow-repeat-frames toggle (live while a run is in progress)
        self.allow_repeat_check.toggled.connect(self._on_allow_repeat_toggled)

        # Reload probe/mask when paths change
        self.probe_file_edit.editingFinished.connect(self._try_load_probe)
        self.pixel_mask_file_edit.editingFinished.connect(self._try_load_mask)

        # RPI parameters window
        self.setup_params_btn.clicked.connect(self._on_setup_params)

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
        self.autolevel_check.toggled.connect(self._on_autolevel_toggled)
        self.zoom_illum_check.toggled.connect(self._on_zoom_illum_toggled)

    # ------------------------------------------------------------------ #
    # Settings persistence                                                 #
    # ------------------------------------------------------------------ #

    def _load_settings(self):
        cfg = configparser.ConfigParser()
        cfg.read(str(_INI_FILE))

        def _get(section, key, fallback):
            return cfg.get(section, key, fallback=fallback)

        self.probe_file_edit.setText(        _get('paths',    'probe_file',       ''))
        self.pixel_mask_file_edit.setText(   _get('paths',    'pixel_mask_file',  ''))
        self.pv_name_edit.setText(           _get('epics',    'pv_name',          ''))
        self.save_dir_edit.setText(          _get('paths',    'save_dir',         ''))
        self.h5_check.setChecked(            _get('h5_replay', 'enabled',          'false').lower() == 'true')
        self.h5_file_edit.setText(           _get('h5_replay', 'h5_file',          ''))
        self.h5_start_spin.setValue(    int(  _get('h5_replay', 'h5_start_idx',     '0')))
        self.h5_stride_spin.setValue(   int(  _get('h5_replay', 'h5_stride',        '1')))
        self.crop_N_spin.setValue(      int(  _get('detector','crop_N',           '512')))
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
        res_mode = _get('rpi', 'res_mode', 'lowres')
        self.res_mode_combo.setCurrentIndex(
            _RES_MODES.index(res_mode) if res_mode in _RES_MODES else 0)
        self.gpu_index_spin.setValue(   int(  _get('rpi', 'gpu_index',    '3')))
        self.probe_prop_um_spin.setValue(float(_get('rpi', 'probe_propagation_um', '0')))
        obj_init = _get('rpi', 'obj_init', 'flat')
        self.obj_init_combo.setCurrentIndex(
            _INITS.index(obj_init) if obj_init in _INITS else _INITS.index('flat')
        )
        method = _get('rpi', 'method', 'CG')
        self.algo_combo.setCurrentIndex(_METHODS.index(method) if method in _METHODS else 0)
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

        # ---- Display ----
        self.unwrap_check.setChecked(         _get('display', 'unwrap',     'false').lower() == 'true')
        self.ramp_check.setChecked(           _get('display', 'ramp',       'false').lower() == 'true')
        self.abs_check.setChecked(            _get('display', 'abs',        'false').lower() == 'true')
        self.zoom_illum_check.setChecked(     _get('display', 'zoom_illum', 'false').lower() == 'true')
        self.autolevel_check.setChecked(      _get('display', 'autolevel',  'true').lower() == 'true')
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
        spl = _get('window', 'splitter', '')
        if spl:
            self.splitter.restoreState(QByteArray.fromHex(spl.encode()))

        # Line edits: keep the tooltip in sync with the (possibly truncated)
        # text so hovering always reveals the full value. Restricted to our own
        # fields (pyqtgraph's ImageView has many internal QLineEdits).
        for le in (self.probe_file_edit,
                   self.pixel_mask_file_edit, self.pv_name_edit,
                   self.save_dir_edit, self.h5_file_edit,
                   self.loss_target_edit, self.reg1_edit, self.reg2_lo_edit,
                   self.reg2_hi_edit):
            le.setToolTip(le.text())
            le.textChanged.connect(lambda text, w=le: w.setToolTip(text))

    def _save_settings(self):
        cfg = configparser.ConfigParser()
        cfg['paths']    = {
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
            'crop_N':        str(self.crop_N_spin.value()),
        }
        cfg['rpi']      = {
            'obj_shape_lowres':    str(self.obj_shape_lowres_spin.value()),
            'obj_shape_highres':   str(self.obj_shape_highres_spin.value()),
            'res_mode':            self._current_res_mode(),
            'obj_init':            self._current_obj_init(),
            'gpu_index':           str(self.gpu_index_spin.value()),
            'method':              self._current_method(),
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
        }
        cfg['display']  = {
            'unwrap':     str(self.unwrap_check.isChecked()).lower(),
            'ramp':       str(self.ramp_check.isChecked()).lower(),
            'abs':        str(self.abs_check.isChecked()).lower(),
            'zoom_illum': str(self.zoom_illum_check.isChecked()).lower(),
            'autolevel':  str(self.autolevel_check.isChecked()).lower(),
            'log':        str(self.log_check.isChecked()).lower(),
        }
        cfg['app']      = {
            'deque_length': str(self.deque_length_spin.value()),
            'autosave':     str(self.autosave_check.isChecked()).lower(),
            'font_size':    str(self.spinBox_fontSize.value()),
        }
        cfg['window']   = {
            'geometry': bytes(self.saveGeometry().toHex()).decode(),
            'splitter': bytes(self.splitter.saveState().toHex()).decode(),
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
        QTimer.singleShot(0, lambda: self._on_font_size_changed(size))

    def _current_font_size(self):
        return self.spinBox_fontSize.value()

    def _on_font_size_changed(self, size):
        for w in (self, self._params_win, self._pv_viewer, self._accum_win):
            if w is not None:
                apply_font_size_to_tree(w, size)
        # EXIT is deliberately oversized; this must run *after* the tree pass,
        # which would otherwise flatten it back to `size`.
        f = self.exit_btn.font()
        f.setPointSize(size + _EXIT_BTN_FONT_DELTA)
        f.setBold(True)
        self.exit_btn.setFont(f)
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
        dlg.exec_()
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
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
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

    # ------------------------------------------------------------------ #
    # Browse helpers                                                       #
    # ------------------------------------------------------------------ #

    def _browse_file(self, line_edit, file_filter):
        # Instantiate directly so setDirectory() is honoured — the static
        # helpers on some Linux/Qt builds ignore the directory argument even
        # with DontUseNativeDialog.
        dlg = QFileDialog(self)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter(file_filter)
        dlg.setDirectory(self._start_dir_from_edit(line_edit))
        if dlg.exec_():
            path = dlg.selectedFiles()[0]
            line_edit.setText(path)
            line_edit.editingFinished.emit()

    def _browse_save_dir(self):
        dlg = QFileDialog(self)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        dlg.setDirectory(self._start_dir_from_edit(self.save_dir_edit))
        if dlg.exec_():
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
        if checked:
            self._frozen_levels = None
        else:
            # Freeze at whatever is on screen right now.
            try:
                lo, hi = self.image_view.getLevels()
                self._frozen_levels = (float(lo), float(hi))
            except Exception:
                self._frozen_levels = None
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
            self.start_btn.setText('Start')
            self.algo_combo.setEnabled(True)
            self.res_mode_combo.setEnabled(True)
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
        self.start_btn.setText('Stop')
        self.algo_combo.setEnabled(False)   # algorithm is fixed while running
        self.res_mode_combo.setEnabled(False)
        self._set_status(f'Starting… ({_METHOD_LABELS[self.algo_combo.currentIndex()]}, '
                         f'{_RES_MODE_LABELS[self.res_mode_combo.currentIndex()]})')
        self._update_prop_status()

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
        self.start_btn.setText('Start')
        self.algo_combo.setEnabled(True)
        self.res_mode_combo.setEnabled(True)
        self._prop_applied_um = None
        self._update_prop_status()

    def _collect_params(self):
        """Read all relevant GUI fields into a plain dict for the worker.

        Returns a superset containing both algorithms' parameters plus the
        selected 'method'; each backend uses only the keys it needs.
        """
        def _safe_float(text, default):
            try:
                return float(text.strip())
            except (ValueError, AttributeError):
                return default

        method = self._current_method()
        # Probe modes is now per-algorithm; pick the active algorithm's value.
        n_probe_modes = (self.cg_probe_modes_spin.value() if method == 'CG'
                         else self.lbfgs_probe_modes_spin.value())
        probe_entry = self._active_probe_entry()
        probe_shape, fill_fraction = self._active_probe_geometry()

        return {
            'method':           method,
            'pv_name':          self.pv_name_edit.text().strip(),
            'h5_mode':          self.h5_check.isChecked(),
            'h5_file':          self.h5_file_edit.text().strip(),
            'h5_start_idx':     self.h5_start_spin.value(),
            'h5_stride':        self.h5_stride_spin.value(),
            'beam_cy':          self._geometry['beam_center_y'],
            'beam_cx':          self._geometry['beam_center_x'],
            'crop_N':           self.crop_N_spin.value(),
            'res_mode':         self._current_res_mode(),
            'obj_shape':        self._current_obj_shape(),
            'obj_init':         self._current_obj_init(),
            'gpu_index':        self.gpu_index_spin.value(),
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

    def snapshot_run_inputs(self):
        """Validate and gather everything a worker needs for one run.

        Shared by Start and the Object Accumulator so both enforce the same
        preconditions. Returns (params, probe, pixel_mask_full); raises
        RuntimeError with a message meant to be shown to the user.
        """
        if self._probe_raw is None:
            raise RuntimeError(
                'No probe is loaded. Open "Setup Initial Probes and Geometry", '
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

        # Levels are always computed here rather than via autoLevels=True:
        # the deque mosaic pads unfilled cells with NaN, which autoLevels
        # propagates into an all-NaN range.
        levels = None
        if self.autolevel_check.isChecked():
            if np.isfinite(disp).any():
                levels = (float(np.nanmin(disp)), float(np.nanmax(disp)))
        else:
            levels = self._frozen_levels

        # With pg.setConfigOptions(imageAxisOrder='row-major'), data[row, col]
        # maps to (y, x) correctly — no transpose needed.
        if levels is None:
            self.image_view.setImage(disp, autoLevels=True, autoRange=False)
        else:
            self.image_view.setImage(disp, autoLevels=False, levels=levels,
                                     autoRange=False)

        self._last_display_img = img   # un-logged, so the readout stays honest
        self._update_pixel_info()
        if allow_zoom:
            self._apply_zoom()

    def _apply_zoom(self):
        """Keep the view on the illuminated area while that box is ticked."""
        if not self.zoom_illum_check.isChecked():
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
            parent       = self,
        )
        apply_font_size_to_tree(dlg, self._current_font_size())
        dlg.exec_()
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
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
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
                'That probe is not loaded — open "Setup Initial Probes and '
                'Geometry" and press "Load registered probes".'
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
                + traceback.format_exc(), self).exec_()
            return
        try:
            win = ObjectAccumulatorWindow(self)
        except Exception:
            _DiagnosticDialog(
                'Could not create the Object Accumulator window.\n\n'
                + traceback.format_exc(), self).exec_()
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

        Left enabled: the RPI Parameters group, "Setup Initial Probes and
        Geometry", the EPICS PV field, font size and EXIT.

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
                crop_N  = self.crop_N_spin.value(),
                h5_mode = self.h5_check.isChecked(),
            )

    def _on_worker_error(self, msg):
        self.start_btn.setText('Start')
        self.algo_combo.setEnabled(True)
        self.res_mode_combo.setEnabled(True)
        first_line = msg.split('\n')[0] if msg else 'Worker error'
        self._set_status(f'Error: {first_line}')
        _DiagnosticDialog(msg, self).exec_()

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
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(600, 650)

        self._last_img    = None
        self._beam_cy     = 0
        self._beam_cx     = 0
        self._crop_N      = 0
        self._h5_mode     = False
        self._first_image = True

        layout = QVBoxLayout(self)

        self._iv = _named(pg.ImageView(), 'pv_image_view')
        layout.addWidget(self._iv)

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
        vb.menu.addAction('Reset Zoom', vb.autoRange)

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
        auto = self._first_image or reset_levels
        self._iv.setImage(display, autoLevels=auto, autoRange=self._first_image)
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
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
