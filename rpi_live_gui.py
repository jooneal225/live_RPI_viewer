"""
RPI Live Viewer — main entry point.

A single GUI that can run either RPI backend:
  * "Fast conjugate gradient RPI"  (CGWorker)
  * "Slow LBFGS RPI"               (LBFGSWorker)

The algorithm is chosen from a combo box in the main window; all RPI parameters
are edited in a separate non-blocking "Setup parameters" window. The currently
selected algorithm is used when Start is pressed, so stopping, switching the
combo, and restarting changes the backend.

Usage:
    python rpi_live_gui.py

CUDA_VISIBLE_DEVICES is set from settings.ini *before* any torch/cdtools
import so that the correct GPU is exposed as device 0.
"""
from __future__ import annotations  # X | Y union syntax back to Python 3.7

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
from pathlib import Path
from collections import deque

import numpy as np
from skimage.restoration import unwrap_phase

import pyqtgraph as pg
from PyQt5 import uic
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QCheckBox, QLabel,
    QComboBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QLineEdit,
    QListWidget, QListWidgetItem,
)
from PyQt5.QtCore import Qt, QByteArray

from rpi_utils import (
    load_probe_file,
    load_pixel_mask,
    crop_mask,
    save_rpi_result,
    remove_phase_ramp,
    autosave_append,
)
from rpi_worker import WORKER_CLASSES
from probe_viewer import ProbeViewer


# Path to the .ui file (same directory as this script)
_UI_FILE    = Path(__file__).parent / 'rpi_live_gui.ui'
_INI_FILE   = Path(__file__).parent / 'settings.ini'

# Combo-box entries → internal method strings (index-aligned).
_METHOD_LABELS = ['Fast conjugate gradient RPI', 'Slow LBFGS RPI']
_METHODS       = ['CG', 'LBFGS']

# RPI parameters that can be stored per-probe (gpu_index is excluded: it is a
# machine setting, not a probe property).
_PROBE_PARAM_KEYS = (
    'method', 'obj_shape',
    'cg_n_iter', 'cg_clear_every', 'cg_loss_target', 'cg_probe_modes',
    'lbfgs_reg_factor1', 'lbfgs_reg_factor2_lo', 'lbfgs_reg_factor2_hi',
    'lbfgs_lr1', 'lbfgs_lr2', 'lbfgs_n_iter1', 'lbfgs_n_iter2', 'lbfgs_probe_modes',
)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi(str(_UI_FILE), self)

        # Runtime state
        self._worker                   = None
        self._results_deque: deque     = deque(maxlen=3)
        self._probe_raw                = None   # (N_opr, N_modes, N, N) complex
        self._probe_path               = None
        self._probe_choices: list      = []     # [(name, path), …] probe library
        self._pixel_mask_full          = None   # 2-D float32
        self._res_m_probe              = None   # float
        self._autosave_h5_path: str | None = None  # set at first frame of each run
        self._pv_viewer: _PVImageViewer | None = None
        self._deque_viewer: _DequeViewer | None = None

        # Reconstruction-image cursor readout state
        self._last_display_img  = None   # processed phase currently shown
        self._last_pixel_size_m = None   # metres/pixel of the current frame
        self._mouse_rc          = None   # (row, col) under the cursor, or None

        # Build the RPI parameter widgets + the non-blocking parameters window,
        # and populate the algorithm combo box.
        self._build_rpi_param_widgets()
        self._build_params_window()
        self.algo_combo.addItems(_METHOD_LABELS)

        # Probe name is set only via the "Set Probe Choice" dialog.
        self.probe_name_edit.setReadOnly(True)

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

        # Auto-load files specified in settings
        self._try_load_probe()
        self._try_load_mask()

    # ------------------------------------------------------------------ #
    # RPI parameter widgets + Setup-parameters window                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _spin(minimum, maximum, value, special=None):
        s = QSpinBox()
        s.setRange(minimum, maximum)
        s.setValue(value)
        if special is not None:
            s.setSpecialValueText(special)
        return s

    @staticmethod
    def _dspin(minimum, maximum, value, decimals, step):
        s = QDoubleSpinBox()
        s.setDecimals(decimals)
        s.setRange(minimum, maximum)
        s.setSingleStep(step)
        s.setValue(value)
        return s

    def _build_rpi_param_widgets(self):
        """Create every RPI parameter widget (defaults; overwritten on load)."""
        # Common parameters
        self.obj_shape_spin   = self._spin(32, 2048, 200)
        self.gpu_index_spin   = self._spin(0, 7, 3)
        self.gpu_index_spin.setToolTip('Applies at next launch (GPU is selected '
                                       'before torch is imported).')
        # Fast conjugate-gradient parameters
        self.n_iter_spin        = self._spin(1, 9999, 100)
        self.clear_every_spin   = self._spin(1, 9999, 10)
        self.loss_target_edit   = QLineEdit('0')
        self.cg_probe_modes_spin = self._spin(0, 32, 0, special='All')
        # Slow LBFGS parameters
        self.reg1_edit          = QLineEdit('1e-5')
        self.reg2_lo_edit       = QLineEdit('1e-8')
        self.reg2_hi_edit       = QLineEdit('1e-6')
        self.lr1_spin           = self._dspin(0.001, 2.0, 0.2, 3, 0.01)
        self.lr2_spin           = self._dspin(0.001, 2.0, 0.1, 3, 0.01)
        self.n_iter1_spin       = self._spin(0, 1000, 20)
        self.n_iter2_spin       = self._spin(0, 1000, 60)
        self.lbfgs_probe_modes_spin = self._spin(0, 32, 0, special='All')

    # Per-probe RPI parameter set: widget getters/setters keyed by name.
    # gpu_index is intentionally excluded (a machine setting, not a probe one).
    def _get_current_rpi_params(self):
        return {
            'method':              self._current_method(),
            'obj_shape':           self.obj_shape_spin.value(),
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
        self.obj_shape_spin.setValue(_i('obj_shape', self.obj_shape_spin.value()))
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
            f"method: {p.get('method', '?')}     obj shape: {p.get('obj_shape', '?')}\n"
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
        win.setWindowTitle('RPI Parameters')
        win.setWindowFlags(win.windowFlags() | Qt.Window)
        outer = QVBoxLayout(win)

        common_box = QGroupBox('Common')
        cf = QFormLayout(common_box)
        cf.addRow('Obj shape:', self.obj_shape_spin)
        cf.addRow('GPU index:', self.gpu_index_spin)
        outer.addWidget(common_box)

        cols = QHBoxLayout()

        cg_box = QGroupBox(_METHOD_LABELS[0])
        cgf = QFormLayout(cg_box)
        cgf.addRow('N iter:',              self.n_iter_spin)
        cgf.addRow('Clear every:',         self.clear_every_spin)
        cgf.addRow('Loss target (0=off):', self.loss_target_edit)
        cgf.addRow('Probe modes:',         self.cg_probe_modes_spin)
        cols.addWidget(cg_box)

        lb_box = QGroupBox(_METHOD_LABELS[1])
        lbf = QFormLayout(lb_box)
        lbf.addRow('Reg factor 1:',    self.reg1_edit)
        lbf.addRow('Reg factor 2 lo:', self.reg2_lo_edit)
        lbf.addRow('Reg factor 2 hi:', self.reg2_hi_edit)
        lbf.addRow('LR 1:',            self.lr1_spin)
        lbf.addRow('LR 2:',            self.lr2_spin)
        lbf.addRow('N iter 1:',        self.n_iter1_spin)
        lbf.addRow('N iter 2:',        self.n_iter2_spin)
        lbf.addRow('Probe modes:',     self.lbfgs_probe_modes_spin)
        cols.addWidget(lb_box)

        outer.addLayout(cols)

        # ---- Per-probe parameter presets ----
        preset_box = QGroupBox('Per-probe parameters')
        pv = QVBoxLayout(preset_box)

        self._params_probe_list = QListWidget()
        self._params_probe_list.setToolTip(
            'Probes from "Set Probe Choice". Each can store its own RPI parameters.'
        )
        self._params_probe_list.currentRowChanged.connect(self._on_params_probe_selected)
        pv.addWidget(self._params_probe_list)

        btn_row = QHBoxLayout()
        self._save_to_probe_btn   = QPushButton('Save RPI parameters to probe')
        self._copy_from_probe_btn = QPushButton('Copy RPI parameters from probe')
        self._save_to_probe_btn.clicked.connect(self._on_save_params_to_probe)
        self._copy_from_probe_btn.clicked.connect(self._on_copy_params_from_probe)
        btn_row.addWidget(self._save_to_probe_btn)
        btn_row.addWidget(self._copy_from_probe_btn)
        pv.addLayout(btn_row)

        saved_box = QGroupBox('Saved parameters for selected probe (read-only)')
        sv = QVBoxLayout(saved_box)
        self._saved_params_label = QLabel('No probe selected.')
        self._saved_params_label.setWordWrap(True)
        self._saved_params_label.setStyleSheet('font-family: monospace;')
        self._saved_params_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sv.addWidget(self._saved_params_label)
        pv.addWidget(saved_box)

        outer.addWidget(preset_box)

        close_btn = QPushButton('Close')
        close_btn.clicked.connect(win.hide)
        outer.addWidget(close_btn)

        self._params_win = win

    def _current_method(self):
        return _METHODS[self.algo_combo.currentIndex()]

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

        # Reload probe/mask when paths change
        self.probe_file_edit.editingFinished.connect(self._try_load_probe)
        self.pixel_mask_file_edit.editingFinished.connect(self._try_load_mask)

        # RPI parameters window
        self.setup_params_btn.clicked.connect(self._on_setup_params)

        # Main controls
        self.view_pv_btn.clicked.connect(self._on_view_pv_image)
        self.view_deque_btn.clicked.connect(self._on_view_deque)
        self.start_btn.clicked.connect(self._on_start_stop)
        self.change_probe_btn.clicked.connect(self._on_change_probe)
        self.save_btn.clicked.connect(self._on_save)

        # Unwrap / ramp toggles rerender last frame
        self.unwrap_check.toggled.connect(self._rerender_last)
        self.ramp_check.toggled.connect(self._on_ramp_toggled)
        self.ramp_check.toggled.connect(self._rerender_last)
        self.ramp_radius_spin.valueChanged.connect(self._rerender_last)

    # ------------------------------------------------------------------ #
    # Settings persistence                                                 #
    # ------------------------------------------------------------------ #

    def _load_settings(self):
        cfg = configparser.ConfigParser()
        cfg.read(str(_INI_FILE))

        def _get(section, key, fallback):
            return cfg.get(section, key, fallback=fallback)

        self.probe_file_edit.setText(        _get('paths',    'probe_file',       ''))
        self.probe_name_edit.setText(        _get('paths',    'probe_name',       ''))
        self.pixel_mask_file_edit.setText(   _get('paths',    'pixel_mask_file',  ''))
        self.pv_name_edit.setText(           _get('epics',    'pv_name',          ''))
        self.save_dir_edit.setText(          _get('paths',    'save_dir',         ''))
        self.h5_check.setChecked(            _get('h5_replay', 'enabled',          'false').lower() == 'true')
        self.h5_file_edit.setText(           _get('h5_replay', 'h5_file',          ''))
        self.h5_start_spin.setValue(    int(  _get('h5_replay', 'h5_start_idx',     '0')))
        self.energy_kev_spin.setValue(  float(_get('beamline','energy_kev',       '12.0')))
        self.detector_dist_spin.setValue(float(_get('beamline','detector_distance','2.0')))
        self.beam_cy_spin.setValue(     int(  _get('detector','beam_center_y',    '512')))
        self.beam_cx_spin.setValue(     int(  _get('detector','beam_center_x',    '512')))
        self.crop_N_spin.setValue(      int(  _get('detector','crop_N',           '512')))

        # ---- RPI: common ----
        self.obj_shape_spin.setValue(   int(  _get('rpi', 'obj_shape',    '200')))
        self.gpu_index_spin.setValue(   int(  _get('rpi', 'gpu_index',    '3')))
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
        self.unwrap_check.setChecked(         _get('display', 'unwrap',      'false').lower() == 'true')
        self.ramp_check.setChecked(           _get('display', 'ramp',        'false').lower() == 'true')
        self.ramp_radius_spin.setValue( int(  _get('display', 'ramp_radius', '0')))
        self._on_ramp_toggled(self.ramp_check.isChecked())

        # ---- App ----
        self.deque_length_spin.setValue(int(  _get('app', 'deque_length', '3')))
        self.autosave_check.setChecked(       _get('app', 'autosave',     'false').lower() == 'true')

        # ---- Probe-choice library (each entry: name, path, per-probe params) ----
        self._probe_choices = []
        n_choices = int(_get('probe_choices', 'count', '0'))
        for i in range(n_choices):
            nm  = _get('probe_choices', f'name{i}', '')
            pth = _get('probe_choices', f'path{i}', '')
            if not pth:
                continue
            params = {}
            sec = f'probe_params_{i}'
            if cfg.has_section(sec):
                for k in _PROBE_PARAM_KEYS:
                    if cfg.has_option(sec, k):
                        params[k] = cfg.get(sec, k)
            self._probe_choices.append({'name': nm, 'path': pth, 'params': params})

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
        for le in (self.probe_file_edit, self.probe_name_edit,
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
            'probe_name':       self.probe_name_edit.text(),
            'pixel_mask_file':  self.pixel_mask_file_edit.text(),
            'save_dir':         self.save_dir_edit.text(),
        }
        cfg['epics']    = {'pv_name': self.pv_name_edit.text()}
        cfg['h5_replay'] = {
            'enabled':      str(self.h5_check.isChecked()).lower(),
            'h5_file':      self.h5_file_edit.text(),
            'h5_start_idx': str(self.h5_start_spin.value()),
        }
        cfg['beamline'] = {
            'energy_kev':        str(self.energy_kev_spin.value()),
            'detector_distance': str(self.detector_dist_spin.value()),
        }
        cfg['detector'] = {
            'beam_center_y': str(self.beam_cy_spin.value()),
            'beam_center_x': str(self.beam_cx_spin.value()),
            'crop_N':        str(self.crop_N_spin.value()),
        }
        cfg['rpi']      = {
            'obj_shape':    str(self.obj_shape_spin.value()),
            'gpu_index':    str(self.gpu_index_spin.value()),
            'method':       self._current_method(),
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
            'unwrap':      str(self.unwrap_check.isChecked()).lower(),
            'ramp':        str(self.ramp_check.isChecked()).lower(),
            'ramp_radius': str(self.ramp_radius_spin.value()),
        }
        cfg['app']      = {
            'deque_length': str(self.deque_length_spin.value()),
            'autosave':     str(self.autosave_check.isChecked()).lower(),
        }
        cfg['window']   = {
            'geometry': bytes(self.saveGeometry().toHex()).decode(),
            'splitter': bytes(self.splitter.saveState().toHex()).decode(),
        }
        probe_choices = {'count': str(len(self._probe_choices))}
        for i, entry in enumerate(self._probe_choices):
            probe_choices[f'name{i}'] = entry['name']
            probe_choices[f'path{i}'] = entry['path']
        cfg['probe_choices'] = probe_choices
        # Per-probe RPI parameter sets (only for probes that have one).
        for i, entry in enumerate(self._probe_choices):
            params = entry.get('params') or {}
            if params:
                cfg[f'probe_params_{i}'] = {k: str(v) for k, v in params.items()}

        with open(str(_INI_FILE), 'w') as f:
            cfg.write(f)

    def closeEvent(self, event):
        self._save_settings()
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        super().closeEvent(event)

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

    def _on_ramp_toggled(self, checked):
        self.label_ramp_radius.setEnabled(checked)
        self.ramp_radius_spin.setEnabled(checked)

    def _on_h5_toggled(self, checked):
        self.h5_file_edit.setEnabled(checked)
        self.h5_browse_btn.setEnabled(checked)
        self.label_h5_start.setEnabled(checked)
        self.h5_start_spin.setEnabled(checked)
        self.pv_name_edit.setEnabled(not checked)

    # ------------------------------------------------------------------ #
    # Start / Stop                                                         #
    # ------------------------------------------------------------------ #

    def _on_start_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()
            self.start_btn.setText('Start')
            self.algo_combo.setEnabled(True)
            self._set_status('Stopped.')
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

        self._autosave_h5_path = None   # new file for each run
        params = self._collect_params()

        # Recreate deque with current length
        self._results_deque = deque(maxlen=params['deque_length'])
        self.save_entry_spin.setMaximum(0)

        worker_cls = WORKER_CLASSES[params['method']]
        self._worker = worker_cls(
            params            = params,
            probe_raw         = self._probe_raw,
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
        self._set_status(f'Starting… ({_METHOD_LABELS[self.algo_combo.currentIndex()]})')

    def _on_worker_finished(self):
        self.start_btn.setText('Start')
        self.algo_combo.setEnabled(True)

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

        return {
            'method':           method,
            'pv_name':          self.pv_name_edit.text().strip(),
            'h5_mode':          self.h5_check.isChecked(),
            'h5_file':          self.h5_file_edit.text().strip(),
            'h5_start_idx':     self.h5_start_spin.value(),
            'beam_cy':          self.beam_cy_spin.value(),
            'beam_cx':          self.beam_cx_spin.value(),
            'crop_N':           self.crop_N_spin.value(),
            'obj_shape':        self.obj_shape_spin.value(),
            'gpu_index':        self.gpu_index_spin.value(),
            'n_probe_modes':    n_probe_modes,  # 0 = use all
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
            'energy_kev':       self.energy_kev_spin.value(),
            'detector_distance':self.detector_dist_spin.value(),
            # Probe metadata passed through to avoid re-reading inside worker
            'res_m_probe':      self._res_m_probe,
        }

    # ------------------------------------------------------------------ #
    # Frame display                                                        #
    # ------------------------------------------------------------------ #

    def _on_frame_ready(self, entry):
        # Update save spin range to reflect how many entries are in the deque
        n = len(self._results_deque)
        self.save_entry_spin.setMaximum(max(0, n - 1))

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

    def _process_phase(self, obj):
        """Apply the current Display settings (unwrap / ramp removal) to the
        phase of one complex object array and return the display phase."""
        phase      = np.angle(obj).astype(np.float32)
        do_unwrap  = self.unwrap_check.isChecked()
        do_ramp    = self.ramp_check.isChecked()

        # Ramp removal requires an unwrapped phase; if the user only wants ramp
        # removal without displaying the unwrapped result, we unwrap internally
        # then wrap back after subtracting the ramp.
        if do_unwrap or do_ramp:
            phase = unwrap_phase(phase).astype(np.float32)
        if do_ramp:
            radius = self.ramp_radius_spin.value() or None  # 0 → None (all pixels)
            phase  = remove_phase_ramp(phase, radius=radius).astype(np.float32)
            if not do_unwrap:
                phase = ((phase + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)
        return phase

    def _render_entry(self, entry):
        phase = self._process_phase(entry['obj'])
        # With pg.setConfigOptions(imageAxisOrder='row-major'), data[row, col]
        # maps to (y, x) correctly — no transpose needed.
        self.image_view.setImage(phase, autoLevels=True, autoRange=False)
        self._last_display_img  = phase
        self._last_pixel_size_m = entry.get('pixel_size_m')
        self._update_pixel_info()

    def _rerender_last(self):
        """Re-display the most recent deque entry when unwrap toggle changes."""
        if not self._results_deque:
            return
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
            parts.append(f'phase: {float(self._last_display_img[r, c]):.2f}')
        self.pixel_info_label.setText('     '.join(parts))

    # ------------------------------------------------------------------ #
    # View deque                                                          #
    # ------------------------------------------------------------------ #

    def _on_view_deque(self):
        if not self._results_deque:
            QMessageBox.information(self, 'Empty deque',
                                    'Run at least one frame first.')
            return
        # Oldest → newest, left → right; each processed with current settings.
        imgs = [self._process_phase(e['obj']) for e in self._results_deque]
        try:
            concat = np.concatenate(imgs, axis=1)
        except ValueError:
            QMessageBox.warning(self, 'Cannot concatenate',
                                'Deque frames have differing shapes.')
            return
        if self._deque_viewer is None:
            self._deque_viewer = _DequeViewer(self)
        self._deque_viewer.set_image(concat, n=len(imgs))
        self._deque_viewer.show()
        self._deque_viewer.raise_()
        self._deque_viewer.activateWindow()

    # ------------------------------------------------------------------ #
    # Change probe dialog                                                  #
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
            current_path = self._probe_path,
            parent       = self,
        )
        dlg.exec_()
        # Keep any add/delete/rename/change-file edits to the library,
        # regardless of accept/cancel, and reflect them in the Setup window.
        self._probe_choices = dlg.get_choices()
        self._refresh_params_probe_list()

        name, path = dlg.get_result()
        if path is not None:
            try:
                probe, res_m = load_probe_file(path)
                self._probe_raw   = probe
                self._probe_path  = path
                self._res_m_probe = res_m
                self.probe_file_edit.setText(path)
                self.probe_name_edit.setText(name)
                self._set_status(
                    f'Probe set: “{name}” — shape={probe.shape}, '
                    f'pixel size={res_m*1e9:.2f} nm'
                )
            except Exception as exc:
                self._set_status(f'Probe set error: {exc}')

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
    # Worker error                                                         #
    # ------------------------------------------------------------------ #

    def _on_view_pv_image(self):
        if self._pv_viewer is None:
            self._pv_viewer = _PVImageViewer(self)
        self._pv_viewer.show()
        self._pv_viewer.raise_()

    def _on_raw_image_ready(self, img):
        if self._pv_viewer is not None and self._pv_viewer.isVisible():
            self._pv_viewer.update_image(
                img,
                beam_cy = self.beam_cy_spin.value(),
                beam_cx = self.beam_cx_spin.value(),
                crop_N  = self.crop_N_spin.value(),
                h5_mode = self.h5_check.isChecked(),
            )

    def _on_worker_error(self, msg):
        self.start_btn.setText('Start')
        self.algo_combo.setEnabled(True)
        first_line = msg.split('\n')[0] if msg else 'Worker error'
        self._set_status(f'Error: {first_line}')
        _DiagnosticDialog(msg, self).exec_()

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def _set_status(self, text):
        self.status_label.setText(text)


# ─────────────────────────────────────────────────────────────────────────── #

class _PVImageViewer(QDialog):
    """Non-modal window showing the latest raw diffraction pattern."""

    def __init__(self, parent=None):
        super().__init__(parent)
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

        self._iv = pg.ImageView()
        layout.addWidget(self._iv)

        # Bottom bar: log checkbox on the left, cursor position on the right
        bottom = QHBoxLayout()
        self._log_check = QCheckBox('log₁₀')
        self._log_check.toggled.connect(self._rerender_reset_levels)
        bottom.addWidget(self._log_check)
        bottom.addStretch()
        self._pos_label = QLabel('')
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


class _DequeViewer(QDialog):
    """Non-modal window showing every deque frame concatenated side-by-side.

    The caller passes an image already built from the current Display settings
    (unwrap / ramp removal applied per frame), so this window only displays it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Deque')
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(900, 400)

        layout = QVBoxLayout(self)
        self._iv = pg.ImageView()
        layout.addWidget(self._iv)

        vb = self._iv.getView()
        vb.menu.addAction('Reset Zoom', vb.autoRange)
        self._first_image = True

    def set_image(self, img, n=1):
        self.setWindowTitle(f'Deque — {n} frame(s), oldest → newest')
        # row-major: data[row, col] → (y, x); autoRange only on first display.
        self._iv.setImage(img, autoLevels=True, autoRange=self._first_image)
        self._first_image = False


class _DiagnosticDialog(QDialog):
    """Resizable, scrollable dialog for worker error diagnostics."""

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Worker Error — Diagnostic')
        self.resize(820, 520)

        layout = QVBoxLayout(self)

        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText(text)
        te.setFontFamily('monospace')
        layout.addWidget(te)

        close_btn = QPushButton('Close')
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
