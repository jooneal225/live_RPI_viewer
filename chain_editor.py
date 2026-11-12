"""
Custom RPI Chain editor — the window behind "Create Custom RPI chain".

Common parameters on top, then a table with one row per step. "+" appends a
step of the algorithm picked in the combo, prefilled from the RPI Parameters
window; "−" removes the selected rows. Cells are edited by typing; a cell
that does not parse turns red and keeps its error in the tooltip, and the
step keeps its last valid value until it is fixed.

The chain lives only in memory: it is empty at every start (the "Custom RPI
Chain" method stays greyed out until a step is added) and is kept across
sessions only through Save/Load JSON.

The step schema, parsing and JSON live in rpi_chain.py.
"""

import copy

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from rpi_chain import (
    ALGOS, ALGO_LABELS, COLUMNS, COLUMN_LABELS, COLUMN_TOOLTIPS, COMMON_RANGES,
    OBJ_INITS, applies, format_cell, parse_cell, read_chain_json,
    step_from_params, total_iterations, validate_common, write_chain_json,
)

# Index-aligned with rpi_chain.OBJ_INITS, as in the main window's Obj init combo.
_INIT_LABELS = ['Random', 'Flat']

_BAD_BRUSH = QBrush(QColor(220, 60, 60, 110))
_NA_BRUSH  = QBrush(QColor(128, 128, 128, 50))


def _named(widget, name):
    """setObjectName and return the widget, so apply_font_size_to_tree
    (which skips unnamed widgets) reaches it."""
    widget.setObjectName(name)
    return widget


class ChainEditor(QDialog):
    """Non-blocking editor for the custom RPI chain.

    common_defaults() -> {n_probe_modes, crop_N, obj_shape, obj_init, n_avg}
        fills the common fields when the window is built and on Wipe.
    step_params() -> MainWindow._get_current_rpi_params()
        prefills each step added with "+".
    """

    # Emitted after any change that can affect is_set(), errors() or snapshot().
    chain_changed = pyqtSignal()

    def __init__(self, common_defaults, step_params, parent=None):
        super().__init__(parent)
        self.setObjectName('chain_win')
        self.setWindowTitle('Custom RPI Chain')
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self._common_defaults = common_defaults
        self._step_params     = step_params
        self._steps: list = []   # step dicts, index-aligned with table rows
        self._bad: list   = []   # per row: {column index: error} for red cells
        self._last_dir    = ''
        self._filling     = False

        outer = QVBoxLayout(self)

        # ---- Common parameters ----
        common_box = _named(QGroupBox('Common'), 'chain_common_box')
        cf = QFormLayout(common_box)
        lo, hi = COMMON_RANGES['n_probe_modes']
        self.n_modes_spin = _named(QSpinBox(), 'chain_n_modes_spin')
        self.n_modes_spin.setRange(lo, hi)
        self.n_modes_spin.setSpecialValueText('All')
        self.n_modes_spin.setToolTip(
            'Probe modes used by every step (0 = all). LBFGS Incoh Obj steps '
            'always use 2 object modes.')
        self.crop_spin = _named(QSpinBox(), 'chain_crop_spin')
        self.crop_spin.setRange(*COMMON_RANGES['crop_N'])
        self.crop_spin.setSingleStep(2)
        self.obj_shape_spin = _named(QSpinBox(), 'chain_obj_shape_spin')
        self.obj_shape_spin.setRange(*COMMON_RANGES['obj_shape'])
        self.obj_shape_spin.setSingleStep(2)
        for w in (self.crop_spin, self.obj_shape_spin):
            w.setToolTip('Must be even.')
        self.obj_init_combo = _named(QComboBox(), 'chain_obj_init_combo')
        self.obj_init_combo.addItems(_INIT_LABELS)
        self.obj_init_combo.setToolTip(
            'Initial object of the first step, made the way that step\'s '
            'algorithm makes it on its own. Later steps start from the object '
            'the previous step ended with.')
        self.n_avg_spin = _named(QSpinBox(), 'chain_n_avg_spin')
        self.n_avg_spin.setRange(*COMMON_RANGES['n_avg'])
        cf.addRow(_named(QLabel('N probe modes:'), 'label_chain_modes'), self.n_modes_spin)
        cf.addRow(_named(QLabel('Det crop:'), 'label_chain_crop'), self.crop_spin)
        cf.addRow(_named(QLabel('Obj shape:'), 'label_chain_obj'), self.obj_shape_spin)
        cf.addRow(_named(QLabel('Obj init:'), 'label_chain_init'), self.obj_init_combo)
        cf.addRow(_named(QLabel('Average N patterns:'), 'label_chain_navg'), self.n_avg_spin)
        outer.addWidget(common_box)
        for w in (self.n_modes_spin, self.crop_spin, self.obj_shape_spin, self.n_avg_spin):
            w.valueChanged.connect(self._on_common_changed)
        self.obj_init_combo.currentIndexChanged.connect(self._on_common_changed)

        # ---- Add / remove steps ----
        add_row = QHBoxLayout()
        add_row.addWidget(_named(QLabel('Algorithm:'), 'label_chain_algo'))
        self.algo_combo = _named(QComboBox(), 'chain_algo_combo')
        self.algo_combo.addItems([ALGO_LABELS[a] for a in ALGOS])
        add_row.addWidget(self.algo_combo)
        self.add_btn = _named(QPushButton('+'), 'chain_add_btn')
        self.add_btn.setStyleSheet(
            'QPushButton { background-color: #2e9e44; color: white; font-weight: bold; }')
        self.add_btn.setToolTip('Add a step of the selected algorithm, with the '
                                'values from the RPI Parameters window.')
        self.add_btn.clicked.connect(self._on_add)
        self.remove_btn = _named(QPushButton('−'), 'chain_remove_btn')
        self.remove_btn.setStyleSheet(
            'QPushButton { background-color: #c0392b; color: white; font-weight: bold; }')
        self.remove_btn.setToolTip('Remove the selected steps (the last one if '
                                   'nothing is selected).')
        self.remove_btn.clicked.connect(self._on_remove)
        add_row.addWidget(self.add_btn)
        add_row.addWidget(self.remove_btn)
        add_row.addStretch(1)
        outer.addLayout(add_row)

        # ---- Steps table ----
        self.table = _named(QTableWidget(0, len(COLUMNS)), 'chain_table')
        self.table.setHorizontalHeaderLabels(COLUMN_LABELS)
        for c, key in enumerate(COLUMNS):
            self.table.horizontalHeaderItem(c).setToolTip(COLUMN_TOOLTIPS[key])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COLUMNS.index('options'), QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed)
        self.table.itemChanged.connect(self._on_item_changed)
        outer.addWidget(self.table, 1)

        self.summary_label = _named(QLabel(''), 'chain_summary_label')
        self.summary_label.setWordWrap(True)
        outer.addWidget(self.summary_label)

        # ---- File / reset / close ----
        btn_row = QHBoxLayout()
        save_btn  = _named(QPushButton('Save JSON…'), 'chain_save_btn')
        load_btn  = _named(QPushButton('Load JSON…'), 'chain_load_btn')
        wipe_btn  = _named(QPushButton('Wipe Custom algorithm'), 'chain_wipe_btn')
        close_btn = _named(QPushButton('Close'), 'chain_close_btn')
        save_btn.clicked.connect(self._on_save)
        load_btn.clicked.connect(self._on_load)
        wipe_btn.clicked.connect(self.wipe)
        close_btn.clicked.connect(self.hide)
        for b in (save_btn, load_btn, wipe_btn, close_btn):
            btn_row.addWidget(b)
        outer.addLayout(btn_row)

        self._set_common(self._common_defaults())
        self.resize(820, 520)
        self._update_summary()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_set(self):
        """True once the chain has at least one step."""
        return bool(self._steps)

    def common(self):
        return {
            'n_probe_modes': self.n_modes_spin.value(),
            'crop_N':        self.crop_spin.value(),
            'obj_shape':     self.obj_shape_spin.value(),
            'obj_init':      OBJ_INITS[self.obj_init_combo.currentIndex()],
            'n_avg':         self.n_avg_spin.value(),
        }

    def snapshot(self):
        """(common, steps) as independent copies, so a running worker never
        sees later edits."""
        return self.common(), copy.deepcopy(self._steps)

    def errors(self):
        """Problems that must be fixed before the chain can run or be saved."""
        if not self._steps:
            return ['The custom chain has no steps.']
        errs = []
        try:
            validate_common(self.common())
        except ValueError as exc:
            errs.append(f'Common: {exc}')
        for row, bad in enumerate(self._bad):
            for col, msg in sorted(bad.items()):
                errs.append(f'Step {row + 1}, {COLUMN_LABELS[col]}: {msg}')
        return errs

    def wipe(self):
        """Empty the table and reset the common fields to their defaults."""
        self._steps, self._bad = [], []
        self.table.setRowCount(0)
        self._set_common(self._common_defaults())
        self._changed()

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _fill_row(self, row):
        """Write step `row` into the table, dropping any red-cell state."""
        step = self._steps[row]
        self._filling = True
        try:
            for c, key in enumerate(COLUMNS):
                item = QTableWidgetItem(format_cell(step, key))
                flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                if key == 'algo':
                    item.setToolTip(COLUMN_TOOLTIPS['algo'])
                elif applies(step['algo'], key):
                    flags |= Qt.ItemFlag.ItemIsEditable
                    item.setToolTip(COLUMN_TOOLTIPS[key])
                else:
                    item.setBackground(_NA_BRUSH)
                    item.setForeground(QBrush(QColor(128, 128, 128)))
                item.setFlags(flags)
                if key != 'options':
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, c, item)
        finally:
            self._filling = False

    def _on_item_changed(self, item):
        if self._filling:
            return
        row, col = item.row(), item.column()
        key = COLUMNS[col]
        if row >= len(self._steps) or key == 'algo':
            return
        step = self._steps[row]
        if not applies(step['algo'], key):
            return
        fields, err = parse_cell(step['algo'], key, item.text())
        self._filling = True
        try:
            if err:
                self._bad[row][col] = err
                item.setBackground(_BAD_BRUSH)
                item.setToolTip(f'Invalid: {err}\n\n{COLUMN_TOOLTIPS[key]}')
            else:
                self._bad[row].pop(col, None)
                step.update(fields)
                # Show the normalized value ('w' → 'weights', '1e-5' → both modes).
                item.setText(format_cell(step, key))
                item.setBackground(QBrush())
                item.setToolTip(COLUMN_TOOLTIPS[key])
        finally:
            self._filling = False
        self._changed()

    def _on_add(self):
        algo = ALGOS[self.algo_combo.currentIndex()]
        self._steps.append(step_from_params(algo, self._step_params()))
        self._bad.append({})
        row = len(self._steps) - 1
        self.table.insertRow(row)
        self._fill_row(row)
        self.table.scrollToItem(self.table.item(row, 0))
        self._changed()

    def _on_remove(self):
        if not self._steps:
            return
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            rows = [len(self._steps) - 1]
        for row in rows:
            del self._steps[row]
            del self._bad[row]
            self.table.removeRow(row)
        self._changed()

    # ------------------------------------------------------------------
    # Common parameters
    # ------------------------------------------------------------------

    def _set_common(self, c):
        for w in (self.n_modes_spin, self.crop_spin, self.obj_shape_spin,
                  self.n_avg_spin, self.obj_init_combo):
            w.blockSignals(True)
        try:
            self.n_modes_spin.setValue(int(c.get('n_probe_modes', 0)))
            self.crop_spin.setValue(int(c.get('crop_N', 512)))
            self.obj_shape_spin.setValue(int(c.get('obj_shape', 100)))
            init = c.get('obj_init', 'flat')
            self.obj_init_combo.setCurrentIndex(
                OBJ_INITS.index(init) if init in OBJ_INITS else OBJ_INITS.index('flat'))
            self.n_avg_spin.setValue(int(c.get('n_avg', 1)))
        finally:
            for w in (self.n_modes_spin, self.crop_spin, self.obj_shape_spin,
                      self.n_avg_spin, self.obj_init_combo):
                w.blockSignals(False)

    def _on_common_changed(self, *_):
        self._changed()

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def _on_save(self):
        errs = self.errors()
        if errs:
            QMessageBox.warning(self, 'Cannot save chain',
                                'Fix these first:\n\n' + '\n'.join(errs))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Custom RPI Chain', self._last_dir, 'JSON files (*.json)')
        if not path:
            return
        if not path.lower().endswith('.json'):
            path += '.json'
        common, steps = self.snapshot()
        try:
            write_chain_json(path, common, steps)
        except Exception as exc:
            QMessageBox.warning(self, 'Save failed', str(exc))
            return
        self._last_dir = path
        self.summary_label.setText(f'Saved to {path}\n' + self._summary_text())

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Custom RPI Chain', self._last_dir,
            'JSON files (*.json);;All files (*)')
        if not path:
            return
        try:
            common, steps = read_chain_json(path)
        except ValueError as exc:
            QMessageBox.warning(self, 'Load failed', str(exc))
            return
        self._last_dir = path
        self._set_common(common)
        self._steps = steps
        self._bad = [{} for _ in steps]
        self.table.setRowCount(len(steps))
        for row in range(len(steps)):
            self._fill_row(row)
        self._changed()
        self.summary_label.setText(f'Loaded {path}\n' + self._summary_text())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _summary_text(self):
        if not self._steps:
            return ('No custom chain set — "Custom RPI Chain" stays greyed out '
                    'in the main window until a step is added.')
        n_bad = sum(len(b) for b in self._bad)
        text = (f'{len(self._steps)} step{"s" if len(self._steps) != 1 else ""}, '
                f'{total_iterations(self._steps)} iterations in total.')
        if n_bad:
            text += f'  {n_bad} invalid cell{"s" if n_bad != 1 else ""} (red) — fix before running.'
        try:
            validate_common(self.common())
        except ValueError as exc:
            text += f'  Common: {exc}.'
        return text

    def _update_summary(self):
        self.summary_label.setText(self._summary_text())

    def _changed(self):
        self._update_summary()
        self.chain_changed.emit()
