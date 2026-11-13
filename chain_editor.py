"""
Custom RPI Chain editor — the window behind "Create Custom RPI chain".

The top half lists every chain held in memory. Clicking one loads it into the
editor below, and edits there change that chain directly. The list's buttons
create, duplicate, delete and reorder chains (dragging also reorders, and
double-clicking renames). "Save JSON…" asks for a file for each chain in
turn; "Load JSON…" takes several files at once and adds one chain per file.

The editor has the chain's common parameters, then a table with one row per
step. "+" appends a step of the algorithm picked in the combo, prefilled from
the RPI Parameters window; "−" removes the selected rows and ▲/▼ move them.
Cells are edited by typing; a cell that does not parse turns red and keeps
its error in the tooltip, and the step keeps its last valid value until it
is fixed.

Chains live only in memory: the list starts with one empty chain at every
start (the "Custom RPI Chain" method stays greyed out until some chain has a
step) and is kept across sessions only through Save/Load JSON. The main
window picks which chain runs, by its id, from the combo under the method.

The step schema, parsing and JSON live in rpi_chain.py.
"""

import copy
import itertools
import os
import re

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFileDialog,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QSizePolicy, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout,
    QWidget,
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

_ID_ROLE = Qt.ItemDataRole.UserRole


def _named(widget, name):
    """setObjectName and return the widget, so apply_font_size_to_tree
    (which skips unnamed widgets) reaches it."""
    widget.setObjectName(name)
    return widget


def _small_btn(text, name, tip, style=None):
    """A QToolButton, which (unlike QPushButton) has no ~80 px minimum width,
    so rows of these stay narrow. Grows horizontally to fill grid cells."""
    b = _named(QToolButton(), name)
    b.setText(text)
    b.setToolTip(tip)
    b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    if style:
        b.setStyleSheet(style)
    return b


_GREEN = 'QToolButton { background-color: #2e9e44; color: white; font-weight: bold; }'
_RED   = 'QToolButton { background-color: #c0392b; color: white; font-weight: bold; }'


class ChainEditor(QDialog):
    """Non-blocking editor for the list of custom RPI chains.

    common_defaults() -> {n_probe_modes, crop_N, obj_shape, obj_init, n_avg}
        fills the common fields of each new chain.
    step_params() -> MainWindow._get_current_rpi_params()
        prefills each step added with "+".

    Each chain is {'id', 'name', 'common', 'steps', 'bad'}, where 'bad' is,
    per step row, {column index: (error, text typed)} for its red cells.
    Callers address chains by 'id', which survives renames and reordering.
    """

    # Emitted after any change that can affect the list of chains, their
    # names, is_set(), errors() or snapshot().
    chain_changed = pyqtSignal()

    def __init__(self, common_defaults, step_params, parent=None):
        super().__init__(parent)
        self.setObjectName('chain_win')
        self.setWindowTitle('Custom RPI Chains')
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self._common_defaults = common_defaults
        self._step_params     = step_params
        self._chains: list = []
        self._ids         = itertools.count(1)
        self._cur         = None    # chain shown in the editor, or None
        self._last_dir    = ''
        self._filling     = False

        outer = QVBoxLayout(self)
        splitter = _named(QSplitter(Qt.Orientation.Vertical), 'chain_splitter')
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        # ---- Chain list ----
        list_box = _named(QGroupBox('Chains'), 'chain_list_box')
        lh = QHBoxLayout(list_box)
        self.chain_list = _named(QListWidget(), 'chain_list')
        self.chain_list.setToolTip(
            'Click a chain to edit it below. Double-click to rename, drag to '
            'reorder. The main window picks which one runs.')
        self.chain_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.chain_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.chain_list.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.chain_list.setMinimumHeight(60)
        self.chain_list.currentRowChanged.connect(self._on_chain_selected)
        self.chain_list.itemChanged.connect(self._on_chain_renamed)
        self.chain_list.model().rowsMoved.connect(self._on_chains_dragged)
        lh.addWidget(self.chain_list, 1)

        lb = QGridLayout()
        lb.setSpacing(3)
        new_btn  = _small_btn('New', 'chain_new_btn', 'Add an empty chain.')
        dup_btn  = _small_btn('Copy', 'chain_dup_btn', 'Duplicate the selected chain.')
        del_btn  = _small_btn('Delete', 'chain_del_btn', 'Delete the selected chain.')
        up_btn   = _small_btn('▲', 'chain_up_btn', 'Move the selected chain up.')
        down_btn = _small_btn('▼', 'chain_down_btn', 'Move the selected chain down.')
        save_btn = _small_btn('Save JSON…', 'chain_save_btn',
                              'Save every chain in the list, asking for a file '
                              'for each in turn.')
        load_btn = _small_btn('Load JSON…', 'chain_load_btn',
                              'Add one chain per selected JSON file.')
        new_btn.clicked.connect(self._on_new_chain)
        dup_btn.clicked.connect(self._on_duplicate_chain)
        del_btn.clicked.connect(self._on_delete_chain)
        up_btn.clicked.connect(lambda: self._move_chain(-1))
        down_btn.clicked.connect(lambda: self._move_chain(+1))
        save_btn.clicked.connect(self._on_save)
        load_btn.clicked.connect(self._on_load)
        lb.addWidget(new_btn, 0, 0)
        lb.addWidget(dup_btn, 0, 1)
        lb.addWidget(del_btn, 1, 0)
        updown = QHBoxLayout()
        updown.setSpacing(3)
        updown.addWidget(up_btn)
        updown.addWidget(down_btn)
        lb.addLayout(updown, 1, 1)
        lb.addWidget(save_btn, 2, 0, 1, 2)
        lb.addWidget(load_btn, 3, 0, 1, 2)
        lb.setRowStretch(4, 1)
        lh.addLayout(lb)
        splitter.addWidget(list_box)

        # ---- Editor for the selected chain ----
        self.editor = _named(QWidget(), 'chain_editor_pane')
        ev = QVBoxLayout(self.editor)
        ev.setContentsMargins(0, 0, 0, 0)

        common_box = _named(QGroupBox('Common'), 'chain_common_box')
        cf = QGridLayout(common_box)
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
        # Two label/field columns.
        for i, (label, name, w) in enumerate((
                ('Det crop:', 'label_chain_crop', self.crop_spin),
                ('Obj shape:', 'label_chain_obj', self.obj_shape_spin),
                ('Probe modes:', 'label_chain_modes', self.n_modes_spin),
                ('Obj init:', 'label_chain_init', self.obj_init_combo),
                ('Avg N dp:', 'label_chain_navg', self.n_avg_spin))):
            r, c = i % 3, 3 * (i // 3)
            cf.addWidget(_named(QLabel(label), name), r, c)
            cf.addWidget(w, r, c + 1)
        cf.setColumnMinimumWidth(2, 12)
        cf.setColumnStretch(1, 1)
        cf.setColumnStretch(4, 1)
        self.n_avg_spin.setToolTip('Average this many diffraction patterns '
                                   'before each reconstruction.')
        ev.addWidget(common_box)
        for w in (self.n_modes_spin, self.crop_spin, self.obj_shape_spin, self.n_avg_spin):
            w.valueChanged.connect(self._on_common_changed)
        self.obj_init_combo.currentIndexChanged.connect(self._on_common_changed)

        # Add / remove / move steps
        add_row = QHBoxLayout()
        add_row.setSpacing(3)
        self.algo_combo = _named(QComboBox(), 'chain_algo_combo')
        self.algo_combo.addItems([ALGO_LABELS[a] for a in ALGOS])
        self.algo_combo.setToolTip('Algorithm of the step "+" adds.')
        self.algo_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.algo_combo.setMinimumContentsLength(8)
        add_row.addWidget(self.algo_combo, 1)
        self.add_btn = _small_btn('+', 'chain_add_btn',
                                  'Add a step of the selected algorithm, with the '
                                  'values from the RPI Parameters window.', _GREEN)
        self.remove_btn = _small_btn('−', 'chain_remove_btn',
                                     'Remove the selected steps (the last one if '
                                     'nothing is selected).', _RED)
        self.step_up_btn   = _small_btn('▲', 'chain_step_up_btn',
                                        'Move the selected steps up.')
        self.step_down_btn = _small_btn('▼', 'chain_step_down_btn',
                                        'Move the selected steps down.')
        self.add_btn.clicked.connect(self._on_add)
        self.remove_btn.clicked.connect(self._on_remove)
        self.step_up_btn.clicked.connect(lambda: self._move_steps(-1))
        self.step_down_btn.clicked.connect(lambda: self._move_steps(+1))
        for b in (self.add_btn, self.remove_btn, self.step_up_btn, self.step_down_btn):
            b.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            b.setMinimumWidth(28)
            add_row.addWidget(b)
        ev.addLayout(add_row)

        self.table = _named(QTableWidget(0, len(COLUMNS)), 'chain_table')
        self.table.setHorizontalHeaderLabels(COLUMN_LABELS)
        for c, key in enumerate(COLUMNS):
            self.table.horizontalHeaderItem(c).setToolTip(COLUMN_TOOLTIPS[key])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed)
        self.table.setMinimumSize(120, 80)
        self.table.itemChanged.connect(self._on_item_changed)
        ev.addWidget(self.table, 1)

        self.summary_label = _named(QLabel(''), 'chain_summary_label')
        self.summary_label.setWordWrap(True)
        ev.addWidget(self.summary_label)
        splitter.addWidget(self.editor)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        close_btn = _named(QPushButton('Close'), 'chain_close_btn')
        close_btn.clicked.connect(self.hide)
        outer.addWidget(close_btn)

        self._add_chain(self._new_chain(self._unique_name('Chain 1')))
        self.resize(560, 620)
        splitter.setSizes([150, 470])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chains(self):
        """[(id, name, n_steps), …] in list order."""
        return [(c['id'], c['name'], len(c['steps'])) for c in self._chains]

    def is_set(self, cid=None):
        """True if chain `cid` (any chain, if None) has at least one step."""
        if cid is None:
            return any(c['steps'] for c in self._chains)
        c = self._find(cid)
        return c is not None and bool(c['steps'])

    def name(self, cid):
        c = self._find(cid)
        return c['name'] if c else ''

    def common(self, cid):
        return dict(self._find(cid)['common'])

    def snapshot(self, cid):
        """(common, steps) of chain `cid` as independent copies, so a running
        worker never sees later edits."""
        c = self._find(cid)
        return dict(c['common']), copy.deepcopy(c['steps'])

    def errors(self, cid):
        """Problems that must be fixed before chain `cid` can run or be saved."""
        c = self._find(cid)
        if c is None:
            return ['The selected custom chain no longer exists.']
        if not c['steps']:
            return [f'Chain "{c["name"]}" has no steps.']
        errs = []
        try:
            validate_common(c['common'])
        except ValueError as exc:
            errs.append(f'Common: {exc}')
        for row, bad in enumerate(c['bad']):
            for col, (msg, _text) in sorted(bad.items()):
                errs.append(f'Step {row + 1}, {COLUMN_LABELS[col]}: {msg}')
        return errs

    def select(self, cid):
        """Show chain `cid` in the editor."""
        for row, c in enumerate(self._chains):
            if c['id'] == cid:
                self.chain_list.setCurrentRow(row)
                return

    # ------------------------------------------------------------------
    # Chain list
    # ------------------------------------------------------------------

    def _find(self, cid):
        return next((c for c in self._chains if c['id'] == cid), None)

    def _new_chain(self, name, common=None, steps=None):
        steps = steps or []
        return {'id': next(self._ids), 'name': name,
                'common': dict(common) if common else self._common_defaults(),
                'steps': steps, 'bad': [{} for _ in steps]}

    def _unique_name(self, name, ignore=None):
        """`name`, or `name (2)`, `name (3)`, … if another chain has it."""
        taken = {c['name'] for c in self._chains if c is not ignore}
        if name not in taken:
            return name
        base = re.sub(r' \(\d+\)$', '', name)
        for i in itertools.count(2):
            cand = f'{base} ({i})'
            if cand not in taken:
                return cand

    def _list_item(self, chain):
        item = QListWidgetItem(chain['name'])
        item.setData(_ID_ROLE, chain['id'])
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                      | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsDragEnabled)
        return item

    def _rebuild_list(self, select_id=None):
        self.chain_list.blockSignals(True)
        try:
            self.chain_list.clear()
            for c in self._chains:
                self.chain_list.addItem(self._list_item(c))
        finally:
            self.chain_list.blockSignals(False)
        row = next((i for i, c in enumerate(self._chains) if c['id'] == select_id), -1)
        if row < 0 and self._chains:
            row = 0
        self.chain_list.setCurrentRow(row)
        # setCurrentRow emits nothing when the row number is unchanged.
        self._on_chain_selected(row)

    def _add_chain(self, chain, at=None):
        at = len(self._chains) if at is None else at
        self._chains.insert(at, chain)
        self._rebuild_list(chain['id'])
        self._changed()

    def _on_new_chain(self):
        self._add_chain(self._new_chain(self._unique_name(f'Chain {len(self._chains) + 1}')))

    def _on_duplicate_chain(self):
        if self._cur is None:
            return
        c = self._cur
        dup = self._new_chain(self._unique_name(c['name']), c['common'],
                              copy.deepcopy(c['steps']))
        dup['bad'] = copy.deepcopy(c['bad'])
        self._add_chain(dup, self._chains.index(c) + 1)

    def _on_delete_chain(self):
        if self._cur is None:
            return
        row = self._chains.index(self._cur)
        if self._cur['steps'] and QMessageBox.question(
                self, 'Delete chain',
                f'Delete chain "{self._cur["name"]}"?') != QMessageBox.StandardButton.Yes:
            return
        del self._chains[row]
        nxt = self._chains[min(row, len(self._chains) - 1)]['id'] if self._chains else None
        self._rebuild_list(nxt)
        self._changed()

    def _move_chain(self, delta):
        if self._cur is None:
            return
        row = self._chains.index(self._cur)
        new = row + delta
        if not 0 <= new < len(self._chains):
            return
        self._chains.insert(new, self._chains.pop(row))
        self._rebuild_list(self._cur['id'])
        self._changed()

    def _on_chains_dragged(self, *_):
        """Reorder self._chains to match the list after a drag."""
        order = [self.chain_list.item(i).data(_ID_ROLE)
                 for i in range(self.chain_list.count())]
        self._chains.sort(key=lambda c: order.index(c['id']))
        self._changed()

    def _on_chain_renamed(self, item):
        c = self._find(item.data(_ID_ROLE))
        if c is None:
            return
        name = self._unique_name(item.text().strip() or c['name'], ignore=c)
        if name != item.text():
            self.chain_list.blockSignals(True)
            item.setText(name)
            self.chain_list.blockSignals(False)
        if name != c['name']:
            c['name'] = name
            self._changed()

    def _on_chain_selected(self, row):
        self._cur = self._chains[row] if 0 <= row < len(self._chains) else None
        self.editor.setEnabled(self._cur is not None)
        c = self._cur
        self._set_common(c['common'] if c else self._common_defaults())
        self.table.setRowCount(len(c['steps']) if c else 0)
        for r in range(self.table.rowCount()):
            self._fill_row(r)
        self._update_summary()

    # ------------------------------------------------------------------
    # Steps table (of the selected chain)
    # ------------------------------------------------------------------

    def _fill_row(self, row):
        """Write step `row` of the selected chain into the table, with its
        red cells showing the text that did not parse."""
        step = self._cur['steps'][row]
        bad = self._cur['bad'][row]
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
                    if c in bad:
                        msg, text = bad[c]
                        item.setText(text)
                        item.setBackground(_BAD_BRUSH)
                        item.setToolTip(f'Invalid: {msg}\n\n{COLUMN_TOOLTIPS[key]}')
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
        if self._filling or self._cur is None:
            return
        row, col = item.row(), item.column()
        key = COLUMNS[col]
        if row >= len(self._cur['steps']) or key == 'algo':
            return
        step = self._cur['steps'][row]
        if not applies(step['algo'], key):
            return
        fields, err = parse_cell(step['algo'], key, item.text())
        self._filling = True
        try:
            if err:
                self._cur['bad'][row][col] = (err, item.text())
                item.setBackground(_BAD_BRUSH)
                item.setToolTip(f'Invalid: {err}\n\n{COLUMN_TOOLTIPS[key]}')
            else:
                self._cur['bad'][row].pop(col, None)
                step.update(fields)
                # Show the normalized value ('w' → 'weights', '1e-5' → both modes).
                item.setText(format_cell(step, key))
                item.setBackground(QBrush())
                item.setToolTip(COLUMN_TOOLTIPS[key])
        finally:
            self._filling = False
        self._changed()

    def _on_add(self):
        if self._cur is None:
            return
        algo = ALGOS[self.algo_combo.currentIndex()]
        self._cur['steps'].append(step_from_params(algo, self._step_params()))
        self._cur['bad'].append({})
        row = len(self._cur['steps']) - 1
        self.table.insertRow(row)
        self._fill_row(row)
        self.table.scrollToItem(self.table.item(row, 0))
        self._changed()

    def _selected_rows(self):
        return sorted({i.row() for i in self.table.selectedIndexes()})

    def _on_remove(self):
        if self._cur is None or not self._cur['steps']:
            return
        rows = self._selected_rows() or [len(self._cur['steps']) - 1]
        for row in reversed(rows):
            del self._cur['steps'][row]
            del self._cur['bad'][row]
            self.table.removeRow(row)
        self._changed()

    def _move_steps(self, delta):
        """Move the selected rows one place up (-1) or down (+1); a row stops
        when it reaches the end or another selected row that has stopped."""
        c = self._cur
        rows = self._selected_rows()
        if c is None or not rows:
            return
        n = len(c['steps'])
        sel = set(rows)
        for row in (rows if delta < 0 else reversed(rows)):
            new = row + delta
            if 0 <= new < n and new not in sel:
                for lst in (c['steps'], c['bad']):
                    lst[row], lst[new] = lst[new], lst[row]
                sel.discard(row)
                sel.add(new)
        for r in range(n):
            self._fill_row(r)
        self.table.clearSelection()
        mode = self.table.selectionMode()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for r in sorted(sel):
            self.table.selectRow(r)
        self.table.setSelectionMode(mode)
        self._changed()

    # ------------------------------------------------------------------
    # Common parameters (of the selected chain)
    # ------------------------------------------------------------------

    def _read_common(self):
        return {
            'n_probe_modes': self.n_modes_spin.value(),
            'crop_N':        self.crop_spin.value(),
            'obj_shape':     self.obj_shape_spin.value(),
            'obj_init':      OBJ_INITS[self.obj_init_combo.currentIndex()],
            'n_avg':         self.n_avg_spin.value(),
        }

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
        if self._cur is not None:
            self._cur['common'] = self._read_common()
        self._changed()

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def _on_save(self):
        """Ask for a file for each chain in turn. A chain with errors is
        reported and skipped; cancelling a file dialog skips that chain."""
        saved, skipped = [], []
        todo = list(self._chains)
        for i, c in enumerate(todo):
            more = i < len(todo) - 1
            errs = self.errors(c['id'])
            if errs:
                skipped.append(c['name'])
                buttons = QMessageBox.StandardButton.Ok
                if more:
                    buttons |= QMessageBox.StandardButton.Abort
                if QMessageBox.warning(
                        self, 'Cannot save chain',
                        f'Chain "{c["name"]}" is skipped. Fix these first:\n\n'
                        + '\n'.join(errs),
                        buttons) == QMessageBox.StandardButton.Abort:
                    break
                continue
            stem = re.sub(r'[^\w.-]+', '_', c['name']).strip('_') or 'chain'
            start = os.path.join(os.path.dirname(self._last_dir), f'{stem}.json')
            path, _ = QFileDialog.getSaveFileName(
                self, f'Save chain "{c["name"]}" ({i + 1} of {len(todo)})',
                start, 'JSON files (*.json)')
            if not path:
                skipped.append(c['name'])
                if more and QMessageBox.question(
                        self, 'Save chains',
                        f'"{c["name"]}" was not saved. Continue with the next '
                        f'chain?') != QMessageBox.StandardButton.Yes:
                    break
                continue
            if not path.lower().endswith('.json'):
                path += '.json'
            common, steps = self.snapshot(c['id'])
            try:
                write_chain_json(path, common, steps, c['name'])
            except Exception as exc:
                QMessageBox.warning(self, 'Save failed', f'{c["name"]}: {exc}')
                skipped.append(c['name'])
                continue
            self._last_dir = path
            saved.append(path)
        msg = f'Saved {len(saved)} of {len(todo)} chain{"s" if len(todo) != 1 else ""}.'
        if skipped:
            msg += f'  Not saved: {", ".join(skipped)}.'
        self.summary_label.setText(msg + '\n' + self._summary_text())

    def _on_load(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Load Custom RPI Chains', self._last_dir,
            'JSON files (*.json);;All files (*)')
        if not paths:
            return
        failed, last = [], None
        for path in paths:
            try:
                common, steps, name = read_chain_json(path)
            except ValueError as exc:
                failed.append(str(exc))
                continue
            name = self._unique_name(name or os.path.splitext(os.path.basename(path))[0])
            last = self._new_chain(name, common, steps)
            self._chains.append(last)
            self._last_dir = path
        if last is not None:
            self._rebuild_list(last['id'])
            self._changed()
        n_ok = len(paths) - len(failed)
        self.summary_label.setText(
            f'Loaded {n_ok} of {len(paths)} file{"s" if len(paths) != 1 else ""}.\n'
            + self._summary_text())
        if failed:
            QMessageBox.warning(self, 'Load failed', '\n\n'.join(failed))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _summary_text(self):
        c = self._cur
        if c is None:
            return 'No chain selected — press "New" or "Load JSON…".'
        if not c['steps']:
            return (f'"{c["name"]}" has no steps yet. A chain can be chosen in '
                    'the main window once it has one.')
        n_bad = sum(len(b) for b in c['bad'])
        text = (f'{len(c["steps"])} step{"s" if len(c["steps"]) != 1 else ""}, '
                f'{total_iterations(c["steps"])} iterations in total.')
        if n_bad:
            text += f'  {n_bad} invalid cell{"s" if n_bad != 1 else ""} (red) — fix before running.'
        try:
            validate_common(c['common'])
        except ValueError as exc:
            text += f'  Common: {exc}.'
        return text

    def _update_summary(self):
        self.summary_label.setText(self._summary_text())

    def _changed(self):
        self._update_summary()
        self.chain_changed.emit()
