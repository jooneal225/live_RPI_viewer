"""
ProbeViewer — modal dialog to manage a library of named probe choices.

Opens from the main window's "Set Probe Choice" button (also shown once at
startup). The user maintains a named list of probe files:
  +            add an entry (file picker + name prompt)
  −            remove the selected entry
  Rename       give the selected entry a new name
  Change File  point the selected entry at a different file (name kept)

Probe data is NOT read from disk just because an entry exists in the list or
is selected — that would mean re-reading every registered probe (or at least
the previously active one) on every dialog open. Instead, "Load registered
probes" reads every entry not already cached; loading is skipped for entries
whose data is already in the shared cache. Selecting a row shows the cached
preview if present, or a "not loaded" placeholder otherwise. Accept returns
the probe data for the highlighted (and loaded) entry directly, so the
caller never needs a second disk read.

Each entry is a dict {'name', 'path', 'params'}; the RPI parameter set in
'params' is owned by the main window and passed through this dialog untouched,
so renaming a probe or changing its file keeps its associated parameters.

`probe_cache` is a dict {path: {'probe': ndarray, 'res_m': float}} owned by
the caller (typically the main window) and mutated in place, so probes loaded
in one dialog session stay loaded for the next.
"""

from pathlib import Path

import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSizePolicy, QListWidget, QListWidgetItem, QInputDialog,
)
from PyQt5.QtCore import Qt

from rpi_utils import load_probe_file


class ProbeViewer(QDialog):
    def __init__(self, choices=None, current_path=None, probe_cache=None, parent=None):
        """
        Parameters
        ----------
        choices : list of dict or None
            The current library; each item {'name', 'path', 'params'}.
        current_path : str or None
            Path of the currently active probe; preselected if present.
        probe_cache : dict or None
            Shared {path: {'probe', 'res_m'}} cache, mutated in place as
            probes are loaded. Pass the same dict across dialog opens to
            avoid re-reading probes already loaded. A fresh dict is used
            (and discarded) if None.
        """
        super().__init__(parent)
        self.setWindowTitle('Set Probe Choice')
        self.resize(680, 760)
        self.setModal(True)

        # Working copy (deep enough that edits here don't touch the caller's
        # list until it reads get_choices()); params dicts are passed through.
        self._choices = [
            {'name': str(c['name']), 'path': str(c['path']),
             'params': dict(c.get('params') or {})}
            for c in (choices or [])
        ]
        self._cache = probe_cache if probe_cache is not None else {}
        self._sel_name = None
        self._sel_path = None

        self._build_ui()
        self._populate_list()

        found = False
        if current_path:
            for i, c in enumerate(self._choices):
                if c['path'] == current_path:
                    self.list_widget.setCurrentRow(i)   # triggers _on_row_changed
                    found = True
                    break
        if not found:
            self._on_row_changed(self.list_widget.currentRow())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Named-probe list with add/remove/rename/change-file controls
        list_row = QHBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.setToolTip('Named probe files. Select one to preview.')
        list_row.addWidget(self.list_widget, 1)

        btn_col = QVBoxLayout()
        self.add_btn = QPushButton('+')
        self.del_btn = QPushButton('−')
        for b, color in ((self.add_btn, '#2ca02c'), (self.del_btn, '#d62728')):
            b.setFixedWidth(32)
            b.setStyleSheet(f'color: white; background-color: {color}; '
                            f'font-weight: bold; font-size: 16px;')
        self.add_btn.setToolTip('Add a probe entry')
        self.del_btn.setToolTip('Remove the selected entry')
        self.rename_btn = QPushButton('Rename')
        self.change_file_btn = QPushButton('Change File')
        self.rename_btn.setToolTip('Rename the selected entry')
        self.change_file_btn.setToolTip('Point the selected entry at another file')
        btn_col.addWidget(self.add_btn)
        btn_col.addWidget(self.del_btn)
        btn_col.addSpacing(12)
        btn_col.addWidget(self.rename_btn)
        btn_col.addWidget(self.change_file_btn)
        btn_col.addStretch()
        list_row.addLayout(btn_col)
        layout.addLayout(list_row)

        # Bulk-load control: reads every not-yet-cached probe on the list.
        self.load_btn = QPushButton('Load registered probes')
        self.load_btn.setToolTip(
            'Read every probe on the list from disk (already-loaded probes '
            'are skipped). Nothing is read from disk until this is pressed.'
        )
        self.load_btn.clicked.connect(self._on_load_registered)
        layout.addWidget(self.load_btn)
        self.load_status_label = QLabel('')
        self.load_status_label.setWordWrap(True)
        layout.addWidget(self.load_status_label)

        # Image view (probe amplitude preview)
        self.image_view = pg.ImageView()
        self.image_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_view, 1)

        # Info labels
        self.path_label = QLabel('No probe selected.')
        self.path_label.setWordWrap(True)
        self.shape_label = QLabel('')
        layout.addWidget(self.path_label)
        layout.addWidget(self.shape_label)

        # Accept / Cancel
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.accept_btn = QPushButton('Accept')
        self.cancel_btn = QPushButton('Cancel')
        btn_row.addWidget(self.accept_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.add_btn.clicked.connect(self._on_add)
        self.del_btn.clicked.connect(self._on_del)
        self.rename_btn.clicked.connect(self._on_rename)
        self.change_file_btn.clicked.connect(self._on_change_file)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        self.accept_btn.clicked.connect(self._on_accept)
        self.cancel_btn.clicked.connect(self.reject)

        self.accept_btn.setEnabled(False)
        self._update_edit_buttons()

    def _populate_list(self):
        self.list_widget.clear()
        for c in self._choices:
            item = QListWidgetItem(c['name'])
            item.setToolTip(c['path'])
            self.list_widget.addItem(item)

    def _update_edit_buttons(self):
        has = self.list_widget.currentRow() >= 0
        self.del_btn.setEnabled(has)
        self.rename_btn.setEnabled(has)
        self.change_file_btn.setEnabled(has)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _load_one(self, name, path):
        """Load one probe into the shared cache. Returns an error string, or
        None on success."""
        try:
            probe, res_m = load_probe_file(path)
        except Exception as exc:
            return f'{name}: {exc}'
        self._cache[path] = {'probe': probe, 'res_m': res_m}
        return None

    def _on_load_registered(self):
        if not self._choices:
            self.load_status_label.setText('No probes registered — use "+" to add one.')
            return
        self.load_btn.setEnabled(False)
        n_loaded, n_cached, errors = 0, 0, []
        for c in self._choices:
            if c['path'] in self._cache:
                n_cached += 1
                continue
            err = self._load_one(c['name'], c['path'])
            if err is None:
                n_loaded += 1
            else:
                errors.append(err)
        self.load_btn.setEnabled(True)

        msg = f'Loaded {n_loaded} probe(s)'
        if n_cached:
            msg += f', {n_cached} already loaded'
        msg += '.'
        if errors:
            msg += f' {len(errors)} failed.'
            self.load_status_label.setToolTip('\n'.join(errors))
        else:
            self.load_status_label.setToolTip('')
        self.load_status_label.setText(msg)

        # Refresh the current row's preview now that the cache is populated.
        self._on_row_changed(self.list_widget.currentRow())

    def _on_add(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select Probe HDF5', '',
            'HDF5 files (*.h5 *.hdf5);;All files (*)'
        )
        if not path:
            return
        name, ok = QInputDialog.getText(
            self, 'Probe name', 'Name for this probe:', text=Path(path).stem
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        self._choices.append({'name': name, 'path': path, 'params': {}})
        err = self._load_one(name, path)
        if err is not None:
            self.load_status_label.setText(f'Error loading: {err}')
        item = QListWidgetItem(name)
        item.setToolTip(path)
        self.list_widget.addItem(item)
        self.list_widget.setCurrentRow(self.list_widget.count() - 1)

    def _on_del(self):
        row = self.list_widget.currentRow()
        if row < 0:
            return
        self._choices.pop(row)
        self.list_widget.takeItem(row)   # selection change → _on_row_changed

    def _on_rename(self):
        row = self.list_widget.currentRow()
        if row < 0:
            return
        name, ok = QInputDialog.getText(
            self, 'Rename probe', 'New name:', text=self._choices[row]['name']
        )
        if not ok or not name.strip():
            return
        self._choices[row]['name'] = name.strip()
        self.list_widget.item(row).setText(name.strip())

    def _on_change_file(self):
        row = self.list_widget.currentRow()
        if row < 0:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, 'Change Probe File', self._choices[row]['path'],
            'HDF5 files (*.h5 *.hdf5);;All files (*)'
        )
        if not path:
            return
        self._choices[row]['path'] = path          # name + params preserved
        self.list_widget.item(row).setToolTip(path)
        err = self._load_one(self._choices[row]['name'], path)
        if err is not None:
            self.load_status_label.setText(f'Error loading: {err}')
        self._on_row_changed(row)                  # reload preview

    def _on_row_changed(self, row):
        self._update_edit_buttons()
        if row < 0 or row >= len(self._choices):
            self._clear_selection()
            return
        name, path = self._choices[row]['name'], self._choices[row]['path']
        data = self._cache.get(path)
        if data is None:
            self._sel_name = None
            self._sel_path = None
            self.accept_btn.setEnabled(False)
            self.path_label.setText(
                f'{name}\n{path}\n(not loaded — click "Load registered probes")'
            )
            self.shape_label.setText('')
            self.image_view.clear()
            return
        self._sel_name = name
        self._sel_path = path
        self._display_probe(data['probe'], f'{name}\n{path}')
        self.accept_btn.setEnabled(True)

    def _clear_selection(self):
        self._sel_name = None
        self._sel_path = None
        self.accept_btn.setEnabled(False)
        self.path_label.setText('No probe selected.')
        self.shape_label.setText('')
        self.image_view.clear()

    def _on_accept(self):
        if self._sel_path is None:
            self.reject()
            return
        self.accept()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _display_probe(self, probe, label):
        """Show amplitude of probe[0, 0] (first OPR, first mode)."""
        self.path_label.setText(label)
        self.shape_label.setText(f'Shape: {probe.shape}   dtype: {probe.dtype}')
        amp = np.abs(probe[0, 0]).astype(np.float32)
        self.image_view.setImage(amp, autoLevels=True, autoRange=True)

    # ------------------------------------------------------------------
    # Result access
    # ------------------------------------------------------------------

    def get_choices(self):
        """Return the (possibly edited) library as a list of dicts."""
        return list(self._choices)

    def get_result(self):
        """
        Returns
        -------
        (name, path, probe, res_m) if a loaded probe was chosen and accepted,
        else (None, None, None, None). `probe`/`res_m` come straight from the
        cache, so the caller never needs to re-read the file.
        """
        if self.result() == QDialog.Accepted and self._sel_path is not None:
            data = self._cache.get(self._sel_path)
            if data is not None:
                return self._sel_name, self._sel_path, data['probe'], data['res_m']
        return None, None, None, None
