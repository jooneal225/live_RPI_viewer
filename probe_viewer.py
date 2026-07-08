"""
ProbeViewer — modal dialog to manage a library of named probe choices.

Opens from the main window's "Set Probe Choice" button. The user maintains a
named list of probe files:
  +            add an entry (file picker + name prompt)
  −            remove the selected entry
  Rename       give the selected entry a new name
  Change File  point the selected entry at a different file (name kept)

Selecting an entry loads that file and previews |probe[0, 0]| (first OPR mode,
first coherent mode). Accept returns the chosen (name, path); the caller then
loads that probe and fills the probe-file / probe-name fields.

Each entry is a dict {'name', 'path', 'params'}; the RPI parameter set in
'params' is owned by the main window and passed through this dialog untouched,
so renaming a probe or changing its file keeps its associated parameters.
"""

from pathlib import Path

import numpy as np
import h5py
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSizePolicy, QListWidget, QListWidgetItem, QInputDialog,
)
from PyQt5.QtCore import Qt


class ProbeViewer(QDialog):
    def __init__(self, choices=None, current_path=None, parent=None):
        """
        Parameters
        ----------
        choices : list of dict or None
            The current library; each item {'name', 'path', 'params'}.
        current_path : str or None
            Path of the currently active probe; preselected if present.
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
        self._sel_name = None
        self._sel_path = None

        self._build_ui()
        self._populate_list()

        if current_path:
            for i, c in enumerate(self._choices):
                if c['path'] == current_path:
                    self.list_widget.setCurrentRow(i)
                    break

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
        self._choices.append({'name': name.strip(), 'path': path, 'params': {}})
        item = QListWidgetItem(name.strip())
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
        self._on_row_changed(row)                  # reload preview

    def _on_row_changed(self, row):
        self._update_edit_buttons()
        if row < 0 or row >= len(self._choices):
            self._clear_selection()
            return
        name, path = self._choices[row]['name'], self._choices[row]['path']
        try:
            with h5py.File(path, 'r') as f:
                probe = f['probe'][()]
        except Exception as exc:
            self.path_label.setText(f'Error loading {path}:\n{exc}')
            self.shape_label.setText('')
            self._clear_selection()
            return
        self._sel_name = name
        self._sel_path = path
        self._display_probe(probe, f'{name}\n{path}')
        self.accept_btn.setEnabled(True)

    def _clear_selection(self):
        self._sel_name = None
        self._sel_path = None
        self.accept_btn.setEnabled(False)

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
        (name, path) if a probe was chosen and accepted, else (None, None).
        """
        if self.result() == QDialog.Accepted and self._sel_path is not None:
            return self._sel_name, self._sel_path
        return None, None
