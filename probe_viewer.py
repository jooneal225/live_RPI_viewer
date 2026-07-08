"""
ProbeViewer — modal dialog to manage a library of named probe choices.

Opens from the main window's "Set Probe Choice" button. The user maintains a
named list of probe files: add with the green +, remove with the red −.
Selecting an entry loads that file and previews |probe[0, 0]| (first OPR mode,
first coherent mode). Accept returns the chosen (name, path); the caller then
loads that probe and fills the probe-file / probe-name fields.

The list itself is owned and persisted by the main window; this dialog edits a
working copy and hands it back via get_choices().
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
        choices : list of (name, path) or None
            The current named-probe library.
        current_path : str or None
            Path of the currently active probe; preselected if present.
        """
        super().__init__(parent)
        self.setWindowTitle('Set Probe Choice')
        self.resize(680, 720)
        self.setModal(True)

        # Working copy of the (name, path) library.
        self._choices = [(str(n), str(p)) for n, p in (choices or [])]
        self._sel_name = None
        self._sel_path = None

        self._build_ui()
        self._populate_list()

        # Preselect the entry matching the currently active probe, if any.
        if current_path:
            for i, (_n, p) in enumerate(self._choices):
                if p == current_path:
                    self.list_widget.setCurrentRow(i)
                    break

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Named-probe list with +/- buttons
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
        btn_col.addWidget(self.add_btn)
        btn_col.addWidget(self.del_btn)
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
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        self.accept_btn.clicked.connect(self._on_accept)
        self.cancel_btn.clicked.connect(self.reject)

        # Accept is enabled only when a valid entry is selected.
        self.accept_btn.setEnabled(False)

    def _populate_list(self):
        self.list_widget.clear()
        for name, path in self._choices:
            item = QListWidgetItem(name)
            item.setToolTip(path)
            self.list_widget.addItem(item)

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
        name = name.strip()
        self._choices.append((name, path))
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

    def _on_row_changed(self, row):
        if row < 0 or row >= len(self._choices):
            self._clear_selection()
            return
        name, path = self._choices[row]
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
        """Return the (possibly edited) library as a list of (name, path)."""
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
