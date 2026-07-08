"""
ProbeViewer — modal dialog for inspecting and replacing the probe.

Opens from the main window's "Change Probe…" button.
Shows |probe[0, 0]| (first OPR mode, first coherent mode).
A new probe file can be loaded; clicking Accept returns it to the caller.
"""

import numpy as np
import h5py
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QSizePolicy,
)
from PyQt5.QtCore import Qt


class ProbeViewer(QDialog):
    def __init__(self, probe=None, probe_path=None, parent=None):
        """
        Parameters
        ----------
        probe : ndarray or None
            Initial probe array to display, shape (N_opr, N_modes, N, N).
        probe_path : str or None
            Path that the current probe was loaded from (informational).
        """
        super().__init__(parent)
        self.setWindowTitle('Probe Viewer')
        self.resize(620, 660)
        self.setModal(True)

        self._probe = probe            # currently displayed probe
        self._new_probe = None         # probe accepted by the user
        self._new_probe_path = None

        self._build_ui()

        if probe is not None:
            label = probe_path if probe_path else 'current probe'
            self._display_probe(probe, label)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Image view
        self.image_view = pg.ImageView()
        self.image_view.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        layout.addWidget(self.image_view)

        # Info labels
        self.path_label = QLabel('No probe loaded.')
        self.path_label.setWordWrap(True)
        self.shape_label = QLabel('')
        layout.addWidget(self.path_label)
        layout.addWidget(self.shape_label)

        # Buttons
        btn_row = QHBoxLayout()
        self.load_btn   = QPushButton('Load New Probe…')
        self.accept_btn = QPushButton('Accept')
        self.cancel_btn = QPushButton('Cancel')
        btn_row.addWidget(self.load_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.accept_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.load_btn.clicked.connect(self._on_load)
        self.accept_btn.clicked.connect(self._on_accept)
        self.cancel_btn.clicked.connect(self.reject)

        # Accept is disabled until a new probe is loaded
        self.accept_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Probe HDF5', '', 'HDF5 files (*.h5 *.hdf5);;All files (*)'
        )
        if not path:
            return
        try:
            with h5py.File(path, 'r') as f:
                probe = f['probe'][()]
        except Exception as exc:
            self.path_label.setText(f'Error loading file: {exc}')
            return

        self._new_probe = probe
        self._new_probe_path = path
        self.accept_btn.setEnabled(True)
        self._display_probe(probe, path)

    def _on_accept(self):
        # If user clicks Accept without loading anything new, do nothing
        if self._new_probe is None:
            self.reject()
            return
        self.accept()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _display_probe(self, probe, label):
        """Show amplitude of probe[0, 0] (first OPR, first mode)."""
        self.path_label.setText(label)
        self.shape_label.setText(
            f'Shape: {probe.shape}   dtype: {probe.dtype}'
        )
        amp = np.abs(probe[0, 0]).astype(np.float32)
        self.image_view.setImage(amp, autoLevels=True, autoRange=True)

    # ------------------------------------------------------------------
    # Result access
    # ------------------------------------------------------------------

    def get_result(self):
        """
        Returns
        -------
        (probe_array, probe_path) if a new probe was accepted,
        (None, None) otherwise.
        """
        if self.result() == QDialog.Accepted and self._new_probe is not None:
            return self._new_probe, self._new_probe_path
        return None, None
