"""
ProbeViewer — modal "Setup Initial Probes and Geometry" dialog.

Opens from the main window's button of the same name (also shown once at
startup). It owns two things:

  * the named probe library
      +            add an entry (file picker + name prompt)
      −            remove the selected entry
      Rename       give the selected entry a new name
      Change File  point the selected entry at a different file (name kept)
      Shape…       set the illuminated shape + filling fraction of the entry

  * the beamline / detector geometry — energy, detector distance and beam
    centre. These used to live on the main panel; they are setup properties
    rather than per-run knobs, so they belong with the probes and travel with
    them in the exported JSON. Crop N deliberately stays on the main window.

Probe data is NOT read from disk just because an entry exists in the list or
is selected — that would mean re-reading every registered probe (or at least
the previously active one) on every dialog open. Instead, "Load registered
probes" reads every entry not already cached; loading is skipped for entries
whose data is already in the shared cache. Selecting a row shows the cached
preview if present, or a "not loaded" placeholder otherwise. Accept returns
the probe data for the highlighted (and loaded) entry directly, so the
caller never needs a second disk read.

Each entry is a dict {'name', 'path', 'enabled', 'shape', 'fill_fraction',
'params'}; the RPI parameter set in 'params' is owned by the main window and
passed through this dialog untouched, so renaming a probe or changing its file
keeps its associated parameters.

`probe_cache` is a dict {path: {'probe': ndarray, 'res_m': float}} owned by
the caller (typically the main window) and mutated in place, so probes loaded
in one dialog session stay loaded for the next.
"""

from pathlib import Path

import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QSizePolicy, QListWidget, QListWidgetItem, QInputDialog,
    QMessageBox,
)
from PyQt5.QtCore import Qt

from rpi_utils import (
    load_probe_file,
    normalize_probe_entry,
    build_config_dict,
    write_config_json,
    read_config_json,
    PROBE_SHAPES,
    DEFAULT_PROBE_SHAPE,
    DEFAULT_FILL_FRACTION,
)


def _named(widget, name):
    """setObjectName(...) and return the widget, so font_utils.apply_font_size_to_tree
    (which skips unnamed widgets) reaches everything we build here."""
    widget.setObjectName(name)
    return widget


DEFAULT_GEOMETRY = {
    'energy_kev':        12.0,
    'detector_distance': 2.0,
    'beam_center_y':     512,
    'beam_center_x':     512,
}


class ProbeViewer(QDialog):
    def __init__(self, choices=None, geometry=None, current_path=None,
                 probe_cache=None, parent=None):
        """
        Parameters
        ----------
        choices : list of dict or None
            The current library; each item {'name', 'path', 'enabled',
            'shape', 'fill_fraction', 'params'}. Partial dicts are accepted
            and filled in with defaults.
        geometry : dict or None
            {'energy_kev', 'detector_distance', 'beam_center_y',
            'beam_center_x'}. Edited in place conceptually — read the result
            back with get_geometry().
        current_path : str or None
            Path of the currently active probe; preselected if present.
        probe_cache : dict or None
            Shared {path: {'probe', 'res_m'}} cache, mutated in place as
            probes are loaded. Pass the same dict across dialog opens to
            avoid re-reading probes already loaded. A fresh dict is used
            (and discarded) if None.
        """
        super().__init__(parent)
        self.setObjectName('ProbeViewer')
        self.setWindowTitle('Setup Initial Probes and Geometry')
        self.resize(720, 860)
        self.setModal(True)

        # Working copy (deep enough that edits here don't touch the caller's
        # list until it reads get_choices()); params dicts are passed through.
        self._choices = [normalize_probe_entry(c) for c in (choices or [])]
        self._geometry = dict(DEFAULT_GEOMETRY)
        self._geometry.update(geometry or {})
        self._cache = probe_cache if probe_cache is not None else {}
        self._sel_name = None
        self._sel_path = None

        self._build_ui()
        self._populate_list()
        self._geometry_to_widgets()

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

        # Named-probe list with add/remove/rename/change-file/shape controls
        list_row = QHBoxLayout()
        self.list_widget = _named(QListWidget(), 'list_widget')
        self.list_widget.setToolTip('Named probe files. Select one to preview.')
        list_row.addWidget(self.list_widget, 1)

        btn_col = QVBoxLayout()
        self.add_btn = _named(QPushButton('+'), 'add_btn')
        self.del_btn = _named(QPushButton('−'), 'del_btn')
        for b, color in ((self.add_btn, '#2ca02c'), (self.del_btn, '#d62728')):
            b.setFixedWidth(32)
            b.setStyleSheet(f'color: white; background-color: {color}; '
                            f'font-weight: bold; font-size: 16px;')
        self.add_btn.setToolTip('Add a probe entry')
        self.del_btn.setToolTip('Remove the selected entry')
        self.rename_btn = _named(QPushButton('Rename'), 'rename_btn')
        self.change_file_btn = _named(QPushButton('Change File'), 'change_file_btn')
        self.shape_btn = _named(QPushButton('Shape / fill…'), 'shape_btn')
        self.rename_btn.setToolTip('Rename the selected entry')
        self.change_file_btn.setToolTip('Point the selected entry at another file')
        self.shape_btn.setToolTip(
            'Set the illuminated shape (circle/square) and filling fraction '
            'of the selected probe. Used for phase unwrapping, ramp removal, '
            'display zoom and object stitching.'
        )
        btn_col.addWidget(self.add_btn)
        btn_col.addWidget(self.del_btn)
        btn_col.addSpacing(12)
        btn_col.addWidget(self.rename_btn)
        btn_col.addWidget(self.change_file_btn)
        btn_col.addWidget(self.shape_btn)
        btn_col.addStretch()
        list_row.addLayout(btn_col)
        layout.addLayout(list_row)

        # Whole-config JSON export/import (separate from the per-session
        # settings.ini persistence the main window handles automatically).
        lib_io_row = QHBoxLayout()
        self.save_lib_btn = _named(QPushButton('Export config…'), 'save_lib_btn')
        self.load_lib_btn = _named(QPushButton('Import config…'), 'load_lib_btn')
        self.save_lib_btn.setToolTip(
            'Write the probe list (name, path, enabled, shape, filling '
            'fraction, per-probe RPI params) plus the beamline and detector '
            'geometry below to a JSON file. Crop N is not included.'
        )
        self.load_lib_btn.setToolTip(
            'Replace the current probe list and geometry with those from a '
            'JSON file.'
        )
        lib_io_row.addWidget(self.save_lib_btn)
        lib_io_row.addWidget(self.load_lib_btn)
        layout.addLayout(lib_io_row)
        self.save_lib_btn.clicked.connect(self._on_export_config)
        self.load_lib_btn.clicked.connect(self._on_import_config)

        # Bulk-load control: reads every not-yet-cached probe on the list.
        self.load_btn = _named(QPushButton('Load registered probes'), 'load_btn')
        self.load_btn.setToolTip(
            'Read every probe on the list from disk (already-loaded probes '
            'are skipped). Nothing is read from disk until this is pressed.'
        )
        self.load_btn.clicked.connect(self._on_load_registered)
        layout.addWidget(self.load_btn)
        self.load_status_label = _named(QLabel(''), 'load_status_label')
        self.load_status_label.setWordWrap(True)
        layout.addWidget(self.load_status_label)

        # Image view (probe amplitude preview) with the illuminated-area outline
        self.image_view = _named(pg.ImageView(), 'image_view')
        self.image_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_view, 1)
        self._shape_outline = pg.PlotDataItem(pen=pg.mkPen('r', width=2))
        self.image_view.getView().addItem(self._shape_outline)

        # Info labels
        self.path_label = _named(QLabel('No probe selected.'), 'path_label')
        self.path_label.setWordWrap(True)
        self.shape_label = _named(QLabel(''), 'shape_label')
        layout.addWidget(self.path_label)
        layout.addWidget(self.shape_label)

        # ---- Beamline / detector geometry -------------------------------
        geo_box = _named(QGroupBox('Beamline / detector geometry'), 'geo_box')
        geo_form = QFormLayout(geo_box)
        self.energy_kev_spin = _named(QDoubleSpinBox(), 'energy_kev_spin')
        self.energy_kev_spin.setDecimals(3)
        self.energy_kev_spin.setRange(0.1, 100.0)
        self.energy_kev_spin.setSingleStep(0.1)
        self.detector_dist_spin = _named(QDoubleSpinBox(), 'detector_dist_spin')
        self.detector_dist_spin.setDecimals(3)
        self.detector_dist_spin.setRange(0.01, 10.0)
        self.detector_dist_spin.setSingleStep(0.01)
        self.beam_cy_spin = _named(QSpinBox(), 'beam_cy_spin')
        self.beam_cy_spin.setMaximum(9999)
        self.beam_cx_spin = _named(QSpinBox(), 'beam_cx_spin')
        self.beam_cx_spin.setMaximum(9999)
        geo_form.addRow(_named(QLabel('Energy (keV):'), 'label_energy'),
                        self.energy_kev_spin)
        geo_form.addRow(_named(QLabel('Det. distance (m):'), 'label_detdist'),
                        self.detector_dist_spin)
        geo_form.addRow(_named(QLabel('Beam center Y:'), 'label_bcy'),
                        self.beam_cy_spin)
        geo_form.addRow(_named(QLabel('Beam center X:'), 'label_bcx'),
                        self.beam_cx_spin)
        layout.addWidget(geo_box)

        # Accept / Cancel
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.accept_btn = _named(QPushButton('Accept'), 'accept_btn')
        self.cancel_btn = _named(QPushButton('Cancel'), 'cancel_btn')
        btn_row.addWidget(self.accept_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.add_btn.clicked.connect(self._on_add)
        self.del_btn.clicked.connect(self._on_del)
        self.rename_btn.clicked.connect(self._on_rename)
        self.change_file_btn.clicked.connect(self._on_change_file)
        self.shape_btn.clicked.connect(self._on_edit_shape)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        self.accept_btn.clicked.connect(self._on_accept)
        self.cancel_btn.clicked.connect(self.reject)

        self.accept_btn.setEnabled(False)
        self._update_edit_buttons()

    def _populate_list(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for c in self._choices:
            item = QListWidgetItem(self._item_text(c))
            item.setToolTip(c['path'])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            enabled = c.get('enabled', True)
            item.setCheckState(Qt.Checked if enabled else Qt.Unchecked)
            item.setForeground(Qt.black if enabled else Qt.gray)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

    @staticmethod
    def _item_text(entry):
        return (f"{entry['name']}   [{entry['shape']} "
                f"{entry['fill_fraction']:.2f}]")

    def _update_edit_buttons(self):
        has = self.list_widget.currentRow() >= 0
        self.del_btn.setEnabled(has)
        self.rename_btn.setEnabled(has)
        self.change_file_btn.setEnabled(has)
        self.shape_btn.setEnabled(has)

    # ------------------------------------------------------------------
    # Geometry widgets
    # ------------------------------------------------------------------

    def _geometry_to_widgets(self):
        self.energy_kev_spin.setValue(float(self._geometry['energy_kev']))
        self.detector_dist_spin.setValue(float(self._geometry['detector_distance']))
        self.beam_cy_spin.setValue(int(self._geometry['beam_center_y']))
        self.beam_cx_spin.setValue(int(self._geometry['beam_center_x']))

    def _widgets_to_geometry(self):
        self._geometry = {
            'energy_kev':        self.energy_kev_spin.value(),
            'detector_distance': self.detector_dist_spin.value(),
            'beam_center_y':     self.beam_cy_spin.value(),
            'beam_center_x':     self.beam_cx_spin.value(),
        }
        return self._geometry

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
        n_loaded, n_cached, n_skipped, errors = 0, 0, 0, []
        for c in self._choices:
            if not c.get('enabled', True):
                n_skipped += 1
                continue
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
        if n_skipped:
            msg += f', {n_skipped} disabled (skipped)'
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
        entry = normalize_probe_entry({'name': name, 'path': path})
        self._choices.append(entry)
        err = self._load_one(name, path)
        if err is not None:
            self.load_status_label.setText(f'Error loading: {err}')
        item = QListWidgetItem(self._item_text(entry))
        item.setToolTip(path)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.list_widget.addItem(item)
        self.list_widget.setCurrentRow(self.list_widget.count() - 1)

    def _on_export_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Configuration', '', 'JSON files (*.json)'
        )
        if not path:
            return
        if not path.lower().endswith('.json'):
            path += '.json'
        try:
            # One flat geometry dict feeds both sections; build_config_dict
            # picks the keys each one owns.
            geo = self._widgets_to_geometry()
            write_config_json(path, build_config_dict(self._choices, geo, geo))
            self.load_status_label.setText(f'Configuration exported to {path}')
        except Exception as exc:
            QMessageBox.warning(self, 'Export failed', str(exc))

    def _on_import_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Import Configuration', '', 'JSON files (*.json)'
        )
        if not path:
            return
        try:
            probes, beamline, detector = read_config_json(path)
        except ValueError as exc:
            QMessageBox.warning(self, 'Import failed', str(exc))
            return
        if self._choices and QMessageBox.question(
            self, 'Replace configuration',
            f'Replace the current {len(self._choices)} probe(s) and the '
            f'geometry below with {len(probes)} probe(s) from file?',
        ) != QMessageBox.Yes:
            return
        self._choices = probes
        self._geometry.update(beamline)
        self._geometry.update(detector)
        self._geometry_to_widgets()
        self._populate_list()
        self._clear_selection()
        self.load_status_label.setText(
            f'Configuration imported from {path} ({len(probes)} probes).'
        )

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
        self.list_widget.item(row).setText(self._item_text(self._choices[row]))

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

    def _on_edit_shape(self):
        row = self.list_widget.currentRow()
        if not (0 <= row < len(self._choices)):
            return
        entry = self._choices[row]
        dlg = _ProbeShapeDialog(entry['shape'], entry['fill_fraction'],
                                entry['name'], self)
        if dlg.exec_() != QDialog.Accepted:
            return
        entry['shape'], entry['fill_fraction'] = dlg.get_result()
        self.list_widget.item(row).setText(self._item_text(entry))
        self._on_row_changed(row)                  # redraw the outline

    def _on_item_changed(self, item):
        """A checkbox (or, incidentally, an item's text) changed."""
        row = self.list_widget.row(item)
        if not (0 <= row < len(self._choices)):
            return
        enabled = item.checkState() == Qt.Checked
        self._choices[row]['enabled'] = enabled
        item.setForeground(Qt.black if enabled else Qt.gray)
        if row == self.list_widget.currentRow():
            self._on_row_changed(row)   # refresh preview / accept state

    def _on_row_changed(self, row):
        self._update_edit_buttons()
        if row < 0 or row >= len(self._choices):
            self._clear_selection()
            return
        entry = self._choices[row]
        name, path = entry['name'], entry['path']
        if not entry.get('enabled', True):
            self._sel_name = None
            self._sel_path = None
            self.accept_btn.setEnabled(False)
            self.path_label.setText(f'{name}\n{path}\n(disabled — cannot be used)')
            self.shape_label.setText('')
            self.image_view.clear()
            self._shape_outline.clear()
            return
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
            self._shape_outline.clear()
            return
        self._sel_name = name
        self._sel_path = path
        self._display_probe(data['probe'], f'{name}\n{path}', entry)
        self.accept_btn.setEnabled(True)

    def _clear_selection(self):
        self._sel_name = None
        self._sel_path = None
        self.accept_btn.setEnabled(False)
        self.path_label.setText('No probe selected.')
        self.shape_label.setText('')
        self.image_view.clear()
        self._shape_outline.clear()

    def _on_accept(self):
        if self._sel_path is None:
            self.reject()
            return
        self.accept()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _display_probe(self, probe, label, entry):
        """Show amplitude of probe[0, 0] (first OPR, first mode), with the
        illuminated-area outline drawn on top so the filling fraction can be
        eyeballed against the real beam."""
        self.path_label.setText(label)
        self.shape_label.setText(
            f'Shape: {probe.shape}   dtype: {probe.dtype}   '
            f'illumination: {entry["shape"]} @ {entry["fill_fraction"]:.2f}'
        )
        amp = np.abs(probe[0, 0]).astype(np.float32)
        self.image_view.setImage(amp, autoLevels=True, autoRange=True)
        self._draw_shape_outline(amp.shape[-1], entry['shape'],
                                 entry['fill_fraction'])

    def _draw_shape_outline(self, n, shape, fill_fraction):
        half = fill_fraction * n / 2.0
        c = (n - 1) / 2.0
        if shape == 'square':
            xs = [c - half, c + half, c + half, c - half, c - half]
            ys = [c - half, c - half, c + half, c + half, c - half]
        else:
            th = np.linspace(0, 2 * np.pi, 180)
            xs = c + half * np.cos(th)
            ys = c + half * np.sin(th)
        self._shape_outline.setData(x=np.asarray(xs, dtype=float),
                                    y=np.asarray(ys, dtype=float))

    # ------------------------------------------------------------------
    # Result access
    # ------------------------------------------------------------------

    def get_choices(self):
        """Return the (possibly edited) library as a list of dicts."""
        return list(self._choices)

    def get_geometry(self):
        """Return the (possibly edited) beamline/detector geometry dict.

        Read straight off the widgets, so geometry edits survive however the
        dialog was dismissed — Accept, Cancel, Esc or the window's close box.
        """
        return dict(self._widgets_to_geometry())

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


class _ProbeShapeDialog(QDialog):
    """Set one probe's illuminated shape and filling fraction.

    These drive phase unwrapping, ramp removal, the display zoom and — most
    importantly — which pixels of a reconstruction get stamped into the
    stitched object, so they are a property of the probe rather than a
    display preference.
    """

    def __init__(self, shape, fill_fraction, probe_name='', parent=None):
        super().__init__(parent)
        self.setObjectName('ProbeShapeDialog')
        self.setWindowTitle('Illuminated area')
        self.setModal(True)

        layout = QVBoxLayout(self)
        header = _named(QLabel(f'Probe: {probe_name}'), 'header_label')
        header.setWordWrap(True)
        layout.addWidget(header)

        form = QFormLayout()
        self.shape_combo = _named(QComboBox(), 'shape_combo')
        self.shape_combo.addItems(PROBE_SHAPES)
        idx = (PROBE_SHAPES.index(shape) if shape in PROBE_SHAPES
               else PROBE_SHAPES.index(DEFAULT_PROBE_SHAPE))
        self.shape_combo.setCurrentIndex(idx)

        self.fill_spin = _named(QDoubleSpinBox(), 'fill_spin')
        self.fill_spin.setDecimals(2)
        self.fill_spin.setRange(0.01, 1.0)
        self.fill_spin.setSingleStep(0.05)
        self.fill_spin.setValue(float(fill_fraction or DEFAULT_FILL_FRACTION))
        self.fill_spin.setToolTip(
            'Illuminated diameter (circle) or side (square) as a fraction of '
            'the object array width.'
        )

        form.addRow(_named(QLabel('Shape:'), 'label_shape'), self.shape_combo)
        form.addRow(_named(QLabel('Filling fraction:'), 'label_fill'), self.fill_spin)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = _named(QPushButton('OK'), 'ok_btn')
        cancel_btn = _named(QPushButton('Cancel'), 'cancel_btn')
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def get_result(self):
        return self.shape_combo.currentText(), self.fill_spin.value()
