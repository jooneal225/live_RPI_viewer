"""
Lineout panel with draggable 25%-75% and erf-fit resolution metrics, and an
optional perpendicular averaging width.

See ``pg_image_tools/__init__.py`` for usage documentation.
"""

import numpy as np
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import pyqtgraph as pg
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from scipy.special import erf


def compute_lineout(image, p1, p2, pixel_size_m, width_px=0):
    """
    Sample ``image`` along the segment p1 -> p2.

    Parameters
    ----------
    image : 2d ndarray, indexed [x, y] (pyqtgraph convention)
    p1, p2 : (x, y) tuples in image pixel coordinates
    pixel_size_m : float, metres per pixel
    width_px : float, optional
        If > 0, average across a swath of this width (in pixels) perpendicular
        to p1->p2 instead of sampling a single-pixel-wide line. Sampled at 2x
        the pixel density of the width, using a simple (unweighted) mean.

    Returns
    -------
    dist_um : 1d ndarray of distances along the segment, in µm
    values : 1d ndarray of interpolated image values
    """
    if image is None:
        return None, None
    x1, y1 = p1
    x2, y2 = p2
    n_pts = max(int(np.hypot(x2 - x1, y2 - y1)), 10)
    xs = np.linspace(x1, x2, n_pts)
    ys = np.linspace(y1, y2, n_pts)

    if width_px <= 0:
        values = map_coordinates(image, [xs, ys], order=1, mode='nearest')
    else:
        length = np.hypot(x2 - x1, y2 - y1)
        if length == 0:
            perp_x, perp_y = 0.0, 0.0
        else:
            perp_x, perp_y = -(y2 - y1) / length, (x2 - x1) / length
        offsets = np.linspace(-width_px / 2, width_px / 2, 2 * width_px + 1)
        acc = np.zeros(n_pts)
        for off in offsets:
            acc += map_coordinates(
                image, [xs + perp_x * off, ys + perp_y * off], order=1, mode='nearest'
            )
        values = acc / len(offsets)

    total_um = np.hypot((x2 - x1) * pixel_size_m * 1e6, (y2 - y1) * pixel_size_m * 1e6)
    dist_um = np.linspace(0, total_um, n_pts)
    return dist_um, values


def _erf_model(x, amp, x0, sigma, offset):
    """Error-function edge model; its derivative is a Gaussian of the given sigma."""
    return amp * erf((x - x0) / (np.sqrt(2) * sigma)) + offset


class LineoutPanel(pg.PlotWidget):
    """
    A pyqtgraph plot of an image lineout, with two draggable vertical cursors
    (blue / green) that report their positions and the value under them, plus
    right-click actions: "Find 25%-75% resolution" measures the edge width
    between them; "Find resolution (erf fit)" fits an error function to the
    same region and reports the FWHM of its derivative; "Lineout width..."
    sets a perpendicular averaging width (in pixels) for the sampled line.

    The two metrics are independent -- running one does not clear the other,
    so both the shaded 25%-75% region and the erf fit curve can be on screen
    together, each named in the legend. Dragging either cursor clears both,
    since both are fitted to the region between them.

    The panel is driven entirely by :meth:`update_lineout`; it never reads data
    itself. It is hidden by default -- call :meth:`set_lineout_visible`.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setLabel('bottom', 'Distance (µm)')
        self.setLabel('left', 'Value')
        self.setVisible(False)

        # Explicit visibility flag rather than isVisible(), so updates issued
        # before the host window is shown still take effect.
        self._visible = False

        self._dist_um = None
        self._values = None
        self._res_lines_positioned = False
        self._res_nm = None
        self._res_region = None
        self._erf_fwhm_nm = None
        self._erf_fit_curve = None
        self._lineout_width_px = 0
        self._last_image = None
        self._last_points = None
        self._last_pixel_size_m = None

        self._res_line1 = pg.InfiniteLine(
            pos=0.0, angle=90, movable=True,
            pen=pg.mkPen(color=(80, 120, 255), width=1.5, style=Qt.PenStyle.DotLine)
        )
        self._res_line2 = pg.InfiniteLine(
            pos=1.0, angle=90, movable=True,
            pen=pg.mkPen(color=(80, 220, 80), width=1.5, style=Qt.PenStyle.DotLine)
        )
        self._res_line1.sigPositionChanged.connect(self._on_res_line_moved)
        self._res_line2.sigPositionChanged.connect(self._on_res_line_moved)

        self._res_text_item = pg.TextItem(text='', color='w', anchor=(1, 0),
                                          fill=pg.mkBrush(0, 0, 0, 180))
        self._res_text_item.setFont(QFont("Courier", 9))
        self.getViewBox().sigRangeChanged.connect(self._reposition_res_text)

        # Static swatches, not tied to the live region/curve items -- they name
        # the two metrics' colours regardless of which (if either) is currently
        # computed, so they survive the clear() in update_lineout().
        self._legend = self.addLegend(offset=(-10, 10))
        self._legend.addItem(
            pg.PlotDataItem(pen=pg.mkPen(color=(160, 0, 200), width=8)),
            '25%-75% resolution',
        )
        self._legend.addItem(
            pg.PlotDataItem(pen=pg.mkPen(color=(255, 165, 0), width=2, style=Qt.PenStyle.DashLine)),
            'erf fit',
        )

        self.find_resolution_action = QtGui.QAction("Find 25%-75% resolution")
        self.find_resolution_action.triggered.connect(self.find_resolution)

        self.find_resolution_erf_action = QtGui.QAction("Find resolution (erf fit)")
        self.find_resolution_erf_action.triggered.connect(self.find_resolution_erf)

        self.lineout_width_action = QtGui.QAction("Lineout width...")
        self.lineout_width_action.triggered.connect(self._open_width_dialog)

        self.getViewBox().menu.addSeparator()
        self.getViewBox().menu.addAction(self.find_resolution_action)
        self.getViewBox().menu.addAction(self.find_resolution_erf_action)
        self.getViewBox().menu.addSeparator()
        self.getViewBox().menu.addAction(self.lineout_width_action)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def lineout_visible(self):
        return self._visible

    def set_lineout_visible(self, visible: bool):
        self._visible = bool(visible)
        self.setVisible(self._visible)

    def update_lineout(self, image, points, pixel_size_m):
        """
        Redraw from ``image`` sampled between the two entries of ``points``.

        ``points`` is a sequence of 0, 1 or 2 (x, y) tuples; with fewer than two
        the panel simply clears. No-op while the panel is hidden.
        """
        if not self._visible:
            return

        self.clear()
        self._res_nm = None       # both regions/curves already removed by clear()
        self._res_region = None
        self._erf_fwhm_nm = None
        self._erf_fit_curve = None
        # The legend's two swatches are proxies, never added via self.addItem(),
        # so clear() (which only touches self.items) leaves them untouched.

        if image is None or points is None or len(points) < 2:
            self._dist_um = None
            self._values = None
            return

        self._last_image = image
        self._last_points = points
        self._last_pixel_size_m = pixel_size_m

        dist_um, values = compute_lineout(
            image, points[0], points[1], pixel_size_m, width_px=self._lineout_width_px
        )
        if dist_um is None:
            return

        self._dist_um = dist_um
        self._values = values
        self.plot(dist_um, values, pen=pg.mkPen('w', width=1))

        marker_pen = pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine)
        self.addItem(pg.InfiniteLine(pos=dist_um[0], angle=90, pen=marker_pen))
        self.addItem(pg.InfiniteLine(pos=dist_um[-1], angle=90, pen=marker_pen))

        # Position metric lines on first use, then keep user-dragged positions
        if not self._res_lines_positioned:
            span = dist_um[-1] - dist_um[0]
            self._res_line1.setPos(dist_um[0] + span / 3)
            self._res_line2.setPos(dist_um[0] + 2 * span / 3)
            self._res_lines_positioned = True

        # Re-add after clear() — ignoreBounds keeps them out of autoscale
        self.addItem(self._res_line1, ignoreBounds=True)
        self.addItem(self._res_line2, ignoreBounds=True)
        self.addItem(self._res_text_item, ignoreBounds=True)
        self._update_res_metric()

    def find_resolution(self):
        """Measure the 25%-75% crossing width between the two cursors."""
        if self._dist_um is None or self._values is None:
            return
        xb = self._res_line1.value()
        xg = self._res_line2.value()
        yb = float(np.interp(xb, self._dist_um, self._values))
        yg = float(np.interp(xg, self._dist_um, self._values))

        lo, hi = min(xb, xg), max(xb, xg)
        mask = (self._dist_um >= lo) & (self._dist_um <= hi)
        x_slice = self._dist_um[mask]
        y_slice = self._values[mask]
        if len(x_slice) < 2:
            return

        level_25 = yb + 0.25 * (yg - yb)
        level_75 = yb + 0.75 * (yg - yb)

        # np.interp requires monotonically increasing xp — sort by y value
        if y_slice[-1] >= y_slice[0]:
            x_25 = float(np.interp(level_25, y_slice, x_slice))
            x_75 = float(np.interp(level_75, y_slice, x_slice))
        else:
            x_25 = float(np.interp(level_25, y_slice[::-1], x_slice[::-1]))
            x_75 = float(np.interp(level_75, y_slice[::-1], x_slice[::-1]))

        # Drop any previous 25%-75% result first -- _clear_2575() also nulls
        # _res_nm, so the new value has to be assigned after it, not before.
        # The erf fit, if any, is left alone: the two metrics coexist.
        self._clear_2575()

        self._res_nm = int(round(abs(x_75 - x_25) * 1e3))

        # Shade the region between the two crossings
        self._res_region = pg.LinearRegionItem(
            values=[min(x_25, x_75), max(x_25, x_75)],
            brush=pg.mkBrush(160, 0, 200, 70),
            movable=False,
        )
        self.addItem(self._res_region, ignoreBounds=True)
        self._update_res_metric()

    def find_resolution_erf(self):
        """
        Fit an error function to the region between the two cursors and report
        the resolution as the FWHM of its derivative (a Gaussian), computed
        directly from the fitted sigma. The fit curve is added to the plot
        until the lineout redraws or a cursor is moved.
        """
        if self._dist_um is None or self._values is None:
            return
        xb = self._res_line1.value()
        xg = self._res_line2.value()

        lo, hi = min(xb, xg), max(xb, xg)
        mask = (self._dist_um >= lo) & (self._dist_um <= hi)
        x_slice = self._dist_um[mask]
        y_slice = self._values[mask]
        if len(x_slice) < 4:
            return

        offset0 = (y_slice[0] + y_slice[-1]) / 2
        amp0 = (y_slice[-1] - y_slice[0]) / 2
        x0_0 = float(np.mean(x_slice))
        sigma0 = max((x_slice[-1] - x_slice[0]) / 4, 1e-6)

        try:
            popt, _ = curve_fit(
                _erf_model, x_slice, y_slice,
                p0=[amp0, x0_0, sigma0, offset0], maxfev=5000,
            )
        except Exception:
            return

        # Drop any previous erf-fit result first -- _clear_erf() also nulls
        # _erf_fwhm_nm, so the new value has to be assigned after it, not before.
        # The 25%-75% result, if any, is left alone: the two metrics coexist.
        self._clear_erf()

        sigma = abs(popt[2])
        self._erf_fwhm_nm = int(round(sigma * 2 * np.sqrt(2 * np.log(2)) * 1e3))

        x_fit = np.linspace(x_slice[0], x_slice[-1], 200)
        y_fit = _erf_model(x_fit, *popt)
        self._erf_fit_curve = pg.PlotDataItem(
            x_fit, y_fit, pen=pg.mkPen(color=(255, 165, 0), width=2, style=Qt.PenStyle.DashLine)
        )
        self.addItem(self._erf_fit_curve, ignoreBounds=True)
        self._update_res_metric()

    @property
    def resolution_nm(self):
        """Last computed 25%-75% width in nm, or None."""
        return self._res_nm

    @property
    def resolution_erf_fwhm_nm(self):
        """Last computed erf-fit FWHM resolution in nm, or None."""
        return self._erf_fwhm_nm

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _on_res_line_moved(self):
        """Called when a metric line is dragged — both metrics were fitted to
        the old region between them, so both are cleared."""
        self._clear_2575()
        self._clear_erf()
        self._update_res_metric()

    def _clear_2575(self):
        if self._res_region is not None:
            try:
                self.removeItem(self._res_region)
            except Exception:
                pass
            self._res_region = None
        self._res_nm = None

    def _clear_erf(self):
        if self._erf_fit_curve is not None:
            try:
                self.removeItem(self._erf_fit_curve)
            except Exception:
                pass
            self._erf_fit_curve = None
        self._erf_fwhm_nm = None

    def _open_width_dialog(self):
        """Popup: choose a perpendicular averaging width (pixels) for the lineout."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Lineout width")
        layout = QtWidgets.QVBoxLayout(dialog)

        form = QtWidgets.QFormLayout()
        combo = QtWidgets.QComboBox()
        combo.addItems([str(i) for i in range(101)])
        combo.setCurrentText(str(self._lineout_width_px))
        form.addRow("Width (pixels):", combo)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._lineout_width_px = int(combo.currentText())
            if self._last_image is not None and self._last_points is not None:
                self.update_lineout(self._last_image, self._last_points, self._last_pixel_size_m)

    def _update_res_metric(self):
        if self._dist_um is None or self._values is None:
            return
        xb = self._res_line1.value()
        xg = self._res_line2.value()
        yb = float(np.interp(xb, self._dist_um, self._values))
        yg = float(np.interp(xg, self._dist_um, self._values))
        delta_x = abs(xg - xb)
        lines = [
            f"Blue  X: {xb:.3f} µm",
            f"Blue  Y: {yb:.4g}",
            f"Green X: {xg:.3f} µm",
            f"Green Y: {yg:.4g}",
            f"ΔX:      {delta_x:.3f} µm",
        ]
        if self._res_nm is not None:
            lines.append(f"Resolution: {self._res_nm} nm")
        if self._erf_fwhm_nm is not None:
            lines.append(f"FWHM (erf fit): {self._erf_fwhm_nm} nm")
        self._res_text_item.setText("\n".join(lines))
        self._reposition_res_text()

    def _reposition_res_text(self):
        vr = self.getViewBox().viewRange()
        x_min, x_max = vr[0]
        y_min, y_max = vr[1]
        mx = (x_max - x_min) * 0.01
        my = (y_max - y_min) * 0.02
        self._res_text_item.setPos(x_max - mx, y_max - my)
