"""
ImagePlotWidget -- a pyqtgraph ImageView with measurement, readout, lineout and
a populated right-click menu.

See ``pg_image_tools/__init__.py`` for usage documentation.
"""

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import pyqtgraph as pg
from scipy.ndimage import median_filter, gaussian_filter

from ._lineout import LineoutPanel

# Every image here is indexed image[x, y]. Pinned per item so that a host that
# sets pg.setConfigOptions(imageAxisOrder='row-major') globally (as
# rpi_live_gui does) can't flip that out from under us; such hosts pass arr.T.
AXIS_ORDER = 'col-major'

# The unwrap-phase ROI's initial extent, as a fraction of the displayed image
ROI_INITIAL_FRAC = 0.5

UNWRAP_CHOICES = ["Entire image", "Rectangular region", "Circular region"]
UNWRAP_KIND_BY_CHOICE = {
    "Entire image": 'full',
    "Rectangular region": 'rect',
    "Circular region": 'circle',
}
UNWRAP_CHOICE_BY_KIND = {v: k for k, v in UNWRAP_KIND_BY_CHOICE.items()}


class ImagePlotWidget(QtWidgets.QWidget):
    """
    Interactive 2D image viewer built on ``pyqtgraph.ImageView``.

    pyqtgraph is the only hard requirement. Every companion widget (info label,
    title label, transpose checkbox, log checkbox) is optional: supply it and it
    is wired up, omit it and that feature is silently skipped.
    """

    #: Emitted after a display option (filter / transpose / log) has been applied
    #: and the cached array re-rendered. A notification, not a request for data.
    sigRedrawRequested = QtCore.pyqtSignal()

    #: Emitted whenever the measurement points change; carries a list of 0, 1 or
    #: 2 (x, y) tuples in image pixel coordinates.
    sigPointsChanged = QtCore.pyqtSignal(list)

    #: Emitted while the cursor is over the image; (x, y) in image pixel coords.
    sigMouseMovedImage = QtCore.pyqtSignal(float, float)

    def __init__(self, container=None, *,
                 info_label=None,
                 title_label=None,
                 transpose_checkbox=None,
                 log_checkbox=None,
                 enable_lineout=True,
                 enable_measure=True,
                 enable_filters=True,
                 enable_compare=True,
                 keep_points=False,
                 title_elide_width=500,
                 parent=None):
        super().__init__(parent)

        self._enable_measure = enable_measure
        self._enable_filters = enable_filters
        self._enable_compare = enable_compare
        self._keep_points = keep_points
        self._title_elide_width = title_elide_width
        self._compare_window = None        # created on first use, kept alive here

        self._info_label = None
        self._title_label = None
        self._transpose_checkbox = None
        self._log_checkbox = None

        # ---- display state ----
        self._raw_data = None
        self._pixel_size_m = 1.0
        self._displayed_shape = None
        self._info_base_text = ''
        self._colorbar_added = False
        self._measure_clicks = []          # up to 2 (x, y) pixel-space coords
        self._active_filter = None         # 'median' | 'gaussian' | None
        self._filter_kernel = 3.0

        # ---- unwrap-phase state ----
        self._unwrap_enabled = False       # mirrors action_unwrap_phase.isChecked()
        self._unwrap_kind = None           # 'full' | 'rect' | 'circle' | None -- remembered
                                            # across toggles so the dialog can default to it
        self._unwrap_roi = None            # pg.RectROI | pg.CircleROI | None ('rect'/'circle' only)
        self._unwrap_mask = None           # bool mask (displayed coords) from the last Apply
                                            # click; None means "pending" -- ROI shown, not applied
        self._unwrap_overlay = None        # floating Apply/Cancel buttons

        # ---- widgets ----
        self.pg_view = pg.ImageView()
        self.pg_view.getImageItem().setOpts(axisOrder=AXIS_ORDER)
        self.lineout = LineoutPanel() if enable_lineout else None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if self.lineout is not None:
            splitter = QtWidgets.QSplitter(Qt.Orientation.Vertical)
            splitter.addWidget(self.pg_view)
            splitter.addWidget(self.lineout)
            splitter.setSizes([700, 200])
            layout.addWidget(splitter)
            self.splitter = splitter
        else:
            layout.addWidget(self.pg_view)
            self.splitter = None

        self._build_overlays()
        self._build_menu()

        # ---- mouse signals ----
        self._mouse_move_proxy = pg.SignalProxy(
            self.pg_view.scene.sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )
        self.pg_view.scene.sigMouseClicked.connect(self._on_mouse_clicked)
        # Keeps the floating unwrap-phase Apply/Cancel overlay anchored on resize.
        self.pg_view.installEventFilter(self)

        # ---- optional companions ----
        self.attach_info_label(info_label)
        self.attach_title_label(title_label)
        self.attach_transpose_checkbox(transpose_checkbox)
        self.attach_log_checkbox(log_checkbox)

        if container is not None:
            self.install_into(container)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def install_into(self, container: QtWidgets.QWidget):
        """Fill ``container`` with this widget, adding a zero-margin layout if needed."""
        layout = container.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self)

    def _build_overlays(self):
        view = self.pg_view.getView()

        # --- Measurement overlay ---
        self._measure_scatter = pg.ScatterPlotItem(
            size=12, pen=pg.mkPen('r', width=2), brush=pg.mkBrush(None), symbol='+'
        )
        view.addItem(self._measure_scatter)

        self._measure_line = pg.PlotCurveItem(
            pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine)
        )
        view.addItem(self._measure_line)

        self._distance_text = pg.TextItem(
            color=(0, 0, 0), anchor=(0.5, 0.5),
            fill=pg.mkBrush(255, 255, 255, 220),
        )
        _font = QFont()
        _font.setPointSize(10)
        self._distance_text.textItem.setFont(_font)
        view.addItem(self._distance_text)
        self._distance_text.setVisible(False)

        # --- Generic scatter overlay (host-driven, e.g. scan positions) ---
        self._scatter_overlay = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 180)
        )
        self._scatter_overlay.setVisible(False)
        view.addItem(self._scatter_overlay)

    def _build_menu(self):
        """
        Populate the ViewBox right-click menu with this widget's own block.

        ``_menu_anchor`` is the first action we add; host actions requested with
        ``at_top=True`` are inserted before it, i.e. above our block but below
        pyqtgraph's built-in entries.
        """
        self._menu = self.pg_view.getView().menu
        self._menu_anchor = None

        self._own_add_separator()
        self.action_copy_image = QtGui.QAction("Copy Image")
        self.action_copy_image.triggered.connect(self.copy_image_to_clipboard)
        self._own_add_action(self.action_copy_image)

        # Zoom-on-image-change behavior is a three-way switch -- exactly one of
        # these is active at a time (see set_image()) -- set off from the rest
        # of the menu by a separator on each side.
        self._own_add_separator()

        self.action_auto_reset_zoom = QtGui.QAction("Auto-reset Zoom")
        self.action_auto_reset_zoom.setCheckable(True)
        self._own_add_action(self.action_auto_reset_zoom)

        # When checked, the view range from just before an image change is
        # remapped onto the new image using the ratio of pixel sizes, so the
        # same physical field of view (position + zoom) stays framed even if
        # the new image has a different pixel size and/or shape.
        self.action_track_zoom = QtGui.QAction("Track Zoom")
        self.action_track_zoom.setCheckable(True)
        self._own_add_action(self.action_track_zoom)

        self.action_no_zoom_tracking = QtGui.QAction("Neither")
        self.action_no_zoom_tracking.setCheckable(True)
        self._own_add_action(self.action_no_zoom_tracking)

        self._zoom_mode_group = QtGui.QActionGroup(self)
        self._zoom_mode_group.setExclusive(True)
        for action in (self.action_auto_reset_zoom, self.action_track_zoom,
                       self.action_no_zoom_tracking):
            self._zoom_mode_group.addAction(action)
        self.action_auto_reset_zoom.setChecked(True)

        self.action_zoom_mode_separator = self._own_add_separator()

        self.action_auto_colorscale = QtGui.QAction("Auto colorscale")
        self.action_auto_colorscale.setCheckable(True)
        self.action_auto_colorscale.setChecked(True)
        self.action_auto_colorscale.triggered.connect(self._on_auto_colorscale_toggled)
        self._own_add_action(self.action_auto_colorscale)

        self.action_lineout = None
        if self.lineout is not None:
            self.action_lineout = QtGui.QAction("Plot Lineout")
            self.action_lineout.setCheckable(True)
            self.action_lineout.triggered.connect(self.set_lineout_visible)
            self._own_add_action(self.action_lineout)

        self.action_compare = None
        if self._enable_compare:
            self.action_compare = QtGui.QAction("Open comparison window")
            self.action_compare.triggered.connect(self.open_comparison_window)
            self._own_add_action(self.action_compare)

        self.action_median_filter = None
        self.action_gaussian_filter = None

        # "Analyze" holds filters (gated by enable_filters) and Unwrap Phase
        # (always available -- it isn't a display filter, it's independent).
        self._own_add_separator()
        self._analyze_menu = QtWidgets.QMenu("Analyze")
        self._own_add_menu(self._analyze_menu)

        if self._enable_filters:
            self.action_median_filter = QtGui.QAction("Median Filter")
            self.action_median_filter.setCheckable(True)
            self.action_median_filter.triggered.connect(
                lambda checked: self.set_filter('median', checked)
            )
            self._analyze_menu.addAction(self.action_median_filter)

            self.action_gaussian_filter = QtGui.QAction("Gaussian Filter")
            self.action_gaussian_filter.setCheckable(True)
            self.action_gaussian_filter.triggered.connect(
                lambda checked: self.set_filter('gaussian', checked)
            )
            self._analyze_menu.addAction(self.action_gaussian_filter)

        self.action_unwrap_phase = QtGui.QAction("Unwrap Phase")
        self.action_unwrap_phase.setCheckable(True)
        self.action_unwrap_phase.triggered.connect(self._on_unwrap_toggled)
        self._analyze_menu.addAction(self.action_unwrap_phase)

    def _note_anchor(self, action):
        if self._menu_anchor is None:
            self._menu_anchor = action
        return action

    def _own_add_action(self, action):
        self._menu.addAction(action)
        return self._note_anchor(action)

    def _own_add_separator(self):
        return self._note_anchor(self._menu.addSeparator())

    def _own_add_menu(self, menu):
        return self._note_anchor(self._menu.addMenu(menu))

    # ------------------------------------------------------------------
    # menu extension API
    # ------------------------------------------------------------------

    def add_menu_action(self, text, callback=None, checkable=False, checked=False,
                        at_top=False, after=None):
        """
        Add a host action to the image's right-click menu and return the QAction.

        at_top : insert above this widget's own block (below pyqtgraph's builtins).
                 Successive at_top calls keep their relative order.
        after  : insert directly after the given QAction (e.g. ``action_no_zoom_tracking``).
                 Takes precedence over ``at_top``.
        """
        action = QtGui.QAction(text, self)
        if checkable:
            action.setCheckable(True)
            action.setChecked(checked)
        if callback is not None:
            action.triggered.connect(callback)
        self._insert_menu_entry(action, at_top=at_top, after=after)
        return action

    def add_menu_separator(self, at_top=False, after=None):
        """Add a separator to the image's right-click menu; returns the QAction."""
        separator = QtGui.QAction(self)
        separator.setSeparator(True)
        self._insert_menu_entry(separator, at_top=at_top, after=after)
        return separator

    def add_menu_submenu(self, menu, at_top=False, after=None):
        """Add a QMenu to the image's right-click menu; returns its QAction."""
        self._insert_menu_entry(menu.menuAction(), at_top=at_top, after=after)
        return menu.menuAction()

    def _insert_menu_entry(self, action, at_top=False, after=None):
        actions = self._menu.actions()
        if after is not None and after in actions:
            idx = actions.index(after) + 1
            if idx < len(actions):
                self._menu.insertAction(actions[idx], action)
            else:
                self._menu.addAction(action)
        elif at_top and self._menu_anchor is not None:
            self._menu.insertAction(self._menu_anchor, action)
        else:
            self._menu.addAction(action)

    # ------------------------------------------------------------------
    # optional companion widgets
    # ------------------------------------------------------------------

    def attach_info_label(self, label):
        """
        QLabel that shows image dimensions plus the live cursor readout.

        Whatever the label already says is left alone until the first image.
        """
        self._info_label = label
        if label is not None and self._raw_data is not None:
            label.setText(self._info_base_text)

    def attach_title_label(self, label):
        """QLabel used by :meth:`set_title` (text is elided, full text as tooltip)."""
        self._title_label = label

    def attach_transpose_checkbox(self, checkbox):
        """QCheckBox that transposes the image on display. Live: toggling redraws."""
        self._transpose_checkbox = self._rebind(
            self._transpose_checkbox, checkbox, self._on_transpose_toggled
        )

    def attach_log_checkbox(self, checkbox):
        """QCheckBox that displays log10(|data|). Live: toggling redraws."""
        self._log_checkbox = self._rebind(
            self._log_checkbox, checkbox, self._on_log_toggled
        )

    @staticmethod
    def _rebind(old, new, slot):
        if old is not None:
            try:
                old.stateChanged.disconnect(slot)
            except TypeError:
                pass
        if new is not None:
            new.stateChanged.connect(slot)
        return new

    def _on_transpose_toggled(self, *_):
        # Transposing changes the coordinate system, so pixel-space overlays go stale.
        self.redraw(clear_overlays=True)
        self.sigRedrawRequested.emit()

    def _on_log_toggled(self, *_):
        self.redraw()
        self.sigRedrawRequested.emit()

    def _on_auto_colorscale_toggled(self, *_):
        # Re-checking it should snap the colorbar back to the current data now,
        # rather than waiting for the next image.
        self.redraw()
        self.sigRedrawRequested.emit()

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------

    def set_image(self, data, pixel_size_m=1.0, autoRange=None):
        """
        Display a 2D array and cache it for later redraws.

        pixel_size_m : metres per pixel; drives every µm / nm readout.
        autoRange    : None means "use the Auto-reset Zoom menu action".
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("set_image expects a 2D numpy array")

        # Captured from the outgoing image, before anything below overwrites
        # it, so the tracked view reflects wherever the user last left it
        # (not wherever it was when "Track Zoom" was checked).
        track_zoom = self.action_track_zoom.isChecked() and self._displayed_shape is not None
        if track_zoom:
            tracked_range = self.pg_view.getView().viewRange()
            tracked_pixel_size = self._pixel_size_m
            tracked_shape = self._displayed_shape

        same_shape = (self._displayed_shape is not None
                      and self._compute_displayed_shape(data) == self._displayed_shape)

        self._raw_data = data
        self._pixel_size_m = float(pixel_size_m)

        # A red ROI that's still on screen from a previous image is applied
        # again automatically, at its current position/size, against the new
        # data -- no Apply click needed. (Apply is only for confirming a
        # freshly-placed ROI, or a drag/resize of an existing one.)
        if self._unwrap_enabled and self._unwrap_kind in ('rect', 'circle') and self._unwrap_roi is not None:
            self._displayed_shape = self._compute_displayed_shape(data)
            self._unwrap_mask = self._roi_mask(self._unwrap_roi, self._unwrap_kind)

        # Measurement coords are image-specific -- unless keep_points is set
        # and the new image has the same shape, as on a live feed where a
        # fresh frame arrives every iteration.
        if not (self._keep_points and same_shape):
            self.clear_measure_points(update_lineout=False)
        self.clear_scatter_overlay()

        if autoRange is None:
            autoRange = self.action_auto_reset_zoom.isChecked() and not track_zoom
        self._render(autoRange=autoRange)   # sets self._displayed_shape for the new data
        self._update_lineout()

        if track_zoom:
            self._apply_tracked_zoom(tracked_range, tracked_pixel_size, tracked_shape)

        if (self._unwrap_enabled and self._unwrap_kind in ('rect', 'circle')
                and self._unwrap_roi is None):
            # No ROI survived (e.g. a clear_image() in between) -- place a
            # fresh one and wait for the user to confirm it with Apply.
            self._place_unwrap_roi()
            self._show_unwrap_overlay()

    def clear_image(self):
        """
        Blank the view: drop the cached array and every overlay.

        For hosts that need an empty plot rather than a stale one, e.g. when
        the file a selection points at cannot be found. A later ``set_image()``
        brings the view back as usual.
        """
        self._remove_unwrap_roi()
        if self._unwrap_overlay is not None:
            self._unwrap_overlay.hide()
        self._unwrap_mask = None
        self._raw_data = None
        self._displayed_shape = None
        self._info_base_text = ''
        self.clear_measure_points(update_lineout=False)
        self.clear_scatter_overlay()
        self.pg_view.clear()
        if self._info_label is not None:
            self._info_label.setText('')
        self._update_lineout()

    def redraw(self, clear_overlays=False):
        """
        Re-render the cached array after a display-option change.

        Preserves the current zoom, measurement points, lineout and scatter
        overlay unless ``clear_overlays`` is set. No-op before the first image.
        """
        if self._raw_data is None:
            return
        if clear_overlays:
            self.clear_measure_points(update_lineout=False)
            self.clear_scatter_overlay()
        self._render(autoRange=False)
        self._update_lineout()

    def _render(self, autoRange):
        data = self._raw_data.astype(np.float32)
        if self._transpose_checked():
            data = data.T
        self._displayed_shape = data.shape   # (nx, ny) in pyqtgraph convention

        data = self._unwrap_for_render(data)

        pix_size_nm = round(self._pixel_size_m * 1e9)
        self._info_base_text = "%.2f×%.2f µm, %d nm pix" % (
            data.shape[0] * self._pixel_size_m * 1e6,
            data.shape[1] * self._pixel_size_m * 1e6,
            pix_size_nm,
        )
        if self._info_label is not None:
            self._info_label.setText(self._info_base_text)

        if self._active_filter == 'median':
            data = median_filter(data, size=max(1, int(self._filter_kernel)))
        elif self._active_filter == 'gaussian':
            data = gaussian_filter(data, sigma=self._filter_kernel)

        if self._log_checked():
            # Floor at the smallest value actually present rather than at eps:
            # detector frames are full of exact zeros, and clipping those to
            # 2e-16 buries the real signal in the top tenth of an 18-decade
            # range. Data with no zeros is unaffected.
            mag = np.abs(data)
            positive = mag[mag > 0]
            floor = float(positive.min()) if positive.size else np.finfo(np.float32).tiny
            data = np.log10(np.clip(mag, a_min=floor, a_max=None))

        # With Auto colorscale off, both the levels and the histogram's
        # zoom/pan stay wherever they are -- except on the very first image,
        # which has no previous levels to hold.
        auto_colorscale = self.action_auto_colorscale.isChecked() or not self._colorbar_added
        # ImageView's own autoRange kwarg is unreliable once a view has
        # already been panned/zoomed once, so drive the reset ourselves
        # (same reasoning as _apply_tracked_zoom driving Track Zoom itself).
        self.pg_view.setImage(data, autoLevels=auto_colorscale, autoRange=False,
                              autoHistogramRange=auto_colorscale)
        if autoRange:
            self.pg_view.getView().autoRange()

        if not self._colorbar_added:
            self.pg_view.ui.histogram.show()
            self._colorbar_added = True

        if auto_colorscale:
            self._sync_levels(data)

    def _sync_levels(self, data):
        """
        Put the color levels and the histogram on the new array's real range.

        ImageView's own autoLevels works off a subsampled min/max, so after a
        few images the histogram's range and its level region can drift out of
        agreement with what is on screen. Setting both from the true extremes
        keeps them in step.
        """
        lo, hi = float(np.nanmin(data)), float(np.nanmax(data))

        if not (np.isfinite(lo) and np.isfinite(hi)):
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                return
            lo, hi = float(finite.min()), float(finite.max())

        if hi <= lo:
            hi = lo + 1.0   # a flat image still needs a non-empty range

        self.set_levels(lo, hi)

    def set_levels(self, lo, hi):
        """
        Set the color levels and fit the histogram to them.

        For hosts that compute their own levels with Auto colorscale off:
        ``image_view.setLevels()`` alone would leave the histogram's zoom/pan
        where it was, now that it no longer follows each new image.
        """
        self.pg_view.ui.histogram.setHistogramRange(lo, hi)
        self.pg_view.setLevels(lo, hi)

    def auto_levels(self):
        """One-off: fit the levels and histogram to the image on screen."""
        if self.pg_view.image is not None:
            self._sync_levels(self.pg_view.image)

    def set_title(self, text):
        """Set the title label, eliding each line in the middle. No-op without a label."""
        if self._title_label is None:
            return
        metrics = self._title_label.fontMetrics()
        elided = "\n".join(
            metrics.elidedText(line, Qt.TextElideMode.ElideMiddle, self._title_elide_width)
            for line in text.split("\n")
        )
        self._title_label.setText(elided)
        self._title_label.setToolTip(text)

    def _compute_displayed_shape(self, data):
        """(nx, ny) ``data`` will have once displayed, without actually rendering it."""
        return data.shape[::-1] if self._transpose_checked() else data.shape

    def _transpose_checked(self):
        return self._transpose_checkbox is not None and self._transpose_checkbox.isChecked()

    def _log_checked(self):
        return self._log_checkbox is not None and self._log_checkbox.isChecked()

    def copy_image_to_clipboard(self):
        """
        Put the image on the clipboard as a PNG: native pixel resolution, with
        the current colormap and levels, oriented as on screen. No overlays.
        """
        image_item = self.pg_view.getImageItem()
        if image_item.image is None:
            return
        pixmap = image_item.getPixmap()
        if pixmap is None or pixmap.isNull():
            return
        # Row 0 of the rendered image is y = 0, which is at the top of the
        # view only when the y axis is inverted.
        view = self.pg_view.getView()
        image = pixmap.toImage().mirrored(view.xInverted(), not view.yInverted())

        png = QtCore.QBuffer()
        png.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
        image.save(png, "PNG")
        mime = QtCore.QMimeData()
        mime.setData("image/png", png.data())
        mime.setImageData(image)          # for apps that ask for a non-PNG format
        QtWidgets.QApplication.clipboard().setMimeData(mime)

    # ------------------------------------------------------------------
    # zoom
    # ------------------------------------------------------------------

    def reset_zoom(self):
        """Fit the view to the image — same as pyqtgraph's built-in "View All"."""
        self.pg_view.getView().autoRange()

    def set_view_range(self, **kwargs):
        """Passthrough to the ViewBox, e.g. ``set_view_range(xRange=(0, 10))``."""
        self.pg_view.getView().setRange(**kwargs)

    def zoom_to_left_square(self, padding=0.05):
        """Zoom to the leftmost H x H square of a wide-and-short image."""
        if self._displayed_shape is None:
            return
        ny = self._displayed_shape[1]   # vertical height H
        self.pg_view.getView().setRange(xRange=(0, ny), yRange=(0, ny), padding=padding)

    def _apply_tracked_zoom(self, old_range, old_pixel_size, old_shape):
        """
        Re-apply a view range captured on the previous image to the new one.

        Each axis is remapped relative to that image's own center (rather
        than its pixel-0 corner), then rescaled by the ratio of pixel sizes.
        A view centered on the old image is therefore still centered on the
        new one, and an off-center crop keeps the same physical offset, even
        when pixel size and/or image shape changed between the two.
        """
        if self._displayed_shape is None or not old_pixel_size:
            return
        new_shape = self._displayed_shape
        scale = old_pixel_size / self._pixel_size_m if self._pixel_size_m else 1.0

        def remap(lo, hi, old_n, new_n):
            center = (lo + hi) / 2
            half = (hi - lo) / 2
            rel_phys = (center - old_n / 2) * scale
            new_center = new_n / 2 + rel_phys
            new_half = half * scale
            return new_center - new_half, new_center + new_half

        (x0, x1), (y0, y1) = old_range
        new_x = remap(x0, x1, old_shape[0], new_shape[0])
        new_y = remap(y0, y1, old_shape[1], new_shape[1])
        self.pg_view.getView().setRange(xRange=new_x, yRange=new_y, padding=0)

    # ------------------------------------------------------------------
    # overlays
    # ------------------------------------------------------------------

    def set_scatter_overlay(self, x, y):
        """Show a scatter overlay at the given image-pixel coordinates."""
        self._scatter_overlay.setData(x=x, y=y)
        self._scatter_overlay.setVisible(True)

    def clear_scatter_overlay(self):
        self._scatter_overlay.setVisible(False)

    @property
    def measure_points(self):
        return list(self._measure_clicks)

    def clear_measure_points(self, update_lineout=True):
        self._measure_clicks = []
        self._measure_scatter.setData(x=[], y=[])
        self._measure_line.setData([], [])
        self._distance_text.setVisible(False)
        self.sigPointsChanged.emit([])
        if update_lineout:
            self._update_lineout()

    # ------------------------------------------------------------------
    # lineout
    # ------------------------------------------------------------------

    def set_lineout_visible(self, visible: bool):
        if self.lineout is None:
            return
        self.lineout.set_lineout_visible(visible)
        if self.action_lineout is not None and self.action_lineout.isChecked() != visible:
            self.action_lineout.setChecked(visible)
        self._update_lineout()

    def _update_lineout(self):
        if self.lineout is None:
            return
        img_item = self.pg_view.getImageItem()
        image = None if img_item is None else img_item.image
        self.lineout.update_lineout(image, self._measure_clicks, self._pixel_size_m)

    # ------------------------------------------------------------------
    # filters
    # ------------------------------------------------------------------

    def set_filter(self, filter_type: str, checked: bool):
        """Enable/disable a display filter, prompting for its kernel width."""
        if not checked:
            self._active_filter = None
        else:
            kernel, ok = QtWidgets.QInputDialog.getDouble(
                self, f"{filter_type.capitalize()} Filter", "Kernel width:",
                value=self._filter_kernel, min=0.1, max=500.0, decimals=1
            )
            if not ok:
                # User cancelled — revert the checkmark
                action = (self.action_median_filter if filter_type == 'median'
                          else self.action_gaussian_filter)
                if action is not None:
                    action.setChecked(False)
                return
            self._filter_kernel = kernel
            self._active_filter = filter_type
            # Uncheck the other filter
            other = (self.action_gaussian_filter if filter_type == 'median'
                     else self.action_median_filter)
            if other is not None:
                other.setChecked(False)

        # Re-render from the cached array — no reload from the host
        self.redraw()
        self.sigRedrawRequested.emit()

    # ------------------------------------------------------------------
    # unwrap phase
    # ------------------------------------------------------------------

    def _on_unwrap_toggled(self, checked: bool):
        """"Unwrap Phase" menu entry is now a toggle: checked = keep unwrapping live."""
        if checked:
            self._enable_unwrap()
        else:
            self._disable_unwrap()

    def _enable_unwrap(self):
        if self._raw_data is None:
            self._set_unwrap_checked(False)
            return

        try:
            from skimage.restoration import unwrap_phase  # noqa: F401  (availability check)
        except ImportError:
            QtWidgets.QMessageBox.warning(
                self, "Unwrap Phase",
                "Unwrap Phase needs scikit-image, which is not installed in this environment."
            )
            self._set_unwrap_checked(False)
            return

        default_index = UNWRAP_CHOICES.index(
            UNWRAP_CHOICE_BY_KIND.get(self._unwrap_kind, UNWRAP_CHOICES[0])
        )
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "Unwrap Phase", "Choose region to unwrap:",
            UNWRAP_CHOICES, default_index, False,
        )
        if not ok:
            self._set_unwrap_checked(False)
            return

        self._unwrap_kind = UNWRAP_KIND_BY_CHOICE[choice]
        self._unwrap_mask = None
        self._unwrap_enabled = True

        if self._unwrap_kind == 'full':
            self._remove_unwrap_roi()
        else:
            self._place_unwrap_roi()
            self._show_unwrap_overlay()

        self.redraw()
        self.sigRedrawRequested.emit()

    def _disable_unwrap(self):
        self._unwrap_enabled = False
        self._unwrap_mask = None
        self._remove_unwrap_roi()
        if self._unwrap_overlay is not None:
            self._unwrap_overlay.hide()
        self._set_unwrap_checked(False)

        self.redraw()
        self.sigRedrawRequested.emit()

    def _set_unwrap_checked(self, checked: bool):
        """Sync the menu checkmark without re-entering ``_on_unwrap_toggled``."""
        action = self.action_unwrap_phase
        if action is not None and action.isChecked() != checked:
            action.blockSignals(True)
            action.setChecked(checked)
            action.blockSignals(False)

    def _place_unwrap_roi(self):
        """(Re)create the draggable/resizable red ROI for the current unwrap kind."""
        self._remove_unwrap_roi()

        nx, ny = self._displayed_shape
        w = max(nx * ROI_INITIAL_FRAC, 4.0)
        h = max(ny * ROI_INITIAL_FRAC, 4.0)
        pos = [(nx - w) / 2, (ny - h) / 2]

        if self._unwrap_kind == 'rect':
            roi = pg.RectROI(pos, [w, h], pen=pg.mkPen('r', width=2), sideScalers=True)
        else:
            side = min(w, h)
            roi = pg.CircleROI([(nx - side) / 2, (ny - side) / 2], [side, side],
                                pen=pg.mkPen('r', width=2))

        roi.setZValue(10)
        self.pg_view.getView().addItem(roi)
        self._unwrap_roi = roi

    def _remove_unwrap_roi(self):
        if self._unwrap_roi is not None:
            self.pg_view.getView().removeItem(self._unwrap_roi)
            self._unwrap_roi = None

    def _apply_unwrap_roi(self):
        """Apply-button handler: unwrap only inside the ROI's *current* position/size."""
        if self._unwrap_roi is None:
            return
        self._unwrap_mask = self._roi_mask(self._unwrap_roi, self._unwrap_kind)
        self.redraw()
        self.sigRedrawRequested.emit()

    def _ensure_unwrap_overlay(self):
        if self._unwrap_overlay is not None:
            return
        overlay = QtWidgets.QWidget(self.pg_view)
        overlay.setStyleSheet(
            "QWidget { background-color: rgba(255, 255, 255, 220); "
            "border: 1px solid palette(mid); border-radius: 4px; }"
        )
        layout = QtWidgets.QHBoxLayout(overlay)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.clicked.connect(self._apply_unwrap_roi)
        btn_cancel = QtWidgets.QPushButton("Cancel")
        # Call _disable_unwrap() directly rather than action.setChecked(False):
        # setChecked() alone doesn't emit `triggered`, so it would desync the
        # checkmark from the actual (still-enabled) unwrap state.
        btn_cancel.clicked.connect(self._disable_unwrap)

        layout.addWidget(btn_apply)
        layout.addWidget(btn_cancel)
        overlay.adjustSize()
        overlay.hide()

        self._unwrap_overlay = overlay

    def _show_unwrap_overlay(self):
        self._ensure_unwrap_overlay()
        self._position_unwrap_overlay()
        self._unwrap_overlay.show()
        self._unwrap_overlay.raise_()

    def _position_unwrap_overlay(self):
        if self._unwrap_overlay is None:
            return
        # Top-left of the plot area -- top-right would sit under pyqtgraph's
        # own histogram/LUT panel, which is docked on the right of pg_view.
        self._unwrap_overlay.move(8, 8)

    def eventFilter(self, obj, event):
        if obj is self.pg_view and event.type() == QtCore.QEvent.Type.Resize:
            if self._unwrap_overlay is not None and self._unwrap_overlay.isVisible():
                self._position_unwrap_overlay()
        return super().eventFilter(obj, event)

    def _roi_mask(self, roi, kind: str):
        """Boolean mask, shape == displayed_shape, True where the ROI covers a pixel."""
        nx, ny = self._displayed_shape
        pos, size = roi.pos(), roi.size()
        x0, y0, w, h = pos.x(), pos.y(), size.x(), size.y()

        xs = np.arange(nx)[:, None]
        ys = np.arange(ny)[None, :]

        if kind == 'rect':
            return (xs >= x0) & (xs < x0 + w) & (ys >= y0) & (ys < y0 + h)

        cx, cy = x0 + w / 2, y0 + h / 2
        r = w / 2
        return (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2

    def _unwrap_for_render(self, data):
        """
        Apply the currently-enabled unwrap, if any, to already-oriented display data.

        Non-destructive -- like the filters, this only affects what gets drawn.
        ``data`` is already transposed to displayed orientation, matching the
        coordinate space ``_roi_mask`` builds masks in.
        """
        if not self._unwrap_enabled:
            return data
        try:
            from skimage.restoration import unwrap_phase
        except ImportError:
            return data   # availability was already checked (with a popup) on enable

        wrapped = np.angle(np.exp(1j * data.astype(np.float64)))   # fold to [-pi, pi)

        if self._unwrap_kind == 'full':
            return unwrap_phase(wrapped).astype(np.float32)

        mask = self._unwrap_mask
        if mask is None or mask.shape != data.shape:
            return data   # pending -- ROI shown, waiting for the Apply button

        result = unwrap_phase(np.ma.array(wrapped, mask=~mask))
        out = data.astype(np.float64)
        out[mask] = np.ma.getdata(result)[mask]
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    # comparison window
    # ------------------------------------------------------------------

    def open_comparison_window(self):
        """
        Show this widget's comparison window, creating it on first use.

        The window is non-blocking and starts empty; its own buttons pull
        snapshots out of this widget. It is kept on ``self`` so that reopening
        raises the existing window rather than making a second one, and so it
        is not garbage-collected the moment this method returns.
        """
        if self._compare_window is None:
            # Imported here so the module is only loaded by hosts that use it
            from ._compare import ImageCompareWindow
            self._compare_window = ImageCompareWindow(self, parent=self)

        self._compare_window.show()
        self._compare_window.raise_()
        self._compare_window.activateWindow()
        return self._compare_window

    @property
    def comparison_window(self):
        """The ``ImageCompareWindow``, or None if it has never been opened."""
        return self._compare_window

    # ------------------------------------------------------------------
    # mouse
    # ------------------------------------------------------------------

    def _on_mouse_moved(self, event):
        pos = event[0]   # SignalProxy wraps args in a tuple
        img_item = self.pg_view.getImageItem()
        if img_item is None or img_item.image is None:
            return
        base = self._info_base_text
        if img_item.sceneBoundingRect().contains(pos):
            pt = img_item.mapFromScene(pos)
            nx, ny = img_item.image.shape[:2]
            x_um = (pt.x() - nx / 2) * self._pixel_size_m * 1e6
            y_um = (pt.y() - ny / 2) * self._pixel_size_m * 1e6
            xi, yi = int(pt.x()), int(pt.y())
            self.sigMouseMovedImage.emit(pt.x(), pt.y())
            if self._info_label is None:
                return
            if 0 <= xi < nx and 0 <= yi < ny:
                intens = img_item.image[xi, yi]
                self._info_label.setText(
                    f"{base}\nx={x_um:.2f}, y={y_um:.2f} µm, I={intens:.3g}"
                )
            else:
                self._info_label.setText(f"{base}\nx={x_um:.2f}, y={y_um:.2f} µm")
        elif self._info_label is not None:
            self._info_label.setText(base)

    def _on_mouse_clicked(self, event):
        if not self._enable_measure:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.scenePos()
        img_item = self.pg_view.getImageItem()
        if img_item is None or img_item.image is None:
            return
        if not img_item.sceneBoundingRect().contains(pos):
            return

        pt = img_item.mapFromScene(pos)
        x, y = pt.x(), pt.y()

        if len(self._measure_clicks) == 2:
            # Third click: clear everything
            self.clear_measure_points()
            return

        self._measure_clicks.append((x, y))
        self._measure_scatter.setData(
            x=[p[0] for p in self._measure_clicks],
            y=[p[1] for p in self._measure_clicks],
        )
        self.sigPointsChanged.emit(list(self._measure_clicks))

        if len(self._measure_clicks) == 2:
            x1, y1 = self._measure_clicks[0]
            x2, y2 = self._measure_clicks[1]
            dist_um = np.sqrt(
                ((x2 - x1) * self._pixel_size_m * 1e6) ** 2 +
                ((y2 - y1) * self._pixel_size_m * 1e6) ** 2
            )
            self._measure_line.setData([x1, x2], [y1, y2])
            # Offset the label perpendicularly away from the line
            dx, dy = x2 - x1, y2 - y1
            seg_len = np.sqrt(dx ** 2 + dy ** 2)
            offset = max(seg_len * 0.07, 30)
            if seg_len > 0:
                px, py = -dy / seg_len * offset, dx / seg_len * offset
            else:
                px, py = 0, offset
            self._distance_text.setPos((x1 + x2) / 2 + px, (y1 + y2) / 2 + py)
            dist_px = int(round(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)))
            nx, ny = img_item.image.shape[:2]
            x1i, y1i = int(x1), int(y1)
            x2i, y2i = int(x2), int(y2)
            if 0 <= x1i < nx and 0 <= y1i < ny and 0 <= x2i < nx and 0 <= y2i < ny:
                I1 = img_item.image[x1i, y1i]
                I2 = img_item.image[x2i, y2i]
                delta_i_line = "\n\N{GREEK CAPITAL LETTER DELTA}I = %.2g" % (I2 - I1)
            else:
                delta_i_line = ""
            self._distance_text.setText(
                f"{dist_um:.3f} µm\n{dist_px} pix{delta_i_line}"
            )
            self._distance_text.setVisible(True)
            self._update_lineout()

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------

    @property
    def image_view(self):
        """The underlying ``pg.ImageView``."""
        return self.pg_view

    @property
    def view(self):
        """The underlying ``pg.ViewBox``."""
        return self.pg_view.getView()

    @property
    def image_item(self):
        """The underlying ``pg.ImageItem``."""
        return self.pg_view.getImageItem()

    @property
    def menu(self):
        """The ViewBox right-click ``QMenu``."""
        return self._menu

    @property
    def displayed_shape(self):
        """(nx, ny) of the displayed image, post-transpose. None before first image."""
        return self._displayed_shape

    @property
    def pixel_size_m(self):
        return self._pixel_size_m

    @property
    def raw_data(self):
        """The array last passed to :meth:`set_image`, before any display transform."""
        return self._raw_data

    @property
    def info_text(self):
        """The dimension/pixel-size string currently shown in the info label."""
        return self._info_base_text
