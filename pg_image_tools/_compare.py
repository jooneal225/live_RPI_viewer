"""
Two-image comparison window.

See ``pg_image_tools/__init__.py`` for usage documentation.
"""

import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator
import pyqtgraph as pg
from scipy.ndimage import zoom as ndi_zoom

from ._image_view import AXIS_ORDER

# Pixel size ratios closer to 1 than this are treated as identical -- resampling
# by 1.0002 costs an interpolation pass and buys nothing.
RESAMPLE_TOLERANCE = 1e-3

# Gap between the two images in side-by-side mode, as a fraction of image 1's width
SIDE_BY_SIDE_GAP_FRAC = 0.03

IMAGE2_OPACITY = 0.5

# The red box's initial extent, as a fraction of image 1
ROI_INITIAL_FRAC = 0.5

# Sub-pixel refinement of the phase-correlation peak, in 1/N of a pixel
PHASE_CORR_UPSAMPLING = 10


class _DraggableImageItem(pg.ImageItem):
    """
    An ImageItem that reports left-button drags instead of acting on them.

    It never moves itself: it emits the drag delta in view coordinates and lets
    the owner decide what that means. In this window image 2's placement is not
    simply "where you dragged it" -- in subtract mode the item holds a
    difference whose extent grows as the two images pull apart -- so placement
    has to stay with the window.

    pyqtgraph hands drag events to whichever item claimed the button during the
    preceding hover, so ``hoverEvent`` has to call ``acceptDrags`` or the
    ViewBox takes the drag as a pan instead.
    """

    #: Emitted while dragging: (delta QPointF in view coords, is_start).
    #: The delta is measured from where the drag began, not from the last event.
    sigDragged = QtCore.pyqtSignal(object, bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draggable = False
        self._drag_start_scene = None

    def hoverEvent(self, ev):
        if not ev.isExit() and self.draggable and self.image is not None:
            ev.acceptDrags(Qt.MouseButton.LeftButton)

    def mouseDragEvent(self, ev):
        if not self.draggable or ev.button() != Qt.MouseButton.LeftButton or self.image is None:
            ev.ignore()
            return
        ev.accept()

        view = self.getViewBox()
        if view is None:
            return
        if ev.isStart():
            self._drag_start_scene = ev.buttonDownScenePos()
        if self._drag_start_scene is None:
            return

        # Map through the view so the image tracks the cursor at any zoom level
        delta = (view.mapSceneToView(ev.scenePos())
                 - view.mapSceneToView(self._drag_start_scene))
        self.sigDragged.emit(QtCore.QPointF(delta), ev.isStart())


class ImageCompareWindow(QtWidgets.QWidget):
    """
    A standalone, non-blocking window for overlaying two images pulled from an
    :class:`~pg_image_tools.ImagePlotWidget`.

    It opens empty. "Set image 1" and "Set image 2" each take a snapshot of
    whatever the source widget is displaying at that moment, together with its
    pixel size. Image 1 is the ground truth: it is drawn solid at the origin and
    its pixel size defines the common grid, so image 2 is resampled onto it when
    the two differ. Image 2 is drawn semi-transparent on top and can be dragged
    into alignment.

    Each image has its own histogram, so levels and colormap are set
    independently. ``Subtract`` leaves image 1 alone and swaps the image 2 slot
    -- and only that slot -- for ``multiplier × image 2 - image 1``, taken over
    the union of the two, so image 2 stays draggable and the difference updates
    as it moves.

    ``Red box`` adds a resizable rectangle that scopes every quantitative
    feature: the overlap it marks is what gets phase-correlated by ``Align``,
    what PSNR and MSE are measured over, what ``Intensity correction`` derives
    its linear map from, and what ``Compute FSC`` transforms.
    """

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle("Image Comparison")

        self._source = source

        # Each slot is None until its button is pressed, else
        # {'data': 2d ndarray, 'pixel_size_m': float, 'label': str}
        self._slots = [None, None]
        self._grid_data = [None, None]            # slots resampled onto the common grid
        self._corrected2 = None                   # image 2's grid after intensity correction
        self._displayed = [None, None]            # what each item actually shows
        self._image2_origin = (0.0, 0.0)          # where the image 2 item is drawn
        self._offset = QtCore.QPointF(0.0, 0.0)   # image 2 drag offset, grid pixels
        self._drag_base = None                    # offset at the start of a drag
        self._vis_before_subtract = None          # restored when subtract goes off
        self._grid_pixel_size_m = 1.0
        self._resample_factor = 1.0
        self._roi_placed = False                  # the box is centred once, then left alone
        self._intensity_map = None                # (gain, offset) actually applied, or None
        self._fsc_window = None                   # kept alive here, like the compare window

        self._build_ui()
        self._rebuild()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # --- plot plus one histogram per image ---
        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw, stretch=1)

        self._plot = self._glw.addPlot(row=0, col=0)
        self._plot.setAspectLocked(True)
        self._plot.invertY(True)    # match the orientation pg.ImageView uses

        self._img1 = pg.ImageItem(axisOrder=AXIS_ORDER)
        self._img1.setZValue(0)
        self._plot.addItem(self._img1)

        self._img2 = _DraggableImageItem(axisOrder=AXIS_ORDER)
        self._img2.setZValue(1)
        self._img2.sigDragged.connect(self._on_image2_dragged)
        self._plot.addItem(self._img2)

        # Above both images, so its handles stay grabbable at any z-order. It
        # is translatable, which is deliberate: a left-drag inside the box moves
        # the box and a left-drag outside it moves image 2.
        self._roi = pg.RectROI(
            [0, 0], [100, 100], pen=pg.mkPen('r', width=2), sideScalers=True,
        )
        self._roi.setZValue(10)
        self._roi.setVisible(False)
        self._roi.sigRegionChanged.connect(self._on_roi_changed)
        self._plot.addItem(self._roi)

        self._hist1 = pg.HistogramLUTItem()
        self._hist1.setImageItem(self._img1)
        self._glw.addItem(self._hist1, row=0, col=1)

        self._hist2 = pg.HistogramLUTItem()
        self._hist2.setImageItem(self._img2)
        self._glw.addItem(self._hist2, row=0, col=2)

        self._mouse_move_proxy = pg.SignalProxy(
            self._glw.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )

        # --- readouts, each paired with the controls that act on it, to its
        # right, rather than stacking label rows above button rows ---
        info_row = QtWidgets.QHBoxLayout()
        self._info_label = QtWidgets.QLabel()
        self._info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info_row.addWidget(self._info_label, stretch=1)

        self._btn_set1 = QtWidgets.QPushButton("Set image 1")
        self._btn_set1.setToolTip("Copy what the source plot is showing into image 1 (ground truth)")
        self._btn_set1.clicked.connect(lambda: self.capture(0))
        self._btn_set2 = QtWidgets.QPushButton("Set image 2")
        self._btn_set2.setToolTip("Copy what the source plot is showing into image 2 (overlay)")
        self._btn_set2.clicked.connect(lambda: self.capture(1))
        self._btn_reset = QtWidgets.QPushButton("Reset offset")
        self._btn_reset.clicked.connect(self.reset_offset)
        info_row.addWidget(self._btn_set1)
        info_row.addWidget(self._btn_set2)
        info_row.addWidget(self._btn_reset)
        layout.addLayout(info_row)

        # --- cursor readout, alongside display and subtract controls ---
        cursor_row = QtWidgets.QHBoxLayout()
        self._cursor_label = QtWidgets.QLabel()
        self._cursor_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cursor_row.addWidget(self._cursor_label, stretch=1)

        self._chk_side_by_side = QtWidgets.QCheckBox("Side-by-side comparison")
        self._chk_side_by_side.setToolTip(
            "Draw both images solid, placed horizontally adjacent instead of overlaid"
        )
        # Not wired in the loop below: it has to lock out side-by-side and
        # re-level image 2, whose range changes completely as a difference.
        self._chk_subtract = QtWidgets.QCheckBox("Subtract")
        self._chk_subtract.setToolTip(
            "Leave image 1 alone and plot (multiplier × image 2) - image 1 in the\n"
            "image 2 slot, over the union of the two. Image 2 stays draggable."
        )
        self._chk_subtract.toggled.connect(self._on_subtract_toggled)
        self._chk_show_both = QtWidgets.QCheckBox("Show both images")
        self._chk_show_both.setChecked(True)
        self._chk_show_both.setToolTip(
            "Overlay both, image 2 semi-transparent.\n"
            "Uncheck to look at one image at a time, fully opaque."
        )
        self._chk_show_only1 = QtWidgets.QCheckBox("Show only image 1")
        self._chk_show_only1.setChecked(True)
        self._chk_show_only1.setToolTip(
            "Which image is the one shown while \"Show both images\" is off.\n"
            "Unchecked shows image 2 instead."
        )
        for chk in (self._chk_side_by_side, self._chk_subtract,
                    self._chk_show_both, self._chk_show_only1):
            cursor_row.addWidget(chk)
        for chk in (self._chk_side_by_side, self._chk_show_both, self._chk_show_only1):
            chk.toggled.connect(self._rebuild)

        cursor_row.addSpacing(12)
        cursor_row.addWidget(QtWidgets.QLabel("multiplier"))
        self._edit_multiplier = QtWidgets.QLineEdit("1.0")
        self._edit_multiplier.setToolTip("Image 2 is scaled by this before the subtraction")
        self._edit_multiplier.setValidator(QDoubleValidator())
        self._edit_multiplier.setMaximumWidth(80)
        self._edit_multiplier.editingFinished.connect(self._on_multiplier_changed)
        cursor_row.addWidget(self._edit_multiplier)
        layout.addLayout(cursor_row)

        # --- metrics readout, alongside the comparison-region controls it scopes ---
        metrics_row = QtWidgets.QHBoxLayout()
        self._metrics_label = QtWidgets.QLabel()
        self._metrics_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        metrics_row.addWidget(self._metrics_label, stretch=1)

        self._chk_roi = QtWidgets.QCheckBox("select comparison region")
        self._chk_roi.setToolTip(
            "A resizable region of interest. Every measurement below is taken\n"
            "over the part of it where both images overlap.\n"
            "Drag inside the box to move it, its handles to resize it; drag\n"
            "outside it to go on moving image 2."
        )
        self._chk_roi.toggled.connect(self._on_roi_enabled)
        metrics_row.addWidget(self._chk_roi)

        self._btn_align = QtWidgets.QPushButton("Align (phase correlation)")
        self._btn_align.setToolTip(
            "Phase-correlate the two images over the comparison region and move\n"
            "image 2 onto image 1, to 1/%d of a pixel." % PHASE_CORR_UPSAMPLING
        )
        self._btn_align.clicked.connect(self.align_phase_correlation)
        metrics_row.addWidget(self._btn_align)

        self._chk_intensity = QtWidgets.QCheckBox("Intensity correction")
        self._chk_intensity.setToolTip(
            "Rescale image 2 linearly so that, inside the comparison region, it\n"
            "has the same mean and standard deviation as image 1. The map is\n"
            "derived from the region but applied to all of image 2, so the\n"
            "display, its histogram, the subtraction and the metrics all see it."
        )
        self._chk_intensity.toggled.connect(self._on_intensity_toggled)
        metrics_row.addWidget(self._chk_intensity)

        self._btn_fsc = QtWidgets.QPushButton("Compute FSC")
        self._btn_fsc.setToolTip(
            "Fourier shell correlation over the comparison region, in its own window.\n"
            "Reuses that window if it is already open, replacing its curves."
        )
        self._btn_fsc.clicked.connect(self.compute_fsc)
        metrics_row.addWidget(self._btn_fsc)
        layout.addLayout(metrics_row)

        # The region starts off, so everything it scopes starts unavailable.
        # _on_roi_enabled keeps the two in step from here on.
        for w in (self._btn_align, self._chk_intensity, self._btn_fsc):
            w.setEnabled(False)

        self.resize(900, 700)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def capture(self, index: int):
        """Pull the source widget's current image into slot ``index`` (0 or 1)."""
        data, pixel_size_m = self._read_source()
        if data is None:
            self._info_label.setText("Source plot has no image to copy.")
            return

        self._slots[index] = {
            'data': np.array(data, dtype=np.float32),   # snapshot, not a live reference
            'pixel_size_m': float(pixel_size_m),
            'label': self._source.info_text,
        }
        self._refresh_grid()
        # The box may have been switched on before there was anything to centre
        # it on, in which case this is the first chance to place it.
        if self._chk_roi.isChecked() and not self._roi_placed:
            self._place_roi()
        self._rebuild(autorange=True)
        self._sync_levels(index)

    def reset_offset(self):
        """Put image 2 back at the origin of the common grid."""
        self._offset = QtCore.QPointF(0.0, 0.0)
        self._rebuild()

    def align_phase_correlation(self):
        """
        Move image 2 onto image 1 by phase-correlating them inside the red box.

        The crop of image 2 is taken at its current offset, so what comes back
        is the *residual* misalignment and is added to the offset rather than
        replacing it. Sub-pixel to 1/``PHASE_CORR_UPSAMPLING`` of a pixel.
        """
        ov1, ov2 = self._roi_overlap()
        if ov1 is None or min(ov1.shape) < 4:
            self._metrics_label.setText(
                "Align: the two images do not overlap inside the red box."
            )
            return

        try:
            from skimage.registration import phase_cross_correlation
        except ImportError:
            self._metrics_label.setText(
                "Align: needs scikit-image, which is not installed in this environment."
            )
            return

        # Indexed rather than unpacked -- skimage's return arity has changed
        # across versions, but the shift has always come first.
        result = phase_cross_correlation(
            np.nan_to_num(ov1), np.nan_to_num(ov2),
            upsample_factor=PHASE_CORR_UPSAMPLING,
        )
        shift = np.asarray(result[0], dtype=float)   # (axis 0, axis 1) == (x, y)
        if shift.size < 2 or not np.all(np.isfinite(shift)):
            self._metrics_label.setText("Align: phase correlation returned no shift.")
            return

        self._offset = QtCore.QPointF(self._offset.x() + float(shift[0]),
                                      self._offset.y() + float(shift[1]))
        self._rebuild()

    def compute_fsc(self):
        """Open the FSC window on the two images' overlap inside the red box."""
        ov1, ov2 = self._roi_overlap()
        if ov1 is None:
            self._metrics_label.setText(
                "FSC: the two images do not overlap inside the red box."
            )
            return

        from ._fsc import FscWindow
        # Replaced rather than refreshed: each press is a snapshot of the
        # current box and offset, and the old one may still be worth comparing.
        self._fsc_window = FscWindow(
            ov1, ov2, pixel_size_m=self._grid_pixel_size_m, parent=self
        )
        self._fsc_window.show()
        self._fsc_window.raise_()
        self._fsc_window.activateWindow()
        return self._fsc_window

    @property
    def offset(self):
        """Image 2's drag offset as an (x, y) tuple in common-grid pixels."""
        return self._offset.x(), self._offset.y()

    @property
    def multiplier(self):
        """The subtraction's image 2 scale factor; 1.0 if the field is unusable."""
        try:
            return float(self._edit_multiplier.text())
        except ValueError:
            return 1.0

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------

    def _read_source(self):
        """(data, pixel_size_m) as currently displayed by the source widget."""
        item = self._source.image_item
        data = None if item is None else item.image
        if data is None:
            data = self._source.raw_data
        if data is None:
            return None, 1.0
        return data, self._source.pixel_size_m

    def _refresh_grid(self):
        """
        Re-derive the common grid and resample the slots onto it.

        Split out from :meth:`_rebuild` because it is the expensive step and
        only a new snapshot can change its result -- dragging must not pay for
        an interpolation pass on every mouse move.
        """
        slot1, slot2 = self._slots

        # Image 1's pixel size is the grid; with no image 1, image 2 keeps its own
        if slot1 is not None:
            self._grid_pixel_size_m = slot1['pixel_size_m']
        elif slot2 is not None:
            self._grid_pixel_size_m = slot2['pixel_size_m']
        else:
            self._grid_pixel_size_m = 1.0

        if slot2 is not None and self._grid_pixel_size_m > 0:
            self._resample_factor = slot2['pixel_size_m'] / self._grid_pixel_size_m
        else:
            self._resample_factor = 1.0

        self._grid_data = [self._resample(0), self._resample(1)]

    def _resample(self, index):
        """
        Slot ``index``'s array on the common grid, or None if the slot is empty.

        Only image 2 is ever resampled; image 1 defines the grid.
        """
        slot = self._slots[index]
        if slot is None:
            return None
        if index == 0 or abs(self._resample_factor - 1.0) <= RESAMPLE_TOLERANCE:
            return slot['data']
        return ndi_zoom(slot['data'], self._resample_factor, order=1)

    def _difference(self, data1, data2):
        """
        ``multiplier × data2 - data1`` over the union of the two.

        Both are already on the common grid, but they need not have the same
        shape and image 2 may have been dragged, so the result spans the
        bounding box of both, each contributing zero where it does not reach.
        Misalignment therefore shows up as the images' edges standing out
        against each other rather than being cropped away. The offset is rounded
        to whole pixels; the union's corner goes in ``_image2_origin`` so the
        result can be drawn back in image 1's coordinates.
        """
        if data1 is None or data2 is None:
            return None

        off_x, off_y = int(round(self._offset.x())), int(round(self._offset.y()))
        x0, y0 = min(0, off_x), min(0, off_y)
        x1 = max(data1.shape[0], off_x + data2.shape[0])
        y1 = max(data1.shape[1], off_y + data2.shape[1])

        out = np.zeros((x1 - x0, y1 - y0), dtype=np.float32)
        ax, ay = off_x - x0, off_y - y0
        out[ax:ax + data2.shape[0], ay:ay + data2.shape[1]] = data2 * self.multiplier
        out[-x0:-x0 + data1.shape[0], -y0:-y0 + data1.shape[1]] -= data1

        self._image2_origin = (x0, y0)
        return out

    # ------------------------------------------------------------------
    # region of interest
    # ------------------------------------------------------------------

    def _overlap_bounds(self):
        """
        ``(xs, xe, ys, ye, ox, oy)`` in common-grid pixels, or None.

        The region is the red box intersected with both images, image 2 sitting
        at the integer-rounded drag offset ``(ox, oy)``. Everything measured in
        this window is measured here, so the one place that gets the index
        arithmetic right is this one. Note the pyqtgraph convention the rest of
        the package follows: array axis 0 is x, so ``pos().x()`` indexes axis 0.
        """
        data1, data2 = self._grid_data
        if not self._chk_roi.isChecked() or data1 is None or data2 is None:
            return None

        ox, oy = int(round(self._offset.x())), int(round(self._offset.y()))
        pos, size = self._roi.pos(), self._roi.size()
        bx0, bx1 = int(round(pos.x())), int(round(pos.x() + size.x()))
        by0, by1 = int(round(pos.y())), int(round(pos.y() + size.y()))

        xs = max(0, ox, bx0)
        xe = min(data1.shape[0], ox + data2.shape[0], bx1)
        ys = max(0, oy, by0)
        ye = min(data1.shape[1], oy + data2.shape[1], by1)

        if xe <= xs or ye <= ys:
            return None
        return xs, xe, ys, ye, ox, oy

    def _roi_overlap(self, grid2=None):
        """
        ``(ov1, ov2)`` -- equally shaped views of both images over the red box.

        ``(None, None)`` when the box is off, an image is missing, or the two
        do not overlap inside it. ``grid2`` defaults to image 2 *after*
        intensity correction, so callers see exactly what is on screen; the
        correction itself passes the raw grid in to avoid chasing its own tail.
        """
        bounds = self._overlap_bounds()
        if bounds is None:
            return None, None
        xs, xe, ys, ye, ox, oy = bounds

        data2 = self._corrected2 if grid2 is None else grid2
        if data2 is None:
            return None, None
        return (self._grid_data[0][xs:xe, ys:ye],
                data2[xs - ox:xe - ox, ys - oy:ye - oy])

    def _intensity_corrected(self):
        """
        Image 2's grid mapped onto image 1's mean and standard deviation.

        The map is fitted inside the red box but applied to the whole array, so
        image 2 stays internally consistent where it extends past the box. Sets
        ``_intensity_map`` to the ``(gain, offset)`` actually applied, or None
        when the correction is off or cannot be fitted.
        """
        grid2 = self._grid_data[1]
        self._intensity_map = None
        if grid2 is None or not self._chk_intensity.isChecked():
            return grid2

        ov1, ov2 = self._roi_overlap(grid2=grid2)
        if ov1 is None:
            return grid2

        finite = np.isfinite(ov1) & np.isfinite(ov2)
        if finite.sum() < 2:
            return grid2
        a, b = ov1[finite], ov2[finite]

        std2 = float(b.std())
        # A flat image 2 has no spread to match -- shift it onto image 1's mean
        # and leave it at that rather than dividing by zero.
        gain = float(a.std()) / std2 if std2 > 0 else 1.0
        offset = float(a.mean()) - gain * float(b.mean())

        self._intensity_map = (gain, offset)
        return (grid2 * gain + offset).astype(np.float32)

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------

    def _rebuild(self, *_args, autorange=False):
        """
        Recompute what each item shows and where, from the current grid.

        Called for every change -- a new snapshot, a checkbox, the multiplier, a
        drag -- so the difference and the placements never go stale.
        """
        subtract = self._chk_subtract.isChecked()
        side_by_side = self._chk_side_by_side.isChecked() and not subtract

        # Intensity correction sits between the grid and everything downstream,
        # so the overlay, the histogram, the difference and the metrics are all
        # looking at the same corrected array.
        self._corrected2 = self._intensity_corrected()

        # Image 1 is left alone either way; only the image 2 slot becomes the
        # difference, so the two histograms keep their separate meanings.
        self._displayed = [self._grid_data[0], self._corrected2]
        if subtract:
            self._displayed[1] = self._difference(self._grid_data[0], self._corrected2)
        elif side_by_side:
            # Adjacent, both solid. Shapes need not match -- nothing is padded.
            width1 = self._grid_data[0].shape[0] if self._grid_data[0] is not None else 0
            self._image2_origin = (width1 * (1 + SIDE_BY_SIDE_GAP_FRAC), 0.0)
        else:
            self._image2_origin = (self._offset.x(), self._offset.y())

        data1, data2 = self._displayed
        for item, data in ((self._img1, data1), (self._img2, data2)):
            if data is None:
                item.clear()
            else:
                # Auto-level only the very first array an item is given; after
                # that the histogram is the user's, and a rebuild triggered by a
                # checkbox or a drag must not throw their levels away.
                item.setImage(data, autoLevels=item.levels is None)

        # One image at a time is always fully opaque -- transparency only earns
        # its keep when there is a second image underneath to see.
        show_both = self._chk_show_both.isChecked()
        if show_both:
            show1, show2 = True, True
            opacity2 = 1.0 if side_by_side else IMAGE2_OPACITY
        else:
            only1 = self._chk_show_only1.isChecked()
            show1, show2 = only1, not only1
            opacity2 = 1.0

        show1 = show1 and data1 is not None
        show2 = show2 and data2 is not None

        self._img1.setPos(0.0, 0.0)
        self._img2.setPos(*self._image2_origin)
        self._img2.setOpacity(opacity2)
        self._img2.draggable = not side_by_side

        self._img1.setVisible(show1)
        self._img2.setVisible(show2)

        self._chk_side_by_side.setEnabled(not subtract)
        self._chk_show_only1.setEnabled(not show_both)
        self._btn_reset.setEnabled(not side_by_side)

        if autorange:
            self._plot.getViewBox().autoRange()
        self._update_info()
        self._update_metrics()

    def _sync_levels(self, index):
        """
        Put one image's histogram and levels on its array's true range.

        Same reasoning as ``ImagePlotWidget._sync_levels``: leaving it to
        autoLevels lets the histogram's range drift out of step with the levels
        after a few images.
        """
        data = self._displayed[index]
        if data is None:
            return
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1.0   # a flat image still needs a non-empty range

        hist = self._hist1 if index == 0 else self._hist2
        hist.setHistogramRange(lo, hi)
        hist.setLevels(lo, hi)

    # ------------------------------------------------------------------
    # interaction
    # ------------------------------------------------------------------

    def _on_subtract_toggled(self, checked):
        """
        Subtract owns the layout: side-by-side is cleared and locked out, and
        only the difference is shown, it already carrying image 1. The
        visibility boxes remain the user's to change afterwards; turning
        subtract back off restores whatever they were before.
        """
        if checked:
            self._vis_before_subtract = (self._chk_show_both.isChecked(),
                                         self._chk_show_only1.isChecked())
            self._chk_side_by_side.setChecked(False)   # each may rebuild on its own
            self._chk_show_both.setChecked(False)
            self._chk_show_only1.setChecked(False)     # i.e. the image 2 slot alone
        elif self._vis_before_subtract is not None:
            show_both, show_only1 = self._vis_before_subtract
            self._chk_show_both.setChecked(show_both)
            self._chk_show_only1.setChecked(show_only1)
        self._rebuild(autorange=True)
        # The difference straddles zero -- nothing like image 2's own range
        self._sync_levels(1)

    def _on_multiplier_changed(self):
        self._rebuild()
        if self._chk_subtract.isChecked():
            self._sync_levels(1)

    def _on_roi_enabled(self, checked):
        """
        Show or hide the red box and everything it scopes.

        The box is centred on image 1 the first time it appears and then left
        where the user puts it, the same rule the lineout's metric cursors
        follow. Turning it off drops the intensity correction with it, which
        moves image 2's range, hence the re-level.
        """
        if checked and not self._roi_placed:
            self._place_roi()
        self._roi.setVisible(checked)
        for w in (self._btn_align, self._chk_intensity, self._btn_fsc):
            w.setEnabled(checked)

        self._rebuild()
        if self._chk_intensity.isChecked():
            self._sync_levels(1)

    def _place_roi(self):
        """Centre the box on image 1 (or image 2, failing that) at half its extent."""
        data = self._grid_data[0]
        if data is None:
            data = self._grid_data[1]
        if data is None:
            return

        w = max(data.shape[0] * ROI_INITIAL_FRAC, 4.0)
        h = max(data.shape[1] * ROI_INITIAL_FRAC, 4.0)
        self._roi.blockSignals(True)
        self._roi.setSize([w, h])
        self._roi.setPos([(data.shape[0] - w) / 2, (data.shape[1] - h) / 2])
        self._roi.blockSignals(False)
        self._roi_placed = True

    def _on_roi_changed(self):
        """
        The box moved or was resized.

        Only the metrics need redoing, unless the intensity correction is on --
        its map is fitted inside the box, so moving the box changes the pixels.
        """
        if self._chk_intensity.isChecked():
            self._rebuild()
        else:
            self._update_metrics()

    def _on_intensity_toggled(self, _checked):
        """
        Not wired straight to ``_rebuild``: the correction rescales image 2, so
        its histogram has to follow it onto the new range.
        """
        self._rebuild()
        self._sync_levels(1)

    def _on_image2_dragged(self, delta, is_start):
        """Apply a drag delta to the offset. Placement is _rebuild's business."""
        if is_start or self._drag_base is None:
            self._drag_base = QtCore.QPointF(self._offset)
        self._offset = self._drag_base + delta
        self._rebuild()

    def _on_mouse_moved(self, event):
        pos = event[0]   # SignalProxy wraps args in a tuple
        if not self._plot.sceneBoundingRect().contains(pos):
            self._cursor_label.setText('')
            return
        pt = self._plot.getViewBox().mapSceneToView(pos)

        # Report position relative to image 1's centre, as the source widget does
        data1 = self._displayed[0]
        cx = data1.shape[0] / 2 if data1 is not None else 0.0
        cy = data1.shape[1] / 2 if data1 is not None else 0.0
        x_um = (pt.x() - cx) * self._grid_pixel_size_m * 1e6
        y_um = (pt.y() - cy) * self._grid_pixel_size_m * 1e6

        # Topmost visible image under the cursor wins, i.e. what you are looking at
        for name, item, data in (("image 2", self._img2, self._displayed[1]),
                                 ("image 1", self._img1, self._displayed[0])):
            if data is None or not item.isVisible():
                continue
            ix = int(np.floor(pt.x() - item.pos().x()))
            iy = int(np.floor(pt.y() - item.pos().y()))
            if 0 <= ix < data.shape[0] and 0 <= iy < data.shape[1]:
                self._cursor_label.setText(
                    "x=%.2f, y=%.2f um, I=%.2e  [%s]" % (x_um, y_um, data[ix, iy], name)
                )
                return

        self._cursor_label.setText("x=%.2f, y=%.2f um" % (x_um, y_um))

    # ------------------------------------------------------------------
    # readout
    # ------------------------------------------------------------------

    def _update_info(self):
        lines = []
        for i, slot in enumerate(self._slots):
            name = "Image %d%s" % (i + 1, " (ground truth)" if i == 0 else "")
            if slot is None or self._grid_data[i] is None:
                lines.append(f"{name}: not set")
                continue
            shape = self._grid_data[i].shape
            text = "%s: %d×%d px, %d nm pix" % (
                name, shape[0], shape[1], round(slot['pixel_size_m'] * 1e9)
            )
            if i == 1 and abs(self._resample_factor - 1.0) > RESAMPLE_TOLERANCE:
                text += " → resampled ×%.4g onto image 1's grid" % self._resample_factor
            lines.append(text)

        if self._chk_subtract.isChecked():
            diff = self._displayed[1]
            if diff is None:
                lines.append("Subtract: needs both images")
            else:
                finite = diff[np.isfinite(diff)]
                extent = ("range %.4g to %.4g" % (float(finite.min()), float(finite.max()))
                          if finite.size else "no finite values")
                lines.append(
                    "Subtract: %.4g × image 2 - image 1 over %d×%d px, %s"
                    % (self.multiplier, diff.shape[0], diff.shape[1], extent)
                )

        dx, dy = self._offset.x(), self._offset.y()
        offset_text = "Image 2 offset: %.1f, %.1f px  (%.3f, %.3f µm)" % (
            dx, dy,
            dx * self._grid_pixel_size_m * 1e6,
            dy * self._grid_pixel_size_m * 1e6,
        )
        if not self._img2.draggable:
            offset_text += "  — dragging disabled"
        lines.append(offset_text)

        if self._intensity_map is not None:
            lines.append(
                "Intensity correction: image 2 × %.4g %+.4g"
                % (self._intensity_map[0], self._intensity_map[1])
            )

        self._info_label.setText("\n".join(lines))

    def _update_metrics(self):
        """
        PSNR, MSE and the offset over the red box's overlap.

        Each degenerate case gets said out loud rather than showing a NaN --
        "no overlap" and "the images are identical" are different answers and
        both are worth knowing.
        """
        if not self._chk_roi.isChecked():
            self._metrics_label.setText('')
            return

        ov1, ov2 = self._roi_overlap()
        if ov1 is None:
            self._metrics_label.setText(
                "Red box: the two images do not overlap inside it."
            )
            return

        finite = np.isfinite(ov1) & np.isfinite(ov2)
        if not finite.any():
            self._metrics_label.setText("Red box: no finite values in the overlap.")
            return
        a, b = ov1[finite], ov2[finite]

        mse = float(np.mean((a - b) ** 2))
        data_range = float(a.max() - a.min())
        if mse <= 0:
            psnr_text = "PSNR: ∞ dB (identical)"
        elif data_range <= 0:
            psnr_text = "PSNR: — (image 1 is flat here)"
        else:
            psnr_text = "PSNR: %.2f dB" % (10.0 * np.log10(data_range ** 2 / mse))

        dx, dy = self._offset.x(), self._offset.y()
        self._metrics_label.setText(
            "Red box overlap: %d×%d px   offset: %.2f, %.2f px (%.3f, %.3f µm)\n"
            "MSE: %.4g   %s"
            % (ov1.shape[0], ov1.shape[1], dx, dy,
               dx * self._grid_pixel_size_m * 1e6,
               dy * self._grid_pixel_size_m * 1e6,
               mse, psnr_text)
        )
