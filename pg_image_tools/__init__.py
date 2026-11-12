"""
pg_image_tools -- reusable interactive 2D image plotting for PyQt6 + pyqtgraph
==============================================================================

This is live_RPI_viewer's PyQt6 port of ptychi-file-browser/pg_image_tools
(which stays on PyQt5). Beyond the binding, it differs in two places: every
ImageItem is pinned to col-major (see Coordinate conventions), and
``ImagePlotWidget`` takes a ``keep_points`` flag. Carry fixes across by hand.

A drop-in interactive image viewer: click-to-measure, live cursor readout, a
lineout panel with a 25%-75% resolution metric, display filters, and a populated
right-click menu. It knows nothing about where your data comes from -- you hand
it a numpy array and a pixel size, it renders and measures.

This package is self-contained. It imports only ``numpy``, ``scipy``, ``PyQt6``
and ``pyqtgraph``, and nothing from whatever project it lives in.


Requirements
------------
numpy, scipy, PyQt6, pyqtgraph.

Two features reach past that, both lazily, so the package imports and runs
without either: the comparison window's ``Align`` button needs
``scikit-image`` (it says so in its readout if the import fails), and the
legacy ``fourier_shell_corr(..., to_print=True)`` plotting path needs
``matplotlib`` -- the FSC window itself does not, plotting in pyqtgraph.


Quick start
-----------
The only thing you need is somewhere to put it::

    from pg_image_tools import ImagePlotWidget

    plot = ImagePlotWidget(self.some_placeholder_widget)
    plot.set_image(my_2d_array, pixel_size_m=8.7e-9)

``ImagePlotWidget`` is a ``QWidget``, so ``container`` is optional -- omit it and
add the widget to a layout yourself, or ``show()`` it as a top-level window.


Full example (Qt Designer host)
-------------------------------
Every companion widget is optional. Supply the ones you have; the features you
skip simply do not appear::

    class MyWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            uic.loadUi("my_window.ui", self)

            self.plot = ImagePlotWidget(
                self.plotContainer,                    # a plain QWidget from the .ui
                info_label=self.label_plot_info,       # optional
                title_label=self.label_plot_title,     # optional
                transpose_checkbox=self.checkBox_transpose,   # optional
                log_checkbox=self.checkBox_logCmap,           # optional
            )

            # Add your own entries to the image's right-click menu
            self.plot.add_menu_action("Copy File Path", self._copy_path, at_top=True)
            self.act_full = self.plot.add_menu_action(
                "Full Probe Zoom", checkable=True,
                after=self.plot.action_no_zoom_tracking,
            )

        def show_array(self, arr, pixel_size_m, caption):
            self.plot.set_title(caption)               # no-op without a title_label
            self.plot.set_image(arr, pixel_size_m=pixel_size_m)

Companions can also be bound after construction with ``attach_info_label()``,
``attach_title_label()``, ``attach_transpose_checkbox()`` and
``attach_log_checkbox()``.


Optional widgets
----------------
================== ============================================ ==========================
argument           when supplied                                when omitted
================== ============================================ ==========================
info_label         shows "W×H µm, N nm pix" plus a live          no readout; cursor
                   ``x=, y=, I=`` line as the cursor moves       position still emitted
                                                                 via sigMouseMovedImage
title_label        set_title() writes into it, each line         set_title() is a no-op
                   elided in the middle, full text as tooltip
transpose_checkbox image is transposed before display;           image never transposed
                   toggling redraws immediately
log_checkbox       displays log10(|data|); toggling redraws      linear display only
================== ============================================ ==========================

Other constructor flags: ``enable_lineout``, ``enable_measure``,
``enable_filters``, ``enable_compare`` (all default True),
``keep_points`` (default False: every ``set_image()`` clears the measurement
points; True keeps them, and the lineout, while the new image has the same
shape -- for live feeds) and ``title_elide_width`` (default 500 px).


Interactions
------------
Image
    * Left-click two points -> cross markers, a dashed connecting line, and a
      label with the separation in µm and in pixels. A third click clears them.
    * Right-click menu: ``Plot Lineout``, ``Auto-reset Zoom`` (re-fit the view on
      every new image), ``Open comparison window``, and ``Analyze > Median
      Filter`` / ``Gaussian Filter`` (each prompts for a kernel width). Zooming
      back out is pyqtgraph's own ``View All``.
    * ``Analyze > Unwrap Phase`` asks which region to unwrap: ``Entire image``
      unwraps immediately; ``Rectangular region`` / ``Circular region`` drop a
      red, draggable, resizable ROI (the circle resizes as a radius) with
      floating ``Apply`` / ``Cancel`` buttons over the plot -- Apply unwraps
      only the pixels inside the ROI, leaving the rest of the image untouched.
      The unwrap replaces the cached array itself (unlike the display filters,
      this isn't reversible by unchecking anything). Needs ``scikit-image``;
      says so in a popup if it is missing.

Comparison window (shown by ``Open comparison window``)
    * A separate, non-blocking window that opens empty. ``Set image 1`` and
      ``Set image 2`` each snapshot whatever the plot is displaying at that
      moment, along with its pixel size.
    * Image 1 is the ground truth: drawn solid at the origin, and its pixel size
      defines the common grid, so image 2 is resampled (linear interpolation)
      whenever the two pixel sizes differ. Image 2 is drawn semi-transparent on
      top and is dragged into alignment with the left mouse button; the offset
      is reported in pixels and µm, and ``Reset offset`` returns it to the
      origin. Dragging image 2 takes over left-drag panning where it lies —
      wheel zoom, right-drag scaling and ``View All`` are unaffected.
    * Each image has its own histogram, so levels and colormap are independent.
    * ``Side-by-side comparison`` draws both solid and places them horizontally
      adjacent instead of overlaid; the two need not have the same shape, and
      the drag offset is remembered for when it is switched back off.
    * ``Show both images`` overlays the two with image 2 semi-transparent.
      Uncheck it to look at one at a time, always fully opaque -- transparency
      only earns its keep when there is something underneath to see. Which one
      is then shown is ``Show only image 1``: checked for image 1, unchecked
      for image 2. That second box is disabled while ``Show both images`` is
      on, there being nothing for it to choose between.
    * ``Subtract`` leaves image 1 alone and puts ``multiplier × image 2 -
      image 1`` in the image 2 slot, taken over the union of the two so that
      image 2 stays draggable and the difference follows it live. Side-by-side
      is cleared and locked out while it is on, the view drops to the
      difference alone (it already carries image 1), and the second histogram
      is re-levelled, the difference straddling zero rather than sitting on
      image 2's own range. Turning subtract back off restores whatever the two
      visibility boxes were. The ``multiplier`` field next to it scales image 2
      before the subtraction; it defaults to 1.0.
    * A second readout under the plot follows the cursor:
      ``x=…, y=… um, I=…`` for the topmost visible image beneath it, with x and
      y measured from image 1's centre.

Red box (the comparison window's second control row)
    * ``Red box`` adds a resizable rectangle, centred on image 1 the first time
      it appears and left where you put it thereafter. Drag inside it to move
      it and its handles to resize it; dragging *outside* it still moves
      image 2. Everything in the row is measured over the region where the box
      and both images overlap, and everything in the row is disabled until the
      box is on.
    * A third readout reports that overlap's size, the current offset in pixels
      and µm, and the ``MSE`` and ``PSNR`` between the two images across it.
      PSNR uses image 1's peak-to-peak inside the box as the data range.
    * ``Align (phase correlation)`` phase-correlates the two over the box and
      moves image 2 onto image 1, to a tenth of a pixel. The crop of image 2 is
      taken at its current offset, so the measured shift is a residual and is
      added to the offset -- press it twice and the second press should barely
      move anything. Needs ``scikit-image``; says so in the readout if it is
      missing.
    * ``Intensity correction`` rescales image 2 linearly so that, inside the
      box, it carries image 1's mean and standard deviation. The map is fitted
      in the box but applied to all of image 2, so the overlay stays consistent
      outside it, and it is applied for real: the display, image 2's histogram,
      the subtraction, the metrics and the FSC all see the corrected array. The
      gain and offset applied are printed in the info readout.
    * ``Compute FSC`` opens a separate, non-blocking window (see below) on the
      overlap.

FSC window (shown by ``Compute FSC``)
    * Plots the Fourier shell correlation of the two overlapping crops against
      spatial frequency in units of Nyquist, with the van Heel & Schatz
      threshold curve and a vertical line where the two cross.
    * The crops are centre-cropped to a square first -- the underlying
      ``fourier_shell_corr`` builds its Tukey window from one axis only.
    * The ``Threshold criterion`` combo chooses ``1/2-bit`` (the default) or
      ``1-bit``. Only the threshold curve is recomputed: ``SNRt`` does not
      enter the correlation, so switching is free.
    * The resolution is reported in nm as the **half-period** at the crossing,
      ``pixel size / f``. A curve that never falls through its threshold says
      so instead of reporting a number.

Lineout panel (shown by ``Plot Lineout``)
    * Plots the image sampled along the two measurement points.
    * Two draggable vertical cursors, blue and green, continuously report their
      x positions, the values under them, and ΔX.
    * Drag them to either side of a hard edge, then right-click the lineout and
      choose ``Find 25%-75% resolution``: the crossing width is shaded and
      reported in nm.
    * ``Find resolution (erf fit)`` fits an error function to the same region
      (added to the plot as a dashed overlay) and reports the FWHM of its
      derivative -- a Gaussian -- computed directly from the fit's sigma.
      The two metrics are independent, so both can be on screen at once, each
      named in a legend; dragging either cursor clears both, since both were
      fitted to the region between them.
    * ``Lineout width...`` opens a popup to choose a perpendicular averaging
      width in pixels (0-100, default 0). Above 0, the lineout is the mean of
      an oversampled swath of parallel samples across that width instead of a
      single-pixel-wide line -- useful for noisy, non-synthetic images.


Data flow and caching
---------------------
``set_image()`` keeps a reference to the raw array. Display-option changes
(filter, transpose, log) re-render from that cache -- the widget never asks the
host to reload, and the current zoom, measurement points, lineout and scatter
overlay survive the redraw. Toggling transpose is the one exception: it changes
the coordinate system, so the pixel-space overlays are cleared.

``sigRedrawRequested`` fires *after* such a redraw. It is a notification for
hosts that keep their own derived state -- not a request for data. Leave it
unconnected if you have none.


Coordinate conventions
----------------------
* Images follow the pyqtgraph convention: axis 0 is x, axis 1 is y. Pass arrays
  already in that orientation (or use the transpose checkbox). This holds even
  under ``pg.setConfigOptions(imageAxisOrder='row-major')``: the widget pins
  its ImageItems to col-major, so a row-major host passes ``arr.T``.
* ``pixel_size_m`` is metres per pixel and drives every µm / nm readout.
  Leave it at 1.0 and the readouts are effectively in pixels × 1e6.
* Measurement points and ``set_scatter_overlay(x, y)`` are in image pixel
  coordinates, origin at the array corner. The cursor readout reports µm
  relative to the image *centre*.


Extension points
----------------
``add_menu_action(text, callback, checkable=, checked=, at_top=, after=)``
    Add an entry to the image's right-click menu and get the QAction back.
    ``at_top=True`` places it above this widget's own block (still below
    pyqtgraph's built-ins); ``after=some_action`` places it directly after an
    existing entry. Also ``add_menu_separator()`` and ``add_menu_submenu()``.

``set_scatter_overlay(x, y)`` / ``clear_scatter_overlay()``
    A red scatter layer in image coordinates for host-supplied points.

``clear_image()``
    Blank the view -- drops the cached array and every overlay, for hosts that
    would otherwise be left showing a stale image.

``open_comparison_window()`` / ``comparison_window``
    The programmatic form of the ``Open comparison window`` menu entry. The
    window is created on first call and reused after that, so calling it again
    raises the existing one. Pass ``enable_compare=False`` to leave the menu
    entry out entirely. ``ImageCompareWindow`` is exported too, for hosts that
    would rather build and place it themselves -- it takes the source
    ``ImagePlotWidget`` as its first argument. On the window itself,
    ``capture(index)``, ``reset_offset()``, ``align_phase_correlation()`` and
    ``compute_fsc()`` are the programmatic forms of its buttons, and ``offset``
    / ``multiplier`` are read-only state.

``FscWindow(data1, data2, pixel_size_m=…)``
    The FSC popup, exported for hosts that want it on their own pair of arrays
    rather than through the comparison window. The two must have the same
    shape; it centre-crops them to a square itself.

``zoom_to_left_square()``, ``reset_zoom()``, ``set_view_range(**kwargs)``
    Zoom helpers. ``reset_zoom()`` is the programmatic form of "View All"; the
    last is a passthrough to ``ViewBox.setRange``.

``image_view``, ``view``, ``image_item``, ``menu``, ``lineout``
    Escape hatches to the underlying pyqtgraph objects for anything not covered.

``displayed_shape``, ``pixel_size_m``, ``raw_data``, ``measure_points``, ``info_text``
    Read-only state.

Signals: ``sigRedrawRequested()``, ``sigPointsChanged(list)``,
``sigMouseMovedImage(float, float)``.


Demo
----
``python -m pg_image_tools`` opens a standalone window on a synthetic test
pattern with all optional widgets wired up. Useful as a smoke test after edits.


Packaging
---------
This folder has no imports from its host project, so it can be moved into its
own repository or turned into a pip package by itself. Keep the top-level module
name ``pg_image_tools`` and ``from pg_image_tools import ImagePlotWidget`` keeps
working unchanged in every dependent project.
"""

from ._image_view import ImagePlotWidget
from ._lineout import LineoutPanel, compute_lineout
from ._compare import ImageCompareWindow
from ._fsc import FscWindow

__all__ = ["ImagePlotWidget", "LineoutPanel", "compute_lineout",
           "ImageCompareWindow", "FscWindow"]
__version__ = "0.1.0"
