"""
Fourier shell correlation popup for the comparison window.

See ``pg_image_tools/__init__.py`` for usage documentation.
"""

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
import pyqtgraph as pg

from .fourier_shell_corr import (
    SNRT_1BIT, SNRT_HALF_BIT, fourier_shell_corr, threshold_curve,
)

# Shell 0 is DC and shell 1 holds a handful of pixels; both sit at ~1 whatever
# the images are, so a crossing found there is an artefact, not a resolution.
FIRST_MEANINGFUL_SHELL = 2


def _square_crop(a, b):
    """``a`` and ``b`` centre-cropped to a common square, or (None, None)."""
    if a is None or b is None or a.shape != b.shape:
        return None, None
    w, h = a.shape
    m = min(w, h)
    if m < 8:
        return None, None
    x0, y0 = (w - m) // 2, (h - m) // 2
    sl = (slice(x0, x0 + m), slice(y0, y0 + m))
    return np.nan_to_num(np.asarray(a, dtype=float))[sl], \
           np.nan_to_num(np.asarray(b, dtype=float))[sl]


def find_crossing(fsc, threshold):
    """
    Where the FSC first falls through its threshold, in units of Nyquist.

    ``fsc`` is the full binned curve and ``threshold`` stops at Nyquist, so the
    former is truncated to the latter's length first. Returns None unless the
    curve is still below threshold at Nyquist -- a noise dip that recovers is
    not a resolution limit. The crossing is interpolated between the bracketing
    shells rather than rounded to one, which otherwise quantises the reported
    resolution in steps of a whole shell.
    """
    n = threshold.size
    if fsc.size < n or n <= FIRST_MEANINGFUL_SHELL:
        return None

    diff = fsc[:n] - threshold
    if not np.isfinite(diff[n - 1]) or diff[n - 1] >= 0:
        return None

    below = np.where(diff[FIRST_MEANINGFUL_SHELL:] < 0)[0]
    if below.size == 0:
        return None
    i = int(below[0]) + FIRST_MEANINGFUL_SHELL

    # Linear interpolation of the zero crossing of diff between i-1 and i
    prev = diff[i - 1]
    if i > 0 and np.isfinite(prev) and prev > 0:
        frac = prev / (prev - diff[i])
        return (i - 1 + frac) / n
    return i / n


class FscWindow(QtWidgets.QWidget):
    """
    A standalone, non-blocking window plotting the Fourier shell correlation of
    two equally-shaped image crops.

    The pair is centre-cropped to a square -- the underlying
    :func:`~pg_image_tools.fourier_shell_corr.fourier_shell_corr` builds its
    Tukey window from one axis only -- transformed once, and then re-thresholded
    for free when the criterion combo changes: ``SNRt`` moves the threshold
    curve but not the correlation.

    The reported resolution is the **half-period** at the crossing,
    ``pixel_size / f``, which is the convention the beamline's other tools use.
    """

    def __init__(self, data1, data2, pixel_size_m=1.0, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle("Fourier Shell Correlation")

        self._pixel_size_m = float(pixel_size_m)
        self._fsc = None            # the correlation curve, independent of SNRt
        self._n_half = 0            # shells up to Nyquist

        self._build_ui()
        self._compute(*_square_crop(data1, data2))
        self._replot()

        self.resize(720, 540)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        self._plot = pg.PlotWidget()
        self._plot.setLabel('bottom', 'Spatial frequency / Nyquist')
        self._plot.setLabel('left', 'Fourier shell correlation')
        self._plot.setXRange(0.0, 1.0)
        self._plot.setYRange(-0.1, 1.1)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.addLegend()
        layout.addWidget(self._plot, stretch=1)

        self._res_label = QtWidgets.QLabel()
        self._res_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._res_label)

        self._info_label = QtWidgets.QLabel()
        self._info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._info_label)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Threshold criterion"))
        self._combo = QtWidgets.QComboBox()
        # Text is cosmetic; the SNRt that matters rides along as item data.
        self._combo.addItem("1/2-bit  (SNRt = %.4g)" % SNRT_HALF_BIT, SNRT_HALF_BIT)
        self._combo.addItem("1-bit  (SNRt = %.4g)" % SNRT_1BIT, SNRT_1BIT)
        self._combo.setToolTip(
            "van Heel & Schatz criterion the FSC is read against.\n"
            "Only the threshold curve changes -- the correlation is not recomputed."
        )
        self._combo.currentIndexChanged.connect(self._replot)
        row.addWidget(self._combo)
        row.addStretch(1)
        layout.addLayout(row)

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------

    def _compute(self, a_sq, b_sq):
        """Run the FSC once. ``SNRt`` is irrelevant here; only the curve is kept."""
        if a_sq is None:
            self._info_label.setText(
                "Overlap is too small or not usable -- need at least 8×8 px in both images."
            )
            return

        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            fsc, threshold, _ = fourier_shell_corr(a_sq, b_sq, to_print=False)
        except Exception as exc:
            self._info_label.setText("FSC failed: %s" % exc)
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._fsc = np.real(fsc)
        self._n_half = threshold.size
        self._info_label.setText(
            "%d×%d px square crop, %d nm pix, %d shells to Nyquist"
            % (a_sq.shape[0], a_sq.shape[1],
               round(self._pixel_size_m * 1e9), self._n_half)
        )

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------

    def _replot(self):
        self._plot.clear()
        # clear() leaves the view where it was, but re-asserting costs nothing
        # and keeps the axes on the fixed FSC ranges after any user zoom.
        self._plot.setXRange(0.0, 1.0)
        self._plot.setYRange(-0.1, 1.1)
        if self._fsc is None:
            self._res_label.setText("Resolution: —")
            return

        snrt = self._combo.currentData()
        label = "1/2-bit threshold" if snrt == SNRT_HALF_BIT else "1-bit threshold"
        threshold = threshold_curve(self._n_half, snrt)

        freq = np.arange(self._n_half, dtype=float) / self._n_half
        self._plot.plot(freq, self._fsc[:self._n_half],
                        pen=pg.mkPen((80, 220, 80), width=2), name="FSC")
        self._plot.plot(freq, threshold,
                        pen=pg.mkPen('r', width=1.5, style=Qt.PenStyle.DashLine), name=label)

        crossing = find_crossing(self._fsc, threshold)
        if crossing is None or crossing <= 0:
            self._res_label.setText(
                "Resolution: —   (the FSC never falls through the %s)" % label
            )
            return

        self._plot.addItem(pg.InfiniteLine(
            pos=crossing, angle=90,
            pen=pg.mkPen((80, 120, 255), width=1.5, style=Qt.PenStyle.DotLine),
        ))
        self._res_label.setText(
            "Resolution: %.1f nm (half-period)   crossing at %.4g × Nyquist = %.2f px"
            % (self._pixel_size_m * 1e9 / crossing, crossing, 1.0 / crossing)
        )
