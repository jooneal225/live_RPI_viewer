"""
Standalone demo / smoke test:  python -m pg_image_tools

Opens an ImagePlotWidget on a synthetic test pattern with every optional widget
wired up, so each feature can be exercised without a host application.
"""

import sys

import numpy as np
from PyQt6 import QtWidgets

from ._image_view import ImagePlotWidget

PIXEL_SIZE_M = 10e-9   # 10 nm pixels


def make_test_pattern(n=512):
    """Siemens-star-ish spokes plus a hard edge and a soft edge to measure."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    cx = cy = n / 2
    theta = np.arctan2(y - cy, x - cx)
    r = np.hypot(x - cx, y - cy)

    img = 0.5 + 0.5 * np.cos(24 * theta)
    img[r > n * 0.42] = 0.0
    img[r < n * 0.04] = 1.0

    # Hard step edge down the left quarter — target for "Find 25%-75% resolution"
    img[:, : n // 8] = 0.0
    img[:, n // 8 : n // 6] = 1.0

    img += 0.02 * np.random.default_rng(0).standard_normal(img.shape)
    return img.T.astype(np.float32)   # axis 0 = x, axis 1 = y


def main():
    app = QtWidgets.QApplication(sys.argv)

    win = QtWidgets.QMainWindow()
    win.setWindowTitle("pg_image_tools demo")
    central = QtWidgets.QWidget()
    win.setCentralWidget(central)
    outer = QtWidgets.QVBoxLayout(central)

    title_label = QtWidgets.QLabel()
    info_label = QtWidgets.QLabel()
    header = QtWidgets.QHBoxLayout()
    header.addWidget(info_label)
    header.addStretch(1)
    header.addWidget(title_label)
    outer.addLayout(header)

    container = QtWidgets.QWidget()
    outer.addWidget(container, stretch=1)

    cb_transpose = QtWidgets.QCheckBox("Transpose")
    cb_log = QtWidgets.QCheckBox("Log")
    footer = QtWidgets.QHBoxLayout()
    footer.addWidget(cb_transpose)
    footer.addWidget(cb_log)
    footer.addStretch(1)
    outer.addLayout(footer)

    plot = ImagePlotWidget(
        container,
        info_label=info_label,
        title_label=title_label,
        transpose_checkbox=cb_transpose,
        log_checkbox=cb_log,
    )

    # A host-supplied menu action, and a scatter overlay in image coordinates
    plot.add_menu_action(
        "Demo: toggle scatter overlay",
        lambda checked: (plot.set_scatter_overlay(*_demo_points())
                         if checked else plot.clear_scatter_overlay()),
        checkable=True,
        at_top=True,
    )
    plot.add_menu_separator(at_top=True)

    data = make_test_pattern()
    plot.set_title("pg_image_tools demo\nsynthetic Siemens star with a hard edge")
    plot.set_image(data, pixel_size_m=PIXEL_SIZE_M)

    win.resize(900, 800)
    win.show()
    print("Click two points on the image to measure; right-click for the menu.")
    sys.exit(app.exec())


def _demo_points(n=512, count=200):
    rng = np.random.default_rng(1)
    return rng.uniform(0, n, count), rng.uniform(0, n, count)


if __name__ == "__main__":
    main()
