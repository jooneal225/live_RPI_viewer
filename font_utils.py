"""Shared helper for applying a global font size across the GUI's windows.

Copied from ptychoSAXS/gui/font_utils.py so that live_RPI_viewer stays a
self-contained repository (the two live in different git repos and are run as
separate processes). The API is identical; the only difference is that this
copy's persisted size lives in live_RPI_viewer/settings.ini under
[app] font_size rather than in the shared QSettings store, so the RPI viewer's
font size is independent of the ptychoSAXS GUIs'.

Kept dependency-free (only PyQt6) so it can be imported from any of this
package's windows without dragging in torch/cdtools/pyqtgraph.
"""

from PyQt6.QtWidgets import QWidget

# Matches the pointsize baked into most of rpi_live_gui.ui's widgets.
DEFAULT_FONT_SIZE = 11


def apply_font_size_to_tree(root, size):
    """Set `size` as the point size of root and every named descendant widget.

    Each widget's existing family/weight/italic are preserved. Anonymous
    internal sub-widgets (spin box arrows, combo box internals, ...) are
    skipped since Qt manages their geometry/font itself.

    NOTE: widgets built in code are only reached if they have been given an
    objectName — see the _named() helper in rpi_live_gui.py.
    """
    for w in [root] + root.findChildren(QWidget):
        if not w.objectName():
            continue
        f = w.font()
        f.setPointSize(size)
        w.setFont(f)
