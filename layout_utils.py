"""Tighter spacing for the control panels next to the image views.

Qt's default group-box layouts leave ~10 px on every side and between rows,
which on a panel of six or seven groups adds up to a scroll bar. Kept
dependency-free (only PyQt6), like font_utils.
"""

from PyQt6.QtWidgets import QGridLayout, QGroupBox, QLayout


def compact_group_boxes(root, margins=(6, 2, 6, 4), spacing=4):
    """Shrink the margins and spacing of every QGroupBox layout under root,
    and the spacing of the layouts nested in them."""
    for box in root.findChildren(QGroupBox):
        lay = box.layout()
        if lay is None:
            continue
        lay.setContentsMargins(*margins)
        stack = [lay]
        while stack:
            lay = stack.pop()
            lay.setSpacing(spacing)
            if isinstance(lay, QGridLayout):
                lay.setHorizontalSpacing(spacing + 2)
            for i in range(lay.count()):
                child = lay.itemAt(i).layout()
                if isinstance(child, QLayout):
                    stack.append(child)
