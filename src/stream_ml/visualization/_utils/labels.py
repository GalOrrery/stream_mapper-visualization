"""Utilities for labelling axes."""

from __future__ import annotations

__all__: list[str] = []

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def set_label(text: str, /, *, ax: Axes, which: str, **kwargs: Any) -> None:
    """Set the label of an axis, appending to the existing label."""
    old_label = getattr(ax, f"get_{which}label")()
    getattr(ax, f"set_{which}label")(
        text + (rf" [{old_label}]" if old_label else ""), **kwargs
    )
