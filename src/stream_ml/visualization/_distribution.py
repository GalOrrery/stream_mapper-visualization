"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from typing import TYPE_CHECKING, Any

from matplotlib import pyplot as plt

from stream_mapper.visualization._defaults import LABEL_DEFAULTS
from stream_mapper.visualization._utils.arg_decorators import make_tuple
from stream_mapper.visualization._utils.plt_decorators import (
    add_savefig_option,
    with_tight_layout,
)

if TYPE_CHECKING:
    from astropy.table import QTable
    from matplotlib.figure import Figure
    from streaam_ml.core.data import Data

    from stream_mapper.core.typing import ArrayLike


@add_savefig_option
@with_tight_layout
@make_tuple("coords")
def coord_panels(
    data: QTable | Data[ArrayLike],
    /,
    coords: tuple[str, ...],
    xcoord: str = "phi1",
    *,
    use_hist: bool = False,
    **kwargs: Any,
) -> Figure:
    """Plot a grid of coordinate panels.

    Parameters
    ----------
    data : `~astropy.table.QTable` or `~stream_mapper.core.data.Data`
        Data with the column names ``coords``. Must have at least ``phi1``
        and ``phi2``.

    coords : tuple[str, ...]
        Tuple of coordinate names to plot.
    xcoord : str, optional
        Coordinate to plot on the x-axis. Default is ``"phi1"``.

    use_hist : bool, optional keyword-only
        Whether to use a histogram or scatter plot, by default `False`.
    **kwargs : Any
        Keyword arguments to pass to stuff.

    Returns
    -------
    `matplotlib.figure.Figure`
        The figure.
    """
    fig, axs = plt.subplots(1, len(coords), figsize=(4 * len(coords), 4))

    for i, c in enumerate(coords):
        ckw = kwargs.pop(c, {})
        ckw.setdefault("rasterized", True)

        if use_hist:
            ckw.setdefault("density", True)
            ckw.setdefault("bins", 100)
            axs[i].hist2d(data[xcoord].flatten(), data[c].flatten(), **ckw)
        else:
            ckw.setdefault("s", 1)
            axs[i].scatter(data[xcoord].flatten(), data[c].flatten(), **ckw)

        # Set labels
        if hasattr(data[xcoord], "unit"):
            axs[i].set_xlabel(f"{xcoord} [{axs[i].get_xlabel()}]")
        else:
            axs[i].set_xlabel(LABEL_DEFAULTS.get(xcoord, xcoord))

        if hasattr(data[c], "unit"):
            axs[i].set_ylabel(f"{LABEL_DEFAULTS.get(c, c)} [{axs[i].get_ylabel()}]")
        else:
            axs[i].set_ylabel(LABEL_DEFAULTS.get(c, c))

    return fig
