"""Stream-ML visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from matplotlib import pyplot as plt

from stream_ml.visualization.defaults import YLABEL_DEFAULTS
from stream_ml.visualization.utils.arg_decorators import make_tuple
from stream_ml.visualization.utils.plt_decorators import (
    add_savefig_option,
    with_tight_layout,
)

__all__: list[str] = []


if TYPE_CHECKING:
    from astropy.table import QTable
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from stream_ml.core import Data
    from stream_ml.core.typing import ArrayLike


########################################################################


@add_savefig_option
def component_likelihood_modelspace(
    data: Data[ArrayLike],
    prob: NDArray[np.floating[Any]],
    *,
    use_hist: bool = False,
    alpha_min: float = 0.05,
    xbins: int = 10,
    ybins: int = 10,
) -> Figure:
    """Plot the data in model space.

    Parameters
    ----------
    data : NDArray[np.floating[Any]]
        The data to plot.
    prob : NDArray[np.floating[Any]]
        The probability of the data.

    use_hist : bool, keyword-only
        Whether to use a histogram or scatter plot.
    alpha_min : float, keyword-only
        The minimum alpha to plot.
    xbins, ybins : int, keyword-only
        The number of bins to use in the x,y-direction.

    Returns
    -------
    `~matplotlib.figure.Figure`
        The figure that was plotted.
    """
    # ---------------------
    # Make figure and axes.
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)

    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax.set(aspect=1)

    # Top histogram.
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)

    # Side histogram.
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    ax_histy.tick_params(axis="y", labelleft=False)

    # ---------------------
    # Plot the data.

    # Sort and shade by probability.
    sorter = np.argsort(prob.flatten())
    alpha = np.copy(prob[sorter])
    alpha[alpha < alpha_min] = alpha_min

    if not use_hist:
        ax.scatter(
            data["phi1"].flatten()[sorter],
            data["phi2"].flatten()[sorter],
            c=prob[sorter],
            cmap="turbo",
            s=1,
            rasterized=True,
            alpha=alpha,
        )
    else:
        ax.hist2d(
            data["phi1"].flatten()[sorter],
            data["phi2"].flatten()[sorter],
            bins=50,
            cmap="turbo",
            weights=alpha,
            rasterized=True,
        )

    ax.set_xlabel(r"$\phi_1'$", fontsize=15)
    ax.set_ylabel(r"$\phi_2'$", fontsize=15)

    ax_histx.hist(data["phi1"].flatten(), bins=xbins, color="gray", density=True)
    ax_histx.set_yscale("log")

    ax_histy.hist(
        data["phi2"].flatten(),
        bins=ybins,
        color="gray",
        orientation="horizontal",
        density=True,
    )
    ax_histy.set_xscale("log")

    return fig


########################################################################


@add_savefig_option
@with_tight_layout
@make_tuple("coords")
def component_likelihood_dataspace(
    data: QTable | Data[ArrayLike],
    prob: NDArray[np.floating[Any]],
    coords: tuple[str, ...],
    *,
    alpha_min: float = 0.05,
    **kwargs: Any,
) -> Figure:
    """Plot the probability in data space.

    Parameters
    ----------
    data : QTable | Data[ArrayLike]
        The data.
    prob : NDArray[np.floating[Any]]
        The probability of each data point.
    coords : tuple[str, ...]
        The coordinate(s) to plot.

    alpha_min : float, optional keyword-only
        The minimum alpha value.
    **kwargs : Any
        Keyword arguments to pass to `~matplotlib.pyplot.subplots`.

    Returns
    -------
    `~matplotlib.figure.Figure`
        The figure.
    """
    # Sort and shade the data by the probability, with the most probable plotted
    # on top.
    sorter = np.argsort(prob.flatten())
    alpha = np.copy(prob)
    alpha[alpha < alpha_min] = alpha_min

    # Make figure, the figsize is a bit of a hack
    fig, axs = plt.subplots(len(coords), 1, figsize=(8, 2.5 * len(coords)))
    axs = cast(
        "np.ndarray[Any, Axes]",
        np.array([axs], dtype=object) if len(coords) == 1 else axs,
    )

    for i, c in enumerate(coords):
        axs[i].scatter(
            np.array(data["phi1"].flatten()[sorter]),
            np.array(data[c].flatten()[sorter]),
            c=prob[sorter],
            cmap="turbo",
            s=1,
            rasterized=True,
            alpha=alpha[sorter],
            **kwargs,
        )
        axs[i].set_xlabel(r"$\phi_1$", fontsize=15)
        axs[i].set_ylabel(YLABEL_DEFAULTS.get(c, c), fontsize=15)

    return fig
