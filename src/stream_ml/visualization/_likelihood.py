"""Stream-ML visualization."""

from __future__ import annotations

__all__: list[str] = []

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from matplotlib import pyplot as plt

from stream_ml.visualization._defaults import LABEL_DEFAULTS
from stream_ml.visualization._utils.arg_decorators import make_tuple
from stream_ml.visualization._utils.plt_decorators import (
    add_savefig_option,
    with_tight_layout,
)

if TYPE_CHECKING:
    from astropy.table import QTable
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from stream_ml.core import Data
    from stream_ml.core.typing import ArrayLike


@add_savefig_option
@with_tight_layout
@make_tuple("coords")
def component_likelihood(
    data: QTable | Data[ArrayLike],
    prob: NDArray[np.floating[Any]],
    coords: tuple[str, ...],
    *,
    alpha_min: float = 0.05,
    **kwargs: Any,
) -> Figure:
    """Plot the probability of a component.

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
        axs[i].set_ylabel(LABEL_DEFAULTS.get(c, c), fontsize=15)

    return fig
