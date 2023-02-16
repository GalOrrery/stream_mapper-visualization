"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.visualization.utils.decorator import (
    add_savefig_option,
    with_ax,
    with_sorter,
)

__all__: list[str] = []

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy import integer
    from numpy.typing import NDArray

    from stream_ml.core import Data
    from stream_ml.core.params import Params
    from stream_ml.core.typing import Array


@add_savefig_option
@with_sorter
@with_ax
def weight(
    data: Data[Array],
    mpars: Params[Array],
    component: str | tuple[str, ...] = "stream",
    *,
    ax: Axes,
    sorter: NDArray[integer[Any]] | bool = True,
    **kwargs: dict[str, Any],
) -> Axes:
    """Plot the weights as a function of phi1."""
    # Iterate over components, plotting the weight
    components = component if isinstance(component, tuple) else (component,)
    for comp in components:
        cmpars = mpars.get_prefixed(comp)

        ax.plot(
            data["phi1"].flatten()[sorter],
            cmpars[(WEIGHT_NAME,)].flatten()[sorter],
            label=comp,
            **kwargs.get(comp, {}),  # TODO: apply sorter
        )

    # Add legend
    ax.legend()

    return ax


@add_savefig_option
@with_sorter
@with_ax
def parameter(
    data: Data[Array],
    mpars: Params[Array],
    component: str | tuple[str, ...] = "stream",
    coord: str | tuple[str, ...] = "phi1",
    param: str = "mu",
    *,
    ax: Axes,
    sorter: NDArray[integer[Any]] | bool = True,
    **kwargs: dict[str, Any],
) -> Axes:
    """Plot a parameter as a function of phi1.

    Parameters
    ----------
    data : Data[Array]
        The data to plot.
    mpars : Params[Array]
        The parameters to plot.

    component : str | tuple[str, ...]
        The component to plot.
    coord : str | tuple[str, ...]
        The coordinate to plot.
    param : str
        The parameter to plot.

    ax : Axes, keyword-only
        The axes to plot on.
    sorter : NDArray[integer[Any]] | bool, keyword-only
        The sorter to use.
    **kwargs : dict[str, Any], keyword-only
        Keyword arguments to pass to `~matplotlib.axes.Axe.plot`.
    """
    # Munge inputs into tuples
    components = component if isinstance(component, tuple) else (component,)
    coords = coord if isinstance(coord, tuple) else (coord,)

    # Iterate over components
    for comp in components:
        cmpars = mpars.get_prefixed(comp)

        # Iterate over coordinates
        for c in coords:
            ax.plot(
                data["phi1"].flatten()[sorter],
                cmpars[(c, param)].flatten()[sorter],
                label=comp,
                **kwargs.get(comp, {}),  # TODO: apply sorter
            )

    # Add legend
    ax.legend()

    return ax
