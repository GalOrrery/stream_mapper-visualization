"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.visualization.utils.arg_decorators import make_tuple, with_sorter
from stream_ml.visualization.utils.plt_decorators import (
    add_savefig_option,
    with_ax,
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
@make_tuple("components")
def weight(
    data: Data[Array],
    mpars: Params[Array],
    components: tuple[str, ...] = ("stream",),
    *,
    ax: Axes,
    sorter: NDArray[integer[Any]] | bool = True,
    **kwargs: dict[str, Any],
) -> Axes:
    """Plot the weights as a function of phi1."""
    # Iterate over components, plotting the weight
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
@make_tuple("components", "coords")
def parameter(  # noqa: PLR0913
    data: Data[Array],
    mpars: Params[Array],
    components: tuple[str, ...] = ("stream",),
    coords: tuple[str, ...] = ("phi1",),
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

    components : tuple[str, ...]
        The component(s) to plot.
    coords : tuple[str, ...]
        The coordinate(s) to plot.
    param : str
        The parameter to plot.

    ax : Axes, keyword-only
        The axes to plot on.
    sorter : NDArray[integer[Any]] | bool, keyword-only
        The sorter to use.
    **kwargs : dict[str, Any], keyword-only
        Keyword arguments to pass to `~matplotlib.axes.Axe.plot`.
    """
    # Iterate over components
    for comp in components:
        cmpars = mpars.get_prefixed(comp)

        # Iterate over coordinates
        for crd in coords:
            ax.plot(
                data["phi1"].flatten()[sorter],
                cmpars[(crd, param)].flatten()[sorter],
                label=f"{comp}[{crd}]",
                **kwargs.get(comp, {}),  # TODO: apply sorter
            )

    # Add legend
    ax.legend()

    return ax
