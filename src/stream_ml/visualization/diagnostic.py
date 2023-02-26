"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matplotlib import gridspec
from matplotlib import pyplot as plt

from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.visualization.defaults import COORD_TO_YLABEL
from stream_ml.visualization.utils.arg_decorators import make_tuple
from stream_ml.visualization.utils.plt_decorators import (
    add_savefig_option,
    with_tight_layout,
)

__all__: list[str] = []


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.gridspec import SubplotSpec

    from stream_ml.core import Data
    from stream_ml.core.api import Model
    from stream_ml.core.params import Params
    from stream_ml.core.typing import Array


def _plot_coordinate_component(
    data: Data[Array],
    pars: Params[Array],
    component: str,
    coord: str,
    *,
    ax: Axes,
    ax_top: Axes,
    y: str = "mu",
    y_err: str = "sigma",
) -> Axes:
    """Plot a single coordinate for a single component.

    Parameters
    ----------
    data : Data[Array]
        The data to plot.
    pars : Params
        The parameters to plot.
    component : str
        The component to plot.
    coord : str
        The coordinate to plot.
    ax : Axes, keyword-only
        The axes to plot on.
    ax_top : Axes, keyword-only
        The axes to plot the weights on.
    y : str, keyword-only
        The name of the mean parameter.
    y_err : str, keyword-only
        The name of the standard deviation parameter.

    Returns
    -------
    Axes
        The axes that were plotted on.
    """
    cpars = pars.get_prefixed(component)

    phi1 = data["phi1"].flatten()
    mu = cpars[coord, y].flatten()
    yerr = cpars[coord, y_err].flatten()

    p = ax.plot(phi1, mu, label=f"{component}")
    fc = p[0].get_color()
    ax.fill_between(phi1, y1=mu + yerr, y2=mu - yerr, facecolor=fc, alpha=0.5)
    ax.fill_between(phi1, y1=mu + 2 * yerr, y2=mu - 2 * yerr, facecolor=fc, alpha=0.25)

    ax_top.plot(phi1, cpars[("weight",)], label=f"{component}[weight]")

    return ax


def _plot_coordinate_panel(  # noqa: PLR0913
    fig: Figure,
    gs: SubplotSpec,
    coord: str,
    data: Data[Array],
    mpars: Params[Array],
    components: tuple[str, ...],
    model_components: tuple[str, ...],
    kwargs: dict[str, Any],
) -> None:
    """Plot a single coordinate for all components."""
    gsi = gs.subgridspec(2, 1, height_ratios=(1, 3))

    # Axes
    ax = fig.add_subplot(gsi[1])
    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(COORD_TO_YLABEL.get(coord, coord))

    # Plot Weight histogram
    ax_top = fig.add_subplot(gsi[0], sharex=ax)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylabel(r"weight")

    if kwargs.get("include_total_weight", True):
        ax_top.plot(
            data["phi1"].flatten(),
            mpars[(WEIGHT_NAME,)].flatten(),
            color="gray",
            label="total weight",
        )

    # Data
    ax.plot(
        data["phi1"].flatten(),
        data[coord].flatten(),
        c="black",
        marker=",",
        linestyle="none",
    )

    # Plot components
    for comp in components:
        _plot_coordinate_component(
            data,
            mpars,
            component=comp,
            coord=coord,
            ax=ax,
            ax_top=ax_top,
            y="mu",
            y_err="sigma",
        )

    # Include background in top plot, if applicable
    if (
        "background" not in components
        and "background" in model_components
        and kwargs.get("include_background_weight", True)
    ):
        background_weight = ("background.weight",)
        ax_top.plot(
            data["phi1"].flatten(),
            mpars[background_weight].flatten(),
            color="black",
            label="background[weight]",
        )

    # Legend
    ax.legend(fontsize=kwargs.get("legend_fontsize", plt.rcParams["font.size"]))
    ax_top.legend(fontsize=kwargs.get("top_legend_fontsize", plt.rcParams["font.size"]))
    ax_top.set_yscale(kwargs.get("top_yscale", "linear"))


@add_savefig_option
@with_tight_layout
@make_tuple("components", "coords")
def astrometric_model_panels(
    model: Model[Array],
    /,
    data: Data[Array],
    mpars: Params[Array],
    *,
    components: tuple[str, ...] = ("stream",),
    coords: tuple[str, ...] = ("phi2",),
    **kwargs: Any,
) -> Figure:
    r"""Diagnostic plot of the model.

    Parameters
    ----------
    model : Model, positional-only
        The model to plot.

    data : Data
        The data to plot. Must be sorted by :math:`\phi_1`.
    mpars : Params[Array]
        The prediction to plot.

    components : tuple[str, ...], keyword-only
        The component(s) to plot.
    coords : tuple[str, ...], keyword-only
        The coordinate(s) to plot.
    savefig : pathlib.Path or str or None, keyword-only
        The path to save the figure to.
    **kwargs : Any
        Additional keyword arguments.

        - figsize : tuple[float, float], optional
           Uses ``plt.rcParams["figure.figsize"]`` by default.
        - legend_fontsize : float, optional
           Uses ``plt.rcParams["legend.fontsize"]`` by default.
        - top_legend_fontsize : float, optional
           Uses ``plt.rcParams["legend.fontsize"]`` by default.
        - top_yscale : str, optional
           Uses ``"linear"`` by default.
        - include_background_weight : bool, optional
           Whether to include the backgground weighgt. `True` by default.
        - include_total_weight : bool, optional
           Whether to include the total weight. `True` by default.

    Returns
    -------
    Figure
        The figure that was plotted.
    """
    # Now figure out how many rows and columns we need
    figsize = kwargs.pop("figsize", plt.rcParams["figure.figsize"])
    fig = plt.figure(constrained_layout=True, figsize=figsize)

    # TODO! option to divide into multiple rows
    gs = gridspec.GridSpec(1, len(coords), figure=fig)  # Main gridspec

    for i, cn in enumerate(coords):
        _plot_coordinate_panel(
            fig=fig,
            gs=gs[i],
            coord=cn,
            data=data,
            mpars=mpars,
            components=components,
            model_components=model.components,
            kwargs=kwargs,
        )

    return fig
