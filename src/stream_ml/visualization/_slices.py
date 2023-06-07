"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

__all__: list[str] = []

import itertools
import re
from typing import TYPE_CHECKING, Any, cast

import numexpr as ne
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch

from stream_ml.visualization._defaults import LABEL_DEFAULTS
from stream_ml.visualization._utils.arg_decorators import make_tuple
from stream_ml.visualization._utils.labels import set_label
from stream_ml.visualization._utils.plt_decorators import (
    add_savefig_option,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from astropy.table import QTable
    from astropy.units import Quantity
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from streaam_ml.core.data import Data

    from stream_ml.core.typing import ArrayLike


CoordT = str | tuple[str, str]  # hist vs plot
IndexT = tuple[int, int]


########################################################################


# TODO: Add option for CMD plot
@add_savefig_option
@make_tuple("coords")
def plot_coordinates_in_slices(  # noqa: PLR0913
    data: QTable | Data[ArrayLike],
    /,
    x_edges: tuple[Quantity | NDArray[Any], ...],
    *,
    xcoord: str = "phi1",
    ycoord: str = "phi2",  # top plot
    coords: tuple[CoordT, ...],
    labels: Mapping[str, str] = LABEL_DEFAULTS,
    figure_kwargs: dict[str, Any] | None = None,
    axtop_kwargs: dict[str, Any] | None = None,
    ax_kwargs: dict[CoordT | IndexT, dict[str, Any]] | None = None,
) -> Figure:
    r"""Bin plot.

    Parameters
    ----------
    data : `~astropy.table.QTable` or `~stream_ml.core.data.Data`
        Data with the column names ``coords``. Must have at least ``phi1`` and
        ``phi2``.
    x_edges : tuple[Quantity, ...]
        Tuple of phi1 bounds.

    xcoord, ycoord : str, optional keyword-only
        Coordinate to plot on the x/y-axis. Default is ``"phi1"``/``"phi2"``.

    coords : tuple[str | tuple[str, str], ...], keyword-only
        Tuple of coordinate names to plot:

        - str: plot the coordinate as a histogram
        - tuple[str, str]: plot the coordinate with the method specified by
          "ax_kwargs" (the default is a scatter plot).

        The strings are evaluated as expreessions with :mod:`numexpr`, with the
        substitution that ``[x]`` is replaced with ``data[x].flatten()``.

    labels : dict[str, str], optional keyword-only
        Dictionary of labels for the columns, by default:

        .. code-block:: python
            {
                "phi1": f"$\\phi_1$",
                "phi2": r"$\\phi_2$",
                "parallax": r"$\varpi$",
                "pm_phi1_cosphi2_unrefl": r"$\\mu_{phi_1}^*$",
                "pm_phi2_unrefl": r"$\\mu_{phi_2}$",
            }

    figure_kwargs : dict[str, Any] | None, optional keyword-only
        Keyword arguments passed to :func:`~matplotlib.pyplot.figure`.

    axtop_kwargs : dict[str, Any] | None, optional keyword-only
        Keyword arguments passed to :func:`~matplotlib.axes.Axes.plot`.

    ax_kwargs : dict | None, optional keyword-only
        Keyword arguments passed to each axis in the grid. Keys can be either:

        - an index (tuple[int, int]). This will be applied to the axis at that
          index.
        - a coordinate name (str). This will be applied to all axes with that
          coordinate.

        The value should be a dictionary of keyword arguments passed to the
        plotting method. If the relevant plot in ``coords`` is a string then the
        plotting method is :func:`~matplotlib.axes.Axes.hist`. If "kind" is in
        the dictionary then the plotting method is specified by ``getattr(ax,
        ax_kwargs[coord]["kind"])``, otherwise it is
        :func:`~matplotlib.axes.Axes.scatter`.

    Returns
    -------
    `matplotlib.figure.Figure`
        The figure.
    """
    fig, ax0, axes = _make_fig_and_axes(
        x_edges=x_edges,
        coords=coords,
        figure_kwargs=figure_kwargs or {"figsize": (20, 20)},
    )

    # -----------------------------------------------------
    # Phi1-Phi2 top plot
    # TODO: make so there can be many top plots, e.g.
    # (phi1, phi2), (phi1, parallax).

    ax0.plot(
        data[xcoord].flatten(),
        data[ycoord].flatten(),
        c="black",
        marker=",",
        linestyle="none",
        **(axtop_kwargs or {}),
    )
    set_label(labels.get(xcoord, xcoord), ax=ax0, which="x", fontsize=15)
    set_label(labels.get(ycoord, ycoord), ax=ax0, which="y", fontsize=15)

    # -----------------------------------------------------
    # Phi1-slice column histograms

    ax_kw = ax_kwargs or {}
    hist_kw_default = {"bins": 10, "color": "gray", "density": True}

    # Iterate over phi1 slices, making a column of histograms for each component
    for col, (left, right) in enumerate(itertools.pairwise(x_edges)):
        # Per-coordinate plot
        coldata = cast(
            "QTable | Data[ArrayLike]",
            data[(data[xcoord].flatten() >= left) & (data[xcoord].flatten() < right)],
        )

        # Iter through rows of the column
        for row, (ax, coord) in enumerate(zip(axes[:, col], coords, strict=True)):
            # Get the kwargs for this plot
            kw = ax_kw.get((row, col), ax_kw.get(coord, {})).copy()

            ld = _make_local_dict(coord, coldata)  # numexpr local dict

            if isinstance(coord, str):
                value = ne.evaluate(re.sub(_brkt_re, _strip_brkt, coord), local_dict=ld)
                ax.hist(value, **(hist_kw_default | kw))

                # Settings
                set_label(labels.get(coord, coord), ax=ax, which="x")
                ax.set_ylabel("fraction")

            else:
                kind = kw.pop("kind", "scatter")
                xstr, ystr = coord
                xvalue = ne.evaluate(re.sub(_brkt_re, _strip_brkt, xstr), local_dict=ld)
                yvalue = ne.evaluate(re.sub(_brkt_re, _strip_brkt, ystr), local_dict=ld)
                getattr(ax, kind)(xvalue, yvalue, **kw)

                # Settings
                ax.set_xlabel(labels.get(xstr, xstr))
                ax.set_ylabel(labels.get(ystr, ystr))

            ax.grid(visible=True)
            ax.set_axisbelow(True)  # grid lines behind data

        # Connect coordinate plots to top plot
        _connect_slices_to_top(fig, ax0, axes[0, col], left=left, right=right)

    # Adjust y-axis labels for all but the first column
    for row, col in itertools.product(range(len(coords)), range(1, len(x_edges) - 1)):
        axes[row, col].tick_params(axis="y", which="both", left=False, labelleft=False)
        axes[row, col].set_ylabel("")

    return fig


# ----------------------------------------------------------------------------

_brkt_re = r"\[(.*?)\]"


def _strip_brkt(m: re.Match[str]) -> str:
    return m[0][1:-1]


def _ensure_brkt(string: str) -> str:
    return f"[{string}]" if (string.find("[") == -1) else string


def _make_local_dict(
    string: str | tuple[str, ...], data: Data[ArrayLike] | QTable
) -> dict[str, ArrayLike]:
    string_ = string if isinstance(string, tuple) else (string,)
    all_matches = itertools.chain(
        *(re.finditer(_brkt_re, _ensure_brkt(c)) for c in string_)
    )
    # TODO! make the key so that it can be used in numexpr
    #       e.g. "x [Mpc]" -> "x_mpc"
    return {_strip_brkt(m): data[_strip_brkt(m)].flatten() for m in all_matches}


# ----------------------------------------------------------------------------


def _make_fig_and_axes(
    *,
    x_edges: tuple[Quantity | NDArray[Any], ...],
    coords: tuple[str | tuple[str, str], ...],
    figure_kwargs: dict[str, Any],
) -> tuple[Figure, Axes, NDArray[np.object_]]:
    # # Connecting line between top plot slice edges and the histograms
    # con_y_top = table["phi2"].min().value
    # Number of rows per column
    nrows = len(coords)

    # Make the plot and GridSpec, with 2 rows and one column per phi1 slice
    fig = plt.figure(**figure_kwargs)
    gs0 = gridspec.GridSpec(
        2,
        len(x_edges) - 1,
        figure=fig,
        height_ratios=(1, nrows),
        wspace=0.02,
        hspace=0.08,
    )

    # Top plot
    ax_top = fig.add_subplot(gs0[0, :])

    # Make axes array
    axes = np.empty((nrows, len(x_edges) - 1), dtype=object)
    # Iterate over columns
    for col in range(len(x_edges) - 1):
        # Make sub-gridspec column of sub-axes
        gsi = gs0[1:, col].subgridspec(nrows, 1, hspace=0.3)
        # Add subplot axes to array
        for row in range(len(coords)):
            axes[row, col] = fig.add_subplot(gsi[row], sharey=axes[row, 0])

    return fig, ax_top, axes


def _connect_slices_to_top(
    fig: Figure,
    axes1: Axes,
    axes2: Axes,
    left: Quantity | NDArray[Any],
    right: Quantity | NDArray[Any],
) -> None:
    """Connect column of axes to top plot with lines."""
    # Add edges to top plot
    axes1.axvline(left, color="tab:red", ls="--")
    axes1.axvline(right, color="tab:red", ls="--")

    # Add connection
    con = ConnectionPatch(
        xyA=(np.array(left), axes1.get_ylim()[0]),
        xyB=(0, 1),
        coordsA="data",
        coordsB="axes fraction",
        axesA=axes1,
        axesB=axes2,
        color="tab:red",
        ls="--",
        zorder=-200,
    )
    axes1.add_artist(con)

    con = ConnectionPatch(
        xyA=(np.array(right), axes1.get_ylim()[0]),
        xyB=(1, 1),
        coordsA="data",
        coordsB="axes fraction",
        axesA=axes1,
        axesB=axes2,
        color="tab:red",
        ls="--",
        zorder=-200,
    )
    axes1.add_artist(con)
