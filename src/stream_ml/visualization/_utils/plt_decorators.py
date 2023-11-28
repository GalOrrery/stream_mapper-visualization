"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

import os
import pathlib
import warnings
from functools import wraps
from typing import TYPE_CHECKING, ParamSpec, TypeVar

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes

    P = ParamSpec("P")
    R = TypeVar("R")
    ForA = TypeVar("ForA", Figure, Axes)


###############################################################################


def add_savefig_option(plotting_func: Callable[P, ForA]) -> Callable[P, ForA]:
    """Add a savefig method to a plotting function.

    Parameters
    ----------
    plotting_func : Callable[ParamSpec, Figure | Axes]
        The plotting function to add a savefig method to.

    Returns
    -------
    Callable[ParamSpec, Figure | Axes]
        The plotting function with a savefig method.
    """

    @wraps(plotting_func)
    def add_savefig_optiodataner(*args: P.args, **kwargs: P.kwargs) -> ForA:
        """Save the figure to a file.

        Parameters
        ----------
        *args : Any
            The arguments to pass to the plotting function.
        **kwargs : P.kwargs
            The keyword arguments to pass to the plotting function.

        Returns
        -------
        `~matplotlib.figure.Figure`
            The figure that was plotted.
        """
        # Get the savefig keyword argument.
        savefig = kwargs.pop("savefig", None)
        if not isinstance(savefig, type(None) | str | os.PathLike | pathlib.Path):
            msg = f"savefig must be a pathlib.Path object, not {type(savefig)}."
            raise TypeError(msg)

        # Call the plotting function.
        fig_or_ax = plotting_func(*args, **kwargs)

        # Save the figure if a path was given.
        if savefig is not None:
            fig = fig_or_ax if isinstance(fig_or_ax, Figure) else fig_or_ax.figure
            fig.savefig(pathlib.Path(savefig).as_posix())

        return fig_or_ax

    return add_savefig_optiodataner


def with_tight_layout(plotting_func: Callable[P, Figure]) -> Callable[P, Figure]:
    """Add tight layout to a plotting function.

    Parameters
    ----------
    plotting_func : Callable[P, Figure]
        The plotting function to add tight layout to.

    Returns
    -------
    Callable[P, Figure]
        The plotting function with tight layout.
    """

    @wraps(plotting_func)
    def with_tight_layout_inner(*args: P.args, **kwargs: P.kwargs) -> Figure:
        fig = plotting_func(*args, **kwargs)

        # Tight layout
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()

        return fig

    return with_tight_layout_inner


def with_ax(plotting_func: Callable[P, R]) -> Callable[P, R]:
    """Pass ax to a plotting function.

    Parameters
    ----------
    plotting_func : Callable[P, R]
        The plotting function to add axes to.

    Returns
    -------
    Callable[P, R]
        The plotting function with axes.
    """

    @wraps(plotting_func)
    def with_ax_inner(*args: P.args, **kwargs: P.kwargs) -> R:
        # Create them if they don't exist.
        kwargs.setdefault("ax", plt.gca())

        # Call the plotting function.
        return plotting_func(*args, **kwargs)

    return with_ax_inner
