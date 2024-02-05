"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

import inspect
from functools import wraps
from typing import TYPE_CHECKING, ParamSpec, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    P = ParamSpec("P")
    R = TypeVar("R")
    ForA = TypeVar("ForA", Figure, Axes)


###############################################################################


def with_sorter(plotting_func: Callable[P, R]) -> Callable[P, R]:
    """Add a sorter to a plotting function.

    Parameters
    ----------
    plotting_func : Callable[P, R]
        The plotting function to add a sorter to.

    Returns
    -------
    Callable[P, R]
        The plotting function with a sorter.

    """
    sig = inspect.signature(plotting_func)

    @wraps(plotting_func)
    def with_sorter_inner(*args: P.args, **kwargs: P.kwargs) -> R:
        # Get the sorter, or create it if it doesn't exist.
        if kwargs.get("sorter") is None:
            ba = sig.bind_partial(*args, **kwargs)
            ba.apply_defaults()
            data = ba.arguments["data"]
            sorter = np.argsort(data["phi1"].flatten())
            kwargs["sorter"] = sorter

        # Call the plotting function.
        return plotting_func(*args, **kwargs)

    return with_sorter_inner


def make_tuple(*arg_names: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Make a function accept a single argument as a tuple.

    Parameters
    ----------
    *arg_names : str
        The name of the argument(s) to make into a tuple of strings.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
        The decorator.

    """

    def make_tuple_inner(plotting_func: Callable[P, R]) -> Callable[P, R]:
        """Make a function accept a single argument as a tuple.

        Parameters
        ----------
        plotting_func : Callable[P, R]
            The plotting function to make a tuple.

        Returns
        -------
        Callable[P, R]
            The plotting function with a tuple.

        """
        sig = inspect.signature(plotting_func)

        @wraps(plotting_func)
        def make_tuple_inner_inner(*args: P.args, **kwargs: P.kwargs) -> R:
            # Get the argument, or create it if it doesn't exist.
            ba = sig.bind_partial(*args, **kwargs)
            ba.apply_defaults()
            for arg_name in arg_names:
                arg = ba.arguments[arg_name]
                ba.arguments[arg_name] = (arg,) if not isinstance(arg, tuple) else arg

            # Call the plotting function.
            return plotting_func(*ba.args, **ba.kwargs)

        return make_tuple_inner_inner

    return make_tuple_inner
