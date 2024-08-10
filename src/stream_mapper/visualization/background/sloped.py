"""Scipy Bounded Exponential Distribution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.stats import rv_continuous

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray


class sloped_distribution(rv_continuous):  # noqa: N801
    """Sloped distribution.

    Parameters
    ----------
    slope : float
        The slope.
    a : float
        Lower bound of the distribution.
    b : float
        Upper bound of the distribution.
    xtol : float, optional
        The tolerance used when calculating the inverse of the CDF.
    seed : int or None, optional
        Seed for the random number generator.

    """

    def __init__(
        self,
        slope: float,
        a: float,
        b: float,
        xtol: float = 1e-14,
        seed: int | None | np.random.RandomState | np.random.Generator = None,
    ) -> None:
        if a > b:
            msg = "a must be less than b"
            raise ValueError(msg)
        if np.abs(slope) > 2 / (b - a) ** 2:
            msg = "slope must be less than 2 / (b - a)^2"
            raise ValueError(msg)
        self.m: np.float64 = np.float64(slope)
        super().__init__(
            0,
            np.float64(a),
            np.float64(b),
            xtol=xtol,
            badvalue=None,
            name=None,
            longname=None,
            shapes=None,
            seed=seed,
        )

    def _pdf(
        self, x: float | NDArray[floating[Any]], *args: Any
    ) -> NDArray[floating[Any]]:
        return cast(
            "NDArray[floating[Any]]",
            np.array(
                self.m * (x - (self.b + self.a) / 2) + 1 / (self.b - self.a),
                dtype=float,
            ),
        )

    def _cdf(
        self, x: float | NDArray[floating[Any]], *args: Any
    ) -> NDArray[floating[Any]]:
        a, b, m = self.a, self.b, self.m
        return cast(
            "NDArray[floating[Any]]",
            np.array(
                (x - a) * (2 + (b - a) * m * (x - b)) / (2 * (b - a)), dtype=float
            ),
        )

    def _ppf(
        self, q: float | NDArray[floating[Any]], *args: Any
    ) -> NDArray[floating[Any]]:
        a, b, m = self.a, self.b, self.m
        return cast(
            "NDArray[floating[Any]]",
            np.array(
                (
                    2
                    + (a - b) * (a + b) * m
                    - np.sqrt((-2 + (a - b) ** 2 * m) ** 2 + 8 * (a - b) ** 2 * m * q)
                )
                / (2 * (a - b) * m),
                dtype=float,
            ),
        )
