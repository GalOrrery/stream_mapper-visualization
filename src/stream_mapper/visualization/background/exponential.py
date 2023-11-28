"""Scipy Bounded Exponential Distribution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.stats import rv_continuous

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray


class exponential_like_distribution(rv_continuous):  # noqa: N801
    """Bounded Exponential Distribution.

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
    small_m_approx_threshold : float, optional keyword-only
        When to switch to an approximation of the PDF
        that is valid for small values of m.
    """

    def __init__(  # noqa: PLR0913
        self,
        slope: float,
        a: float,
        b: float,
        xtol: float = 1e-14,
        seed: int | None | np.random.RandomState | np.random.Generator = None,
        *,
        small_m_approx_threshold: float = 1e-4,
    ) -> None:
        if a > b:
            msg = "a must be less than b"
            raise ValueError(msg)
        self.m: np.float64 = np.float64(slope)
        self._bma: np.float64 = np.float64(b - a)
        self.small_m_approx_threshold = small_m_approx_threshold
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
        self, x: int | NDArray[floating[Any]], *args: Any
    ) -> NDArray[floating[Any]]:
        # Mathematically this is correct, but as m approaches 0, the pdf becomes
        # indeterminate.
        if np.abs(self.m) > self.small_m_approx_threshold:
            out = (-self.m * np.exp(-self.m * (x - self.a))) / (
                np.expm1(-self.m * (self.b - self.a))
            )

        # Instead, we use the order-3 Taylor expansion
        # of the exponential function around m=0.
        else:
            m, a, bma = self.m, self.a, self._bma
            xma = x - a
            out = np.array(
                1 / bma
                + (m * (0.5 - xma / bma))
                + (m**2 / 2 * (bma / 6 - xma + xma**2 / bma))
                + ((m**3 * (2 * xma - bma) * xma * (bma - xma)) / (12 * bma))
            )

        return cast("NDArray[floating[Any]]", out)

    def _cdf(
        self, x: float | NDArray[floating[Any]], *args: Any
    ) -> NDArray[floating[Any]]:
        a: np.float64 = self.a
        b: np.float64 = self.b
        m: np.float64 = self.m
        # Mathematically this is correct, but as m approaches 0, the cdf becomes
        # indeterminate.
        if np.abs(self.m) > self.small_m_approx_threshold:
            out = (np.exp(m * (a - x) - 1)) / (np.exp(m * (a - b)) - 1)

        # Instead, we use the order-3 Taylor expansion of the pdf around m=0.
        else:
            out = np.array(
                ((x - a) / (b - a) / 24)
                * (
                    12
                    + m
                    * (2 * a + 2 * b + a * b * m - (4 + m * (a + b)) * x + m * x**2)
                )
            )

        return cast("NDArray[floating[Any]]", out)

    def _ppf(self, q: float, *args: Any) -> NDArray[floating[Any]]:
        a, b, m = self.a, self.b, self.m
        # Mathematically this is correct, but as m approaches 0, the ppf becomes
        # indeterminate.
        if np.abs(self.m) > self.small_m_approx_threshold:
            out = a - np.log(1 + q * (np.exp(-m * (b - a)) - 1)) / m

        # Instead, we use the order-3 Taylor expansion around m=0.
        # TODO! this does not converge to the correct answer!
        else:
            out = np.array(
                a
                + (b - a) * (1 - 0.5 * (b - a) * m + (b - a) ** 2 * m**2 / 6) * q
                + 0.5 * m * (b - a) ** 2 * (1 - (b - a) * m) * q**2
                + (b - a) ** 3 * m**2 * q**3 / 3
            )

        return cast("NDArray[floating[Any]]", out)
