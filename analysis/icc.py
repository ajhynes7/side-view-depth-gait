"""Intraclass correlation coefficients."""

from dataclasses import dataclass, astuple

import numpy as np

from modules.typing import array_like


@dataclass
class AnovaSquares:
    """Dataclass for Sum of Squares and Mean Squares from ANOVA."""

    BS: float  # Between subjects
    BM: float  # Between measurements

    WS: float  # Within subjects
    WM: float  # Withing measurements

    T: float  # Total
    E: float  # Error

    def __iter__(self):
        """Implement iteration for easier unit testing."""
        return iter(astuple(self))


@dataclass
class SumSquares(AnovaSquares):
    """Sum of Squares from ANOVA for calculating ICCs."""

    pass


@dataclass
class MeanSquares(AnovaSquares):
    """Mean squares from ANOVA for calculating ICCs."""

    pass


def anova_sum_squares(X: array_like) -> SumSquares:
    """Return sum of Squares from ANOVA for calculating ICCs."""

    X = np.array(X)
    n, k = X.shape

    S = X.mean(axis=1)  # Means of each subject (row)
    M = X.mean(axis=0)  # Mean of each measurement/rater (column)

    x_bar = X.mean()

    BS = k * np.sum((S - x_bar) ** 2)
    BM = n * np.sum((M - x_bar) ** 2)

    WS = np.sum((X - S.reshape(-1, 1)) ** 2)
    WM = np.sum((X - M) ** 2)

    T = np.sum((X - x_bar) ** 2)

    E = T - BS - BM  # Sum of squares, error

    return SumSquares(BS, BM, WS, WM, T, E)


def anova_mean_squares(ss: SumSquares, n: int, k: int):
    """Return mean squares from ANOVA for calculating ICCs."""

    BS = ss.BS / (n - 1)  # Mean square between subjects
    BM = ss.BM / (k - 1)  # Mean square between measurements

    WS = ss.WS / (n * (k - 1))  # Mean square within subjects
    WM = ss.WM / (k * (n - 1))  # Mean square within measurements

    T = ss.T / (n * k - 1)  # Mean square, total
    E = ss.E / ((n - 1) * (k - 1))  # Mean square, error

    return MeanSquares(BS, BM, WS, WM, T, E)


def icc(X: array_like, form: int = 1) -> np.float:
    """
    Compute intraclass correlation coefficients (ICCs).

    Parameters
    ----------
    X: (n, k) ndarray
        Array for n subjects and k measurements/raters.
    form: int
        1, 2 (agreement), or 3 (consistency).

    References
    ----------
    Liljequist, D., Elfving, B., & Roaldsen, K. S. (2019).
    Intraclass correlation–A discussion and demonstration of basic features.
    PloS one, 14(7).

    Examples
    --------
    >>> from analysis.icc import icc

    >>> X = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    >>> icc(X).round(4)
    1.0
    >>> icc(X, form=2).round(4)
    1.0
    >>> icc(X, form=3).round(4)
    1.0

    >>> X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    >>> icc(X).round(4)
    0.8182
    >>> icc(X, form=2).round(4)
    0.8333
    >>> icc(X, form=3).round(4)
    1.0

    >>> X = [[7, 9], [10, 13], [8, 4]]
    >>> icc(X).round(4)
    0.5246
    >>> icc(X, form=2).round(4)
    0.463
    >>> icc(X, form=3).round(4)
    0.3676

    >>> X = [[60, 61], [60, 65], [58, 62], [10, 10]]
    >>> icc(X).round(4)
    0.992
    >>> icc(X, form=2).round(4)
    0.992
    >>> icc(X, form=3).round(4)
    0.9957

    """
    X = np.array(X)
    n, k = X.shape

    ss = anova_sum_squares(X)
    ms = anova_mean_squares(ss, n, k)

    if form == 1:
        num = ms.BS - ms.WS
        denom = ms.BS + (k - 1) * ms.WS

    elif form == 2:
        num = ms.BS - ms.E
        denom = ms.BS + (k - 1) * ms.E + (k / n) * (ms.BM - ms.E)

    elif form == 3:
        num = ms.BS - ms.E
        denom = ms.BS + (k - 1) * ms.E

    return num / denom
