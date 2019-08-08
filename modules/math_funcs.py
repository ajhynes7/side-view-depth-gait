"""Math functions."""

from typing import Tuple

import numpy as np


def limits(x: np.ndarray, tolerance: float) -> Tuple[float, float]:
    """
    Return lower and upper bounds (x ± tolerance).

    Parameters
    ----------
    x : {int, float, ndarray}
        Input value or array
    tolerance : {int, float}
        Tolerance that defines lower and upper limits.

    Returns
    -------
    lower_lim, upper_lim : {int, float}
        Lower and upper limits.

    Examples
    --------
    >>> limits(10, 2)
    (8, 12)

    >>> limits(np.array([2, 3]), 5)
    (array([-3, -2]), array([7, 8]))

    """
    lower_lim, upper_lim = x - tolerance, x + tolerance

    return lower_lim, upper_lim


def norm_ratio(a: float, b: float) -> float:
    """
    Return a normalized ratio between two positive inputs.

    If ratio a / b is greater than one, the reciprocal is returned instead.
    Returns nan if either input is zero.

    Parameters
    ----------
    a, b : float
        Positive inputs.

    Returns
    -------
    float
        Ratio between a and b, with value in range (0, 1].

    Examples
    --------
    >>> norm_ratio(5, 10)
    0.5

    >>> norm_ratio(10, 5)
    0.5

    >>> norm_ratio(5, 0)
    nan

    >>> norm_ratio(5, 5)
    1.0

    """
    if a == 0 or b == 0:
        return np.nan

    ratio = np.divide(a, b)

    if ratio > 1:
        ratio = np.reciprocal(ratio)

    return ratio
