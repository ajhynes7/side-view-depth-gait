"""Math functions."""

import numpy as np
from dpcontracts import require


def limits(x, tolerance):
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


@require("The inputs must have length two.", lambda args: all(len(x) == 2 for x in [args.limits_a, args.limits_b]))
@require(
    "The upper limit must be >= the lower limit.",
    lambda args: all(lims[1] >= lims[0] for lims in [args.limits_a, args.limits_b]),
)
def check_overlap(limits_a, limits_b):
    """
    Check if two ranges of numbers overlap.

    Parameters
    ----------
    limits_a, limits_b : tuple
        Tuple of form (min, max).

    Returns
    -------
    bool
        True if the ranges overlap; false otherwise.

    Examples
    --------
    >>> check_overlap([0, 1], [2, 3])
    False

    >>> check_overlap([1, 1], [1, 2])
    True

    >>> check_overlap([4, 7], [5, 10])
    True

    >>> check_overlap([5, 10], [4, 7])
    True

    """
    min_a, max_a = limits_a
    min_b, max_b = limits_b

    return min_a <= max_b and max_a >= min_b


def norm_ratio(a, b):
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
