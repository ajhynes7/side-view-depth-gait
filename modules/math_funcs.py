"""Math functions."""

import numpy as np
from statsmodels.robust import mad


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


def within_mad(array, c=1):
    """
    Check if elements are within limits defined by the median absolute deviation (MAD).

    The limits are defined as median +- c * MAD.

    Parameters
    ----------
    array : array_like
        (n,) Input 1D array.
    c : number
        Coefficient of MAD (default 1).

    Returns
    -------
    ndarray
        (n,) Boolean mask.
        Element is true if value is within the limits; false otherwise.

    Examples
    --------
    >>> array = [1, 2, 3, 4, 100, 5]

    >>> mad_filter(array)
    array([False, True, True, True, False, True])

    >>> mad_filter(array, c=3)
    array([True, True, True, True, False, True])

    """
    array = np.array(array)

    median = np.median(array)
    mad_ = mad(array)

    limits_mad = limits(median, c * mad_)

    return np.logical_and(array > limits_mad[0], array < limits_mad[1])
