"""Module for statistical calculations."""

from typing import NamedTuple, Tuple

import numpy as np
from numpy import ndarray

import modules.math_funcs as mf
from modules.typing import array_like


class BlandAltman(NamedTuple):
    """Container for Bland-Altman results."""

    bias: float
    lower_limit: float
    upper_limit: float
    range_: float


def relative_difference(x, y):
    """
    Relative difference between values x and y.

    Calculated as (x - y) / mean(x, y).

    Parameters
    ----------
    x, y : {int, float, ndarray}
        Input values or arrays.

    Returns
    -------
    {float, ndarray}
        Relative difference.

    Examples
    --------
    >>> x = np.array([3, 3, 100])
    >>> y = np.array([1, 2, 110])

    >>> np.round(relative_difference(x, y), 3)
    array([ 1.   ,  0.4  , -0.095])

    >>> relative_difference(2, 3)
    -0.4

    """
    difference = x - y
    mean_ = (x + y) / 2

    return difference / mean_


def relative_error(measured, actual):
    """
    Return the relative errors between measured and actual values.

    Calculated as (measured - actual) / actual.

    Parameters
    ----------
    measured : {int, float, ndarray}
        Measured value or array.

    actual : {int, float, ndarray}
        Actual value or array.

    Returns
    -------
    error : {float, ndarray}
        Relative error.

    Examples
    --------
    >>> relative_error(2, 5)
    -0.6

    >>> x = np.array([1, 2])
    >>> y = np.array([2, 2])

    >>> relative_error(x, y)
    array([-0.5,  0. ])

    """
    return (measured - actual) / actual


def bland_altman(differences: ndarray) -> BlandAltman:
    """
    Calculate measures for Bland-Altman analysis.

    Compare measurements of a new device to those of a validated device.

    Parameters
    ----------
    differences : ndarray
        Differences (relative or absolute) between measurements of two devices.

    Returns
    -------
    BlandAltman : namedtuple
        namedtuple with Bland-Altman parameters.

    Examples
    --------
    >>> measures_1 = np.array([1, 2, 3])
    >>> measures_2 = np.array([2, 2, 3])
    >>> differences = relative_difference(measures_1, measures_2)
    >>> results = bland_altman(differences)

    >>> np.round(results.bias, 2)
    -0.22

    >>> np.round(results.lower_limit, 2)
    -0.84

    >>> np.round(results.upper_limit, 2)
    0.39

    """
    bias, standard_dev = differences.mean(), differences.std()

    lower_limit, upper_limit = mf.limits(bias, 1.96 * standard_dev)

    return BlandAltman(bias=bias, lower_limit=lower_limit, upper_limit=upper_limit, range_=upper_limit - lower_limit)


def icc(matrix: Sequence, form: Tuple[int, int] = (1, 1)) -> float:
    """
    Return an intraclass correlation coefficient (ICC).

    Parameters
    ----------
    matrix : (N, K) array_like
        Array for N subjects and K raters.
    form : tuple, optional
        The ICC form using Shrout and Fleiss (1979) convention.
        The default is (1, 1).

    Returns
    -------
    float
        ICC of the specified form.

    References
    ----------
    Shrout and Fleiss (1979)
    Koo and Li (2016)

    Examples
    --------
    >>> matrix = [[7, 9], [10, 13], [8, 4]]

    >>> icc(matrix).round(4)
    0.5246
    >>> icc(matrix, form=(2, 1)).round(4)
    0.463
    >>> icc(matrix, form=(3, 1)).round(4)
    0.3676

    >>> matrix = [[60, 61], [60, 65], [58, 62], [10, 10]]

    >>> icc(matrix).round(4)
    0.992
    >>> icc(matrix, form=(2, 1)).round(4)
    0.992
    >>> icc(matrix, form=(3, 1)).round(4)
    0.9957

    """
    matrix = np.array(matrix)

    # Number of subjects (n) and number of raters (k)
    n, k = matrix.shape

    # Total sum of squares
    ss_total = matrix.var(ddof=1) * (n * k - 1)

    # Mean square for rows
    ms_r = matrix.mean(axis=1).var(ddof=1) * k

    # Mean square for residual sources of variance
    ms_w = matrix.var(axis=1, ddof=1).sum() / n

    # Mean square for columns
    ms_c = matrix.mean(axis=0).var(ddof=1) * n

    # Mean square for error
    ms_e = (ss_total - ms_r * (n - 1) - ms_c * (k - 1)) / ((n - 1) * (k - 1))

    if form == (1, 1):
        num = ms_r - ms_w
        denom = ms_r + (k - 1) * ms_w

    elif form == (2, 1):
        num = ms_r - ms_e
        denom = ms_r + (k - 1) * ms_e + k / n * (ms_c - ms_e)

    elif form == (3, 1):
        num = ms_r - ms_e
        denom = ms_r + (k - 1) * ms_e

    return num / denom
