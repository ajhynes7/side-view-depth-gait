"""Module for statistical calculations."""

from typing import NamedTuple

from numpy import ndarray

import modules.math_funcs as mf


class BlandAltman(NamedTuple):
    """Container for Bland-Altman results."""

    bias: float
    lower_limit: float
    upper_limit: float
    range_: float


def relative_difference(x: ndarray, y: ndarray) -> ndarray:
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
    >>> import numpy as np

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


def relative_error(measured: ndarray, actual: ndarray) -> ndarray:
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
    >>> import numpy as np

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
    >>> import numpy as np

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
