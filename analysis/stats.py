"""Module for statistical calculations."""

from collections import namedtuple

import numpy as np
import pandas as pd
from statsmodels import robust

import analysis.math_funcs as mf


def relative_difference(x, y, absolute=False):
    """
    Relative difference between values x and y.

    Calculated as (x - y) / mean(x, y).

    Parameters
    ----------
    x, y : {int, float, ndarray}
        Input values or arrays.
    absolute : bool, optional
        If True, the absolute relative error is returned
        (the default is False).

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

    >>> relative_difference(2, 3, absolute=True)
    0.4

    """
    difference = x - y
    mean_ = (x + y) / 2

    rel_difference = difference / mean_

    if absolute:
        rel_difference = abs(rel_difference)

    return rel_difference


def relative_error(measured, actual, absolute=False):
    """
    Return the relative errors between measured and actual values.

    Calculated as (measure - actual) / actual.

    Parameters
    ----------
    measured : {int, float, ndarray}
        Measured value or array.

    actual : {int, float, ndarray}
        Actual value or array.

    absolute : bool, optional
        If True, the absolute relative error is returned
        (the default is False).

    Returns
    -------
    error : {float, ndarray}
        Relative error.

    Examples
    --------
    >>> relative_error(2, 5)
    -0.6

    >>> relative_error(3, 5, absolute=True)
    0.4

    >>> x = np.array([1, 2])
    >>> y = np.array([2, 2])
    >>> relative_error(x, y, absolute=True)
    array([0.5, 0. ])

    """
    error = (measured - actual) / actual

    if absolute:
        error = abs(error)

    return error


def mad_outliers(x, c):
    """
    Remove outliers from an array using the median absolute deviation (MAD).

    Values beyond the median ± c(MAD) are set to NaN.

    Parameters
    ----------
    x : array_like
        Input array.
    c : {int, float}
        Coefficient for MAD.

    Returns
    -------
    x_filtered : ndarray
        Array with same shape as input x, but with outliers set to NaN
        and all values as floats.

    Examples
    --------
    >>> mad_outliers([2.0, 3.0, 100.0, 3.0], 2.5)
    array([ 2.,  3., nan,  3.])

    >>> mad_outliers([5, 6, 4, 20], 3)
    array([ 5.,  6.,  4., nan])

    """
    mad = robust.mad(x)
    median = np.median(x)

    lower_bound = median - c * mad
    upper_bound = median + c * mad

    x_filtered = np.array(x).astype(float)
    x_filtered[np.logical_or(x < lower_bound, x > upper_bound)] = np.nan

    return x_filtered


def bland_altman(differences):
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

    params = 'bias lower_limit upper_limit range_'

    BlandAltman = namedtuple('BlandAltman', params)

    return BlandAltman(
        bias=bias,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        range_=upper_limit - lower_limit)


def compare_measurements(df_1, df_2, compare_funcs):
    """
    Compare measurements taken by two devices.

    Parameters
    ----------
    df_1, df_2 : DataFrame
        Measurements of devices 1 and 2.
        Columns are measurement names
        Rows are individual measurements.
    compare_funcs : dict
        Keys are metric names, e.g., relative error
        Values are functions of form column_1, column_2 -> scalar

    Returns
    -------
    df_results : DataFrame
        Columns are measurement names, e.g. stride length
        Rows are metric names, e.g. relative error

    """
    measurements = df_1.columns
    metrics = compare_funcs.keys()

    df_results = pd.DataFrame(index=metrics, columns=measurements)

    for measurement in measurements:
        for metric, func in compare_funcs.items():

            df_results.loc[metric, measurement] = func(df_1[measurement],
                                                       df_2[measurement])

    return df_results
