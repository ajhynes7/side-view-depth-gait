import numpy as np

from statsmodels import robust

import modules.general as gen


class BlandAltman:
    """
    Bland-Altman analysis.

    Compare measurements of a new device to those of a validated device.

    Parameters
    ----------
    x_new : ndarray
        Measurements of new device.
    x_valid : ndarray
        Measurements of validate device.
    percent : bool, optional
        If True, the percent difference is used.
        If False (default) the relative difference is used.

    Attributes
    ----------
    means : ndarray
        Means of corresponding measurements.
    differences : ndarray
        Differences between corresponding measurements.
    bias : float
        Mean of the differences.
        Indicates the bias of the new device compared to the validated one.
    limits_of_agreement : tuple
        Tuple of form (lower_limit, upper_limit).
        Bias minus/plus 1.96 standard deviations.
    range : float
        Range of the limits of agreement.
        A low range indicates stronger agreement.

    Examples
    --------
    >>> x_new = np.array([1, 2, 3])
    >>> x_valid = np.array([2, 2, 3])

    >>> results = BlandAltman(x_new, x_valid, percent=True)

    >>> results.means
    array([1.5, 2. , 3. ])

    >>> np.round(results.differences)
    array([-67.,   0.,   0.])

    >>> round(results.bias)
    -22.0

    >>> np.round(results.limits_of_agreement)
    array([-84.,  39.])

    >>> round(results.range)
    123.0

    """

    def __init__(self, x_new, x_valid, percent=False):

        self.x_new, self.x_valid = x_new, x_valid
        self.percent = percent

    @property
    def means(self):

        return (self.x_new + self.x_valid) / 2

    @property
    def differences(self):

        diffs = relative_difference(self.x_new, self.x_valid)

        if self.percent:
            diffs *= 100

        return diffs

    @property
    def bias(self):

        return self.differences.mean()

    @property
    def limits_of_agreement(self):

        standard_dev = self.differences.std()

        lower_lim, upper_lim = gen.limits(self.bias, 1.96 * standard_dev)

        return lower_lim, upper_lim

    @property
    def range(self):

        lower_lim, upper_lim = self.limits_of_agreement

        return upper_lim - lower_lim


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
    >>> x = np.array([3, 3, 4])
    >>> y = np.array([1, 2, 4])

    >>> relative_difference(x, y)
    array([1. , 0.4, 0. ])

    >>> relative_difference(2, 3)
    -0.4

    """
    difference = x - y

    average = (x + y) / 2

    rel_difference = difference / average

    if absolute:
        rel_difference = abs(rel_difference)

    return rel_difference


def relative_error(measured, actual, absolute=False):
    """
    Return the relative errors between measured and actual values.

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
