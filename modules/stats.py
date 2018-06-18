import numpy as np

from statsmodels import robust

import modules.general as gen


class BlandAltman:

    def differences(x, y, percent=False):
        """
        Return means and differences of two sets of measurements.

        Parameters
        ----------
        x : ndarray
            Measurements from device A.
        y : ndarray
            Measurements from device B.
        percent : bool, optional
            If True, the percent difference is returned.
            If False (default) the relative difference is returned.

        Returns
        -------
        means : ndarray
            Means of measurements from devices A and B.
        diffs : ndarray
            Differences of measurements.

        Examples
        --------
        >>> x = np.array([1, 2, 3])
        >>> y = np.array([2, 2, 3])

        >>> means, diffs = BlandAltman.differences(x, y, percent=True)

        >>> means
        array([1.5, 2. , 3. ])

        >>> np.round(diffs)
        array([-67.,   0.,   0.])

        """
        means = (x + y) / 2

        diffs = relative_difference(x, y)

        if percent:
            diffs *= 100

        return means, diffs

    def limits_of_agreement(diffs):
        """
        Calculate bias and limits of agreement for Bland Altman analysis.

        Parameters
        ----------
        diffs : ndarray
            Differences between measurements from devices A and B.

        Returns
        -------
        bias : float
            Mean of the differences.
        lower_lim : float
            Bias minus 1.96 standard deviations.
        upper_lim : float
            Bias plus 1.96 standard deviations.

        Examples
        --------
        >>> diffs = np.array([-67, 0, 0])
        >>> bias, lower_lim, upper_lim = BlandAltman.limits_of_agreement(diffs)

        >>> round(bias)
        -22.0

        >>> round(lower_lim)
        -84.0

        >>> round(upper_lim)
        40.0

        """
        bias = diffs.mean()
        standard_dev = diffs.std()

        lower_lim, upper_lim = gen.limits(bias, 1.96 * standard_dev)

        return bias, lower_lim, upper_lim


def mad_outliers(x, c):
    """
    Remove outliers from an array of data using the
    median absolute deviation (MAD).

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


def relative_difference(x, y, absolute=False):
    """
    Relative difference between two observations A and B.

    Calculated as (A - B) / mean(A, B).

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

    relative_difference = difference / average

    if absolute:
        relative_difference = abs(relative_difference)

    return relative_difference


if __name__ == "__main__":

    import doctest
    doctest.testmod()
