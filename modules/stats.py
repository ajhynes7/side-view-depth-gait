import numpy as np

from statsmodels import robust


def mad_outliers(x, c):
    """
    Remove outliers from an array of data using the
    Median Absolute Deviation (MAD).

    Values beyond the Median ± c(MAD) are set to NaN.

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
    >>> import numpy.testing as npt
    >>> x_filtered = mad_outliers([2.0, 3.0, 100.0, 3.0], 2.5)
    >>> npt.assert_array_equal(x_filtered, [2, 3, np.nan, 3])

    >>> x_filtered = mad_outliers([5, 6, 4, 20], 3)
    >>> npt.assert_array_equal(x_filtered, [5, 6, 4, np.nan])

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


def percent_difference(x, y, absolute=False):
    """
    [description]

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
        Percent difference.

    Examples
    --------
    >>> x = np.array([3, 3, 4])
    >>> y = np.array([1, 2, 4])

    >>> percent_difference(x, y)
    array([1. , 0.4, 0. ])

    >>> percent_difference(2, 3)
    -0.4

    """
    difference = x - y

    average = np.mean([x, y], axis=0)

    percent_difference = difference / average

    if absolute:
        percent_difference = abs(percent_difference)

    return percent_difference


if __name__ == "__main__":

    import doctest
    doctest.testmod()
