import numpy as np
from statsmodels import robust


def mad_outliers(x, c):
    """
    Removes outliers from an array of data using the
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
    Return the relative error between a measured and actual value.

    Parameters
    ----------
    measured : {int, float}
        Measured value.

    actual : {int, float}
        Actual value.

    absolute : bool, optional
        If True, the absolute relative error is returned
        (the default is False).

    Returns
    -------
    error
        Relative error.

    Examples
    --------
    >>> relative_error(2, 5)
    -0.6

    >>> relative_error(3, 5, absolute=True)
    0.4

    """
    error = (measured - actual) / actual

    if absolute:
        error = abs(error)

    return error


if __name__ == "__main__":

    import doctest
    doctest.testmod()
