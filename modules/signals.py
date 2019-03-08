"""Module for signal processing."""

import numpy as np


def root_mean_square(array):
    """
    Return the root mean square of an array.

    Parameters
    ----------
    array : ndarray
        Input array.

    Returns
    -------
    float
        Root mean square.

    Examples
    --------
    >>> array = np.array([0, 1])
    >>> root_mean_square(array) == np.sqrt(2) / 2
    True

    """
    return np.sqrt(sum(array ** 2) / array.size)


def nan_normalize(array):
    """
    Map all values in an array to the range [0, 1].

    The array can contain nan values.

    Parameters
    ----------
    array : array_like
        Input array.
        Max and min values should be different to avoid division by zero.

    Returns
    -------
    ndarray
        Normalized array.

    Raises
    ------
    ValueError
        When minimum and maximum values are the same.

    Examples
    --------
    >>> array = [i for i in range(5)]
    >>> np.array_equal(nan_normalize(array), [0, 0.25, 0.5, 0.75, 1])
    True

    >>> nan_normalize([1, 2, np.nan, 3])
    array([0. , 0.5, nan, 1. ])

    """
    max_value = np.nanmax(array)
    min_value = np.nanmin(array)

    array_range = max_value - min_value

    if array_range == 0:
        raise ValueError("Division by zero.")

    return (array - min_value) / array_range
