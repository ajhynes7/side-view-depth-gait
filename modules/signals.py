"""Module for signal processing."""

import numpy as np


def root_mean_square(x):
    """
    Return the root mean square of an array.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    float
        Root mean square.

    Examples
    --------
    >>> x = np.array([0, 1])
    >>> root_mean_square(x) == np.sqrt(2) / 2
    True

    """
    return np.sqrt(sum(x**2) / x.size)


def normalize(x):
    """
    Map all values in an array to the range [0, 1].

    Parameters
    ----------
    x : array_like
        Input array.
        Max and min values should be different to avoid division by zero.

    Returns
    -------
    ndarray
        Normalized array.

    Examples
    --------
    >>> x = [i for i in range(5)]
    >>> np.array_equal(normalize(x), [0, 0.25, 0.5, 0.75, 1])
    True

    """
    max_value = np.nanmax(x)
    min_value = np.nanmin(x)

    return (x - min_value) / (max_value - min_value)
