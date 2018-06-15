import math

import numpy as np


def gaussian(x, sigma=1, mu=0):
    """
    Function that defines the Gaussian (normal) distribution.

    Parameters
    ----------
    x : {int, float}
        Main input value.
    mu : {int, float}
        Mean of distribution.
    sigma : {int, float}
        Standard deviation.

    Returns
    -------
    float
        Output of Gaussian function.

    Examples
    --------
    >>> round(gaussian(0), 4)
    0.3989

    >>> round(gaussian(0, sigma=0.5), 4)
    0.7979

    >>> round(gaussian(-0.5, mu=1, sigma=0.5), 4)
    0.0089
    
    """
    np.seterr(under='ignore')

    coeff = 1.0 / np.sqrt(2 * np.pi * sigma**2)
    exponent = np.exp(- (x - mu)**2 / (2 * sigma**2))

    return coeff * exponent


def sigmoid(x, a=1):
    """
    Sigmoid function (produces the sigmoid curve).

    Parameters
    ----------
    x : {int, float}
        Main argument to the function.

    a : {int, float}, optional
        Coefficient for x (the default is 1)
        This changes the shape of the sigmoid curve.

    Returns
    -------
    float
        Output of sigmoid function.

    Examples
    --------
    >>> sigmoid(0)
    0.5

    >>> round(sigmoid(1, a=2), 4)
    0.8808

    """
    return 1 / (1 + math.exp(-a * x))


def root_mean_square(x):
    """
    Return the root mean square of an array.

    Parameters
    ----------
    x : ndarray
        Input numpy array.

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


def norm_ratio(a, b):
    """
    Normalized ratio between two positive inputs.
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


def normalize_array(x):
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
    >>> np.array_equal(normalize_array(x), [0, 0.25, 0.5, 0.75, 1])
    True

    """
    max_value = np.nanmax(x)
    min_value = np.nanmin(x)

    return (x - min_value) / (max_value - min_value)


def centre_of_mass(points, masses):
    """
    Compute the centre of mass of a set of points.

    Parameters
    ----------
    points : ndarray
        List of points in space
    masses : array_like
        Mass of each point

    Returns
    -------
    ndarray
        Centre of mass as a 1-D array

    Examples
    --------
    >>> points = np.array([[1, 2, 3], [-1, 2, 5], [3, 5, 7]])
    >>> masses = [1, 1, 1]
    >>> centre_of_mass(points, masses)
    array([1., 3., 5.])

    >>> points = np.array([[-1, 10], [5, 2]])
    >>> masses = [0, 1]
    >>> centre_of_mass(points, masses)
    array([5., 2.])

    """
    total = 0

    for point, mass in zip(points, masses):
        total += mass * point

    return total / sum(masses)


if __name__ == "__main__":

    import doctest
    doctest.testmod()
