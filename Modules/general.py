import numpy as np
import math


def sigmoid(x, a=1):

    return 1 / (1 + math.exp(-a * x))


def relative_error(measured, actual, absolute=False):

    error = (measured - actual) / actual

    if absolute:
        error = abs(error)

    return error


def normalize_array(x):

    max_value = np.nanmax(x)
    min_value = np.nanmin(x)

    return (x - min_value) / (max_value - min_value)


def score_func(measured, actual):

    absolute_error = relative_error(measured, actual, absolute=True)
    normalized_error = sigmoid(absolute_error)

    return math.log(-normalized_error + 1) + 1


def ratio_func(a, b):
    """
    Ratio between two positive inputs.
    If ratio a / b is less than one, the reciprocal is returned instead.

    Parameters
    ----------
    a, b : float
        Positive inputs

    Returns
    -------
    float
        Ratio between a and b
    """
    if a == 0 or b == 0:
        return np.nan

    ratio = np.divide(a, b)

    if ratio < 1:
        ratio = np.reciprocal(ratio)

    return ratio


def matrix_from_labels(expected_values, labels):

    n_rows = len(labels)

    mat = np.full((n_rows, n_rows), np.nan)

    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):

            if label_j in expected_values[label_i]:
                mat[i, j] = expected_values[label_i][label_j]

    return mat


def inside_spheres(dist_matrix, point_nums, r):
    """
    Given n points, m of these points are centres of spheres.
    Calculates which of the n points are contained inside these m spheres.

    Parameters
    ----------
    dist_matrix : ndarray
        | (n, n) distance matrix
        | Element (i, j) is distance from point i to point j

    point_nums : array_like
        | (m, ) List of points that are the sphere centres
        | Numbers between 1 and n

    r : float
        Radius of spheres

    Returns
    -------
    in_spheres : array_like
        (n,) array of bools
        Element i is true if point i is in the set of spheres
    """
    n_points = len(dist_matrix)

    in_spheres = np.full(n_points, False)

    for i in point_nums:

        distances = dist_matrix[i, :]

        in_current_sphere = distances <= r
        in_spheres = in_spheres | in_current_sphere

    return in_spheres
