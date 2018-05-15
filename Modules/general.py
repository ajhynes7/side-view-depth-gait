import numpy as np


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
        return np.inf

    ratio = np.divide(a, b)

    if ratio < 1:
        return np.reciprocal(ratio)


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
