"""Operations dealing with linear algebra, such as projection and distances."""

import numpy as np


def best_fit_line(points):
    """
    Return the line of best fit for a set of points.

    The direction of the line depends on the order of the points.

    Parameters
    ----------
    points : ndarray
         (n, d) array of n points with dimension d.

    Returns
    -------
    centroid : ndarray
        Centroid of points. Line of best fit passes through centroid.
    direction : ndarray
        Unit direction vector for line of best fit.
        Right singular vector which corresponds to the largest
        singular value of A.

    Raises
    ------
    ValueError
        When fewer than two points are input (line would be underdefined).

    Examples
    --------
    >>> points = np.array([[1, 0], [2, 0], [3, 0]])
    >>> centroid, direction = best_fit_line(points)

    >>> centroid
    array([2., 0.])

    >>> direction
    array([1., 0.])

    >>> _, direction = best_fit_line(np.flip(points, axis=0))
    >>> direction.astype(int)
    array([-1,  0])

    """
    n_points, _ = points.shape
    if n_points < 2:
        raise ValueError('At least two points required.')

    # Ensure that points have no nan values
    points = points[~np.isnan(points).any(axis=1)]

    centroid = np.mean(points, axis=0)

    _, _, vh = np.linalg.svd(points - centroid)

    direction = vh[0, :]

    return centroid, direction


def line_coordinate_system(line_point, direction, points):
    """
    Represent points in a one-dimensional coordinate system defined by a line.

    The input line point acts as the origin of the coordinate system.

    The line is analagous to an x-axis. The output coordinates represent the
    x-values of points on this line.

    Parameters
    ----------
    line_point : ndarray
        Point on line.
    direction : ndarray
        Direction vector of line.
    points : ndarray
        (n, d) array of n points with dimension d.

    Returns
    -------
    coordinates : ndarray
        One-dimensional coordinates.

    Examples
    --------
    >>> line_point = np.array([0, 0])
    >>> direction = np.array([1, 0])

    >>> points = np.array([[10, 2], [3, 4], [-5, 5]])

    >>> line_coordinate_system(line_point, direction, points)
    array([10,  3, -5])

    """
    vectors = points - line_point

    coordinates = np.apply_along_axis(np.dot, 1, vectors, direction)

    return coordinates


def side_value_2d(point_a, point_b, direction):
    """
    Return value for side of point A relative to point B given a direction.

    A positive value indicates that A is to the right of B.

    Parameters
    ----------
    point_a, point_b : array_like
        2D points A and B
    direction : array_like
        2D direction vector.

    Returns
    -------
    float
        Value indicating the side of A relative to B.

    Examples
    --------
    >>> side_value_2d([10, 2], [-5, 1], [0, 1])
    15.0

    >>> side_value_2d([10, 2], [-5, 1], [0, -1])
    -15.0

    >>> side_value_2d([10, 2], [8, -2], [1, 0])
    -4.0

    >>> side_value_2d([10, 2], [8, -2], [-1, 0])
    4.0

    """
    vector_to_a = np.subtract(point_a, point_b)

    return np.float(np.cross(vector_to_a, direction))
