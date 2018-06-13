import numpy as np
from numpy.linalg import norm

import modules.general as gen


def unit(v):
    """
    Return the unit vector of v.

    Parameters
    ----------
    v : array_like
        Input vector.

    Returns
    -------
    ndarray
        Unit vector.

    Examples
    --------
    >>> unit([5, 0, 0])
    array([1., 0., 0.])

    >>> unit([0, -2])
    array([ 0., -1.])

    """
    return v / norm(v)


def consecutive_dist(points):
    """
    Calculate the distance between each consecutive pair of points.

    Parameters
    ----------
    points : array_like
        List of points.

    Yields
    ------
    float
        Distance between two consecutive points.

    Examples
    --------
    >>> points = [[1, 1], [2, 1], [0, 1]]
    >>> list(consecutive_dist(points))
    [1.0, 2.0]

    """
    for point_1, point_2 in gen.pairwise(points):

        vector = np.subtract(point_1, point_2)
        yield norm(vector)


def closest_point(candidate_points, target_point):
    """
    Return the closest point to a target from a set of candidates.

    Parameters
    ----------
    candidate_points : ndarray
        (n, dim) array of n points.
    target_point : array_like
        Target position

    Returns
    -------
    close_point : ndarray
        Closest point from the set of candidates.
    close_index : int
        Row index of the closest point in the candidates array.

    Examples
    --------
    >>> candidates = np.array([[3, 4, 5], [2, 1, 5]])
    >>> target = [2, 1, 4]

    >>> close_point, close_index = closest_point(candidates, target)

    >>> close_point
    array([2, 1, 5])

    >>> close_index
    1

    """
    vectors_to_target = candidate_points - target_point
    distances_to_target = norm(vectors_to_target, axis=1)

    close_index = np.argmin(distances_to_target)
    close_point = candidate_points[close_index, :]

    return close_point, close_index


def dist_point_line(P, A, B):
    """
    Distance from a point to a line.

    Parameters
    ----------
    P : ndarray
        Point in space.
    A : ndarray
        Point A on line.
    B : ndarray
        Point B on line.

    Returns
    -------
    float
        Distance from point to plane.

    Examples
    --------
    >>> A, B = np.array([0, 0]), np.array([1, 0])

    >>> dist_point_line(np.array([0, 5]), A, B)
    5.0

    >>> dist_point_line(np.array([10, 0]), A, B)
    0.0

    """
    num = norm(np.cross(P - A, P - B))
    denom = norm(A - B)

    return num / denom


def dist_point_plane(P, P_plane, normal):
    """
    Distance from a point to a plane.

    Parameters
    ----------
    P : ndarray
        Point in space.
    normal : ndarray
        Normal of plane.
    plane_pt : ndarray
        Point on plane.

    Returns
    -------
    float
        Distance from point to plane.

    Examples
    --------
    >>> P_plane, normal = np.array([0, 0, 0]), np.array([0, 0, 1])

    >>> dist_point_plane(np.array([10, 2, 5]), P_plane, normal)
    5.0

    """
    n_hat = unit(normal)

    return abs(np.dot(n_hat, P - P_plane))


def proj_point_line(P, A, B):
    """
    Project a point onto a line.

    Parameters
    ----------
    P : ndarray
        Point in space.
    A : ndarray
        Point A on line.
    B : ndarray
        Point B on line.

    Returns
    -------
    ndarray
        Projection of point P onto the line.

    Examples
    --------
    >>> A, B = np.array([0, 0]), np.array([1, 0])

    >>> proj_point_line(np.array([0, 5]), A, B)
    array([0., 0.])

    """

    AP = P - A  # Vector from A to point
    AB = B - A  # Vector from A to B

    # Project point onto line
    return A + np.dot(AP, AB) / norm(AB)**2 * AB


def proj_point_plane(P, P_plane, normal):
    """
    Project a point onto a plane.

    Parameters
    ----------
    P : ndarray
        Point in space..
    P_plane : ndarray
        Point on plane..
    normal : ndarray
        Normal vector of plane..

    Returns
    -------
    ndarray
        Projection of point P onto the plane..

    Examples
    --------
    >>> P_plane, normal = np.array([0, 0, 0]), np.array([0, 0, 1])

    >>> proj_point_plane(np.array([10, 2, 5]), P_plane, normal)
    array([10.,  2.,  0.])

    """
    unit_normal = unit(normal)

    return P - np.dot(P - P_plane, unit_normal) * unit_normal


def best_fit_line(points):
    """
    Find the line of best fit for a set of multi-dimensional points.
    Uses singular value decomposition.

    The direction of the line depends on the order of the points.

    Parameters
    ----------
    points : ndarray
         (n, d) array of n points with dimension d.

    Returns
    -------
    centroid : ndarray
        Centroid of all the points. Line of best fit passes through centroid.
    direction : ndarray
        Unit direction vector for line of best fit.
        Right singular vector which corresponds to the largest
        singular value of A.

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
    # Ensure that points have no nan values
    points = points[~np.isnan(points).any(axis=1)]

    centroid = np.mean(points, axis=0)
    A = points - centroid

    _, _, vh = np.linalg.svd(A)

    direction = vh[0, :]

    return centroid, direction


def angle_direction(target_direction, forward, up):
    """
    Find the direction (right or left) of a target,
    given an orientation specifying the forward and up directions.

    All input arrays have shape (3,).

    Parameters
    ----------
    target : array_like
        Vector in direction of a target point.
    forward : array_like
        Vector for forward direction.
     up : array_like
        Vector for up direction.

    Returns
    -------
    int
        Value is 1 if target is to the left,
        -1 if to the right, 0 if straight ahead.

    Examples
    --------
    >>> fwd, up = [0, 1, 0], [0, 0, 1]
    >>> angle_direction([1, 1, 0], fwd, up)
    -1

    >>> angle_direction(np.array([-1, 1, 0]), fwd, up)
    1

    >>> angle_direction(np.array([0.0, -1.0, 0.0]), fwd, up)
    0

    """
    perpendicular = np.cross(forward, target_direction)

    signed = np.sign(np.dot(perpendicular, up))

    return int(signed)


def angle_between(x, y, degrees=False):
    """
    Compute the angle between vectors x and y.

    Parameters
    ----------
    x, y : array_like
        Input vectors

    degrees : bool, optional
        Set to true for angle in degrees rather than radians.

    Returns
    -------

    Examples
    --------
    >>> angle_between([1, 0], [1, 0])
    0.0

    >>> x, y = [1, 0], [1, 1]
    >>> round(angle_between(x, y, degrees=True))
    45.0

    >>> x, y = [1, 0], [-2, 0]
    >>> round(angle_between(x, y, degrees=True))
    180.0

    """
    dot_product = np.dot(x, y)

    cos_theta = dot_product / (norm(x) * norm(y))

    theta = np.arccos(cos_theta)

    if degrees:
        theta = np.rad2deg(theta)

    return theta


# Raise an exception for any floating-point errors
np.seterr(all='raise')

if __name__ == "__main__":

    import doctest
    doctest.testmod()
