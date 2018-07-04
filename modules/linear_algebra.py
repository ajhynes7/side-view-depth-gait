"""Operations dealing with linear algebra, such as projection and distances."""

import numpy as np
from numpy.linalg import norm

import general as gen


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
    return gen.divide_no_error(v, norm(v))


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


def dist_point_line(point, line_point_1, line_point_2):
    """
    Distance from a point to a line.

    Parameters
    ----------
    point : ndarray
        Point in space.
    line_point_1 : ndarray
        Point A on line.
    line_point_2 : ndarray
        Point B on line.

    Returns
    -------
    float
        Distance from point to plane.

    Examples
    --------
    >>> line_point_1, line_point_2 = np.array([0, 0]), np.array([1, 0])

    >>> dist_point_line(np.array([0, 5]), line_point_1, line_point_2)
    5.0

    >>> dist_point_line(np.array([10, 0]), line_point_1, line_point_2)
    0.0

    """
    num = norm(np.cross(point - line_point_1, point - line_point_2))
    denom = norm(line_point_1 - line_point_2)

    return gen.divide_no_error(num, denom)


def dist_point_plane(point, plane_point, normal):
    """
    Distance from a point to a plane.

    Parameters
    ----------
    point : ndarray
        Point in space.
    plane_point : ndarray
        Point on plane.
    normal : ndarray
        Normal of plane.

    Returns
    -------
    float
        Distance from point to plane.

    Examples
    --------
    >>> plane_point, normal = np.array([0, 0, 0]), np.array([0, 0, 1])

    >>> dist_point_plane(np.array([10, 2, 5]), plane_point, normal)
    5.0

    """
    n_hat = unit(normal)

    return abs(np.dot(n_hat, point - plane_point))


def project_vector(u, v):
    """
    Project vector u onto vector v.

    Parameters
    ----------
    u, v : array_like
        Input vectors.

    Returns
    -------
    ndarray
        Projection of vector x onto vector y.

    Examples
    --------
    >>> u, v = [10, 5], [0, 1]

    >>> project_vector(u, v)
    array([0., 5.])

    >>> project_vector(u, [0, 8])
    array([0., 5.])

    """
    unit_v = unit(v)

    return np.dot(u, unit_v) * unit_v


def project_point_line(point, line_point_1, line_point_2):
    """
    Project a point onto a line.

    Parameters
    ----------
    point : ndarray
        Point in space.
    line_point_1 : ndarray
        Point A on line.
    line_point_2 : ndarray
        Point B on line.

    Returns
    -------
    ndarray
        Projection of point P onto the line.

    Examples
    --------
    >>> line_point_1, line_point_2 = np.array([0, 0]), np.array([1, 0])

    >>> project_point_line(np.array([0, 5]), line_point_1, line_point_2)
    array([0., 0.])

    """
    vec_1 = point - line_point_1  # Vector from A to point
    vec_1_2 = line_point_2 - line_point_1  # Vector from A to B

    # Project point onto line
    return line_point_1 + gen.divide_no_error(np.dot(vec_1, vec_1_2),
                                              norm(vec_1_2)**2) * vec_1_2


def project_point_plane(point, plane_point, normal):
    """
    Project a point onto a plane.

    Parameters
    ----------
    point : ndarray
        Point in space.
    plane_point : ndarray
        Point on plane.
    normal : ndarray
        Normal vector of plane.

    Returns
    -------
    ndarray
        Projection of point P onto the plane..

    Examples
    --------
    >>> plane_point, normal = np.array([0, 0, 0]), np.array([0, 0, 1])

    >>> project_point_plane(np.array([10, 2, 5]), plane_point, normal)
    array([10.,  2.,  0.])

    """
    unit_normal = unit(normal)

    return point - np.dot(point - plane_point, unit_normal) * unit_normal


def best_fit_line(points):
    """
    Find the line of best fit for a set of multi-dimensional points.

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

    _, _, vh = np.linalg.svd(points - centroid)

    direction = vh[0, :]

    return centroid, direction


def angle_direction(target_direction, forward, up):
    """
    Find the direction (right or left) of a target.

    An orientation is defined by specifying the forward and up directions.

    Parameters
    ----------
    target_direction : array_like
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
    theta : float
        Angle between vectors.

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

    cos_theta = gen.divide_no_error(dot_product, (norm(x) * norm(y)))

    theta = np.arccos(cos_theta)

    if degrees:
        theta = np.rad2deg(theta)

    return theta


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
    array([10.,  3., -5.])

    """
    line_point_2 = line_point + direction

    projected_points = np.apply_along_axis(project_point_line, 1, points,
                                           line_point, line_point_2)

    vectors = projected_points - line_point

    coordinates = np.apply_along_axis(np.dot, 1, vectors, direction)

    return coordinates


if __name__ == "__main__":

    import doctest
    doctest.testmod()
