"""Functions related to spatial points."""

from itertools import accumulate

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist


def assign_to_closest(points, targets):
    """
    Assign each point to the closest target.

    Parameters
    ----------
    points : ndarray
        (n, d) array of n points with dimension d.
    targets : ndarray
        (n_targets, d) array of target points with dimension d.

    Returns
    -------
    ndarray
        (n,) array of labels.
        Label i corresponds to the target closest to point i.

    Examples
    --------
    >>> points = np.array([[0, 0], [1, 1], [50, 0], [2, 2], [100, 100]])
    >>> targets = np.array([[5, 5], [40, 0], [105, 100]])

    >>> assign_to_closest(points, targets)
    array([0, 0, 1, 0, 2])

    """
    dist_matrix = cdist(points, targets)

    return np.argmin(dist_matrix, axis=1)


def consecutive_dist(points):
    """
    Calculate the distance between each consecutive pair of points.

    Parameters
    ----------
    points : array_like
        List of points.

    Returns
    ------
    ndarray
        Distance between consecutive points.

    Examples
    --------
    >>> points = [[1, 1], [2, 1], [0, 1]]
    >>> consecutive_dist(points)
    array([1., 2.])

    """
    differences = np.diff(points, axis=0)

    return norm(differences, axis=1)


def correspond_points(points_prev, points_curr):
    """
    Order points so they correspond to points from the previous frame.

    Only two points are allowed on each frame.

    The correspondence minimizes the total distance travelled by the points
    from the previous frame to the current.

    Parameters
    ----------
    points_prev, points_curr : ndarray
        Previous and current points.
        (2, d) array of 2 points with dimension d.

    Returns
    -------
    points_ordered : ndarray
        Current points in the order that corresponds to previous points.

    Raises
    ------
    ValueError
        When either input does not have two rows.

    Examples
    --------
    >>> points_prev = np.array([[0, 0], [10, 11]])
    >>> points_curr = np.array([[10, 10], [2, 3]])

    >>> correspond_points(points_prev, points_curr)
    array([[ 2,  3],
           [10, 10]])

    >>> points_prev = np.array([[-1, 1, 3], [4, 5, 2]])
    >>> points_curr = np.array([[-2, 1, 4], [5, 5, 1]])

    >>> correspond_points(points_prev, points_curr)
    array([[-2,  1,  4],
           [ 5,  5,  1]])

    """
    inputs = [points_prev, points_curr]
    if not all(len(points) == 2 for points in inputs):
        raise ValueError("Inputs do not have two rows of points.")

    dist_matrix = cdist(points_prev, points_curr)

    sum_diagonal = dist_matrix.trace()
    sum_reverse = np.fliplr(dist_matrix).trace()

    points_ordered = points_curr

    if sum_reverse < sum_diagonal:

        points_ordered = np.flip(points_curr, axis=0)

    return points_ordered


def track_two_objects(points_1, points_2):
    """
    Assign points in time to two distinct objects.

    Minimizes the distance travelled across consecutive frames.

    Parameters
    ----------
    points_1, points_2 : array_like
        (n, d) array of n points with dimension d.
        The points represent the position of an object at consective frames.

    Returns
    -------
    points_assigned_1, points_assigned_2 : ndarray
        (n, d) array of points.
        The points now correspond to objects 1 and 2.

    Examples
    --------
    >>> points_1 = [[-1, 1], [0, 1], [12, 11], [3, 5], [5, 6]]
    >>> points_2 = [[11, 10], [12, 12], [1, 3], [14, 13], [15, 15]]

    >>> points_new_1, points_new_2 = track_two_objects(points_1, points_2)

    >>> points_new_1
    array([[-1,  1],
           [ 0,  1],
           [ 1,  3],
           [ 3,  5],
           [ 5,  6]])

    >>> points_new_2
    array([[11, 10],
           [12, 12],
           [12, 11],
           [14, 13],
           [15, 15]])

    """
    # Results of corresponding consecutive pairs of points
    point_series = accumulate(zip(points_1, points_2), correspond_points)

    # Unpack the points and convert to ndarray
    point_list_1, point_list_2 = zip(*point_series)
    points_assigned_1 = np.stack(point_list_1)
    points_assigned_2 = np.stack(point_list_2)

    return points_assigned_1, points_assigned_2


def closest_point(points, target):
    """
    Select the closest point to a target from a set of points.

    Parameters
    ----------
    points : ndarray
        Points in space. Each row is a position vector.
    target : ndarray
        Target position.

    Returns
    -------
    point_closest : ndarray
        The closest point to the target.
    index_closest : int
        Index of the closest point.

    Examples
    --------
    >>> points = np.array([[1, 2], [2, 3], [10, 11]])
    >>> target = np.array([10, 10])
    >>> point_closest, index = closest_point(points, target)

    >>> point_closest
    array([10, 11])

    >>> index
    2

    """
    distances = norm(points - target, axis=1)

    index_closest = np.argmin(distances)
    point_closest = points[index_closest]

    return point_closest, index_closest


def position_accuracy(points, targets, max_dist=10):

    distances = norm(points - targets, axis=1)

    return np.mean(distances <= max_dist)
