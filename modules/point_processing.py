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
    -------
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


def apply_to_real_points(func, points):
    """
    Apply a function only to points that contain no nans.

    The output has the same length as the number of input points.

    """
    is_nan_point = np.isnan(points).any(axis=1)
    points_real = points[~is_nan_point]

    output_real = func(points_real)

    output = np.full(points.shape[0], np.nan)
    output[~is_nan_point] = output_real

    return output


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


def closest_proposals(proposals, targets):

    closest = np.zeros(targets.shape)

    for i, target in enumerate(targets):

        # Proposals for current target
        target_proposals = proposals[i]

        close_point, _ = closest_point(target_proposals, target)
        closest[i, :] = close_point

    return closest


def assign_pair(point_pair, target_pair):

    dist_matrix = cdist(point_pair, target_pair)

    sum_diagonal = dist_matrix.trace()
    sum_reverse = np.fliplr(dist_matrix).trace()

    assigned_pair = point_pair

    if sum_reverse < sum_diagonal:
        assigned_pair = np.flip(point_pair, axis=0)

    return assigned_pair


def match_pairs(points_1, points_2, targets_1, targets_2):

    points_shape = points_1.shape
    assigned_1 = np.zeros(points_shape)
    assigned_2 = np.zeros(points_shape)

    for i in range(points_shape[0]):

        point_pair = np.vstack((points_1[i], points_2[i]))
        target_pair = np.vstack((targets_1[i], targets_2[i]))

        assigned_pair = assign_pair(point_pair, target_pair)

        assigned_1[i, :] = assigned_pair[0]
        assigned_2[i, :] = assigned_pair[1]

    return assigned_1, assigned_2


def position_accuracy(points, targets, max_dist=10):
    """
    Calculate ratio of points within a distance from corresponding targets.

    Parameters
    ----------
    points : ndarray
        (n, d) array of n positions of dimension d.
    targets : ndarray
        (n, d) array of n target positions of dimension d.
    max_dist : {int, float}
        Maximum distance that a point can be from its target to be counted.

    Returns
    -------
    float
        Ratio of points within the max distance from their targets.

    Examples
    --------
    >>> points = np.array([[1, 2], [2, 3], [10, 11], [15, -2]])
    >>> targets = np.array([[1, 3], [10, 3], [12, 13], [14, -3]])

    >>> position_accuracy(points, targets, max_dist=0)
    0.0

    >>> position_accuracy(points, targets, max_dist=5)
    0.75

    >>> position_accuracy(points, targets, max_dist=10)
    1.0

    """
    distances = norm(points - targets, axis=1)

    return np.mean(distances <= max_dist)


def double_position_accuracy(points_1, points_2, targets_1, targets_2, max_dist=10):

    within_dist_1 = norm(points_1 - targets_1, axis=1) <= max_dist
    within_dist_2 = norm(points_2 - targets_2, axis=1) <= max_dist

    within_both = np.column_stack((within_dist_1, within_dist_2)).all(axis=1)

    return within_both.mean()
