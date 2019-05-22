"""Functions related to spatial points."""

from copy import copy

import numpy as np
from dpcontracts import require
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


@require("There must be two points in each input.", lambda args: all(len(points) == 2 for points in args))
def check_correspondence(points_prev, points_curr):
    """
    Check if current points correspond to points from the previous frame.

    Only two points are allowed on each frame.

    The correspondence minimizes the total distance travelled by the points
    from the previous frame to the current.

    Parameters
    ----------
    points_prev, points_curr : array_like
        Previous and current points.
        (2, d) array of 2 points with dimension d.

    Returns
    -------
    bool
        True if the points are corresponded; false otherwise.

    Examples
    --------
    >>> points_prev = [[0, 0], [7, 8]]
    >>> points_curr = [[7, 9], [1, 1]]
    >>> check_correspondence(points_prev, points_curr)
    False

    >>> points_prev = [[0, 0, 0], [5, 5, 5]]
    >>> points_curr = [[1, 1, 1], [6, 6, 6]]
    >>> check_correspondence(points_prev, points_curr)
    True

    >>> points_prev = [[-1, 1, 3], [4, 5, 2]]
    >>> points_curr = [[-2, 1, 4], [5, 5, 1]]
    >>> check_correspondence(points_prev, points_curr)
    True

    """
    dist_matrix = cdist(points_prev, points_curr)

    sum_diagonal = dist_matrix.trace()
    sum_reverse = np.fliplr(dist_matrix).trace()

    return sum_diagonal < sum_reverse


def track_two_objects(points_a, points_b):
    """
    Assign points in time to two distinct objects.

    Minimizes the distance travelled across consecutive frames.

    Parameters
    ----------
    points_a, points_b : array_like
        (n, d) array of n points with dimension d.
        The points represent the position of an object at consecutive frames.

    Returns
    -------
    array_correspondence : ndarray
        (n,) array.
        Element i is 1 if the points at frame i are correctly assigned; 0 otherwise.

    Examples
    --------
    >>> points_a = [[0, 0], [11, 11], [2, 2], [3, 3], [14, 14]]
    >>> points_b = [[10, 10], [1, 1], [12, 12], [13, 13], [4, 4]]

    >>> track_two_objects(points_a, points_b)
    array([ True, False,  True,  True, False])

    """
    n_points = len(points_a)
    array_correspondence = np.full(n_points, True)

    point_prev_a = points_a[0]
    point_prev_b = points_b[0]

    for i in range(1, n_points):

        point_curr_a = points_a[i]
        point_curr_b = points_b[i]

        points_prev = [point_prev_a, point_prev_b]
        points_curr = [point_curr_a, point_curr_b]

        correspondence = check_correspondence(points_prev, points_curr)

        point_prev_a = point_curr_a
        point_prev_b = point_curr_b

        if not correspondence:
            # The current points do not correspond to the previous.
            # Swap the previous points.
            point_prev_a, point_prev_b = point_prev_b, point_prev_a

        array_correspondence[i] = correspondence

    return array_correspondence


def correspond_points(points_a, points_b, array_correspondence):
    """
    Return points correctly assigned to groups A and B.

    The points correspond with the points from the previous frame.

    Parameters
    ----------
    points_a, points_b : array_like
        (n, d) array of n points with dimension d.
        The points represent the position of an object at consecutive frames.
    array_correspondence : ndarray
        (n) array of booleans.
        Element i is 1 if the points at frame i are correctly assigned; false otherwise.

    Returns
    -------
    points_a_true, points_b_true : array_like
        (n, d) array of n points with dimension d.
        Points correctly assigned to groups A and B.

    Examples
    --------
    >>> points_a = [[0, 0], [11, 11], [2, 2], [3, 3], [14, 14]]
    >>> points_b = [[10, 10], [1, 1], [12, 12], [13, 13], [4, 4]]

    >>> array_correspondence = track_two_objects(points_a, points_b)
    >>> points_a, points_b = correspond_points(points_a, points_b, array_correspondence)

    >>> points_a
    [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]

    >>> points_b
    [[10, 10], [11, 11], [12, 12], [13, 13], [14, 14]]

    """
    points_a_true = copy(points_a)
    points_b_true = copy(points_b)

    for i, correspondence in enumerate(array_correspondence):

        if not correspondence:
            points_a_true[i] = points_b[i]
            points_b_true[i] = points_a[i]

    return points_a_true, points_b_true


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
