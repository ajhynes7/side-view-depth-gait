"""Functions related to spatial points."""

import numpy as np
from numpy.linalg import norm


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
