"""Functions related to spatial points."""

from typing import Tuple

import numpy as np
from dpcontracts import require
from numpy import ndarray
from numpy.linalg import norm
from scipy.spatial.distance import cdist

from modules.typing import array_like


def consecutive_dist(points: array_like) -> ndarray:
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


def closest_point(points: array_like, target: array_like) -> Tuple[ndarray, int]:
    """
    Select the closest point to a target from a set of points.

    Parameters
    ----------
    points : (N, D) array_like
        Points in space. Each row is a position vector.
    target : (D,) array_like
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
    distances = norm(np.subtract(points, target), axis=1)

    # Wrap in int() to satisfy mypy.
    index_closest = int(np.argmin(distances))

    point_closest = points[index_closest]

    return point_closest, index_closest


@require(
    "The args must have the same length.", lambda args: len(set(map(len, args))) == 1
)
def closest_proposals(proposals: array_like, targets: array_like) -> ndarray:
    """
    Return closest proposal to each target.

    Parameters
    ----------
    proposals : (N,) Sequence
        Each element is a list of position proposals.
    targets : (N, D) ndarray
        Each row is a target position.

    Returns
    -------
    ndarray
        Each row i is the closest proposal to the target i.

    Examples
    --------
    >>> proposals = [
    ...     [[0, 1], [1, 1], [1, 5]],
    ...     [[4, 5], [2, 3], [2, 2]],
    ...     [[1, 9], [4, 2], [1, 5]],
    ... ]

    >>> targets = [[0, 0], [1, 1], [2, 2]]

    >>> closest_proposals(proposals, targets)
    array([[0, 1],
           [2, 2],
           [4, 2]])

    """
    return np.array(
        [closest_point(points, target)[0] for points, target in zip(proposals, targets)]
    )


@require(
    "The args must have the same length.", lambda args: len(set(map(len, args))) == 1
)
def assign_pair(pair_points: array_like, pair_targets: array_like) -> ndarray:
    """
    Assign a pair of points to a pair of targets by minimizing point-target distance.

    Parameters
    ----------
    pair_points : (2,) array_like
        Pair of points.
    pair_targets : (2,) array_like
        Pair of target points.

    Returns
    -------
    pair_assigned : ndarray
        The pair of points assigned to the pair of targets.
        The order of the points now matches the targets.

    Examples
    --------
    >>> pair_points = ([0, 0], [1, 1])
    >>> pair_targets = ([1, 2], [0, 1])

    >>> assign_pair(pair_points, pair_targets)
    array([[1, 1],
           [0, 0]])

    """
    dist_matrix = cdist(pair_points, pair_targets)

    sum_diagonal = dist_matrix.trace()
    sum_reverse = np.fliplr(dist_matrix).trace()

    pair_assigned = np.array(pair_points)

    if sum_reverse < sum_diagonal:
        pair_assigned = np.flip(pair_points, axis=0)

    return pair_assigned


@require(
    "The arrays must have the same shape",
    lambda args: len({x.shape for x in args}) == 1,
)
def match_pairs(
    points_1: ndarray, points_2: ndarray, targets_1: ndarray, targets_2: ndarray
) -> Tuple[ndarray, ndarray]:
    """
    Match two sets of points to two sets of targets.

    Parameters
    ----------
    points_1, points_2 : (N, D) ndarray
        Positions of objects 1 and 2.
    targets_1, targets_2 : (N, D) ndarray
        Target positions of objects 1 and 2.

    Returns
    -------
    assigned_1, assigned_2 : (N, D) ndarray
        The points assigned to the closest targets.

    Examples
    --------
    >>> points_1 = np.array([[0, 1], [6, 2], [7, 1]])
    >>> points_2 = np.array([[5, 1], [1, 0], [2, 0]])

    >>> targets_1 = np.array([[0, 0], [1, 0], [2, 0]])
    >>> targets_2 = np.array([[5, 0], [6, 0], [7, 0]])

    >>> assigned_1, assigned_2 = match_pairs(points_1, points_2, targets_1, targets_2)

    >>> assigned_1
    array([[0, 1],
           [1, 0],
           [2, 0]])

    >>> assigned_2
    array([[5, 1],
           [6, 2],
           [7, 1]])

    """
    pairs_points = zip(points_1, points_2)
    pairs_targets = zip(targets_1, targets_2)

    assignments = [assign_pair(a, b) for a, b in zip(pairs_points, pairs_targets)]

    assigned_1, assigned_2 = zip(*assignments)

    return np.array(assigned_1), np.array(assigned_2)


@require(
    "The arrays must have the same shape",
    lambda args: args.points.shape == args.targets.shape,
)
def position_accuracy(points: ndarray, targets: ndarray, max_dist: float = 10) -> float:
    """
    Calculate ratio of points within a distance from corresponding targets.

    Parameters
    ----------
    points : (N, D) ndarray
        N positions of dimension D.
    targets : (N, D) ndarray
        N target positions of dimension D.
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

    # Wrap in float() to satisfy mypy.
    return float(np.mean(distances <= max_dist))


@require(
    "The arrays must have the same shape",
    lambda args: len(
        {
            x.shape
            for x in [args.points_1, args.points_2, args.targets_1, args.targets_2]
        }
    )
    == 1,
)
def double_position_accuracy(
    points_1: ndarray,
    points_2: ndarray,
    targets_1: ndarray,
    targets_2: ndarray,
    max_dist: float = 10,
) -> float:
    """
    Return ratio of both sets of points being within both targets.

    Parameters
    ----------
    points_1, points_2 : (N, D) ndarray
        Input points.
    target_1, targets_2 : (N, D) ndarray
        Input targets.

    Returns
    -------
    float
        Double position accuracy.

    Examples
    --------
    >>> points_1 = np.array([[0, 1], [1, 5], [2, 5], [3, 4]])
    >>> points_2 = np.array([[5, 5], [6, 8], [7, 1], [8, 3]])

    >>> targets_1 = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    >>> targets_2 = np.array([[5, 5], [6, 6], [7, 7], [8, 8]])

    >>> double_position_accuracy(points_1, points_2, targets_1, targets_2, max_dist=0)
    0.0
    >>> double_position_accuracy(points_1, points_2, targets_1, targets_2, max_dist=5)
    0.75
    >>> double_position_accuracy(points_1, points_2, targets_1, targets_2, max_dist=10)
    1.0

    """
    within_dist_1 = norm(points_1 - targets_1, axis=1) <= max_dist
    within_dist_2 = norm(points_2 - targets_2, axis=1) <= max_dist

    within_both = np.column_stack((within_dist_1, within_dist_2)).all(axis=1)

    return within_both.mean()
