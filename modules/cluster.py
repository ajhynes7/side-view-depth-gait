"""Module for clustering points in space."""

from queue import Queue

import numpy as np
from scipy.spatial.distance import cdist


def dbscan(points, eps=0.5, min_pts=5):
    """
    Cluster points with DBSCAN algorithm.

    Parameters
    ----------
    points : array_like
        (n, d) array of n points with dimension d.
    eps : float, optional
        Maximum distance between two points for one to be
        considered in the neighbourhood of the other.
    min_pts : int, optional
        Number of points in a neighbourhood for a point to be considered
        a core point.

    Returns
    -------
    labels : ndarray
        (n,) array of cluster labels.

    Examples
    --------
    >>> points = [[0, 0], [1, 0], [2, 0], [0, 5], [1, 5], [2, 5]]

    >>> dbscan(points, eps=1, min_pts=2)
    array([0, 0, 0, 1, 1, 1])

    """
    dist_matrix = cdist(points, points)

    n_points = len(points)
    labels = np.zeros(n_points, dtype=int)

    label_cluster = 0

    for idx_pt in range(n_points):

        if labels[idx_pt] != 0:
            # Only unlabelled points can be considered as seed points.
            continue

        set_neighbours = region_query(dist_matrix, eps, idx_pt)

        if len(set_neighbours) < min_pts:
            # The neighbourhood of the point is smaller than the minimum.
            # The point is marked as noise.
            labels[idx_pt] = -1

        else:
            label_cluster += 1

            # Assign the point to the current cluster
            labels[idx_pt] = label_cluster

            grow_cluster(dist_matrix, labels, set_neighbours, label_cluster, eps, min_pts)

    # Subtract 1 from non-noise labels so they begin at zero (this is consistent with scikit-learn).
    labels[labels != -1] -= 1

    return labels


def grow_cluster(dist_matrix, labels, set_neighbours, label_cluster, eps, min_pts):
    """
    Grow a cluster starting from a seed point.

    Parameters
    ----------
    dist_matrix : ndarray
        (n, n) distance matrix.
    labels : ndarray
        (n,) array of cluster labels.
    set_neighbours : set
        Set of indices for neighbours of the seed point.
    label_cluster : int
        Label of the current cluster.
    eps : float
        Maximum distance between two points for one to be
        considered in the neighbourhood of the other.
    min_pts : int
        Number of points in a neighbourhood for a point to be considered
        a core point.

    Examples
    --------
    >>> points = [[0, 0], [1, 0], [2, 0], [0, 5], [1, 5], [2, 5]]

    >>> idx_pt, label_cluster = 0, 1
    >>> eps, min_pts = 1, 2

    >>> dist_matrix = cdist(points, points)
    >>> labels = np.zeros(len(points))

    >>> set_neighbours = region_query(dist_matrix, eps, idx_pt)
    >>> grow_cluster(dist_matrix, labels, set_neighbours, label_cluster, eps, min_pts)

    >>> labels
    array([1., 1., 1., 0., 0., 0.])

    """
    # Initialize a queue with the current neighbourhood.
    queue_search = Queue()

    for i in set_neighbours:
        queue_search.put(i)

    while not queue_search.empty():

        # Consider the next point in the queue.
        idx_next = queue_search.get()

        label_next = labels[idx_next]

        if label_next == -1:
            # This neighbour was labelled as noise.
            # It is now a border point of the cluster.
            labels[idx_next] = label_cluster

        elif label_next == 0:
            # The neighbour was unclaimed.
            # Add the next point to the cluster.
            labels[idx_next] = label_cluster

            set_neighbours_next = region_query(dist_matrix, eps, idx_next)

            if len(set_neighbours_next) >= min_pts:
                # The next point is a core point.
                # Add its neighbourhood to the queue to be searched.
                for i in set_neighbours_next:
                    queue_search.put(i)


def region_query(dist_matrix, eps, idx_pt):
    """
    Find which points are within a distance `eps` of the point with index `idx_pt`.

    Parameters
    ----------
    dist_matrix : ndarray
        (n, n) distance matrix.
    eps : float
        Maximum distance between two points for one to be
        considered in the neighbourhood of the other.
    idx_pt : int
        Index of the current point.

    Returns
    -------
    set
        Set of indices for the points neighbouring the current point.

    Examples
    --------
    >>> points = [[0, 0], [1, 0], [2, 0], [0, 5], [1, 5], [2, 5]]

    >>> dist_matrix = cdist(points, points)

    >>> region_query(dist_matrix, 0.5, 0)
    {0}
    >>> region_query(dist_matrix, 1, 0)
    {0, 1}
    >>> region_query(dist_matrix, 5, 0)
    {0, 1, 2, 3}

    """
    return set(np.nonzero(dist_matrix[idx_pt] <= eps)[0])
