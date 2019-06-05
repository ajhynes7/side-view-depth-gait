"""Module for establishing a motion correspondence between multiple points."""

import numpy as np
from numpy.linalg import norm


def proximal_uniformity(X_frames, phi_k_minus_1, indices_points):

    m = len(phi_k_minus_1)
    p, q, r = indices_points

    X_k_minus_1, X_k, X_k_plus_1 = X_frames

    cost_matrix_1 = np.zeros((m, m))
    cost_matrix_2 = np.zeros_like(cost_matrix_1)

    for x in range(m):
        for z in range(m):

            point_a = X_k_minus_1[x]
            point_b = X_k[phi_k_minus_1[x]]
            point_c = X_k_plus_1[z]

            vector_ab = np.subtract(point_b, point_a)
            vector_bc = np.subtract(point_c, point_b)

            cost_matrix_1[x, z] = norm(vector_ab - vector_bc)
            cost_matrix_2[x, z] = norm(vector_bc)

    C_1 = np.sum(cost_matrix_1)
    C_2 = np.sum(cost_matrix_2)

    point_p = X_k_minus_1[p]
    point_q = X_k[q]
    point_r = X_k_plus_1[r]

    vector_pq = np.subtract(point_q, point_p)
    vector_qr = np.subtract(point_r, point_q)

    return norm(vector_pq - vector_qr) / C_1 + norm(vector_qr) / C_2


def frame_correspondence(X_frames, phi_k_minus_1):

    # Number of points on each frame.
    m = len(phi_k_minus_1)

    # %% Construct matrix M.

    M = np.full((m, m), np.nan)

    for i in range(m):
        for j in range(m):

            for p in range(m):

                if phi_k_minus_1[p] == i:
                    indices_points = [p, i, j]
                    M[i, j] = proximal_uniformity(X_frames, phi_k_minus_1, indices_points)

    # %% Compute phi_k, the assignment of points in frame k to points in frame k + 1

    phi_k = np.zeros_like(phi_k_minus_1)

    for _ in range(m):

        # %% Construct priority matrix B.
        B = np.full((m, m), np.nan)

        for i in range(m):

            if np.isnan(M[i]).all():
                # This row has been already masked with NaN.
                continue

            # Find the minimum column M.
            l_i = np.nanargmin(M[i])

            # Sum of row i of M, excluding column l_i.
            sum_row = np.nansum(M[i]) - M[i, l_i]

            # Sum of column l_i of M, excluding row i.
            sum_col = np.nansum(M[:, l_i]) - M[i, l_i]

            B[i, l_i] = sum_row + sum_col

        # Find row and column of max value of B.
        row_max, col_max = np.nonzero(B == np.nanmax(B))

        if len(row_max) > 1:
            row_max, col_max = row_max[0], col_max[0]

        phi_k[row_max] = col_max

        # Mask the max row and column in M.
        M[row_max, :] = np.nan
        M[:, col_max] = np.nan

    return phi_k.astype(int)


def correspond_motion(list_points, correspondence_initial):
    """
    Establish a motion correspondence between points over multiple frames.

    Parameters
    ----------
    list_points : list
        Each element is an (m, d) array of m points with dimension d.
    correspondence_initial : array_like
        (m,) array
        Initial correspondence of points from frames f_0 to f_1.

    Returns
    -------
    assignment : ndarray
        (n, m) array for n frames and m points.
        Element (i, j) is the label of point j on frame i.

    References
    ----------
    Rangarajan, K., & Shah, M. (1991). Establishing motion correspondence.
    CVGIP: image understanding, 54(1), 56-73.

    """
    n = len(list_points)  # Number of frames.
    m = len(correspondence_initial)  # Number of points.

    # Begin with previously known correspondence between frames.
    list_phi = [None] * (n - 1)
    list_phi[0] = correspondence_initial

    for k in range(1, n - 1):

        # Points from previous, current, and next frame.
        X_frames = (list_points[k - 1], list_points[k], list_points[k + 1])

        list_phi[k] = frame_correspondence(X_frames, list_phi[k - 1])

    # %% Assign labels to the points.

    assignment = np.zeros((n, m))
    assignment[0] = range(m)

    for k in range(1, n):
        correspondence_prev = list_phi[k - 1]
        assignment[k, :] = assignment[k - 1, correspondence_prev]

    return assignment

def assign_points(points_stacked, assignment):
    """
    Assign points to their respective groups with an assignment matrix.

    Parameters
    ----------
    points_stacked : ndarray
        (n_frames, n_dim, n_points) array for points over multiple frames.
    assignment : ndarray
        (n_frames, n_points) array.
        Each row is the assignment of labels

    Examples
    --------
    >>> points_a = [[1, 6], [2, 6], [3, 0], [4, 3]]
    >>> points_b = [[1, 0], [2, 0], [3, 6], [4, 6]]
    >>> points_c = [[1, 3], [2, 3], [3, 3], [4, 0]]

    >>> points_stacked = np.dstack((points_a, points_b, points_c))
    >>> assignment = correspond_motion(points_stacked, [0, 1, 2])

    >>> points_stacked = np.dstack((points_a, points_b, points_c))
    >>> points_stacked_assigned = assign_points(points_stacked, assignment)

    >>> points_stacked_assigned[:, :, 0]
    array([[1, 6],
           [2, 6],
           [3, 6],
           [4, 6]])

    >>> points_stacked_assigned[:, :, 1]
    array([[1, 0],
           [2, 0],
           [3, 0],
           [4, 0]])

    >>> points_stacked_assigned[:, :, 2]
    array([[1, 3],
           [2, 3],
           [3, 3],
           [4, 3]])

    """
    points_stack_assigned = np.zeros_like(points_stacked)
    n_frames, _, n_points = points_stacked.shape

    for i in range(n_frames):
        row_labels = assignment[i]

        for j in range(n_points):
            points_stack_assigned[i, :, row_labels[j]] = points_stacked[i, :, j]

    return points_stack_assigned
