import numpy as np
from general import centre_of_mass, closest_point
from scipy.spatial.distance import cdist


def shift_to_convergence(points, mean_pos, shift_func, radius=1):

    n_points = len(points)
    within_radius = np.full(n_points, True)

    dist_matrix = cdist(points, points)

    while True:

        prev_within_radius = within_radius

        _, nearest_index = closest_point(points, mean_pos)
        distances = dist_matrix[:, nearest_index]

        within_radius = distances <= radius

        if np.array_equal(within_radius, prev_within_radius):
            break

        mean_pos = shift_func(points, mean_pos, radius)

    return mean_pos, within_radius


def mean_shift(points, shift_func, radius=1):

    n_points, n_dimensions = points.shape

    index_matrix = np.full((n_points, n_points), False)
    all_centroids = np.full((n_points, n_dimensions), np.nan)

    for i, mean_pos in enumerate(points):
        # Shift mean until convergence
        mean_pos, within_radius = shift_to_convergence(points, mean_pos, shift_func, radius)

        index_matrix[i, :] = within_radius
        all_centroids[i, :] = mean_pos

    _, unique_indices, labels = np.unique(index_matrix,
                                          return_index=True, return_inverse=True, axis=0)

    centroids = all_centroids[unique_indices, :]
    k, _ = centroids.shape

    return labels, centroids, k

