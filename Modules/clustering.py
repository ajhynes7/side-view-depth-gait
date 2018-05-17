import numpy as np
from general import gaussian, centre_of_mass, closest_point 
from scipy.spatial.distance import cdist
from functools import partial


def gaussian_kernel_shift(points, mean_pos, radius):
    
    distances = np.linalg.norm(points - mean_pos, axis=1)
    
    # Gaussian kernel with standard deviation set to the radius parameter
    K = partial(gaussian, mu=0, sigma=radius)
    masses = K(distances)

    return centre_of_mass(points, masses)

    
def flat_kernel_shift(points, mean_pos, radius):
    
    distances = np.linalg.norm(points - mean_pos, axis=1)
    
    within_radius = distances <= radius
    
    points_in_radius = points[within_radius, :]
    n_in_radius, _ = points_in_radius.shape
    
    masses = np.ones(n_in_radius)

    return centre_of_mass(points_in_radius, masses)


def shift_to_convergence(points, mean_pos, shift_func, radius, eps):

    while True:
    
        prev_mean_pos = mean_pos

        mean_pos = shift_func(points, mean_pos, radius)
        
        if np.linalg.norm(prev_mean_pos - mean_pos) < eps:
            # Mean position has converged
            
            distances = np.linalg.norm(points - mean_pos, axis=1)
            in_radius = distances <= radius
            break

    return mean_pos, in_radius


def mean_shift(points, shift_func, radius=1, eps=0.001):

    n_points, n_dimensions = points.shape

    index_matrix = np.full((n_points, n_points), False)
    all_centroids = np.full((n_points, n_dimensions), np.nan)

    for i, mean_pos in enumerate(points):
        # Shift mean until convergence
        mean_pos, in_radius = shift_to_convergence(points, mean_pos, shift_func, radius, eps)

        index_matrix[i, :] = in_radius
        all_centroids[i, :] = mean_pos

    _, unique_indices, labels = np.unique(index_matrix,
                                          return_index=True, return_inverse=True, axis=0)

    centroids = all_centroids[unique_indices, :]
    k, _ = centroids.shape

    return labels, centroids, k




