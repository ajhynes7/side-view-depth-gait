import numpy as np

def norm_ratio(a, b):
    if a <= b:
        return a / b
    else:
        return b / a


def inside_spheres(dist_matrix, point_nums, r):
    
    n_points = len(dist_matrix)
    
    in_spheres = np.full(n_points, False)

    for i in point_nums:
        
        distances = dist_matrix[i, :]
        
        in_current_sphere = distances <= r
        in_spheres = in_spheres | in_current_sphere
            
    return in_spheres