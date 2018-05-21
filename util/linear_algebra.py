import numpy as np
from numpy.linalg import norm


def unit(v):
    """
    Unit vector of v

    Parameters
    ----------
    v : array_like
        Input vector

    Returns
    -------
    array_like
        Unit vector

    """
    return v / norm(v)


def dist_point_line(P, A, B):
    """
    Distance from a point to a line.

    Parameters
    ----------
    P : array_like
        Point in space
    A : array_like
        Point A on line
    B : array_like
        Point B on line

    Returns
    -------
    float
        Distance from point to plane

    """
    num = norm(np.cross(P - A, P - B))
    denom = norm(A - B)

    return num / denom


def dist_point_plane(P, P_plane, normal):
    """
    Distance from a point to a plane.

    Parameters
    ----------
    P : array_like
        Point in space
    normal : array_like
        Normal of plane
    plane_pt : array_like
        Point on plane

    Returns
    -------
    float
        Distance from point to plane

    """
    n_hat = unit(normal)

    return abs(np.dot(n_hat, P - P_plane))


def proj_point_line(P, A, B):
    """
    Distance from a point to a line.

    Parameters
    ----------
    P : array_like
        Point in space
    A : array_like
        Point A on line
    B : array_like
        Point B on line

    Returns
    -------
    array_like
        Projection of point P onto the line

    """
    AP = P - A 	# Vector from A to point
    AB = B - A  # Vector from A to B

    # Project point onto line
    return A + np.dot(AP, AB) / norm(AB)**2 * AB


def proj_point_plane(P, P_plane, normal):
    """
    Projects a point onto a plane.

    Parameters
    ----------
    P : array_like
        Point in space
    P_plane : array_like
        Point on plane
    normal : array_like
        Normal vector of plane

    Returns
    -------
    array_like
        Projection of point P onto the plane
    """

    unit_normal = unit(normal)

    return P - np.dot(P - P_plane, unit_normal) * unit_normal


def best_fit_line(points):
    """
    Finds the line of best fit for a set of multi-dimensional points.
    Uses singular value decomposition.

    Parameters
    ----------
    points : ndarray
         Each row is a position vector

    Returns
    -------
    centroid : array_like
        Centroid of all the points. Line of best fit passes through centroid
    direction : array_like
        Direction vector for line of best fit
        Right singular vector which corresponds to the largest
        singular value of A
    """

    # Ensure that points have no nan values
    points = points[~np.isnan(points).any(axis=1)]

    centroid = np.mean(points, axis=0)
    A = points - centroid

    _, _, vh = np.linalg.svd(A)

    direction = vh[0, :]

    return centroid, direction


def angle_direction(target_direction, forward, up):
    """
    Finds the direction (right or left) of a target,
    given an orientation specifying the forward and up directions

    Parameters
    ----------
    target : array_like
        Vector in direction of a target point
    forward : array_like
        Vector for forward direction
     up : array_like
        Vector for up direction

    Returns
    -------
    int
        Value is 1 if target is to the left,
        -1 if to the right, 0 if straight ahead
    """
    perpendicular = np.cross(forward, target_direction)

    return np.sign(np.dot(perpendicular, up))
