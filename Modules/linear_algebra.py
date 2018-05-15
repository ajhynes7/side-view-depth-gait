
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
    v_unit : array_like
        Unit vector

    """
    return v / norm(v)


def dist_point_plane(P, normal, plane_pt):
    """
    Distance from a point to a plane.

    Parameters
    ----------
    P : array_like
        Position in space

    normal : array_like
        Normal of plane

    plane_pt : array_like
        Point on plane

    Returns
    -------
    d : float
        Distance from point to plane

    """
    n_hat = unit(normal)

    return abs(np.dot(n_hat, P - plane_pt))


def dist_point_line(P, A, B):
    """
    Distance from a point to a line.

    Parameters
    ----------
    P : array_like
        Position in space

    A : array_like
        Point A on line

    B : array_like
        Point B on line

    Returns
    -------
    d : float
        Distance from point to plane

    """
    num = norm(np.cross(P - A, P - B))
    denom = norm(A - B)

    return num / denom


def proj_point_line(P, A, B):
    """
    Distance from a point to a line.

    Parameters
    ----------
    P : array_like
        Position in space

    A : array_like
        Point A on line

    B : array_like
        Point B on line

    Returns
    -------
    P_line : array_like
        Projection of point P onto line

    """
    AP = P - A 	# Vector from A to point
    AB = B - A  # Vector from A to B

    # Project point onto line
    return A + np.dot(AP, AB) / norm(AB)**2 * AB
