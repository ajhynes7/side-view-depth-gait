import numpy as np
from numpy.linalg import norm


def assign_swing_stance(foot_points_i, foot_points_f):
    """
    Assign initial and final foot points to the stance and swing foot.

    Returns the combination that minimizes
    the distance travelled by the stance foot.

    Parameters
    ----------
    foot_points_i : ndarray
        The two initial foot points (at start of stride).
    foot_points_f : ndarray
        The two final foot points (at end of stride).

    Returns
    -------
    points_i : ndarray
        Initial foot points in order of (stance, swing).
    points_f : ndarray
        Final foot points in order of (stance, swing).

    """
    min_dist = np.inf
    points_i, points_f = [], []

    for a in range(2):
        for b in range(2):

            stance_i = foot_points_i[a, :]
            stance_f = foot_points_f[b, :]

            swing_i = foot_points_i[1 - a, :]
            swing_f = foot_points_f[1 - b, :]

            d_stance = norm(stance_f - stance_i)

            if d_stance < min_dist:

                min_dist = d_stance

                points_i = np.array([stance_i, swing_i])
                points_f = np.array([stance_f, swing_f])

    return points_i, points_f
