"""Module for assigning left/right sides to the feet."""

from collections import namedtuple

import numpy as np
from skimage.measure import LineModelND, ransac
from skspatial.objects import Vector
from statsmodels.robust import mad


def fit_ransac(points):
    """Fit a line to 3D points with RANSAC."""

    model, is_inlier = ransac(
        points, LineModelND, min_samples=int(0.9 * len(points)), residual_threshold=3 * min(mad(points))
    )

    return model, is_inlier


def compute_basis(frames, points_head, points_a, points_b):
    """Return origin and basis vectors of new coordinate system found with RANSAC."""

    model_ransac, is_inlier = fit_ransac(points_head)

    frames_inlier = frames[is_inlier]
    points_head_inlier = points_head[is_inlier]
    points_a_inlier = points_a[is_inlier]
    points_b_inlier = points_b[is_inlier]

    points_mean = (points_a_inlier + points_b_inlier) / 2

    vector_up = Vector(np.median(points_head_inlier - points_mean, axis=0)).unit()

    vector_forward = Vector(model_ransac.params[1]).unit()
    vector_perp = vector_up.cross(vector_forward)

    point_origin = model_ransac.params[0]

    Basis = namedtuple('Basis', 'origin, forward, up, perp')
    basis = Basis(point_origin, vector_forward, vector_up, vector_perp)

    return basis, frames_inlier, points_a_inlier, points_b_inlier


def assign_sides_grouped(frames_grouped, values_side_grouped, labels_grouped):

    is_stance_grouped = labels_grouped != -1

    labels_unique = np.unique(labels_grouped[is_stance_grouped])

    # Assume all stance phases are from the left foot.
    is_label_l = np.ones_like(labels_unique, dtype=bool)

    for i, label in enumerate(labels_unique):

        is_cluster = labels_grouped == label

        frames_cluster = frames_grouped[is_cluster]
        is_frame_cluster = np.in1d(frames_grouped, frames_cluster)

        is_foot_swing = is_frame_cluster & ~is_cluster

        value_side_foot_stance = np.median(values_side_grouped[is_cluster])
        value_side_foot_swing = np.median(values_side_grouped[is_foot_swing])

        if value_side_foot_stance > value_side_foot_swing:
            is_label_l[i] = False

    labels_unique_l = labels_unique[is_label_l]
    labels_unique_r = labels_unique[~is_label_l]

    is_stance_grouped_l = np.in1d(labels_grouped, labels_unique_l)
    is_stance_grouped_r = np.in1d(labels_grouped, labels_unique_r)

    labels_grouped_l = np.copy(labels_grouped)
    labels_grouped_r = np.copy(labels_grouped)

    labels_grouped_l[~is_stance_grouped_l] = -1
    labels_grouped_r[~is_stance_grouped_r] = -1

    return labels_grouped_l, labels_grouped_r
