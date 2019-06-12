"""Module for assigning left/right sides to the feet."""

from collections import namedtuple

import numpy as np
from dpcontracts import require, ensure
from skimage.measure import LineModelND, ransac
from skspatial.objects import Vector
from statsmodels.robust import mad

import modules.numpy_funcs as nf


@require("The input points must be 3D.", lambda args: args.points.shape[1] == 3)
@ensure("The output points must be 2D.", lambda _, result: result.shape[1] == 2)
def reduce_dimension(points):

    return np.column_stack((points[:, 0], points[:, 2]))


def fit_ransac(points):
    """Fit a line to 3D points with RANSAC."""

    model, is_inlier = ransac(
        points, LineModelND, min_samples=int(0.9 * len(points)), residual_threshold=3 * min(mad(points))
    )

    return model, is_inlier


def compute_basis(frames, points_a, points_b):
    """Return origin and basis vectors of new coordinate system found with RANSAC."""

    frames_grouped = np.repeat(frames, 2)
    points_grouped = nf.interweave_rows(points_a, points_b)

    points_grouped_2d = reduce_dimension(points_grouped)

    model_ransac, is_inlier = fit_ransac(points_grouped_2d)
    point_origin, vector_forward = model_ransac.params

    vector_up = [0, 0, 1]
    vector_perp = Vector(vector_forward).cross(vector_up)[:-1]

    frames_grouped_inlier = frames_grouped[is_inlier]
    points_grouped_inlier = points_grouped_2d[is_inlier]

    Basis = namedtuple('Basis', 'origin, forward, up, perp')
    basis = Basis(point_origin, vector_forward, vector_up, vector_perp)

    return basis, frames_grouped_inlier, points_grouped_inlier


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
