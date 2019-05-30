"""Module for detecting the phases of a foot during a walking pass."""

from collections import namedtuple

import numpy as np
import pandas as pd
from skimage.measure import LineModelND, ransac
from skspatial.objects import Vector
from statsmodels.robust import mad

import modules.cluster as cl
import modules.numpy_funcs as nf


def fit_ransac(points):
    """Fit a line to 3D points with RANSAC."""

    thresh = 3 * min(mad(points))
    model, is_inlier = ransac(points, LineModelND, min_samples=2, residual_threshold=thresh)

    return model, is_inlier


def compute_basis(frames, points_head, points_a, points_b):

    points_head_grouped = nf.interweave_rows(points_head, points_head)
    points_foot_grouped = nf.interweave_rows(points_a, points_b)

    frames_column = frames.reshape(-1, 1)
    frames_grouped = nf.interweave_rows(frames_column, frames_column)

    model_ransac, is_inlier = fit_ransac(points_foot_grouped)

    points_head_inlier = points_head_grouped[is_inlier]
    points_foot_inlier = points_foot_grouped[is_inlier]
    frames_grouped_inlier = frames_grouped[is_inlier]

    vector_up = Vector(np.median(points_head_inlier - points_foot_inlier, axis=0)).unit()
    vector_forward = Vector(model_ransac.params[1]).unit()
    vector_perp = vector_up.cross(vector_forward)

    point_origin = model_ransac.params[0]

    Basis = namedtuple('Basis', 'origin, forward, up, perp')
    basis = Basis(point_origin, vector_forward, vector_up, vector_perp)

    return basis, points_foot_inlier, frames_grouped_inlier


def label_stance_phases(frames, points_2d):

    return cl.dbscan_st(points_2d, eps_spatial=5)


def stance_props(frames, points_foot, labels_stance):
    """
    Return properties of each stance phase from one foot in a walking pass.

    """
    labels_unique = np.unique(labels_stance[labels_stance != -1])

    Stance = namedtuple('Stance', ['frame_i', 'frame_f', 'position'])

    def yield_props():

        for label in labels_unique:

            is_cluster = labels_stance == label

            points_foot_cluster = points_foot[is_cluster]
            point_foot_med = np.median(points_foot_cluster, axis=0)

            frames_cluster = frames[is_cluster]
            frame_i = frames_cluster.min()
            frame_f = frames_cluster.max()

            yield Stance(frame_i, frame_f, point_foot_med)

    return pd.DataFrame(yield_props())


def assign_sides_pass(df_stance):
    """Assign sides to detected stance clusters in a walking pass."""

    points_2d = np.stack(df_stance.position)
    is_left = points_2d[:, 0] < 0

    df_stance['side'] = ['L' if x else 'R' for x in is_left]

    return df_stance
