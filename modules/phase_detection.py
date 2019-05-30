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

    return cl.dbscan_st(points_2d, frames, eps_spatial=5, eps_temporal=15)


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

    df_stance = df_stance.sort_values('frame_i')

    points_stance = np.stack(df_stance.position)
    points_stance_even = points_stance[::2]
    points_stance_odd = points_stance[1::2]

    value_side_even = np.median(points_stance_even[:, 0])
    value_side_odd = np.median(points_stance_odd[:, 0])

    # Initially assume even stances belong to the left foot.
    side_even, side_odd = 'L', 'R'

    if value_side_even > value_side_odd:
        side_even, side_odd = side_odd, side_even

    n_stances = df_stance.shape[0]

    sides = np.full(n_stances, None)
    sides[::2] = side_even
    sides[1::2] = side_odd

    nums_stance = np.zeros(n_stances).astype(int)
    nums_stance[::2] = np.arange(points_stance_even.shape[0])
    nums_stance[1::2] = np.arange(points_stance_odd.shape[0])

    return df_stance.assign(side=sides, num_stance=nums_stance)
