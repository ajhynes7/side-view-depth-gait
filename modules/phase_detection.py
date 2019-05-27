"""Module for detecting the phases of a foot during a walking pass."""

from collections import namedtuple

import numpy as np
import pandas as pd
from skimage.measure import LineModelND, ransac
from sklearn.cluster import DBSCAN
from skspatial.objects import Vector
from statsmodels.robust import mad

import modules.math_funcs as mf


def fit_ransac(points_foot):
    """Fit a line to the foot points with RANSAC."""

    thresh = 3 * min(mad(points_foot))
    model, is_inlier = ransac(points_foot, LineModelND, min_samples=2, residual_threshold=thresh)

    return model, is_inlier


def compute_basis(points_head, points_a, points_b, model_ransac):

    # Define the up direction as the median of vectors to the head.
    points_mean = (points_a + points_b) / 2
    vector_up = Vector(np.median(points_head - points_mean, axis=0)).unit()

    point_origin, vector_forward = model_ransac.params
    vector_perp = vector_up.cross(vector_forward)

    Basis = namedtuple('Basis', 'origin, forward, up, perp')
    basis = Basis(point_origin, vector_forward, vector_up, vector_perp)

    return basis


def label_stance_phases(points_2d):

    return DBSCAN(eps=5).fit(points_2d).labels_


def stance_props(frames, points_foot, labels_stance):
    """
    Return properties of each stance phase from one foot in a walking pass.

    """
    labels_unique = np.unique(labels_stance[labels_stance != -1])

    Stance = namedtuple('Stance', ['frame_i', 'frame_f', 'position'])

    def yield_props():

        for label in labels_unique:

            is_cluster = labels_stance == label

            frames_cluster = np.unique(frames[is_cluster])
            points_foot_cluster = points_foot[is_cluster]

            point_foot_med = np.median(points_foot_cluster, axis=0)

            # Remove outlier frames from the cluster.
            frames_cluster = mf.mad_filter(frames_cluster, c=3)

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
