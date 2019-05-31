"""Module for detecting the phases of a foot during a walking pass."""

from collections import namedtuple

import numpy as np
import pandas as pd
from skimage.measure import LineModelND, ransac
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from skspatial.objects import Vector
from statsmodels.robust import mad

import modules.math_funcs as mf
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


def label_stance_phases(signal):

    points_to_cluster = signal.reshape(-1, 1)

    return DBSCAN(eps=5).fit(points_to_cluster).labels_.flatten()


def filter_stances(frames, signal, labels):

    labels_unique = np.unique(labels[labels != -1])

    regressor = LinearRegression()

    def yield_slopes():

        for label in labels_unique:

            is_cluster = labels == label
            frames_cluster = frames[is_cluster].reshape(-1, 1)
            signal_cluster = signal[is_cluster].reshape(-1, 1)

            model_linear = regressor.fit(frames_cluster, signal_cluster)
            yield float(model_linear.coef_)

    slopes = [*yield_slopes()]

    is_inlier_label = mf.within_mad(slopes, c=3)

    for i, label in enumerate(labels_unique):

        if not is_inlier_label[i]:
            labels[labels == label] = -1

    return labels


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


def detect_side_stances(frames, points, coords_forward, is_side):
    """Detect stance phases of the left or right foot."""

    points_side = points[is_side]
    frames_side = frames[is_side].flatten()
    signal_side = coords_forward[is_side].flatten()

    # Detect stance phases from the signal.
    labels_side = label_stance_phases(signal_side)

    # Remove outlier stances from the labels.
    labels_side = filter_stances(frames_side, signal_side, labels_side)

    return stance_props(frames_side, points_side, labels_side)
