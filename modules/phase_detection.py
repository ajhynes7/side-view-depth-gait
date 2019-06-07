"""Module for detecting the phases of a foot during a walking pass."""

from collections import namedtuple

import numpy as np
import pandas as pd
from dpcontracts import require
from sklearn.cluster import DBSCAN
from skspatial.transformation import transform_coordinates


def stance_props(frames, points_foot, labels_stance):
    """Return properties of each stance phase from one foot in a walking pass."""

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


@require("The frames must correspond to the points.", lambda args: args.frames.size == args.points_side.shape[0])
def detect_side_stances(frames, points_side, basis):

    signal_side = transform_coordinates(points_side, basis.origin, [basis.forward])
    labels_stance = label_stance_phases(signal_side)

    return stance_props(frames, points_side, labels_stance)


def reassign_stances(frames, signal_l, signal_r, labels_l, labels_r):

    is_stance_l = labels_l != -1
    is_stance_r = labels_r != -1

    frames_stance_l = frames[is_stance_l]
    frames_stance_r = frames[is_stance_r]

    signal_stance_l = signal_l[is_stance_l]
    signal_stance_r = signal_r[is_stance_r]

    frames_stance_stacked = np.concatenate((frames_stance_l, frames_stance_r))
    signal_stance_stacked = np.vstack((signal_stance_l, signal_stance_r))

    is_from_l = np.vstack((np.ones_like(signal_stance_l), np.zeros_like(signal_stance_r))).astype(bool).flatten()

    labels_stance_stacked = label_stance_phases(signal_stance_stacked)

    labels_unique = np.unique(labels_stance_stacked[labels_stance_stacked != -1])

    is_stance_stacked_l = np.zeros_like(labels_stance_stacked).astype(bool).flatten()

    for label in labels_unique:

        is_cluster = labels_stance_stacked == label

        # Ratio of the cluster that originates from the left foot.
        ratio_l = is_from_l[is_cluster].mean()

        if ratio_l > 0.5:
            is_stance_stacked_l[is_cluster] = True

    frames_stance_stacked_l = frames_stance_stacked[is_stance_stacked_l]
    frames_stance_stacked_r = frames_stance_stacked[~is_stance_stacked_l]

    is_frame_stance_l = np.in1d(frames, frames_stance_stacked_l)
    is_frame_stance_r = np.in1d(frames, frames_stance_stacked_r)

    labels_filt_l = np.copy(labels_l)
    labels_filt_r = np.copy(labels_r)

    labels_filt_l[~is_frame_stance_l] = -1
    labels_filt_r[~is_frame_stance_r] = -1

    return labels_filt_l, labels_filt_r


def detect_stances(frames, points_a, points_b, basis):

    points_foot_grouped = nf.interweave_rows(points_a, points_b)

    frames_column = frames.reshape(-1, 1)
    frames_grouped = nf.interweave_rows(frames_column, frames_column).flatten()

    signal_grouped = transform_coordinates(points_foot_grouped, basis.origin, [basis.forward])
    values_side_grouped = transform_coordinates(points_foot_grouped, basis.origin, [basis.perp])

    labels_grouped = cl.dbscan_st(signal_grouped, times=frames_grouped, eps_spatial=5, eps_temporal=10, min_pts=7)

    labels_grouped_l, labels_grouped_r = sa.assign_sides_grouped(frames_grouped, values_side_grouped, labels_grouped)

    df_stance_l = stance_props(frames_grouped, points_foot_grouped, labels_grouped_l).assign(side='L')
    df_stance_r = stance_props(frames_grouped, points_foot_grouped, labels_grouped_r).assign(side='R')

    df_stance = pd.concat((df_stance_l, df_stance_r), sort=False)

    return df_stance
