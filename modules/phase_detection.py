"""Module for detecting the phases of a foot during a walking pass."""

from collections import namedtuple

import numpy as np
import pandas as pd
from skspatial.transformation import transform_coordinates

import modules.cluster as cl
import modules.numpy_funcs as nf
import modules.side_assignment as sa


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


def detect_stances(frames_grouped, points_foot_grouped, basis):

    signal_grouped = transform_coordinates(points_foot_grouped, basis.origin, [basis.forward])
    values_side_grouped = transform_coordinates(points_foot_grouped, basis.origin, [basis.perp])

    labels_grouped = cl.dbscan_st(signal_grouped, times=frames_grouped, eps_spatial=5, eps_temporal=10, min_pts=7)

    labels_grouped_l, labels_grouped_r = sa.assign_sides_grouped(frames_grouped, values_side_grouped, labels_grouped)

    df_stance_l = stance_props(frames_grouped, points_foot_grouped, labels_grouped_l).assign(side='L')
    df_stance_r = stance_props(frames_grouped, points_foot_grouped, labels_grouped_r).assign(side='R')

    df_stance = pd.concat((df_stance_l, df_stance_r), sort=False)

    return df_stance
