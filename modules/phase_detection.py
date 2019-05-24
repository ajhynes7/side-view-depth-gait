"""Module for detecting the phases of a foot during a walking pass."""

from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

import modules.math_funcs as mf


def stance_props(frames_grouped, points_foot_grouped, labels):
    """
    Return properties of each stance phase of a walking pass.

    """
    labels_unique = np.unique(labels[labels != -1])

    Stance = namedtuple('Stance', ['frame_i', 'frame_f', 'position'])

    for label in labels_unique:

        is_cluster = labels == label

        frames_cluster = np.unique(frames_grouped[is_cluster])
        points_foot_cluster = points_foot_grouped[is_cluster]

        point_foot_med = np.median(points_foot_cluster, axis=0)

        frame_i = frames_cluster.min()
        frame_f = frames_cluster.max()

        yield Stance(frame_i, frame_f, point_foot_med)


def detect_stance_phases(frames, points_a, points_b):

    frames_grouped = np.concatenate((frames, frames))
    points_foot_grouped = np.vstack((points_a, points_b))

    dbscan = DBSCAN(eps=7, min_samples=7).fit(points_foot_grouped)
    labels = dbscan.labels_

    df_stance = pd.DataFrame(stance_props(frames_grouped, points_foot_grouped, labels))

    return df_stance
