"""Module for detecting the phases of a foot during a walking pass."""

from collections import namedtuple

import numpy as np
import pandas as pd
from dpcontracts import require
from sklearn.cluster import DBSCAN


def label_stance_phases(signal):
    """Detect stance phases and return corresponding labels."""

    labels = DBSCAN(eps=5, min_samples=10).fit(signal.reshape(-1, 1)).labels_

    return labels.flatten()


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


@require("The points must be 2D.", lambda args: args.points_2d_side.shape[1] == 2)
@require("The frames must correspond to the points.", lambda args: args.frames.size == args.points_2d_side.shape[0])
def detect_side_stances(frames, points_2d_side):

    signal_side = points_2d_side[:, 1]
    labels_stance = label_stance_phases(signal_side)

    return stance_props(frames, points_2d_side, labels_stance)
