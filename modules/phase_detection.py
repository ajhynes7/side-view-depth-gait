"""Module for detecting the phases of a foot during a walking pass."""

import copy
from collections import namedtuple
from itertools import product

import numpy as np
from sklearn.cluster import DBSCAN
from statsmodels.robust import mad

import modules.math_funcs as mf
import modules.numpy_funcs as nf


def detect_stance_phases(signal):

    dbscan = DBSCAN(eps=5).fit(signal.reshape(-1, 1))

    labels_stance = dbscan.labels_.astype(float)
    labels_stance[labels_stance == -1] = np.nan

    return labels_stance


def stance_stats(signal, labels_stance):

    groups_stance = [*nf.group_by_label(signal, labels_stance)]

    StanceStats = namedtuple('StanceStats', ['number', 'limits', 'count'])

    for i, group_stance in enumerate(groups_stance):

        med_value = np.median(group_stance)
        mad_ = mad(group_stance)
        count = len(group_stance)

        limits = mf.limits(med_value, mad_)

        yield StanceStats(i, limits, count)


def filter_stances(signal_l, signal_r, labels_l, labels_r):

    stats_stance_l = stance_stats(signal_l, labels_l)
    stats_stance_r = stance_stats(signal_r, labels_r)

    pairs_stats = product(stats_stance_l, stats_stance_r)

    labels_filt_l = copy.copy(labels_l)
    labels_filt_r = copy.copy(labels_r)

    for stats_l, stats_r in pairs_stats:

        if mf.check_overlap(stats_l.limits, stats_r.limits):
            # The left and right stance phases overlap.
            # Keep the phase with more values.

            if stats_l.count < stats_r.count:
                # Delete the left stance phase.
                labels_filt_l[labels_l == stats_l.number] = np.nan

            elif stats_l.count > stats_r.count:
                # Delete the right stance phase.
                labels_filt_r[labels_r == stats_r.number] = np.nan

    return labels_filt_l, labels_filt_r


def stance_medians(frames, points, labels_stance):

    groups_frames = nf.group_by_label(frames, labels_stance)
    groups_points = nf.group_by_label(points, labels_stance)

    Stance = namedtuple('Stance', ['num_stance', 'position', 'frame_i', 'frame_f'])

    for i, (group_frames, group_points) in enumerate(zip(groups_frames, groups_points)):

        point_med = np.median(group_points, axis=0)

        frame_i, frame_f = group_frames[[0, -1]]

        yield Stance(i, point_med, frame_i, frame_f)
