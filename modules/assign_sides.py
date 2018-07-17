"""Functions for assigning correct sides to the feet."""

import numpy as np
import pandas as pd
from numpy.linalg import norm

import modules.signals as sig
import modules.numpy_funcs as nf
import modules.pandas_funcs as pf
import modules.sliding_window as sw
import modules.linear_algebra as lin
import modules.point_processing as pp


def evaluate_foot_side(head_points, foot_points_1, foot_points_2, direction):
    """
    Yield a value indicating the side (left/right) of a foot.

    A positive value indicates right, while negative indicates left.

    Parameters
    ----------
    head_points : ndarray
        (n, 3) array of head positions.
    foot_points_1, foot_points_2 : ndarray
        (n, 3) array of foot positions.
    direction : ndarray
        Vector for direction of motion.

    Returns
    -------
    side_values : ndarray
        (n, ) array of values indicating left/right direction for foot 1.

    """
    side_values = np.zeros(len(head_points))

    zipped = zip(head_points, foot_points_1, foot_points_2)

    for i, (head, foot_1, foot_2) in enumerate(zipped):

        mean_foot = (foot_1 + foot_2) / 2
        up = head - mean_foot

        target = foot_1 - mean_foot

        side_values[i] = lin.target_side_value(direction, up, target)

    return side_values


def assign_sides_portion(df_walk, direction):
    """
    Assign correct sides to feet during a portion of a walking pass.

    Parameters
    ----------
    df_walk : DataFrame
        Data for a portion of a walking pass.
        Three columns: HEAD, L_FOOT, R_FOOT.
    direction : ndarray
        Vector for direction of motion.

    Returns
    -------
    df_assigned : DataFrame
        Walking data after foot sides have been assigned.

    """
    foot_points_l = np.stack(df_walk.L_FOOT)
    foot_points_r = np.stack(df_walk.R_FOOT)

    # Find a motion correspondence so the foot sides do not switch abruptly
    foot_points_l, foot_points_r = pp.track_two_objects(foot_points_l,
                                                        foot_points_r)

    df_assigned = df_walk.copy()
    df_assigned.L_FOOT = pf.series_of_rows(foot_points_l, index=df_walk.index)
    df_assigned.R_FOOT = pf.series_of_rows(foot_points_r, index=df_walk.index)

    head_points = np.stack(df_walk.HEAD)
    side_values = evaluate_foot_side(head_points, foot_points_l,
                                     foot_points_r, direction)

    if np.sum(side_values) > 0:

        # The left foot should be labelled the right foot, and vice versa
        df_assigned = pf.swap_columns(df_assigned, 'L_FOOT', 'R_FOOT')

    return df_assigned


def assign_sides_pass(df_pass, direction_pass):
    """
    Assign correct sides to feet over a full walking pass.

    Parameters
    ----------
    df_pass : DataFrame
        Head and foot positions at each frame in a walking pass.
        Three columns: HEAD, L_FOOT, R_FOOT.
    direction_pass : ndarray
        Direction of motion for the walking pass.

    Returns
    -------
    DataFrame
        New DataFrame for walking pass with feet assigned to correct sides.

    """
    frames = df_pass.index.values

    foot_pos_l = np.stack(df_pass.L_FOOT)
    foot_pos_r = np.stack(df_pass.R_FOOT)
    norms = np.apply_along_axis(norm, 1, foot_pos_l - foot_pos_r)

    signal = pd.Series(1 - sig.normalize(norms), index=frames)

    rms = sig.root_mean_square(signal.values)

    peak_frames, _ = sw.detect_peaks(frames, signal, window_length=3,
                                     min_height=rms)

    labels = nf.label_by_split(frames, peak_frames)

    grouped_dfs = [*nf.group_by_label(df_pass, labels)]

    assigned_dfs = [assign_sides_portion(x, direction_pass) for x in
                    grouped_dfs]

    return pd.concat(assigned_dfs)
