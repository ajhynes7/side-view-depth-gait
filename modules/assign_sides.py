"""Functions for assigning correct sides to the feet."""

import numpy as np
from numpy.linalg import norm
import pandas as pd

import modules.linear_algebra as lin
import modules.numpy_funcs as nf
import modules.pandas_funcs as pf
import modules.point_processing as pp
import modules.signals as sig
import modules.sliding_window as sw


def direction_of_pass(df_pass):
    """
    Return vector representing overall direction of motion for a walking pass.

    Parameters
    ----------
    df_pass : DataFrame
        Head and foot positions at each frame in a walking pass.
        Three columns: HEAD, L_FOOT, R_FOOT.

    Returns
    -------
    line_point : ndarray
        Point that lies on line of motion.
    direction_pass : ndarray
        Direction of motion for the walking pass.

    """
    # All head positions on one walking pass
    head_points = np.stack(df_pass.HEAD)

    # Line of best fit for head positions
    line_point, line_direction = lin.best_fit_line(head_points)

    vector_start_end = head_points[-1, :] - head_points[0, :]

    direction_pass = line_direction
    if np.dot(line_direction, vector_start_end) < 0:
        # The direction of the best fit line should be reversed
        direction_pass = -line_direction

    return line_point, direction_pass


def evaluate_foot_side(foot_points_1, foot_points_2, direction):
    """
    Yield a value indicating the side (left/right) of a foot.

    A positive value indicates right, while negative indicates left.

    Parameters
    ----------
    foot_points_1, foot_points_2 : ndarray
        (n, d) array of foot positions.
    direction : ndarray
        Vector for direction of motion.

    Returns
    -------
    side_values : ndarray
        (n, ) array of values indicating left/right direction for foot 1.

    """
    mean_foot_points = (foot_points_1 + foot_points_2) / 2
    targets = foot_points_1 - mean_foot_points

    side_values = np.array([np.cross(direction, target) for target in targets])

    return side_values


def assign_sides_portion(df_walk, direction):
    """
    Assign correct sides to feet during a section of a walking pass.

    The feet are assigned by establishing a motion correspondence for the
    section of the walking pass, then calculating a value to assign one
    tracked foot to left or right.

    Parameters
    ----------
    df_walk : DataFrame
        Data for a section of a walking pass.
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
    foot_points_l, foot_points_r = pp.track_two_objects(
        foot_points_l, foot_points_r)

    df_assigned = df_walk.copy()
    df_assigned.L_FOOT = pf.series_of_rows(foot_points_l, index=df_walk.index)
    df_assigned.R_FOOT = pf.series_of_rows(foot_points_r, index=df_walk.index)

    side_values = evaluate_foot_side(foot_points_l, foot_points_r, direction)

    if np.sum(side_values) > 0:
        # The left foot should be labelled the right foot, and vice versa
        df_assigned = pf.swap_columns(df_assigned, 'L_FOOT', 'R_FOOT')

    return df_assigned


def assign_sides_pass(df_pass, direction_pass):
    """
    Assign correct sides to feet over a full walking pass.

    The pass is split into multiple sections of frames. The splits occur when
    the feet are together. The feet are assigned to left/right on each section
    of frames.

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

    foot_points_l = np.stack(df_pass.L_FOOT)
    foot_points_r = np.stack(df_pass.R_FOOT)
    norms = norm(foot_points_l - foot_points_r, axis=1)

    # Detect peaks in the inverted foot distance signal.
    # These peaks are the frames when the feet are close together.
    signal = 1 - sig.nan_normalize(norms)
    rms = sig.root_mean_square(signal)
    peak_frames, _ = sw.detect_peaks(
        frames, signal, window_length=3, min_height=rms)

    labels = nf.label_by_split(frames, peak_frames)

    grouped_dfs = [*nf.group_by_label(df_pass, labels)]

    assigned_dfs = [
        assign_sides_portion(x, direction_pass) for x in grouped_dfs
    ]

    return pd.concat(assigned_dfs)
