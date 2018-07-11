"""Functions for assigning correct sides to the feet."""
import numpy as np
import pandas as pd

import modules.signals as sig
import modules.general as gen
import modules.math_funcs as mf
import modules.pandas_funcs as pf
import modules.linear_algebra as lin
import modules.point_processing as pp


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
    line_point, direction_pass = lin.best_fit_line(head_points)

    return line_point, direction_pass


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

    Yields
    ------
    float
        Value indicating left/right direction for foot 1.

    """
    for head, foot_1, foot_2 in zip(head_points, foot_points_1, foot_points_2):

        mean_foot = (foot_1 + foot_2) / 2
        up = head - mean_foot

        target = foot_1 - mean_foot

        yield lin.target_side_value(direction, up, target)


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
    # Find a motion correspondence so the foot sides do not switch abruptly
    foot_points_l, foot_points_r = pp.track_two_objects(df_walk.L_FOOT,
                                                        df_walk.R_FOOT)

    df_assigned = df_walk.copy()
    df_assigned.L_FOOT = pf.series_of_rows(foot_points_l, index=df_walk.index)
    df_assigned.R_FOOT = pf.series_of_rows(foot_points_r, index=df_walk.index)

    head_points = np.stack(df_walk.HEAD)
    side_values = evaluate_foot_side(head_points, foot_points_l,
                                     foot_points_r, direction)

    if sum(side_values) > 0:

        # The left foot should be labelled the right foot, and vice versa
        df_assigned = pf.swap_columns(df_assigned, 'L_FOOT', 'R_FOOT')

    return df_assigned


def assign_sides_pass(df_pass):
    """
    Assign correct sides to feet over a full walking pass.

    Parameters
    ----------
    df_pass : DataFrame
        Head and foot positions at each frame in a walking pass.
        Three columns: HEAD, L_FOOT, R_FOOT.

    Returns
    -------
    DataFrame
        New DataFrame for walking pass with feet assigned to correct sides.

    """
    _, direction_pass = direction_of_pass(df_pass)

    foot_dist = (df_pass.L_FOOT - df_pass.R_FOOT).apply(np.linalg.norm)

    deriv = sig.window_derivative(foot_dist, 5)
    deriv_2 = sig.window_derivative(deriv, 5).dropna()

    masses = mf.normalize_array(deriv_2.values)
    peak_frames, mid_frames = sig.mean_shift_peaks(deriv_2, masses=masses,
                                                   kernel='gaussian', radius=3)

    frames = df_pass.index.values
    labels = gen.label_by_split(frames, peak_frames)

    grouped_dfs = list(gen.group_by_label(df_pass, labels))

    assigned_dfs = [assign_sides_portion(x, direction_pass) for x in
                    grouped_dfs]

    return pd.concat(assigned_dfs)
