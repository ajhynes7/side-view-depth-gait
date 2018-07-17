"""
Module for calculating gait metrics from 3D body part positions.

Common Parameters
-----------------
df_pass : DataFrame
    DataFrame for one walking pass.
    Index values are frames.
    Columns must include 'L_FOOT', 'R_FOOT'.
    Elements are position vectors.
df_contact : DataFrame
    Each row represents a frame when a foot contacts the floor.
    Columns include 'number', 'side', 'frame'.
df_gait : DataFrame
    Each row represents a stride.
    Columns include gait metrics, e.g. 'stride_length', and the side and
    stride number.

"""
import numpy as np
import pandas as pd
from numpy.linalg import norm

import modules.pandas_funcs as pf
import modules.assign_sides as asi
import modules.sliding_window as sw
import modules.linear_algebra as lin
import modules.phase_detection as pde


def stride_metrics(side_x_i, side_y, side_x_f, *, fps=30):

    foot_x_i, foot_x_f = side_x_i.position, side_x_f.position
    foot_y = side_y.position

    foot_y_proj = lin.project_point_line(foot_y, foot_x_i, foot_x_f)

    stride_length = norm(foot_x_f - foot_x_i)
    stride_time = norm(side_x_f.frame - side_x_i.frame) / fps

    stride_velocity = stride_length / stride_time

    metrics = {'number': side_x_i.number,
               'side': side_x_i.side,

               'stride_length': stride_length,
               'stride_time': stride_time,
               'stride_velocity': stride_velocity,

               'absolute_step_length': norm(foot_x_f - foot_y),
               'step_length': norm(foot_x_f - foot_y_proj),
               'stride_width': norm(foot_y - foot_y_proj),

               'step_time': (side_x_f.frame - side_y.frame) / fps
               }

    return metrics


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


def foot_signal(foot_interest, foot_other, direction_pass):
    """
    Return a signal from foot data that is used to detect contact frames.

    Parameters
    ----------
    foot_interest, foot_other : ndarray
        Rows are foot positions.
        The first array is the foot of interest (left or right).
    direction_pass : ndarray
        Direction of motion for the walking pass.

    Returns
    -------
    signal : ndarray
        Signal from foot data.

    """
    foot_difference = foot_interest - foot_other

    signal = np.apply_along_axis(np.dot, 1, foot_difference, direction_pass)

    return signal


def foot_contacts_to_gait(df_contact):

    foot_tuples = df_contact.itertuples(index=False)

    def yield_metrics():

        for foot_tuple in sw.generate_window(foot_tuples, n=3):

            yield stride_metrics(*foot_tuple)

    return pd.DataFrame(yield_metrics())


def walking_pass_metrics(df_pass, direction_pass):
    """
    Calculate gait metrics from a single walking pass in front of the camera.

    Parameters
    ----------
    df_pass
        See module docstring.
    direction_pass : ndarray
        Direction of motion for the walking pass.

    Returns
    -------
    df_gait
        See module docstring.

    """
    frames = df_pass.index.values

    foot_l, foot_r = np.stack(df_pass.L_FOOT), np.stack(df_pass.R_FOOT)
    signal_l = foot_signal(foot_l, foot_r, direction_pass)

    series_l = pd.Series(signal_l, index=frames)

    split_frames = pde.frames_of_interest(series_l)

    df_phase_l = pde.foot_phases(split_frames, direction_pass, df_pass.L_FOOT)
    df_phase_r = pde.foot_phases(split_frames, direction_pass, df_pass.R_FOOT)

    df_grouped_l = pde.group_stance_frames(df_phase_l, '_L')
    df_grouped_r = pde.group_stance_frames(df_phase_r, '_R')
    grouped_dfs = [df_grouped_l, df_grouped_r]

    df_concat = pd.concat(grouped_dfs).sort_values('frame').reset_index()

    df_contact = pf.split_column(df_concat, column='index', delim='_',
                                 new_columns=['number', 'side'])

    df_gait = foot_contacts_to_gait(df_contact)
    df_gait = pf.column_to_suffixes(df_gait, groupby_col='side',
                                    merge_col='number')

    return df_gait


def combine_walking_passes(pass_dfs):
    """
    Combine gait metrics from all walking passes in a trial.

    Parameters
    ----------
    pass_dfs : list
        Each element is a df_pass (see module docstring).

    Returns
    -------
    df_final : DataFrame
        Each row represents a single stride.
        There can be multiple strides in a walking pass.
        Columns are gait metrics for left/right sides.

    """
    df_list = []
    for i, df_pass in enumerate(pass_dfs):

        _, direction_pass = direction_of_pass(df_pass)

        # Assign correct sides to feet
        df_assigned = asi.assign_sides_pass(df_pass, direction_pass)

        df_gait = walking_pass_metrics(df_assigned, direction_pass)
        df_gait['pass'] = i  # Add column to record the walking pass

        df_list.append(df_gait)

    df_combined = pd.concat(df_list, sort=True)

    df_final = df_combined.reset_index(drop=True)

    df_final = df_final.set_index('pass')   # Set the index to the pass column
    df_final = df_final.sort_index(axis=1)  # Sort the columns alphabetically

    return df_final
