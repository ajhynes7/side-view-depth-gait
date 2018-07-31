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

import modules.assign_sides as asi
import modules.linear_algebra as lin
import modules.numpy_funcs as nf
import modules.pandas_funcs as pf
import modules.phase_detection as pde
import modules.sliding_window as sw


def iterpolate_points(df_pass):
    """
    Interpolate points for missing frames in a walking pass.

    Parameters
    ----------
    df_pass : See module docstring.

    Returns
    -------
    df_interp : DataFrame
        DataFrame for walking pass with no gaps in the frames.

    """
    # Make a dataframe of the walking pass with no missing frames
    df_consec = pf.make_index_consecutive(df_pass)
    df_consec = df_consec.applymap(lambda x: x if isinstance(x, np.ndarray)
                                   else np.full(3, np.nan))

    df_interp = df_consec.copy()

    for column_name in df_consec:

        points = np.stack(df_consec[column_name])
        points_interp = np.apply_along_axis(nf.interp_nan, 0, points)

        points_series = pf.series_of_rows(points_interp, index=df_consec.index)

        df_interp[column_name] = points_series

    return df_interp


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


def foot_contacts_to_gait(df_contact):
    """
    Calculate gait metrics from all instances of the feet contacting the floor.

    Parameters
    ----------
    df_contact : DataFrame
        Each row represents an instance of a foot contacting the floor.
        Columns are 'frame', 'position', 'number', 'side'.

    Returns
    -------
    df_gait : DataFrame
        Each row represents a set of gait metrics calculated from one stride.
        Columns include gait metric names, e.g., stride_velocity.
        Columns also include 'number' and 'side'.

    """
    foot_tuples = df_contact.itertuples(index=False)

    def yield_metrics():

        for foot_tuple in sw.generate_window(foot_tuples, n=3):

            yield stride_metrics(*foot_tuple)

    df_gait = pd.DataFrame(yield_metrics())

    return df_gait


def stride_metrics(foot_x_i, foot_y, foot_x_f, *, fps=30):
    """
    Calculate gait metrics from a single stride.

    Parameters
    ----------
    foot_x_i : namedtuple
        Single result from pandas DataFrame.itertuples() method.
        Includes fields of 'frame', 'position', 'number', and 'side'.
        Represents the initial foot on side x.
    foot_y : namedtuple
        Represents the foot on side y.
    foot_x_f : namedtuple
        Represents the final foot on side x.
    fps : int, optional
        Camera frame rate in frames per second (default 30).

    Returns
    -------
    metrics : dict
        Dictionary containing gait metric names and values.

    """
    pos_x_i, pos_x_f = foot_x_i.position, foot_x_f.position
    pos_y = foot_y.position

    pos_y_proj = lin.project_point_line(pos_y, pos_x_i, pos_x_f)

    stride_length = norm(pos_x_f - pos_x_i)
    stride_time = (foot_x_f.frame - foot_x_i.frame) / fps

    stride_velocity = stride_length / stride_time

    metrics = {'number': foot_x_i.number,
               'side': foot_x_i.side,

               'stride_length': stride_length,
               'stride_time': stride_time,
               'stride_velocity': stride_velocity,

               'absolute_step_length': norm(pos_x_f - pos_y),
               'step_length': norm(pos_x_f - pos_y_proj),
               'stride_width': norm(pos_y - pos_y_proj),

               'step_time': (foot_x_f.frame - foot_y.frame) / fps
               }

    return metrics


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
    # df_phase_l = pde.get_phase_dataframe(df_pass.L_FOOT, direction_pass)
    # df_phase_r = pde.get_phase_dataframe(df_pass.R_FOOT, direction_pass)

    df_contact_l = pde.get_contacts(df_pass.L_FOOT, direction_pass, '_L')
    df_contact_r = pde.get_contacts(df_pass.R_FOOT, direction_pass, '_R')

    contact_dfs = [df_contact_l, df_contact_r]

    df_stacked = pd.concat(contact_dfs).sort_values('frame').reset_index()

    df_contact = pf.split_column(df_stacked, column='index', delim='_',
                                 new_columns=['number', 'side'])

    df_gait = foot_contacts_to_gait(df_contact)

    if not df_gait.empty:

        df_gait = pf.split_and_merge(df_gait, merge_col='number',
                                     split_col='side', split_vals=('L', 'R'))

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

        df_pass = iterpolate_points(df_pass)

        # Ensure there are no missing frames in the walking pass
        df_pass = pf.make_index_consecutive(df_pass)

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
