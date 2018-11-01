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
    Columns include 'stride', 'side', 'frame'.
df_gait : DataFrame
    Each row represents a stride.
    Columns include gait metrics, e.g. 'stride_length', and the side and
    stride number.

"""
import numpy as np
from numpy.linalg import norm
import pandas as pd

import modules.assign_sides as asi
import modules.linear_algebra as lin
import modules.numpy_funcs as nf
import modules.pandas_funcs as pf
import modules.phase_detection as pde
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


def stride_metrics(foot_x_i, foot_x_f, foot_y, *, fps=30):
    """
    Calculate gait metrics from a single stride.

    Parameters
    ----------
    foot_x_i : namedtuple
        Single result from pandas DataFrame.itertuples() method.
        Includes fields of 'frame', 'position', 'stride', and 'side'.
        Represents the initial foot on side x.
    foot_x_f : namedtuple
        Represents the final foot on side x.
    foot_y : namedtuple
        Represents the foot on side y.
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

    stride_vector = pos_x_f - pos_x_i

    stride_length = norm(stride_vector)
    stride_time = (foot_x_f.frame - foot_x_i.frame) / fps

    stride_velocity = stride_length / stride_time

    metrics = {
        'stride': foot_x_i.stride,
        'side': foot_x_i.side,
        'stride_length': stride_length,
        'stride_time': stride_time,
        'stride_velocity': stride_velocity,
        'step_length': norm(pos_x_f - pos_y_proj),
        'stride_width': norm(pos_y - pos_y_proj),
    }

    return metrics


def stance_metrics(is_stance_l, is_stance_r):
    """
    Calculate gait metrics involved with the stance to swing ratio.

    Parameters
    ----------
    is_stance_l : ndarray
        Vector of booleans.
        Element is True if corresponding left foot is in the stance phase.
    is_stance_r : bool
        Vector of booleans.
        Element is True if corresponding right foot is in the stance phase.

    Returns
    -------
    metrics : dict
        Dictionary with stance metrics, e.g. double stance percentage.

    """
    stance_vectors = [is_stance_l, is_stance_r, is_stance_l & is_stance_r]

    stance_l, stance_r, stance_double = [nf.ratio_nonzero(x) * 100
                                         for x in stance_vectors]

    df_stance = pd.DataFrame({
        'side': ['L', 'R'],
        'stance_percentage': [stance_l, stance_r],
        'stance_percentage_double': stance_double,
    })

    return df_stance


def foot_contacts_to_gait(df_contact):
    """
    Calculate gait metrics from all instances of the feet contacting the floor.

    Parameters
    ----------
    df_contact : DataFrame
        Each row represents an instance of a foot contacting the floor.
        Columns are 'frame', 'position', 'stride', 'side'.

    Returns
    -------
    df_gait : DataFrame
        Each row represents a set of gait metrics calculated from one stride.
        Columns include gait metric names, e.g., stride_velocity.
        Columns also include 'stride' and 'side'.

    """
    foot_tuples = df_contact.itertuples(index=False)

    def yield_metrics():
        """Inner function to yield metrics for each stride."""
        for foot_tuple in sw.generate_window(foot_tuples, n=3):

            foot_x_i, foot_y, foot_x_f = foot_tuple

            yield stride_metrics(foot_x_i, foot_x_f, foot_y)

    df_gait = pd.DataFrame(yield_metrics())

    return df_gait


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
    df_pass_metrics : DataFrame
        All metrics for a walking pass.
        Columns are metric names.

    """
    df_phase_l = pde.get_phase_dataframe(df_pass.L_FOOT, direction_pass)
    df_phase_r = pde.get_phase_dataframe(df_pass.R_FOOT, direction_pass)

    is_stance_l = df_phase_l.phase.values == 'stance'
    is_stance_r = df_phase_r.phase.values == 'stance'
    df_stance = stance_metrics(is_stance_l, is_stance_r)

    df_contact_l = pde.get_contacts(df_pass.L_FOOT, direction_pass)
    df_contact_r = pde.get_contacts(df_pass.R_FOOT, direction_pass)

    df_contact_l['side'] = 'L'
    df_contact_r['side'] = 'R'

    df_contact = pd.concat([df_contact_l, df_contact_r]).sort_values('frame')

    df_gait = foot_contacts_to_gait(df_contact)

    if not df_gait.empty:
        df_pass_metrics = pd.merge(df_gait, df_stance, left_on='side',
                                   right_on='side')

        return df_pass_metrics


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
    list_dfs = []
    for i, df_pass in enumerate(pass_dfs):

        _, direction_pass = direction_of_pass(df_pass)

        # Assign correct sides to feet
        df_pass = asi.assign_sides_pass(df_pass, direction_pass)

        # Ensure there are no missing frames in the walking pass
        df_pass = pf.make_index_consecutive(df_pass)
        df_pass = df_pass.applymap(
            lambda x: x if isinstance(x, np.ndarray)
            else np.full(direction_pass.size, np.nan))

        df_pass_metrics = walking_pass_metrics(df_pass, direction_pass)

        if df_pass_metrics is not None:
            # Add column to record the walking pass
            df_pass_metrics['pass'] = i

        list_dfs.append(df_pass_metrics)

    df_trial = pd.concat(list_dfs, sort=True).reset_index(drop=True)

    return df_trial
