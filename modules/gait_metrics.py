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
from collections import namedtuple

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


def spatial_parameters(pos_a_i, pos_b, pos_a_f):
    """
    Calculate spatial gait parameters.

    Positions are input in temporal order
    (foot A initial, foot B, foot A final).

    Parameters
    ----------
    pos_a_i : ndarray
        Initial position of foot A.
    pos_b : ndarray
        Position of foot B.
    pos_a_f : ndarray
        Final position of foot A.

    Returns
    -------
    Spatial : namedtuple
        namedtuple including stride length, step length, and stride width.

    Examples
    --------
    >>> pos_l_1 = np.array([764.253, 28.798])
    >>> pos_r_1 = np.array([696.834, 37.141])

    >>> pos_l_2 = np.array([637.172, 24.508])
    >>> pos_r_2 = np.array([579.102, 35.457])

    >>> pos_l_3 = np.array([518.030, 30.507])

    >>> np.round(spatial_parameters(pos_l_1, pos_r_1, pos_l_2), 1)
    array([127.2,  60.1,  10.6])

    >>> np.round(spatial_parameters(pos_r_1, pos_l_2, pos_r_2), 1)
    array([117.7,  57.9,  11.8])

    >>> np.round(spatial_parameters(pos_l_2, pos_r_2, pos_l_3), 1)
    array([119.3,  60.7,   8. ])

    """
    stride_length = norm(pos_a_f - pos_a_i)

    pos_b_proj = lin.project_point_line(pos_b, pos_a_i, pos_a_f)
    step_length = norm(pos_a_f - pos_b_proj)
    stride_width = norm(pos_b - pos_b_proj)

    Spatial = namedtuple('Spatial',
                         ['stride_length', 'step_length', 'stride_width'])

    return Spatial(stride_length, step_length, stride_width)


def stride_parameters(foot_a_i, foot_b, foot_a_f, *, fps=30):
    """
    Calculate gait parameters from a single stride.

    Parameters
    ----------
    foot_a_i : namedtuple
        Single result from pandas DataFrame.itertuples() method.
        Includes fields of 'frame', 'position', 'stride', and 'side'.
        Represents the initial foot on side A.
    foot_b : namedtuple
        Represents the foot on side B.
    foot_a_f : namedtuple
        Represents the final foot on side A.
    fps : int, optional
        Camera frame rate in frames per second (default 30).

    Returns
    -------
    parameters : dict
        Dictionary containing gait parameter names and values.

    """
    pos_a_i, pos_a_f = foot_a_i.position, foot_a_f.position
    pos_b = foot_b.position

    spatial = spatial_parameters(pos_a_i, pos_b, pos_a_f)

    stride_time = (foot_a_f.frame - foot_a_i.frame) / fps
    stride_velocity = spatial.stride_length / stride_time

    parameters = {
        'stride': foot_a_i.stride,
        'side': foot_a_i.side,
        'stride_length': spatial.stride_length,
        'step_length': spatial.step_length,
        'stride_width': spatial.stride_width,
        'stride_time': stride_time,
        'stride_velocity': stride_velocity,
    }

    return parameters


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

    stance_l, stance_r, stance_double = [
        nf.ratio_nonzero(x) * 100 for x in stance_vectors
    ]

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

            yield stride_parameters(*foot_tuple)

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
        df_pass_metrics = pd.merge(
            df_gait, df_stance, left_on='side', right_on='side')

        head_points = np.stack(df_pass.HEAD)

        vectors = np.diff(head_points[::27], axis=0)
        norms = np.linalg.norm(vectors, axis=1)
        speed = np.nanmedian(norms)

        df_pass_metrics.stride_velocity = speed

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
