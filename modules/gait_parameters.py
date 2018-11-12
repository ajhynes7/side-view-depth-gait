"""
Module for calculating gait parameters from 3D body part positions.

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
    Columns include gait parameters, e.g. 'stride_length', and the side and
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
        namedtuple consisting of absolute step length, step length,
        stride length,  and stride width.

    Examples
    --------
    >>> pos_l_1 = np.array([764.253, 28.798])
    >>> pos_r_1 = np.array([696.834, 37.141])

    >>> pos_l_2 = np.array([637.172, 24.508])
    >>> pos_r_2 = np.array([579.102, 35.457])

    >>> pos_l_3 = np.array([518.030, 30.507])

    >>> np.round(spatial_parameters(pos_l_1, pos_r_1, pos_l_2), 1)
    array([ 61. ,  60.1, 127.2,  10.6])

    >>> np.round(spatial_parameters(pos_r_1, pos_l_2, pos_r_2), 1)
    array([ 59.1,  57.9, 117.7,  11.8])

    >>> np.round(spatial_parameters(pos_l_2, pos_r_2, pos_l_3), 1)
    array([ 61.3,  60.7, 119.3,   8. ])

    """
    stride_length = norm(pos_a_f - pos_a_i)

    pos_b_proj = lin.project_point_line(pos_b, pos_a_i, pos_a_f)

    absolute_step_length = norm(pos_a_f - pos_b)
    step_length = norm(pos_a_f - pos_b_proj)
    stride_width = norm(pos_b - pos_b_proj)

    Spatial = namedtuple('Spatial',
                         ['absolute_step_length', 'step_length',
                          'stride_length', 'stride_width'])

    return Spatial(absolute_step_length, step_length,
                   stride_length, stride_width)


def stride_parameters(foot_a_i, foot_b, foot_a_f, *, fps=30):
    """
    Calculate gait parameters from a single stride.

    Parameters
    ----------
    foot_a_i : namedtuple
        Represents the initial foot on side A.
        Includes fields of 'stride', 'frame', 'side', 'position'.
    foot_b : namedtuple
        Represents the foot on side B.
    foot_a_f : namedtuple
        Represents the final foot on side A.
    fps : int, optional
        Camera frame rate in frames per second (default 30).

    Returns
    -------
    dict
        Dictionary containing gait parameter names and values.

    Examples
    --------
    >>> Foot = namedtuple('Foot', ['stride', 'frame', 'side', 'position'])

    >>> foot_l_1 = Foot(1, 200, 'L', np.array([764.253, 28.798]))
    >>> foot_r_1 = Foot(1, 215, 'R', np.array([696.834, 37.141]))
    >>> foot_l_2 = Foot(2, 230, 'R', np.array([637.172, 24.508]))

    >>> params = stride_parameters(foot_l_1, foot_r_1, foot_l_2)

    >>> params['side']
    'L'
    >>> params['stride']
    1
    >>> np.round(params['step_length'], 1)
    60.1
    >>> np.round(params['stride_width'], 1)
    10.6
    >>> np.round(params['stride_velocity'], 1)
    127.2

    """
    pos_a_i, pos_a_f = foot_a_i.position, foot_a_f.position
    pos_b = foot_b.position

    spatial = spatial_parameters(pos_a_i, pos_b, pos_a_f)

    stride_time = (foot_a_f.frame - foot_a_i.frame) / fps
    stride_velocity = spatial.stride_length / stride_time

    stride_info = {'stride': foot_a_i.stride, 'side': foot_a_i.side}

    temporal_params = {'stride_time': stride_time,
                       'stride_velocity': stride_velocity}

    return {**stride_info, **temporal_params, **spatial._asdict()}


def stance_parameters(is_stance_l, is_stance_r):
    """
    Calculate gait parameters involved with the stance to swing ratio.

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
    parameters : dict
        Dictionary with stance parameters, e.g. double stance percentage.

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
    Calculate gait parameters from all instances of feet contacting the floor.

    Parameters
    ----------
    df_contact : DataFrame
        Each row represents an instance of a foot contacting the floor.
        Columns are 'frame', 'position', 'stride', 'side'.

    Returns
    -------
    df_gait : DataFrame
        Each row represents a set of gait parameters from one stride.
        Columns include gait parameter names, e.g., stride_velocity.
        Columns also include 'stride' and 'side'.

    """
    foot_tuples = df_contact.itertuples(index=False)

    def yield_parameters():
        """Inner function to yield parameters for each stride."""
        for foot_tuple in sw.generate_window(foot_tuples, n=3):

            yield stride_parameters(*foot_tuple)

    df_gait = pd.DataFrame(yield_parameters())

    return df_gait


def walking_pass_parameters(df_pass, direction_pass):
    """
    Calculate gait parameters from a single walking pass.

    Parameters
    ----------
    df_pass
        See module docstring.
    direction_pass : ndarray
        Direction of motion for the walking pass.

    Returns
    -------
    df_pass_parameters : DataFrame
        All parameters for a walking pass.
        Columns are parameters names.

    """
    df_phase_l = pde.get_phase_dataframe(df_pass.L_FOOT, direction_pass)
    df_phase_r = pde.get_phase_dataframe(df_pass.R_FOOT, direction_pass)

    is_stance_l = df_phase_l.phase.values == 'stance'
    is_stance_r = df_phase_r.phase.values == 'stance'
    df_stance = stance_parameters(is_stance_l, is_stance_r)

    df_contact_l = pde.get_contacts(df_pass.L_FOOT, direction_pass)
    df_contact_r = pde.get_contacts(df_pass.R_FOOT, direction_pass)

    df_contact_l['side'] = 'L'
    df_contact_r['side'] = 'R'

    df_contact = pd.concat([df_contact_l, df_contact_r]).sort_values('frame')

    df_gait = foot_contacts_to_gait(df_contact)

    if not df_gait.empty:
        df_pass_parameters = pd.merge(
            df_gait, df_stance, left_on='side', right_on='side')

        return df_pass_parameters


def combine_walking_passes(pass_dfs):
    """
    Combine gait parameters from all walking passes in a trial.

    Parameters
    ----------
    pass_dfs : list
        Each element is a df_pass (see module docstring).

    Returns
    -------
    df_final : DataFrame
        Each row represents a single stride.
        There can be multiple strides in a walking pass.
        Columns are gait parameters for left/right sides.

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

        df_pass_parameters = walking_pass_parameters(df_pass, direction_pass)

        if df_pass_parameters is not None:
            # Add column to record the walking pass
            df_pass_parameters['pass'] = i

        list_dfs.append(df_pass_parameters)

    df_trial = pd.concat(list_dfs, sort=True).reset_index(drop=True)

    return df_trial
