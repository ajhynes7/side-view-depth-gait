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

import modules.linear_algebra as lin
import modules.pandas_funcs as pf
import modules.phase_detection as pde
import modules.sliding_window as sw


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

    Spatial = namedtuple('Spatial', [
        'absolute_step_length', 'step_length', 'stride_length', 'stride_width'
    ])

    return Spatial(absolute_step_length, step_length, stride_length,
                   stride_width)


def stride_parameters(foot_a_i, foot_b, foot_a_f, *, fps=30):
    """
    Calculate gait parameters from a single stride.

    Parameters
    ----------
    foot_a_i : namedtuple
        Represents the initial foot on side A.
        Includes fields of 'stride', 'side', 'position', 'first_contact',
        'last_contact'.
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
    >>> names = ['stride', 'side', 'position', 'first_contact', 'last_contact']
    >>> Foot = namedtuple('Foot', names)

    >>> foot_l_1 = Foot(1, 'L', np.array([764, 28]), 180, 220)
    >>> foot_r_1 = Foot(1, 'R', np.array([696, 37]), 200, 230)
    >>> foot_l_2 = Foot(2, 'R', np.array([637, 24]), 230, 245)

    >>> params = stride_parameters(foot_l_1, foot_r_1, foot_l_2)

    >>> params['side']
    'L'
    >>> params['stride']
    1
    >>> np.round(params['step_length'], 1)
    59.4
    >>> np.round(params['stride_width'], 1)
    11.1
    >>> np.round(params['stride_velocity'], 1)
    76.2
    >>> params['stance_percentage']
    80.0

    """
    pos_a_i, pos_a_f = foot_a_i.position, foot_a_f.position
    pos_b = foot_b.position

    spatial = spatial_parameters(pos_a_i, pos_b, pos_a_f)

    stride_time = (foot_a_f.first_contact - foot_a_i.first_contact) / fps
    stance_time = (foot_a_i.last_contact - foot_a_i.first_contact) / fps

    stance_percentage = (stance_time / stride_time) * 100
    stride_velocity = spatial.stride_length / stride_time

    stride_info = {'stride': foot_a_i.stride, 'side': foot_a_i.side}

    temporal_params = {
        'stride_time': stride_time,
        'stance_percentage': stance_percentage,
        'stride_velocity': stride_velocity
    }

    return {**stride_info, **temporal_params, **spatial._asdict()}


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
    df_contact_l = pde.get_contacts(df_pass.L_FOOT, direction_pass)
    df_contact_r = pde.get_contacts(df_pass.R_FOOT, direction_pass)

    df_contact_l['side'] = 'L'
    df_contact_r['side'] = 'R'

    # Combine left and right contact instances and sort by frame
    df_contact = pd.concat([df_contact_l, df_contact_r]).sort_values('frame')

    df_pass_parameters = foot_contacts_to_gait(df_contact)

    return df_pass_parameters


def combine_walking_passes(df_assigned, direction_series):
    """
    Combine gait parameters from all walking passes in a trial.

    Parameters
    ----------
    df_assigned : DataFrame
        Each row contains head and foot positions
        after assigning L/R sides to the feet.
        MultiIndex of form (pass, frame)

    direction_series : Series
        Row i is the direction vector of walking pass i.

    Returns
    -------
    df_trial : DataFrame
        Each row represents a single stride.
        There can be multiple strides in a walking pass.
        Columns are gait parameters for left/right sides.

    """
    # List of DataFrames with gait parameters
    param_dfs = []

    n_passes = direction_series.size

    for pass_number in range(n_passes):

        df_assigned_pass = df_assigned.loc[pass_number]
        direction_pass = direction_series.loc[pass_number]

        df_pass = df_assigned_pass

        # Ensure there are no missing frames in the walking pass
        df_pass = pf.make_index_consecutive(df_pass)
        df_pass = df_pass.applymap(
            lambda x: x if isinstance(x, np.ndarray)
            else np.full(direction_pass.size, np.nan))

        df_pass_parameters = walking_pass_parameters(df_pass, direction_pass)

        if df_pass_parameters is not None:
            # Add column to record the walking pass
            df_pass_parameters['pass'] = pass_number

        param_dfs.append(df_pass_parameters)

    # Gait parameters of full walking trial
    df_trial = pd.concat(param_dfs, sort=True).reset_index(drop=True)

    return df_trial
