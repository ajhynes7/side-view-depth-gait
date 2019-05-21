"""Module for calculating gait parameters from 3D body part positions."""

import pandas as pd
from dpcontracts import require, ensure
from skspatial.objects import Vector, Line

import modules.phase_detection as pde
import modules.sliding_window as sw


def spatial_parameters(pos_a_i, pos_b, pos_a_f):
    """
    Calculate spatial gait parameters for a stride.

    Positions are input in temporal order
    (foot A initial, foot B, foot A final).

    Parameters
    ----------
    pos_a_i : array_like
        Initial position of foot A.
    pos_b : array_like
        Position of foot B.
    pos_a_f : array_like
        Final position of foot A.

    Returns
    -------
    Spatial : namedtuple
        namedtuple consisting of absolute step length, step length,
        stride length,  and stride width.

    Examples
    --------
    >>> pos_l_1 = [764.253, 28.798]
    >>> pos_r_1 = [696.834, 37.141]

    >>> pos_l_2 = [637.172, 24.508]
    >>> pos_r_2 = [579.102, 35.457]

    >>> pos_l_3 = [518.030, 30.507]

    >>> np.round(spatial_parameters(pos_l_1, pos_r_1, pos_l_2), 1)
    array([ 61. ,  60.1, 127.2,  10.6])

    >>> np.round(spatial_parameters(pos_r_1, pos_l_2, pos_r_2), 1)
    array([ 59.1,  57.9, 117.7,  11.8])

    >>> np.round(spatial_parameters(pos_l_2, pos_r_2, pos_l_3), 1)
    array([ 61.3,  60.7, 119.3,   8. ])

    """
    vector_a = Vector.from_points(pos_a_i, pos_a_f)
    line_a = Line(point=pos_a_i, direction=vector_a)

    pos_b_proj = line_a.project_point(pos_b)

    stride_length = vector_a.norm()
    absolute_step_length = Vector.from_points(pos_b, pos_a_f).norm()
    step_length = Vector.from_points(pos_b_proj, pos_a_f).norm()

    stride_width = line_a.distance_point(pos_b)

    return {
        'absolute_step_length': absolute_step_length,
        'step_length': step_length,
        'stride_length': stride_length,
        'stride_width': stride_width,
    }


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

    dict_spatial = spatial_parameters(pos_a_i, pos_b, pos_a_f)

    stride_time = (foot_a_f.frame_i - foot_a_i.frame_i) / fps
    stance_time = (foot_a_i.frame_f - foot_a_i.frame_i) / fps

    stance_percentage = (stance_time / stride_time) * 100
    stride_velocity = dict_spatial['stride_length'] / stride_time

    dict_temporal = {
        'stride_time': stride_time,
        'stance_percentage': stance_percentage,
        'stride_velocity': stride_velocity,
    }

    return {**dict_spatial, **dict_temporal}


@require("All input arrays must have the same length.", lambda args: len(set(len(x) for x in args)) == 1)
@ensure(
    "The output must have the required columns.",
    lambda _, result: set(result.columns) == {'num_stance', 'position', 'frame_i', 'frame_f', 'side'},
)
def labels_to_stances(frames, points_l, points_r, labels_l, labels_r):
    """
    Return a DataFrame with median positions and initial/final frames
    for each stance phase in a walking pass.

    Parameters
    ----------
    frames : ndarray
        (n,) array of frames for the walking pass.
    points_l, points_r : ndarray
        (n, 2) arrays for left and right foot points.
    labels_l, labels_r : ndarray
        (n,) array of labels indicating stance phases.

    Returns
    -------
    df_stance : DataFrame
        Each row is a stance phase.
        The columns are 'num_stance', 'position', 'frame_i', 'frame_f', 'side'.

    """
    df_stance_l = pd.DataFrame(pde.stance_medians(frames, points_l, labels_l))
    df_stance_r = pd.DataFrame(pde.stance_medians(frames, points_r, labels_r))

    df_stance_l['side'] = 'L'
    df_stance_r['side'] = 'R'

    df_stance = pd.concat((df_stance_l, df_stance_r), ignore_index=True)

    # Sort stance phases by median frame.
    df_stance = df_stance.sort_values('frame_i').reset_index(drop=True)

    return df_stance


@require(
    "DataFrame must include the required columns.",
    lambda args: set(args.df_stance.columns) == {'num_stance', 'position', 'frame_i', 'frame_f', 'side'},
)
def stances_to_gait(df_stance):
    """
    Calculate gait parameters from all instances of feet contacting the floor.

    Parameters
    ----------
    df_stance : DataFrame
        Each row represents a stance phase.
        Columns are 'num_stance', 'position', 'frame_i', 'frame_f', 'side'.

    Returns
    -------
    df_gait : DataFrame
        Each row represents a set of gait parameters from one stride.
        Columns include gait parameter names, e.g., stride_velocity.

    """
    tuples_stance = df_stance.itertuples(index=False)

    def yield_parameters():
        """Inner function to yield parameters for each stride."""
        for stance_a_i, stance_b, stance_a_f in sw.generate_window(tuples_stance, n=3):

            dict_stride = stride_parameters(stance_a_i, stance_b, stance_a_f)

            dict_stride['side'] = stance_a_i.side
            dict_stride['num_stance'] = stance_a_i.num_stance

            yield dict_stride

    df_gait = pd.DataFrame(yield_parameters()).set_index(['num_stance', 'side'])

    return df_gait


@require("The frames must correspond to the points.", lambda args: args.frames.shape == (args.points_l.shape[0],))
@require("The points must 3D.", lambda args: all(x.shape[1] == 3 for x in [args.points_l, args.points_r]))
@require("The signals must 1D arrays.", lambda args: all(x.ndim == 1 for x in [args.signal_l, args.signal_r]))
@ensure("The output must contain gait params.", lambda _, result: 'stride_length' in result.columns)
@ensure(
    "The output must have the required MultiIndex.",
    lambda _, result: result.index.names == ['num_stance', 'side'],
)
def walking_pass_parameters(frames, points_l, points_r, signal_l, signal_r):
    """
    Calculate gait parameters from a single walking pass.

    Parameters
    ----------

    Returns
    -------
    df_gait_pass : DataFrame
        Gait parameters of the walking pass.
        The columns include parameters names.

    """
    labels_l = pde.detect_stance_phases(signal_l)
    labels_r = pde.detect_stance_phases(signal_r)

    # Filter out false stance phases
    labels_l, labels_r = pde.filter_stances(signal_l, signal_r, labels_l, labels_r)

    df_stance = labels_to_stances(frames, points_l, points_r, labels_l, labels_r)
    df_gait_pass = stances_to_gait(df_stance)

    return df_gait_pass
