"""Module for calculating gait parameters from 3D body part positions."""

import numpy as np
import pandas as pd
from dpcontracts import require, ensure
from skspatial.objects import Vector, Line
from skspatial.transformation import transform_coordinates

import modules.numpy_funcs as nf
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
    dict
        Dictionary consisting of absolute step length, step length,
        stride length, and stride width.

    Examples
    --------
    >>> import numpy as np

    >>> pos_l_1 = [764.253, 28.798]
    >>> pos_r_1 = [696.834, 37.141]

    >>> pos_l_2 = [637.172, 24.508]
    >>> pos_r_2 = [579.102, 35.457]

    >>> pos_l_3 = [518.030, 30.507]

    >>> values = list(spatial_parameters(pos_l_1, pos_r_1, pos_l_2).values())
    >>> np.round(values, 1)
    array([ 61. ,  60.1, 127.2,  10.6])

    >>> values = list(spatial_parameters(pos_r_1, pos_l_2, pos_r_2).values())
    >>> np.round(values, 1)
    array([ 59.1,  57.9, 117.7,  11.8])

    >>> values = list(spatial_parameters(pos_l_2, pos_r_2, pos_l_3).values())
    >>> np.round(values, 1)
    array([ 61.3,  60.7, 119.3,   8. ])

    """
    line_a = Line.from_points(pos_a_i, pos_a_f)

    pos_b_proj = line_a.project_point(pos_b)

    stride_length = line_a.direction.norm()
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
        Includes fields of 'position', 'frame_i', 'frame_f'.
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
    >>> from collections import namedtuple
    >>> import numpy as np

    >>> names = ['position', 'frame_i', 'frame_f']
    >>> Foot = namedtuple('Foot', names)

    >>> foot_l_1 = Foot([764, 28], 180, 220)
    >>> foot_r_1 = Foot([696, 37], 200, 230)
    >>> foot_l_2 = Foot([637, 24], 230, 245)

    >>> params = stride_parameters(foot_l_1, foot_r_1, foot_l_2)

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


@require(
    "The stance DataFrame must include the required columns.",
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
    DataFrame
        Each row represents a set of gait parameters from one stride.
        Columns include gait parameter names, e.g., stride_velocity.

    """
    df_stance = df_stance.sort_values('frame_i')

    def yield_parameters():
        """Inner function to yield parameters for each stride."""
        tuples_stance = df_stance.itertuples(index=False)

        for stance_a_i, stance_b, stance_a_f in sw.generate_window(tuples_stance, n=3):

            if stance_a_i.side == stance_a_f.side != stance_b.side:
                # The sides must alternate between L and R (either L-R-L or R-L-R).

                dict_stride = stride_parameters(stance_a_i, stance_b, stance_a_f)
                dict_stride['side'] = stance_a_i.side

                yield dict_stride

    df_gait = pd.DataFrame(yield_parameters())

    if not df_gait.empty:
        df_gait = df_gait.set_index('side')

    return df_gait

@require(
    "The points must be 3D.",
    lambda args: all(x.shape[1] == 3 for x in [args.points_head, args.points_a, args.points_b]),
)
@ensure("The output must contain gait params.", lambda _, result: 'stride_length' in result.columns)
@ensure("The output must have the required MultiIndex.", lambda _, result: result.index.names == ['num_stride', 'side'])
def walking_pass_parameters(frames, points_head, points_a, points_b):
    """
    Calculate gait parameters from a single walking pass.

    Parameters
    ----------
    frames : ndarray
        (n,) array of frames for the pass.
    points_head : ndarray
        (n, 3) array of head points.
    points_a, points_b : ndarray
        (n, 3) array of foot points A and B.

    Returns
    -------
    DataFrame
        Gait parameters of the walking pass.
        The columns include parameters names.

    """
    basis, points_foot_inlier, frames_grouped_inlier = pde.compute_basis(frames, points_head, points_a, points_b)

    # Convert foot points into new coordinates defined by forward, up, and perpendicular directions.
    points_transformed = transform_coordinates(points_foot_inlier, basis.origin, (basis.up, basis.perp, basis.forward))
    coords_up, coords_perp, coords_forward = np.split(points_transformed, 3, 1)

    points_2d = np.column_stack((coords_perp, coords_forward))
    labels_stance = pde.label_stance_phases(frames_grouped_inlier, points_2d)

    return (
        pde.stance_props(frames_grouped_inlier, points_2d, labels_stance)
        .pipe(pde.assign_sides_pass)
        .pipe(stances_to_gait)
    )
