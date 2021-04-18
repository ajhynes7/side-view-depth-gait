"""Module for calculating gait parameters from 3D body part positions."""

from typing import Any, Dict, Iterator

import numpy as np
import pandas as pd
import xarray as xr
from dpcontracts import require, ensure
from skspatial.objects import Vector, Line

import modules.phase_detection as pde
import modules.side_assignment as sa
import modules.sliding_window as sw
from modules.phase_detection import Stance
from modules.typing import array_like


def spatial_parameters(
    point_a_i: array_like, point_b: array_like, point_a_f: array_like
) -> Dict[str, np.float64]:
    """
    Calculate spatial gait parameters for a stride.

    Positions are input in temporal order
    (foot A initial, foot B, foot A final).

    Parameters
    ----------
    point_a_i : array_like
        Initial position of foot A.
    point_b : array_like
        Position of foot B.
    point_a_f : array_like
        Final position of foot A.

    Returns
    -------
    dict
        Dictionary consisting of stride length, absolute step length, step length,
        and stride width.

    Examples
    --------
    >>> import numpy as np

    >>> point_l_1 = [764.253, 28.798]
    >>> point_r_1 = [696.834, 37.141]

    >>> point_l_2 = [637.172, 24.508]
    >>> point_r_2 = [579.102, 35.457]

    >>> point_l_3 = [518.030, 30.507]

    >>> values = list(spatial_parameters(point_l_1, point_r_1, point_l_2).values())
    >>> np.round(values, 1)
    array([127.2,  61. ,  60.1,  10.6])

    >>> values = list(spatial_parameters(point_r_1, point_l_2, point_r_2).values())
    >>> np.round(values, 1)
    array([117.7,  59.1,  57.9,  11.8])

    >>> values = list(spatial_parameters(point_l_2, point_r_2, point_l_3).values())
    >>> np.round(values, 1)
    array([119.3,  61.3,  60.7,   8. ])

    """
    line_a = Line.from_points(point_a_i, point_a_f)

    point_b_proj = line_a.project_point(point_b)

    stride_length = line_a.direction.norm()

    absolute_step_length = Vector.from_points(point_b, point_a_f).norm()
    step_length = Vector.from_points(point_b_proj, point_a_f).norm()

    stride_width = Vector.from_points(point_b_proj, point_b).norm()

    return {
        'stride_length': stride_length,
        'absolute_step_length': absolute_step_length,
        'step_length': step_length,
        'stride_width': stride_width,
    }


def stride_parameters(
    foot_a_i: Stance, foot_b: Stance, foot_a_f: Stance, *, fps: float = 30
) -> Dict[str, np.float64]:
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
    point_a_i, point_a_f = foot_a_i.position, foot_a_f.position
    point_b = foot_b.position

    dict_spatial = spatial_parameters(point_a_i, point_b, point_a_f)

    stride_time = (foot_a_f.frame_i - foot_a_i.frame_i) / fps
    stance_time = (foot_a_i.frame_f - foot_a_i.frame_i) / fps

    stride_velocity = dict_spatial['stride_length'] / stride_time

    stance_percentage = (stance_time / stride_time) * 100

    # Convert floats to np.float64 to satisfy mypy.
    dict_temporal = {
        'stride_time': np.float64(stride_time),
        'stride_velocity': stride_velocity,
        'stance_percentage': np.float64(stance_percentage),
    }

    return {**dict_spatial, **dict_temporal}


@require(
    "The stance DataFrame must include the required columns.",
    lambda args: set(args.df_stance.columns)
    == {'num_stride', 'position', 'frame_i', 'frame_f', 'side'},
)
@require(
    "The stances must be sorted by initial frame.",
    lambda args: args.df_stance.frame_i.is_monotonic,
)
def stances_to_gait(df_stance: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate gait parameters from all instances of feet contacting the floor.

    Parameters
    ----------
    df_stance : DataFrame
        Each row represents a stance phase.
        Columns are 'position', 'frame_i', 'frame_f', 'side'.

    Returns
    -------
    DataFrame
        Each row represents a set of gait parameters from one stride.
        Columns include gait parameter names, e.g., stride_velocity.

    """

    def yield_parameters() -> Iterator[Dict[str, Any]]:
        """Inner function to yield parameters for each stride."""
        tuples_stance = df_stance.itertuples(index=False)

        for stance_a_i, stance_b, stance_a_f in sw.generate_window(tuples_stance, n=3):

            sides = ''.join([stance_a_i.side, stance_b.side, stance_a_f.side])

            if sides in ('LRL', 'RLR'):
                # The sides must alternate between L and R (either L-R-L or R-L-R).

                dict_stride = stride_parameters(stance_a_i, stance_b, stance_a_f)

                dict_stride['side'] = stance_a_i.side
                dict_stride['num_stride'] = stance_a_i.num_stride

                yield dict_stride

    df_gait = pd.DataFrame(yield_parameters())

    if not df_gait.empty:
        df_gait = df_gait.set_index(['side', 'num_stride'])

    return df_gait


@require(
    "The layers must include head and two feet.",
    lambda args: set(args.points_stacked.layers.values)
    == {'points_a', 'points_b', 'points_head'},
)
@ensure(
    "The output must contain gait params.",
    lambda _, result: 'stride_length' in result.columns if not result.empty else True,
)
@ensure(
    "The output must have the required index.",
    lambda _, result: result.index.names == ['side', 'num_stride']
    if not result.empty
    else True,
)
def walking_pass_parameters(points_stacked: xr.DataArray) -> pd.DataFrame:
    """
    Calculate gait parameters from a single walking pass.

    Parameters
    ----------
    points_stacked : xarray.DataArray
        (N_frames, N_dims, N_layers) array of points.
        The layers are 'points_head', 'points_a', 'points_b'.

    Returns
    -------
    DataFrame
        Gait parameters of the walking pass.
        The columns include parameters names.

    """
    basis, points_grouped_inlier = sa.compute_basis(points_stacked)

    labels_grouped_l, labels_grouped_r = pde.label_stances(points_grouped_inlier, basis)
    df_stance = pde.get_stance_dataframe(
        points_grouped_inlier, labels_grouped_l, labels_grouped_r
    )

    return df_stance.pipe(stances_to_gait)
