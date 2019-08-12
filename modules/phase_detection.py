"""Module for detecting the phases of a foot during a walking pass."""

from collections import namedtuple
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from numpy import ndarray
from skspatial.transformation import transform_coordinates

import modules.cluster as cl
import modules.side_assignment as sa


def stance_props(points_foot: xr.DataArray, labels_stance: ndarray) -> pd.DataFrame:
    """Return properties of each stance phase from one foot in a walking pass."""

    frames = points_foot.coords['frames'].values

    labels_unique = np.unique(labels_stance[labels_stance != -1])

    Stance = namedtuple('Stance', ['frame_i', 'frame_f', 'position'])

    def yield_props():

        for label in labels_unique:

            is_cluster = labels_stance == label

            points_foot_cluster = points_foot[is_cluster]
            point_foot_med = np.median(points_foot_cluster, axis=0)

            frames_cluster = frames[is_cluster]
            frame_i = frames_cluster.min()
            frame_f = frames_cluster.max()

            yield Stance(frame_i, frame_f, point_foot_med)

    return pd.DataFrame(yield_props())


def label_stances(points_foot_grouped: xr.DataArray, basis: sa.Basis) -> Tuple[ndarray, ndarray]:
    """Label all stance phases in a walking pass."""

    frames_grouped = points_foot_grouped.coords['frames'].values

    signal_grouped = transform_coordinates(points_foot_grouped, basis.origin, [basis.forward])
    values_side_grouped = transform_coordinates(points_foot_grouped, basis.origin, [basis.perp])

    labels_grouped = cl.dbscan_st(signal_grouped, times=frames_grouped, eps_spatial=5, eps_temporal=10, min_pts=7)
    labels_grouped_l, labels_grouped_r = sa.assign_sides_grouped(frames_grouped, values_side_grouped, labels_grouped)

    return labels_grouped_l, labels_grouped_r


def get_stance_dataframe(points_foot_grouped: xr.DataArray, labels_grouped_l: ndarray, labels_grouped_r: ndarray) -> pd.DataFrame:
    """Return DataFrame where each row is a stance phase."""

    df_stance_l = stance_props(points_foot_grouped, labels_grouped_l).assign(side='L')
    df_stance_r = stance_props(points_foot_grouped, labels_grouped_r).assign(side='R')

    return (
        pd.concat((df_stance_l, df_stance_r), sort=False).rename_axis('num_stride').sort_values('frame_i').reset_index()
    )
