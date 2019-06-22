"""Module for assigning left/right sides to the feet."""

from collections import namedtuple

import numpy as np
import xarray as xr
from dpcontracts import require
from skimage.measure import LineModelND, ransac
from skspatial.objects import Vector
from statsmodels.robust import mad

import modules.numpy_funcs as nf


def fit_ransac(points):
    """
    Fit a line to the foot points with RANSAC.

    Parameters
    ----------
    points : (N, D) ndarray
        Input points.

    Returns
    -------
    model : LineModelND
        Linear model fitted with RANSAC.
        Params consist of the point and unit vector defining the line.
    is_inlier : (N,) ndarray
        Boolean mask indicating inlier points.

    """
    model, is_inlier = ransac(
        points, LineModelND, min_samples=int(0.5 * len(points)), residual_threshold=2.5 * mad(points[:, 2], c=1)
    )

    return model, is_inlier


@require(
    "The layers must include head and two feet.",
    lambda args: set(args.points_stacked.layers.values) == {'points_a', 'points_b', 'points_head'},
)
def compute_basis(points_stacked):
    """
    Return origin and basis vectors of new coordinate system found with RANSAC.

    Parameters
    ----------
    points_stacked : xarray.DataArray
        (N_frames, N_dims, N_layers) array of points.

    Returns
    -------
    basis : namedtuple
        Basis of new coordinate system (origin point and three unit vectors).
        Fields include 'origin', 'forward', 'up', 'perp'.
    points_grouped_inlier : xarray.DataArray
        (N_frames, N_dims) array.
        Grouped foot points that are marked inliers by RANSAC.

    """
    frames = points_stacked.coords['frames'].values

    points_head = points_stacked.sel(layers='points_head')
    points_a = points_stacked.sel(layers='points_a')
    points_b = points_stacked.sel(layers='points_b')

    points_foot_mean = (points_a + points_b) / 2

    vectors_up = points_head - points_foot_mean
    vector_up = Vector(np.median(vectors_up, axis=0)).unit()

    frames_grouped = np.repeat(frames, 2)
    points_grouped = nf.interweave_rows(points_a, points_b)

    model_ransac, is_inlier = fit_ransac(points_grouped)
    point_origin, vector_forward = model_ransac.params

    vector_perp = Vector(vector_up).cross(vector_forward)

    frames_grouped_inlier = frames_grouped[is_inlier]
    points_grouped_inlier = points_grouped[is_inlier]

    points_grouped_inlier = xr.DataArray(
        points_grouped_inlier, coords=(frames_grouped_inlier, ['x', 'y', 'z']), dims=('frames', 'cols')
    )

    Basis = namedtuple('Basis', 'origin, forward, up, perp')
    basis = Basis(point_origin, vector_forward, vector_up, vector_perp)

    return basis, points_grouped_inlier


def assign_sides_grouped(frames_grouped, values_side_grouped, labels_grouped):
    """
    Assign left/right sides to clusters representing stance phases.

    Parameters
    ----------
    frames_grouped : (N_grouped,) ndarray
        Frames corresponding to grouped foot points.
    values_side_grouped : (N_grouped,) ndarray
        Values related to the side (left/right) of each foot point.
    labels_grouped : (N_grouped,) ndarray
        Labels indicating detected clusters (stance phases of the feet).
        Non-cluster elements are marked with -1.

    Returns
    -------
    labels_grouped_l, labels_grouped_r : (N_grouped,) ndarray
        Arrays of labels for the left and right sides.

    """
    labels_unique = np.unique(labels_grouped[labels_grouped != -1])
    set_labels_r = set()

    for label in labels_unique:

        is_cluster = labels_grouped == label

        # Element is True for all cluster frames in the grouped array.
        is_frame_cluster = np.in1d(frames_grouped, frames_grouped[is_cluster])

        # Element is True if the corresponding foot point occurred on a frame in the cluster,
        # but is not a part of the cluster itself. This means it is a swing foot.
        is_foot_swing = is_frame_cluster & ~is_cluster

        value_side_stance = np.median(values_side_grouped[is_cluster])

        if is_foot_swing.sum() > 0:
            value_side_swing = np.median(values_side_grouped[is_foot_swing])
        else:
            # It's possible that there are no swing feet in the cluster.
            # In this case, the swing value is assumed to be zero.
            value_side_swing = 0

        if value_side_stance > value_side_swing:
            # The current label is on the right side.
            set_labels_r.add(label)

    set_labels_l = set(labels_unique) - set_labels_r

    is_label_grouped_l = np.in1d(labels_grouped, list(set_labels_l))
    is_label_grouped_r = np.in1d(labels_grouped, list(set_labels_r))

    labels_grouped_l = np.copy(labels_grouped)
    labels_grouped_r = np.copy(labels_grouped)

    labels_grouped_l[~is_label_grouped_l] = -1
    labels_grouped_r[~is_label_grouped_r] = -1

    return labels_grouped_l, labels_grouped_r
