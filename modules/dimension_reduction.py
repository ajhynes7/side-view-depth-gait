"""Module for converting 3D foot points to a 2D coordinate system."""

from collections import namedtuple

import numpy as np
from dpcontracts import require, ensure
from skimage.measure import LineModelND, ransac
from skspatial.objects import Vector
from statsmodels.robust import mad

import modules.numpy_funcs as nf


def fit_ransac(points):
    """Fit a line to 3D points with RANSAC."""

    model, is_inlier = ransac(
        points, LineModelND, min_samples=int(0.9 * len(points)), residual_threshold=3 * min(mad(points))
    )

    return model, is_inlier


def compute_basis(frames, points_head, points_a, points_b):
    """Return origin and basis vectors of new coordinate system found with RANSAC."""

    model_ransac, is_inlier = fit_ransac(points_head)

    frames_inlier = frames[is_inlier]
    points_head_inlier = points_head[is_inlier]
    points_a_inlier = points_a[is_inlier]
    points_b_inlier = points_b[is_inlier]

    points_mean = (points_a_inlier + points_b_inlier) / 2

    vector_up = Vector(np.median(points_head_inlier - points_mean, axis=0)).unit()

    vector_forward = Vector(model_ransac.params[1]).unit()
    vector_perp = vector_up.cross(vector_forward)

    point_origin = model_ransac.params[0]

    Basis = namedtuple('Basis', 'origin, forward, up, perp')
    basis = Basis(point_origin, vector_forward, vector_up, vector_perp)

    return basis, frames_inlier, points_a_inlier, points_b_inlier


@require("The input frames must be a 1D array.", lambda args: args.frames_grouped.ndim == 1)
@ensure(
    "The output frames must correspond to the points.""",
    lambda _, result: result[0].size == result[1].shape[0],
)
@ensure("The output points must be all finite.", lambda _, result: np.isfinite(result[1]).all())
def split_points(frames_grouped, points_grouped):
    """Separate points corresponding to the same frame."""

    frames_unique, counts_unique = np.unique(frames_grouped, return_counts=True)

    n_objects = counts_unique.max()
    is_max_count = counts_unique == n_objects

    frames_final = frames_unique[is_max_count]
    n_frames_final = frames_final.size

    n_dim = points_grouped.shape[1]

    points_stacked = np.full((n_frames_final, n_dim, n_objects), np.nan)

    for i, frame in enumerate(frames_final):

        is_frame = frames_grouped == frame

        points_stacked[i] = points_grouped[is_frame].T

    return frames_final, points_stacked
