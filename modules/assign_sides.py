"""Module for assigning correct sides to the feet."""

import numpy as np
from dpcontracts import require, ensure
from skspatial.objects import Vector
from skspatial.transformation import transform_coordinates

import modules.numpy_funcs as nf
import modules.point_processing as pp
import modules.signals as sig
import modules.sliding_window as sw


@require("The input points must be 3D.", lambda args: all(x.shape[1] == 3 for x in args))
@ensure("The output points must be 2D.", lambda _, results: all(x.shape[1] == 2 for x in results))
def convert_to_2d(points_head, points_a, points_b):

    points_foot = np.vstack((points_a, points_b))

    point_foot_med = np.median(points_foot, axis=0)
    point_head_med = np.median(points_head, axis=0)

    # Up direction defined as vector from median foot point to
    # median head point
    vector_up = Vector.from_points(point_foot_med, point_head_med).unit()

    # Forward direction defined as median vector from one frame to next
    points_foot_mean = (points_a + points_b) + 2
    differences = np.diff(points_foot_mean, axis=0)
    vector_forward = Vector(np.median(differences, axis=0)).unit()

    vector_perp = vector_up.cross(vector_forward)
    vectors_basis = (vector_perp, vector_forward)

    # Represent the 3D foot points in a new 2D coordinate system
    points_2d_a = transform_coordinates(points_a, point_foot_med, vectors_basis)
    points_2d_b = transform_coordinates(points_b, point_foot_med, vectors_basis)

    return points_2d_a, points_2d_b


@require("The points must be 2D", lambda args: all(x.shape[1] == 2 for x in (args.points_2d_a, args.points_2d_b)))
@ensure(
    "The outputs must have the same size as the inputs.",
    lambda args, results: args.points_2d_a.shape == results[0].shape == results[1].shape,
)
def assign_sides_pass(frames, points_2d_a, points_2d_b):

    distances_feet = np.linalg.norm(points_2d_a - points_2d_b, axis=1)

    # Find local minima in the foot distances.
    signal = 1 - sig.nan_normalize(distances_feet)
    rms = sig.root_mean_square(signal)
    peak_frames, _ = sw.detect_peaks(frames, signal, window_length=3, min_height=rms)

    # Split the pass into portions between local minima.
    labels = np.array(nf.label_by_split(frames, peak_frames))
    labels_unique = np.unique(labels)

    points_2d_l = np.zeros_like(points_2d_a)
    points_2d_r = np.zeros_like(points_2d_b)

    for label in labels_unique:

        is_portion = labels == label

        points_portion_a = points_2d_a[is_portion]
        points_portion_b = points_2d_b[is_portion]

        # Find a motion correspondence so the foot sides do not switch abruptly.
        points_tracked_a, points_tracked_b = pp.track_two_objects(points_portion_a, points_portion_b)

        value_side_a = points_tracked_a[:, 0].sum()
        value_side_b = points_tracked_b[:, 0].sum()

        # Assume that points A are left foot points, and vice versa.
        points_tracked_l = points_tracked_a
        points_tracked_r = points_tracked_b

        if value_side_a > value_side_b:
            # Swap left and right foot points.
            points_tracked_l, points_tracked_r = points_tracked_r, points_tracked_l

        points_2d_l[is_portion] = points_tracked_l
        points_2d_r[is_portion] = points_tracked_r

    return points_2d_l, points_2d_r
