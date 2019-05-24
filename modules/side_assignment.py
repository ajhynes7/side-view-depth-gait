"""Module for assigning correct sides to the feet."""

import numpy as np
from dpcontracts import require
from skspatial.objects import Vector
from skspatial.transformation import transform_coordinates


@require("The input points must be 3D.", lambda args: all(x.shape[1] == 3 for x in args))
def median_basis(points_head, points_a, points_b):

    points_foot_mean = (points_a + points_b) / 2

    # Forward direction defined as median vector from one frame to next.
    differences = np.diff(points_foot_mean, axis=0)
    vector_forward = Vector(np.median(differences, axis=0)).unit()

    # Up direction defined as median of vectors from foot to head.
    vectors_up = points_head - points_foot_mean
    vector_up = Vector(np.median(vectors_up, axis=0)).unit()

    vector_perp = vector_up.cross(vector_forward)
    vectors_basis = (vector_forward, vector_up, vector_perp)

    origin = np.median(np.vstack((points_a, points_b)), axis=0)

    return origin, vectors_basis


@require(
    "The median point and perpendicular vector must be 3D.",
    lambda args: all(x.size == 3 for x in [args.point_med, args.vector_perp]),
)
def assign_sides_pass(df_stance, point_med, vector_perp):

    # Ensure that the stances are in temporal order.
    df_stance = df_stance.sort_values('frame_i')

    df_stance_even = df_stance[::2]
    df_stance_odd = df_stance[1::2]

    points_stance_even = np.stack(df_stance_even.position)
    points_stance_odd = np.stack(df_stance_odd.position)

    # Convert stance points to 1D coordinate system.
    # The values are coordinate along the perpendicular direction.
    values_side_even = transform_coordinates(points_stance_even, point_med, [vector_perp])
    values_side_odd = transform_coordinates(points_stance_odd, point_med, [vector_perp])

    value_side_even = np.median(values_side_even)
    value_side_odd = np.median(values_side_odd)

    # Initially assume even stances belong to the left foot.
    side_even, side_odd = 'L', 'R'

    if value_side_even > value_side_odd:
        side_even, side_odd = side_odd, side_even

    n_stances = df_stance.shape[0]

    sides = np.full(n_stances, None)
    sides[::2] = side_even
    sides[1::2] = side_odd

    nums_stance = np.zeros(n_stances).astype(int)
    nums_stance[::2] = np.arange(len(values_side_even))
    nums_stance[1::2] = np.arange(len(values_side_odd))

    df_stance['side'] = sides
    df_stance['num_stance'] = nums_stance

    return df_stance
