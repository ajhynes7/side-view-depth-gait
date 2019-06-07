"""Unit tests for finding a motion correspondence of multiple points."""

import numpy as np
from numpy.testing import assert_array_equal

import modules.motion_correspondence as mc


def test_correspondence():

    list_points = [
        [[1, 5], [1, 3], [1, 0]],
        [[2, 5], [2, 3], [2, 0]],
        [[3, 5], [3, 3], [3, 1]],
        [[4, 5], [4, 3], [4, 1]],
        [[5, 5], [5, 3], [5, 1]],
        [[6, 6], [6, 2], [6, 1]],
        [[7, 8], [7, 1], [7, 2]],
        [[8, 6], [8, 0], [8, 4]],
        [[9, 4], [9, 0], [9, 5]],
        [[10, 6], [10, 0], [10, 4]],
    ]

    assignment_expected = [
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 2, 1],
        [0, 2, 1],
        [0, 2, 1],
        [1, 2, 0],
    ]

    points_stacked = np.swapaxes(list_points, 1, 2)
    assignment = mc.correspond_motion(points_stacked, [0, 1, 2])

    assert_array_equal(assignment, assignment_expected)
