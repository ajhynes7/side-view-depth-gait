"""Tests for linear algebra module."""

import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays

from numpy.linalg import norm

import modules.linear_algebra as lin


floats = st.floats(min_value=-1e6, max_value=1e6)
ints = st.integers(min_value=-1e6, max_value=1e6)

non_zero_vector = st.lists(ints, min_size=2, max_size=5).filter(lambda x:
                                                                any(x))

cross_vector = st.lists(ints, min_size=3, max_size=3).filter(lambda x: any(x))

array_3 = arrays('int', (3,), ints)


@given(non_zero_vector)
def test_unit(vector):
    """
    Tests for converting a vector to a unit vector.

    The unit vector must have a norm of one and the unit operation must be
    idempotent.

    """
    unit_vector = lin.unit(vector)

    assert np.allclose(norm(unit_vector), 1)

    # Unit function is idempotent
    assert np.allclose(unit_vector, lin.unit(unit_vector))


@given(non_zero_vector, non_zero_vector)
def test_perpendicular(u, v):
    """Two vectors must have an angle of 90 deg if they are perpendicular."""
    assume(len(u) == len(v))

    angle_90 = lin.angle_between(u, v, degrees=True) == 90

    assert lin.is_perpendicular(u, v) == angle_90


@given(cross_vector, cross_vector)
def test_parallel(u, v):
    """If two vectors are parallel, the angle between them must be 0 or 180."""
    angle_uv = lin.angle_between(u, v, degrees=True)

    if lin.is_parallel(u, v):
        angle_0 = np.isclose(angle_uv, 0, atol=1e-5)
        angle_180 = np.isclose(angle_uv, 180)

        assert (angle_0 or angle_180)


@given(cross_vector, cross_vector, cross_vector)
def test_collinear(point_a, point_b, point_c):

    dist_ab = norm(np.subtract(point_a, point_b))
    dist_bc = norm(np.subtract(point_b, point_c))
    dist_ac = norm(np.subtract(point_a, point_c))

    dists = [dist_ab, dist_bc, dist_ac]

    if lin.is_collinear(point_a, point_b, point_c):

        max_index = np.argmax(dists)

        max_dist = dists[max_index]
        non_max_dists = dists[:max_index] + dists[max_index+1:]

        assert np.isclose(max_dist, sum(non_max_dists))


def test_consecutive_dist():

    lengths = [*lin.consecutive_dist(points)]

    npt.assert_array_equal(np.round(lengths, 4), [2.2361, 9.0554, 8.1854])


def test_closest_point():

    target = [2, 3, 4]
    close_point, close_index = lin.closest_point(np.array(points), target)

    assert close_index == 3


@given(array_3, array_3, array_3)
def test_project_point_line(point_p, point_a, point_b):
    """Tests for projecting a point onto a line."""
    if norm(point_a - point_b) == 0:

        with pytest.raises(Exception):
            lin.project_point_line(point_p, point_a, point_b)

    else:

        point_proj = lin.project_point_line(point_p, point_a, point_b)

        vector_ab = np.subtract(point_a, point_b)
        vector_proj = np.subtract(point_p, point_proj)

        assert lin.is_collinear(point_a, point_b, point_proj, atol=1e-3)
        assert lin.is_perpendicular(vector_ab, vector_proj, atol=1e-3)

        # The order of the line points should not matter
        point_proj_2 = lin.project_point_line(point_p, point_b, point_a)
        assert(np.allclose(point_proj, point_proj_2))


@given(array_3, array_3, array_3)
def test_project_point_plane(point, point_plane, normal):
    """Tests for projecting a point onto a plane."""
    if norm(normal) == 0:

        with pytest.raises(Exception):
            lin.project_point_plane(point, point_plane, normal)

    else:

        point_proj = lin.project_point_plane(point, point_plane, normal)

        vector_proj_point = point - point_proj
        vector_proj_plane = point_plane - point_proj

        dist_proj_point = norm(vector_proj_point)
        dist_proj_plane = norm(vector_proj_plane)

        assert lin.is_parallel(normal, vector_proj_point, atol=1e-3)

        if not (np.isclose(dist_proj_point, 0) or
                np.isclose(dist_proj_plane, 0)):

            assert lin.is_perpendicular(vector_proj_plane, vector_proj_point,
                                        atol=0.1)


@pytest.mark.parametrize("test_input, expected", [
    (np.array([1, 1, 0]), 'straight'),
    (np.array([-1, 5, 0]), 'straight'),
    (np.array([0, 5, 1]), 'left'),
    (np.array([0, -5, -10]), 'right'),
    (np.array([4, 2, 1]), 'left'),
])
def test_target_side(test_input, expected):

    forward = np.array([1, 0, 0])
    up = np.array([0, 1, 0])

    assert lin.target_side(test_input, forward, up) == expected


# def test_best_fit_line_1():

#     points = np.random.rand(10, 3)
#     _, direction = lin.best_fit_line(points)

#     points_reversed = np.flip(points, axis=0)

#     _, direction_reversed = lin.best_fit_line(points_reversed)

#     # The two vectors should be parallel
#     cross_prod = np.cross(direction_reversed, direction)
#     npt.assert_array_almost_equal(cross_prod, np.array([0, 0, 0]))

#     npt.assert_allclose(norm(direction), 1)


# @pytest.mark.parametrize("points, centroid, direction", [
#     (np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), np.array([1, 0, 0]),
#      np.array([1, 0, 0])),
#     (np.array([[0, 0], [4, 0]]), np.array([2, 0]), np.array([1, 0])),
#     (np.array([[0, 0], [0, -10]]), np.array([0, -5]), np.array([0, -1])),
# ])
# def test_best_fit_line_2(points, centroid, direction):

#     centroid_calc, direction_calc = lin.best_fit_line(points)

#     npt.assert_allclose(centroid, np.round(centroid_calc, 2))
#     npt.assert_allclose(direction, direction_calc)


# @pytest.mark.parametrize("points, centroid, normal", [
#     (np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]]), np.array([0.33, 0, 0.33]),
#      np.array([0, -1, 0])),
#     (np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0]]), np.array([1, 1, 0]),
#      np.array([0, 0, 1])),
# ])
# def test_best_fit_plane(points, centroid, normal):

#     centroid_calc, normal_calc = lin.best_fit_plane(points)

#     npt.assert_allclose(centroid, np.round(centroid_calc, 2))
#     npt.assert_allclose(normal, normal_calc)


# @pytest.mark.parametrize("a, b, expected", [
#     (np.array([2, 0]), np.array([-2, 0]), np.pi),
#     (np.array([5, 5, 5]), np.array([1, 1, 1]), 0),
#     (np.array([1, 0]), np.array([1, 1]), np.pi / 4),
#     (np.array([1, 0]), np.array([-5, -5]), 3 * np.pi / 4),
# ])
# def test_angle_between(a, b, expected):

#     angle = lin.angle_between(a, b)

#     npt.assert_allclose(angle, expected)

points = [[1, 2, 3], [2, 2, 5], [-1, 10, 2], [2, 3, 5]]
