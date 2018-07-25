"""Tests for linear algebra module."""

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays
from numpy.linalg import norm

import modules.linear_algebra as lin


floats = st.floats(min_value=-1e6, max_value=1e6)
ints = st.integers(min_value=-1e6, max_value=1e6)

n_points = st.one_of(st.integers(min_value=2, max_value=10))

non_zero_vector = st.lists(ints, min_size=2, max_size=5).filter(lambda x:
                                                                any(x))

shapes = st.tuples(st.integers(min_value=2, max_value=10),
                   st.integers(min_value=2, max_value=10))

# Strategy for generating shapes of numpy arrays with dimension 2 or 3
shapes_2_3 = st.tuples(st.integers(min_value=2, max_value=10),
                       st.integers(min_value=2, max_value=3))

cross_vector = st.lists(ints, min_size=3, max_size=3).filter(lambda x: any(x))

point_3 = arrays('int', (3,), ints)

points = arrays('int', shapes, ints)
points_2_3 = arrays('int', shapes_2_3, ints)


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
    """Test function that checks for collinearity."""
    dist_ab = norm(np.subtract(point_a, point_b))
    dist_bc = norm(np.subtract(point_b, point_c))
    dist_ac = norm(np.subtract(point_a, point_c))

    dists = [dist_ab, dist_bc, dist_ac]

    if lin.is_collinear(point_a, point_b, point_c):

        max_index = np.argmax(dists)

        max_dist = dists[max_index]
        non_max_dists = dists[:max_index] + dists[max_index+1:]

        assert np.isclose(max_dist, sum(non_max_dists))


@given(point_3, point_3, point_3)
def test_project_point_line(point_p, point_a, point_b):
    """Tests for projecting a point onto a line."""
    if norm(point_a - point_b) == 0:

        with pytest.raises(Exception):
            lin.project_point_line(point_p, point_a, point_b)

    else:

        point_proj = lin.project_point_line(point_p, point_a, point_b)

        vector_ab = np.subtract(point_a, point_b)
        vector_proj = np.subtract(point_p, point_proj)

        assert lin.is_collinear(point_a, point_b, point_proj, atol=0.1)
        assert lin.is_perpendicular(vector_ab, vector_proj, atol=0.1)

        # The order of the line points should not matter
        point_proj_2 = lin.project_point_line(point_p, point_b, point_a)
        assert(np.allclose(point_proj, point_proj_2))


@given(point_3, point_3, point_3)
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


@given(points_2_3, st.sampled_from([lin.best_fit_line, lin.best_fit_plane]))
def test_best_fit(points, best_fit_func):
    """Tests for the line of best fit in multidimensional space."""
    n_unique = len(np.unique(points, axis=0))
    assume(n_unique >= 3)

    centroid, direction = best_fit_func(points)

    points_reversed = np.flip(points, axis=0)
    centroid_rev, direction_rev = best_fit_func(points_reversed)

    assert np.allclose(centroid, centroid_rev)
    assert np.isclose(norm(direction), 1)

    assert lin.is_parallel(direction, direction_rev)


"""Parameterized tests"""


@pytest.mark.parametrize("test_input, expected", [
    (np.array([1, 1, 0]), 'straight'),
    (np.array([-1, 5, 0]), 'straight'),
    (np.array([0, 5, 1]), 'left'),
    (np.array([0, -5, -10]), 'right'),
    (np.array([4, 2, 1]), 'left'),
])
def test_target_side(test_input, expected):
    """Test specific examples of determining the side of a target."""
    forward = np.array([1, 0, 0])
    up = np.array([0, 1, 0])

    assert lin.target_side(test_input, forward, up) == expected


@pytest.mark.parametrize("points, centroid, direction", [
    (np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), np.array([1, 0, 0]),
     np.array([1, 0, 0])),
    (np.array([[0, 0], [4, 0]]), np.array([2, 0]), np.array([1, 0])),
    (np.array([[0, 0], [0, -10]]), np.array([0, -5]), np.array([0, -1])),
])
def test_best_fit_line_examples(points, centroid, direction):
    """Test specific examples of best fit line."""
    centroid_calc, direction_calc = lin.best_fit_line(points)

    assert np.allclose(centroid, np.round(centroid_calc, 2))
    assert np.allclose(direction, direction_calc)


@pytest.mark.parametrize("points, centroid, normal", [
    (np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]]), np.array([0.33, 0, 0.33]),
     np.array([0, -1, 0])),
    (np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0]]), np.array([1, 1, 0]),
     np.array([0, 0, 1])),
])
def test_best_fit_plane_examples(points, centroid, normal):
    """Test specific examples of best fit plane."""
    centroid_calc, normal_calc = lin.best_fit_plane(points)

    assert np.allclose(centroid, np.round(centroid_calc, 2))
    assert np.allclose(normal, normal_calc)


@pytest.mark.parametrize("a, b, expected", [
    (np.array([2, 0]), np.array([-2, 0]), np.pi),
    (np.array([5, 5, 5]), np.array([1, 1, 1]), 0),
    (np.array([1, 0]), np.array([1, 1]), np.pi / 4),
    (np.array([1, 0]), np.array([-5, -5]), 3 * np.pi / 4),
])
def test_angle_between_examples(a, b, expected):
    """Test specific examples of the angle between two vectors."""
    angle = lin.angle_between(a, b)

    assert np.allclose(angle, expected)
