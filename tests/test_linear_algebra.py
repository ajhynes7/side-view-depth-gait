"""Tests for linear algebra module."""

import numpy as np
from numpy.linalg import norm

import hypothesis.strategies as st
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays

import pytest

import modules.linear_algebra as lin

floats = st.floats(min_value=-1e6, max_value=1e6)
ints = st.integers(min_value=-1e6, max_value=1e6)
ints_nonzero = ints.filter(lambda x: x != 0)

n_points = st.one_of(st.integers(min_value=2, max_value=10))

non_zero_vector = st.lists(
    ints, min_size=2, max_size=5).filter(lambda x: any(x))

shapes = st.tuples(
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=2, max_value=10))

# Strategy for generating shapes of numpy arrays with dimension 2 or 3
shapes_2_3 = st.tuples(
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=2, max_value=3))

array_like_nonzero = st.lists(
    ints, min_size=3, max_size=3).filter(lambda x: any(x))

point_3 = arrays('int', (3, ), ints)

points_2_3 = arrays('int', shapes_2_3, ints)


def is_perpendicular(u, v, **kwargs):
    """
    Check if two vectors are perpendicular.

    The vectors are perpendicular if their dot product is zero.

    Parameters
    ----------
    u, v : array_like
        Input vectors
    kwargs : dict, optional
        Additional keywords passed to `np.isclose`.

    Returns
    -------
    bool
        True if vectors are perpendicular.

    Examples
    --------
    >>> is_perpendicular([0, 1], [1, 0])
    True

    >>> is_perpendicular([-1, 5], [3, 4])
    False

    >>> is_perpendicular([2, 0, 0], [0, 0, 2])
    True

    The zero vector is perpendicular to all vectors.

    >>> is_perpendicular([0, 0, 0], [1, 2, 3])
    True

    """
    return np.isclose(np.dot(u, v), 0, **kwargs)


def is_parallel(u, v, **kwargs):
    """
    Check if two vectors are parallel.

    Parameters
    ----------
    u, v : array_like
        Input vectors
    kwargs : dict, optional
        Additional keywords passed to `np.allclose`.

    Returns
    -------
    bool
        True if vectors are parallel.

    Examples
    --------
    >>> is_parallel([0, 1], [1, 0])
    False

    >>> is_parallel([-1, 5], [2, -10])
    True

    >>> is_parallel([1, 2, 3], [3, 6, 9])
    True

    """
    return np.allclose(np.cross(u, v), 0, **kwargs)


def is_collinear(point_a, point_b, point_c, **kwargs):
    """
    Check if three points are collinear.

    Points A, B, C are collinear if AB is parallel to AC.

    Parameters
    ----------
    point_a, point_b, point_c : ndarray
        Input points.
    kwargs : dict, optional
        Additional keywords passed to `np.allclose`.

    Returns
    -------
    bool
        True if points are collinear.

    Examples
    --------
    >>> is_collinear([0, 1], [1, 0], [1, 2])
    False

    >>> is_collinear([1, 1], [2, 2], [5, 5])
    True

    """
    vector_ab = np.subtract(point_a, point_b)
    vector_ac = np.subtract(point_a, point_c)

    return is_parallel(vector_ab, vector_ac, **kwargs)


def angle_between(u, v, degrees=False):
    """
    Compute the angle between vectors u and v.

    Parameters
    ----------
    u, v : array_like
        Input vectors

    degrees : bool, optional
        Set to true for angle in degrees rather than radians.

    Returns
    -------
    theta : float
        Angle between vectors.

    Examples
    --------
    >>> angle_between([1, 0], [1, 0])
    0.0

    >>> u, v = [1, 0], [1, 1]
    >>> round(angle_between(u, v, degrees=True))
    45.0

    >>> u, v = [1, 0], [-2, 0]
    >>> round(angle_between(u, v, degrees=True))
    180.0

    >>> u, v = [1, 1, 1], [1, 1, 1]
    >>> angle_between(u, v)
    0.0

    """
    cos_theta = np.dot(lin.unit(u), lin.unit(v))

    # The allowed domain for arccos is [-1, 1]
    if cos_theta > 1:
        cos_theta = 1
    elif cos_theta < -1:
        cos_theta = -1

    theta = np.arccos(cos_theta)

    if degrees:
        theta = np.rad2deg(theta)

    return theta


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

    angle_90 = angle_between(u, v, degrees=True) == 90

    assert is_perpendicular(u, v) == angle_90


@given(array_like_nonzero, array_like_nonzero)
def test_parallel(u, v):
    """If two vectors are parallel, the angle between them must be 0 or 180."""
    angle_uv = angle_between(u, v, degrees=True)

    if is_parallel(u, v):
        angle_0 = np.isclose(angle_uv, 0, atol=1e-5)
        angle_180 = np.isclose(angle_uv, 180)

        assert (angle_0 or angle_180)


@given(array_like_nonzero, array_like_nonzero, array_like_nonzero)
def test_collinear(point_a, point_b, point_c):
    """Test function that checks for collinearity."""
    dist_ab = norm(np.subtract(point_a, point_b))
    dist_bc = norm(np.subtract(point_b, point_c))
    dist_ac = norm(np.subtract(point_a, point_c))

    dists = [dist_ab, dist_bc, dist_ac]

    if is_collinear(point_a, point_b, point_c):

        max_index = np.argmax(dists)

        max_dist = dists[max_index]
        non_max_dists = dists[:max_index] + dists[max_index + 1:]

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

        assert is_collinear(point_a, point_b, point_proj, atol=0.1)
        assert is_perpendicular(vector_ab, vector_proj, atol=0.1)

        # The order of the line points should not matter
        point_proj_2 = lin.project_point_line(point_p, point_b, point_a)
        assert np.allclose(point_proj, point_proj_2)


@given(points_2_3)
def test_best_fit_line(points):
    """Tests for the line of best fit in multidimensional space."""
    points = np.unique(points, axis=0)

    # At least two unique points needed to define a line.
    assume(len(points) >= 2)

    centroid, direction = lin.best_fit_line(points)

    points_reversed = np.flip(points, axis=0)
    centroid_rev, direction_rev = lin.best_fit_line(points_reversed)

    assert np.allclose(centroid, centroid_rev)
    assert np.isclose(norm(direction), 1)

    assert is_parallel(direction, direction_rev)


@given(array_like_nonzero, array_like_nonzero, array_like_nonzero,
       ints_nonzero)
def test_target_side_value(forward, up, target, c):
    """Test evaluating the side (left/right) of a target."""
    assume(not is_parallel(forward, up))

    value = lin.target_side_value(forward, up, target)
    value_scaled = lin.target_side_value(forward, up, c * np.array(target))

    if value != 0:
        # The target is to the left or right of forward

        if abs(c) > 1:
            assert abs(value_scaled) > abs(value)
        elif abs(c) < 1:
            assert abs(value_scaled) < abs(value)

    scaled_forward = lin.target_side_value(c * np.array(forward), up, target)
    scaled_up = lin.target_side_value(forward, c * np.array(up), target)

    # Scaling the forward or up vectors does not change the
    # magnitude of the result.
    if c > 0:
        assert np.isclose(value, scaled_forward)
        assert np.isclose(value, scaled_up)
    else:
        assert np.isclose(value, -scaled_forward)
        assert np.isclose(value, -scaled_up)


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


@pytest.mark.parametrize("a, b, expected", [
    (np.array([2, 0]), np.array([-2, 0]), np.pi),
    (np.array([5, 5, 5]), np.array([1, 1, 1]), 0),
    (np.array([1, 0]), np.array([1, 1]), np.pi / 4),
    (np.array([1, 0]), np.array([-5, -5]), 3 * np.pi / 4),
])
def test_angle_between_examples(a, b, expected):
    """Test specific examples of the angle between two vectors."""
    angle = angle_between(a, b)

    assert np.allclose(angle, expected)
