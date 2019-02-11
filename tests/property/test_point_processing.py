"""Tests for processing points."""

import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays
from scipy.spatial.distance import cdist

import modules.point_processing as pp


ints = st.integers(min_value=-1e6, max_value=1e6)


@st.composite
def array_like_2d(draw):
    """Generate a 2D array_like object that represents a set of points."""
    n_dim = draw(st.integers(min_value=1, max_value=5))
    n_points = draw(st.integers(min_value=1, max_value=50))

    points = draw(
        st.lists(
            st.lists(ints, min_size=n_dim, max_size=n_dim),
            min_size=n_points,
            max_size=n_points))

    return points


@given(array_like_2d())
def test_consecutive_dist(points):
    """Test finding the distance between consecutive pairs of points."""
    points = np.array(points)
    assume(points.shape[0] > 1 and points.shape[1] > 1)

    lengths = pp.consecutive_dist(points)

    assert lengths.ndim == 1
    assert lengths.size == len(points) - 1

    # The consecutive lengths can also be found from the distance matrix.
    # This is a more expensive function because the distance is
    # calculated between all pairs of points.
    dist_matrix = cdist(points, points)

    assert np.array_equal(lengths, np.diag(dist_matrix, k=1))


@given(st.data())
def test_correspond_points(data):
    """Test corresponding two current points to two previous points."""
    n_dim = data.draw(st.integers(min_value=1, max_value=5))

    point_pair = arrays(int, (2, n_dim), ints)
    points_prev, points_curr = data.draw(point_pair), data.draw(point_pair)

    points_ordered = pp.correspond_points(points_prev, points_curr)

    assert np.all(points_ordered.shape == points_curr.shape)

    assert np.array_equal(
        np.unique(points_ordered, axis=0), np.unique(points_curr, axis=0))


@given(st.data())
def test_track_two_objects(data):
    """Test tracking two objects by assigning positions to the objects."""
    n_dim = data.draw(st.integers(min_value=1, max_value=5))
    n_points = data.draw(st.integers(min_value=2, max_value=50))

    array_like = st.lists(
        st.lists(ints, min_size=n_dim, max_size=n_dim),
        min_size=n_points,
        max_size=n_points)

    points_1, points_2 = data.draw(array_like), data.draw(array_like)

    points_new_1, points_new_2 = pp.track_two_objects(points_1, points_2)

    sum_1 = sum(pp.consecutive_dist(points_1))
    sum_2 = sum(pp.consecutive_dist(points_2))

    sum_new_1 = sum(pp.consecutive_dist(points_new_1))
    sum_new_2 = sum(pp.consecutive_dist(points_new_2))

    assert sum_new_1 + sum_new_2 <= sum_1 + sum_2
