"""Tests for processing points."""

import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given
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
            max_size=n_points,
        )
    )

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
