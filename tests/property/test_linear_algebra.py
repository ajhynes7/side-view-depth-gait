"""Tests for linear algebra module."""

import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays
from numpy.linalg import norm

import modules.linear_algebra as lin


ints = st.integers(min_value=-1e6, max_value=1e6)

# Strategy for generating shapes of numpy arrays with dimension 2 or 3
shapes_2_3 = st.tuples(
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=2, max_value=3),
)

array_like_nonzero = st.lists(ints, min_size=3, max_size=3).filter(
    lambda x: any(x)
)

point_3 = arrays('int', (3,), ints)

points_2_3 = arrays('int', shapes_2_3, ints)


@given(points_2_3)
def test_best_fit_line(points):
    """Tests for the line of best fit in multidimensional space."""
    points = np.unique(points, axis=0)

    # At least two unique points needed to define a line.
    assume(len(points) >= 2)

    centroid, direction = lin.best_fit_line(points)

    points_reversed = np.flipud(points)
    centroid_rev, direction_rev = lin.best_fit_line(points_reversed)

    assert np.allclose(centroid, centroid_rev)
    assert np.isclose(norm(direction), norm(direction_rev))

    assert np.isclose(norm(direction), 1)
