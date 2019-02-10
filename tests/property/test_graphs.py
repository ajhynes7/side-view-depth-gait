"""Tests for functions dealing with graphs."""

import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays

import modules.graphs as gr

floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=True)
ints = st.integers(min_value=-1e6, max_value=1e6)


@st.composite
def square_array(draw):
    """Generate a square numpy array."""
    n = draw(st.integers(min_value=1, max_value=50))

    return draw(arrays('float', (n, n), st.floats(allow_nan=False)))


@given(square_array())
def test_adj_list_conversion(adj_matrix):
    """Test converting between adjacency list and matrix."""
    adj_list = gr.adj_matrix_to_list(adj_matrix)

    adj_matrix_new = gr.adj_list_to_matrix(adj_list)

    assert np.array_equal(adj_matrix, adj_matrix_new)
