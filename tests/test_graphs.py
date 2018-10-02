"""Tests for functions dealing with graphs."""

import numpy as np
import pytest

import hypothesis.strategies as st
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


@pytest.mark.parametrize("test_input, expected", [
    (5, [1, 2, 4, 5]),
    (3, [1, 2, 3]),
    (0, [0]),
])
def test_paths(test_input, expected):

    prev, _ = gr.dag_shortest_paths(G, V, source_nodes)
    assert gr.trace_path(prev, test_input) == expected


def test_path_weight():

    prev, _ = gr.dag_shortest_paths(G, V, {0})
    shortest_path = gr.trace_path(prev, 5)

    assert gr.weight_along_path(G, shortest_path) == 9
    assert gr.weight_along_path(G, range(6)) == 27


def test_min_shortest_path():

    node_labels = {0: 1, 1: 2, 2: 2, 3: 3, 4: 4, 5: 4}

    prev, dist = gr.dag_shortest_paths(G, V, source_nodes)

    min_path = gr.min_shortest_path(prev, dist, node_labels, 4)

    assert min_path == [1, 2, 4]


G = {
    0: {
        1: 2,
        2: 5
    },
    1: {
        3: 10,
        2: 4
    },
    2: {
        3: 3,
        4: 1
    },
    3: {
        4: 15,
        5: 6
    },
    4: {
        5: 3
    },
    5: {}
}

source_nodes = {0, 1}
V = [v for v in G]
