"""Unit tests for graph functions."""

import numpy as np
import pytest
from numpy import nan

import modules.graphs as gr


@pytest.mark.parametrize(
    "adj_list, matrix_expected",
    [
        ({0: {1: 2}}, np.array([[nan, 2], [nan, nan]])),
        ({0: {1: 2}, 1: {0: -5, 1: 0}}, np.array([[nan, 2], [-5, 0]])),
        (
            {0: {1: 2}, 1: {2: -5}, 2: {}},
            np.array([[nan, 2, nan], [nan, nan, -5], [nan, nan, nan]]),
        ),
    ],
)
def test_adj_list_to_matrix(adj_list, matrix_expected):

    if len(adj_list) != len(matrix_expected):

        with pytest.raises(Exception):
            gr.adj_list_to_matrix(adj_list)

    else:

        adj_matrix = gr.adj_list_to_matrix(adj_list)

        assert np.allclose(adj_matrix, matrix_expected, equal_nan=True)

        assert gr.adj_matrix_to_list(adj_matrix) == adj_list

def test_paths(target_node, path):

    prev, _ = gr.dag_shortest_paths(G, V, source_nodes)
    assert gr.trace_path(prev, target_node) == path



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
