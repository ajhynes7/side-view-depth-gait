"""Unit tests for graph functions."""

import pytest

import modules.graphs as gr


@pytest.mark.parametrize("target_node, path", [
    (5, [1, 2, 4, 5]),
    (3, [1, 2, 3]),
    (0, [0]),
])
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
