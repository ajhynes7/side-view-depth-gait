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
