"""Unit tests for graph functions."""

import numpy as np
import pytest
from numpy import nan

import modules.graphs as gr


@pytest.fixture
def directed_acyclic_graph():
    """Fixture for a directed acyclic graph."""
    graph = {
        0: {1: 2, 2: 5},
        1: {3: 10, 2: 4},
        2: {3: 3, 4: 1},
        3: {4: 15, 5: 6},
        4: {5: 3},
        5: {},
    }

    order = list(graph)

    nodes_source = {0, 1}

    return graph, order, nodes_source


@pytest.fixture
def label_graph():
    """Fixture for converting labelled nodes into a graph."""
    node_labels = {
        0: 'head',
        1: 'hip',
        2: 'hip',
        3: 'thigh',
        4: 'thigh',
        5: 'thigh',
    }

    label_adj_list = {'head': {'hip': 60}, 'hip': {'thigh': 15}, 'thigh': {}}

    graph_expected = {
        0: {1: 60, 2: 60},
        1: {3: 15, 4: 15, 5: 15},
        2: {3: 15, 4: 15, 5: 15},
        3: {},
        4: {},
        5: {},
    }

    return node_labels, label_adj_list, graph_expected


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


@pytest.mark.parametrize(
    "node_target, path",
    [
        (0, [0]),
        (1, [1]),
        (2, [1, 2]),
        (3, [1, 2, 3]),
        (4, [1, 2, 4]),
        (5, [1, 2, 4, 5]),
    ],
)
def test_shortest_paths(directed_acyclic_graph, node_target, path):

    graph, order, nodes_source = directed_acyclic_graph

    prev, _ = gr.dag_shortest_paths(graph, order, nodes_source)
    assert gr.trace_path(prev, node_target) == path


def test_labelled_nodes_to_graph(label_graph):

    node_labels, label_adj_list, graph_expected = label_graph

    graph_from_labels = gr.labelled_nodes_to_graph(node_labels, label_adj_list)

    assert graph_from_labels == graph_expected
