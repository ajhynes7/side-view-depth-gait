import pytest
import numpy as np
import numpy.testing as npt

from util import graphs as gr


def test_adj_list_conversion():

    G = {0: {1: 5},
         1: {2: -4},
         2: {1: 3}
         }

    M = np.array([[np.nan, 5, np.nan],
                  [np.nan, np.nan, -4],
                  [np.nan, 3, np.nan]])

    npt.assert_array_equal(gr.adj_list_to_matrix(G), M)

    G_converted = gr.adj_matrix_to_list(M)
    assert G == G_converted


@pytest.mark.parametrize("test_input, expected", [
(5, [1, 2, 4, 5]),
(3, [1, 2, 3]),
(0, [0]),
])
def test_paths(test_input, expected):
    
    prev, dist = gr.dag_shortest_paths(G, V, source_nodes)

    assert gr.trace_path(prev, test_input) == expected


def test_path_weight():

    prev, dist = gr.dag_shortest_paths(G, V, source_nodes)

    prev, dist = gr.dag_shortest_paths(G, V, {0})
    shortest_path = gr.trace_path(prev, 5)

    assert gr.weight_along_path(G, shortest_path) == 9
    assert gr.weight_along_path(G, range(6)) == 27




G = {0: {1: 2, 2: 5},
     1: {3: 10, 2: 4},
     2: {3: 3, 4: 1},
     3: {4: 15, 5: 6},
     4: {5: 3},
     5: {}
     }

source_nodes = {0, 1}
V = [v for v in G]
