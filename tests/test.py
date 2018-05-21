import unittest
import numpy as np
import numpy.testing as npt

from util import linear_algebra as lin
from util import graphs as gr


class TestLinearAlgebra(unittest.TestCase):

    def test_line_distance(self):
        """ Testing the linear algebra module """

        P = np.array([2, 3, 4])
        A = np.array([1, 5, 4])
        B = np.array([2, 10, 8])

        P_proj = lin.proj_point_line(P, A, B)
        d = lin.dist_point_line(P, A, B)

        npt.assert_allclose(d, 1.752549)
        npt.assert_allclose(d, np.linalg.norm(P_proj - P))

        low, high = -10, 10  # Limits for random numbers

        for _ in range(10):

            dim = np.random.randint(2, 3)  # Vector dimension

            # Generate random arrays
            P, A, B = [np.random.uniform(low, high, dim)
                       for _ in range(3)]

            P_proj = lin.proj_point_line(P, A, B)
            d = lin.dist_point_line(P, A, B)

            npt.assert_allclose(d, np.linalg.norm(P_proj - P))

    def test_plane_distance(self):
        """ Tests functions related to a point and plane """

        low, high = -10, 10

        for _ in range(10):

            dim = np.random.randint(1, 5)  # Vector dimension

            # Generate random arrays
            P, P_plane, normal = [np.random.uniform(low, high, dim)
                                  for _ in range(3)]

            P_proj = lin.proj_point_plane(P, P_plane, normal)
            d = lin.dist_point_plane(P, P_plane, normal)

            npt.assert_allclose(d, np.linalg.norm(P_proj - P))

    def test_unit(self):

        low, high = -10, 10

        for i in range(10):

            dim = np.random.randint(1, 5)  # Vector dimension

            v = np.random.uniform(low, high, dim)

            npt.assert_allclose(np.linalg.norm(lin.unit(v)), 1)

    def test_angle_direction(self):

        forward = np.array([0, 1, 0])
        up = np.array([0, 0, 1])

        x1 = lin.angle_direction(np.array([1, 1, 0]), forward, up)
        x2 = lin.angle_direction(np.array([-1, 5, 0]), forward, up)
        x3 = lin.angle_direction(np.array([0, 5, 0]), forward, up)
        x4 = lin.angle_direction(np.array([0, -5, -10]), forward, up)
        x5 = lin.angle_direction(np.array([4, 2, 1]), forward, up)

        self.assertEqual(x1, -1)
        self.assertEqual(x2, 1)
        self.assertEqual(x3, 0)
        self.assertEqual(x4, 0)
        self.assertEqual(x5, -1)


class TestGraphs(unittest.TestCase):

    def test_adj_list_conversion(self):

        G = {0: {1: 5},
             1: {2: -4},
             2: {1: 3}
             }

        M = np.array([[np.nan, 5, np.nan],
                      [np.nan, np.nan, -4],
                      [np.nan, 3, np.nan]])

        npt.assert_array_equal(gr.adj_list_to_matrix(G), M)

        G_converted = gr.adj_matrix_to_list(M)
        self.assertDictEqual(G, G_converted)

    def test_paths(self):

        G = {0: {1: 2, 2: 5},
             1: {3: 10, 2: 4},
             2: {3: 3, 4: 1},
             3: {4: 15, 5: 6},
             4: {5: 3},
             5: {}
             }

        source_nodes = {0, 1}
        V = [v for v in G]

        prev, dist = gr.dag_shortest_paths(G, V, source_nodes)

        self.assertEqual(gr.trace_path(prev, 5), [1, 2, 4, 5])
        self.assertEqual(gr.trace_path(prev, 3), [1, 2, 3])
        self.assertEqual(gr.trace_path(prev, 0), [0])

        prev, dist = gr.dag_shortest_paths(G, V, {0})
        shortest_path = gr.trace_path(prev, 5)
        path_weight = gr.weight_along_path(G, shortest_path)
        self.assertEqual(path_weight, 9)

        self.assertEqual(gr.weight_along_path(G, range(6)), 27)


if __name__ == '__main__':
    unittest.main()
