import unittest
import numpy as np
import numpy.testing as npt

import sys
sys.path.insert(0, '..//Modules/')
import linear_algebra as lin

class TestCalc(unittest.TestCase):

    def test_line_distance(self):
        """ Testing the linear algebra module """

        P = np.array([2, 3, 4])
        A = np.array([1, 5, 4])
        B = np.array([2, 10, 8])

        P_proj = lin.proj_point_line(P, A, B)
        d = lin.dist_point_line(P, A, B)

        npt.assert_allclose(d, 1.752549)
        npt.assert_allclose(d, np.linalg.norm(P_proj - P))

        low, high= -10, 10  # Limits for random numbers 

        for _ in range(10):

            dim = np.random.randint(2, 3)  # Vector dimension

            # Generate random arrays
            P, A, B = [np.random.uniform(low, high, dim)
                                  for _ in range(3)]

            P_proj = lin.proj_point_line(P, A, B)
            d = lin.dist_point_line(P, A, B)

            npt.assert_allclose(d, np.linalg.norm(P_proj - P))

    def test_plane_distance(self):
        """ Tests functiions related to a point and plane """

        low, high= -10, 10

        for _ in range(10):

            dim = np.random.randint(1, 5)  # Vector dimension

            # Generate random arrays
            P, P_plane, normal = [np.random.uniform(low, high, dim)
                                  for _ in range(3)]

            P_proj = lin.proj_point_plane(P, P_plane, normal)
            d = lin.dist_point_plane(P, P_plane, normal)

            npt.assert_allclose(d, np.linalg.norm(P_proj - P))

    def test_unit(self):

        low, high= -10, 10

        for i in range(10):

            dim = np.random.randint(1, 5)  # Vector dimension

            v = np.random.uniform(low, high, dim)

            npt.assert_allclose(np.linalg.norm(lin.unit(v)), 1)

    def test_angle_direction(self):

        forward = np.array([0, 1, 0])
        up = np.array([0, 0, -1])
 
        x1 = lin.angle_direction(np.array([3, 5, 0]), forward, up)
        x2 = lin.angle_direction(np.array([2, -1, 0]), forward, up)
        x3 = lin.angle_direction(np.array([-2, 1, 0]), forward, up)
        x4 = lin.angle_direction(np.array([0, 10, 5]), forward, up)

        self.assertEqual(x1, 1)
        self.assertEqual(x2, 1)
        self.assertEqual(x3, -1)
        self.assertEqual(x4, 0)


if __name__ == '__main__':
    unittest.main()