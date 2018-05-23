import pytest
import numpy as np
import numpy.testing as npt

from util import linear_algebra as lin


def test_line_distance():

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


def test_plane_distance():

    low, high = -10, 10

    for _ in range(10):

        dim = np.random.randint(1, 5)  # Vector dimension

        # Generate random arrays
        P, P_plane, normal = [np.random.uniform(low, high, dim)
                              for _ in range(3)]

        P_proj = lin.proj_point_plane(P, P_plane, normal)
        d = lin.dist_point_plane(P, P_plane, normal)

        npt.assert_allclose(d, np.linalg.norm(P_proj - P))


def test_unit():

    low, high = -10, 10

    for _ in range(10):

        dim = np.random.randint(1, 5)  # Vector dimension
        v = np.random.uniform(low, high, dim)

        npt.assert_allclose(np.linalg.norm(lin.unit(v)), 1)


@pytest.mark.parametrize("test_input, expected", [
    (np.array([1, 1, 0]), -1),
    (np.array([-1, 5, 0]), 1),
    (np.array([0, 5, 0]), 0),
    (np.array([0, -5, -10]), 0),
    (np.array([4, 2, 1]), -1),
])
def test_angle_direction(test_input, expected):

    forward = np.array([0, 1, 0])
    up = np.array([0, 0, 1])

    assert lin.angle_direction(test_input, forward, up) == expected
