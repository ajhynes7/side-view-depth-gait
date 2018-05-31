import pytest
import numpy as np
import numpy.testing as npt

import modules.linear_algebra as lin


def test_unit():

    low, high = -10, 10

    for _ in range(10):

        dim = np.random.randint(1, 5)  # Vector dimension
        v = np.random.uniform(low, high, dim)

        npt.assert_allclose(np.linalg.norm(lin.unit(v)), 1)


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


def test_best_fit_line():

    points = np.random.rand(10, 3)
    _, direction = lin.best_fit_line(points)

    points_reversed = np.flip(points, axis=0)

    _, direction_reversed = lin.best_fit_line(points_reversed)

    # The two vectors should be parallel
    cross_prod = np.cross(direction_reversed, direction)
    npt.assert_array_almost_equal(cross_prod, np.array([0, 0, 0]))

    npt.assert_allclose(np.linalg.norm(direction), 1)


@pytest.mark.parametrize("a, b, expected", [
    (np.array([2, 0]), np.array([-2, 0]), np.pi),
    (np.array([5, 5, 5]), np.array([1, 1, 1]), 0),
    (np.array([1, 0]), np.array([1, 1]), np.pi / 4),
    (np.array([1, 0]), np.array([-5, -5]), 3 * np.pi / 4),
])
def test_angle_between(a, b, expected):

    angle = lin.angle_between(a, b)

    npt.assert_allclose(angle, expected)


