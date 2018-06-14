import numpy as np
import pytest

import numpy.testing as npt

import modules.math_funcs as mf


def test_gaussian():

    assert round(mf.gaussian(0), 4) == 0.3989


def test_sigmoid():

    assert mf.sigmoid(0) == 0.5

    assert mf.sigmoid(0, 10) == 0.5


def test_root_mean_square():

    x = np.array([0, 1, 2])
    assert np.isclose(mf.root_mean_square(x), np.sqrt(5 / 3))

    x = np.array([0, 1, 2, 3])
    assert np.isclose(mf.root_mean_square(x), np.sqrt(14 / 4))


def test_norm_ratio():

    for i in range(10):

        a = np.random.randint(1, 10)
        b = np.random.randint(1, 10)

        r = mf.norm_ratio(a, b)

        assert r > 0 and r <= 1

    assert np.isnan(mf.norm_ratio(0, 5))


def test_normalize_array():

    for i in range(10):

        dim = np.random.randint(2, 5)  # Vector dimension
        v = np.random.uniform(0, 10, dim)

        normalized = mf.normalize_array(v)

        in_range = np.logical_and(np.all(normalized >= 0),
                                  np.all(normalized <= 1))

        assert in_range


def test_centre_of_mass():

    points = np.array([[0, 1], [0, -1]])
    masses = [10, 10]

    centre = mf.centre_of_mass(points, masses)

    npt.assert_almost_equal(centre, [0, 0])

    masses = [2, 1]
    centre = mf.centre_of_mass(points, masses)
    npt.assert_almost_equal(centre, [0, 1/3])

    points = np.array([[1, 1], [0, 0]])
    masses = [5, 10]
    centre = mf.centre_of_mass(points, masses)
    npt.assert_almost_equal(centre, [1/3, 1/3])

    with pytest.raises(Exception):
        points = [[1, 1], [0, 0]]
        centre = mf.centre_of_mass(points, masses)
        npt.assert_almost_equal(centre, [1/3, 1/3])
