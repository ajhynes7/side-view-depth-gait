
import pytest
import numpy as np
import numpy.testing as npt

import analysis.math_funcs as mf


def test_centre_of_mass():

    points = np.array([[0, 1], [0, -1]])
    masses = np.array([10, 10])

    centre = mf.centre_of_mass(points, masses)

    npt.assert_almost_equal(centre, [0, 0])

    masses = np.array([2, 1])
    centre = mf.centre_of_mass(points, masses)
    npt.assert_almost_equal(centre, [0, 1 / 3])

    points = np.array([[1, 1], [0, 0]])
    masses = np.array([5, 10])
    centre = mf.centre_of_mass(points, masses)
    npt.assert_almost_equal(centre, [1 / 3, 1 / 3])

    with pytest.raises(Exception):
        points = [[1, 1], [0, 0]]
        centre = mf.centre_of_mass(points, masses)
        npt.assert_almost_equal(centre, [1 / 3, 1 / 3])