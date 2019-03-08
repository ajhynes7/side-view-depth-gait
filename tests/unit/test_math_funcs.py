import pytest
import numpy as np
from numpy.testing import assert_allclose

import analysis.math_funcs as mf


@pytest.mark.parametrize(
    "x, tolerance, limits_expected",
    [
        (5, 1, (4, 6)),
        (10, 5, (5, 15)),
        (-5, 1.5, (-6.5, -3.5)),
        (np.array([1, 2]), 1, (np.array([0, 1]), np.array([2, 3]))),
    ],
)
def test_limits(x, tolerance, limits_expected):

    assert_allclose(mf.limits(x, tolerance), limits_expected)


@pytest.mark.parametrize(
    "a, b, ratio_expected",
    [(4, 2, 0.5), (2, 4, 0.5), (15, 3, 0.2), (0, 5, np.nan), (4, 0, np.nan)],
)
def test_norm_ratio(a, b, ratio_expected):

    assert np.isclose(mf.norm_ratio(a, b), ratio_expected, equal_nan=True)
