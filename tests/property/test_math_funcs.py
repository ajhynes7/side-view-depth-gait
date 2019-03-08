"""Test math functions."""

import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_array_equal

import analysis.math_funcs as mf


st_floats_nonneg = st.floats(min_value=0, max_value=1e6)


@given(st_floats_nonneg, st_floats_nonneg.filter(lambda x: x > 0))
def test_limits(x, tolerance):

    lower_limit, upper_limit = mf.limits(x, tolerance)

    assert upper_limit > lower_limit

    array_limits = np.array(x) + np.array([-tolerance, tolerance])

    assert_array_equal(array_limits, (lower_limit, upper_limit))


@given(st_floats_nonneg, st_floats_nonneg)
def test_norm_ratio(a, b):
    """
    Test the normalized ratio between two values A and B.

    If either value is zero, the ratio is undefined.

    The ratio must be in the interval (0, 1].

    """
    ratio = mf.norm_ratio(a, b)

    if a == 0 or b == 0:
        assert np.isclose(ratio, np.nan, equal_nan=True)

    else:
        assert ratio > 0 and ratio <= 1
        assert np.isclose(ratio * max([a, b]), min([a, b]))
