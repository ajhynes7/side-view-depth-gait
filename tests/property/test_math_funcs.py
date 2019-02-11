"""Test math functions."""

import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given

import analysis.math_funcs as mf


@given(st.floats(min_value=0, max_value=1e10), st.floats(min_value=0, max_value=1e10))
def test_norm_ratio(a, b):
    """The ratio must be greater than zero and less than or equal to one."""
    assume(a != 0 and b != 0)

    ratio = mf.norm_ratio(a, b)

    assert ratio > 0 and ratio <= 1
    assert np.isclose(ratio * max([a, b]), min([a, b]))
