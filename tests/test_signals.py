"""Test functions for signal processing."""

import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays

import modules.signals as sig

ints = st.integers(min_value=-1e6, max_value=1e6)

# Generates either an integer or a nan
ints_or_nan = st.one_of(ints, st.just(np.nan))

array_length = st.integers(min_value=1, max_value=50)
array_1d = arrays('int', (array_length), ints)


@given(array_1d)
def test_root_mean_square(array):
    """Test that rms is at least zero and between the min and max."""
    rms = sig.root_mean_square(array)

    assert rms >= 0
    assert np.min(array) <= rms <= np.max(np.abs(array))


@given(st.lists(ints_or_nan, min_size=1, max_size=50))
def test_normalize(array):
    """Test that all real values are in range [0, 1]."""
    assume(~np.all(np.isnan(array)))
    assume(np.nanmax(array) - np.nanmin(array) > 0)

    normalized = sig.normalize(array)
    real_values = normalized[~np.isnan(normalized)]

    assert np.logical_and(np.all(real_values >= 0), np.all(real_values <= 1))
