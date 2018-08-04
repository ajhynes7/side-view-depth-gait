"""Tests for functions using numpy."""

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import modules.numpy_funcs as nf

regular_ints = st.integers(min_value=-1e6, max_value=1e6)
positive_ints = st.integers(min_value=1, max_value=1e6)
list_lengths = st.integers(min_value=1, max_value=50)

array_like_1d = st.lists(regular_ints, min_size=1, max_size=50)


@st.composite
def same_len_lists(draw):
    """Generate two lists with the same length."""
    n = draw(st.integers(min_value=1, max_value=50))
    fixed_length_list = st.lists(regular_ints, min_size=n, max_size=n)

    return (draw(fixed_length_list), draw(fixed_length_list))


@given(array_like_1d)
def test_to_column(array):
    """Test converting an array_like to a column ndarray."""
    column_array = nf.to_column(array)
    n_rows, n_cols = column_array.shape

    assert n_rows == len(array) and n_cols == 1


@given(array_like_1d)
def test_ratio_non_zero(array):
    """Test finding the ratio of non zero elements to all elements."""
    ratio = nf.ratio_nonzero(array)

    assert isinstance(ratio, float)
    assert ratio <= len(array)


@given(array_like_1d)
def test_all_consecutive(array):
    """Test checking that an array has all consecutive values."""
    consecutive_test_2 = np.ptp(array) + 1 == len(array)

    assert nf.all_consecutive(array) == consecutive_test_2


@given(array_like_1d)
def test_unique_no_sort(array):
    """Test returning unique values while preserving original order."""
    unique_preserved = nf.unique_no_sort(array)
    unique_regular = np.unique(array)

    if nf.is_sorted(array):
        # If the array is already sorted, the two unique arrays should be equal
        assert np.array_equal(unique_preserved, unique_regular)

    else:
        # Sorting the preserved order should return the regular unique array
        assert np.array_equal(np.sort(unique_preserved), unique_regular)


@given(array_like_1d)
def test_map_to_whole(array):
    """Test mapping elements in an array to whole numbers starting at zero."""
    mapped = nf.map_to_whole(array)

    unique, counts = np.unique(array, return_counts=True)
    unique_mapped, counts_mapped = np.unique(mapped, return_counts=True)

    # Whole numbers from zero to the number of unique values.
    whole_nums = np.arange(len(unique_mapped))

    assert np.array_equal(unique_mapped, whole_nums)
    assert set(counts_mapped) == set(counts)


@given(same_len_lists())
def test_expand_arrays(lists):
    """Test expanding arrays x, y so that x is all consecutive."""
    x, y = lists

    assert x is not y

    if not np.array_equal(x, np.unique(x)):

        with pytest.raises(Exception):
            x_expanded, y_expanded = nf.expand_arrays(x, y)

    else:
        x_expanded, y_expanded = nf.expand_arrays(x, y)

        assert nf.all_consecutive(x_expanded)

        assert len(x_expanded) == len(y_expanded)
        assert len(nf.remove_nan(y_expanded)) == len(x)
