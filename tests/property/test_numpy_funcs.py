"""Tests for functions using numpy."""

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays

import modules.numpy_funcs as nf


ints = st.integers(min_value=-1e6, max_value=1e6)
positive_ints = st.integers(min_value=1, max_value=1e6)
array_lengths = st.integers(min_value=1, max_value=50)

# Generates either an integer or a nan
ints_or_nan = st.one_of(ints, st.just(np.nan))

array_1d = arrays(int, array_lengths, ints)
array_like_1d = st.lists(ints, min_size=1, max_size=50)


@given(array_like_1d)
def test_to_column(array):
    """Test converting an array_like to a column ndarray."""
    column_array = nf.to_column(array)
    n_rows, n_cols = column_array.shape

    assert n_rows == len(array) and n_cols == 1


@given(arrays(float, array_lengths, st.floats(allow_nan=True)))
def test_remove_nan(array):
    """Test removing nans from an array."""
    # Assume that the input has at least one nan
    assume(np.any(np.isnan(array)))

    removed = nf.remove_nan(array)
    assert not np.any(np.isnan(removed))


@given(st.data())
def test_find_indices(data):
    """Test finding the indices of elements in an array."""
    array = data.draw(array_1d)

    # Random subset of the array
    n = data.draw(st.integers(min_value=1, max_value=len(array)))
    values_to_find = np.random.choice(array, n)

    indices = nf.find_indices(array, values_to_find)

    assert set(values_to_find) == set(array[indices])


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


@given(st.data())
def test_group_by_label(data):
    """Test splitting an array into groups using the labels of the elements."""
    n = data.draw(array_lengths)
    int_arrays = arrays(int, n, ints)

    array, labels = data.draw(int_arrays), data.draw(int_arrays)

    groups = list(nf.group_by_label(array, labels))

    # There is a group for each label
    assert len(groups) == len(np.unique(labels))


@given(st.data())
def test_label_by_split(data):
    """Test labelling an array with values used to split the labels."""
    n_1 = data.draw(array_lengths)
    n_2 = data.draw(array_lengths)

    array = data.draw(arrays(int, n_1, ints))
    split_vals = data.draw(arrays(int, n_2, ints))

    labels = nf.label_by_split(array, split_vals)

    assert len(labels) == len(array)
    assert max(labels) == np.sum(np.in1d(array, split_vals))


@given(st.data())
def test_large_boolean_groups(data):
    """Test finding groups of elements with enough True values."""
    n = data.draw(array_lengths)

    array_bool = data.draw(arrays(bool, n, st.booleans()))
    labels = data.draw(arrays(int, n, ints))

    min_length_1, min_length_2 = np.random.choice(n), np.random.choice(n)

    good_labels_1 = nf.large_boolean_groups(array_bool, labels, min_length_1)
    good_labels_2 = nf.large_boolean_groups(array_bool, labels, min_length_2)

    if min_length_1 > min_length_2:
        assert good_labels_1 <= good_labels_2

    elif min_length_1 < min_length_2:
        assert good_labels_1 >= good_labels_2

    if good_labels_1 < good_labels_2:
        assert min_length_1 > min_length_2


@given(array_like_1d)
def test_make_consecutive(array):
    """Test making an array have consecutive values from min to max."""
    array_consec, index = nf.make_consecutive(array)

    assert np.array_equal(array_consec[index], np.unique(array))
    assert nf.all_consecutive(array_consec)


@given(st.data())
def test_expand_arrays(data):
    """Test expanding arrays x, y so that x is all consecutive."""
    n = data.draw(array_lengths)
    lists = st.lists(ints, min_size=n, max_size=n)

    x, y = data.draw(lists), data.draw(lists)

    assume(not np.array_equal(x, y))
    assert len(x) == len(y)

    if not np.array_equal(x, np.unique(x)):

        with pytest.raises(Exception):
            x_expanded, y_expanded = nf.expand_arrays(x, y)

    else:
        x_expanded, y_expanded = nf.expand_arrays(x, y)

        assert nf.all_consecutive(x_expanded)

        assert len(x_expanded) == len(y_expanded)
        assert len(nf.remove_nan(y_expanded)) == len(x)
