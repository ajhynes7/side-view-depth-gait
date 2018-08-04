"""Tests for functions using numpy."""

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import modules.numpy_funcs as nf

regular_ints = st.integers(min_value=-1e6, max_value=1e6)
positive_ints = st.integers(min_value=1, max_value=1e6)
list_lengths = st.integers(min_value=1, max_value=50)


@st.composite
def same_len_lists(draw):

    n = draw(st.integers(min_value=1, max_value=50))
    fixed_length_list = st.lists(regular_ints, min_size=n, max_size=n)

    return (draw(fixed_length_list), draw(fixed_length_list))


def test_divide_no_error():

    assert nf.divide_no_error(10, 5) == 2

    with pytest.raises(Exception):

        nf.divide_no_error(0, 4)
        nf.divide_no_error(4, 0)
        nf.divide_no_error(5, np.nan)


def test_dict_to_array():

    d = {0: 1, 4: 10}

    x = nf.dict_to_array(d)
    y = [1, np.nan, np.nan, np.nan, 10]

    assert np.allclose(x, y, equal_nan=True)


@given(st.lists(list_lengths, min_size=1))
def test_make_consecutive(array):

    consecutive, index = nf.make_consecutive(array)

    assert nf.all_consecutive(consecutive)
    assert np.array_equal(np.unique(array), consecutive[index])


@given(same_len_lists())
def test_expand_arrays(lists):

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
