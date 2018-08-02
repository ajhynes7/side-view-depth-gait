"""Tests for functions dealing with iterables."""

from itertools import tee

import numpy as np
import hypothesis.strategies as st
from hypothesis import given

import modules.iterable_funcs as itf

sequences = st.one_of(st.lists(elements=st.integers(),
                               min_size=1, max_size=50),
                      st.text(min_size=1, max_size=50),
                      )


@given(sequences)
def test_pairwise(seq):
    """Test returning pairwise values from a sequence."""
    if len(seq) >= 2:

        pairs = [*itf.pairwise(seq)]

        assert len(seq) == len(pairs) + 1


@given(st.iterables(elements=st.floats(), max_size=50))
def test_iterable_to_dict(it):
    """Test converting an iterable to a dictionary."""
    it_1, it_2 = tee(it)

    d = itf.iterable_to_dict(it_1)

    assert isinstance(d, dict)
    assert len(d) == len(list(it_2))


@given(st.iterables(elements=st.integers(), max_size=50),
       st.dictionaries(keys=st.integers(), values=st.integers()),
       )
def test_map_with_dict(it, mapping):
    """Test mapping an iterator with a dictionary."""
    it_1, it_2 = tee(it)

    mapped = itf.map_with_dict(it_1, mapping)

    assert isinstance(mapped, list)
    assert len(mapped) == len(list(it_2))


@given(st.iterables(elements=st.integers(min_value=-1e6, max_value=1e6),
                    min_size=1, max_size=50),
       st.iterables(elements=st.integers(min_value=0, max_value=10),
                    min_size=1, max_size=50),
       )
def test_repeat_by_element(it, repeat_nums):
    """Test repeating elements in an iterable."""
    it_1, it_2 = tee(it)
    repeat_1, repeat_2 = tee(repeat_nums)

    repeated = list(itf.repeat_by_element(it_1, repeat_1))

    n_iter_elements = len(list(it_2))
    repeat_values = list(repeat_2)

    assert len(repeated) == sum(repeat_values[:n_iter_elements])


@given(sequences)
def test_label_repeated_elements(seq):
    """Test labelling groups of consecutive repeated elements."""
    seq_sorted = sorted(seq)

    labels_1 = list(itf.label_repeated_elements(seq))
    labels_2 = list(itf.label_repeated_elements(seq_sorted))

    unique_vals = np.unique(seq)

    assert len(seq) == len(labels_1) == len(labels_2)

    assert len(unique_vals) - 1 <= max(labels_2) <= max(labels_1)
