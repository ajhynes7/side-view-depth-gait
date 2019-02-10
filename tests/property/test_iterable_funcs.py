"""Tests for functions dealing with iterables."""

from itertools import tee

import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given

import modules.iterable_funcs as itf

sequences = st.one_of(
    st.lists(elements=st.integers(), min_size=1, max_size=50),
    st.text(min_size=1, max_size=50))


def all_unique(iterable):
    """Check if all elements in iterable are unique."""
    return len(set(iterable)) == len(iterable)


@given(sequences)
def test_pairwise(seq):
    """Test returning pairwise values from a sequence."""
    if len(seq) >= 2:

        pairs = [*itf.pairwise(seq)]

        assert len(seq) == len(pairs) + 1


@given(st.iterables(elements=st.floats(), min_size=1, max_size=50))
def test_iterable_to_dict(iterable):
    """Test converting an iterable to a dictionary."""
    iterable_1, iterable_2 = tee(iterable)

    list_iterable = list(iterable_1)

    dict_ = itf.iterable_to_dict(iterable_2)

    assert list(dict_.items()) == list(enumerate(list_iterable))


@given(
    st.iterables(elements=st.integers(), max_size=50),
    st.dictionaries(keys=st.integers(), values=st.integers(), min_size=1),
    )
def test_map_with_dict(iterable, mapping):
    """Test mapping an iterator with a dictionary."""
    assume(all_unique(mapping.values()))

    it_1, it_2 = tee(iterable)
    list_iterable = list(it_1)

    list_mapped = itf.map_with_dict(it_2, mapping)

    assert isinstance(list_mapped, list)
    assert len(list_mapped) == len(list_iterable)

    set_iterable = set(list_iterable)
    set_keys = set(mapping.keys())
    set_intersection = set_iterable.intersection(set_keys)

    mapping_reversed = {value: key for key, value in mapping.items()}
    list_mapped_reversed = itf.map_with_dict(list_mapped, mapping_reversed)
    set_reversed = set(list_mapped_reversed) - {None}

    assert set_reversed == set_intersection


@given(sequences)
def test_label_repeated_elements(seq):
    """Test labelling groups of consecutive repeated elements."""
    seq_sorted = sorted(seq)

    labels_1 = list(itf.label_repeated_elements(seq))
    labels_2 = list(itf.label_repeated_elements(seq_sorted))

    unique_vals = np.unique(seq)

    assert len(seq) == len(labels_1) == len(labels_2)

    assert len(unique_vals) - 1 <= max(labels_2) <= max(labels_1)
