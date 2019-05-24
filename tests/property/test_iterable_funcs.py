"""Tests for functions dealing with iterables."""

from itertools import tee

import hypothesis.strategies as st
from hypothesis import given

import modules.iterable_funcs as itf

sequences = st.one_of(
    st.lists(elements=st.integers(), min_size=1, max_size=50),
    st.text(min_size=1, max_size=50),
)


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
