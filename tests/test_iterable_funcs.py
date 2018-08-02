"""Tests for functions dealing with iterables."""

from itertools import tee

import hypothesis.strategies as st
from hypothesis import given

import modules.iterable_funcs as itf

sequences = st.one_of(st.lists(elements=st.integers(), max_size=50),
                      st.text(max_size=50),
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
