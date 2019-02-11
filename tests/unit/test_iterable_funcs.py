"""Unit tests for iterable functions."""

import pytest

import modules.iterable_funcs as itf


@pytest.mark.parametrize("sequence, expected", [
    ([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)]),
    ("abcd", [('a', 'b'), ('b', 'c'), ('c', 'd')]),
])
def test_pairwise(sequence, expected):

    assert [*itf.pairwise(sequence)] == expected


@pytest.mark.parametrize("iterable, expected", [
    ([1, 2, 3, 4], {0: 1, 1: 2, 2: 3, 3: 4}),
    (range(3), {0: 0, 1: 1, 2: 2}),
    ("brown", {0: 'b', 1: 'r', 2: 'o', 3: 'w', 4: 'n'}),
])
def test_iterable_as_dict(iterable, expected):

    assert itf.iterable_to_dict(iterable) == expected


@pytest.mark.parametrize("iterable, mapping, expected", [
    ([1, 2, 3, 4], {1: 5, 2: 10, 3: 6, 4: 8, 5: 20}, [5, 10, 6, 8]),
    ([1, 2, 3, 4], {1: 5, 2: 10}, [5, 10, None, None]),
    ("cow", {1: 5, 'a': 5, 'c': 4, 'w': 'moo'}, [4, None, 'moo']),
])
def test_map_with_dict(iterable, mapping, expected):

    assert itf.map_with_dict(iterable, mapping) == expected


@pytest.mark.parametrize("sequence, expected", [
    ([1, 2, 3, 4], [0, 1, 2, 3]),
    ([1, 2, 3, 2], [0, 1, 2, 3]),
    ([1, 2, 3, 3], [0, 1, 2, 2]),
    ("aabcbddee", [0, 0, 1, 2, 3, 4, 4, 5, 5]),
])
def test_label_repeated_elements(sequence, expected):

    assert [*itf.label_repeated_elements(sequence)] == expected
