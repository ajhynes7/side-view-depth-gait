"""Unit tests for iterable functions."""

import pytest

import modules.iterable_funcs as itf


@pytest.mark.parametrize(
    "sequence, expected",
    [
        ([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)]),
        ("abcd", [('a', 'b'), ('b', 'c'), ('c', 'd')]),
    ],
)
def test_pairwise(sequence, expected):

    assert [*itf.pairwise(sequence)] == expected


@pytest.mark.parametrize(
    "iterable, expected",
    [
        ([1, 2, 3, 4], {0: 1, 1: 2, 2: 3, 3: 4}),
        (range(3), {0: 0, 1: 1, 2: 2}),
        ("brown", {0: 'b', 1: 'r', 2: 'o', 3: 'w', 4: 'n'}),
    ],
)
def test_iterable_as_dict(iterable, expected):

    assert itf.iterable_to_dict(iterable) == expected
