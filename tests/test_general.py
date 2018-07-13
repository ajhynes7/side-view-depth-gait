import pytest
import numpy as np

import numpy.testing as npt

import modules.general as gen


def test_divide_no_error():

    assert gen.divide_no_error(10, 5) == 2

    with pytest.raises(Exception):

        gen.divide_no_error(0, 4)
        gen.divide_no_error(4, 0)
        gen.divide_no_error(5, np.nan)


def test_pairwise():

    x = [1, 2, 3, 4]

    assert [*gen.pairwise(x)] == [(1, 2), (2, 3), (3, 4)]


def test_strings_with_any_substrings():

    strings = ['Oatmeal', 'Bobcat', 'Lightbulb', 'Strikeout']

    substrings = ['oat', 'out']
    str_idx, sub_idx = gen.strings_with_any_substrings(strings, substrings)

    assert str_idx == [3]
    assert sub_idx == [1]

    substrings = ['Oat', 'out']
    str_idx, sub_idx = gen.strings_with_any_substrings(strings, substrings)

    assert str_idx == [0, 3]
    assert sub_idx == [0, 1]


def test_iterable_to_dict():

    assert gen.iterable_to_dict([]) == {}

    x = [1, 2, 3, 10, 50]
    d = gen.iterable_to_dict(x)

    assert d == {0: 1, 1: 2, 2: 3, 3: 10, 4: 50}


def test_dict_to_array():

    d = {0: 1, 4: 10}

    x = gen.dict_to_array(d)
    y = [1, np.nan, np.nan, np.nan, 10]

    npt.assert_array_equal(x, y)
