"""Test functions for string manipulation."""

import modules.string_funcs as sf


def test_strings_with_any_substrings():

    strings = ['Oatmeal', 'Bobcat', 'Lightbulb', 'Strikeout']

    substrings = ['oat', 'out']
    str_idx, sub_idx = sf.strings_with_any_substrings(strings, substrings)

    assert str_idx == [3]
    assert sub_idx == [1]

    substrings = ['Oat', 'out']
    str_idx, sub_idx = sf.strings_with_any_substrings(strings, substrings)

    assert str_idx == [0, 3]
    assert sub_idx == [0, 1]
