"""Functions related to strings."""


def strings_with_any_substrings(strings, substrings):
    """
    Yield each string that contains any of the substrings.

    The function is case sensitive.

    Parameters
    ----------
    strings: iterable
        Iterable of strings.
    substrings : iterable
        Iterable of substrings.

    Returns
    -------
    string_index : list
        Index to the matched strings.

    substring_index : list
        Index to the matched substrings.

    Examples
    --------
    >>> strings = ['Wildcat', 'Sheepdog', 'Moose', 'Tomcat']
    >>> substrings = ['cat', 'dog']

    >>> x, y = strings_with_any_substrings(strings, substrings)

    >>> x
    [0, 1, 3]

    >>> y
    [0, 1, 0]

    """
    string_index, substring_index = [], []

    for i, string in enumerate(strings):
        for j, substring in enumerate(substrings):

            if substring in string:

                string_index.append(i)
                substring_index.append(j)
                break

    return string_index, substring_index


def any_in_string(string, substrings):
    """
    Check if any substrings are in a string.

    Parameters
    ----------
    string : str
        Input string.
    substrings : iterable
        Sequence of substrings.

    Returns
    -------
    bool
        True if at least one of the substrings is contained in the string.

    Examples
    --------
    >>> any_in_string('Hoover dam', ['dog', 'cat', 'dam'])
    True

    >>> any_in_string('Hoover dam', ('dog', 'cat'))
    False

    """
    matched, _ = strings_with_any_substrings([string], substrings)

    return len(matched) > 0
