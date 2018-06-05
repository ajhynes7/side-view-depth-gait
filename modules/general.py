

def pairwise(x):
    """
    Return a zip object that is used to iterate
    through consecutive pairs in an iterable.

    Parameters
    ----------
    x : iterable
        Any iterable.

    Returns
    -------
    zip
        Zip object.

    Examples
    --------
    >>> for a, b in pairwise([1, 2, 3, 4]): print(a, b)
    1 2
    2 3
    3 4
    """

    return zip(x[:-1], x[1:])


def strings_with_any_substrings(strings, substrings):
    """
    Given a list of strings and a list of substrings,
    yield each string that contains any of the substrings.

    Parameters
    ----------
    strings: iterable
        Iterable of strings.
    substrings : iterable
        Iterable of substrings

    Yields
    ------
    str
        String that contains of the substrings.

    Examples
    --------
    >>> strings = ['Wildcat', 'Sheepdog', 'Moose']
    >>> substrings = ['cat', 'dog']

    >>> list(strings_with_any_substrings(strings, substrings))
    ['Wildcat', 'Sheepdog']

    """
    for s in strings:
        for s_sub in substrings:

            if s_sub in s:
                yield s
                break


if __name__ == "__main__":

    import doctest
    doctest.testmod()
