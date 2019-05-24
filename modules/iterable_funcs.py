"""Functions related to iterables."""


def pairwise(seq):
    """
    Return a zip object that contains consecutive pairs of a sequence.

    Parameters
    ----------
    seq : sequence
        Any sequence type (e.g. list, tuple, range, string).

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
    return zip(seq[:-1], seq[1:])


def iterable_to_dict(it):
    """
    Convert an iterable to a dictionary.

    Parameters
    ----------
    it : iterable
        Any iterable (e.g. set, dict, generator, sequence type).

    Returns
    -------
    dict
        Dictionary version of iterable.

    Examples
    --------
    >>> iterable_to_dict([1, 1, 2, 4, 5, 10, 20])
    {0: 1, 1: 1, 2: 2, 3: 4, 4: 5, 5: 10, 6: 20}

    >>> iterable_to_dict((i + 1 for i in range(3)))
    {0: 1, 1: 2, 2: 3}

    """
    return {i: value for i, value in enumerate(it)}
