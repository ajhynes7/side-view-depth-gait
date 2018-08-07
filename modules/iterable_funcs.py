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


def map_with_dict(it, mapping):
    """
    Map items in a sequence using a dictionary.

    The keys of the dictionary are mapped to their values.

    Parameters
    ----------
    it : iterable
        Any iterable (e.g. set, dict, generator, sequence type).
    mapping : dict
        Dictionary used for mapping.

    Returns
    -------
    list
        New sequence with mapped items.

    Examples
    --------
    >>> map_with_dict(['R', 'L', 'R'], {'R': 'Right', 'L': 'Left'})
    ['Right', 'Left', 'Right']

    >>> map_with_dict([0, 0, 1], {0: 10, 1: 5, 2: 8})
    [10, 10, 5]

    >>> map_with_dict((i for i in range(5)), {1: 2, 2: 3})
    [None, 2, 3, None, None]

    """
    return [*map(mapping.get, it)]


def label_repeated_elements(seq):
    """
    Assign a label to each group of consecutive repeated elements.

    Parameters
    ----------
    seq : sequence
        Any sequence type (e.g. list, tuple, range, string).

    Yields
    ------
    int
        Current label value.

    Examples
    --------
    >>> seq = [0, 0, 1, 0, 0, 1]
    >>> [*label_repeated_elements(seq)]
    [0, 0, 1, 2, 2, 3]

    >>> [*label_repeated_elements("abbbddg")]
    [0, 1, 1, 1, 2, 2, 3]

    """
    current_value, count = seq[0], 0

    for x in seq:
        if x != current_value:
            current_value = x
            count += 1

        yield count
