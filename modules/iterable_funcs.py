"""Functions related to iterables."""

from itertools import chain, repeat


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


def repeat_by_element(it, repeat_nums):
    """
    Repeat each element in an iterable by a corresponding element value.

    Parameters
    ----------
    it : iterable
        Any iterable (e.g. set, dict, generator, sequence type).
    repeat_nums : iterable
        Iterable of integers.

    Returns
    -------
    itertools.chain
        Iterator with repeated elements.

    Examples
    --------
    >>> [*repeat_by_element([1, 2, 3], [2, 2, 3])]
    [1, 1, 2, 2, 3, 3, 3]

    >>> [*repeat_by_element([1, 2, 3], [0, 2, 3])]
    [2, 2, 3, 3, 3]

    >>> [*repeat_by_element((i for i in range(5)), (i for i in range(10)))]
    [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]

    >>> [*repeat_by_element("abc", (2, 0, 3, 4))]
    ['a', 'a', 'c', 'c', 'c']

    >>> [*repeat_by_element("abc", (i for i in range(10)))]
    ['b', 'c', 'c']

    """
    repeated = (repeat(x, n) for x, n in zip(it, repeat_nums))

    return chain(*repeated)


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
