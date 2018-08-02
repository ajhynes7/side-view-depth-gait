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
    >>> x = [1, 1, 2, 4, 5, 10, 20]

    >>> iterable_to_dict(x)
    {0: 1, 1: 1, 2: 2, 3: 4, 4: 5, 5: 10, 6: 20}

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


def repeat_by_list(array, repeat_nums):
    """
    Repeat each element in an array by a corresponding number.

    Parameters
    ----------
    array : iterable
        Input iterable.
    repeat_nums : iterable
        Iterable of numbers.

    Returns
    -------
    itertools.chain
        Iterator with repeated elements.

    Examples
    --------
    >>> [*repeat_by_list([1, 2, 3], [2, 2, 3])]
    [1, 1, 2, 2, 3, 3, 3]

    >>> [*repeat_by_list([1, 2, 3], [0, 2, 3])]
    [2, 2, 3, 3, 3]

    >>> [*repeat_by_list("abc", (2, 0, 3, 4))]
    ['a', 'a', 'c', 'c', 'c']

    """
    repeated = (repeat(x, n) for x, n in zip(array, repeat_nums))

    return chain(*repeated)


def label_repeated_elements(array):
    """
    Assign a label to each group of repeated elements in an iterable.

    Parameters
    ----------
    array : iterable
        Input iterable.

    Yields
    ------
    int
        Current label value.

    Examples
    --------
    >>> array = [0, 0, 1, 0, 0, 1]
    >>> [*label_repeated_elements(array)]
    [0, 0, 1, 2, 2, 3]

    >>> [*label_repeated_elements("abbbddg")]
    [0, 1, 1, 1, 2, 2, 3]

    """
    current_value, count = array[0], 0

    for x in array:

        if x != current_value:
            current_value = x
            count += 1

        yield count
