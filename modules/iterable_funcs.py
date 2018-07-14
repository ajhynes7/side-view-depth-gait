"""Functions related to iterables."""

from itertools import chain, repeat

import modules.numpy_funcs as nf


def pairwise(x):
    """
    Return a zip object that contains consecutive pairs in an iterable.

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


def iterable_to_dict(x):
    """
    Convert an iterable to a dictionary.

    Parameters
    ----------
    x : iterable
        Any iterable.

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
    return {i: value for i, value in enumerate(x)}


def map_with_dict(seq, mapping):
    """
    Map items in a sequence using a dictionary.

    The keys of the dictionary are mapped to their values.

    Parameters
    ----------
    seq : iterable
        Input sequence.
    mapping : dict
        Dictionary used for mapping.

    Returns
    -------
    list
        New sequence with mapped items.

    Examples
    --------
    >>> seq = ['R', 'L', 'R']
    >>> mapping = {'R': 'Right', 'L': 'Left'}

    >>> map_with_dict(seq, mapping)
    ['Right', 'Left', 'Right']

    >>> seq = [0, 0, 1]
    >>> mapping = {0: 10, 1: 5, 2: 8}

    >>> map_with_dict(seq, mapping)
    [10, 10, 5]

    """
    return [*map(mapping.get, seq)]


def map_sort(x):
    """
    Map elements in an array to whole numbers in order (0, 1, 2, ...).

    Parameters
    ----------
    x : array_like
        Input array of numbers.

    Returns
    -------
    list
        List of sorted labels.

    Examples
    --------
    >>> map_sort([1, 1, 1, 0, 0, 2, 2, 3, 3, 3])
    [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]

    >>> map_sort([3, 3, 5, 5, 5, 10, 2])
    [0, 0, 1, 1, 1, 2, 3]

    """
    unique = nf.unique_no_sort(x)

    output_values = [i for i, _ in enumerate(unique)]

    mapping = {k: v for k, v in zip(unique, output_values)}

    return map_with_dict(x, mapping)


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
