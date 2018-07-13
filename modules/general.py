"""General functions."""

import numpy as np
from itertools import chain, repeat


def get_properties(class_object):
    """
    Calculate all properties of a class instance.

    Parameters
    ----------
    class_object : object
        Instance of class.

    Returns
    -------
    property_dict : dict
        Dictionary containing all properties of class instance.
        Dict keys are property names, dict values are property values.

    Notes
    -----
    Example usage:

    get_properties(steve)
    {'name': 'Steve', 'age': 8, 'weight': 30}

    """
    property_dict = {}

    class_name = class_object.__class__

    for var in vars(class_name):

        if isinstance(getattr(class_name, var), property):

            property_dict[var] = getattr(class_object, var)

    return property_dict


def limits(x, tolerance):
    """
    Return lower and upper bounds (x ± tolerance).

    Parameters
    ----------
    x : {int, float, ndarray}
        Input value or array
    tolerance : {int, float}
        Tolerance that defines lower and upper limits.

    Returns
    -------
    lower_lim, upper_lim : {int, float}
        Lower and upper limits.

    Examples
    --------
    >>> limits(10, 2)
    (8, 12)

    >>> limits(np.array([2, 3]), 5)
    (array([-3, -2]), array([7, 8]))

    """
    lower_lim, upper_lim = x - tolerance, x + tolerance

    return lower_lim, upper_lim


def divide_no_error(a, b):
    """
    Divide without allowing any floating-point errors.

    Parameters
    ----------
    a, b : {int, float}
        Input values

    Returns
    -------
    float
        Result of division

    Examples
    --------
    >>> divide_no_error(10, 2)
    5.0

    >>> divide_no_error(10, 0)
    Traceback (most recent call last):
    ZeroDivisionError: division by zero

    """
    np.seterr(all='raise')

    return a / b


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


def dict_to_array(d):
    """
    Convert a dictionary to a numpy array.

    Each dictionary key is an index to the array.

    Parameters
    ----------
    d : Input dictionary.
        All keys must be ints.

    Returns
    -------
    x : ndarray
        1-D numpy array.

    Examples
    --------
    >>> d = {0: 1, 1: 1, 2: 2, 3: 4, 4: 5, 5: 10, 8: 20}

    >>> dict_to_array(d)
    array([ 1.,  1.,  2.,  4.,  5., 10., nan, nan, 20.])

    """
    n_elements = max(d.keys()) + 1

    x = np.full(n_elements, np.nan)

    for element in d:
        x[element] = d[element]

    return x


def unique_no_sort(x):
    """
    Return array of unique elements with order preserved.

    The original numpy unique() returns a sorted list.
    The order is determined by the first appearance of each unique element.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    ndarray
        Unique elements of x with original order preserved.

    Examples
    --------
    >>> unique_no_sort([1, 1, 1, 0, 0, 3, 3, 5, 5])
    array([1, 0, 3, 5])

    >>> unique_no_sort([1, 1, 1, 0, 0, 5, 3, 3, 5])
    array([1, 0, 5, 3])

    """
    unique, return_ind = np.unique(x, return_index=True)

    return unique[np.argsort(return_ind)]


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
    return list(map(mapping.get, seq))


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
    unique = unique_no_sort(x)

    output_values = [i for i, _ in enumerate(unique)]

    mapping = {k: v for k, v in zip(unique, output_values)}

    return map_with_dict(x, mapping)


def group_by_label(array, labels):
    """
    Group elements of an array by their corresponding labels.

    Parameters
    ----------
    array : ndarray
        Input array.
    labels : array_like
        Label of each element in the input array.

    Yields
    ------
    ndarray
        Subset of input array that corresponds to one label.

    Examples
    --------
    >>> array = np.array([10, 2, 4, 8])
    >>> labels = [0, 1, 2, 1]

    >>> for x in group_by_label(array, labels): print(x)
    [10]
    [2 8]
    [4]

    >>> array = np.array([[1, 2], [2, 4], [5, 0]])
    >>> labels = [0, 1, 1]

    >>> for x in group_by_label(array, labels): print(x)
    [[1 2]]
    [[2 4]
     [5 0]]

    """
    for label in np.unique(labels):
        yield array[labels == label]


def find_indices(array, elems_to_find):
    """
    Return the indices of array elements that match those in another array.

    Parameters
    ----------
    array : array_like
        Input array.
    elems_to_find : array_like
        Array of elements to find in the first array.

    Returns
    -------
    ndarray
        Indices to array.

    Examples
    --------
    >>> array = [1, 2, 3, 5, 10]
    >>> elems_to_find = [2, 3, 15]

    >>> find_indices(array, elems_to_find)
    array([1, 2])

    >>> find_indices(array, [8, 11])
    array([], dtype=int64)

    """
    return np.in1d(array, elems_to_find).nonzero()[0]


def label_by_split(array, split_vals):
    """
    Produce an array of labels by splitting the array at given values.

    Parameters
    ----------
    array : array_like
        Input array of n values.
    split_vals : array_like
        The input array is split into sub-arrays at these values.
        If a split value does not exist in the array, a new label is not
        created for it.

    Returns
    -------
    labels : ndarray
        (n, ) array of labels. If no splits are made, the labels are all zero.

    Examples
    --------
    >>> label_by_split([1, 2, 3, 4, 5], [2, 4])
    array([0, 1, 1, 2, 2])

    >>> label_by_split([10, 2, 3, -1, 5], [2, 4])
    array([0, 1, 1, 1, 1])

    >>> label_by_split([10, 2, 3, -1, 5], [2, 5])
    array([0, 1, 1, 1, 2])

    >>> label_by_split([1, 2, 3], [8, 5])
    array([0, 0, 0])

    """
    split_indices = find_indices(array, split_vals)
    sub_arrays = np.split(array, split_indices)

    labels = np.zeros(len(array), dtype=int)

    for i, sub_array in enumerate(sub_arrays):

        labels[np.in1d(array, sub_array)] = i

    return labels


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
