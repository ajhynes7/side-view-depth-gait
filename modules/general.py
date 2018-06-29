"""General functions."""

import numpy as np
import itertools


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


def map_sort(x):
    """
    Map elements in an array to whole numbers in order (0, 1, 2, ...).

    Parameters
    ----------
    x : array_like
        Input array.

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

    d = {k: v for k, v in zip(unique, output_values)}

    return list(map(d.get, x))


def window(sequence, n=2):
    """
    Generate a sliding window of width n from an iterable.

    Adapted from an itertools recipe.

    Parameters
    ----------
    sequence : iterable
        Input sequence.
    n : int, optional
        Width of sliding window (default 2)

    Yields
    ------
    result : tuple
        Tuple containing n elements from input sequence.

    """
    iterator = iter(sequence)

    result = tuple(itertools.islice(iterator, n))

    if len(result) == n:
        yield result

    for elem in iterator:

        result = result[1:] + (elem, )

        yield result


if __name__ == "__main__":

    import doctest
    doctest.testmod()
