import numpy as np


def get_properties(class_name, class_object):

    property_dict = {}

    for var in vars(class_name):

        if isinstance(getattr(class_name, var), property):

            property_dict[var] = getattr(class_object, var)

    return property_dict


def limits(x, tolerance):
    """
    Return lower and upper limits (x ± tolerance).

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
    Divide without allowing any floating-point errors
    (e.g., division by zero).

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

    The function is case sensitive.

    Parameters
    ----------
    strings: iterable
        Iterable of strings.
    substrings : iterable
        Iterable of substrings.

    Returns
    -------
    matched_strings : list
        List of strings that contain one of the substrings.

    substring_indices : list
        Index to the substring contained by each string.

    Examples
    --------
    >>> strings = ['Wildcat', 'Sheepdog', 'Moose', 'Tomcat']
    >>> substrings = ['cat', 'dog']

    >>> x, y = strings_with_any_substrings(strings, substrings)

    >>> x
    ['Wildcat', 'Sheepdog', 'Tomcat']

    >>> y
    [0, 1, 0]

    """
    matched_strings, substring_index = [], []

    for string in strings:
        for i, substring in enumerate(substrings):

            if substring in string:
                substring_index.append(i)
                matched_strings.append(string)
                break

    return matched_strings, substring_index


def iterable_to_dict(x):
    """
    Converts an iterable to a dictionary.

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


if __name__ == "__main__":

    import doctest
    doctest.testmod()
