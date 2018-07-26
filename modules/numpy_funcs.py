"""Functions related to numpy arrays or operations."""

import numpy as np

import modules.iterable_funcs as itf


def remove_nan(array):
    """
    Remove nan values from an array.

    Parameters
    ----------
    array : ndarray
        Input array.

    Returns
    -------
    ndarray
        Array with nan elements removed.

    Examples
    --------
    >>> array = np.array([1, 2, 3, np.nan, 4])

    >>> remove_nan(array)
    array([1.0, 2.0, 3.0, 4.0])

    """
    return array[~np.isnan(array)]


def ratio_nonzero(x):
    """
    Return the ratio of nonzero elements to all elements.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    float
        Ratio of nonzero elements

    Examples
    --------
    >>> ratio_nonzero([0, 1, 2, 3, 0])
    0.6

    >>> ratio_nonzero([True, True, True, False])
    0.75

    """
    return np.count_nonzero(x) / len(x)


def to_column(x):
    """
    Convert a 1D array to a 2D column.

    Parameters
    ----------
    x : array_like
        Input array with n elements.

    Returns
    -------
    ndarray
        (n, 1) array.

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> x.shape
    (3,)

    >>> column = to_column(x)
    >>> column.shape
    (3, 1)

    """
    return np.array(x).reshape(-1, 1)


def all_consecutive(array):
    """
    Check if elements in array are all consecutive.

    Parameters
    ----------
    array : array_like
        Input array.

    Returns
    -------
    bool
        True if all elements are consecutive

    Examples
    --------
    >>> all_consecutive([1, 2, 3])
    True

    >>> all_consecutive([1, 2, 4])
    False

    >>> all_consecutive([1.1, 2.1, 3.1])
    True

    """
    return np.all(np.diff(array) == 1)


def is_sorted(array):
    """
    Check if array is sorted in ascending order.

    Parameters
    ----------
    array : array_like
        Input array.

    Returns
    -------
    bool
        True if array is in ascending order.

    Examples
    --------
    >>> is_sorted([1, 2, 3])
    True

    >>> is_sorted([-10, -5, 0, 8])
    True

    >>> is_sorted([-5, 0, 4, 3, 6, 8])
    False

    """
    return np.all(np.diff(array) >= 0)


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

    return itf.map_with_dict(x, mapping)


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


def label_by_split(array, split_vals):
    """
    Produce an array of labels by splitting the input array at given values.

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


def expand_arrays(x, y):
    """
    Expand arrays so that x is all consecutive numbers and y aligns with x.

    Parameters
    ----------
    x, y : array_like
        Input arrays.

    Returns
    -------
    x_exp, y_exp : ndarray
        Expanded arrays.

    Examples
    --------
    >>> x = [0, 1, 2, 5, 8, 9]
    >>> y = [7, 6, 3, 4, 1, 5]
    >>> x_exp, y_exp = expand_arrays(x, y)

    >>> x_exp
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    >>> y_exp
    array([ 7.,  6.,  3., nan, nan,  4., nan, nan,  1.,  5.])

    >>> x_exp, y_exp = expand_arrays([5, 8, 15], [7, 6, 3])

    >>> x_exp
    array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])

    >>> y_exp
    array([ 7., nan, nan,  6., nan, nan, nan, nan, nan, nan,  3.])

    """
    min_x, max_x = np.min(x), np.max(x)
    x_exp = np.arange(min_x, max_x + 1)

    y_exp = np.full(x_exp.size, np.nan)
    y_exp[x - min_x] = y

    return x_exp, y_exp


def is_idempotent(f, x):
    """
    Verify that a function is idempotent.

    For a function f:
        f(x) == f(f(x))

    Parameters
    ----------
    f : function
        Function to test for idempotence.
    x : any
        Input to function.

    Returns
    -------
    bool
        True if function is idempotent.

    Examples
    --------
    >>> is_idempotent(abs, -5)
    True

    >>> is_idempotent(lambda x: x + 10, 0)
    False

    """
    return np.allclose(f(x), f(f(x)))


def label_consecutive_true(bool_array):
    """
    Assign a unique label to each group of consecutive True values in an array.

    Parameters
    ----------
    bool_array : array_like
        Array of `n` boolean values.

    Returns
    -------
    ndarray
        (n,) array of labels.
        Each True element receives a label while each False element
        is set to nan.

    Examples
    --------
    >>> x = [True, True, False, False, False, True, True, False, False, True]

    >>> label_consecutive_true(bool_array)
    array([ 0.,  0., nan, nan, nan,  1.,  1., nan, nan,  2.])

    """
    labels = np.fromiter(itf.label_repeated_elements(bool_array), 'float')
    labels[~np.array(bool_array)] = np.nan

    unique = np.unique(remove_nan(labels))
    n_unique = len(unique)

    map_dict = {k: v for k, v in zip(unique, range(n_unique))}

    return np.array(itf.map_with_dict(labels, map_dict), dtype='float')
