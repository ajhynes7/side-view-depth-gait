"""Functions related to numpy arrays or operations."""

import numpy as np


def to_column(array_1d):
    """
    Convert a 1D numpy array to a 2D column.

    Parameters
    ----------
    array_1d : ndarray
        (n, ) array.

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
    return array_1d.reshape(-1, 1)


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
