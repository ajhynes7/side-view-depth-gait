"""Functions related to numpy arrays or operations."""

import numpy as np

import modules.iterable_funcs as itf


def to_column(array):
    """
    Convert a 1D array to a 2D column.

    Parameters
    ----------
    array : array_like
        Input array with n elements.

    Returns
    -------
    ndarray
        (n, 1) array.

    Examples
    --------
    >>> array = np.array([1, 2, 3])
    >>> array.shape
    (3,)

    >>> column = to_column(array)
    >>> column.shape
    (3, 1)

    """
    return np.array(array).reshape(-1, 1)


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
    array([1., 2., 3., 4.])

    """
    return array[~np.isnan(array)]


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
    return np.nonzero(np.in1d(array, elems_to_find))[0]


def ratio_nonzero(array):
    """
    Return the ratio of nonzero elements to all elements.

    Parameters
    ----------
    array : array_like
        Input array.

    Returns
    -------
    float
        Ratio of nonzero elements.

    Examples
    --------
    >>> ratio_nonzero([0, 1, 2, 3, 0])
    0.6

    >>> ratio_nonzero([True, True, True, False])
    0.75

    """
    return np.count_nonzero(array) / len(array)


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


def unique_no_sort(array):
    """
    Return array of unique elements with order preserved.

    The original numpy unique() returns a sorted list.
    The order is determined by the first appearance of each unique element.

    Parameters
    ----------
    array : array_like
        Input array.

    Returns
    -------
    ndarray
        Unique elements of array with original order preserved.

    Examples
    --------
    >>> unique_no_sort([1, 1, 1, 0, 0, 3, 3, 5, 5])
    array([1, 0, 3, 5])

    >>> unique_no_sort([1, 1, 1, 0, 0, 5, 3, 3, 5])
    array([1, 0, 5, 3])

    """
    unique, return_ind = np.unique(array, return_index=True)

    return unique[np.argsort(return_ind)]


def map_to_whole(array):
    """
    Map elements in an array to whole numbers in order (0, 1, 2, ...).

    Parameters
    ----------
    array : array_like
        Input array of numbers.

    Returns
    -------
    list
        List of sorted labels.

    Examples
    --------
    >>> map_to_whole([1, 1, 1, 0, 0, 2, 2, 3, 3, 3])
    [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]

    >>> map_to_whole([3, 3, 5, 5, 5, 10, 2])
    [0, 0, 1, 1, 1, 2, 3]

    """
    unique = unique_no_sort(array)
    output_values = [i for i, _ in enumerate(unique)]

    mapping = {k: v for k, v in zip(unique, output_values)}

    return itf.map_with_dict(array, mapping)


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


def large_boolean_groups(array_bool, labels, min_length=2):
    """
    Find groups of values in a boolean array that have enough True elements.

    Parameters
    ----------
    array_bool : array_like
        Input array.
    labels : ndarray
        Label of each element in boolean array.
        The labels define the groups of elements.
    min_length : int, optional
        Minimum allowed number of True elements in a group.

    Returns
    -------
    good_labels : set
        Set of labels corresponding to groups with enough True elements.

    Examples
    --------
    >>> array = np.array([True, True, True, False, False, True, False])
    >>> labels = np.array([0, 1, 2, 1, 1, 2, 0])
    >>> large_boolean_groups(array, labels)
    {2}

    >>> large_boolean_groups(array, labels, min_length=5)
    set()

    """
    good_labels = set()

    for label in set(labels):

        if np.sum(array_bool[labels == label]) >= min_length:

            good_labels.add(label)

    return good_labels


def make_consecutive(array):
    """
    Convert an array to one with consecutive numbers.

    The numbers are consecutive from the the min to the max of the input array.

    Parameters
    ----------
    array : ndarray
        Input array.

    Returns
    -------
    consecutive : ndarray
        Array of consecutive numbers.
    index : ndarray
        Index to the consecutive array.
        consecutive[index] returns the unique values of the input array.

    Examples
    --------
    >>> array = np.array([1, 3, 4, 6, 7, 9])

    >>> consecutive, index = make_consecutive(array)
    >>> consecutive
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> index
    array([0, 2, 3, 5, 6, 8])
    >>> consecutive[index]
    array([1, 3, 4, 6, 7, 9])

    >>> consecutive, index = make_consecutive([1, 1, 2, 5])
    >>> consecutive
    array([1, 2, 3, 4, 5])
    >>> index
    array([0, 1, 4])

    """
    min_val, max_val = np.min(array), np.max(array)
    consecutive = np.arange(min_val, max_val + 1)

    index = np.nonzero(np.in1d(consecutive, array))[0]

    return consecutive, index


def expand_arrays(x, y):
    """
    Expand arrays so that x is all consecutive numbers and y aligns with x.

    The input x values must be unique.

    Parameters
    ----------
    x, y : array_like
        Input arrays.

    Returns
    -------
    x_expanded, y_expanded : ndarray
        Expanded arrays.

    Raises
    ------
    ValueError
        When there are duplicate x values.

    Examples
    --------
    >>> x = [0, 1, 2, 5, 8, 9]
    >>> y = [7, 6, 3, 4, 1, 5]
    >>> x_expanded, y_expanded = expand_arrays(x, y)
    >>> x_expanded
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> y_expanded
    array([ 7.,  6.,  3., nan, nan,  4., nan, nan,  1.,  5.])

    >>> x_expanded, y_expanded = expand_arrays([5, 8, 15], [7, 6, 3])
    >>> x_expanded
    array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
    >>> y_expanded
    array([ 7., nan, nan,  6., nan, nan, nan, nan, nan, nan,  3.])

    """
    if not np.array_equal(x, np.unique(x)):
        raise ValueError('x values are not unique.')

    x_expanded, index = make_consecutive(x)

    y_expanded = np.full(len(x_expanded), np.nan)
    y_expanded[index] = y

    return x_expanded, y_expanded
