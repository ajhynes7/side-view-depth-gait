"""Functions related to numpy arrays or operations."""

import numpy as np
from scipy.interpolate import interp1d

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
    array([1., 2., 3., 4.])

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
    return np.nonzero(np.in1d(array, elems_to_find))[0]


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


def filter_consecutive_true(bool_array, min_length=2):
    """
    Remove sequences of consecutive True values that are too short.

    Consecutive sequences of True with a length less than the specified minimum
    are set to False.

    Parameters
    ----------
    bool_array : array_like
        Array of booleans.
    min_length : int, optional
        Minimum allowed length of consecutive True values (default 2).

    Returns
    -------
    filt_array : ndarray
        Filtered array with short groups of consecutive True set to False.

    Examples
    --------
    >>> array = [True, True, True, False, True, True, False, False, True]

    >>> filter_consecutive_true(array, min_length=3)
    array([ True,  True,  True, False, False, False, False, False, False])

    """
    filt_array = np.array(bool_array)

    group_labels = np.fromiter(itf.label_repeated_elements(bool_array), 'int')
    unique_labels, unique_counts = np.unique(group_labels, return_counts=True)

    for label, count in zip(unique_labels, unique_counts):

        if np.all(filt_array[group_labels == label]) and count < min_length:
            filt_array[group_labels == label] = False

    return filt_array


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


def fill_with_previous(array):
    """
    Fill nan values in an array with the most recent non-nan value.

    Parameters
    ----------
    array : array_like
        Input array.

    Returns
    -------
    array_filled : array_like
        Array with nan values filled.

    Examples
    --------
    >>> fill_with_previous([0, 0, np.nan, 1, 1, 2, np.nan, 3])
    [0, 0, 0, 1, 1, 2, 2, 3]

    >>> fill_with_previous([np.nan, 0, 1, 2])
    [nan, 0, 1, 2]

    """
    array_filled = array.copy()
    previous_number = np.nan

    is_nan = np.isnan(array)

    for i, value in enumerate(array):

        if is_nan[i]:
            array_filled[i] = previous_number
        else:
            previous_number = value

    return array_filled


def interp_nan(array):
    """
    Fill nan values of a 1D array using linear interpolation.

    Parameters
    ----------
    array : ndarray
        (n, ) input array.

    Returns
    -------
    ndarray
        (n, ) array with interpolated values.

    Examples
    --------
    >>> interp_nan(np.array([1, np.nan, np.nan, np.nan, 3]))
    array([1. , 1.5, 2. , 2.5, 3. ])

    >>> array = np.array([1, 1, 3, np.nan, np.nan, 5, np.nan, 8, 10])
    >>> np.round(interp_nan(array), 2)
    array([ 1.  ,  1.  ,  3.  ,  3.67,  4.33,  5.  ,  6.5 ,  8.  , 10.  ])

    >>> interp_nan(np.array([1, 2, 3]))
    array([1., 2., 3.])

    NaN values on the ends are not interpolated.

    >>> interp_nan(np.array([np.nan, 1, 2, 3, np.nan]))
    array([nan,  1.,  2.,  3., nan])

    """
    not_nan = ~np.isnan(array)

    indices = np.arange(len(array))

    x, y = indices[not_nan], array[not_nan]
    f = interp1d(x, y, bounds_error=False)

    return np.where(not_nan, array, f(indices))
