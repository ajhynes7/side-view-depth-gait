"""Functions related to NumPy arrays or operations."""

from typing import Sequence

import numpy as np
from numpy import ndarray


def interweave_rows(array_a: Sequence, array_b: Sequence) -> ndarray:
    """
    Interweave the rows of two arrays.

    Parameters
    ----------
    array_a, array_b : (N, D) array_like
        Input arrays with N rows.

    Returns
    -------
    array_c : (2 * N, D) ndarray
        Array with rows alternating from arrays A and B.

    Examples
    --------
    >>> array_a = [[1, 2], [3, 4], [5, 6]]
    >>> array_b = [[-1, -2], [-3, -4], [-5, -6]]

    >>> interweave_rows(array_a, array_b)
    array([[ 1,  2],
           [-1, -2],
           [ 3,  4],
           [-3, -4],
           [ 5,  6],
           [-5, -6]])

    """
    array_a = np.array(array_a)
    n_points, n_dim = array_a.shape

    array_c = np.empty((2 * n_points, n_dim), dtype=array_a.dtype)

    array_c[::2] = array_a
    array_c[1::2] = array_b

    return array_c


def label_by_split(indices_split: Sequence, n_elements: int) -> ndarray:
    """
    Return an array of labels from split indices.

    The indices mark different sections of the array.

    Parameters
    ----------
    indices_split : array_like
        1D array of indices to split on.
    n_elements: int
        Desired length of the output array.

    Returns
    -------
    ndarray
        Array of labels.
        Each label represents a section between the split indices.

    Examples
    --------
    >>> label_by_split([1, 3], 4)
    array([0, 1, 1, 2])

    >>> label_by_split([2, 3, 5], 8)
    array([0, 0, 1, 2, 2, 3, 3, 3])

    >>> label_by_split([2, 3, 5], 10)
    array([0, 0, 1, 2, 2, 3, 3, 3, 3, 3])

    The number of elements can be smaller than the max split index.

    >>> label_by_split([2, 3, 5], 4)
    array([0, 0, 1, 2])

    """

    def yield_label_sections():

        list_sections = np.split(np.zeros(n_elements), indices_split)

        for i, array_section in enumerate(list_sections):
            yield array_section + i

    return np.concatenate([*yield_label_sections()]).astype(int)


def filter_labels(labels: Sequence, min_elements: int) -> ndarray:
    """
    Return an array of labels with small groups marked as noise.

    Parameters
    ----------
    labels : array_like
        1D array of N labels.
    min_elements : int
        Minimum allowed number of labels in a group.

    Returns
    -------
    labels_filtered : (N,) ndarray
        Array of labels. Labels belong to small groups changed to -1.

    Examples
    --------
    >>> filter_labels([0, 0, 0, 1, 1, 2, 2, 2], 1)
    array([0, 0, 0, 1, 1, 2, 2, 2])

    >>> filter_labels([0, 0, 0, 1, 1, 2, 2, 2], 3)
    array([ 0,  0,  0, -1, -1,  2,  2,  2])

    The input labels do not need to be sorted.

    >>> filter_labels([2, 0, 1, 0, 1, 0, 2, 2], 3)
    array([ 2,  0, -1,  0, -1,  0,  2,  2])

    """
    labels = np.array(labels)
    labels_filtered = np.array(labels)

    for label in np.unique(labels):

        is_label = labels == label

        if is_label.sum() < min_elements:
            labels_filtered[is_label] = -1

    return labels_filtered
