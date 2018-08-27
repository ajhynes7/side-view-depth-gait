"""
Functions involving a sliding window.

The window contains n elements from a larger iterable.

"""
import itertools

import numpy as np
import pandas as pd

import modules.numpy_funcs as nf


def generate_window(sequence, n=2):
    """
    Generate a sliding window of width n from an iterable.

    Adapted from an itertools recipe.

    Parameters
    ----------
    sequence : iterable
        Input sequence.
    n : int, optional
        Width of sliding window (default 2).

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


def detect_peaks(x, y, *, window_length=3, min_height=0):
    """
    Detect peaks using a sliding window.

    Parameters
    ----------
    x, y : array_like
        Input arrays representing x and y coordinates.
    window_length : int, optional
        Length of sliding window (default 3).
    min_height : int, optional
        Minimum allowed height of peaks (default 0).

    Returns
    -------
    x_peak, y_peak : ndarray
        x and y values at the detected peaks.

    Examples
    --------
    >>> x, y = [10, 11, 12, 13, 14, 15], [3, 2, 10, 5, 3, 2]

    >>> detect_peaks(x, y)
    (array([12]), array([10.]))

    >>> x, y = [10, 11, 12, 30, 31, 32], [3, 2, 10, 5, 3, 2]

    >>> detect_peaks(x, y)
    (array([12, 30]), array([10.,  5.]))

    >>> detect_peaks(x, y, min_height=6)
    (array([12]), array([10.]))

    >>> detect_peaks(x, y, window_length=20)
    (array([12]), array([10.]))

    """
    # Ensure there are no gaps in the x-values (must be all consecutive).
    x, y = nf.expand_arrays(x, y)

    # Initially assume that all elements are peaks.
    all_indices = [i for i, _ in enumerate(x)]
    peak_set = set(all_indices)

    for window in generate_window(all_indices, n=window_length):

        indices = np.array(window)
        values = y[indices]

        if np.all(np.isnan(values)):
            continue

        # Index of the max value in the window.
        max_index = indices[np.nanargmax(values)]

        # Mark all indices other than the max index as a non-peak.
        non_peaks = set(indices)
        non_peaks.remove(max_index)
        peak_set = peak_set - non_peaks

    # Keep only indices of array values that reach min height.
    peak_set = {i for i in peak_set if y[i] >= min_height}
    peak_indices = sorted(peak_set)

    x_peak, y_peak = x[peak_indices], y[peak_indices]

    return x_peak, y_peak


def apply_to_padded(array, func, *args, **kwargs):
    """
    Apply a function to a sliding window of a padded array.

    Parameters
    ----------
    array : array_like
        Input array.
    func : function
        Function to apply to each window.
    r : int, optional
        Radius of sliding window (default 1).

    Returns
    -------
    list
        Result of applying input function to each window.

    Examples
    --------
    >>> x = [1, 1, 2, 5, 3, 4, 6, 8]
    >>> x = [float(i) for i in x]

    >>> apply_to_padded(x, np.nansum, 2, 'constant', constant_values=np.nan)
    [4.0, 9.0, 12.0, 15.0, 20.0, 26.0, 21.0, 18.0]

    >>> apply_to_padded(x, np.nansum, 3, 'constant', constant_values=np.nan)
    [9.0, 12.0, 16.0, 22.0, 29.0, 28.0, 26.0, 21.0]

    """
    pad_width = args[0]
    n = 2 * pad_width + 1

    padded = np.pad(array, *args, **kwargs)

    return [func(x) for x in generate_window(padded, n)]
