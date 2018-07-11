"""
Functions involving a sliding window.

The window contains n elements from a larger iterable.

"""
import itertools

import numpy as np
import pandas as pd


def generate_window(sequence, n=2):
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


def derivative(signal, n=3):
    """
    Compute the derivative of a discrete signal with a sliding window.

    The slope of the best fit line is found for each window.

    Parameters
    ----------
    signal : Series
        Index values are frames.
    n : int, optional
        Number of elements in sliding window (default 3).
        The number must be odd, so that the median of the window frames is a
        whole number.

    Returns
    -------
    deriv : Series
        Series of same length as the input signal.
        Index values are frames.

    Examples
    --------
    >>> frames = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12]
    >>> data = [10, 11, 14, 15, 18, 20, 15, 10, -2, -10]
    >>> signal = pd.Series(data, index=frames)

    >>> derivative(signal, n=3)
    1           NaN
    2      2.000000
    3      1.333333
    5      1.333333
    6      1.666667
    8     -1.000000
    9     -5.000000
    10    -8.500000
    11   -10.000000
    12          NaN
    dtype: float64

    """
    assert n % 2 != 0

    frames = signal.index.values

    frame_windows = generate_window(frames, n)
    signal_windows = generate_window(signal, n)

    middle_index = int(n / 2)

    deriv = pd.Series(index=frames)

    frame_list, deriv_list = [], []

    for x, y in zip(frame_windows, signal_windows):

        rise = y[-1] - y[0]
        run = x[-1] - x[0]

        frame_list.append(x[middle_index])
        deriv_list.append(rise / run)

    deriv[frame_list] = deriv_list

    return deriv


def detect_peaks(array, *, window_length=3, min_height=0):
    """
    Detect peaks in an array using a sliding window.

    Parameters
    ----------
    array : ndarray
        Input array.
    window_length : int, optional
        Length of sliding window (default 3).
    min_height : int, optional
        Minimum allowed height of peaks (default 0).

    Returns
    -------
    peak_indices : list
        Indices to the array where a peak value occurs.

    """
    all_indices = [i for i, _ in enumerate(array)]
    peak_set = set(all_indices)

    for window in generate_window(all_indices, n=window_length):

        indices = np.array(window)
        values = array[indices]

        if np.all(np.isnan(values)):
            continue

        max_index = indices[np.nanargmax(values)]

        non_peaks = set(indices)
        non_peaks.remove(max_index)
        peak_set = peak_set - non_peaks

    # Keep only indices of array values that reach min height
    peak_set = {x for x in peak_set if array[x] >= min_height}

    peak_indices = sorted(peak_set)

    return peak_indices
