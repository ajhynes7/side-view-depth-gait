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
        The number must be odd so the window has a middle element.

    Returns
    -------
    deriv : Series
        Series of same length as the input signal.
        Index values are frames.

    Raises
    ------
    ValueError
        When the sliding window length is an even number.

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
    if n % 2 == 0:
        raise ValueError("The sliding window length is even.")

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


def apply_to_padded(array, func, r=1):
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
    >>> array = [1, 1, 2, 5, 3, 4, 6, 8]
    >>> apply_to_padded(array, np.nansum, r=1)
    >>> apply_to_padded(, np.nansum, r=2)
    [4.0, 9.0, 12.0, 15.0, 20.0, 26.0, 21.0, 18.0]

    """
    n = 2*r + 1
    floats = np.array(array).astype(float)

    padded = np.pad(floats, r, 'constant', constant_values=np.nan)
    windows = [*generate_window(padded, n)]

    return [func(x) for x in windows]
