"""Module for signal processing."""

import numpy as np
import pandas as pd
from scipy.stats import linregress

import modules.general as gen
import modules.mean_shift as ms
import modules.linear_algebra as lin


def filter_by_function(signal, func):
    """
    Filter a signal using a given function.

    The function takes the signal as input and returns a single value.
    The values below and above this value are returned.

    Parameters
    ----------
    signal : ndarray
        Input array.
    func : Function
        Function of form: f(signal) -> value

    Returns
    -------
    signal_lower, signal_upper : ndarray
        Values below and above the output of the function.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 5, 6])
    >>> x_lower, x_upper = filter_by_function(x, np.mean)

    >>> x_lower
    array([1, 2, 3])

    >>> x_upper
    array([5, 6])

    """
    value = func(signal)

    assert ~np.isnan(value)

    signal_lower = signal[signal < value]
    signal_upper = signal[signal > value]

    return signal_lower, signal_upper


def root_mean_square(x):
    """
    Return the root mean square of an array.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    float
        Root mean square.

    Examples
    --------
    >>> x = np.array([0, 1])
    >>> root_mean_square(x) == np.sqrt(2) / 2
    True

    """
    return np.sqrt(sum(x**2) / x.size)


def mean_shift_peaks(signal, **kwargs):
    """
    Find peaks in a signal using mean shift clustering.

    The frames (x values) of the signal are clustered with mean shift.
    Middle and peak frames are returned for each cluster found.

    Parameters
    ----------
    signal : Series
        Index values are frames.

    kwargs : dict, optional
        Additional keywords passed to mean shift cluster function.

    Returns
    -------
    peak_frames : ndarray
        Frames where values are at a peak.
    mid_frames : ndarray
        Frames closest to cluster centroids.

    """
    frames = signal.index.values.reshape(-1, 1)

    # Find centres of foot distance peaks with mean shift
    labels, centroids, k = ms.cluster(frames, **kwargs)

    # Find frames with highest foot distance in each mean shift cluster
    peak_frames = [signal[labels == i].idxmax() for i in range(k)]

    # Find the frames closest to the mean shift centroids
    mid_frames = [lin.closest_point(frames, x)[0].item()
                  for x in centroids]

    return np.unique(peak_frames), np.unique(mid_frames)


def derivative(signal):
    """
    First derivative of discrete signal.

    Parameters
    ----------
    signal : Series
        Index values are frames.

    Returns
    -------
    deriv : Series
        First derivative.

    """
    frames = signal.index.values

    deriv = pd.Series(index=frames)

    for t_prev, t_curr, t_next in gen.window(frames, 3):

        delta_t = t_next - t_prev

        delta_signal = signal[t_next] - signal[t_prev]

        deriv[t_curr] = delta_signal / delta_t

    return deriv


def window_derivative(signal, n=3):
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

    >>> window_derivative(signal, n=3)
    1           NaN
    2      2.000000
    3      1.214286
    5      1.214286
    6      1.571429
    8     -0.714286
    9     -5.000000
    10    -8.500000
    11   -10.000000
    12          NaN
    dtype: float64

    """
    assert n % 2 != 0

    frames = signal.index.values

    frame_windows = gen.window(frames, n)
    signal_windows = gen.window(signal, n)

    deriv = pd.Series(index=frames)

    for x, y in zip(frame_windows, signal_windows):

        frame = np.median(x)
        slope = linregress(x, y).slope

        deriv[frame] = slope

    return deriv
