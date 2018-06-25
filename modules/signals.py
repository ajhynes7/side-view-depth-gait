import numpy as np

import modules.mean_shift as ms
import modules.linear_algebra as lin


def root_mean_square(x):
    """
    Return the root mean square of an array.

    Parameters
    ----------
    x : ndarray
        Input numpy array.

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


def root_mean_filter(x):

    rms = root_mean_square(x)

    return x[x > rms]


def mean_shift_peaks(time_series, r=1):
    """
    Find peaks in the foot distance data.

    Applies mean shift to the foot distance values
    greater than the root mean square.

    Parameters
    ----------
    foot_dist : pandas Series
        Distance between feet at each frame.
        Index values are frame numbers.
    r : {int, float}, optional
        Radius for mean shift clustering (default is 1).

    Returns
    -------
    peak_frames : ndarray
        Frames where foot distance is at a peak.
    mid_frames : ndarray
        Frames closest to cluster centroids.

    """
    frames = time_series.index.values.reshape(-1, 1)

    # Find centres of foot distance peaks with mean shift
    labels, centroids, k = ms.cluster(frames, kernel='gaussian', radius=r)

    # Find frames with highest foot distance in each mean shift cluster
    peak_frames = [time_series[labels == i].idxmax() for i in range(k)]

    # Find the frames closest to the mean shift centroids
    mid_frames = [lin.closest_point(frames, x)[0].item()
                  for x in centroids]

    return np.unique(peak_frames), np.unique(mid_frames)
