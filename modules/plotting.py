"""Functions for plotting points and visualizing results."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def scatter_labels(points, labels):
    """
    Scatter points that are coloured by label.

    Parameters
    ----------
    points : ndarray
        (n, d) array of n points with dimension d.
    labels : ndarray
        (n, ) array of point labels.

    """
    for label in np.unique(labels):

        points_label = points[labels == label]

        plt.scatter(points_label[:, 0], points_label[:, 1])


def scatter_pos(points, color='b'):
    """
    Plot 3D positions.

    Parameters
    ----------
    points : ndarray
        (n, 3) array of n points.
    color : str, optional
        Colour of all points (default 'b')

    """
    fig, ax = plt.subplots()

    ax = Axes3D(fig)

    # The z values of the points represent depth values,
    # while y values represent height
    # Thus, y and z are switched for plotting
    for point in points:
        ax.scatter(point[0], point[2], point[1], c=color, depthshade=False)

    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(100, 300)
    ax.set_zlim3d(-100, 100)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')


def scatter2(points):
    """
    Produce a 2D scatter plot from an array of points.

    Parameters
    ----------
    points : ndarray
        (n, 2) array of n points in two dimensions.

    """
    plt.scatter(points[:, 0], points[:, 1])


def scatter_order(points, order):
    """
    Produce a scatter plot using a specified order of the (x, y) coordinates.

    Parameters
    ----------
    points : ndarray
        (n, d) array of n points in d dimensions.
    order : array_like
        Order of the coordinates

    """
    plt.scatter(points[:, order[0]], points[:, order[1]])


def scatter_dataframe(df):
    """
    Produce a scatter plot from a pandas DataFrame containing vectors.

    Parameters
    ----------
    df : pandas DataFrame
        Elements are vectors as 1D numpy arrays.
        Column names are used in the legend.

    """
    for column, values in df.items():
        points = np.vstack(values)

        scatter2(points)

    plt.legend(df.columns)


def scatter_series(series):
    """
    Plot a pandas series containing vectors.

    Parameters
    ----------
    series : pandas Series
        Values are vectors as 1D numpy arrays.
        Index names are used in the legend.

    """
    # Array of points
    points = np.vstack(series)

    for point in points:

        plt.scatter(point[0], point[1])

    plt.legend(series.index)


def plot_foot_peaks(foot_dist, peak_frames):
    """
    Plot the peaks in the foot distance signal.

    Parameters
    ----------
    foot_dist : ndarray
        Foot distance signal.
    peak_frames : iterable
        Sequence of frames where a peak occurs.

    """
    _, ax = plt.subplots()

    ax.plot(foot_dist, color='k', linewidth=0.7)

    ax.vlines(x=peak_frames, ymin=0, ymax=foot_dist.max(), colors='r')

    plt.xlabel('Frame number')
    plt.ylabel('Distance between feet [cm]')


def compare_measurements(x, y, **kwargs):
    """
    Scatter plot of measurements from two devices.

    Straight line shows ideal results (when measurements are equal).

    Parameters
    ----------
    x : array_like
        Measurements of device A.
    y : array_like
        Measurements of device B.
    **kwargs
        Keyword arguments for scatter plot.

    """
    _, ax = plt.subplots()

    ax.scatter(x, y, **kwargs)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]

    # Plot both limits against each other
    ax.plot(lims, lims, 'k-')


def plot_bland_altman(means, diffs, bias, limits, percent=True):
    """
    Produce a Bland-Altman plot.

    Parameters
    ----------
    means : array_like
        Means of measurements from devices A and B.
    diffs : array_like
        Differences of measurements.
    bias : {int, float}
            Mean of the differences.
    limits : tuple
        Tuple of form (lower_limit, upper_limit).
        Bias minus/plus 1.96 standard deviations.
    percent : bool, optional
            If True, the y label shows percent difference.
            If False (default) the y label shows regular difference.

    """
    lower_limit, upper_limit = limits

    plt.scatter(means, diffs, c='black', s=5)

    plt.axhline(y=bias, color='k', linestyle='-')
    plt.axhline(y=lower_limit, color='k', linestyle='--')
    plt.axhline(y=upper_limit, color='k', linestyle='--')

    plt.xlabel('Mean of measurements')

    if percent:
        plt.ylabel('Percent difference between measurements')
    else:
        plt.ylabel('Difference between measurements')

    plt.annotate('Bias', xy=(120, bias - 2))
    plt.annotate('Lower limit', xy=(120, lower_limit - 2))
    plt.annotate('Upper limit', xy=(120, upper_limit - 2))
