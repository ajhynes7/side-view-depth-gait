"""Functions for plotting points and visualizing results."""

import numpy as np
import matplotlib.pyplot as plt

import modules.linear_algebra as lin


def scatter_labels(points, labels):
    """
    Scatter points that are coloured by label.

    Parameters
    ----------
    points : ndarray
        (n, 2) array of n points with dimension 2.
    labels : ndarray
        (n, ) array of point labels.

    """
    for label in np.unique(labels):

        points_label = points[labels == label]

        plt.scatter(points_label[:, 0], points_label[:, 1])


def scatter2(points, **kwargs):
    """
    Produce a 2D scatter plot.

    Parameters
    ----------
    points : ndarray
        (n, 2) array of n points in two dimensions.
    kwargs : dict, optional
        Additional keywords passed to `scatter`.

    """
    if points.ndim == 1:
        # Convert to 2d array
        points = points.reshape(1, -1)

    plt.scatter(points[:, 0], points[:, 1], **kwargs)


def scatter3(ax, points, **kwargs):
    """
    Produce a 3D scatter plot.

    Parameters
    ----------
    ax : Axes3D object
        Axis for plotting.
    points : ndarray
        (n, 3) array of n points in three dimensions.
        One-dimensional array with shape (3, ) also allowed.
    kwargs : dict, optional
        Additional keywords passed to `scatter`.

    """
    if points.ndim == 1:
        # Convert to 2d array
        points = points.reshape(1, -1)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **kwargs)


def scatter_order(points, order):
    """
    Produce a scatter plot using a specified order of the (x, y) coordinates.

    Parameters
    ----------
    points : ndarray
        (n, 2) array of n points with dimension 2.
    order : array_like
        Order of the coordinates

    """
    plt.scatter(points[:, order[0]], points[:, order[1]])


def scatter_series(series):
    """
    Produce a scatter plot from a pandas Series.

    The index is used as the x-values.

    Parameters
    ----------
    series : Series
        Input pandas Series.

    """
    plt.scatter(series.index, series)


def plot_plane(ax, point, normal, *, x_range=range(10), y_range=range(10),
               **kwargs):
    """
    Plot a 3D plane.

    Parameters
    ----------
    ax : Axes3D object
        Axis for plotting.
    point : ndarray
        Point on plane.
    normal : ndarray
        Normal vector of plane.
    x_range, y_range : iterable, optional
        Range of x and y used to plot the plane (default range(10))
    kwargs: dict, optional
        Additional keywords passed to `plot_surface`.

    """
    a, b, c, d = lin.plane_coefficients(point, normal)

    x_grid, y_grid = np.meshgrid(x_range, y_range)

    z_grid = (-a * x_grid - b * y_grid - d) / c

    ax.plot_surface(x_grid, y_grid, z_grid, **kwargs)


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
    kwargs : dict, optional
        Additional keywords passed to `plot_surface`.

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
