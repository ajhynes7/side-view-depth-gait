"""Module for working with images."""

import numpy as np


def image_to_real(point_image, x_res, y_res, f_xz, f_yz):
    """
    Convert image coordinates to real world coordinates.

    Parameters
    ----------
    point_image : array_like
        Point in image coordinates.
    x_res, y_res : int
        Resolution of image in x and y axes.
    f_xz, f_yz : {float, int}
        Conversion factors for x and y.

    Returns
    -------
    point_real : ndarray
        Point in real world coordinates.

    Examples
    --------
    >>> point_image = [2239.39, -719.69, 3]
    >>> x_res, y_res = 640, 480
    >>> f_xz, f_yz = 1.11146664619446, 0.833599984645844

    >>> point_real = image_to_real(point_image, x_res, y_res, f_xz, f_yz)

    >>> np.round(point_real)
    array([10.,  5.,  3.])

    """
    x_image, y_image, z_image = point_image

    f_normalized_x = x_image / x_res - 0.5
    f_normalized_y = 0.5 - y_image / y_res

    x_real = f_normalized_x * z_image * f_xz
    y_real = f_normalized_y * z_image * f_yz
    z_real = z_image

    point_real = np.array([x_real, y_real, z_real])

    return point_real


def real_to_image(point_real, x_res, y_res, f_xz, f_yz):
    """
    Convert real world coordinates to image coordinates.

    Parameters
    ----------
    point_real : array_like
        Point in real world coordinates.
    x_res, y_res : int
        Resolution of image in x and y axes.
    f_xz, f_yz : {float, int}
        Conversion factors for x and y.

    Returns
    -------
    point_image : ndarray
        Point in image coordinates.

    Examples
    --------
    >>> point_real = [10, 5, 3]
    >>> x_res, y_res = 640, 480
    >>> f_xz, f_yz = 1.11146664619446, 0.833599984645844

    >>> point_image = real_to_image(point_real, x_res, y_res, f_xz, f_yz)

    >>> np.round(point_image, 2)
    array([2239.39, -719.69,    3.  ])

    """
    f_coeff_x = x_res / f_xz
    f_coeff_y = y_res / f_yz

    x_real, y_real, z_real = point_real

    x_image = f_coeff_x * x_real / z_real + 0.5 * x_res
    y_image = 0.5 * y_res - f_coeff_y * y_real / z_real
    z_image = z_real

    point_image = np.array([x_image, y_image, z_image])

    return point_image


def recalibrate_positions(positions_real_old, x_res_old, y_res_old, x_res,
                          y_res, f_xz, f_yz):

    positions_real = np.full(positions_real_old.shape, np.nan)

    for i, pos_real_old in enumerate(positions_real_old):

        pos_image = real_to_image(pos_real_old, x_res_old, y_res_old, f_xz,
                                  f_yz)
        positions_real[i] = image_to_real(pos_image, x_res, y_res, f_xz, f_yz)

    return positions_real
