"""Module for working with images."""

import numpy as np
from numpy import ndarray

from modules.typing import array_like


def image_to_real(point_image: array_like, x_res: int, y_res: int, f_xz: float, f_yz: float) -> ndarray:
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


def real_to_image(point_real: array_like, x_res: int, y_res: float, f_xz: float, f_yz: float) -> ndarray:
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


def rgb_to_label(image_rgb: ndarray, rgb_vectors: array_like) -> ndarray:
    """
    Convert an RGB image to a label image.

    Parameters
    ----------
    image_rgb : ndarray
        (n_rows, n_cols, 3) image
    rgb_vectors : array_like
        Each element is a [R, G, B] vector.

    Returns
    -------
    label_image : ndarray
        (n_rows, n_cols) image.
        2D label image.

    """
    label_image = np.zeros(image_rgb.shape[:-1])

    for i, rgb_vector in enumerate(rgb_vectors):

        mask = np.all(image_rgb == rgb_vector, axis=-1)
        label_image[mask] = i + 1

    return label_image


def recalibrate_positions(
    positions_real_orig: ndarray, x_res_orig: int, y_res_orig: int, x_res: int, y_res: int, f_xz: float, f_yz: float
) -> ndarray:
    """
    Change real world coordinates using new camera calibration parameters.

    Parameters
    ----------
    positions_real_orig : ndarray
        Original positions in real world coordinates.
    x_res_orig, y_res_orig : int
        Original image resolutions.
    x_res, y_res : int
        New image resolutions.
    f_xz, f_yz : int
        Focal parameters.

    Returns
    -------
    positions_real : ndarray
        Positions in new real world coordinates.

    """
    positions_real = np.full(positions_real_orig.shape, np.nan)

    for i, pos_real_orig in enumerate(positions_real_orig):

        pos_image = real_to_image(pos_real_orig, x_res_orig, y_res_orig, f_xz, f_yz)
        positions_real[i] = image_to_real(pos_image, x_res, y_res, f_xz, f_yz)

    return positions_real


# Camera calibration parameters.

# While the depth image files are 640 x 480, there is a large border in the
# image, resulting in a smaller real resolution.
# This needs to be taken into account when converting between
# image coordinates and real world coordinates.

# Original resolutions of depth images
X_RES_ORIG, Y_RES_ORIG = 640, 480

# Estimates of actual resolutions
X_RES, Y_RES = 565, 430

# Coefficients used to convert between image and real coordinates
# calculated for the Kinect v1
F_XZ, F_YZ = 1.11146664619446, 0.833599984645844
