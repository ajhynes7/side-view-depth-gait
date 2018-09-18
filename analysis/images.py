"""Module for working with images."""

import numpy as np


def focal_length(resolution, field_of_view):
    """
    Calculate focal length of camera from image resolution and field of view.

    Parameters
    ----------
    resolution : int
        x or y resolution in pixels.
    field_of_view : {int, float}
        Field of camera view in degrees.

    Returns
    -------
    float
        Focal length in pixels.

    Examples
    --------
    >>> f_x, f_y = focal_length(640, 57), focal_length(480, 43)

    >>> np.round(f_x, 2)
    589.37

    >>> np.round(f_y, 2)
    609.28

    """
    return 0.5 * resolution / np.tan(0.5 * np.deg2rad(field_of_view))


def image_to_real(position_image, x_res, y_res, f_x, f_y):
    """
    Convert image coordinates to real world coordinates.

    Parameters
    ----------
    position_image : array_like
        Position in image coordinates.
    x_res, y_res : int
        Resolution of image for x and y in pixels
    f_x, f_y : {float, int}
        Focal lengths for x and y in pixels.

    Returns
    -------
    ndarray
        Position in real world coordinates.

    Examples
    --------
    >>> position_image = [477, 348, 334.1]
    >>> x_res, y_res = 640, 480
    >>> fov_x, fov_y = 57, 43

    >>> f_x, f_y = focal_length(x_res, fov_x), focal_length(y_res, fov_y)

    >>> position_real = image_to_real(position_image, x_res, y_res, f_x, f_y)

    >>> np.round(position_real, 2)
    array([ 89.  , -59.22, 334.1 ])

    """
    # Coordinates using top left corner of image as origin
    x_image, y_image, z = position_image

    # Coordinates using centre of image as origin
    x_view = x_image - 0.5 * x_res
    y_view = 0.5 * y_res - y_image

    x_real = x_view / f_x * z
    y_real = y_view / f_y * z

    return np.array([x_real, y_real, z])


def real_to_image(position_real, x_res, y_res, f_x, f_y):
    """
    Convert real world coordinates to image coordinates.

    Parameters
    ----------
    position_real : array_like
        Position in image coordinates.
    x_res, y_res : int
        Resolution of image for x and y in pixels
    f_x, f_y : {float, int}
        Focal lengths for x and y in pixels.

    Returns
    -------
    ndarray
        Position in real world coordinates.

    Examples
    --------
    >>> position_real = [89, -59.22, 334.1]
    >>> x_res, y_res = 640, 480
    >>> fov_x, fov_y = 57, 43

    >>> f_x, f_y = focal_length(x_res, fov_x), focal_length(y_res, fov_y)

    >>> position_image = real_to_image(position_real, x_res, y_res, f_x, f_y)

    >>> np.round(position_image, 2)
    array([477. , 348. , 334.1])

    """
    x_real, y_real, z = position_real

    # Coordinates using centre of image as origin
    x_view = x_real * f_x / z
    y_view = y_real * f_y / z

    # Coordinates using top left corner of image as origin
    x_image = x_view + 0.5 * x_res
    y_image = 0.5 * y_res - y_view

    return np.array([x_image, y_image, z])


def recalibrate_positions(positions_real_old, x_res_old, y_res_old,
                          x_res, y_res, f_x, f_y):

    positions_real = np.full(positions_real_old.shape, np.nan)

    for i, pos_real_old in enumerate(positions_real_old):

        pos_image = real_to_image(pos_real_old, x_res_old, y_res_old, f_x, f_y)
        positions_real[i] = image_to_real(pos_image, x_res, y_res, f_x, f_y)

    return positions_real
