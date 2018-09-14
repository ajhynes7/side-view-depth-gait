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

    x_view = x_real * f_x / z
    y_view = y_real * f_y / z

    x_image = x_view + 0.5 * x_res
    y_image = 0.5 * y_res - y_view

    return np.array([x_image, y_image, z])


def image_to_points(img):
    """
    Convert an image to a list of points.

    Each point represents a pixel in the image.

    The (x, y) image position serves as the x and y coordinates of the point.
    The image value at position (x, y) is the z coordinate of the point.

    Parameters
    ----------
    img : ndarray
        Input image.

    Returns
    -------
    points : ndarray
        (n, 3) array of n points.

    Examples
    --------
    >>> img = np.array([[10, 2], [3, 4]])

    >>> image_to_points(img)
    array([[ 0.,  0., 10.],
           [ 0.,  1.,  3.],
           [ 1.,  0.,  2.],
           [ 1.,  1.,  4.]])

    """
    n_rows, n_cols = img.shape
    points = np.full((img.size, 3), np.nan)

    count = 0

    for x in range(n_cols):
        for y in range(n_rows):
            z = img[y, x]
            points[count] = [x, y, z]

            count += 1

    return points


def points_to_image(points):
    """
    Convert a list of points to an image.

    The x and y coordinates of a point correspond to the (x, y) image position.
    The z coordinate of the point is the value of the (x, y) image position.

    Parameters
    ----------
    points : ndarray
        (n, 3) array of n points.

    Returns
    -------
    img : ndarray
        Output image.

    Examples
    --------
    >>> points = np.array([[0, 0, 10], [0, 1, 3], [1, 0, 2], [1, 1, 4]])

    >>> points_to_image(points)
    array([[10.,  2.],
           [ 3.,  4.]])

    """
    z_values = points[:, -1]  # Save the float z-values

    # Convert to int so that x and y can be indices to the image array
    points = points.astype(int)

    img_shape = np.max(points[:, [1, 0]], axis=0) + 1
    img = np.full(img_shape, np.nan)

    for i, _ in enumerate(points):
        x, y = points[i, :2]
        z = z_values[i]

        img[y, x] = z

    return img
