"""Module for working with images."""
import numpy as np


def real_to_proj(point_real, x_res, y_res, f_xz, f_yz):
    """
    Convert real world coordinates to projected coordinates.

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
    point_proj : ndarray
        Point in projected coordinates.

    Examples
    --------
    >>> point_real = [10, 5, 3]
    >>> x_res, y_res = 640, 480
    >>> f_xz, f_yz = 1.11146664619446, 0.833599984645844

    >>> point_proj = real_to_proj(point_real, x_res, y_res, f_xz, f_yz)

    >>> np.round(point_proj, 2)
    array([2239.39, -719.69,    3.  ])

    """
    f_coeff_x = x_res / f_xz
    f_coeff_y = y_res / f_yz

    x_real, y_real, z_real = point_real

    x_proj = f_coeff_x * x_real / z_real + 0.5 * x_res

    y_proj = 0.5 * y_res - f_coeff_y * y_real / z_real

    z_proj = z_real

    point_proj = np.array([x_proj, y_proj, z_proj])

    return point_proj


def proj_to_real(point_proj, x_res, y_res, f_xz, f_yz):
    """
    Convert projected coordinates to real world coordinates.

    Parameters
    ----------
    point_proj : array_like
        Point in projected coordinates.
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
    >>> point_proj = [2239.39, -719.69, 3]
    >>> x_res, y_res = 640, 480
    >>> f_xz, f_yz = 1.11146664619446, 0.833599984645844

    >>> point_real = proj_to_real(point_proj, x_res, y_res, f_xz, f_yz)

    >>> np.round(point_real)
    array([10.,  5.,  3.])

    """
    x_proj, y_proj, z_proj = point_proj

    f_normalized_x = x_proj / x_res - 0.5
    f_normalized_y = 0.5 - y_proj / y_res

    x_real = f_normalized_x * z_proj * f_xz
    y_real = f_normalized_y * z_proj * f_yz

    z_real = z_proj

    point_real = np.array([x_real, y_real, z_real])

    return point_real


def image_coords_to_real(x_res, y_res, f_xz, f_yz, img, x, y):
    """
    Return real world coordinates from a pixel position on a depth image.

    Parameters
    ----------
    img : ndarray
        Input image.
    x_res, y_res : int
        Resolution of image in x and y axes.
    f_xz, f_yz : {float, int}
        Conversion factors for x and y.
    x, y : int
        Coordinates of pixel on image.

    Returns
    -------
    point_real : ndarray
        Point in real world coordinates.

    Examples
    --------
    >>> img = np.array([[10, 2], [3, 4]])
    >>> x, y = 0, 1
    >>> x_res, y_res = 640, 480
    >>> f_xz, f_yz = 1.11146664619446, 0.833599984645844

    >>> point_real = image_coords_to_real(x_res, y_res, f_xz, f_yz, img, x, y)
    >>> np.round(point_real, 2)
    array([-1.67,  1.25,  3.  ])

    """
    z = img[y, x]
    point_proj = [x, y, z]

    point_real = proj_to_real(point_proj, x_res, y_res, f_xz, f_yz)

    return point_real


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
    # Number of rows and columns in the output image.
    img_shape = np.max(points[:, [1, 0]], axis=0) + 1

    img = np.full(img_shape, np.nan)

    for point in points:

        x, y, z = point
        img[y, x] = z

    return img


if __name__ == "__main__":

    import doctest
    doctest.testmod()
