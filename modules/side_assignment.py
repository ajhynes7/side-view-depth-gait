"""Module for assigning left/right sides to the feet."""

import numpy as np
from numpy.linalg import norm
from scipy.signal import find_peaks
from skimage.measure import LineModelND, ransac
from sklearn.preprocessing import normalize
from skspatial.transformation import transform_coordinates
from statsmodels.robust import mad

import modules.motion_correspondence as mc
import modules.numpy_funcs as nf


def fit_points_pass(points_pass):

    model, is_inlier = ransac(
        points_pass, LineModelND, min_samples=int(0.9 * len(points_pass)), residual_threshold=3 * min(mad(points_pass))
    )

    return model, is_inlier


def split_walking_pass(points_a, points_b):
    """Split the walking pass between local minima in the foot distance signal."""

    distances_foot = norm(points_a - points_b, axis=1)
    distances_normalized = normalize(distances_foot.reshape(-1, 1), norm='max', axis=0).flatten()

    indices_min, _ = find_peaks(1 - distances_normalized, prominence=0.5, width=1)

    n_points = points_a.shape[0]
    labels_sections = nf.label_by_split(indices_min, n_points)

    return labels_sections


def assign_sides_section(points_a, points_b, basis):
    """

    The coordinate system is:
        X - Along Zeno Walkway
        Y - Height
        Z - Depth from Kinect camera

    Examples
    --------
    >>> points_a = np.array([[0, 0, 200], [1, 0, 200], [2, 0, 200], [3, 0, 200]])
    >>> points_b = np.array([[0, 0, 220], [1, 0, 220], [2, 0, 220], [3, 0, 220]])

    >>> points_l, points_r = assign_sides_section(points_a, points_b, [1, 0, 0])

    >>> points_l
    array([[  0,   0, 220],
           [  1,   0, 220],
           [  2,   0, 220],
           [  3,   0, 220]])

    >>> points_r
    array([[  0,   0, 200],
           [  1,   0, 200],
           [  2,   0, 200],
           [  3,   0, 200]])

    The sides switch if the walking direction switches.

    >>> points_a[:, 0] *= -1
    >>> points_b[:, 0] *= -1

    >>> points_l, points_r = assign_sides_section(points_head, points_a, points_b)

    >>> points_l
    array([[  0,   0, 200],
           [ -1,   0, 200],
           [ -2,   0, 200],
           [ -3,   0, 200]])

    >>> points_r
    array([[  0,   0, 220],
           [ -1,   0, 220],
           [ -2,   0, 220],
           [ -3,   0, 220]])

    """
    # Assume side A is left, B is right.
    points_l, points_r = points_a, points_b

    values_side_l = transform_coordinates(points_a, basis.origin, [basis.perp])
    values_side_r = transform_coordinates(points_b, basis.origin, [basis.perp])

    value_side_l = np.median(values_side_l)
    value_side_r = np.median(values_side_r)

    if value_side_l > value_side_r:

        points_l, points_r = points_r, points_l

    return points_l, points_r


def assign_sides_pass(frames, points_a, points_b, basis):
    """Assign left/right sides to feet in a walking pass."""

    labels_sections = split_walking_pass(points_a, points_b)

    # Mark small sections with label of -1.
    labels_sections = nf.filter_labels(labels_sections, min_elements=3)
    is_noise = labels_sections == -1

    points_stacked = np.dstack((points_a, points_b))

    def yield_assigned_sections():
        """Yield each section of a walking pass with foot points assigned to left/right."""

        labels_unique = np.unique(labels_sections[~is_noise])

        for label in labels_unique:

            is_section = labels_sections == label

            points_stacked_section = points_stacked[is_section]

            assignment = mc.correspond_motion(points_stacked_section, [0, 1])
            points_stacked_assigned = mc.assign_points(points_stacked_section, assignment)

            points_section_a = points_stacked_assigned[:, :, 0]
            points_section_b = points_stacked_assigned[:, :, 1]

            points_section_l, points_section_r = assign_sides_section(points_section_a, points_section_b, basis)

            yield points_section_l, points_section_r

    list_points_l, list_points_r = zip(*yield_assigned_sections())

    points_l = np.vstack(list_points_l)
    points_r = np.vstack(list_points_r)

    frames_lr = frames[~is_noise]

    return frames_lr, points_l, points_r
