"""Module for assigning left/right sides to the feet."""

import numpy as np
from dpcontracts import require, ensure
from numpy.linalg import norm
from scipy.signal import find_peaks
from sklearn.preprocessing import normalize

import modules.motion_correspondence as mc


def split_walking_pass(frames, points_a, points_b):
    """Split the walking pass between local minima in the foot distance signal."""

    distances_foot = norm(points_a - points_b, axis=1)
    distances_normalized = normalize(distances_foot.reshape(-1, 1), norm='max', axis=0).flatten()

    indices_min, _ = find_peaks(1 - distances_normalized, prominence=0.5, width=1)

    points_stacked = np.dstack((points_a, points_b))

    # Split into sections between the detected peaks and remove sections that are too short.
    # The sections should have at least three frames so motion correspondence algorithm can be run.
    list_points = [x for x in np.split(points_stacked, indices_min) if len(x) >= 3]
    list_frames = [x for x in np.split(frames, indices_min) if len(x) >= 3]

    return list_frames, list_points


@require("The points must be 2D.", lambda args: all(x.shape[1] == 2 for x in [args.points_a, args.points_b]))
def assign_sides_pass(frames, points_a, points_b):
    """Assign left/right sides to feet in a walking pass."""

    list_frames, list_points = split_walking_pass(frames, points_a, points_b)

    def yield_assigned_sections():
        """Yield each section of a walking pass with foot points assigned to left/right."""

        for points_stacked_section in list_points:

            assignment = mc.correspond_motion(points_stacked_section, [0, 1])
            points_stacked_assigned = mc.assign_points(points_stacked_section, assignment)

            points_section_l = points_stacked_assigned[:, :, 0]
            points_section_r = points_stacked_assigned[:, :, 1]

            value_side_l = np.median(points_section_l[:, 0])
            value_side_r = np.median(points_section_r[:, 0])

            if value_side_l > value_side_r:

                points_section_l, points_section_r = points_section_r, points_section_l

            points_stacked_section_lr = np.dstack((points_section_l, points_section_r))

            yield points_stacked_section_lr

    points_stacked_lr = np.vstack([*yield_assigned_sections()])
    frames_lr = np.concatenate(list_frames)

    points_l = points_stacked_lr[:, :, 0]
    points_r = points_stacked_lr[:, :, 1]

    return frames_lr, points_l, points_r
