"""
Module for calculating gait metrics from 3D body part positions
of a walking person.

"""
import numpy as np
from numpy.linalg import norm
import pandas as pd

import modules.general as gen
import modules.linear_algebra as lin
from modules.signals import mean_shift_peaks, root_mean_filter


class Stride:

    def __init__(self, stance, swing_i, swing_f):

        self.swing_i, self.swing_f = swing_i, swing_f
        self.stance = stance

        self.stance_proj = lin.proj_point_line(self.stance, self.swing_i,
                                               self.swing_f)

        assert Stride.is_stride(stance, swing_i, swing_f)

    def __str__(self):

        string = "Stride(frame_i={self.frame_i}, frame_f={self.frame_f})"

        return string.format(self=self)

    @staticmethod
    def is_stride(stance, swing_i, swing_f):

        swing_same_side = swing_i.side == swing_f.side

        swing_consec = swing_i.contact_number == swing_f.contact_number - 1

        same_contact = swing_i.contact_number == stance.contact_number

        swing_stance_diff_side = swing_i.side != stance.side

        tests = [swing_same_side, swing_consec, same_contact,
                 swing_stance_diff_side]

        return np.all(tests)

    @property
    def stride_length(self):

        return norm(self.swing_f - self.swing_i)

    @property
    def stride_time(self):

        return (self.frame_f - self.frame_i) / 30

    @property
    def stride_velocity(self):

        return self.stride_length / self.stride_time

    @property
    def step_length(self):

        return norm(self.stance_proj - self.swing_i)

    @property
    def stride_width(self):

        return norm(self.stance_proj - self.stance)

    @property
    def absolute_step_length(self):

        return norm(self.stance - self.swing_i)


def split_by_pass(df, frame_labels):
    """
    Split a DataFrame into separate DataFrames for each walking pass.
    The new DataFrames are ordered by image frame number.

    Parameters
    ----------
    df : pandas DataFrame
        Index contains image frames.
    frame_labels : ndarray
        Label of each image frame.
        Label indicates the walking pass.

    Returns
    -------
    pass_dfs : list
        List containing DataFrame for each walking pass.

    """
    # Put labels in order so that walking pass
    # DataFrames will be ordered by frame.
    frame_labels = np.array(gen.map_sort(frame_labels))

    pass_dfs = [df[frame_labels == i] for i in np.unique(frame_labels)]

    return pass_dfs


def foot_contacts(df_pass, direction_pass):
    """
    Estimate the frames where foot makes contact with floor.

    Separate arrays are returned for left and right feet.

    Parameters
    ----------
    df_pass : pandas DataFrame
        DataFrame for walking pass.
        Columns must include 'L_FOOT', 'R_FOOT'.

    direction_pass : ndarray
        Direction of motion for walking pass.

    Returns
    -------
    df_pass_contact : pandas DataFrame
        Columns are 'L_FOOT', 'R_FOOT'.
        Index is step number (0, 1, 2, ...).
        Elements are frames where contact occurs.

    """
    right_to_left = df_pass.L_FOOT - df_pass.R_FOOT

    projections_l = right_to_left.apply(np.dot, args=(direction_pass,))
    projections_r = -projections_l

    contacts_l, _ = mean_shift_peaks(root_mean_filter(projections_l), r=10)
    contacts_r, _ = mean_shift_peaks(root_mean_filter(projections_r), r=10)

    df_peaks_l = pd.DataFrame(contacts_l, columns=['L_FOOT'])
    df_peaks_r = pd.DataFrame(contacts_r, columns=['R_FOOT'])

    df_pass_contact = df_peaks_l.join(df_peaks_r, how='outer')

    return df_pass_contact


def gait_dataframe(df, peak_frames, peak_labels, metrics_func):
    """
    Produces a pandas DataFrame containing gait metrics from a walking trial.

    Parameters
    ----------
    df : DataFrame
        Index is the frame numbers.
        Columns include 'HEAD', 'L_FOOT', 'R_FOOT'.
        Each element is a position vector.
    peak_frames : array_like
        Array of all frames with a detected peak in the foot distance data.
    peak_labels : dict
        Label of each peak frame.
        The labels are determined by clustering the peak frames.

    Returns
    -------
    gait_df : DataFrame
        Index is final peak frame used to calculate gait metrics.
        Columns are gait metric names.

    """
    gait_list, frame_list = [], []

    for frame_i, frame_f in gen.pairwise(peak_frames):

        if peak_labels[frame_i] == peak_labels[frame_f]:

            metrics = metrics_func(df, frame_i, frame_f)

            gait_list.append(metrics)
            frame_list.append(frame_f)

    gait_df = pd.DataFrame(gait_list, index=frame_list)
    gait_df.index.name = 'Frame'

    return gait_df
