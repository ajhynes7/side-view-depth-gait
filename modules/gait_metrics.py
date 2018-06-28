"""
Module for calculating gait metrics from 3D body part positions
of a walking person.

"""
import numpy as np
from numpy.linalg import norm
import pandas as pd

import modules.general as gen
import modules.linear_algebra as lin
import modules.pose_estimation as pe
import modules.pandas_funcs as pf
from modules.signals import mean_shift_peaks, root_mean_filter


class Stride:

    def __init__(self, swing_i, stance, swing_f):

        self.swing_i, self.swing_f = swing_i, swing_f
        self.stance = stance

        pos_p = stance.position
        pos_a, pos_b = swing_i.position, swing_f.position

        self.stance_proj = lin.proj_point_line(pos_p, pos_a, pos_b)

    def __str__(self):

        string = "Stride(side={self.side}, number={self.number})"

        return string.format(self=self)

    @property
    def side(self):

        return self.swing_i.side

    @property
    def number(self):

        return self.swing_i.number

    @property
    def step_length(self):

        return norm(self.stance_proj - self.swing_f.position)

    @property
    def step_time(self):

        return (self.swing_f.frame - self.stance.frame) / 30

    @property
    def stride_width(self):

        return norm(self.stance.position - self.stance_proj)

    @property
    def stride_length(self):

        return norm(self.swing_f.position - self.swing_i.position)

    @property
    def stride_time(self):

        return (self.swing_f.frame - self.swing_i.frame) / 30

    @property
    def stride_velocity(self):

        return self.stride_length / self.stride_time


def foot_contacts_to_gait(df_foot):
    """


    Parameters
    ----------
    df_foot :  DataFrame
        [description]

    Returns
    -------
    df_gait : DataFrame
        [description]

    """
    foot_tuples = df_foot.itertuples(index=False)

    property_dict = {}

    for i, foot_tuple in enumerate(gen.window(foot_tuples, n=3)):

        stride_instance = Stride(*foot_tuple)

        property_dict[i] = gen.get_properties(stride_instance)

    # By setting the orient, the keys of the dictionary become the index
    df_gait = pd.DataFrame.from_dict(property_dict, orient='index')

    return df_gait


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
    df_contact : pandas DataFrame
        Columns are 'number', 'part', 'frame'.
        Each row represents a frame when a foot contacts the floor.

    """
    right_to_left = df_pass.L_FOOT - df_pass.R_FOOT

    projections_l = right_to_left.apply(np.dot, args=(direction_pass,))
    projections_r = -projections_l

    contacts_l, _ = mean_shift_peaks(root_mean_filter(projections_l), r=10)
    contacts_r, _ = mean_shift_peaks(root_mean_filter(projections_r), r=10)

    df_peaks_l = pd.DataFrame(contacts_l, columns=['L_FOOT'])
    df_peaks_r = pd.DataFrame(contacts_r, columns=['R_FOOT'])

    df_joined = df_peaks_l.join(df_peaks_r, how='outer')

    # Reshape data to have one frame per row
    series_contact = df_joined.stack().sort_values().astype(int)

    df_contact = series_contact.reset_index()

    df_contact.columns = ['number', 'part', 'frame']

    return df_contact


def walking_pass_metrics(df_pass):
    """
    Calculate gait metrics from a single walking pass in front of the camera.

    Parameters
    ----------
    df_pass : DataFrame
        Index is the frame numbers.
        Columns must include L_FOOT', 'R_FOOT'.
        Elements are position vectors.

    Returns
    -------
    df_gait : DataFrame
        Each row represents a stride.
        Columns include gait metrics, e.g. 'stride_length', and the side and
        stride number.

    """
    # Enforce consistent sides for the feet on all walking passes.
    # Also calculate the general direction of motion for each pass.
    df_pass, direction = pe.consistent_sides(df_pass)

    # Estimate frames where foot contacts floor
    df_contact = foot_contacts(df_pass, direction)

    # Add column with corresponding foot positions
    df_contact = pf.column_from_lookup(df_contact, df_pass, column='position',
                                       lookup_cols=('frame', 'part'))

    # For simplicity, substitute the part 'R_FOOT' with a side 'R'
    df_contact['side'] = df_contact.part.str[0]
    df_contact = df_contact.drop('part', axis=1)

    df_gait = foot_contacts_to_gait(df_contact)

    return df_gait


def combine_walking_passes(pass_dfs):

    list_ = []
    for i, df_pass in enumerate(pass_dfs):

        df_gait = walking_pass_metrics(df_pass)
        df_gait['pass'] = i  # Add column to record the walking pass

        list_.append(df_gait)

    df_combined = pd.concat(list_)

    # Reset the index because there are repeated index elements
    df_combined = df_combined.reset_index(drop=True)

    # Split DataFrame by side (right and left)
    df_l, df_r = [x for _, x in df_combined.groupby('side')]

    df_final = pd.merge(df_l, df_r, how='outer', left_on='pass',
                        right_on='pass', suffixes=['_L', '_R'])

    strings_to_drop = ['side', 'pass', 'number']
    df_final = pf.drop_any_like(df_final, strings_to_drop, axis=1)

    return df_final


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
