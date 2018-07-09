"""
Module for calculating gait metrics from 3D body part positions.

Common Parameters
-----------------
df_pass : DataFrame
    DataFrame for one walking pass.
    Index values are frames.
    Columns must include 'L_FOOT', 'R_FOOT'.
    Elements are position vectors.
df_contact : DataFrame
    Each row represents a frame when a foot contacts the floor.
    Columns include 'number', 'side', 'frame'.
df_gait : DataFrame
    Each row represents a stride.
    Columns include gait metrics, e.g. 'stride_length', and the side and
    stride number.

"""
import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.signal import medfilt

import modules.general as gen
import modules.linear_algebra as lin
import modules.pose_estimation as pe
import modules.pandas_funcs as pf
import modules.signals as sig
import modules.math_funcs as mf


class Stride:

    def __init__(self, swing_i, stance, swing_f):

        self.swing_i, self.swing_f = swing_i, swing_f
        self.stance = stance

        pos_p = stance.position
        pos_a, pos_b = swing_i.position, swing_f.position

        self.stance_proj = lin.project_point_line(pos_p, pos_a, pos_b)

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


def foot_signal(foot_interest, foot_other, direction_pass):
    """
    Return a signal from foot data that is used to detect contact frames.

    Parameters
    ----------
    foot_interest, foot_other : Series
        Index values are frames.
        Values are foot positions.
        The first series is the foot of interest (left or right).
    direction_pass : ndarray
        Direction of motion for the walking pass.

    Returns
    -------
    signal : Series
        Signal from foot data.

    """
    vectors_to_foot = foot_interest - foot_other

    signal = vectors_to_foot.apply(np.dot, args=(direction_pass,))

    return signal


def detect_foot_contacts(signal, min_peak_height):
    """
    Detect frames where foot first contacts the floor.

    This is achieved by locating peaks in a foot signal.

    Parameters
    ----------
    signal : Series
        Signal from foot data.
    min_peak_height : {int, float}
        Minimum allowed height for signal peak.

    Returns
    -------
    contact_frames : ndarray
        Frames where foot of interest contacts the floor.

    """
    signal[:] = medfilt(signal)  # Apply a median filter to remove noise

    signal_upper = signal[signal > 0]

    norm_values = mf.normalize_array(signal_upper.values)

    contact_frames, _ = sig.mean_shift_peaks(signal_upper, masses=norm_values,
                                             kernel='gaussian', radius=5,
                                             eps=1e-3)

    # Filter the contact frames to remove false peaks
    contact_values = signal[contact_frames]
    contact_frames = contact_frames[contact_values > min_peak_height]

    return contact_frames


def join_foot_contacts(contacts_l, contacts_r):
    """
    Combine arrays of foot contact frames into a DataFrame object.

    Parameters
    ----------
    contacts_l, contacts_r : array_like
        Left and right contact frames.

    Returns
    -------
    df_contact : DataFrame
        Each row represents a first contact of a foot on the floor.
        Columns are 'number', 'side', and 'frame'.

    Examples
    --------
    >>> contacts_l = [214, 256]
    >>> contacts_r = [234]

    >>> join_foot_contacts(contacts_l, contacts_r)
       number side  frame
    0       0    L    214
    1       0    R    234
    2       1    L    256

    """
    df_l = pd.DataFrame(contacts_l, columns=['L'])
    df_r = pd.DataFrame(contacts_r, columns=['R'])

    df_joined = df_l.join(df_r, how='outer')

    # Reshape data to have one frame per row
    series_contact = df_joined.stack().sort_values().astype(int)

    df_contact = series_contact.reset_index()
    df_contact.columns = ['number', 'side', 'frame']

    return df_contact


def lookup_contact_positions(df_pass, df_contact, side_to_part):
    """
    Add foot positions at the contact frames.

    Parameters
    ----------
    df_pass, df_contact
        See module docstring.
    side_to_part : dict
        Dictionary mapping sides to body parts.
        e.g. {'L': 'L_FOOT', 'R': 'R_FOOT'}

    """
    parts = gen.map_with_dict(df_contact.side, side_to_part)
    frames = df_contact.frame

    df_contact['position'] = df_pass.lookup(frames, parts)


def foot_contacts_to_gait(df_contact):
    """
    Calculate gait metrics using instances when the feet contact the floor.

    Parameters
    ----------
    df_contact
        See module docstring.

    Returns
    -------
    df_gait
        See module docstring.

    """
    foot_tuples = df_contact.itertuples(index=False)

    property_dict = {}

    for i, foot_tuple in enumerate(gen.window(foot_tuples, n=3)):

        stride_instance = Stride(*foot_tuple)

        property_dict[i] = gen.get_properties(stride_instance)

    # By setting the orient, the keys of the dictionary become the index
    df_gait = pd.DataFrame.from_dict(property_dict, orient='index')

    return df_gait


def walking_pass_metrics(df_pass):
    """
    Calculate gait metrics from a single walking pass in front of the camera.

    Parameters
    ----------
    df_pass
        See module docstring.

    Returns
    -------
    df_gait
        See module docstring.

    """
    # Calculate the general direction of motion for each pass.
    line_point, direction_pass = pe.direction_of_pass(df_pass)

    # Enforce consistent sides for the feet on all walking passes.
    verified_sides = list(pe.verify_sides_pass(df_pass, direction_pass))
    df_cons = pe.enforce_consistency(df_pass, verified_sides)

    signal_l = foot_signal(df_cons.L_FOOT, df_cons.R_FOOT, direction_pass)
    signal_r = -signal_l

    min_peak_height = sig.root_mean_square(signal_l)

    contacts_l = detect_foot_contacts(signal_l, min_peak_height)
    contacts_r = detect_foot_contacts(signal_r, min_peak_height)

    df_contact = join_foot_contacts(contacts_l, contacts_r)

    # Add a column of foot positions
    side_to_part = {'L': 'L_FOOT', 'R': 'R_FOOT'}
    lookup_contact_positions(df_cons, df_contact, side_to_part)

    df_gait = foot_contacts_to_gait(df_contact)

    return df_gait


def combine_walking_passes(pass_dfs):
    """
    Combine gait metrics from all walking passes in a trial.

    Parameters
    ----------
    pass_dfs : list
        Each element is a df_pass (see module docstring).

    Returns
    -------
    df_final : DataFrame
        Each row represents a single stride.
        There can be multiple strides in a walking pass.
        Columns are gait metrics for left/right sides.

    """
    list_ = []
    for i, df_pass in enumerate(pass_dfs):

        df_gait = walking_pass_metrics(df_pass)
        df_gait['pass'] = i  # Add column to record the walking pass

        list_.append(df_gait)

    df_combined = pd.concat(list_, sort=True)

    # Reset the index because there are repeated index elements
    df_combined = df_combined.reset_index(drop=True)

    # Split the DataFrame by body side
    # There may be an empty DataFrame if one side has no values
    df_l = df_combined[df_combined.side == 'L']
    df_r = df_combined[df_combined.side == 'R']

    df_final = pd.merge(df_l, df_r, how='outer', left_on='pass',
                        right_on='pass', suffixes=['_L', '_R'])

    df_final = df_final.set_index('pass')   # Set the index to the pass column
    df_final = df_final.sort_index(axis=1)  # Sort the columns alphabetically

    strings_to_drop = ['side', 'number']
    df_final = pf.drop_any_like(df_final, strings_to_drop, axis=1)

    return df_final
