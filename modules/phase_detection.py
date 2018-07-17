"""Module for detecting the phases of a foot during a walking pass."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import modules.signals as sig
import modules.numpy_funcs as nf
import modules.pandas_funcs as pf
import modules.iterable_funcs as itf
import modules.linear_algebra as lin


def frames_of_interest(foot_signal):
    """
    Return frames of interest from a foot signal.

    These frames are peaks and troughs in the foot signal.

    Parameters
    ----------
    foot_signal : Series
        Signal from foot data that resembles a sinusoid.

    Returns
    -------
    frames_interest : ndarray
        Sorted array of frames.

    """
    signal_1 = sig.normalize(foot_signal)
    signal_2 = 1 - signal_1

    rms_1 = sig.root_mean_square(signal_1)
    rms_2 = sig.root_mean_square(signal_2)

    frames_1 = sig.detect_peaks(signal_1, window_length=3, min_height=rms_1)
    frames_2 = sig.detect_peaks(signal_2, window_length=3, min_height=rms_2)

    frames_interest = np.sort(np.append(frames_1, frames_2))

    return frames_interest


def get_step_signal(direction_pass, foot_series_pass):
    """
    Return a signal that resembles multiple upwards steps.

    This is achieved by representing foot points as 1D values along the
    direction of the walking pass.

    The signal is used to detect the stance and swing phases of the foot.

    Parameters
    ----------
    direction_pass : ndarray
        Vector for direction of a walking pass.
    foot_series_pass : Series
        Positions of a foot during a walking pass.
        Index values are frames.
        Values are foot positions.

    Returns
    -------
    step_signal : Series
        Signal with multiple steps.
        Index values are frames.

    """
    line_point = np.array([0, 0, 0])
    points = np.stack(foot_series_pass)

    x_values = lin.line_coordinate_system(line_point, direction_pass, points)

    step_signal = pd.Series(x_values, index=foot_series_pass.index)

    return step_signal


def detect_phases(step_signal, frames_interest):
    """
    Return the phase (stance/swing) of each frame in a walking pass.

    Parameters
    ----------
    step_signal : Series
        Signal with multiple steps.
        Index values are frames.
    frames_interest : ndarray
        Sorted array of frames.

    Returns
    -------
    frame_phases : Series
        Indicates the walking phase of the corresponding frames.
        Each element is either 'stance' or 'swing'.

    """
    frames = step_signal.index.values

    split_labels = nf.label_by_split(frames, frames_interest)
    sub_signals = list(nf.group_by_label(step_signal, split_labels))

    variances = [*map(np.var, sub_signals)]
    variance_array = np.array(variances).reshape(-1, 1)

    k_means = KMeans(n_clusters=2, random_state=0).fit(variance_array)
    variance_labels = k_means.labels_

    sub_signal_lengths = [*map(len, sub_signals)]
    expanded_labels = [*itf.repeat_by_list(variance_labels,
                                           sub_signal_lengths)]

    stance_label = np.argmin(k_means.cluster_centers_)
    swing_label = 1 - stance_label
    phase_dict = {stance_label: 'stance', swing_label: 'swing'}

    phase_strings = itf.map_with_dict(expanded_labels, phase_dict)
    frame_phases = pd.Series(phase_strings, index=frames)

    return frame_phases


def get_phase_dataframe(frame_phases):
    """
    Return a DataFrame displaying the phase and phase number of each frame.

    The phase number is a count of the phase occurence
    (e.g., stance 0, 1, ...).

    Parameters
    ----------
    frame_phases : Series
        Indicates the walking phase of the corresponding frames.
        Each element is either 'stance' or 'swing'.

    Returns
    -------
    df_phase : DataFrame
        Index is 'frame'.
        Columns are 'phase', 'number'.

    """
    df_phase = pd.DataFrame({'phase': frame_phases}, dtype='category')
    df_phase.index.name = 'frame'

    phase_strings = frame_phases.values
    phase_labels = np.array([*itf.label_repeated_elements(phase_strings)])

    is_stance = df_phase.phase == 'stance'
    is_swing = df_phase.phase == 'swing'

    stance_labels = [*itf.label_repeated_elements(phase_labels[is_stance])]
    swing_labels = [*itf.label_repeated_elements(phase_labels[is_swing])]

    frames = frame_phases.index
    stance_series = pd.Series(stance_labels, index=frames[is_stance])
    swing_series = pd.Series(swing_labels, index=frames[is_swing])

    df_phase['number'] = pd.concat([stance_series, swing_series])

    return df_phase


def foot_phases(frames_interest, direction_pass, foot_series_pass):
    """
    Return a DataFrame with stride phases for one foot during a walking pass.

    Parameters
    ----------
    frames_interest : ndarray
        Sorted array of frames.
    direction_pass : ndarray
        Vector for direction of a walking pass.
    foot_series_pass : Series
        Positions of a foot during a walking pass.
        Index values are frames.
        Values are foot positions.

    Returns
    -------
    df_phase : DataFrame
        Index is 'frame'.
        Columns are 'phase', 'number', 'position'.

    """
    step_signal = get_step_signal(direction_pass, foot_series_pass)
    frame_phases = detect_phases(step_signal, frames_interest)

    df_phase = get_phase_dataframe(frame_phases)
    df_phase['position'] = foot_series_pass

    return df_phase


def group_stance_frames(df_phase, suffix):

    df_stance = df_phase[df_phase.phase == 'stance'].reset_index()

    column_funcs = {'frame': list, 'position': np.stack}
    df_grouped = pf.apply_to_grouped(df_stance, 'number', column_funcs)

    df_grouped.frame = df_grouped.frame.apply(np.median)
    df_grouped.index = df_grouped.index.astype('str') + suffix

    return df_grouped
