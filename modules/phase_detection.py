"""Module for detecting the phases of a foot during a walking pass."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import modules.iterable_funcs as itf
import modules.linear_algebra as lin
import modules.numpy_funcs as nf
import modules.pandas_funcs as pf
import modules.point_processing as pp
import modules.sliding_window as sw


def detect_phases(step_signal):
    """
    Detect the stance/swing phases of a foot during a walking pass.

    Parameters
    ----------
    step_signal : ndarray
        (n, ) array of values indicating the motion of one foot.

    Returns
    -------
    is_stance : ndarray
        (n, ) array of boolean values.
        Element is True if the corresponding foot is in the stance phase.

    """
    pad_width = 5
    variances = sw.apply_to_padded(
        step_signal, np.nanvar, pad_width, 'reflect', reflect_type='odd')

    points_to_cluster = nf.to_column(nf.remove_nan(np.array(variances)))
    k_means = KMeans(n_clusters=2, random_state=0).fit(points_to_cluster)

    signal_labels = pp.assign_to_closest(
        nf.to_column(variances), k_means.cluster_centers_)

    stance_label = np.argmin(k_means.cluster_centers_)
    is_stance = np.logical_and(signal_labels == stance_label,
                               ~np.isnan(variances))

    # Filter groups of stance frames that are too small

    labels = np.fromiter(itf.label_repeated_elements(is_stance), 'int')

    is_real = ~np.isnan(step_signal)
    good_labels = nf.large_boolean_groups(
        is_stance & is_real, labels, min_length=10)

    good_elements = np.in1d(labels, list(good_labels))

    is_stance &= good_elements

    return is_stance


def get_phase_dataframe(foot_series, direction_pass):
    """
    Return a DataFrame displaying the phase and phase number of each frame.

    The phase number is a count of the phase occurrence.
    (e.g., stance 0, 1, ...).

    Parameters
    ----------
    foot_series : ndarray
        Index is 'frame'.
        Values are foot positions.
    direction_pass : ndarray
        Direction of motion for the walking pass.

    Returns
    -------
    df_phase : DataFrame
        Index is 'frame'.
        Columns are 'phase', 'position', 'stride'.

    """
    frames = foot_series.index.values
    foot_points = np.stack(foot_series)

    line_point = np.zeros(direction_pass.shape)
    step_signal = lin.line_coordinate_system(line_point, direction_pass,
                                             foot_points)

    is_stance = detect_phases(step_signal)

    is_stance_series = pd.Series(is_stance, index=frames)
    is_stance_series.replace({True: 'stance', False: 'swing'}, inplace=True)

    df_phase = pd.DataFrame({'phase': is_stance_series}, dtype='category')
    df_phase['position'] = foot_series

    # Unique label for each distinct phase in the walking pass.
    # e.g. swing, stance, swing section get labelled 0, 1, 2.
    phase_labels = np.array([*itf.label_repeated_elements(is_stance)])

    is_swing = ~is_stance

    # Count of each phase in the walking pass.
    stance_numbers = [*itf.label_repeated_elements(phase_labels[is_stance])]
    swing_numbers = [*itf.label_repeated_elements(phase_labels[is_swing])]

    stance_series = pd.Series(stance_numbers, index=frames[is_stance])
    swing_series = pd.Series(swing_numbers, index=frames[is_swing])
    df_phase['stride'] = pd.concat([stance_series, swing_series])

    df_phase.index.name = 'frame'

    return df_phase


def group_stance_frames(df_phase):
    """
    Create a DataFrame of stance frames grouped by contact number.

    Parameters
    ----------
    df_phase : DataFrame
        Index is 'frame'.
        Columns are 'phase', 'stride', 'position',
        'first_contact', 'last_contact'.

    Returns
    -------
    df_grouped : DataFrame
        Columns are 'frame', 'position'

    Examples
    --------
    >>> pos = [[1, 2], [3, 4]]
    >>> d = {'phase': ['stance', 'stance'], 'stride': [0, 0], 'position': pos}

    >>> df = pd.DataFrame(d, index=[175, 176])
    >>> df.index.name = 'frame'

    >>> group_stance_frames(df)
       stride  frame    position  first_contact  last_contact
    0       0  175.5  [2.0, 3.0]            175           176

    """
    df_stance = df_phase[df_phase.phase == 'stance'].reset_index()

    column_funcs = {'frame': np.median, 'position': np.stack}
    df_grouped = pf.apply_to_grouped(df_stance, 'stride', column_funcs)

    df_grouped.position = df_grouped.position.apply(np.nanmedian, axis=0)

    # Record first and last stance frame of each stride
    contact_frames = df_stance.groupby('stride').frame.agg(['min', 'max'])
    contact_frames.columns = ['first_contact', 'last_contact']

    df_grouped = pd.concat((df_grouped, contact_frames), axis=1)

    return df_grouped.reset_index()


def get_contacts(foot_series, direction_pass):
    """
    Return a DataFrame containing contact frames and positions for one foot.

    A contact frame is when the foot is contacting the floor
    (in a stance phase).

    Parameters
    ----------
    foot_series : ndarray
        Index is 'frame'.
        Values are foot positions.
    direction_pass : ndarray
        Direction of motion for the walking pass.

    Returns
    -------
    DataFrame
        Columns are 'frame', 'position'

    """
    df_phase = get_phase_dataframe(foot_series, direction_pass)

    return group_stance_frames(df_phase)
