"""Module for detecting the phases of a foot during a walking pass."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import modules.iterable_funcs as itf
import modules.linear_algebra as lin
import modules.numpy_funcs as nf
import modules.pandas_funcs as pf
import modules.sliding_window as sw


def detect_phases(foot_points, direction_pass):
    """
    Detect the stance/swing phases of a foot during a walking pass.

    Parameters
    ----------
    foot_points : ndarray
        (n, 3) array of n foot positions.
    direction_pass : ndarray
        Vector for direction of a walking pass.

    Returns
    -------
    is_stance : ndarray
        (n, ) array of boolean values.
        Element is True if the corresponding foot is in the stance phase.

    """
    step_signal = lin.line_coordinate_system(np.zeros(3),
                                             direction_pass, foot_points)

    variances = sw.apply_to_padded(step_signal, np.nanvar, r=5)
    variance_array = nf.to_column(variances)

    k_means = KMeans(n_clusters=2, random_state=0).fit(variance_array)

    stance_label = np.argmin(k_means.cluster_centers_)
    is_stance = k_means.labels_ == stance_label

    return is_stance


def get_phase_dataframe(frames, is_stance):
    """
    Return a DataFrame displaying the phase and phase number of each frame.

    The phase number is a count of the phase occurrence.
    (e.g., stance 0, 1, ...).

    Parameters
    ----------
    frames : ndarray
        Frames of the walking pass.
    is_stance : ndarray
        Element is True if the corresponding foot is in the stance phase.

    Returns
    -------
    df_phase : DataFrame
        Index is 'frame'.
        Columns are 'phase', 'number'.

    """
    is_stance_series = pd.Series(is_stance, index=frames)
    is_stance_series.replace({True: 'stance', False: 'swing'}, inplace=True)

    df_phase = pd.DataFrame({'phase': is_stance_series}, dtype='category')

    # Unique label for each distinct phase in the walking pass.
    # e.g. swing, stance, swing section get labelled 0, 1, 2.
    phase_labels = np.array([*itf.label_repeated_elements(is_stance)])

    is_swing = ~is_stance

    # Count of each phase in the walking pass.
    stance_numbers = [*itf.label_repeated_elements(phase_labels[is_stance])]
    swing_numbers = [*itf.label_repeated_elements(phase_labels[is_swing])]

    stance_series = pd.Series(stance_numbers, index=frames[is_stance])
    swing_series = pd.Series(swing_numbers, index=frames[is_swing])
    df_phase['number'] = pd.concat([stance_series, swing_series])

    df_phase.index.name = 'frame'

    return df_phase


def group_stance_frames(df_phase, suffix):
    """
    Create a DataFrame of stance frames grouped by contact number.

    Parameters
    ----------
    df_phase : DataFrame
        Index is 'frame'.
        Columns are 'phase', 'number', 'position'.
    suffix : str
        Suffix to add to values of the index ('_L' or '_R').

    Returns
    -------
    df_grouped : DataFrame
        Index values have the suffix appended.
        Columns are 'frame', 'position'

    Examples
    --------
    >>> pos = [[1, 2], [3, 4]]
    >>> d = {'phase': ['stance', 'stance'], 'number': [0, 0], 'position': pos}

    >>> df = pd.DataFrame(d, index=[175, 176])
    >>> df.index.name = 'frame'

    >>> group_stance_frames(df, '_L')
         frame    position
    0_L  175.5  [2.0, 3.0]

    """
    df_stance = df_phase[df_phase.phase == 'stance'].reset_index()

    column_funcs = {'frame': np.median, 'position': np.stack}
    df_grouped = pf.apply_to_grouped(df_stance, 'number', column_funcs)

    df_grouped.position = df_grouped.position.apply(np.median, axis=0)
    df_grouped.index = df_grouped.index.astype('str') + suffix

    return df_grouped
