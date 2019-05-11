"""Plot foot distance and step signal."""

from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.cluster import DBSCAN

import analysis.plotting as pl
import modules.assign_sides as asi
import modules.numpy_funcs as nf
import modules.pandas_funcs as pf
import modules.phase_detection as pde
import modules.signals as sig
import modules.sliding_window as sw


def main():

    df_selected = pd.read_pickle(join('data', 'kinect', 'df_selected.pkl'))

    trial_name = df_selected.index.levels[0][0]
    df_selected_trial = df_selected.loc[trial_name]

    frames = df_selected_trial.index

    # Cluster frames with mean shift to locate the walking passes
    clustering = DBSCAN(eps=5).fit(nf.to_column(frames))
    labels = clustering.labels_

    # Sort labels so that the frames are in temporal order
    labels = nf.map_to_whole(labels)

    # DataFrames for each walking pass in a trial
    pass_dfs = list(nf.group_by_label(df_selected_trial, labels))

    df_pass = pass_dfs[0]

    # Convert to 2D
    df_pass = df_pass.applymap(lambda point: asi.convert_to_2d(point))

    # %% Foot Distance

    frames = df_pass.index.values

    foot_pos_l = np.stack(df_pass.L_FOOT)
    foot_pos_r = np.stack(df_pass.R_FOOT)
    norms = norm(foot_pos_l - foot_pos_r, axis=1)

    signal = 1 - sig.nan_normalize(norms)
    rms = sig.root_mean_square(signal)

    peak_frames, _ = sw.detect_peaks(frames, signal, window_length=3, min_height=rms)

    fig = plt.figure()

    foot_dist = pd.Series(norms, index=frames)

    pl.scatter_series(foot_dist, c='k', s=15)
    plt.plot(foot_dist, c='k', linewidth=0.7)
    plt.vlines(x=peak_frames, ymin=foot_dist.max(), ymax=foot_dist.min(), color='r')
    plt.xlabel('Frame')
    plt.ylabel('Foot Distance [cm]')

    fig.savefig(join('figures', 'foot_dist.pdf'), format='pdf', dpi=1200)

    # %% Step Signal

    line_pass = asi.direction_of_pass(df_pass)

    # Assign correct sides to feet
    df_pass = asi.assign_sides_pass(df_pass, line_pass.direction)

    # Ensure there are no missing frames in the walking pass
    df_pass = pf.make_index_consecutive(df_pass)
    df_pass = df_pass.applymap(lambda x: x if isinstance(x, np.ndarray) else np.full(3, np.nan))

    foot_series = df_pass.R_FOOT
    frames_pass = df_pass.index.values

    foot_points = np.stack(foot_series)

    step_signal = line_pass.transform_points(foot_points)

    is_stance = pde.detect_phases(step_signal)

    fig = plt.figure()

    points = np.column_stack((frames_pass, step_signal))
    pl.scatter_labels(points, ~is_stance)
    plt.xlabel('Frame')
    plt.ylabel('Signal')
    plt.legend(['Stance', 'Swing'])

    fig.savefig(join('figures', 'step_signal.pdf'), format='pdf', dpi=1200)


if __name__ == '__main__':
    main()
