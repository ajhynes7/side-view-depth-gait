"""Create figures using functions from the main code."""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import pandas as pd
from sklearn.cluster import DBSCAN

import analysis.plotting as pl
import modules.assign_sides as asi
import modules.gait_parameters as gp
import modules.linear_algebra as lin
import modules.numpy_funcs as nf
import modules.pandas_funcs as pf
import modules.phase_detection as pde
import modules.signals as sig
import modules.sliding_window as sw


def main():

    file_name = '2014-12-08_P004_Post_000'

    hypo_dir = os.path.join('data', 'kinect', 'processed', 'hypothesis')
    best_pos_dir = os.path.join('data', 'kinect', 'best_pos')

    hypo_paths = glob.glob(os.path.join(hypo_dir, '*.pkl'))
    best_pos_paths = glob.glob(os.path.join(best_pos_dir, '*.pkl'))

    hypo_paths = [x for x in hypo_paths if file_name in x]
    best_pos_paths = [x for x in best_pos_paths if file_name in x]

    df_head_feet = pd.read_pickle(best_pos_paths[0])

    # Convert all position vectors to float type
    # so they can be easily input to linear algebra functions
    df_head_feet = df_head_feet.applymap(pd.to_numeric)

    # Cluster frames with mean shift to locate the walking passes
    frames = df_head_feet.index

    # Cluster frames with mean shift to locate the walking passes
    clustering = DBSCAN(eps=5).fit(nf.to_column(frames))
    labels = clustering.labels_

    # Sort labels so that the frames are in temporal order
    labels = nf.map_to_whole(labels)

    # DataFrames for each walking pass in a trial
    pass_dfs = list(nf.group_by_label(df_head_feet, labels))

    df_pass = pass_dfs[0]

    # %% Foot Distance

    frames = df_pass.index.values

    foot_pos_l = np.stack(df_pass.L_FOOT)
    foot_pos_r = np.stack(df_pass.R_FOOT)
    norms = np.apply_along_axis(norm, 1, foot_pos_l - foot_pos_r)

    signal = 1 - sig.nan_normalize(norms)
    rms = sig.root_mean_square(signal)

    peak_frames, _ = sw.detect_peaks(frames, signal,
                                     window_length=3, min_height=rms)

    fig = plt.figure()

    foot_dist = pd.Series(norms, index=frames)

    pl.scatter_series(foot_dist, c='k', s=15)
    plt.plot(foot_dist, c='k', linewidth=0.7)

    plt.vlines(x=peak_frames, ymin=foot_dist.max(),
               ymax=foot_dist.min(), color='r')

    plt.xlabel('Frame')
    plt.ylabel('Foot Distance [cm]')

    save_path = os.path.join('figures', 'foot_dist.pdf')
    fig.savefig(save_path, format='pdf', dpi=1200)

    # %% Step Signal

    _, direction_pass = gp.direction_of_pass(df_pass)

    # Assign correct sides to feet
    df_pass = asi.assign_sides_pass(df_pass, direction_pass)

    # Ensure there are no missing frames in the walking pass
    df_pass = pf.make_index_consecutive(df_pass)
    df_pass = df_pass.applymap(lambda x: x if isinstance(x, np.ndarray)
                               else np.full(3, np.nan))

    foot_series = df_pass.R_FOOT
    frames_pass = df_pass.index.values

    foot_points = np.stack(foot_series)

    step_signal = lin.line_coordinate_system(np.zeros(3), direction_pass,
                                             foot_points)

    is_stance = pde.detect_phases(step_signal)

    fig = plt.figure()

    points = np.column_stack((frames_pass, step_signal))
    pl.scatter_labels(points, ~is_stance)
    plt.xlabel('Frame')
    plt.ylabel('Signal')
    plt.legend(['Stance', 'Swing'])

    save_path = os.path.join('figures', 'step_signal.pdf')
    fig.savefig(save_path, format='pdf', dpi=1200)


if __name__ == '__main__':
    main()
