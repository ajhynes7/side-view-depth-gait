"""Script to calculate gait metrics from Kinect data."""

import glob
import os

import numpy as np
from numpy.linalg import norm
import pandas as pd
from sklearn.cluster import MeanShift

import analysis.stats as st
import modules.gait_metrics as gm
import modules.numpy_funcs as nf

load_dir = os.path.join('data', 'kinect', 'best_pos')
save_dir = os.path.join('data', 'results')

save_name = 'kinect_gait_metrics.csv'

# All files with .pkl extension
file_paths = sorted(glob.glob(os.path.join(load_dir, '*.pkl')))
save_path = os.path.join(save_dir, save_name)

df_metrics = pd.read_csv(save_path, index_col=0)

for file_path in file_paths:

    df_head_feet = pd.read_pickle(file_path)
    frames = df_head_feet.index

    # Convert points to floats to allow future calculations (e.g. np.isnan)
    df_head_feet = df_head_feet.applymap(lambda point: point.astype(np.float))

    # Remove outliers
    dist_to_foot_l = (df_head_feet.HEAD - df_head_feet.L_FOOT).apply(norm)
    dist_to_foot_r = (df_head_feet.HEAD - df_head_feet.R_FOOT).apply(norm)

    to_filter = st.relative_error(
        dist_to_foot_l, dist_to_foot_r, absolute=True)
    filtered = st.mad_outliers(to_filter, 2)

    good_frames = np.unique(frames[~np.isnan(filtered)])
    keep_frame = np.in1d(frames, good_frames)

    df_head_feet = df_head_feet[keep_frame]
    frames = df_head_feet.index

    # Cluster frames with mean shift to locate the walking passes
    mean_shift = MeanShift(bandwidth=60).fit(nf.to_column(frames))
    labels = mean_shift.labels_

    # Sort labels so that the frames are in temporal order
    labels = nf.map_to_whole(labels)

    # DataFrames for each walking pass in a trial
    pass_dfs_3d = nf.group_by_label(df_head_feet, labels)

    # Reduce dimension of head and foot positions on each walking pass
    pass_dfs_2d = []
    for df_pass in pass_dfs_3d:
        pass_dfs_2d.append(
            df_pass.applymap(lambda point: np.array([point[2], point[0]])))

    df_trial = gm.combine_walking_passes(pass_dfs_2d)
    row_metrics = df_trial.apply(np.nanmedian, axis=0)

    base_name = os.path.basename(file_path)  # File with extension
    file_name = os.path.splitext(base_name)[0]  # File with no extension

    # Fill in row of gait metrics DataFrame
    df_metrics.loc[file_name] = row_metrics

df_metrics.to_csv(save_path, na_rep='NaN')
