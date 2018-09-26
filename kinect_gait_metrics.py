"""Script to calculate gait metrics from Kinect data."""

import glob
import os

import numpy as np
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

    # Remove outliers

    mean_foot = (df_head_feet.L_FOOT + df_head_feet.R_FOOT) / 2
    dist_head_feet = (df_head_feet.HEAD - mean_foot).apply(np.linalg.norm)

    dist_filtered = st.mad_outliers(dist_head_feet, 2)
    keep_frame = ~np.isnan(dist_filtered)

    df_head_feet = df_head_feet[keep_frame]

    # Cluster frames with mean shift to locate the walking passes
    frames = df_head_feet.index
    mean_shift = MeanShift(bandwidth=60).fit(nf.to_column(frames))
    labels = mean_shift.labels_

    # Sort labels so that the frames are in temporal order
    labels = nf.map_to_whole(labels)

    # DataFrames for each walking pass in a trial
    pass_dfs = nf.group_by_label(df_head_feet, labels)

    df_trial = gm.combine_walking_passes(pass_dfs)
    row_metrics = df_trial.apply(np.nanmedian, axis=0)

    base_name = os.path.basename(file_path)  # File with extension
    file_name = os.path.splitext(base_name)[0]  # File with no extension

    # Fill in row of gait metrics DataFrame
    df_metrics.loc[file_name] = row_metrics

df_metrics.to_csv(save_path, na_rep='NaN')
