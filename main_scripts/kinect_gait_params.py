"""Script to calculate gait parameters from Kinect data."""

import os

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

import modules.gait_parameters as gp
import modules.numpy_funcs as nf


load_dir = os.path.join('data', 'kinect', 'best_pos')
save_dir = os.path.join('data', 'kinect', 'gait_params')
match_dir = os.path.join('data', 'matching')

df_match = pd.read_csv(os.path.join(match_dir, 'match_kinect_zeno.csv'))


for file_name in df_match.kinect:

    file_path = os.path.join(load_dir, file_name + '.pkl')

    df_head_feet = pd.read_pickle(file_path)
    frames = df_head_feet.index

    # Convert points to floats to allow future calculations (e.g. np.isnan)
    df_head_feet = df_head_feet.applymap(lambda point: point.astype(np.float))

    # Cluster frames with mean shift to locate the walking passes
    clustering = DBSCAN(eps=5).fit(nf.to_column(frames))
    labels = clustering.labels_

    # Drop frames identified as noise
    is_noise = labels == -1
    frames_to_drop = frames[is_noise]
    df_head_feet = df_head_feet.drop(frames_to_drop)
    labels = labels[~is_noise]

    # Sort labels so that the frames are in temporal order
    labels = nf.map_to_whole(labels)

    # DataFrames for each walking pass in a trial
    pass_dfs_3d = list(nf.group_by_label(df_head_feet, labels))

    # Reduce dimension of head and foot positions on each walking pass
    pass_dfs_2d = []
    for df_pass in pass_dfs_3d:
        pass_dfs_2d.append(
            df_pass.applymap(lambda point: np.array([point[2], point[0]])))

    df_trial = gp.combine_walking_passes(pass_dfs_2d)

    save_path = os.path.join(save_dir, file_name + '.pkl')
    df_trial.to_pickle(save_path)
