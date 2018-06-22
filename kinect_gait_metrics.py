import os
import glob

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

import modules.gait_metrics as gm


def main():

    load_dir = os.path.join('data', 'kinect', 'best pos')
    save_dir = os.path.join('data', 'results')

    save_name = 'kinect_gait_metrics.csv'

    # All files with .pkl extension
    file_paths = glob.glob(os.path.join(load_dir, '*.pkl'))
    save_path = os.path.join(save_dir, save_name)

    df_metrics = pd.read_csv(save_path, index_col=0)

    for file_path in file_paths:

        df_head_feet = pd.read_pickle(file_path)

        # %% Peak detection

        # Cluster frames with k means to locate the 4 walking passes
        frames = df_head_feet.index.values.reshape(-1, 1)
        k_means = KMeans(n_clusters=4, random_state=0).fit(frames)

        foot_diff = df_head_feet.L_FOOT - df_head_feet.R_FOOT
        foot_dist = foot_diff.apply(np.linalg.norm)

        # Detect peaks in the foot distance data
        peak_frames, mid_frames = gm.foot_dist_peaks(foot_dist, r=5)

        # %% Gait metrics

        # Dictionary that maps image frames to cluster labels
        label_dict = dict(zip(frames.flatten(), k_means.labels_))

        df_head_metrics = gm.gait_dataframe(df_head_feet, mid_frames,
                                            label_dict, gm.head_metrics)

        df_foot_metrics = gm.gait_dataframe(df_head_feet, peak_frames,
                                            label_dict, gm.foot_metrics)

        # Series with mean of each gait metric
        gait_results = pd.concat([df_head_metrics.mean(),
                                  df_foot_metrics.mean()])

        # %% Fill in row of results DataFrame

        base_name = os.path.basename(file_path)     # File with extension
        file_name = os.path.splitext(base_name)[0]  # File with no extension

        df_metrics.loc[file_name] = gait_results

    df_metrics.to_csv(save_path)


if __name__ == '__main__':

    main()
