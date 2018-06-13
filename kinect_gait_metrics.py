import os
import glob

import pandas as pd
import numpy as np

import modules.gait_metrics as gm

from sklearn.cluster import KMeans


def main():

    for file_path in file_paths:

        df_head_feet = pd.read_pickle(file_path)

        # %% Peak detection

        # Cluster frames with k means to locate the 4 walking passes
        frames = df_head_feet.index.values.reshape(-1, 1)
        k_means = KMeans(n_clusters=4, random_state=0).fit(frames)

        foot_dist = df_head_feet.apply(lambda row: np.linalg.norm(
                                       row['L_FOOT'] - row['R_FOOT']), axis=1)

        # Detect peaks in the foot distance data
        peak_frames = gm.foot_dist_peaks(foot_dist, k_means.labels_, r=5)

        # %% Gait metrics

        # Dictionary that maps image frames to cluster labels
        label_dict = dict(zip(frames.flatten(), k_means.labels_))

        gait_df = gm.gait_dataframe(df_head_feet, peak_frames, label_dict)

        # %% Fill in row of results DataFrame

        df_metrics = pd.read_csv(save_path, index_col=0)

        base_name = os.path.basename(file_path)     # File with extension
        file_name = os.path.splitext(base_name)[0]  # File with no extension

        gait_results = gait_df.mean()
        df_metrics.loc[file_name] = gait_results

        df_metrics.to_csv(save_path)


if __name__ == '__main__':

    load_dir = os.path.join('data', 'kinect', 'best pos')
    save_dir = os.path.join('data', 'results')

    save_name = 'kinect_gait_metrics.csv'

    # All files with .pkl extension
    file_paths = glob.glob(os.path.join(load_dir, '*.pkl'))

    save_path = os.path.join(save_dir, save_name)

    main()
