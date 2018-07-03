import os
import glob

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import modules.gait_metrics as gm
import modules.linear_algebra as lin


def main():

    load_dir = os.path.join('data', 'kinect', 'best pos')
    save_dir = os.path.join('data', 'results')

    save_name = 'kinect_gait_metrics.csv'

    # All files with .pkl extension
    file_paths = glob.glob(os.path.join(load_dir, '*.pkl'))
    save_path = os.path.join(save_dir, save_name)

    df_metrics = pd.read_csv(save_path, index_col=0)

    # Define function to project each body part position onto a plane
    plane_point = np.array([0, 0, 0])
    normal = np.array([0, 1, 0])

    def project_point(x):
        return lin.project_point_plane(x, plane_point, normal)

    for file_path in file_paths:

        df_head_feet = pd.read_pickle(file_path)

        # Convert all position vectors to float type
        # so they can be easily input to linear algebra functions
        df_head_feet = df_head_feet.applymap(pd.to_numeric)

        # Project all points onto same plane
        df_head_feet = df_head_feet.applymap(project_point)

        # Cluster frames with k means to locate the 4 walking passes
        frames = df_head_feet.index.values.reshape(-1, 1)
        k_means = KMeans(n_clusters=4, random_state=0).fit(frames)

        pass_dfs = gm.split_by_pass(df_head_feet, k_means.labels_)

        df_gait = gm.combine_walking_passes(pass_dfs)

        # %% Fill in row of results DataFrame

        base_name = os.path.basename(file_path)     # File with extension
        file_name = os.path.splitext(base_name)[0]  # File with no extension

        df_metrics.loc[file_name] = df_gait.mean()

    df_metrics.to_csv(save_path, na_rep='NaN')


if __name__ == '__main__':

    main()
