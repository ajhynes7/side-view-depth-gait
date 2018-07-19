"""Script to calculate gait metrics from Kinect data."""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import modules.gait_metrics as gm
import modules.iterable_funcs as itf
import modules.numpy_funcs as nf


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

        # Convert all position vectors to float type
        # so they can be easily input to linear algebra functions
        df_head_feet = df_head_feet.applymap(pd.to_numeric)

        # Cluster frames with k means to locate the 4 walking passes
        frames = df_head_feet.index.values.reshape(-1, 1)
        k_means = KMeans(n_clusters=4, random_state=0).fit(frames)

        # Sort labels so that the frames are in temporal order
        labels = itf.map_sort(k_means.labels_)

        # DataFrames for each walking pass in a trial
        pass_dfs = nf.group_by_label(df_head_feet, labels)

        df_trial = gm.combine_walking_passes(pass_dfs)
        row_metrics = df_trial.apply(np.nanmedian, axis=0)

        base_name = os.path.basename(file_path)     # File with extension
        file_name = os.path.splitext(base_name)[0]  # File with no extension

        # Fill in row of gait metrics DataFrame
        df_metrics.loc[file_name] = row_metrics

    df_metrics.to_csv(save_path, na_rep='NaN')


if __name__ == '__main__':

    main()
