"""Assign left and right sides to selected foot positions."""

from os.path import join

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

import modules.assign_sides as asi
import modules.numpy_funcs as nf


def main():

    load_dir = join('data', 'kinect', 'best_pos')
    save_dir = join('data', 'kinect', 'assigned')

    # List of trials to run
    running_path = join('data', 'kinect', 'running', 'trials_to_run.csv')
    trials_to_run = pd.read_csv(running_path, header=None, squeeze=True).values

    for trial_name in trials_to_run[:2]:

        df_selected = pd.read_pickle(join(load_dir, trial_name + '.pkl'))

        frames = df_selected.index

        # Cluster frames to locate the walking passes
        clustering = DBSCAN(eps=5).fit(nf.to_column(frames))
        labels = clustering.labels_

        # Drop frames identified as noise
        is_noise = labels == -1
        frames_to_drop = frames[is_noise]
        df_selected = df_selected.drop(frames_to_drop)
        labels = labels[~is_noise]

        # Sort labels so that the frames are in temporal order
        labels = nf.map_to_whole(labels)

        # DataFrames for each walking pass in a trial
        pass_dfs_3d = list(nf.group_by_label(df_selected, labels))

        # Reduce dimension of head and foot positions on each walking pass
        pass_dfs_2d = []
        for df_pass in pass_dfs_3d:
            pass_dfs_2d.append(
                df_pass.applymap(lambda point: np.array([point[2], point[0]])))

        # %% Assign sides to each walking pass and combine passses

        dict_passes = {}

        for i, df_pass in enumerate(pass_dfs_2d):
            _, direction_pass = asi.direction_of_pass(df_pass)

            # Assign correct sides to feet
            dict_passes[i] = asi.assign_sides_pass(df_pass, direction_pass)

        df_assigned = pd.concat(dict_passes)
        df_assigned.index = df_assigned.index.set_names('pass', level=0)

        save_path = join(save_dir, trial_name + '.pkl')
        df_assigned.to_pickle(save_path)


if __name__ == '__main__':
    main()
