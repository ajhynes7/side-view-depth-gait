"""Assign left and right sides to selected foot positions."""

from os.path import join

import pandas as pd
from sklearn.cluster import DBSCAN

import modules.assign_sides as asi
import modules.numpy_funcs as nf


def main():

    load_dir = join('data', 'kinect', 'best_pos')

    assigned_dir = join('data', 'kinect', 'assigned')
    direction_dir = join('data', 'kinect', 'direction')

    # List of trials to run
    running_path = join('data', 'kinect', 'running', 'trials_to_run.csv')
    trials_to_run = pd.read_csv(running_path, header=None, squeeze=True).values

    for trial_name in trials_to_run:

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

        # Project points onto X-Z plane
        df_selected_2d = df_selected.applymap(
            lambda point: asi.convert_to_2d(point)
        )

        # Sort labels so that the frames are in temporal order
        labels = nf.map_to_whole(labels)

        # %% Assign sides to each walking pass and combine passes
        # DataFrames for each walking pass in a trial

        pass_dfs_2d = list(nf.group_by_label(df_selected_2d, labels))

        dict_passes, pass_directions = {}, []

        for i, df_pass in enumerate(pass_dfs_2d):
            _, direction_pass = asi.direction_of_pass(df_pass)

            pass_directions.append(direction_pass)
            df_pass_assigned = asi.assign_sides_pass(df_pass, direction_pass)

            # Assign correct sides to feet
            dict_passes[i] = df_pass_assigned

        df_assigned = pd.concat(dict_passes)
        df_assigned.index = df_assigned.index.set_names('pass', level=0)

        direction_series = pd.Series(pass_directions)
        direction_series.index.name = 'pass'

        df_assigned.to_pickle(join(assigned_dir, trial_name + '.pkl'))
        direction_series.to_pickle(join(direction_dir, trial_name + '.pkl'))


if __name__ == '__main__':
    main()
