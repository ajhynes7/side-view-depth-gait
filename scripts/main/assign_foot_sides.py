"""Assign left and right sides to selected foot positions."""

import pickle
from os.path import join

import pandas as pd
from sklearn.cluster import DBSCAN

import modules.assign_sides as asi
import modules.numpy_funcs as nf


def main():

    df_selected = pd.read_pickle(join('data', 'kinect', 'df_selected.pkl'))

    dict_trials, dict_lines_fit = {}, {}

    for trial_name, df_trial in df_selected.groupby(level=0):

        # Drop the first level of the multi-index (the trial name)
        # Now the index is just the frames
        df_trial = df_trial.droplevel(0)
        frames = df_trial.index

        # Cluster frames to locate the walking passes
        clustering = DBSCAN(eps=5).fit(nf.to_column(frames))
        labels = clustering.labels_

        # Drop frames identified as noise
        is_noise = labels == -1
        frames_to_drop = frames[is_noise]
        df_trial = df_trial.drop(frames_to_drop)
        labels = labels[~is_noise]

        # Project points onto X-Z plane
        df_trial_2d = df_trial.applymap(lambda point: asi.convert_to_2d(point))

        # Sort labels so that the frames are in temporal order
        labels = nf.map_to_whole(labels)

        # %% Assign sides to each walking pass and combine passes
        # DataFrames for each walking pass in a trial

        pass_dfs_2d = list(nf.group_by_label(df_trial_2d, labels))

        dict_passes, list_lines_passes = {}, []

        for num_pass, df_pass in enumerate(pass_dfs_2d):

            # Line of best fit for the walking pass
            line_pass = asi.direction_of_pass(df_pass)

            # Assign correct sides to feet
            df_pass_assigned = asi.assign_sides_pass(df_pass, line_pass.direction)

            dict_passes[num_pass] = df_pass_assigned
            list_lines_passes.append(line_pass)

        df_assigned_trial = pd.concat(dict_passes)
        df_assigned_trial.index = df_assigned_trial.index.set_names('pass', level=0)

        dict_trials[trial_name] = df_assigned_trial
        dict_lines_fit[trial_name] = list_lines_passes

    df_assigned = pd.concat(dict_trials)

    # Save the assigned positions for each trial
    df_assigned.to_pickle(join('data', 'kinect', 'df_assigned.pkl'))

    # Save the best fit lines for each walking pass of each trial
    with open(join('data', 'kinect', 'dict_lines_fit.pkl'), 'wb') as handle:
        pickle.dump(dict_lines_fit, handle)


if __name__ == '__main__':
    main()
