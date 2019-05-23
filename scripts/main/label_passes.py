"""Cluster the frames of walking trials to determine walking passes."""

from os.path import join

import pandas as pd
from sklearn.cluster import DBSCAN


def main():

    df_selected = pd.read_pickle(join('data', 'kinect', 'df_selected.pkl'))
    trial_names = df_selected.index.get_level_values(0).unique()

    dict_trials = dict.fromkeys(trial_names)

    for trial_name in trial_names:

        df_selected_trial = df_selected.loc[trial_name]
        frames = df_selected_trial.index.values

        # Cluster frames to locate the walking passes
        clustering = DBSCAN(eps=5).fit(frames.reshape(-1, 1))
        labels = clustering.labels_

        df_selected_trial['pass'] = labels

        # Drop frames marked as noise
        frames_to_drop = frames[labels == -1]
        df_selected_trial = df_selected_trial.drop(frames_to_drop)

        # Add the pass number to the index
        df_selected_trial = df_selected_trial.set_index('pass', append=True)
        df_selected_trial = df_selected_trial.reorder_levels(['pass', 'frame'])

        dict_trials[trial_name] = df_selected_trial

    df_selected_passes = pd.concat(dict_trials)
    df_selected_passes.index = df_selected_passes.index.rename('trial_name', level=0)

    df_selected_passes.to_pickle(join('data', 'kinect', 'df_selected_passes.pkl'))


if __name__ == '__main__':
    main()
