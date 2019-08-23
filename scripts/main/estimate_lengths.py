"""Estimate lengths of the body for each trial."""

import time
from os.path import join

import pandas as pd

import modules.pose_estimation as pe


def main():

    df_hypo = pd.read_pickle(join('data', 'kinect', 'df_hypo.pkl'))
    trials_to_run = df_hypo.index.get_level_values(0).unique()

    t = time.time()

    # %% Calculate lengths for each walking trial

    list_lengths = []

    for i, (trial_name, df_hypo_trial) in enumerate(df_hypo.groupby(level=0)):

        print(trial_name)

        lengths_estimated = pe.estimate_lengths(df_hypo_trial, atol=0.1)

        list_lengths.append(lengths_estimated)

    df_lengths = pd.DataFrame(list_lengths, index=trials_to_run)
    df_lengths.to_csv(join('data', 'kinect', 'kinect_lengths.csv'))

    # %% Calculate run-time metrics

    time_elapsed = time.time() - t

    trials_run = len(trials_to_run)

    print(
        """
        Number of trials: {}\n
        Total time: {}\n
        """.format(
            trials_run, round(time_elapsed, 2)
        )
    )


if __name__ == '__main__':
    main()
