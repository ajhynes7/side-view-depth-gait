"""Estimate lengths of the body for each trial."""

from os.path import join
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import modules.iterable_funcs as itf
import modules.pose_estimation as pe


def main():

    kinect_dir = join('data', 'kinect')
    hypo_dir = join(kinect_dir, 'processed', 'hypothesis')

    # List of trials to run
    running_path = join('data', 'kinect', 'running', 'trials_to_run.csv')
    trials_to_run = pd.read_csv(running_path, header=None, squeeze=True).values

    # Read first DataFrame with position hypotheses to get number of lengths
    df_hypo = pd.read_pickle(join(hypo_dir, trials_to_run[0] + '.pkl'))

    n_part_types = df_hypo.shape[1]
    n_lengths = n_part_types - 1

    part_labels = range(n_part_types)

    df_lengths = pd.DataFrame(
        index=trials_to_run, columns=range(n_lengths))
    df_lengths.index.name = 'trial_name'

    t = time.time()
    trials_run, frames_run = 0, 0

    # %% Calculate lengths for each walking trial

    for trial_name in trials_to_run:

        file_path = join(kinect_dir, 'processed', 'hypothesis',
                         trial_name + '.pkl')
        df_hypo = pd.read_pickle(file_path).dropna()

        n_frames = df_hypo.shape[0]
        trial_lengths = np.zeros((n_frames, n_lengths))

        for i in range(n_frames):

            frame_series = df_hypo.iloc[i]

            population, labels = pe.get_population(frame_series, part_labels)
            dist_matrix = cdist(population, population)

            for idx_a, idx_b in itf.pairwise(part_labels):

                pair_distances = dist_matrix[labels == idx_a][:, labels ==
                                                              idx_b]

                trial_lengths[i, idx_a] = np.percentile(pair_distances, 25)

        # Fill in row of lengths DataFrame
        df_lengths.loc[trial_name] = np.median(trial_lengths, axis=0)

        trials_run += 1
        frames_run += n_frames

    df_lengths.to_csv(join(kinect_dir, 'lengths', 'kinect_lengths.csv'))

    # %% Calculate run-time metrics

    time_elapsed = time.time() - t
    frames_per_second = round(frames_run / time_elapsed)

    print("""
    Number of trials: {}\n
    Number of frames: {}\n
    Total time: {}\n
    Frames per second: {}""".format(trials_run, frames_run,
                                    round(time_elapsed, 2), frames_per_second))


if __name__ == '__main__':
    main()
