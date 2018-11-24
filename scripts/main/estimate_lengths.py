"""Estimate lengths of the body for each trial."""

import glob
import os
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import modules.iterable_funcs as itf
import modules.pose_estimation as pe
import modules.string_funcs as sf


def cost_func(a, b):
    """Cost function for weighting edges of graph."""
    return (a - b)**2


def main():

    lower_part_types = [
        'HEAD', 'HIP', 'UPPER_LEG', 'KNEE', 'LOWER_LEG', 'FOOT'
    ]

    load_dir = os.path.join('data', 'kinect', 'processed', 'hypothesis')
    save_dir = os.path.join('data', 'kinect', 'lengths')

    save_name = 'kinect_lengths.csv'

    file_paths = glob.glob(os.path.join(load_dir, '*.pkl'))
    file_names = [os.path.splitext(os.path.basename(x))[0] for x in file_paths]

    df_lengths = pd.DataFrame(
        index=file_names, columns=range(len(lower_part_types) - 1))
    df_lengths.index.name = 'file_name'

    t = time.time()
    trials_run, frames_run = 0, 0

    n_lengths = len(lower_part_types) - 1

    # %% Calculate lengths for each walking trial

    # List of trials to run
    running_path = os.path.join('data', 'kinect', 'running',
                                'trials_to_run.csv')
    trials_to_run = pd.read_csv(running_path, header=None, squeeze=True).values

    # Pairs of consecutive body part indices
    index_pairs = list(itf.pairwise(range(len(lower_part_types))))

    for file_name in trials_to_run:

        file_path = os.path.join(load_dir, file_name + '.pkl')

        df = pd.read_pickle(file_path)

        # Select frames with data
        string_index, part_labels = sf.strings_with_any_substrings(
            df.columns, lower_part_types)

        lower_parts = df.columns[string_index]

        df_lower = df[lower_parts].dropna(axis=0)
        n_frames = df_lower.shape[0]

        trials_run += 1
        frames_run += n_frames

        trial_lengths = np.zeros((n_frames, n_lengths))

        for i, (_, frame_series) in enumerate(df_lower.iterrows()):

            population, labels = pe.get_population(frame_series, part_labels)
            dist_matrix = cdist(population, population)

            for idx_a, idx_b in index_pairs:

                pair_distances = dist_matrix[labels == idx_a][:, labels ==
                                                              idx_b]

                trial_lengths[i, idx_a] = np.percentile(pair_distances, 25)

        # Fill in row of lengths DataFrame
        df_lengths.loc[file_name] = np.median(trial_lengths, axis=0)

    save_path = os.path.join(save_dir, save_name)
    df_lengths.to_csv(save_path)

    # %% Calculate run-time metrics

    time_elapsed = time.time() - t
    frames_per_second = round(frames_run / time_elapsed)

    print("""
    Number of trials: {}\n
    Number of frames: {}\n
    Total time: {}\n
    Frames per second: {}""".format(
        trials_run, frames_run, round(time_elapsed, 2),
        frames_per_second))


if __name__ == '__main__':
    main()
