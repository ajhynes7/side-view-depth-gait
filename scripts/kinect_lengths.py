"""Script to estimate lengths of the body for each trial."""

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
    return (a - b) ** 2


lower_part_types = ['HEAD', 'HIP', 'UPPER_LEG', 'KNEE', 'LOWER_LEG', 'FOOT']

load_dir = os.path.join('data', 'kinect', 'processed', 'hypothesis')
save_dir = os.path.join('data', 'kinect', 'lengths')
match_dir = os.path.join('data', 'matching')

save_name = 'kinect_lengths.csv'

df_match = pd.read_csv(os.path.join(match_dir, 'match_kinect_zeno.csv'))

df_lengths = pd.DataFrame(index=df_match.kinect,
                          columns=range(len(lower_part_types) - 1))
df_lengths.index.name = 'file_name'

t = time.time()
total_frames = 0

n_lengths = len(lower_part_types) - 1

# Pairs of consecutive body part indices
index_pairs = list(itf.pairwise(range(len(lower_part_types))))


# %% Calculate lengths for each file

for file_name in df_match.kinect:

    file_path = os.path.join(load_dir, file_name + '.pkl')

    base_name = os.path.basename(file_path)  # File with extension
    file_name = os.path.splitext(base_name)[0]  # File with no extension

    df = pd.read_pickle(file_path)

    # Select frames with data
    string_index, part_labels = sf.strings_with_any_substrings(
        df.columns, lower_part_types)

    lower_parts = df.columns[string_index]

    df_lower = df[lower_parts].dropna(axis=0)

    n_frames = df_lower.shape[0]
    total_frames += n_frames

    trial_lengths = np.zeros((n_frames, n_lengths))

    for i, (_, frame_series) in enumerate(df_lower.iterrows()):

        population, labels = pe.get_population(frame_series, part_labels)
        dist_matrix = cdist(population, population)

        for idx_a, idx_b in index_pairs:

            pair_distances = dist_matrix[labels == idx_a][:, labels == idx_b]

            trial_lengths[i, idx_a] = np.percentile(pair_distances, 25)

    # Fill in row of lengths DataFrame
    df_lengths.loc[file_name] = np.median(trial_lengths, axis=0)

save_path = os.path.join(save_dir, save_name)
df_lengths.to_csv(save_path)


# %% Calculate run-time metrics

n_trials = df_match.shape[0]

time_elapsed = time.time() - t
frames_per_second = round(total_frames / time_elapsed)


print("""
Number of trials: {}\n
Number of frames: {}\n
Total time: {}\n
Frames per second: {}""".format(n_trials,
                                total_frames,
                                round(time_elapsed, 2),
                                frames_per_second))
