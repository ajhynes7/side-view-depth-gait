"""Script to estimate lengths of the body for each trial."""

import glob
import os
import time

import pandas as pd

import modules.pose_estimation as pe
import modules.string_funcs as sf


def cost_func(a, b):
    """Cost function for weighting edges of graph."""
    return (a - b)**2


# Parameters

n_frames_est = 60  # Number of frames used to estimate lengths on each trial

lower_part_types = ['HEAD', 'HIP', 'UPPER_LEG', 'KNEE', 'LOWER_LEG', 'FOOT']

load_dir = os.path.join('data', 'kinect', 'processed', 'hypothesis')
save_dir = os.path.join('data', 'kinect', 'lengths')

save_name = 'kinect_lengths.csv'
save_path = os.path.join(save_dir, save_name)

# All files with .pkl extension
file_paths = glob.glob(os.path.join(load_dir, '*.pkl'))

# Find the Kinect files that have matching Zeno files
match_dir = os.path.join('data', 'matching')
df_match = pd.read_csv(os.path.join(match_dir, 'match_kinect_zeno.csv'))

# Create DataFrame of lengths between consecutive parts
file_names = [os.path.splitext(os.path.basename(path))[0] for path in file_paths]
df_lengths = pd.DataFrame(index=file_names, columns=range(len(lower_part_types) - 1))
df_lengths.index.name = 'File'

# Select best positions from each Kinect data file
t = time.time()
total_frames = 0

# Calculate lengths for each file

for file_path in file_paths:

    base_name = os.path.basename(file_path)  # File with extension
    file_name = os.path.splitext(base_name)[0]  # File with no extension

    df = pd.read_pickle(file_path)

    # Select frames with data
    string_index, part_labels = sf.strings_with_any_substrings(
        df.columns, lower_part_types)

    lower_parts = df.columns[string_index]

    df_lower = df[lower_parts].dropna(axis=0)

    population_series = df_lower.apply(
        lambda row: pe.get_population(row, part_labels)[0], axis=1)

    label_series = df_lower.apply(
        lambda row: pe.get_population(row, part_labels)[1], axis=1)

    # Estimate lengths between consecutive parts
    lengths = pe.estimate_lengths(
        population_series, label_series, cost_func, n_frames_est, eps=0.01)

    # Fill in row of lengths DataFrame
    df_lengths.loc[file_name] = lengths


n_trials = len(file_paths)
total_frames = n_trials * n_frames_est

time_elapsed = time.time() - t
frames_per_second = round(total_frames / time_elapsed)

df_lengths.to_csv(save_path)

print("""
Number of trials: {}\n
Number of frames: {}\n
Total time: {}\n
Frames per second: {}""".format(n_trials,
                                total_frames,
                                round(time_elapsed, 2),
                                frames_per_second))
