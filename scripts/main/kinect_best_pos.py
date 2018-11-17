"""Select the best body part positions from multiple hypotheses."""

import os
import time

import numpy as np
import pandas as pd

import analysis.math_funcs as mf
import modules.pose_estimation as pe
import modules.string_funcs as sf


def cost_func(a, b):
    """Cost function for weighting edges of graph."""
    return (a - b)**2


def score_func(a, b):
    """Score function for scoring links between body parts."""
    x = 1 / mf.norm_ratio(a, b)
    return -(x - 1)**2 + 1


def main():

    lower_part_types = [
        'HEAD', 'HIP', 'UPPER_LEG', 'KNEE', 'LOWER_LEG', 'FOOT'
    ]

    radii = [i for i in range(5, 30, 5)]

    part_connections = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                                 [3, 5], [1, 3]])

    # Reading data
    load_dir = os.path.join('data', 'kinect', 'processed', 'hypothesis')
    save_dir = os.path.join('data', 'kinect', 'best_pos')

    # DataFrame with lengths between body parts
    length_path = os.path.join('data', 'kinect', 'lengths',
                               'kinect_lengths.csv')
    df_length = pd.read_csv(length_path, index_col=0)

    t = time.time()
    total_frames = 0

    # %% Select best positions from each Kinect data file

    # List of trials to run
    running_path = os.path.join('data', 'kinect', 'running',
                                'trials_to_run.csv')
    trials_to_run = pd.read_csv(running_path, header=None, squeeze=True).values

    for file_name in trials_to_run:

        file_path = os.path.join(load_dir, file_name + '.pkl')

        df = pd.read_pickle(file_path)

        base_name = os.path.basename(file_path)  # File with extension
        file_name = os.path.splitext(base_name)[0]  # File with no extension

        lengths = df_length.loc[file_name]  # Read estimated lengths for trial

        # Select frames with data
        string_index, part_labels = sf.strings_with_any_substrings(
            df.columns, lower_part_types)

        lower_parts = df.columns[string_index]

        df_lower = df[lower_parts].dropna(axis=0)

        population_series = df_lower.apply(
            lambda row: pe.get_population(row, part_labels)[0], axis=1)

        label_series = df_lower.apply(
            lambda row: pe.get_population(row, part_labels)[1], axis=1)

        # Expected lengths for all part connections,
        # including non-adjacent (e.g., knee to foot)
        label_adj_list = pe.lengths_to_adj_list(part_connections, lengths)

        # Select paths to feet on each frame

        # List of image frames with data
        frames = population_series.index.values

        best_pos_list = []

        for f in frames:
            population = population_series.loc[f]
            labels = label_series.loc[f]

            # Select the best two shortest paths
            pos_1, pos_2 = pe.process_frame(population, labels, label_adj_list,
                                            radii, cost_func, score_func)

            best_pos_list.append((pos_1, pos_2))

        df_best_pos = pd.DataFrame(
            best_pos_list, index=frames, columns=['Side A', 'Side B'])

        # Head and foot positions as series
        head_pos = df_best_pos['Side A'].apply(lambda row: row[0, :])
        foot_pos_1 = df_best_pos['Side A'].apply(lambda row: row[-1, :])
        foot_pos_2 = df_best_pos['Side B'].apply(lambda row: row[-1, :])

        # Combine into new DataFrame
        df_head_feet = pd.concat([head_pos, foot_pos_1, foot_pos_2], axis=1)
        df_head_feet.columns = ['HEAD', 'L_FOOT', 'R_FOOT']
        df_head_feet.index.name = 'Frame'

        # Save data
        save_path = os.path.join(save_dir, file_name) + '.pkl'
        df_head_feet.to_pickle(save_path)

        total_frames += len(frames)

    # %% Calculate run-time metrics

    time_elapsed = time.time() - t
    frames_per_second = np.round(total_frames / time_elapsed)

    print("""
    Number of trials: {}\n
    Number of frames: {}\n
    Total time: {}\n
    Frames per second: {}""".format(
        len(trials_to_run), total_frames, np.round(time_elapsed, 2),
        frames_per_second))


if __name__ == '__main__':
    main()
