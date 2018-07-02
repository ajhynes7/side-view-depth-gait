import os
import glob

import pandas as pd
import numpy as np

import modules.general as gen
import modules.math_funcs as mf
import modules.pose_estimation as pe
import modules.stats as st


def main():

    # %% Parameters

    def cost_func(a, b): return (a - b)**2

    def score_func(a, b):

        x = 1 / mf.norm_ratio(a, b)

        return -(x - 1)**2 + 1

    lower_part_types = ['HEAD', 'HIP', 'UPPER_LEG', 'KNEE', 'LOWER_LEG',
                        'FOOT']

    radii = [i for i in range(0, 30, 5)]

    part_connections = np.matrix('0 1; 1 2; 2 3; 3 4; 4 5; 3 5; 1 3')

    # %% Reading data

    load_dir = os.path.join('data', 'kinect', 'processed', 'hypothesis')
    save_dir = os.path.join('data', 'kinect', 'best pos')

    length_path = os.path.join('data', 'results', 'kinect_lengths.csv')

    # All files with .pkl extension
    file_paths = glob.glob(os.path.join(load_dir, '*.pkl'))

    # DataFrame with lengths between body parts
    df_length = pd.read_csv(length_path, index_col=0)

    # %% Select best positions from each Kinect data file

    for file_path in file_paths:

        df = pd.read_pickle(file_path)

        base_name = os.path.basename(file_path)     # File with extension
        file_name = os.path.splitext(base_name)[0]  # File with no extension

        lengths = df_length.loc[file_name]  # Read lengths

        # Select frames with data
        string_index, part_labels = gen.strings_with_any_substrings(
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
            pos_1, pos_2 = pe.process_frame(population, labels, label_adj_list,
                                            radii, cost_func, score_func)

            best_pos_list.append((pos_1, pos_2))

        df_best_pos = pd.DataFrame(best_pos_list, index=frames,
                                   columns=['Side A', 'Side B'])

        # Head and foot positions
        head_pos = df_best_pos['Side A'].apply(lambda row: row[0, :])
        foot_pos_1 = df_best_pos['Side A'].apply(lambda row: row[-1, :])
        foot_pos_2 = df_best_pos['Side B'].apply(lambda row: row[-1, :])

        # Combine into new DataFrame
        df_head_feet = pd.concat([head_pos, foot_pos_1, foot_pos_2], axis=1)
        df_head_feet.columns = ['HEAD', 'L_FOOT', 'R_FOOT']
        df_head_feet.index.name = 'Frame'

        # Remove outlier frames
        y_foot_1 = df_head_feet.apply(lambda row: row['L_FOOT'][1],
                                      axis=1).values

        y_foot_2 = df_head_feet.apply(lambda row: row['R_FOOT'][1],
                                      axis=1).values

        y_foot_filtered_1 = st.mad_outliers(y_foot_1, 2)
        y_foot_filtered_2 = st.mad_outliers(y_foot_2, 2)

        good_frame_1 = ~np.isnan(y_foot_filtered_1)
        good_frame_2 = ~np.isnan(y_foot_filtered_2)

        df_head_feet = df_head_feet[good_frame_1 & good_frame_2]

        # Save data
        save_path = os.path.join(save_dir, file_name) + '.pkl'
        df_head_feet.to_pickle(save_path)


if __name__ == '__main__':

    main()
