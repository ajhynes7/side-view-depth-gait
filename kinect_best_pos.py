import os
import pandas as pd
import numpy as np

import modules.general as gen
import modules.math_funcs as mf
import modules.pose_estimation as pe
import modules.stats as st


# %% Parameters

def cost_func(a, b): return (a - b)**2


def score_func(a, b):

    x = 1 / mf.norm_ratio(a, b)

    return -(x - 1)**2 + 1


lower_part_types = ['HEAD', 'HIP', 'UPPER_LEG', 'KNEE', 'LOWER_LEG', 'FOOT']

radii = [i for i in range(0, 30, 5)]

part_connections = np.matrix('0 1; 1 2; 2 3; 3 4; 4 5; 3 5; 1 3')


# %% Read DataFrame

trial_id = '2014-12-22_P007_Pre_004'

load_directory = '../../MEGA/Data/Kinect Zeno/Kinect processed'
save_directory = '../../MEGA/Data/Kinect Zeno/Kinect best pos'

load_path = os.path.join(load_directory, trial_id + '.pkl')
save_path = os.path.join(save_directory, trial_id + '.pkl')

df = pd.read_pickle(load_path)


# %% Select frames with data

lower_parts, part_labels = gen.strings_with_any_substrings(
    df.columns, lower_part_types)
df_lower = df[lower_parts].dropna(axis=0)

population_series = df_lower.apply(
    lambda row: pe.get_population(row, part_labels)[0], axis=1)

label_series = df_lower.apply(
    lambda row: pe.get_population(row, part_labels)[1], axis=1)


# %% Estimate lengths between consecutive parts

lengths = pe.estimate_lengths(population_series, label_series,
                              cost_func, 60, eps=0.01)

# Expected lengths for all part connections,
# including non-adjacent (e.g., knee to foot)
label_adj_list = pe.lengths_to_adj_list(part_connections, lengths)


# %% Select paths to feet on each frame

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
L_foot_pos = df_best_pos['Side A'].apply(lambda row: row[-1, :])
R_foot_pos = df_best_pos['Side B'].apply(lambda row: row[-1, :])

# Combine into new DataFrame
df_head_feet = pd.concat([head_pos, L_foot_pos, R_foot_pos], axis=1)
df_head_feet.columns = ['HEAD', 'L_FOOT', 'R_FOOT']
df_head_feet.index.name = 'Frame'


# %% Remove outlier frames

y_foot_L = df_head_feet.apply(lambda row: row['L_FOOT'][1], axis=1).values
y_foot_R = df_head_feet.apply(lambda row: row['R_FOOT'][1], axis=1).values

y_foot_L_filtered = st.mad_outliers(y_foot_L, 2)
y_foot_R_filtered = st.mad_outliers(y_foot_R, 2)

good_frame_L = ~np.isnan(y_foot_L_filtered)
good_frame_R = ~np.isnan(y_foot_R_filtered)

df_head_feet = df_head_feet[good_frame_L & good_frame_R]


# %% Save data

df_head_feet.to_pickle(save_path)
