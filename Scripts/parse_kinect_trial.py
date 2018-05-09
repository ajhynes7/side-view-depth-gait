import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


import sys
sys.path.insert(0, '../Modules/')

import graphs as gr
import pose_estimation as pe
import general as gen
import plotting as pl


def read_positions(frame_df, part, max_num_coords):
    """
    Inputs
    ------

    Outputs
    -------
    """
    row = frame_df[frame_df['Part'] == part]
    row_coordinates = row.loc[:, range(max_num_coords)].as_matrix()

    n_coordinates = np.sum(~np.isnan(row_coordinates))
    n_points = int(n_coordinates / 3)

    points = np.full([n_points, 3], np.nan)

    for idx, i in enumerate(range(0, n_coordinates, 3)):
        points[idx, :] = row_coordinates[0, i:i+3]

    return points

file_path = '../../../MEGA/Data/Kinect trials/2014-12-22_P007_Pre_004.txt'
column_names = [i for i in range(-2, 60)]

df = pd.read_csv(file_path, skiprows=range(22), header=None,\
                 names=column_names, sep='\t', engine='python')

# Change some column names
df.rename(columns={-2: 'Frame', -1: 'Part'}, inplace=True) 

# Replace any strings with nan in the Frame column
df['Frame'] = df['Frame'].replace(r'[^0-9]', np.nan, regex=True)

df['Frame'] = pd.to_numeric(df['Frame'])

parts = df.groupby('Part').groups.keys()  # Part names


# Parameters
max_num_coords = 60
radii = [i for i in range(30)]

cost_func = lambda a, b: (a - b)**2
score_func = lambda x : -(x - 1)**2 + 1


# Dataframe for current image frame
frame_df = df[df['Frame'] == 624]


part_types = ['HEAD', 'HIP', 'UPPER_LEG', 'KNEE', 'LOWER_LEG', 'FOOT']
lengths = [63.9626,   19.3718,   12.8402,   22.0421,   20.5768]

edges = np.matrix('0 1;  \
                   1 2;  \
                   1 3;  \
                   2 3;  \
                   3 4;  \
                   3 5;  \
                   4 5')

is_simple = np.array([1, 1, 0, 1, 1, 0, 1]).astype(bool)

edges_simple = edges[is_simple, :]

population_dict = {part: read_positions(frame_df, part, max_num_coords) \
                   for part in parts}
population, labels = pe.get_population(population_dict, part_types)


expected_lengths        = pe.lengths_lookup(edges, lengths)
expected_lengths_simple = pe.lengths_lookup(edges_simple, lengths)

dist_matrix = cdist(population, population)

ratio_func = lambda a, b: 1.0 / gen.norm_ratio(a, b)

# %%

ratio_matrix = pe.distances_to_adj_matrix(dist_matrix, labels, \
                                          expected_lengths, ratio_func)
score_matrix = score_func(ratio_matrix)

M = pe.distances_to_adj_matrix(dist_matrix, labels, \
                               expected_lengths_simple, cost_func)
G = gr.adj_matrix_to_list(M)

source_nodes = tuple(np.where(labels == 0)[0])
prev, dist = gr.dag_shortest_paths(G, G.keys(), source_nodes)


path_matrix = pe.paths_to_foot(prev, labels)

filtered_score_matrix = pe.filter_by_path(score_matrix, path_matrix,\
                                          expected_lengths_simple)


# %%


foot_A, foot_B = pe.select_best_feet(dist_matrix, filtered_score_matrix,\
                                  path_matrix, radii)

path_A, path_B = path_matrix[foot_A, :], path_matrix[foot_B, :]

pop_A, pop_B = population[path_A, :], population[path_B, :]


# %% Visual results


colours = ['blue', 'green', 'red', 'orange', 'gray', 'black']
pl.scatter_colour(pop_A, colours, part_types)

plt.xlim(-100, 100)
plt.ylim(-100, 100)

