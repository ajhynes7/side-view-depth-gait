import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

import sys
sys.path.insert(0, '../Modules/')
sys.path.insert(0, '../Shared code/')

from peakdet import peakdet
import pose_estimation as pe
import read_data as rd


max_num_coords = 60

file_path = '../../../MEGA/Data/Kinect trials/2014-12-22_P007_Pre_004.txt'
column_names = [i for i in range(-2, max_num_coords)]

df = pd.read_csv(file_path, skiprows=22, header=None,\
                 names=column_names, sep='\t', engine='python')

# Change some column names
df.rename(columns={-2: 'Frame', -1: 'Part'}, inplace=True)

# Replace any strings with nan in the Frame column
df.Frame = df.Frame.replace(r'[^0-9]', np.nan, regex=True)

df.Frame = pd.to_numeric(df.Frame)


# %% Inputs

parts = df.groupby('Part').groups.keys()  # Part names

n_frames = int(max(df.Frame)) + 1

part_types = ['HEAD', 'HIP', 'UPPER_LEG', 'KNEE', 'LOWER_LEG', 'FOOT']
lengths = [63.9626,   19.3718,   12.8402,   22.0421,   20.5768]
radii = [i for i in range(0, 30, 5)]

edges = np.matrix('0 1;  \
                   1 2;  \
                   2 3;  \
                   3 4;  \
                   4 5;  \
                   3 5;  \
                   1 3')


#%%  Process all frames in dataset

chosen_pos_dict = {k: {} for k in ['HEAD', 'L_FOOT', 'R_FOOT']}

start_frame, end_frame = 600, 700

total = 0
for f in range(600, 700):

    # Dataframe for current image frame
    df_current = df[df.Frame == f]


    pop_dict = {part: rd.read_positions(df_current, part, max_num_coords)\
                for part in parts}

    t = time()
    pop_A, pop_B = pe.process_frame(df_current, pop_dict, part_types, edges,\
                                    lengths, radii)

    total += time() - t

    chosen_pos_dict['HEAD'][f]    = pop_A[0, :]
    chosen_pos_dict['L_FOOT'][f]  = pop_A[-1, :]
    chosen_pos_dict['R_FOOT'][f]  = pop_B[-1, :]



# %% Gait metrics

chosen_pos_df = pd.DataFrame(chosen_pos_dict)

# Specify order of columns
chosen_pos_df = chosen_pos_df[['HEAD', 'L_FOOT', 'R_FOOT']]

foot_dist_func = lambda row: np.linalg.norm(row[1] - row[2])
chosen_pos_df['Foot_dist'] = chosen_pos_df.apply(foot_dist_func, axis=1)

dist_peaks, _ = peakdet(chosen_pos_df.Foot_dist, 0.01)
dist_peaks[:, 0] += start_frame


# %% Visual results


#colours = ['blue', 'green', 'red', 'orange', 'gray', 'black']
#pl.scatter_colour(pop_A, colours, part_types)

plt.figure()
plt.scatter(pop_A[:, 0], pop_A[:, 1], color='b')
plt.scatter(pop_B[:, 0], pop_B[:, 1], color='r')
plt.xlim(-100, 100)
plt.ylim(-100, 100)

plt.figure()
plt.plot(chosen_pos_df.Foot_dist)
plt.scatter(dist_peaks[:, 0], dist_peaks[:, 1], color='r', alpha=1)