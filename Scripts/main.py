import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

import sys
sys.path.insert(0, '../Modules/')
sys.path.insert(0, '../Shared code/')

import pose_estimation as pe
from gait_metrics import gait_dataframe
from peakdet import peakdet


# %% Read dataframe

directory = '../../../MEGA/Data/Kinect Zeno/Kinect processed'
file_name = '2014-12-22_P007_Pre_004.pkl'

load_path = os.path.join(directory, file_name)
df = pd.read_pickle(load_path)

# %% Parameters

lower_part_types = ['HEAD', 'HIP', 'UPPER_LEG', 'KNEE', 'LOWER_LEG', 'FOOT']
lengths = [63.9626,   19.3718,   12.8402,   22.0421,   20.5768]
radii = [i for i in range(0, 30, 5)]

edges = np.matrix('0 1;  \
                   1 2;  \
                   2 3;  \
                   3 4;  \
                   4 5;  \
                   3 5;  \
                   1 3')

# %%

func = lambda x: pe.process_frame(x.to_dict(), lower_part_types, edges, lengths, radii)
best_pos_series = df.apply(func, axis=1)


# Each row i is a tuple containing the best positions for frame i
# Split each tuple into columns of a dataframe
df_best_pos = pd.DataFrame(best_pos_series.values.tolist(),\
                           columns=['Side A', 'Side B'])

df_best_pos = df_best_pos.dropna()

# Extract the head and feet positions
head_pos = df_best_pos['Side A'].apply(lambda row: row[0, :])
L_foot_pos = df_best_pos['Side A'].apply(lambda row: row[-1, :])
R_foot_pos = df_best_pos['Side B'].apply(lambda row: row[-1, :])

# Combine into new dataframe
df_head_feet = pd.concat([head_pos, L_foot_pos, R_foot_pos], axis=1) 
df_head_feet.columns = ['HEAD', 'L_FOOT', 'R_FOOT']
df_head_feet.index.name = 'Frame'


dist_func = lambda row: np.linalg.norm(row['L_FOOT'] - row['R_FOOT'])
foot_dist = df_head_feet.apply(dist_func, axis=1)


# %% Gait metrics


# Detect peaks in the foot distance data
# Pass in the foot distance index so the peak x-values align with the frames
dist_peaks, _ = peakdet(foot_dist, 20, foot_dist.index)

peak_frames, peak_vals = dist_peaks[:, 0], dist_peaks[:, 1]
peak_frames = peak_frames.astype(int)

# Cluster the peak frame numbers, to assign frame to passes
kmeans = KMeans(n_clusters=4, random_state=0).fit(peak_frames.reshape(-1, 1))


gait_df = gait_dataframe(df_head_feet, peak_frames, kmeans.labels_)


# %% Visual results

plt.figure()
plt.plot(foot_dist, color='k', linewidth=0.7)
plt.scatter(peak_frames, peak_vals, cmap='Set1', c=kmeans.labels_)
plt.xlabel('Frame number')
plt.ylabel('Distance between feet [cm]')
plt.show()

