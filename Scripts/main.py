import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

import sys
sys.path.insert(0, '../Modules/')
sys.path.insert(0, '../Shared code/')

import pose_estimation as pe  # noqa (ignore PEP 8 style)
from gait_metrics import gait_dataframe  # noqa
from peakdet import peakdet  # noqa
from general import mad_outliers

# %% Read DataFrame

directory = '../../../MEGA/Data/Kinect Zeno/Kinect processed'
file_name = '2014-12-22_P007_Pre_004.pkl'

load_path = os.path.join(directory, file_name)
df = pd.read_pickle(load_path)


# %% Parameters

lower_part_types = ['HEAD', 'HIP', 'UPPER_LEG', 'KNEE', 'LOWER_LEG', 'FOOT']


lengths = [62.1080, 20.1733, 14.1756, 19.4509, 20.4996]

radii = [i for i in range(0, 30, 5)]

edges = np.matrix('0 1; 1 2; 2 3; 3 4; 4 5; 3 5; 1 3')


# %% Select positions

# Find the best positions for each image frame
best_pos_series = df.apply(lambda x: pe.process_frame(x.to_dict(),
                           lower_part_types, edges, lengths, radii), axis=1)

n_frames = len(df)
for f in range(n_frames):
    row = df.loc[f]

    if f == 1147:
        print('h')

    pe.process_frame(row.to_dict(), lower_part_types, edges, lengths, radii)


# %% Extract head and foot positions

# Each row i is a tuple containing the best positions for frame i
# Split each tuple into columns of a DataFrame
df_best_pos = pd.DataFrame(best_pos_series.values.tolist(),
                           columns=['Side A', 'Side B'])

df_best_pos = df_best_pos.dropna()

# Extract the head and feet positions
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

y_foot_L_filtered = mad_outliers(y_foot_L, 2)
y_foot_R_filtered = mad_outliers(y_foot_R, 2)

good_frame_L = ~np.isnan(y_foot_L_filtered)
good_frame_R = ~np.isnan(y_foot_R_filtered)

df_head_feet = df_head_feet[good_frame_L & good_frame_R]


# %% Gait metrics


foot_dist = df_head_feet.apply(lambda row: np.linalg.norm(
                               row['L_FOOT'] - row['R_FOOT']), axis=1)

# Detect peaks in the foot distance data
# Pass in the foot distance index so the peak x-values align with the frames
dist_peaks, _ = peakdet(foot_dist, 20, foot_dist.index)

peak_frames, peak_values = dist_peaks[:, 0], dist_peaks[:, 1]
peak_frames = peak_frames.astype(int)

# Cluster the peak frame numbers, to assign frame to passes
k_means = KMeans(n_clusters=4, random_state=0).fit(peak_frames.reshape(-1, 1))


gait_df = gait_dataframe(df_head_feet, peak_frames, k_means.labels_)


# %% Visual results

plt.figure()
plt.plot(foot_dist, color='k', linewidth=0.7)
plt.scatter(peak_frames, peak_values, cmap='Set1', c=k_means.labels_)
plt.xlabel('Frame number')
plt.ylabel('Distance between feet [cm]')
plt.show()

