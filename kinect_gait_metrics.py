import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.signal import medfilt

import modules.pose_estimation as pe
import modules.gait_metrics as gm


# %% Read DataFrame

trial_id = '2014-12-22_P007_Pre_004'

load_directory = '../../MEGA/Data/Kinect Zeno/Kinect best pos'
load_path = os.path.join(load_directory, trial_id + '.pkl')

df_head_feet = pd.read_pickle(load_path)


# %% Enforce consistency of the body part sides

# Cluster frames with k means to locate the 4 walking passes
frames = df_head_feet.index.values.reshape(-1, 1)
k_means = KMeans(n_clusters=4, random_state=0).fit(frames)

# Dictionary that maps image frames to cluster labels
label_dict = dict(zip(frames.flatten(), k_means.labels_))

switch_sides = pe.consistent_sides(df_head_feet, k_means.labels_)

for frame, switch in switch_sides.iteritems():
    if switch:
        df_head_feet.loc[frame, ['L_FOOT', 'R_FOOT']] = \
            df_head_feet.loc[frame, ['R_FOOT', 'L_FOOT']].values


# %% Peak detection

foot_dist = df_head_feet.apply(lambda row: np.linalg.norm(
                               row['L_FOOT'] - row['R_FOOT']), axis=1)


# # Apply median filter to foot distance signal
# filtered = medfilt(foot_dist)
# foot_dist = pd.Series(filtered, index=foot_dist.index)


# Detect peaks in the foot distance data
# Pass in the foot distance index so the peak x-values align with the frames
peak_frames = gm.foot_dist_peaks(foot_dist, k_means.labels_, r=5)


# %% Gait metrics

gait_df = gm.gait_dataframe(df_head_feet, peak_frames, label_dict)


write_dir = '../../MEGA/Data/Kinect Zeno/Results'
write_filename = 'kinect_gait_metrics.csv'

write_path = os.path.join(write_dir, write_filename)

write_df = pd.read_csv(write_path, index_col=0)

# Fill in row of dataframe
write_df.loc[trial_id] = gait_df.mean()

write_df.to_csv(write_path)


# %% Visual results

fig, ax = plt.subplots()

plt.plot(foot_dist, color='k', linewidth=0.7)
plt.xlabel('Frame number')
plt.ylabel('Distance between feet [cm]')
ax.vlines(x=peak_frames, ymin=0, ymax=50, colors='r')

plt.show()
