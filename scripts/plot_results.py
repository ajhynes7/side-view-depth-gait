import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

load_dir = os.path.join('results', 'dataframes')
save_dir = os.path.join('results', 'figures')

df_total_k = pd.read_pickle(os.path.join(load_dir, 'df_total_k.pkl'))
df_total_z = pd.read_pickle(os.path.join(load_dir, 'df_total_z.pkl'))

# Columns that represent gait parameters
gait_params = df_total_k.select_dtypes(float).columns

df_trials_k = df_total_k.groupby('trial_id').median()[gait_params]
df_trials_z = df_total_z.groupby('trial_id').median()[gait_params]

df_sides_k = df_total_k.groupby(['trial_id', 'side']).median()[gait_params]
df_sides_z = df_total_z.groupby(['trial_id', 'side']).median()[gait_params]


# %% Plotting

plt.rc('text', usetex=True)

font = {'family': 'serif',
        'weight': 'bold',
        'size': 14,
        }
plt.rc('font', **font)  # pass in the font dict as kwargs


for gait_param in gait_params:

    fig, ax = plt.subplots()

    scatter_l = ax.scatter(df_sides_z.xs('L', level='side')[gait_param],
                           df_sides_k.xs('L', level='side')[gait_param],
                           c='b', s=10)

    scatter_r = ax.scatter(df_sides_z.xs('R', level='side')[gait_param],
                           df_sides_k.xs('R', level='side')[gait_param],
                           c='r', s=10)

    ax.legend([scatter_l, scatter_r], ['L', 'R'])

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]

    # Plot equality line
    ax.plot(lims, lims, 'k-')

    # Remove right and top borders
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.xlabel("Zeno Walkway [cm]")
    plt.ylabel("Kinect [cm]")

    fig.savefig(os.path.join(save_dir, 'compare_{}.pdf'.format(gait_param)),
                format='pdf', dpi=1200)
