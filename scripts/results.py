"""Calculate results of comparing Kinect and Zeno gait parameters."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

import analysis.stats as st


def combine_trials(load_dir, matched_file_names):
    """Combine dataframes from all matched walking trials."""
    list_dfs = []

    for i, file_name in enumerate(matched_file_names):

        file_path = os.path.join(load_dir, file_name + '.pkl')

        df_device = pd.read_pickle(file_path)
        df_device['trial_id'] = i
        df_device['file_name'] = file_name

        list_dfs.append(df_device)

    return pd.concat(list_dfs).reset_index(drop=True)


load_dir_k = os.path.join('data', 'kinect', 'gait_params')
load_dir_z = os.path.join('data', 'zeno', 'gait_params')
match_dir = os.path.join('data', 'matching')

df_match = pd.read_csv(os.path.join(match_dir, 'match_kinect_zeno.csv'))


# Convert match table to dictionary for easy file matching
dict_match = {x.zeno: x.kinect for x in df_match.itertuples()}

matched_file_names_k = list(dict_match.values())
matched_file_names_z = list(dict_match.keys())

df_total_k = combine_trials(load_dir_k, matched_file_names_k)
df_total_z = combine_trials(load_dir_z, matched_file_names_z)

# Save total dataframes for easy analysis
df_total_k.to_pickle(os.path.join('results', 'dataframes', 'df_total_k.pkl'))
df_total_z.to_pickle(os.path.join('results', 'dataframes', 'df_total_z.pkl'))

# Columns that represent gait parameters
gait_params = df_total_k.select_dtypes(float).columns

# Remove negative values from Zeno data
df_total_z = df_total_z.applymap(
    lambda x: np.nan if isinstance(x, float) and x < 0 else x)

df_trials_k = df_total_k.groupby('trial_id').median()[gait_params]
df_trials_z = df_total_z.groupby('trial_id').median()[gait_params]

df_sides_k = df_total_k.groupby(['trial_id', 'side']).median()[gait_params]
df_sides_z = df_total_z.groupby(['trial_id', 'side']).median()[gait_params]

# Calculate results
funcs = {
    'pearson': lambda a, b: pearsonr(a, b)[0],
    'spearman': lambda a, b: spearmanr(a, b)[0],
    'abs_rel_error':
    lambda a, b: st.relative_error(a, b, absolute=True).mean(),
    'bias': lambda a, b: st.bland_altman(st.relative_difference(a, b)).bias,
    'range': lambda a, b: st.bland_altman(st.relative_difference(a, b)).range_,
}

df_results = st.compare_measurements(df_trials_k, df_trials_z, funcs)
df_results.to_csv(
    os.path.join('results', 'spreadsheets', 'results_grouped.csv'))\


# %% Plotting

save_dir = os.path.join('results', 'figures')

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
