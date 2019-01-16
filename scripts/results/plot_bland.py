"""Generate Bland-Altman and direct comparison plots."""

from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis.stats as st


def main():

    dataframe_dir = join('results', 'dataframes')
    save_dir = join('results', 'plots')

    # Load dataframes with all results
    df_total_k = pd.read_pickle(join(dataframe_dir, 'df_total_k.pkl'))
    df_total_z = pd.read_pickle(join(dataframe_dir, 'df_total_z.pkl'))

    # Columns that represent gait parameters
    gait_params = df_total_k.select_dtypes(float).columns

    df_trials_k = df_total_k.groupby('trial_id').median()
    df_trials_z = df_total_z.groupby('trial_id').median()

    for param in gait_params:

        measures_k, measures_z = df_trials_k[param], df_trials_z[param]

        means = (measures_k + measures_z) / 2
        differences = st.relative_difference(measures_k, measures_z)

        bland_alt = st.bland_altman(differences)

        # %% Bland-Altman plot

        fig, ax = plt.subplots()

        ax.scatter(means, differences, c='k', s=20)

        # Enlarge y range
        y_lims = np.array(ax.get_ylim())
        y_range = y_lims[1] - y_lims[0]
        ax.set_ylim(y_lims + 0.5 * y_range * np.array([-1, 1]))

        # Remove right and top borders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Horizontal lines for bias and limits of agreement
        ax.axhline(y=bland_alt.bias, color='k', linestyle='-')
        ax.axhline(y=bland_alt.lower_limit, color='k', linestyle=':')
        ax.axhline(y=bland_alt.upper_limit, color='k', linestyle=':')

        # Format y labels as percentages
        ax.set_yticklabels(
            [r'{:.0f}\%'.format(x * 100) for x in ax.get_yticks()])

        ax.tick_params(labelsize=25)
        ax.locator_params(nbins=5)

        plt.xlabel('Mean of two measurements [cm]', fontsize=30)
        plt.ylabel('Relative difference', fontsize=30)

        fig.savefig(
            join(save_dir, 'bland_{}.pdf'.format(param)),
            bbox_inches='tight')

        # %% Direct comparison plot

        fig, ax = plt.subplots()

        ax.scatter(measures_z, measures_k, c='k', s=20)

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]

        # Plot equality line
        ax.plot(lims, lims, 'k-')

        # Remove right and top borders
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        ax.tick_params(labelsize=25)
        ax.locator_params(nbins=5)

        plt.xlabel("Zeno Walkway [cm]", fontsize=30)
        plt.ylabel("Kinect [cm]", fontsize=30)

        fig.savefig(
            join(save_dir, 'compare_{}.pdf'.format(param)),
            bbox_inches='tight')


if __name__ == '__main__':
    main()
