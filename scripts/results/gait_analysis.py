"""Create plots and tables for comparing Kinect to Zeno Walkway."""

from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis.stats as st


def main():

    dir_plots = join('results', 'plots')
    dir_tables = join('results', 'tables')
    dir_dataframes = join('results', 'dataframes')

    # Load all results
    df_total_k = pd.read_pickle(join(dir_dataframes, 'df_total_k.pkl'))
    df_total_z = pd.read_pickle(join(dir_dataframes, 'df_total_z.pkl'))

    gait_params = df_total_k.select_dtypes(float).columns

    df_trials_k = df_total_k.groupby('trial_id').median()
    df_trials_z = df_total_z.groupby('trial_id').median()

    icc_21, icc_31 = [], []
    bland_alt_tuples = []

    for param in gait_params:

        measures_k = df_trials_k[param]
        measures_z = df_trials_z[param]

        means = (measures_k + measures_z) / 2
        differences = st.relative_difference(measures_k, measures_z)

        bland_alt = st.bland_altman(differences)

        # %% Bland-Altman plot

        fig_1, ax_1 = plt.subplots()

        ax_1.scatter(means, differences, c='k', s=20)

        # Horizontal lines for bias and limits of agreement
        ax_1.axhline(y=bland_alt.bias, color='k', linestyle='-')
        ax_1.axhline(y=bland_alt.lower_limit, color='k', linestyle=':')
        ax_1.axhline(y=bland_alt.upper_limit, color='k', linestyle=':')

        # Enlarge y range
        y_lims = np.array(ax_1.get_ylim())
        y_range = y_lims[1] - y_lims[0]
        ax_1.set_ylim(y_lims + 0.5 * y_range * np.array([-1, 1]))

        # Reduce number of ticks on the axes, so that figure can be small
        ax_1.locator_params(nbins=5)

        ax_1.tick_params(labelsize=25)  # Increase font of tick labels

        # Format y labels as percentages.
        # This line needs to be after the others,
        # or else the y-axis may be incorrect.
        ax_1.set_yticklabels([f'{(x * 100):.0f}\%' for x in ax_1.get_yticks()])

        # Remove right and top borders
        ax_1.spines['right'].set_visible(False)
        ax_1.spines['top'].set_visible(False)

        plt.xlabel('Mean of two measurements [cm]', fontsize=30)
        plt.ylabel('Relative difference', fontsize=30)
        plt.tight_layout()

        fig_1.savefig(join(dir_plots, 'bland_{}.png'.format(param)))

        # %% Direct comparison plot

        fig_2, ax_2 = plt.subplots()

        ax_2.scatter(measures_z, measures_k, c='k', s=20)

        lims = [
            np.min([ax_2.get_xlim(), ax_2.get_ylim()]),
            np.max([ax_2.get_xlim(), ax_2.get_ylim()]),
        ]

        # Plot equality line
        ax_2.plot(lims, lims, 'k-')

        # Remove right and top borders
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        ax_2.tick_params(labelsize=25)

        plt.xlabel("Zeno Walkway [cm]", fontsize=30)
        plt.ylabel("Kinect [cm]", fontsize=30)
        plt.tight_layout()

        fig_2.savefig(join(dir_plots, 'compare_{}.png'.format(param)))

        # %% Collect data for Bland and ICC tables

        bland_alt_tuples.append(bland_alt)

        measures = np.column_stack((measures_k, measures_z))
        icc_21.append(st.icc(measures, type_icc=(2, 1)))
        icc_31.append(st.icc(measures, type_icc=(3, 1)))

    df_bland = pd.DataFrame.from_records(
        bland_alt_tuples, index=gait_params, columns=bland_alt._fields
    )

    df_icc = pd.DataFrame(
        {'ICC(2, 1)': icc_21, 'ICC(3, 1)': icc_31}, index=gait_params
    )

    with open(join(dir_tables, 'bland_altman.txt'), 'w') as file:
        file.write(np.round(df_bland, 3).to_latex())

    with open(join(dir_tables, 'icc.txt'), 'w') as file:
        file.write(np.round(df_icc, 3).to_latex())


if __name__ == '__main__':
    main()
