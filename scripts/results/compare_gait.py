"""Compare the gait parameters of the Kinect and Zeno Walkway."""

from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis.stats as st


def main():

    dir_plots = join('results', 'plots')

    # Gait parameters from all trials with matching IDs.
    df_matched_k = pd.read_pickle(join('data', 'kinect', 'df_matched.pkl'))
    df_matched_z = pd.read_pickle(join('data', 'zeno', 'df_matched.pkl'))

    gait_params = df_matched_k.columns

    # %% Calculate results per trial.

    df_trials_k = df_matched_k.groupby('trial_id').median()
    df_trials_z = df_matched_z.groupby('trial_id').median()

    dict_icc, dict_bland = {}, {}

    for param in gait_params:

        measures_k = df_trials_k[param]
        measures_z = df_trials_z[param]

        measures = np.column_stack((measures_k, measures_z))

        icc_21 = st.icc(measures, form=(2, 1))
        icc_31 = st.icc(measures, form=(3, 1))

        means = measures.mean(axis=1)
        differences = st.relative_difference(measures_k, measures_z)
        bland_alt = st.bland_altman(differences)

        dict_icc[param] = {'ICC_21': icc_21, 'ICC_31': icc_31}
        dict_bland[param] = bland_alt._asdict()

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
        ax_1.set_yticklabels([f'{(x * 100):.0f}' for x in ax_1.get_yticks()])

        # Remove right and top borders
        ax_1.spines['right'].set_visible(False)
        ax_1.spines['top'].set_visible(False)

        plt.xlabel('Mean of two measurements [cm]', fontsize=30)
        plt.ylabel(r'Relative difference [\%]', fontsize=30)
        plt.tight_layout()

        fig_1.savefig(join(dir_plots, 'bland_{}.png'.format(param)))

        # %% Direct comparison plots

        fig_2, ax_2 = plt.subplots()

        ax_2.scatter(measures_z, measures_k, c='k', s=20)

        lims = [np.min([ax_2.get_xlim(), ax_2.get_ylim()]), np.max([ax_2.get_xlim(), ax_2.get_ylim()])]

        # Plot equality line
        ax_2.plot(lims, lims, 'k-')

        # Remove right and top borders
        ax_2.spines['right'].set_visible(False)
        ax_2.spines['top'].set_visible(False)

        ax_2.tick_params(labelsize=25)

        plt.xlabel("Zeno Walkway [cm]", fontsize=30)
        plt.ylabel("Kinect [cm]", fontsize=30)
        plt.tight_layout()

        fig_2.savefig(join(dir_plots, 'compare_{}.png'.format(param)))

    df_icc = pd.DataFrame.from_dict(dict_icc, orient='index')
    df_bland = pd.DataFrame.from_dict(dict_bland, orient='index')

    # %% Calculate results for left and right sides.

    df_sides_k = df_matched_k.groupby(['trial_id', 'side']).median()
    df_sides_z = df_matched_z.groupby(['trial_id', 'side']).median()

    dict_icc_sides, dict_bland_sides = {}, {}

    for side in ['L', 'R']:

        df_side_k = df_sides_k[df_sides_k.index.get_level_values('side') == side]
        df_side_z = df_sides_z[df_sides_z.index.get_level_values('side') == side]

        for param in gait_params:

            measures_k = df_side_k[param]
            measures_z = df_side_z[param]

            measures = np.column_stack((measures_k, measures_z))

            icc_21 = st.icc(measures, form=(2, 1))
            icc_31 = st.icc(measures, form=(3, 1))

            differences = st.relative_difference(measures_k, measures_z)
            bland_alt = st.bland_altman(differences)

            dict_icc_sides[(param, side)] = {'ICC_21': icc_21, 'ICC_31': icc_31}
            dict_bland_sides[(param, side)] = bland_alt._asdict()

    df_icc_sides = pd.DataFrame.from_dict(dict_icc_sides, orient='index').unstack()
    df_bland_sides = pd.DataFrame.from_dict(dict_bland_sides, orient='index').unstack()

    # %% Save results as LaTeX tables.

    dict_dfs = {
        'icc': df_icc,
        'bland_altman': df_bland,
        'icc_sides': df_icc_sides,
        'bland_altman_sides': df_bland_sides,
    }

    for file_name, df in dict_dfs.items():

        with open(join('results', 'tables', file_name + '.txt'), 'w') as file:
            file.write(df.round(2).to_latex())


if __name__ == '__main__':
    main()
