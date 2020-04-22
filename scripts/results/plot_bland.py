"""Create Bland-Altman plots."""

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis.stats as st


def get_units(param_gait: str):
    """Return the appropriate units for the gait parameter."""

    if param_gait == 'stance_percentage':
        units = r'\%'

    elif param_gait == 'stride_time':
        units = 's'

    elif param_gait == 'stride_velocity':
        units = 'cm/s'

    else:
        units = 'cm'

    return units


def main():

    dir_plots = join('results', 'plots')

    if not os.path.exists(dir_plots):
        os.makedirs(dir_plots)

    # Gait parameters from all trials with matching IDs.
    df_matched_k = pd.read_pickle(join('data', 'kinect', 'df_matched.pkl'))
    df_matched_z = pd.read_pickle(join('data', 'zeno', 'df_matched.pkl'))

    gait_params = df_matched_k.columns

    # %% Calculate results per trial.

    df_trials_k = df_matched_k.groupby('trial_id').median()
    df_trials_z = df_matched_z.groupby('trial_id').median()

    dict_bland = {}

    for param in gait_params:

        measures_k = df_trials_k[param]
        measures_z = df_trials_z[param]

        measures = np.column_stack((measures_k, measures_z))

        means = measures.mean(axis=1)
        differences = st.relative_difference(measures_k, measures_z)
        bland_alt = st.bland_altman(differences)

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

        units = get_units(param)

        plt.xlabel(f"Mean of two measurements [{units}]", fontsize=30)
        plt.ylabel(r"Relative difference [\%]", fontsize=30)
        plt.tight_layout()

        fig_1.savefig(join(dir_plots, f'bland_{param}.png'))

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

        plt.xlabel(f"Zeno Walkway [{units}]", fontsize=30)
        plt.ylabel(f"Kinect [{units}]", fontsize=30)
        plt.tight_layout()

        fig_2.savefig(join(dir_plots, f'compare_{param}.png'))


if __name__ == '__main__':
    main()
