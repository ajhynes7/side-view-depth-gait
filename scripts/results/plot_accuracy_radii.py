"""Plot accuracy vs radii used to select feet."""

from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import modules.point_processing as pp


def main():

    df_radii = pd.read_pickle(join('data', 'kinect', 'df_radii.pkl'))
    df_truth = pd.read_pickle(join('data', 'kinect', 'df_truth.pkl'))
    df_hypo = pd.read_pickle(join('data', 'kinect', 'df_hypo.pkl'))

    # Truth positions on frames with head and both feet
    df_truth = df_truth.loc[:, ['HEAD', 'L_FOOT', 'R_FOOT']].dropna()

    # Trials and frames common to ground truth and selected positions
    index_intersection = df_truth.index.intersection(df_radii.loc[0].index)

    index_sorted = index_intersection.sort_values(('trial_name', 'frame'))[0]

    df_truth = df_truth.loc[index_sorted]
    df_hypo = df_hypo.loc[index_sorted]

    truth_l = np.stack(df_truth.L_FOOT)
    truth_r = np.stack(df_truth.R_FOOT)

    # %% Create modified truth

    # All foot proposals on each frame
    proposals_foot = df_hypo.apply(lambda row: row.population[row.labels == row.labels.max()], axis=1)

    truth_mod_l = pp.closest_proposals(proposals_foot, truth_l)
    truth_mod_r = pp.closest_proposals(proposals_foot, truth_r)

    truth_mod_accs = []

    for radius, df_radius in df_radii.groupby(level=0):

        # Drop the first level to have same MultiIndex as df_truth
        df_radius.index = df_radius.index.droplevel(0)
        df_radius = df_radius.loc[index_sorted]

        selected_l = np.stack(df_radius.L_FOOT)
        selected_r = np.stack(df_radius.R_FOOT)

        # Match selected positions with truth
        matched_l, matched_r = pp.match_pairs(selected_l, selected_r, truth_l, truth_r)

        truth_mod_accs.append(pp.double_position_accuracy(matched_l, matched_r, truth_mod_l, truth_mod_r) * 100)

    # %% Create plot of accuracy vs radii

    fig, ax = plt.subplots()

    radii = df_radii.index.get_level_values(0).unique()
    ax.plot(radii, truth_mod_accs, '-o', c='k')

    ax.set_aspect(0.1)

    ax.set_xlabel('Radius [cm]')
    ax.set_ylabel(r'Accuracy [\%]')

    fig.savefig(join('results', 'plots', 'accuracy_radii.pdf'), dpi=1200)


if __name__ == '__main__':
    main()
