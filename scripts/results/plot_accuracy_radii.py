"""Plot accuracy vs radii used to select feet."""

import glob
from os.path import basename, join, splitext

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import modules.point_processing as pp
from scripts.results.table_pose import combine_dataframes


def main():

    kinect_dir = join('data', 'kinect')

    df_truth = pd.read_pickle('results/dataframes/df_truth.pkl')
    trial_names = df_truth.index.levels[0]

    df_hypo = combine_dataframes(
        join(kinect_dir, 'processed', 'hypothesis'), trial_names
    )
    df_selected = combine_dataframes(join(kinect_dir, 'best_pos'), trial_names)

    foot_parts = ['L_FOOT', 'R_FOOT']

    # Truth positions on frames with head and both feet
    df_truth = df_truth.loc[:, ['HEAD'] + foot_parts].dropna()

    # Take frames with truth and selected positions
    index_intersect = df_truth.index.intersection(df_selected.index)

    df_truth = df_truth.loc[index_intersect]
    df_hypo = df_hypo.loc[index_intersect]

    truth_l = np.stack(df_truth.L_FOOT)
    truth_r = np.stack(df_truth.R_FOOT)

    # Create modified truth
    proposals = df_hypo.FOOT.values
    truth_mod_l = pp.closest_proposals(proposals, np.stack(df_truth.L_FOOT))
    truth_mod_r = pp.closest_proposals(proposals, np.stack(df_truth.R_FOOT))

    # %% Combine dataframes for different radii

    file_paths = sorted(glob.glob(join(kinect_dir, 'best_pos_radii', '*.pkl')))

    dataframe_dict = {}
    for file_path in file_paths:
        file_name = splitext(basename(file_path))[0]
        trial_name, radius = file_name.split('_radius_')

        dataframe_dict[(float(radius), trial_name)] = pd.read_pickle(file_path)

    df_radii = pd.concat(dataframe_dict)
    df_radii.index.names = ['radius', 'trial_name', 'frame']

    radii = df_radii.index.levels[0]

    # %% Calculate accuracy for different radii

    radii = np.arange(0, 11, 1)

    truth_accs, truth_mod_accs = [], []

    for radius in radii:

        df_selected = df_radii.loc[radius].loc[index_intersect]

        selected_l = np.stack(df_selected.L_FOOT)
        selected_r = np.stack(df_selected.R_FOOT)

        # Match selected positions with truth
        matched_l, matched_r = pp.match_pairs(
            selected_l, selected_r, truth_l, truth_r
        )

        truth_accs.append(
            pp.double_position_accuracy(
                matched_l, matched_r, truth_l, truth_r
            ) * 100
        )

        truth_mod_accs.append(
            pp.double_position_accuracy(
                matched_l, matched_r, truth_mod_l, truth_mod_r
            ) * 100
        )

    # %% Create plot of accuracy vs radii

    fig = plt.figure()

    plt.plot(radii, truth_accs, '-o', c='b')
    plt.plot(radii, truth_mod_accs, '-o', c='r')

    plt.xlabel('Radius [cm]')
    plt.ylabel(r'Accuracy [\%]')

    plt.legend(['Truth', 'Modified'])

    fig.savefig(join('results', 'plots', 'accuracy_radii.pdf'), dpi=1200)


if __name__ == '__main__':
    main()
