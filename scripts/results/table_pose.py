"""Compute metrics for pose estimation."""

from os.path import join

import numpy as np
import pandas as pd

import modules.point_processing as pp


def combine_dataframes(load_dir, trial_names):
    """Combine dataframes from different walking trials."""
    dataframe_dict = {}

    for trial_name in trial_names:
        dataframe_dict[trial_name] = pd.read_pickle(
            join(load_dir, trial_name) + '.pkl')

    return pd.concat(dataframe_dict)


def main():

    kinect_dir = join('data', 'kinect')

    df_truth = pd.read_pickle('results/dataframes/df_truth.pkl')
    trial_names = df_truth.index.get_level_values(0).unique().values

    df_hypo = combine_dataframes(
        join(kinect_dir, 'processed', 'hypothesis'), trial_names)
    df_selected = combine_dataframes(join(kinect_dir, 'best_pos'), trial_names)
    df_assigned = combine_dataframes(join(kinect_dir, 'assigned'), trial_names)

    foot_parts = ['L_FOOT', 'R_FOOT']

    # Truth positions on frames with head and both feet
    df_truth = df_truth.loc[:, ['HEAD'] + foot_parts].dropna()

    # Drop index level for walking pass
    # This gives it the same index as the other dataframes
    df_assigned.index = df_assigned.index.droplevel(1)

    # %% Take frames with truth and selected positions

    index_intersect = df_truth.index.intersection(df_selected.index)

    df_truth = df_truth.loc[index_intersect]
    df_hypo = df_hypo.loc[index_intersect]
    df_selected = df_selected.loc[index_intersect]
    df_assigned = df_assigned.loc[index_intersect]

    # %% Obtain numpy arrays from dataframes

    proposals = df_hypo.FOOT.values

    truth_l = np.stack(df_truth.L_FOOT)
    truth_r = np.stack(df_truth.R_FOOT)

    selected_l = np.stack(df_selected.L_FOOT)
    selected_r = np.stack(df_selected.R_FOOT)

    assigned_l = np.stack(df_assigned.L_FOOT)
    assigned_r = np.stack(df_assigned.R_FOOT)

    # Match selected positions with truth
    matched_l, matched_r = pp.match_pairs(selected_l, selected_r, truth_l,
                                          truth_r)

    # Create modified truth
    truth_mod_l = pp.closest_proposals(proposals, np.stack(df_truth.L_FOOT))
    truth_mod_r = pp.closest_proposals(proposals, np.stack(df_truth.R_FOOT))

    # %% Convert points to 2D (since side assignment is in 2D)

    truth_2d_l = truth_l[:, [2, 0]]
    truth_2d_r = truth_r[:, [2, 0]]

    truth_mod_2d_l = truth_mod_l[:, [2, 0]]
    truth_mod_2d_r = truth_mod_r[:, [2, 0]]

    # %% Calculate accuracies

    acc_matched_l = pp.position_accuracy(matched_l, truth_l)
    acc_matched_r = pp.position_accuracy(matched_r, truth_r)

    acc_matched_mod_l = pp.position_accuracy(matched_l, truth_mod_l)
    acc_matched_mod_r = pp.position_accuracy(matched_r, truth_mod_r)

    acc_assigned_l = pp.position_accuracy(assigned_l, truth_2d_l)
    acc_assigned_r = pp.position_accuracy(assigned_r, truth_2d_r)

    acc_assigned_mod_l = pp.position_accuracy(assigned_l, truth_mod_2d_l)
    acc_assigned_mod_r = pp.position_accuracy(assigned_r, truth_mod_2d_r)

    df_acc_matched = pd.DataFrame(
        index=['Left', 'Right'],
        columns=['Truth', 'Modified'],
        data=[[acc_matched_l, acc_matched_mod_l],
              [acc_matched_r, acc_matched_mod_r]])

    df_acc_assigned = pd.DataFrame(
        index=['Left', 'Right'],
        columns=['Truth', 'Modified'],
        data=[[acc_assigned_l, acc_assigned_mod_l],
              [acc_assigned_r, acc_assigned_mod_r]])

    print(pp.double_position_accuracy(matched_l, matched_r, truth_l, truth_r))
    print(pp.double_position_accuracy(matched_l, matched_r, truth_mod_l, truth_mod_r))

    print(df_acc_matched)
    print(df_acc_assigned)


if __name__ == '__main__':
    main()
