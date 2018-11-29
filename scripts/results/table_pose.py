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
    df_selected = df_selected.loc[index_intersect]
    df_assigned = df_assigned.loc[index_intersect]
    frames = df_truth.index

    # %% Create modified truth

    df_truth_modified = pd.DataFrame(index=frames, columns=foot_parts)

    for frame in frames:

        foot_proposals = df_hypo.loc[frame, 'FOOT']
        true_foot_l, true_foot_r = df_truth.loc[frame, foot_parts]

        proposal_closest_l, _ = pp.closest_point(foot_proposals,
                                                 true_foot_l.reshape(1, -1))
        proposal_closest_r, _ = pp.closest_point(foot_proposals,
                                                 true_foot_r.reshape(1, -1))

        df_truth_modified.loc[frame, foot_parts[0]] = proposal_closest_l
        df_truth_modified.loc[frame, foot_parts[1]] = proposal_closest_r

    # %% Match selected foot positions to truth

    df_selected_matched = pd.DataFrame(index=frames, columns=foot_parts)

    for frame in frames:

        points_selected = np.stack(df_selected.loc[frame][foot_parts])
        points_truth = np.stack(df_truth_modified.loc[frame])

        matched_l, matched_r = pp.correspond_points(points_truth,
                                                    points_selected)

        df_selected_matched.loc[frame, foot_parts[0]] = matched_l
        df_selected_matched.loc[frame, foot_parts[1]] = matched_r

    # %% Accuracies of foot positions matched to truth

    acc_matched_l = pp.position_accuracy(
        np.stack(df_selected_matched.L_FOOT), np.stack(df_truth.L_FOOT))
    acc_matched_r = pp.position_accuracy(
        np.stack(df_selected_matched.R_FOOT), np.stack(df_truth.R_FOOT))

    acc_matched_mod_l = pp.position_accuracy(
        np.stack(df_selected_matched.L_FOOT),
        np.stack(df_truth_modified.L_FOOT))
    acc_matched_mod_r = pp.position_accuracy(
        np.stack(df_selected_matched.R_FOOT),
        np.stack(df_truth_modified.R_FOOT))

    df_acc_matched = pd.DataFrame(
        index=['Left', 'Right'],
        columns=['Truth', 'Modified'],
        data=[[acc_matched_l, acc_matched_mod_l],
              [acc_matched_r, acc_matched_mod_r]])

    # %% Accuracies of foot positions after assigning sides

    # Convert points to 2D (the sides are assigned to 2D foot points)
    df_truth_2d = df_truth.applymap(lambda point: point[[2, 0]])
    df_truth_modified_2d = df_truth_modified.applymap(
        lambda point: point[[2, 0]])

    acc_assigned_l = pp.position_accuracy(
        np.stack(df_assigned.L_FOOT), np.stack(df_truth_2d.L_FOOT))
    acc_assigned_r = pp.position_accuracy(
        np.stack(df_assigned.R_FOOT), np.stack(df_truth_2d.R_FOOT))

    acc_assigned_mod_l = pp.position_accuracy(
        np.stack(df_assigned.L_FOOT), np.stack(df_truth_modified_2d.L_FOOT))
    acc_assigned_mod_r = pp.position_accuracy(
        np.stack(df_assigned.R_FOOT), np.stack(df_truth_modified_2d.R_FOOT))

    df_acc_assigned = pd.DataFrame(
        index=['Left', 'Right'],
        columns=['Truth', 'Modified'],
        data=[[acc_assigned_l, acc_assigned_mod_l],
              [acc_assigned_r, acc_assigned_mod_r]])

    print(df_acc_matched)
    print(df_acc_assigned)


if __name__ == '__main__':
    main()
