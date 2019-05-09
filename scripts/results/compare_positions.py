"""Compare selected and assigned positions to ground truth."""

from os.path import join

import numpy as np
import pandas as pd

import modules.assign_sides as asi
import modules.point_processing as pp


def main():

    kinect_dir = join('data', 'kinect')

    df_hypo = pd.read_pickle(join(kinect_dir, 'df_hypo.pkl'))
    df_selected = pd.read_pickle(join(kinect_dir, 'df_selected.pkl'))
    df_assigned = pd.read_pickle(join(kinect_dir, 'df_assigned.pkl'))
    df_truth = pd.read_pickle(join(kinect_dir, 'df_truth.pkl'))

    # Truth positions on frames with head and both feet
    df_truth = df_truth.loc[:, ['HEAD', 'L_FOOT', 'R_FOOT']].dropna()

    # Trials and frames common to ground truth and selected positions
    index_intersection = df_truth.index.intersection(df_selected.index)

    index_sorted = index_intersection.sort_values(('trial_name', 'frame'))[0]

    # The assigned DataFrame had an extra index for the walking pass
    # This is dropped so the MultiIndex is the same as the other DataFrames
    df_assigned.index = df_assigned.index.droplevel('pass')

    # # Take the trials and frames shared by ground truth and the others
    df_hypo = df_hypo.loc[index_sorted]
    df_selected = df_selected.loc[index_sorted]
    df_assigned = df_assigned.loc[index_sorted]
    df_truth = df_truth.loc[index_sorted]

    # %% Obtain NumPy arrays from DataFrames

    truth_head = np.stack(df_truth.HEAD)
    truth_l = np.stack(df_truth.L_FOOT)
    truth_r = np.stack(df_truth.R_FOOT)

    selected_head = np.stack(df_selected.HEAD)
    selected_l = np.stack(df_selected.L_FOOT)
    selected_r = np.stack(df_selected.R_FOOT)

    assigned_l = np.stack(df_assigned.L_FOOT)
    assigned_r = np.stack(df_assigned.R_FOOT)

    # Match selected positions with truth
    matched_l, matched_r = pp.match_pairs(selected_l, selected_r, truth_l, truth_r)

    # %% Create modified truth

    # All foot proposals on each frame
    proposals_foot = df_hypo.apply(
        lambda row: row.population[row.labels == row.labels.max()], axis=1
    )

    truth_mod_l = pp.closest_proposals(proposals_foot, truth_l)
    truth_mod_r = pp.closest_proposals(proposals_foot, truth_r)

    # %% Convert points to 2D (since side assignment is in 2D)

    truth_2d_l = np.apply_along_axis(asi.convert_to_2d, 1, truth_l)
    truth_2d_r = np.apply_along_axis(asi.convert_to_2d, 1, truth_r)

    truth_mod_2d_l = np.apply_along_axis(asi.convert_to_2d, 1, truth_mod_l)
    truth_mod_2d_r = np.apply_along_axis(asi.convert_to_2d, 1, truth_mod_r)

    # %% Accuracies

    acc_head = pp.position_accuracy(truth_head, selected_head)

    acc_matched_l = pp.position_accuracy(matched_l, truth_l)
    acc_matched_r = pp.position_accuracy(matched_r, truth_r)

    acc_matched_mod_l = pp.position_accuracy(matched_l, truth_mod_l)
    acc_matched_mod_r = pp.position_accuracy(matched_r, truth_mod_r)

    acc_assigned_l = pp.position_accuracy(assigned_l, truth_2d_l)
    acc_assigned_r = pp.position_accuracy(assigned_r, truth_2d_r)

    acc_assigned_mod_l = pp.position_accuracy(assigned_l, truth_mod_2d_l)
    acc_assigned_mod_r = pp.position_accuracy(assigned_r, truth_mod_2d_r)

    # %% Accuracies of left and right combined
    # This is more challenging because *both* feet must be
    # within a distance d from the truth

    acc_matched = pp.double_position_accuracy(matched_l, matched_r, truth_l, truth_r)
    acc_matched_mod = pp.double_position_accuracy(matched_l, matched_r, truth_mod_l, truth_mod_r)

    acc_assigned = pp.double_position_accuracy(assigned_l, assigned_r, truth_2d_l, truth_2d_r)
    acc_assigned_mod = pp.double_position_accuracy(assigned_l, assigned_r, truth_mod_2d_l, truth_mod_2d_r)

    # %% Organize into DataFrames

    df_acc_head = pd.DataFrame(
        index=['Truth'], columns=['Head'], data=acc_head
    )

    df_acc_matched = pd.DataFrame(
        index=['Truth', 'Modified'],
        columns=['Left', 'Right', 'Both'],
        data=[
            [acc_matched_l, acc_matched_r, acc_matched],
            [acc_matched_mod_l, acc_matched_mod_r, acc_matched_mod],
        ],
    )

    df_acc_assigned = pd.DataFrame(
        index=['Truth', 'Modified'],
        columns=['Left', 'Right', 'Both'],
        data=[
            [acc_assigned_l, acc_assigned_r, acc_assigned],
            [acc_assigned_mod_l, acc_assigned_mod_r, acc_assigned_mod],
        ],
    )

    # %% Save DataFrames as LaTeX tables

    dict_frames = {
        'accuracy_head': df_acc_head,
        'accuracy_matched': df_acc_matched,
        'accuracy_assigned': df_acc_assigned,
    }

    table_dir = join('results', 'tables')

    for file_name, data_frame in dict_frames.items():

        with open(join(table_dir, file_name + '.txt'), 'w') as file:
            file.write(data_frame.round(2).to_latex())


if __name__ == '__main__':
    main()
