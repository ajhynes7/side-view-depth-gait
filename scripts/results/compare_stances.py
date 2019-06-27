"""Compare left/right stance positions to ground truth."""

from os.path import join

import numpy as np
import pandas as pd
import xarray as xr
from dpcontracts import ensure

import modules.phase_detection as pde
import modules.point_processing as pp
import modules.side_assignment as sa
import modules.xarray_funcs as xrf


@ensure("The arrays must have the same shape", lambda _, result: result[0].shape == result[1].shape)
def match_frames(list_passes_x, df_truth_trial_x):

    points_trial_x = xr.concat(list_passes_x, dim='frames')

    truth_trial_x = xr.DataArray(
        np.stack(df_truth_trial_x), coords=(df_truth_trial_x.index.values, range(3)), dims=('frames', 'cols')
    )

    # Take frames common to ground truth and selected stance positions.
    frames_int_x = np.intersect1d(points_trial_x.frames, truth_trial_x.frames)

    points_trial_x = points_trial_x.loc[frames_int_x]
    truth_trial_x = truth_trial_x.loc[frames_int_x]

    return points_trial_x, truth_trial_x


def main():

    kinect_dir = join('data', 'kinect')

    df_hypo = pd.read_pickle(join(kinect_dir, 'df_hypo.pkl'))
    df_selected = pd.read_pickle(join(kinect_dir, 'df_selected.pkl'))
    df_selected_passes = pd.read_pickle(join(kinect_dir, 'df_selected_passes.pkl'))
    df_truth = pd.read_pickle(join(kinect_dir, 'df_truth.pkl'))

    # Trials and frames common to ground truth and selected positions
    index_intersection = df_truth.index.intersection(df_selected.index)

    index_sorted, _ = index_intersection.sort_values(('trial_name', 'frame'))

    # Take the trials and frames shared by ground truth and the others
    df_hypo = df_hypo.loc[index_sorted]
    df_selected = df_selected.loc[index_sorted]
    df_truth = df_truth.loc[index_sorted]

    trial_names = index_sorted.get_level_values(level=0).unique()

    list_points_l, list_points_r = [], []
    list_truth_l, list_truth_r = [], []
    list_truth_mod_l, list_truth_mod_r = [], []

    for trial_name in trial_names:

        df_hypo_trial = df_hypo.loc[trial_name]
        df_truth_trial = df_truth.loc[trial_name]
        df_selected_trial = df_selected_passes.loc[trial_name]

        list_passes_l, list_passes_r = [], []

        for num_pass, df_pass in df_selected_trial.groupby(level=0):

            frames = df_pass.reset_index().frame.values

            points_head = np.stack(df_pass.HEAD)
            points_a = np.stack(df_pass.L_FOOT)
            points_b = np.stack(df_pass.R_FOOT)

            points_stacked = xr.DataArray(
                np.dstack((points_a, points_b, points_head)),
                coords=(frames, range(3), ['points_a', 'points_b', 'points_head']),
                dims=('frames', 'cols', 'layers'),
            )

            basis, points_grouped_inlier = sa.compute_basis(points_stacked)

            labels_grouped_l, labels_grouped_r = pde.label_stances(points_grouped_inlier, basis)

            points_pass_l = points_grouped_inlier[labels_grouped_l != -1]
            points_pass_r = points_grouped_inlier[labels_grouped_r != -1]

            # Ensure all frames are unique by taking mean of points on the same frame.
            points_pass_l = xrf.unique_frames(points_pass_l, lambda rows: np.mean(rows, axis=0))
            points_pass_r = xrf.unique_frames(points_pass_r, lambda rows: np.mean(rows, axis=0))

            list_passes_l.append(points_pass_l)
            list_passes_r.append(points_pass_r)

        # Combine stance points from all walking passes in the trial, and take frames common to
        # trial points and the truth.
        points_trial_l, truth_trial_l = match_frames(list_passes_l, df_truth_trial.L_FOOT.dropna())
        points_trial_r, truth_trial_r = match_frames(list_passes_r, df_truth_trial.R_FOOT.dropna())

        proposals_foot_trial = df_hypo_trial.apply(lambda row: row.population[row.labels == row.labels.max()], axis=1)
        array_proposals_l = np.array(proposals_foot_trial.reindex(truth_trial_l.frames))
        array_proposals_r = np.array(proposals_foot_trial.reindex(truth_trial_r.frames))

        truth_mod_trial_l = pp.closest_proposals(array_proposals_l, truth_trial_l.values)
        truth_mod_trial_r = pp.closest_proposals(array_proposals_r, truth_trial_r.values)

        list_points_l.append(points_trial_l)
        list_points_r.append(points_trial_r)

        list_truth_l.append(truth_trial_l)
        list_truth_r.append(truth_trial_r)

        list_truth_mod_l.append(truth_mod_trial_l)
        list_truth_mod_r.append(truth_mod_trial_r)

    points_l = np.vstack(list_points_l)
    points_r = np.vstack(list_points_r)

    truth_l = np.vstack(list_truth_l)
    truth_r = np.vstack(list_truth_r)

    truth_mod_l = np.vstack(list_truth_mod_l)
    truth_mod_r = np.vstack(list_truth_mod_r)

    acc_stance_l = pp.position_accuracy(points_l, truth_l)
    acc_stance_r = pp.position_accuracy(points_r, truth_r)

    acc_stance_mod_l = pp.position_accuracy(points_l, truth_mod_l)
    acc_stance_mod_r = pp.position_accuracy(points_r, truth_mod_r)

    data = [
        [acc_stance_l, acc_stance_r],
        [acc_stance_mod_l, acc_stance_mod_r]
    ]

    df_acc = pd.DataFrame(data, columns=['Left', 'Right'], index=['Truth', 'Modified'])

    with open(join('results', 'tables', 'accuracy_stance.txt'), 'w') as file:
        file.write(df_acc.round(3).to_latex())


if __name__ == '__main__':
    main()
