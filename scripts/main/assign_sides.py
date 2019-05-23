"""Assign left and right sides to selected foot positions."""

from os.path import join

import numpy as np
import pandas as pd

import modules.pandas_funcs as pf
import modules.point_processing as pp
import modules.side_assignment as sa


def main():

    kinect_dir = join('data', 'kinect')

    df_selected_passes = pd.read_pickle(join(kinect_dir, 'df_selected_passes.pkl'))

    tuples_trial_pass = df_selected_passes.index.droplevel('frame').values

    dict_assigned_2d = dict.fromkeys(tuples_trial_pass)
    dict_assigned_3d = dict.fromkeys(tuples_trial_pass)

    for tuple_trial_pass, df_pass in df_selected_passes.groupby(level=[0, 1]):

        df_pass = df_pass.loc[tuple_trial_pass]
        frames = df_pass.index.values

        points_head = np.stack(df_pass.HEAD)
        points_3d_a = np.stack(df_pass.L_FOOT)
        points_3d_b = np.stack(df_pass.R_FOOT)

        # Convert the 3D foot points to a 2D coordinate system.
        points_2d_a, points_2d_b = sa.convert_to_2d(points_head, points_3d_a, points_3d_b)

        # Split the walking pass into portions between the feet coming together.
        labels_portions = sa.split_walking_pass(frames, points_2d_a, points_2d_b)

        # Assign left and right sides to the 2D foot positions.
        array_assignment = sa.assign_sides_pass(points_2d_a, points_2d_b, labels_portions)

        points_2d_l, points_2d_r = pp.correspond_points(points_2d_a, points_2d_b, array_assignment)
        points_3d_l, points_3d_r = pp.correspond_points(points_3d_a, points_3d_b, array_assignment)

        df_2d = pf.points_to_dataframe([points_2d_l, points_2d_r], ['L_FOOT', 'R_FOOT'], frames)
        df_3d = pf.points_to_dataframe([points_3d_l, points_3d_r], ['L_FOOT', 'R_FOOT'], frames)

        dict_assigned_2d[tuple_trial_pass] = df_2d
        dict_assigned_3d[tuple_trial_pass] = df_3d

    df_assigned_2d = pd.concat(dict_assigned_2d)
    df_assigned_3d = pd.concat(dict_assigned_3d)

    index_names = ['trial_name', 'num_pass', 'frame']
    df_assigned_2d.index.names = index_names
    df_assigned_3d.index.names = index_names

    # Save the assigned positions for each trial
    df_assigned_2d.to_pickle(join(kinect_dir, 'df_assigned_2d.pkl'))
    df_assigned_3d.to_pickle(join(kinect_dir, 'df_assigned_3d.pkl'))


if __name__ == '__main__':
    main()
