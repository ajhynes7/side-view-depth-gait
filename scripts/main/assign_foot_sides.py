"""Assign left and right sides to selected foot positions."""

from os.path import join

import numpy as np
import pandas as pd

import modules.assign_sides as asi
import modules.pandas_funcs as pf


def main():

    df_selected_passes = pd.read_pickle(join('data', 'kinect', 'df_selected_passes.pkl'))

    tuples_trial_pass = df_selected_passes.index.droplevel('frame').values
    dict_assigned = dict.fromkeys(tuples_trial_pass)

    for tuple_trial_pass, df_pass in df_selected_passes.groupby(level=[0, 1]):

        df_pass = df_pass.loc[tuple_trial_pass]
        frames = df_pass.index.values

        points_head = np.stack(df_pass.HEAD)
        points_a = np.stack(df_pass.L_FOOT)
        points_b = np.stack(df_pass.R_FOOT)

        # Reduce the dimension of the foot points.
        points_2d_a, points_2d_b = asi.convert_to_2d(points_head, points_a, points_b)

        # Assign left/right sides to the feet.
        points_l, points_r = asi.assign_sides_pass(frames, points_2d_a, points_2d_b)

        dict_assigned[tuple_trial_pass] = pf.points_to_dataframe([points_l, points_r], ['L_FOOT', 'R_FOOT'], frames)

    df_assigned = pd.concat(dict_assigned)
    df_assigned.index.names = ['trial_name', 'num_pass', 'frame']

    # Save the assigned positions for each trial
    df_assigned.to_pickle(join('data', 'kinect', 'df_assigned.pkl'))


if __name__ == '__main__':
    main()
