"""Calculate gait parameters from Kinect data."""

from os.path import join

import numpy as np
import pandas as pd

import modules.gait_parameters as gp


def main():

    df_assigned_2d = pd.read_pickle(join('data', 'kinect', 'df_assigned_2d.pkl'))
    df_assigned_3d = pd.read_pickle(join('data', 'kinect', 'df_assigned_3d.pkl'))

    tuples_trial_pass = df_assigned_2d.index.droplevel('frame').values
    dict_gait = dict.fromkeys(tuples_trial_pass)

    for tuple_trial_pass, df_pass_2d in df_assigned_2d.groupby(level=[0, 1]):

        frames = df_pass_2d.reset_index().frame.values

        df_pass_3d = df_assigned_3d.loc[tuple_trial_pass]

        points_2d_l = np.stack(df_pass_2d.L_FOOT)
        points_2d_r = np.stack(df_pass_2d.R_FOOT)

        points_3d_l = np.stack(df_pass_3d.L_FOOT)
        points_3d_r = np.stack(df_pass_3d.R_FOOT)

        # The 1D foot signal is the Y coordinate of the 2D points.
        signal_l = points_2d_l[:, 1]
        signal_r = points_2d_r[:, 1]

        df_gait_pass = gp.walking_pass_parameters(frames, points_3d_l, points_3d_r, signal_l, signal_r)

        dict_gait[tuple_trial_pass] = df_gait_pass

    df_gait = pd.concat(dict_gait)
    df_gait.index = df_gait.index.rename(level=[0, 1], names=['trial_name', 'num_pass'])

    # Save the gait parameters for each trial
    df_gait.to_pickle(join('data', 'kinect', 'df_gait.pkl'))


if __name__ == '__main__':
    main()
