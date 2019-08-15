"""Calculate gait parameters from Kinect data."""

from os.path import join

import numpy as np
import pandas as pd
import xarray as xr

import modules.gait_parameters as gp


def main():

    df_selected_passes = pd.read_pickle(join('data', 'kinect', 'df_selected_passes.pkl'))

    tuples_trial_pass = df_selected_passes.index.droplevel('frame').values
    dict_gait = dict.fromkeys(tuples_trial_pass)

    for tuple_trial_pass, df_pass in df_selected_passes.groupby(level=[0, 1]):

        print(tuple_trial_pass)

        frames = df_pass.reset_index().frame.values

        points_head = np.stack(df_pass.HEAD)
        points_a = np.stack(df_pass.L_FOOT)
        points_b = np.stack(df_pass.R_FOOT)

        points_stacked = xr.DataArray(
            np.dstack((points_a, points_b, points_head)),
            coords={
                'frames': frames,
                'cols': range(3),
                'layers': ['points_a', 'points_b', 'points_head'],
            }
        )

        df_gait_pass = gp.walking_pass_parameters(points_stacked)

        if not df_gait_pass.empty:
            dict_gait[tuple_trial_pass] = df_gait_pass

    df_gait = pd.concat(dict_gait, sort=False)
    df_gait.index = df_gait.index.rename(['trial_name', 'num_pass'], level=[0, 1])

    # Save the gait parameters for each trial
    df_gait.to_pickle(join('data', 'kinect', 'df_gait.pkl'))


if __name__ == '__main__':
    main()
