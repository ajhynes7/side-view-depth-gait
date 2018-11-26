"""Calculate gait parameters from Kinect data."""

from os.path import join

import pandas as pd

import modules.gait_parameters as gp


def main():

    kinect_dir = join('data', 'kinect')

    df_match = pd.read_csv(join('data', 'matching', 'match_kinect_zeno.csv'))

    for trial_name in df_match.kinect:

        df_assigned = pd.read_pickle(
            join(kinect_dir, 'assigned', trial_name + '.pkl'))
        direction_series = pd.read_pickle(
            join(kinect_dir, 'direction', trial_name + '.pkl'))

        df_trial = gp.combine_walking_passes(df_assigned, direction_series)

        save_path = join(kinect_dir, 'gait_params', trial_name + '.pkl')
        df_trial.to_pickle(save_path)


if __name__ == '__main__':
    main()
