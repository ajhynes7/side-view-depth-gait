"""Calculate gait parameters from Kinect data."""

import pickle
from os.path import join

import pandas as pd

import modules.gait_parameters as gp


def main():

    # Table of matching Kinect and Zeno trials
    df_match = pd.read_csv(join('data', 'matching', 'match_kinect_zeno.csv'))

    # DataFrame of head and assigned L/R foot positions
    df_assigned = pd.read_pickle(join('data', 'kinect', 'df_assigned.pkl'))

    # Lines of best fit for each walking pass of each trial
    with open(join('data', 'kinect', 'dict_lines_fit.pkl'), 'rb') as handle:
        dict_lines_fit = pickle.load(handle)

    dict_gait = {}

    for trial_name in df_match.kinect:

        print(trial_name)  # Print current trial just to show progress

        df_assigned_trial = df_assigned.loc[trial_name]

        # Best fit lines for the trial (one for each walking pass)
        lines_trial = dict_lines_fit[trial_name]

        dict_gait[trial_name] = gp.combine_walking_passes(df_assigned_trial, lines_trial)

    df_gait = pd.concat(dict_gait)

    # Save the gait parameters for each trial
    df_gait.to_pickle(join('data', 'kinect', 'df_gait.pkl'))


if __name__ == '__main__':
    main()
