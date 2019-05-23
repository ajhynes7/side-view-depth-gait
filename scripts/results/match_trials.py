"""Create Kinect and Zeno DataFrames with matching trial IDs."""

from os.path import join

import pandas as pd


def main():

    df_gait_k = pd.read_pickle(join('data', 'kinect', 'df_gait.pkl'))
    df_gait_z = pd.read_pickle(join('data', 'zeno', 'df_gait.pkl'))

    df_match = pd.read_csv(join('data', 'matching', 'match_kinect_zeno.csv'), index_col=0)

    dict_k, dict_z = {}, {}

    for tuple_row in df_match.itertuples():

        trial_id = tuple_row.Index

        trial_name_k = tuple_row.kinect
        trial_name_z = tuple_row.zeno

        dict_k[(trial_id, trial_name_k)] = df_gait_k.loc[trial_name_k]
        dict_z[(trial_id, trial_name_z)] = df_gait_z.loc[trial_name_z]

    # Gait parameters with matching trial IDs
    df_matched_k = pd.concat(dict_k)
    df_matched_z = pd.concat(dict_z)

    df_matched_k.index = df_matched_k.index.rename(['trial_id', 'trial_name'], level=[0, 1])
    df_matched_z.index = df_matched_z.index.rename(['trial_id', 'trial_name'], level=[0, 1])

    df_matched_k.to_pickle(join('data', 'kinect', 'df_matched.pkl'))
    df_matched_z.to_pickle(join('data', 'zeno', 'df_matched.pkl'))


if __name__ == '__main__':
    main()
