"""Count strides measured by the Kinect."""

from os.path import join

import pandas as pd


def main():

    df_gait = pd.read_pickle(join('data', 'kinect', 'df_gait.pkl'))

    list_stats = ['count', 'median']

    stats_per_trial = df_gait.groupby('trial_name').size().agg(list_stats)
    stats_per_pass = df_gait.groupby(['trial_name', 'num_pass']).size().agg(list_stats)

    df_stats = pd.concat([stats_per_trial, stats_per_pass], axis=1).set_axis(
        ["Per trial", "Per pass"], axis=1
    )

    with open(join('results', 'tables', 'stride_counts.csv'), 'w') as file:
        file.write(df_stats.to_csv())


if __name__ == '__main__':
    main()
