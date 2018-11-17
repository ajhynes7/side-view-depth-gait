"""Create LaTeX table of body length results."""

import os

import numpy as np
import pandas as pd


def main():

    load_path = os.path.join('data', 'kinect', 'lengths', 'kinect_lengths.csv')

    df_lengths = pd.read_csv(load_path, index_col=0)

    match_dir = os.path.join('data', 'matching')
    df_match = pd.read_csv(os.path.join(match_dir, 'match_kinect_zeno.csv'))

    # Body lengths of trials that have matching Zeno data
    df_lengths_matched = df_lengths.loc[df_match.kinect].reset_index()

    # Extract date and participant from file name
    pattern = '(?P<date>\d{4}-\d{2}-\d{2})_P(?P<participant>\d{3})'
    df_regex = df_lengths_matched.file_name.str.extract(pattern)
    df_expanded = pd.concat((df_lengths_matched, df_regex), axis=1, sort=False)

    df_grouped = df_expanded.groupby('participant').agg(['mean', 'std'])
    df_grouped_round = np.round(df_grouped, 2)

    save_dir = os.path.join('results', 'tables')

    with open(os.path.join(save_dir, 'lengths.txt'), 'w') as file:
        file.write(df_grouped_round.to_latex())


if __name__ == '__main__':
    main()
