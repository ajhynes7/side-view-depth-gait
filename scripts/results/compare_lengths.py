"""Create table of estimated and ground truth lengths."""

import os

import numpy as np
import pandas as pd

import analysis.stats as st
import modules.point_processing as pp


def main():

    # Ground truth positions from labelled trials
    df_truth = pd.read_pickle(
        os.path.join('results', 'dataframes', 'df_truth.pkl'))
    part_names = df_truth.columns.values
    trial_names = df_truth.index.levels[0].values

    # DataFrame with lengths between body parts
    df_lengths = pd.read_csv(
        os.path.join('data', 'kinect', 'lengths', 'kinect_lengths.csv'),
        index_col=0)
    df_lengths.columns = range(len(df_lengths.columns))

    parts_l = ['HEAD'] + [x for x in part_names if x[0] == 'L']
    parts_r = ['HEAD'] + [x for x in part_names if x[0] == 'R']

    dict_lengths = {}

    for i, trial_name in enumerate(trial_names):

        df_truth_l = df_truth.loc[:, parts_l]
        df_truth_r = df_truth.loc[:, parts_r]

        lengths_l = df_truth_l.apply(
            lambda row: pp.consecutive_dist(np.stack(row)), axis=1)
        lengths_r = df_truth_r.apply(
            lambda row: pp.consecutive_dist(np.stack(row)), axis=1)

        lengths_truth = np.median(
            np.stack(pd.concat((lengths_l, lengths_r))), axis=0)

        df_length_compare = pd.DataFrame({
            'Estimated':
            df_lengths.loc[trial_name],
            'Ground Truth':
            pd.Series(lengths_truth),
        })

        df_length_compare['Relative Error'] = df_length_compare.apply(
            lambda row: st.relative_error(row['Estimated'],
                                          row['Ground Truth']),
            axis=1)

        dict_lengths[i + 1] = df_length_compare

    df_final = pd.concat(dict_lengths)
    df_final.index.names = ['Trial', 'Link']

    df_final = df_final.swaplevel().sort_index(level=0)

    save_path = os.path.join('results', 'tables', 'length_comparison.txt')
    with open(save_path, 'w') as file:

        file.write(np.round(df_final, 2).to_latex())
