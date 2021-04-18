"""Compare estimated and ground truth lengths."""

from os.path import join

import numpy as np
import pandas as pd

import analysis.stats as st
import modules.point_processing as pp


def main():

    # Ground truth positions from labelled trials
    df_truth = pd.read_pickle(join('data', 'kinect', 'df_truth.pkl'))

    # Estimated lengths between body parts for each walking trial
    df_lengths = pd.read_csv(join('data', 'kinect', 'kinect_lengths.csv'), index_col=0)

    # Convert column names from strings to ints so indices will align
    # with the ground truth lengths
    df_lengths.columns = df_lengths.columns.astype(int)

    part_names = df_truth.columns.values
    parts_l = ['HEAD'] + [x for x in part_names if x[0] == 'L']
    parts_r = ['HEAD'] + [x for x in part_names if x[0] == 'R']

    df_truth_l = df_truth.loc[:, parts_l].dropna()
    df_truth_r = df_truth.loc[:, parts_r].dropna()

    # Measured lengths of labelled trials on each frame
    lengths_truth_l = df_truth_l.apply(
        lambda row: pp.consecutive_dist(np.stack(row)), axis=1
    )
    lengths_truth_r = df_truth_r.apply(
        lambda row: pp.consecutive_dist(np.stack(row)), axis=1
    )

    labelled_trial_names = df_truth.index.get_level_values(0).unique()

    dict_lengths = {}

    for i, trial_name in enumerate(labelled_trial_names):

        lengths_trial_l = lengths_truth_l.loc[trial_name]
        lengths_trial_r = lengths_truth_r.loc[trial_name]

        # All measured lengths for the trial
        lengths_trial = pd.concat((lengths_trial_l, lengths_trial_r))

        # Median lengths for the trial
        lengths_truth = np.median(np.stack(lengths_trial), axis=0)

        df_compare_trial = pd.DataFrame(
            {
                'Estimated': df_lengths.loc[trial_name],
                'Ground Truth': pd.Series(lengths_truth),
            }
        )

        dict_lengths[i + 1] = df_compare_trial

    df_length_comparison = pd.concat(dict_lengths)
    df_length_comparison.index.names = ['Trial', 'Link']

    # Switch the multi-index levels so that Head-Hip, etc. is the outer level
    df_length_comparison = df_length_comparison.swaplevel().sort_index(level=0)

    # Add a column for relative error between estimated and truth lengths
    df_length_comparison['Relative Error'] = df_length_comparison.apply(
        lambda row: st.relative_error(row['Estimated'], row['Ground Truth']), axis=1
    )

    save_path = join('results', 'tables', 'length_comparison.csv')
    with open(save_path, 'w') as file:

        file.write(df_length_comparison.round(2).to_csv())


if __name__ == '__main__':
    main()
