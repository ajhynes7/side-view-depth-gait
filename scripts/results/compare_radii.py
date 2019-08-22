"""Calculate accuracies for various sphere radii."""

from os.path import join

import numpy as np
import pandas as pd

import modules.pose_estimation as pe


def main():

    kinect_dir = join('data', 'kinect')

    df_hypo = pd.read_pickle(join(kinect_dir, 'df_hypo.pkl'))
    df_truth = pd.read_pickle(join(kinect_dir, 'df_truth.pkl'))

    df_length = pd.read_csv(join(kinect_dir, 'kinect_lengths.csv'), index_col=0)

    labelled_trial_names = df_truth.index.get_level_values(0).unique()
    df_hypo_labelled = df_hypo.loc[labelled_trial_names]

    list_dfs_radii = []
    radii_max = range(11)

    for r_max in radii_max:

        radii = [i for i in range(r_max + 1)]

        # Pre-allocate array to hold best head and foot positions
        # on each frame
        array_selected = np.full((df_hypo_labelled.shape[0], 3), fill_value=None)

        index_row = 0

        for trial_name, df_hypo_trial in df_hypo_labelled.groupby(level=0):

            lengths = df_length.loc[trial_name]  # Read estimated lengths for trial

            for tuple_frame in df_hypo_trial.itertuples():

                population, labels = tuple_frame.population, tuple_frame.labels

                # Select the best two shortest paths
                pos_1, pos_2 = pe.process_frame(population, labels, lengths, radii, pe.cost_func, pe.score_func)

                # Positions of the best head and two feet
                array_selected[index_row, 0] = pos_1[0, :]
                array_selected[index_row, 1] = pos_1[-1, :]
                array_selected[index_row, 2] = pos_2[-1, :]

                index_row += 1

        # DataFrame of selected head and foot positions.
        # The left and right feet are just assumptions at this point.
        # They are later given correct L/R labels.
        df_selected = pd.DataFrame(array_selected, index=df_hypo_labelled.index, columns=['HEAD', 'L_FOOT', 'R_FOOT'])

        list_dfs_radii.append(df_selected)

    df_radii = pd.concat(list_dfs_radii, keys=radii_max, names=['max_radius'])

    df_radii.to_pickle(join(kinect_dir, 'df_radii.pkl'))


if __name__ == '__main__':
    main()
