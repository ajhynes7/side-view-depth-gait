"""Select the best body part positions from multiple joint proposals."""

import time
from os.path import join

import numpy as np
import pandas as pd

import analysis.math_funcs as mf
import modules.pose_estimation as pe
from modules.constants import PART_CONNECTIONS


def cost_func(a, b):
    """Cost function for weighting edges of graph."""
    return (a - b) ** 2


def score_func(a, b):
    """Score function for scoring links between body parts."""
    x = 1 / mf.norm_ratio(a, b)
    return -(x - 1) ** 2 + 1


def main():

    radii = [i for i in range(6)]

    # DataFrame with position hypotheses (join proposals) for all trials
    df_hypo = pd.read_pickle(join('data', 'kinect', 'df_hypo.pkl'))

    n_frames_total = df_hypo.shape[0]

    # DataFrame with expected lengths between body parts
    length_path = join('data', 'kinect', 'kinect_lengths.csv')
    df_length = pd.read_csv(length_path, index_col=0)

    # Pre-allocate array to hold best head and foot positions
    # on each frame
    array_selected = np.full((n_frames_total, 3), fill_value=None)

    t = time.time()
    index_row = 0

    for trial_name, df_trial in df_hypo.groupby(level=0):

        print(trial_name)  # Print current trial just to show progress

        lengths = df_length.loc[trial_name]  # Read estimated lengths for trial

        # Expected lengths for all part connections,
        # including non-adjacent (e.g., knee to foot)
        label_adj_list = pe.lengths_to_adj_list(PART_CONNECTIONS, lengths)

        for tuple_frame in df_trial.itertuples():

            population, labels = tuple_frame.population, tuple_frame.labels

            # Select the best two shortest paths
            pos_1, pos_2 = pe.process_frame(
                population,
                labels,
                label_adj_list,
                radii,
                cost_func,
                score_func,
            )

            # Positions of the best head and two feet
            array_selected[index_row, 0] = pos_1[0, :]
            array_selected[index_row, 1] = pos_1[-1, :]
            array_selected[index_row, 2] = pos_2[-1, :]

            index_row += 1

    # DataFrame of selected head and foot positions.
    # The left and right feet are just assumptions at this point.
    # They are later given correct L/R labels.
    df_selected = pd.DataFrame(
        array_selected,
        index=df_hypo.index,
        columns=['HEAD', 'L_FOOT', 'R_FOOT'],
    )

    df_selected.to_pickle(join('data', 'kinect', 'df_selected.pkl'))

    # %% Calculate run-time metrics

    time_elapsed = time.time() - t

    frames_per_second = round(n_frames_total / time_elapsed)

    print(
        """
        Number of frames: {}\n
        Total time: {}\n
        Frames per second: {}""".format(
            n_frames_total,
            round(time_elapsed, 2),
            frames_per_second,
        )
    )


if __name__ == '__main__':
    main()
