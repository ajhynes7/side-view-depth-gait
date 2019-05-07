"""Transform raw data from the Kinect into a pandas DataFrame."""

from os.path import join

import numpy as np
import pandas as pd

import analysis.images as im
import modules.pose_estimation as pe
from scripts.pre_processing.constants import PART_TYPES


def main():

    load_dir = join('data', 'kinect', 'raw')

    # Number of columns for the position coordinates
    # Number should be sufficiently large and divisible by 3
    n_coord_cols = 99

    # List of trials to run
    running_path = join('data', 'kinect', 'running', 'trials_to_run.csv')
    trials_to_run = pd.read_csv(running_path, header=None, squeeze=True).values

    dict_trials = {}

    for trial_name in trials_to_run:

        print(trial_name)

        file_path = join(load_dir, trial_name + '.txt')
        df_raw = pd.read_csv(
            file_path,
            skiprows=range(22),
            header=None,
            names=[i for i in range(-2, n_coord_cols)],
            sep='\t',
            skipfooter=1,  # The last night says "Quit button pressed"
            engine='python',
        )

        # Label some columns
        df_raw.rename(columns={-2: 'frame', -1: 'part'}, inplace=True)

        # Crop the DataFrame at the max frame number
        # (the text file loops back to the beginning)
        max_frame = df_raw.frame.max()
        last_index = np.nonzero(df_raw.frame.values == max_frame)[0][-1]
        df_cropped = df_raw.iloc[:last_index]

        df_cropped = df_cropped.set_index(['frame', 'part'])

        # Drop the first three numeric columns
        # (these are the coordinates of the confidence position)
        df_hypo_raw = df_cropped.drop([0, 1, 2], axis=1)

        # Drop rows that are all nans
        df_hypo_raw = df_hypo_raw.dropna(how='all')

        # Convert elements floats because they
        # are 3D coordinates
        df_hypo_raw = df_hypo_raw.astype(np.float)

        # Extract unique index values
        frames = df_hypo_raw.index.get_level_values(0).unique()

        df_hypo_types = pd.DataFrame(index=frames, columns=PART_TYPES)

        for part_type in PART_TYPES:

            # Booleans marking rows of one part type (e.g. FOOT)
            row_is_type = df_hypo_raw.index.get_level_values(1).str.contains(part_type)

            # Only contains rows for one part type
            df_hypo_raw_type = df_hypo_raw[row_is_type]

            for frame, df_frame in df_hypo_raw_type.groupby(level=0):

                # Combine the coordinates of body parts with the same type
                # e.g. L_FOOT and R_FOOT
                coords_part_type = df_frame.values.flatten()
                coords_part_type = coords_part_type[~np.isnan(coords_part_type)]

                if coords_part_type.size == 0:
                    continue

                # Reshape into array of 3D points
                points_part_type = coords_part_type.reshape(-1, 3)

                # The hypothetical positions now need to be converted from
                # real to image then back to real using new parameters.
                points_part_type = im.recalibrate_positions(
                    points_part_type,
                    im.X_RES_ORIG,
                    im.Y_RES_ORIG,
                    im.X_RES,
                    im.Y_RES,
                    im.F_XZ,
                    im.F_YZ,
                )

                df_hypo_types.loc[frame, part_type] = points_part_type

        dict_trials[trial_name] = df_hypo_types

    # DataFrame of all frames with position hypotheses for each body part type
    df_concat = pd.concat(dict_trials).dropna()

    # %% Convert the columns of part types into two columns:
    # 'population' and 'labels'.

    part_labels = range(len(PART_TYPES))
    df_pop_labels = df_concat.apply(
        lambda row: pe.get_population(row, part_labels), axis=1
    )

    df_hypo_final = pd.DataFrame(
        index=df_pop_labels.index, columns=['population', 'labels']
    )

    df_hypo_final['population'] = df_pop_labels.apply(lambda row: row[0])
    df_hypo_final['labels'] = df_pop_labels.apply(lambda row: row[1])

    df_hypo_final.to_pickle(join('data', 'kinect', 'df_hypo.pkl'))


if __name__ == '__main__':
    main()
