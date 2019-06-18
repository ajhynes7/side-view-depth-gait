"""Calculate relative error."""

from os.path import join

import pandas as pd


def main():

    # Gait parameters from all trials with matching IDs.
    df_matched_k = pd.read_pickle(join('data', 'kinect', 'df_matched.pkl'))
    df_matched_z = pd.read_pickle(join('data', 'zeno', 'df_matched.pkl'))

    # Take only the gait parameters measured by Kinect.
    df_matched_z = df_matched_z[df_matched_k.columns]

    df_sides_k = df_matched_k.groupby(['trial_id', 'side']).mean()
    df_sides_z = df_matched_z.groupby(['trial_id', 'side']).mean()

    dict_error = {}

    for side in ['L', 'R']:

        df_side_k = df_sides_k[df_sides_k.index.get_level_values('side') == side]
        df_side_z = df_sides_z[df_sides_z.index.get_level_values('side') == side]

        dict_error[side] = ((df_side_k - df_side_z).abs() / df_side_z * 100).max()

    df_error_sides = pd.DataFrame.from_dict(dict_error)

    with open(join('results', 'tables', 'rel_error_sides.txt'), 'w') as file:
        file.write(df_error_sides.round(2).to_latex())


if __name__ == '__main__':
    main()
