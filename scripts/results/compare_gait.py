"""Compare the gait parameters of the Kinect and Zeno Walkway."""

from os.path import join

import numpy as np
import pandas as pd

import analysis.stats as st


def main():

    # Gait parameters from all trials with matching IDs.
    df_matched_k = pd.read_pickle(join('data', 'kinect', 'df_matched.pkl'))
    df_matched_z = pd.read_pickle(join('data', 'zeno', 'df_matched.pkl'))

    # Ensure Kinect and Zeno DataFrames have the same MultiIndex.
    assert df_matched_k.index.names == df_matched_z.index.names

    # Ensure that all Kinect parameters are non-negative.
    assert np.all(df_matched_k >= 0)

    # Take absolute value of Zeno parameters.
    df_matched_z = df_matched_z.applymap(lambda x: abs(x))

    gait_params = df_matched_k.columns

    # %% Calculate results per trial.

    df_trials_k = df_matched_k.groupby('trial_id').median()
    df_trials_z = df_matched_z.groupby('trial_id').median()

    dict_icc, dict_bland = {}, {}

    for param in gait_params:

        measures_k = df_trials_k[param]
        measures_z = df_trials_z[param]

        measures = np.column_stack((measures_k, measures_z))

        icc_21 = st.icc(measures, form=(2, 1))
        icc_31 = st.icc(measures, form=(3, 1))

        differences = st.relative_difference(measures_k, measures_z)
        bland_alt = st.bland_altman(differences)

        dict_icc[param] = {'ICC_21': icc_21, 'ICC_31': icc_31}
        dict_bland[param] = bland_alt._asdict()

    df_icc = pd.DataFrame.from_dict(dict_icc, orient='index')
    df_bland = pd.DataFrame.from_dict(dict_bland, orient='index')

    # %% Calculate results for left and right sides.

    df_sides_k = df_matched_k.groupby(['trial_id', 'side']).median()
    df_sides_z = df_matched_z.groupby(['trial_id', 'side']).median()

    dict_icc_sides, dict_bland_sides = {}, {}

    for side in ['L', 'R']:

        df_side_k = df_sides_k[df_sides_k.index.get_level_values('side') == side]
        df_side_z = df_sides_z[df_sides_z.index.get_level_values('side') == side]

        for param in gait_params:

            measures_k = df_side_k[param]
            measures_z = df_side_z[param]

            measures = np.column_stack((measures_k, measures_z))

            icc_21 = st.icc(measures, form=(2, 1))
            icc_31 = st.icc(measures, form=(3, 1))

            differences = st.relative_difference(measures_k, measures_z)
            bland_alt = st.bland_altman(differences)

            dict_icc_sides[(param, side)] = {'ICC_21': icc_21, 'ICC_31': icc_31}
            dict_bland_sides[(param, side)] = bland_alt._asdict()

    df_icc_sides = pd.DataFrame.from_dict(dict_icc_sides, orient='index').unstack()
    df_bland_sides = pd.DataFrame.from_dict(dict_bland_sides, orient='index').unstack()

    # %% Save results as LaTeX tables.

    dict_dfs = {
        'icc': df_icc,
        'bland_altman': df_bland,
        'icc_sides': df_icc_sides,
        'bland_altman_sides': df_bland_sides,
    }

    for file_name, df in dict_dfs.items():

        with open(join('results', 'tables', file_name + '.txt'), 'w') as file:
            file.write(df.round(2).to_latex())


if __name__ == '__main__':
    main()
