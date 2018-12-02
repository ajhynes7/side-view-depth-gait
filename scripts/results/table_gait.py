"""Create tables of results for gait analysis."""

from os.path import join

import numpy as np
import pandas as pd

import analysis.stats as st


def main():

    dataframe_dir = join('results', 'dataframes')

    # Load all results
    df_total_k = pd.read_pickle(join(dataframe_dir, 'df_total_k.pkl'))
    df_total_z = pd.read_pickle(join(dataframe_dir, 'df_total_z.pkl'))

    gait_params = df_total_k.select_dtypes(float).columns

    df_trials_k = df_total_k.groupby('trial_id').median()
    df_trials_z = df_total_z.groupby('trial_id').median()

    icc_21, icc_31 = [], []
    bland_alt_tuples = []

    for param in gait_params:

        measures_1 = df_trials_k[param]
        measures_2 = df_trials_z[param]

        differences = st.relative_difference(measures_1, measures_2)
        bland_alt = st.bland_altman(differences)
        bland_alt_tuples.append(bland_alt)

        measures = np.column_stack((measures_1, measures_2))
        icc_21.append(st.icc(measures, type=(2, 1)))
        icc_31.append(st.icc(measures, type=(3, 1)))

    df_bland = pd.DataFrame.from_records(
        bland_alt_tuples, index=gait_params, columns=bland_alt._fields)

    df_icc = pd.DataFrame({
        'ICC(2, 1)': icc_21,
        'ICC(3, 1)': icc_31},
        index=gait_params,
    )

    with open(join('results', 'tables', 'bland_altman.txt'), 'w') as file:
        file.write(np.round(df_bland, 3).to_latex())

    with open(join('results', 'tables', 'icc.txt'), 'w') as file:
        file.write(np.round(df_icc, 3).to_latex())


if __name__ == '__main__':
    main()
