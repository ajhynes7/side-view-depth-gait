"""Create LaTeX table of Bland-Altman results."""

import os

import numpy as np
import pandas as pd

import analysis.stats as st


def main():

    dataframe_dir = os.path.join('results', 'dataframes')

    # Load all results
    df_total_k = pd.read_pickle(os.path.join(dataframe_dir, 'df_total_k.pkl'))
    df_total_z = pd.read_pickle(os.path.join(dataframe_dir, 'df_total_z.pkl'))

    gait_params = df_total_k.select_dtypes(float).columns

    df_trials_k = df_total_k.groupby('trial_id').median()
    df_trials_z = df_total_z.groupby('trial_id').median()

    bland_alt_tuples = []

    for param in gait_params:
        differences = st.relative_difference(df_trials_k[param],
                                             df_trials_z[param])
        bland_alt = st.bland_altman(differences)

        bland_alt_tuples.append(bland_alt)

    df_bland = pd.DataFrame.from_records(
        bland_alt_tuples, index=gait_params, columns=bland_alt._fields)

    df_bland.to_csv(
        os.path.join('results', 'spreadsheets', 'bland_altman.csv'))\

    df_bland_round = np.round(df_bland, 4)

    save_dir = os.path.join('results', 'tables')

    with open(os.path.join(save_dir, 'bland_altman.txt'), 'w') as file:
        file.write(df_bland_round.to_latex())


if __name__ == '__main__':
    main()
