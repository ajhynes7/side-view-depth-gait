from os.path import join
from typing import Tuple, Optional

import numpy as np
import pandas as pd

import analysis.stats as st
from analysis.icc import icc
from modules.typing import array_like


def calc_trial_results(
    df_matched_k: pd.DataFrame, df_matched_z: pd.DataFrame, vector_filter: Optional[array_like] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate ICC and Bland-Altman results."""

    df_trials_k = df_matched_k.groupby('trial_id').median()
    df_trials_z = df_matched_z.groupby('trial_id').median()

    if vector_filter is None:
        vector_filter = np.ones(df_trials_k.shape[0])

    assert vector_filter.ndim == 1
    assert df_trials_k.shape[0] == df_trials_z.shape[0] == vector_filter.size

    dict_icc, dict_bland = {}, {}

    gait_params = df_matched_k.columns

    for param in gait_params:

        measures_k = df_trials_k[param][vector_filter]
        measures_z = df_trials_z[param][vector_filter]

        measures = np.column_stack((measures_k, measures_z))

        icc_21 = icc(measures, form=2)
        icc_31 = icc(measures, form=3)

        differences = st.relative_difference(measures_k, measures_z)
        bland_alt = st.bland_altman(differences)

        dict_icc[param] = {'ICC_21': icc_21, 'ICC_31': icc_31}
        dict_bland[param] = bland_alt._asdict()

    df_icc = pd.DataFrame.from_dict(dict_icc, orient='index')
    df_bland = pd.DataFrame.from_dict(dict_bland, orient='index')

    return df_icc, df_bland


def main():

    # Gait parameters from all trials with matching IDs.
    df_matched_k = pd.read_pickle(join('data', 'kinect', 'df_matched.pkl'))
    df_matched_z = pd.read_pickle(join('data', 'zeno', 'df_matched.pkl'))

    df_trial_types = pd.read_csv(join('data', 'matching', 'trial_types.csv'))
    df_matched_types = df_trial_types.set_index("trial_name").loc[df_matched_k.reset_index().trial_name.unique()]
    trials_kinect_normal_walking = df_matched_types.loc[df_matched_types.type == "A"].index.values

    trials_kinect = df_matched_k.reset_index().trial_name.unique()
    is_normal_walking = np.in1d(trials_kinect, trials_kinect_normal_walking)

    # Results from combining normal and dual-task walking
    df_icc, df_bland = calc_trial_results(df_matched_k, df_matched_z)

    df_icc_normal, df_bland_normal = calc_trial_results(df_matched_k, df_matched_z, is_normal_walking)
    df_icc_dual_task, df_bland_dual_task = calc_trial_results(df_matched_k, df_matched_z, ~is_normal_walking)

    df_icc_split = df_icc_normal.join(df_icc_dual_task, lsuffix='_normal', rsuffix='_dual_task')
    df_bland_split = df_bland_normal.join(df_bland_dual_task, lsuffix='_normal', rsuffix='_dual_task')

    dict_dfs = {
        'icc': df_icc,
        'bland_altman': df_bland,
        'icc_split': df_icc_split,
        'bland_altman_split': df_bland_split,
    }

    for file_name, df in dict_dfs.items():

        with open(join('results', 'tables', file_name + '.txt'), 'w') as file:
            file.write(df.round(2).to_latex())


if __name__ == '__main__':
    main()
