from os.path import join
from typing import Tuple, Optional

import numpy as np
import pandas as pd

import analysis.stats as st
from analysis.icc import icc
from modules.typing import array_like


def calc_trial_results(
    df_matched_k: pd.DataFrame,
    df_matched_z: pd.DataFrame,
    vector_filter: Optional[array_like] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate ICC and Bland-Altman results."""

    df_trials_k = df_matched_k.groupby('trial_id').median()
    df_trials_z = df_matched_z.groupby('trial_id').median()

    if vector_filter is None:
        vector_filter = np.full(df_trials_k.shape[0], True)

    assert vector_filter.ndim == 1
    assert df_trials_k.shape[0] == df_trials_z.shape[0] == vector_filter.size

    dict_icc, dict_bland, dict_bland_relative = {}, {}, {}

    gait_params = df_matched_k.columns

    for param in gait_params:

        measures_k = df_trials_k[param][vector_filter]
        measures_z = df_trials_z[param][vector_filter]

        differences = measures_k - measures_z
        differences_relative = st.relative_difference(measures_k, measures_z)

        bland_altman = st.bland_altman(differences)
        bland_altman_relative = st.bland_altman(differences_relative)

        dict_bland[param] = bland_altman._asdict()
        dict_bland_relative[param] = bland_altman_relative._asdict()

        measures = np.column_stack((measures_k, measures_z))
        icc_21 = icc(measures, form=2)
        icc_31 = icc(measures, form=3)

        dict_icc[param] = {'ICC_21': icc_21, 'ICC_31': icc_31}

    df_icc = pd.DataFrame.from_dict(dict_icc, orient='index')
    df_bland = pd.DataFrame.from_dict(dict_bland, orient='index')
    df_bland_relative = pd.DataFrame.from_dict(dict_bland_relative, orient='index')

    return df_bland, df_bland_relative, df_icc


def concat_tables(df_bland, df_bland_relative, df_icc):

    return pd.concat(
        [df_bland, df_bland_relative * 100, df_icc],
        axis=1,
        keys=["Bland Altman", "Bland Altman Relative [%]", "ICC"],
    ).round(2)


def main():

    # Gait parameters from all trials with matching IDs.
    df_matched_k = pd.read_pickle(join('data', 'kinect', 'df_matched.pkl'))
    df_matched_z = pd.read_pickle(join('data', 'zeno', 'df_matched.pkl'))

    df_trial_types = pd.read_csv(join('data', 'matching', 'trial_types.csv'))
    df_matched_types = df_trial_types.set_index("trial_name").loc[
        df_matched_k.reset_index().trial_name.unique()
    ]
    trials_kinect_normal_walking = df_matched_types.loc[
        df_matched_types.type == "A"
    ].index.values

    trials_kinect = df_matched_k.reset_index().trial_name.unique()
    is_normal_walking = np.in1d(trials_kinect, trials_kinect_normal_walking)

    # Results from combining normal and dual-task walking
    df_bland, df_bland_relative, df_icc = calc_trial_results(df_matched_k, df_matched_z)

    df_bland_normal, df_bland_rel_normal, df_icc_normal = calc_trial_results(
        df_matched_k, df_matched_z, vector_filter=is_normal_walking
    )
    df_bland_dual_task, df_bland_rel_dual_task, df_icc_dual_task = calc_trial_results(
        df_matched_k, df_matched_z, vector_filter=~is_normal_walking
    )

    df_concat = concat_tables(df_bland, df_bland_relative, df_icc)
    df_concat_normal = concat_tables(
        df_bland_normal, df_bland_rel_normal, df_icc_normal
    )
    df_concat_dual_task = concat_tables(
        df_bland_dual_task, df_bland_rel_dual_task, df_icc_dual_task
    )

    df_total = pd.concat(
        [df_concat, df_concat_normal, df_concat_dual_task],
        keys=["Grouped", "Normal pace", "Dual task"],
    )

    with open(join('results', 'tables', 'bland_icc.csv'), 'w') as file:
        file.write(df_total.to_csv())


if __name__ == '__main__':
    main()
