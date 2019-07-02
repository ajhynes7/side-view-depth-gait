"""Plot stride width relative difference vs. Zeno stride width."""

from os.path import join

import matplotlib.pyplot as plt
import pandas as pd


def main():

    df_matched_k = pd.read_pickle(join('data', 'kinect', 'df_matched.pkl'))
    df_matched_z = pd.read_pickle(join('data', 'zeno', 'df_matched.pkl'))

    df_trials_k = df_matched_k.groupby(['trial_id']).median()
    df_trials_z = df_matched_z.groupby(['trial_id']).median()

    df_trial_names_k = df_matched_k.groupby(['trial_name']).median()

    pattern_participant = r'P(?P<participant>\d{3})'
    df_regex = df_trial_names_k.index.get_level_values('trial_name').str.extract(pattern_participant)

    df_diff_rel = (
        ((df_trials_k - df_trials_z) / (0.5 * (df_trials_k + df_trials_z)))
        .dropna(axis=1)  # Drop nan columns (gait params not shared by Kinect and Zeno)
        .assign(participant=df_regex.participant)
    )

    df_sw = pd.DataFrame({
        'rel_diff': df_diff_rel.stride_width.values,
        'zeno': df_trials_z.stride_width.values,
        'participant': df_regex.participant,
    })

    df_sw.participant = (
        pd.Categorical(df_sw.participant, ordered=True)
        .rename_categories({
            '004': 'Participant 1', '005': 'Participant 2', '006': 'Participant 3', '007': 'Participant 4'
        })
    )

    participants_unique = df_sw.participant.unique().sort_values()

    fig = plt.figure()

    for participant in participants_unique:

        is_participant = df_sw.participant == participant

        x = df_sw.zeno[is_participant]
        y = df_sw.rel_diff[is_participant] * 100

        plt.scatter(x, y, label=participant, s=100, edgecolor='k')

    plt.legend()

    plt.xlabel('Zeno Stride Width [cm]')
    plt.ylabel(r'Relative Difference [\%]')

    fig.savefig(join('results', 'plots', 'scatter_stride_width.pdf'))


if __name__ == '__main__':
    main()
