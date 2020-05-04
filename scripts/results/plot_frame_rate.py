"""Compare accuracy of stride velocity to the frame rate used."""

from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.icc import icc


def main():

    # Gait parameters from all trials with matching IDs.
    df_matched_k = pd.read_pickle(join('data', 'kinect', 'df_matched.pkl'))
    df_matched_z = pd.read_pickle(join('data', 'zeno', 'df_matched.pkl'))

    df_trials_k = df_matched_k.groupby('trial_id').median()
    df_trials_z = df_matched_z.groupby('trial_id').median()

    param = 'stride_velocity'

    list_fps = range(20, 35)
    list_icc_21 = []

    for fps in list_fps:

        measures_k = df_trials_k[param] * fps / 30
        measures_z = df_trials_z[param]

        measures = np.column_stack((measures_k, measures_z))

        icc_21 = icc(measures, form=2)

        list_icc_21.append(icc_21)

    fig, ax = plt.subplots()

    ax.scatter(list_fps, list_icc_21, c='k')

    ax.set_aspect(0.8)

    ax.set_xlabel("Camera frame rate [fps]")
    ax.set_ylabel("$ICC_{2,1}$ of Stride Velocity")

    fig.savefig(join('results', 'plots', 'scatter_frame_rate.pdf'))


if __name__ == '__main__':
    main()
