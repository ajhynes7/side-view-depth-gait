"""Run all results scripts."""

import matplotlib.pyplot as plt

from scripts.results import (align_frames, gait_analysis, plot_accuracy_radii,
                             process_ground_truth, process_trials,
                             table_length_compare, table_lengths, table_pose)


def main():

    # Customize font
    plt.rc('text', usetex=True)
    font = {'family': 'serif', 'weight': 'bold', 'size': 14}
    plt.rc('font', **font)

    align_frames.main()
    #process_ground_truth.main()
    process_trials.main()

    table_lengths.main()
    table_length_compare.main()
    table_pose.main()

    plot_accuracy_radii.main()

    gait_analysis.main()


if __name__ == '__main__':
    main()
