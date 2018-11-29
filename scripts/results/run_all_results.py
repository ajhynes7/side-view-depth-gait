"""Run all results scripts."""

import matplotlib.pyplot as plt

from scripts.results import (plot_labels, plot_results, process_ground_truth,
                             process_trials, table_bland, table_length_compare,
                             table_lengths, table_pose)


def main():

    # Customize font
    plt.rc('text', usetex=True)
    font = {'family': 'serif', 'weight': 'bold', 'size': 14}
    plt.rc('font', **font)

    process_trials.main()
    process_ground_truth.main()

    plot_results.main()

    table_bland.main()
    table_lengths.main()
    table_length_compare.main()
    table_pose.main()


if __name__ == '__main__':
    main()
