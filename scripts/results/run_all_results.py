"""Run all results scripts."""

from scripts.results import (plot_labels, plot_results, process_ground_truth,
                             process_trials, table_bland, table_length_compare,
                             table_lengths, table_pose)


def main():

    process_trials.main()
    process_ground_truth.main()

    plot_results.main()
    plot_labels.main()

    table_bland.main()
    table_lengths.main()
    table_length_compare.main()
    table_pose.main()


if __name__ == '__main__':
    main()
