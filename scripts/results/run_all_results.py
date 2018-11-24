"""Run all results scripts."""

import scripts.results.plot_labels as plot_labels
import scripts.results.plot_results as plot_results
import scripts.results.process_ground_truth as process_ground_truth
import scripts.results.process_trials as process_trials
import scripts.results.table_bland as table_bland
import scripts.results.table_length_compare as table_length_compare
import scripts.results.table_lengths as table_lengths
import scripts.results.table_pose as table_pose


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
