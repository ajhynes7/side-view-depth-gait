"""Run all results scripts."""

import scripts.results.bland_table as bland_table
import scripts.results.combine_trials as combine_trials
import scripts.results.length_table as length_table
import scripts.results.plot_results as plot_results
import scripts.results.process_ground_truth as process_ground_truth


def main():

    # Save dataframes
    combine_trials.main()
    process_ground_truth.main()

    plot_results.main()

    bland_table.main()
    length_table.main()


if __name__ == '__main__':
    main()
