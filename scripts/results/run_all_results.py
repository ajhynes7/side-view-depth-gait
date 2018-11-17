"""Run all result scripts."""

import scripts.results.bland_table as bland_table
import scripts.results.combine_trials as combine_trials
import scripts.results.length_table as length_table
import scripts.results.plot_results as plot_results


def main():

    combine_trials.main()

    # Plots
    plot_results.main()

    # Tables
    bland_table.main()
    length_table.main()


if __name__ == 'main':
    main()
