"""Run all results scripts."""

import matplotlib
from matplotlib import rc

from scripts.results import (
    align_frames,
    compare_gait,
    match_trials,
    plot_accuracy_radii,
    process_ground_truth,
    table_length_compare,
    table_lengths,
    table_pose,
)


def main():

    matplotlib.rcParams.update({"pgf.texsystem": "pdflatex"})

    # Customize font
    rc('font', **{'family': 'serif', 'weight': 'bold', 'size': 14})
    rc('text', usetex=True)

    # %% Comparison with labelled depth images

    align_frames.main()
    process_ground_truth.main()

    table_lengths.main()
    table_length_compare.main()
    table_pose.main()

    plot_accuracy_radii.main()

    # %% Comparison with Zeno Walkway

    match_trials.main()
    compare_gait.main()


if __name__ == '__main__':
    main()
