"""Run all results scripts."""

import matplotlib
from matplotlib import rc

from scripts.results import (
    align_frames,
    gait_analysis,
    plot_accuracy_radii,
    process_ground_truth,
    process_trials,
    table_length_compare,
    table_lengths,
    table_pose,
)


def main():

    matplotlib.rcParams.update({"pgf.texsystem": "pdflatex"})

    # Customize font
    rc('font', **{'family': 'serif', 'weight': 'bold', 'size': 14})
    rc('text', usetex=True)

    align_frames.main()
    process_ground_truth.main()
    process_trials.main()

    table_lengths.main()
    table_length_compare.main()
    table_pose.main()

    plot_accuracy_radii.main()

    gait_analysis.main()


if __name__ == '__main__':
    main()
