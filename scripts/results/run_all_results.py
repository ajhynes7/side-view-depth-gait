"""Run all results."""

import matplotlib
from matplotlib import rc

from scripts.results import (
    align_frames,
    compare_gait,
    compare_lengths,
    compare_positions,
    compare_radii,
    group_lengths,
    match_trials,
    plot_accuracy_radii,
    plot_bland,
    process_ground_truth,
)


def main():

    matplotlib.rcParams.update({"pgf.texsystem": "pdflatex"})

    # Customize font
    rc('font', **{'family': 'serif', 'weight': 'bold', 'size': 14})
    rc('text', usetex=True)

    # %% Estimated lengths grouped by participant

    group_lengths.main()

    # %% Comparison with labelled images

    align_frames.main()
    process_ground_truth.main()

    compare_lengths.main()
    compare_positions.main()

    # Foot selection accuracy with different radii
    compare_radii.main()
    plot_accuracy_radii.main()

    # %%  Comparison with Zeno Walkway

    match_trials.main()
    compare_gait.main()
    plot_bland.main()


if __name__ == '__main__':
    main()
