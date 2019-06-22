"""Run all results."""


from scripts.results import (
    align_frames,
    compare_gait,
    compare_lengths,
    compare_positions,
    compare_radii,
    group_lengths,
    make_plots,
    match_trials,
    process_ground_truth,
)


def main():

    # %% Estimated lengths grouped by participant

    group_lengths.main()

    # %% Comparison with labelled images

    align_frames.main()
    process_ground_truth.main()

    compare_lengths.main()
    compare_positions.main()

    # Foot selection accuracy with different radii
    compare_radii.main()

    # %%  Comparison with Zeno Walkway

    match_trials.main()
    compare_gait.main()

    make_plots.main()


if __name__ == '__main__':
    main()
