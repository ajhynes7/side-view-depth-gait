"""Run all main scripts."""

from scripts.main import (
    assign_foot_sides,
    calc_gait_params,
    estimate_lengths,
    label_passes,
    select_proposals,
)


def main():

    estimate_lengths.main()
    select_proposals.main()
    label_passes.main()
    assign_foot_sides.main()
    calc_gait_params.main()


if __name__ == '__main__':
    main()
