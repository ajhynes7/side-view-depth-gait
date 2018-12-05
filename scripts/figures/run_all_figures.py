"""Run all scripts to generate figures."""

import matplotlib.pyplot as plt

from scripts.figures import (body_graph, create_figures_code, joint_proposals,
                             spheres, truth_positions)


def main():

    # Customize font
    plt.rc('text', usetex=True)
    font = {'family': 'serif', 'weight': 'bold', 'size': 14}
    plt.rc('font', **font)

    # Run scripts to make figures
    body_graph.main()
    joint_proposals.main()
    spheres.main()
    create_figures_code.main()
    truth_positions.main()


if __name__ == '__main__':
    main()
