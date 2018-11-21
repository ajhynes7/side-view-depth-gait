"""Run all scripts to generate figures."""

import matplotlib.pyplot as plt

import scripts.figures.body_graph as body_graph
import scripts.figures.create_figures_code as create_figures_code
import scripts.figures.spheres as spheres


def main():

    # Customize font
    plt.rc('text', usetex=True)
    font = {'family': 'serif', 'weight': 'bold', 'size': 14}
    plt.rc('font', **font)

    # Run scripts to make figures
    body_graph.main()
    spheres.main()
    create_figures_code.main()


if __name__ == '__main__':
    main()
