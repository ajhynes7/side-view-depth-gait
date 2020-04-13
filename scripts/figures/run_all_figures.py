"""Run all scripts to generate figures."""

import os

import matplotlib
from matplotlib import rc

from scripts.figures import body_graph, joint_proposals, signal, truth_positions


def main():

    dir_figures = 'figures'

    if not os.path.exists(dir_figures):
        os.makedirs(dir_figures)

    matplotlib.rcParams.update({"pgf.texsystem": "pdflatex"})

    # Customize font
    rc('font', **{'family': 'serif', 'weight': 'bold', 'size': 14})
    rc('text', usetex=True)

    # Run scripts to make figures
    body_graph.main()
    joint_proposals.main()
    signal.main()
    truth_positions.main()


if __name__ == '__main__':
    main()
