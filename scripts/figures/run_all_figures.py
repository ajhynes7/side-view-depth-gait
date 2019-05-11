"""Run all scripts to generate figures."""

import matplotlib
from matplotlib import rc

from scripts.figures import body_graph, joint_proposals, signals, truth_positions


def main():

    matplotlib.rcParams.update({"pgf.texsystem": "pdflatex"})

    # Customize font
    rc('font', **{'family': 'serif', 'weight': 'bold', 'size': 14})
    rc('text', usetex=True)

    # Run scripts to make figures
    body_graph.main()
    joint_proposals.main()
    signals.main()
    truth_positions.main()


if __name__ == '__main__':
    main()
