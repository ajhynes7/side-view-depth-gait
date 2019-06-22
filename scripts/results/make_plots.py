"""Make all plots of results."""

import matplotlib
from matplotlib import rc

from scripts.results import plot_accuracy_radii, plot_bland


def main():

    matplotlib.rcParams.update({"pgf.texsystem": "pdflatex"})

    # Customize font
    rc('font', **{'family': 'serif', 'weight': 'bold', 'size': 14})
    rc('text', usetex=True)

    plot_accuracy_radii.main()
    plot_bland.main()


if __name__ == '__main__':
    main()
