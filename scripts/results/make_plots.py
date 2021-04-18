"""Make all plots of results."""

import os
from os.path import join

import matplotlib
from matplotlib import rc

from scripts.results import (
    plot_accuracy_radii,
    plot_bland,
    plot_stride_width,
    plot_frame_rate,
)


def main():

    os.makedirs(join("results", "plots"), exist_ok=True)

    matplotlib.rcParams.update({"pgf.texsystem": "pdflatex"})

    # Customize font
    rc('font', **{'family': 'serif', 'weight': 'bold', 'size': 14})
    rc('text', usetex=True)

    plot_accuracy_radii.main()
    plot_bland.main()

    plot_stride_width.main()
    plot_frame_rate.main()


if __name__ == '__main__':
    main()
