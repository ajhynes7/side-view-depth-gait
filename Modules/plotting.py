import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def scatter_colour(points, colours, labels):
    """


    Parameters
    ----------


    Returns
    -------

    """
    plt.figure()

    for i, c in enumerate(colours):
        plt.scatter(points[i, 0], points[i, 1], color=c,
                    label='{i}'.format(i=labels[i]))

    plt.legend(loc='best')
    plt.show()


def scatter_pos(points):
    """


    Parameters
    ----------


    Returns
    -------

    """
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(points[:, 0], points[:, 2], points[:, 1])

    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(100, 300)
    ax.set_zlim3d(-100, 100)
