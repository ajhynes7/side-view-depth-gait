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


def scatter_pos(fig, point_list, colors='b'):
    """


    Parameters
    ----------


    Returns
    -------

    """

    ax = Axes3D(fig)

    # The z values of the points represent depth values,
    # while y values represent height
    # Thus, y and z are switched for plotting
    for i, points in enumerate(point_list):
        ax.scatter(points[:, 0], points[:, 2], points[:, 1],
                   c=colors[i], depthshade=False)

    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(100, 300)
    ax.set_zlim3d(-100, 100)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')


def scatter2(points):
    plt.scatter(points[:, 0], points[:, 1])
