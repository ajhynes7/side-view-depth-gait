"""Generate diagram of body part graph."""

import numpy as np
import matplotlib.pyplot as plt

import analysis.plotting as pl
import modules.iterable_funcs as itf


def generate_points(n_points_per_set):
    """Yield points showing the nodes of the graph."""
    n_sets = len(n_points_per_set)

    for i, n_points in enumerate(n_points_per_set):

        y_vals = (n_sets - i) * np.ones(n_points)
        x_vals = np.linspace(0, 1, n_points + 1, endpoint=False)[1:]

        yield np.stack([(x, y) for x, y in zip(x_vals, y_vals)])


points_per_set = [2, 3, 5, 2, 4, 5]
part_types = ['Head', 'Hip', 'Thigh', 'Knee', 'Calf', 'Foot']

fig = plt.figure()

point_sets = [*generate_points(points_per_set)]

for points_a, points_b in itf.pairwise(point_sets):

    points = np.vstack([points_a, points_b])

    pl.scatter2(points, c='k', s=50)
    pl.connect_two_sets(points_a, points_b, c='k')

# Highlight path in red
prev_path_point = np.array([])
for row_points in point_sets:

    path_point = row_points[np.random.randint(row_points.shape[0])]

    pl.scatter2(path_point, c='r', s=100, zorder=3)

    if prev_path_point.size > 0:
        pl.connect_points(prev_path_point, path_point, c='r', zorder=3)

    prev_path_point = path_point

# Label rows with body part names
y_coords = [points[0, 1] for points in point_sets]

for ii, part_type in enumerate(part_types):
    plt.text(1, y_coords[ii] - 0.075, part_type, fontsize=12)

plt.xlim((0, 1.1))
plt.axis('off')
plt.show()

# fig.savefig('body_graph.pdf', format='pdf', dpi=1200)
