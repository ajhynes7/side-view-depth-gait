"""Create figures showing 2D positions of joint proposals."""

import os

import numpy as np
import matplotlib.pyplot as plt

import analysis.plotting as pl
import modules.numpy_funcs as nf


def plot_spheres(paths):
    """Plot circles around 2D joint proposals."""
    for path in paths:
        pl.scatter2(
            population[path, :], s=1e3, facecolors='none', edgecolors='k')


def plot_links(label_adj_list, path_matrix):
    """Plot the relevant score matrix links depending on the spheres.""" 
    for path in path_matrix:
        for u in label_adj_list:
            for v in label_adj_list[u]:
                a, b = path[u], path[v]
                point_a, point_b = population[a], population[b]
                pl.connect_points(point_a, point_b, c='k', linewidth=0.5)


load_dir = os.path.join('data', 'saved_variables')

population = np.load(os.path.join(load_dir, 'population.npy'))
labels = np.load(os.path.join(load_dir, 'labels.npy'))
path_matrix = np.load(os.path.join(load_dir, 'path_matrix.npy'))

part_types = ['Head', 'Hip', 'Thigh', 'Knee', 'Calf', 'Foot']

# ### Add noisy foot

population = np.vstack([population, [-20, -20, 300]])
labels = np.append(labels, max(labels))

path_extra = np.append(path_matrix[-1, :-1], len(labels) - 1)
path_matrix = np.vstack([path_matrix, path_extra])

# ## Customize font

plt.rc('text', usetex=True)
font = {'family': 'serif', 'weight': 'bold', 'size': 12}

plt.rc('font', **font)  # pass in the font dict as kwargs

# ### Plot population

fig = plt.figure()

pl.scatter_labels(population, labels)

plt.legend(part_types)
plt.xlim((-150, 0))
plt.xlabel('X')
plt.ylabel('Y', rotation=0)
plt.show()

fig.savefig('labelled_points.pdf', format='pdf', dpi=1200)

# ### Show score matrix filtering

label_adj_list = {0: {1}, 1: {2, 3}, 2: {3}, 3: {4, 5}, 4: {5}}

fig = plt.figure()

plt.subplot(1, 2, 1)
pl.scatter_labels(population, labels)

# Plot links of unfiltered score matrix
population_groups = list(nf.group_by_label(population, labels))

for u in label_adj_list:
    for v in label_adj_list[u]:
        points_u, points_v = population_groups[u], population_groups[v]
        pl.connect_two_sets(points_u, points_v, c='k', linewidth=0.07)

plt.axis('off')

plt.subplot(1, 2, 2)
pl.scatter_labels(population, labels)
plt.legend(part_types)

# Plot links of filtered score matrix
for path in path_matrix:
    for u in label_adj_list:
        for v in label_adj_list[u]:
            a, b = path[u], path[v]

            point_a, point_b = population[a], population[b]
            pl.connect_points(point_a, point_b, c='k', linewidth=0.5)

plt.axis('off')
plt.show()

fig.savefig('score_matrix.png', format='png')

# ### Plot spheres


fig = plt.figure()

ax1 = plt.subplot(1, 3, 1)
path_matrix_1 = path_matrix[[-1, 2], :]
pl.scatter_labels(population, labels)
plot_spheres(path_matrix_1)

ax2 = plt.subplot(1, 3, 2)
path_matrix_2 = path_matrix[[1, 3], :]
pl.scatter_labels(population, labels)
plot_spheres(path_matrix_2)

ax3 = plt.subplot(1, 3, 3)
path_matrix_3 = path_matrix[[1, 2], :]
pl.scatter_labels(population, labels)
plot_spheres(path_matrix_3)

ax1.set_xlabel('(a)')
ax2.set_xlabel('(b)')
ax3.set_xlabel('(c)')

ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax3.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

ax1.tick_params(axis='y', which='both', left=False, labelleft=False)
ax2.tick_params(axis='y', which='both', left=False, labelleft=False)
ax3.tick_params(axis='y', which='both', left=False, labelleft=False)

ax1.set_xlim([-150, 50])
ax2.set_xlim([-150, 50])
ax3.set_xlim([-150, 50])

plt.subplots_adjust(wspace=0, hspace=0)

plt.show()

fig.savefig('spheres.pdf', format='pdf', dpi=1200)