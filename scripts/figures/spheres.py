"""Generate diagram of sphere process to select best feet."""

from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

import analysis.plotting as pl
import modules.pose_estimation as pe
from scripts.main.select_proposals import cost_func, score_func


def main():

    part_connections = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                                 [3, 5], [1, 3]])

    lengths = np.array([60, 20, 15, 20, 20])

    part_types = ['Head', 'Hip', 'Thigh', 'Knee', 'Calf', 'Foot']

    load_dir = join('data', 'saved_variables')
    population = np.load(join(load_dir, 'population.npy'))
    labels = np.load(join(load_dir, 'labels.npy'))

    # Remove some points to make the figure clearer
    to_keep = np.concatenate((np.arange(29), [31]))
    population = population[to_keep]
    labels = labels[to_keep]

    population[:, -1] = 0  # Remove depth

    # Add noisy foot
    population = np.vstack([population, [-20, 0, 0]])
    labels = np.append(labels, max(labels))

    # %% Calculate paths

    label_adj_list = pe.lengths_to_adj_list(part_connections, lengths)

    # Define a graph with edges between consecutive parts
    # (e.g. knee to calf, not knee to foot)
    cons_label_adj_list = pe.only_consecutive_labels(label_adj_list)

    # Run shortest path algorithm on the body graph
    prev, dist = pe.pop_shortest_paths(population, labels, cons_label_adj_list,
                                       cost_func)

    # Get shortest path to each foot
    paths, _ = pe.paths_to_foot(prev, dist, labels)

    n_pop = population.shape[0]
    path_extra = np.append(paths[-1, :-1], n_pop - 1)
    paths = np.vstack([paths, path_extra])

    # %% Plot joint proposals

    fig = plt.figure()

    pl.scatter_labels(population, labels, edgecolor='k', s=50)
    plt.legend(part_types, loc=[0.45, 0.5], edgecolor='k')
    plt.axis('equal')
    plt.axis('off')

    save_path = join('figures', 'sphere_proposals.pdf')
    fig.savefig(save_path, dpi=1200)

    # %% Plot proposals on shortest paths

    pop_reduced, paths_reduced = pe.reduce_population(population, paths)
    labels_reduced = labels[np.unique(paths)]

    fig = plt.figure()

    pl.scatter_labels(pop_reduced, labels_reduced, edgecolor='k', s=50)
    plt.axis('equal')
    plt.axis('off')

    save_path = join('figures', 'sphere_proposals_reduced.pdf')
    fig.savefig(save_path, dpi=1200)

    # %% Plot spheres

    r = 10

    n_pop_reduced = pop_reduced.shape[0]
    path_vectors = pe.get_path_vectors(paths_reduced, n_pop_reduced)

    dist_matrix = cdist(pop_reduced, pop_reduced)
    score_matrix = pe.get_scores(dist_matrix, paths_reduced, label_adj_list,
                                 score_func)

    pairs = [[1, 2], [2, 3], [0, 1]]

    n_figs = len(pairs)

    for i in range(n_figs):
        fig, ax = plt.subplots()

        has_sphere = np.any(path_vectors[pairs[i]], 0)

        within_radius = dist_matrix < r
        inside_spheres = pe.in_spheres(within_radius, has_sphere)
        pl.plot_links(pop_reduced, score_matrix, inside_spheres)

        pl.scatter_labels(
            pop_reduced, labels_reduced, s=50, edgecolor='k', zorder=5)

        has_sphere = np.any(path_vectors[pairs[i]], 0)
        pl.plot_spheres(pop_reduced[has_sphere], r, ax)

        plt.axis('equal')
        plt.axis('off')

        save_path = join('figures', 'spheres_{}.pdf')
        fig.savefig(save_path.format(i), dpi=1200)


if __name__ == '__main__':
    main()
