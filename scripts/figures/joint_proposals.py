"""Plot multiple figures involving joint proposals."""

import glob
import pickle
import re
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.cm import get_cmap
from scipy.spatial.distance import cdist

import analysis.images as im
import analysis.plotting as pl
import modules.pose_estimation as pe
from modules.constants import TYPE_CONNECTIONS, PART_CONNECTIONS


def scatter_parts(points, labels, name_cmap: str = 'Set1', **kwargs):
    """
    Plot scatter points of body parts with different colours and shape depending on the label.

    Parameters
    ----------
    points : (N, 2) ndarray
        Array of N points with dimension 2.
    labels : (N,) ndarray
        Array of point labels.
    name_cmap: str
        Name of the colormap.

    """

    labels_unique = np.unique(labels)

    cmap = get_cmap(name_cmap)

    cycler_ = cycler(color=cmap.colors[:6]) + cycler(marker=['o', '^', 's', '*', 'P', 'X'])

    for label, dict_format in zip(labels_unique, cycler_):

        points_label = points[labels == label]

        plt.scatter(points_label[:, 0], points_label[:, 1], **dict_format, **kwargs)


def main():

    kinect_dir = join('data', 'kinect')

    df_truth = pd.read_pickle(join(kinect_dir, 'df_truth.pkl'))
    labelled_trial_names = df_truth.index.get_level_values(0).unique()

    # Specify image file
    trial_name = labelled_trial_names[0]
    file_index = 271

    # %% Load depth image

    depth_dir = join(kinect_dir, 'labelled_trials', trial_name, 'depth16bit')
    depth_paths = sorted(glob.glob(join(depth_dir, '*.png')))

    depth_path = depth_paths[file_index]
    depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    # %% Obtain joint proposals for the depth image

    df_hypo = pd.read_pickle(join(kinect_dir, 'df_hypo.pkl'))

    match_object = re.search(r'(\d+).png', depth_path)
    image_number = int(match_object.group(1))

    # Load dictionary to convert image numbers to frames
    with open(join(kinect_dir, 'alignment', "{}.pkl".format(trial_name)), 'rb') as handle:
        image_to_frame = pickle.load(handle)

    frame = image_to_frame[image_number]
    population, labels = df_hypo.loc[trial_name].loc[frame]

    points_image = np.apply_along_axis(im.real_to_image, 1, population, im.X_RES, im.Y_RES, im.F_XZ, im.F_YZ)

    # %% Plot joint proposals on depth image

    part_types = ['Head', 'Hip', 'Thigh', 'Knee', 'Calf', 'Foot']

    fig, ax = plt.subplots()

    ax.imshow(depth_image, cmap='gray')
    scatter_parts(points_image[:, :2], labels, edgecolors='k', s=100)
    plt.legend(part_types, framealpha=1, loc='upper left', fontsize=12)

    ax.set_yticks([])
    ax.set_xticks([])

    fig.savefig(join('figures', 'joint_proposals_image.png'))

    # %% Plot joint proposals

    legend_location = [0.2, 0.6]

    fig = plt.figure()
    scatter_parts(population, labels, edgecolor='k', s=100)
    plt.axis('equal')
    plt.axis('off')
    plt.legend(part_types, loc=legend_location, edgecolor='k')
    fig.savefig(join('figures', 'joint_proposals.pdf'), dpi=1200)

    # %% Plot reduced joint proposals

    df_lengths = pd.read_csv(join(kinect_dir, 'kinect_lengths.csv'), index_col=0)
    lengths = df_lengths.loc[trial_name]

    label_adj_list_types = pe.lengths_to_adj_list(TYPE_CONNECTIONS, lengths)

    # Run shortest path algorithm on the body graph
    dist_matrix = cdist(population, population)
    prev, dist = pe.pop_shortest_paths(dist_matrix, labels, label_adj_list_types, pe.cost_func)

    # Get shortest path to each foot
    paths, _ = pe.paths_to_foot(prev, dist, labels)

    pop_reduced, paths_reduced = pe.reduce_population(population, paths)
    labels_reduced = labels[np.unique(paths)]

    fig = plt.figure()
    scatter_parts(pop_reduced, labels_reduced, edgecolor='k', s=100)
    plt.axis('equal')
    plt.axis('off')
    fig.savefig(join('figures', 'joint_proposals_reduced.pdf'), dpi=1200)

    # %% Plot spheres

    r = 10
    pairs = [[0, 1], [2, 3], [1, 2]]

    n_pop_reduced = pop_reduced.shape[0]
    path_vectors = pe.get_path_vectors(paths_reduced, n_pop_reduced)

    label_adj_list_parts = pe.lengths_to_adj_list(PART_CONNECTIONS, lengths)

    dist_matrix_reduced = cdist(pop_reduced, pop_reduced)
    score_matrix = pe.get_scores(dist_matrix_reduced, paths_reduced, label_adj_list_parts, pe.score_func)

    n_figs = len(pairs)

    for i in range(n_figs):
        fig, ax = plt.subplots()

        scatter_parts(pop_reduced, labels_reduced, s=100, edgecolor='k', zorder=5)

        if i == 0:
            # Add legend to first figures
            plt.legend(part_types, loc=legend_location, edgecolor='k')

        has_sphere = np.any(path_vectors[pairs[i]], 0)
        within_radius = dist_matrix_reduced < r

        inside_spheres = pe.in_spheres(within_radius, has_sphere)
        pl.plot_links(pop_reduced, score_matrix, inside_spheres)

        has_sphere = np.any(path_vectors[pairs[i]], 0)
        pl.plot_spheres(pop_reduced[has_sphere], r, ax)

        plt.axis('equal')
        plt.axis('off')

        save_path = join('figures', 'spheres_{}.pdf')
        fig.savefig(save_path.format(i), dpi=1200)


if __name__ == '__main__':
    main()
