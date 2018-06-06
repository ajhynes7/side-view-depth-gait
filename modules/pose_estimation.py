import itertools
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist

import modules.graphs as gr
import modules.linear_algebra as lin
import modules.general as gen


def ratio_func(a, b):
    """
    Ratio between two positive inputs.
    If ratio a / b is less than one, the reciprocal is returned instead.

    Parameters
    ----------
    a, b : float
        Positive inputs.

    Returns
    -------
    float
        Ratio between a and b.
    """
    if a == 0 or b == 0:
        return np.nan

    ratio = np.divide(a, b)

    if ratio < 1:
        ratio = np.reciprocal(ratio)

    return ratio


def get_population(frame_series, part_labels):
    """
    Return the population of part hypotheses from one image frame.

    Parameters
    ----------
    frame_series : pandas series
        Index of the series is body parts.
        Values of the series are part hypotheses.
    part_labels : array_like
        Label for each body part in the series.
        e.g. L_FOOT and R_FOOT both have the label 5.

    Returns
    -------
    points : ndarray
        (n, d) array of n points with dimension d.
    labels : ndarray
        1-D array of labels.
        The labels are sorted in ascending order.

    Examples
    --------
    >>> head_points = np.array([-45, 66, 238]).reshape(-1, 3)
    >>> foot_points = np.array([[-26., -57, 249], [-74, -58, 260]])

    >>> frame_series = pd.Series({'L_FOOT': foot_points, 'HEAD': head_points})
    >>> part_labels = [5, 0]

    >>> points, labels = get_population(frame_series, part_labels)

    >>> points
    array([[-45.,  66., 238.],
           [-26., -57., 249.],
           [-74., -58., 260.]])

    >>> labels
    array([0, 5, 5])

    """
    pop_list, label_list = [], []

    for index_points, label in zip(frame_series, part_labels):

        for point in index_points:

            pop_list.append(point)
            label_list.append(label)

    population, labels = np.array(pop_list), np.array(label_list)

    # Sort the labels and apply the sorting to the points
    sort_index = np.argsort(labels)
    population, labels = population[sort_index], labels[sort_index]

    return population, labels


def lengths_lookup(edges, lengths):
    """


    Parameters
    ----------


    Returns
    -------

    """
    last_part = edges.max()

    expected_lengths = {i: {} for i in range(last_part+1)}

    n_rows = len(edges)

    for i in range(n_rows):
        u, v = edges[i, 0], edges[i, 1]

        expected_lengths[u][v] = sum(lengths[u:v])

    return expected_lengths


def dist_to_adj_matrix(dist_matrix, labels, expected_lengths, cost_func):
    """


    Parameters
    ----------


    Returns
    -------

    """
    n_nodes = len(dist_matrix)

    adj_matrix = np.full((n_nodes, n_nodes), np.nan)

    for i in range(n_nodes):
        label_i = labels[i]

        for j in range(n_nodes):
            label_j = labels[j]
            measured_length = dist_matrix[i, j]

            if label_j in expected_lengths[label_i]:
                expected_length = expected_lengths[label_i][label_j]

                adj_matrix[i, j] = cost_func(expected_length, measured_length)

    return adj_matrix


def paths_to_foot(prev, dist, labels):
    """
    Finds the path to each foot node

    Parameters
    ----------
    prev : array_like

    dist : array_like

    labels : array_like

    Returns
    -------
    path_matrix : array_like

    """

    max_label = max(labels)

    foot_index = np.where(labels == max_label)[0]
    n_feet = len(foot_index)

    path_matrix = np.full((n_feet, max_label+1), np.nan)
    path_dist = np.full(n_feet, np.nan)

    for i, foot in enumerate(foot_index):

        path_matrix[i, :] = gr.trace_path(prev, foot)
        path_dist[i] = dist[foot]

    return path_matrix.astype(int), path_dist


def get_score_matrix(population, labels, expected_lengths, score_func):
    
    label_dict = gen.iterable_to_dict(labels)
    dist_matrix = cdist(population, population)

    expected_adj_list = gr.labelled_nodes_to_graph(label_dict, expected_lengths)

    expected_matrix = gr.adj_list_to_matrix(expected_adj_list)

    vectorized_score_func = np.vectorize(score_func)

    score_matrix = vectorized_score_func(dist_matrix, expected_matrix)
    score_matrix[np.isnan(score_matrix)] = 0

    return score_matrix, dist_matrix


def frame_paths(population, labels, expected_lengths, cost_func):

    # Represent population as a weighted directed acyclic graph
    pop_graph = gr.points_to_graph(
        population, labels, expected_lengths, cost_func)

    # Run shortest path algorithm
    head_nodes = np.where(labels == 0)[0]  # Source nodes
    order = pop_graph.keys()  # Topological ordering of the nodes
    prev, dist = gr.dag_shortest_paths(pop_graph, order, head_nodes)

    # Get shortest path to each foot
    path_matrix, path_dist = paths_to_foot(prev, dist, labels)

    return path_matrix, path_dist


def filter_by_path(input_matrix, path_matrix, part_connections):
    """


    Parameters
    ----------


    Returns
    -------

    """
    filtered_matrix = np.zeros(input_matrix.shape)
    n_paths, n_path_nodes = path_matrix.shape

    for i in range(n_paths):
        for j in range(n_path_nodes):
            for k in range(n_path_nodes):

                if k in part_connections[j]:
                    # These nodes in the path are connected in the body part graph
                    A, B = path_matrix[i, j], path_matrix[i, k]
                    filtered_matrix[A, B] = input_matrix[A, B]

    return filtered_matrix


def inside_spheres(dist_matrix, point_nums, r):
    """
    Given n points, m of these points are centres of spheres.
    Calculates which of the n points are contained inside these m spheres.

    Parameters
    ----------
    dist_matrix : ndarray
        | (n, n) distance matrix
        | Element (i, j) is distance from point i to point j

    point_nums : array_like
        | (m, ) List of points that are the sphere centres
        | Numbers between 1 and n

    r : float
        Radius of spheres

    Returns
    -------
    in_spheres : array_like
        (n,) array of bools
        Element i is true if point i is in the set of spheres
    """
    n_points = len(dist_matrix)

    in_spheres = np.full(n_points, False)

    for i in point_nums:

        distances = dist_matrix[i, :]

        in_current_sphere = distances <= r
        in_spheres = in_spheres | in_current_sphere

    return in_spheres


def inside_radii(dist_matrix, path_matrix, radii):
    """


    Parameters
    ----------


    Returns
    -------

    """
    in_spheres_list = []

    for path in path_matrix:
        temp = []

        for r in radii:
            in_spheres = inside_spheres(dist_matrix, path, r)
            temp.append(in_spheres)

        in_spheres_list.append(temp)

    return in_spheres_list


def select_best_feet(dist_matrix, score_matrix, path_matrix, radii):
    """


    Parameters
    ----------


    Returns
    -------

    """
    n_paths = len(path_matrix)
    n_radii = len(radii)

    in_spheres_list = inside_radii(dist_matrix, path_matrix, radii)

    # All possible pairs of paths
    combos = list(itertools.combinations(range(n_paths), 2))

    n_combos = len(combos)

    votes, combo_scores = np.zeros(n_combos), np.zeros(n_combos)

    for i in range(n_radii):

        for ii, combo in enumerate(combos):

            in_spheres_1 = in_spheres_list[combo[0]][i]
            in_spheres_2 = in_spheres_list[combo[1]][i]

            in_spheres = in_spheres_1 | in_spheres_2

            temp = score_matrix[in_spheres, :]
            score_subset = temp[:, in_spheres]

            combo_scores[ii] = np.sum(score_subset)

        max_score = max(combo_scores)

        # Winning combos for this radius
        radius_winners = combo_scores == max_score

        # Votes go to the winners
        votes = votes + radius_winners

    winning_combo = np.argmax(votes)
    foot_1, foot_2 = combos[winning_combo]

    return foot_1, foot_2


def foot_to_pop(population, path_matrix, path_dist, foot_1, foot_2):

    path_1, path_2 = path_matrix[foot_1, :], path_matrix[foot_2, :]
    pop_1, pop_2 = population[path_1, :], population[path_2, :]

    # Select the head along the minimum shortest path
    min_path = path_matrix[np.argmin(path_dist), :]
    head_pos = population[min_path[0], :]

    pop_1[0, :], pop_2[0, :] = head_pos, head_pos

    return pop_1, pop_2


def process_frame(population, labels, expected_lengths_all, radii, cost_func, score_func):
    """


    Parameters
    ----------


    Returns
    -------

    """
    
    path_matrix, path_dist = frame_paths(population, labels, expected_lengths_all, cost_func)
    
    score_matrix, dist_matrix = get_score_matrix(population, labels, expected_lengths_all, score_func)
    
    filtered_score_matrix = filter_by_path(score_matrix, path_matrix,
                                        expected_lengths_all)
    
    foot_1, foot_2 = select_best_feet(dist_matrix, filtered_score_matrix,
                                      path_matrix, radii)

    pop_1, pop_2 = foot_to_pop(population, path_matrix, path_dist, foot_1, foot_2)

    return pop_1, pop_2


def assign_LR(foot_A, foot_B, line_vector):

    # Up direction defined as positive y
    up = np.array([0, 1, 0])

    # Points on line
    line_point_A = (foot_A + foot_B) / 2

    # Vector from mean foot position to current foot
    target_direction = foot_A - line_point_A

    # Check if point is left or right of the line
    return lin.angle_direction(target_direction, line_vector, up)


def consistent_sides(df_head_feet, frame_labels):
    """
    Parameters
    ----------
    df_head_feet : DataFrame
    frame_labels : array_like

    Returns
    -------
    switch_sides : Series
    """

    frames = df_head_feet.index.values

    n_frames = len(frame_labels)
    n_labels = frame_labels.max() + 1

    switch_sides = pd.Series(np.full(n_frames, False), index=frames)

    # Loop through each main cluster, i.e., each walking pass
    for i in range(n_labels):

        cluster_i = frame_labels == i

        # All head positions on one walking pass
        head_points = np.stack(tuple(df_head_feet['HEAD'][cluster_i]))

        # Line of best fit for head positions
        _, direction = lin.best_fit_line(head_points)

        cluster_frames = frames[frame_labels == i]

        for frame in cluster_frames:

            foot_A = df_head_feet.loc[frame, 'L_FOOT']
            foot_B = df_head_feet.loc[frame, 'R_FOOT']

            side_of_A = assign_LR(foot_A, foot_B, direction)

            if side_of_A == 1:
                # A value of 1 indicates that foot A is on the right
                # of the line of best fit
                # Thus, the sides should be switched
                switch_sides.loc[frame] = True

    return switch_sides


if __name__ == "__main__":

    import doctest
    doctest.testmod()
