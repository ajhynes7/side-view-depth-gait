import itertools
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist

import modules.graphs as gr
import modules.linear_algebra as lin


def matrix_from_labels(expected_values, labels):

    n_rows = len(labels)

    mat = np.full((n_rows, n_rows), np.nan)

    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):

            if label_j in expected_values[label_i]:
                mat[i, j] = expected_values[label_i][label_j]

    return mat


def score_func(measured, actual):

    absolute_error = relative_error(measured, actual, absolute=True)
    normalized_error = sigmoid(absolute_error)

    return math.log(-normalized_error + 1) + 1


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


def get_population(population_dict, part_types):
    """
    Parameters
    ----------

    population_dict : dict

    part_types : array_like
        List of strings for the types of body parts.

    Returns
    -------
    population : array_like
        n x 3 matrix of all part position hypotheses.

    labels :
    """
    population_list, label_list = [], []

    for i, part_type in enumerate(part_types):
        for full_part_name in population_dict:

            if part_type in full_part_name:
                points = population_dict[full_part_name]
                n_points, _ = points.shape

                population_list.append(points)
                label_list.extend([i for _ in range(n_points)])

    # Convert list to numpy matrix
    population = np.concatenate(population_list)
    labels = np.array(label_list)

    assert(population.shape[0] == len(labels))

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


def filter_by_path(input_matrix, path_matrix, expected_lengths):
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

                if k in expected_lengths[j]:
                    # These nodes in the path are connected in the body graph
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


def process_frame(pop_dict, part_types, edges, lengths, radii):
    """


    Parameters
    ----------


    Returns
    -------

    """
    def cost_func(a, b): return (a - b)**2

    def score_func(x): return -(x - 1)**2 + 1

    population, labels = get_population(pop_dict, part_types)

    if len(np.unique(labels)) != len(part_types):
        return np.nan, np.nan

    n_lengths = len(lengths)
    edges_simple = edges[range(n_lengths), :]

    expected_lengths = lengths_lookup(edges, lengths)
    expected_lengths_simple = lengths_lookup(edges_simple, lengths)

    dist_matrix = cdist(population, population)
    expected_matrix = matrix_from_labels(expected_lengths, labels)

    vectorized_ratio_func = np.vectorize(ratio_func)
    ratio_matrix = vectorized_ratio_func(dist_matrix, expected_matrix)

    score_matrix = score_func(ratio_matrix)
    score_matrix[np.isnan(score_matrix)] = 0

    adj_matrix = dist_to_adj_matrix(dist_matrix, labels,
                                    expected_lengths_simple, cost_func)

    adj_list = gr.adj_matrix_to_list(adj_matrix)

    source_nodes = np.where(labels == 0)[0]
    prev, dist = gr.dag_shortest_paths(adj_list, adj_list.keys(), source_nodes)

    path_matrix, path_dist = paths_to_foot(prev, dist, labels)

    filtered_score_matrix = filter_by_path(score_matrix, path_matrix,
                                           expected_lengths)

    foot_1, foot_2 = select_best_feet(dist_matrix, filtered_score_matrix,
                                      path_matrix, radii)

    path_1, path_2 = path_matrix[foot_1, :], path_matrix[foot_2, :]
    pop_1, pop_2 = population[path_1, :], population[path_2, :]

    # Select the head along the minimum shortest path
    min_path = path_matrix[np.argmin(path_dist), :]
    head_pos = population[min_path[0], :]

    pop_1[0, :], pop_2[0, :] = head_pos, head_pos

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
        centroid, direction = lin.best_fit_line(head_points)

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
