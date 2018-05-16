import itertools
import numpy as np
import graphs as gr
import general as gen

from scipy.spatial.distance import cdist


def consecutive_lengths(points):

    n_points = len(points)
    lengths = np.zeros((n_points-1, 1))

    for i in range(n_points - 1):

        point_1 = points[i, :]
        point_2 = points[i + 1, :]

        lengths[i] = np.linalg.norm(point_1 - point_2)

    return lengths


def get_population(population_dict, part_types):
    """
    Parameters
    ----------

    population_dict : dict

    part_types : array_like
        List of strings for the types of body parts

    Returns
    -------
    population : array_like
        n x 3 matrix of all part position hypotheses

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
            in_spheres = gen.inside_spheres(dist_matrix, path, r)
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
    expected_matrix = gen.matrix_from_labels(expected_lengths, labels)

    vectorized_ratio_func = np.vectorize(gen.ratio_func)
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

    # Add in scores for foot height
    pop_height = population[:, 1].reshape(-1, 1)
    height_matrix = cdist(pop_height, pop_height)

    foot_label = labels.max()
    is_foot = (labels == foot_label).reshape(-1, 1)
    boolean_matrix = is_foot @ is_foot.T

    filtered_score_matrix[boolean_matrix] -= 0.01 * height_matrix[boolean_matrix]

    foot_1, foot_2 = select_best_feet(dist_matrix, filtered_score_matrix,
                                      path_matrix, radii)

    path_1, path_2 = path_matrix[foot_1, :], path_matrix[foot_2, :]
    pop_1, pop_2 = population[path_1, :], population[path_2, :]

    # Select the head along the minimum shortest path
    min_path = path_matrix[np.argmin(path_dist), :]
    head_pos = population[min_path[0], :]

    pop_1[0, :], pop_2[0, :] = head_pos, head_pos

    return pop_1, pop_2
