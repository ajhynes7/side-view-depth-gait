import itertools
import numpy as np
import graphs as gr
import general as gen
import read_data as rd

from scipy.spatial.distance import cdist


def get_population(population_dict, part_types):

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

    last_part = edges.max()

    expected_lengths = {i: {} for i in range(last_part+1)}

    n_rows = len(edges)

    for i in range(n_rows):
            u, v = edges[i, 0], edges[i, 1]

            expected_lengths[u][v] = sum(lengths[u:v])

    return expected_lengths


def dist_to_adj_matrix(dist_matrix, labels, expected_lengths, cost_func):

    n_nodes = len(dist_matrix)

    M = np.full((n_nodes, n_nodes), np.nan)

    for i in range(n_nodes):
        label_A = labels[i]

        for j in range(n_nodes):
            label_B = labels[j]

            if label_B in expected_lengths[label_A]:
                expected_length = expected_lengths[label_A][label_B]
                measured_length = dist_matrix[i, j]

                M[i, j] = cost_func(expected_length, measured_length)

    return M


def paths_to_foot(prev, dist, labels):
    """
    Params
    ------
    
    Outputs
    -------
    """

    max_label = max(labels)

    foot_index = np.where(labels == max_label)[0]
    n_feet = len(foot_index)

    path_matrix = np.full((n_feet, max_label+1), np.nan)
    path_dist   = np.full(n_feet, np.nan)

    for i, foot in enumerate(foot_index):

        path_matrix[i, :] = gr.trace_path(prev, foot)
        path_dist[i] = dist[foot]

    return path_matrix.astype(int), path_dist


def filter_by_path(input_matrix, path_matrix, expected_lengths):

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

    in_spheres_list = []

    for path in path_matrix:
        temp = []

        for r in radii:
            in_spheres = gen.inside_spheres(dist_matrix, path, r)
            temp.append(in_spheres)

        in_spheres_list.append(temp)

    return in_spheres_list


def select_best_feet(dist_matrix, score_matrix, path_matrix, radii):

    n_paths = len(path_matrix)
    n_radii = len(radii)

    in_spheres_list = inside_radii(dist_matrix, path_matrix, radii)

    # All possible pairs of paths
    combos = list(itertools.combinations(range(n_paths), 2))

    n_combos = len(combos)

    votes, combo_scores = np.zeros(n_combos), np.zeros(n_combos)

    for i in range(n_radii):

        for ii, combo in enumerate(combos):

            in_spheres_A = in_spheres_list[combo[0]][i]
            in_spheres_B = in_spheres_list[combo[1]][i]

            in_spheres = in_spheres_A | in_spheres_B

            temp = score_matrix[in_spheres, :]
            score_subset = temp[:, in_spheres]

            combo_scores[ii] = np.sum(score_subset)

        max_score = max(combo_scores)

        # Winning combos for this radius
        radius_winners = combo_scores == max_score;

        # Votes go to the winners
        votes = votes + radius_winners;

    winning_combo = np.argmax(votes)

    foot_A = combos[winning_combo][0]
    foot_B = combos[winning_combo][1]

    return foot_A, foot_B


def process_frame(frame_df, pop_dict, part_types, edges, lengths, radii):

    cost_func   = lambda a, b: (a - b)**2
    score_func  = lambda x : -(x - 1)**2 + 1
    ratio_func  = lambda a, b: 1.0 / gen.norm_ratio(a, b)

    n_lengths = len(lengths)
    edges_simple = edges[range(n_lengths), :]

    population, labels = get_population(pop_dict, part_types)

    expected_lengths        = lengths_lookup(edges, lengths)
    expected_lengths_simple = lengths_lookup(edges_simple, lengths)

    dist_matrix = cdist(population, population)

    ratio_matrix = dist_to_adj_matrix(dist_matrix, labels, \
                                              expected_lengths, ratio_func)
    score_matrix = score_func(ratio_matrix)

    M = dist_to_adj_matrix(dist_matrix, labels, \
                                   expected_lengths_simple, cost_func)
    G = gr.adj_matrix_to_list(M)

    source_nodes = np.where(labels == 0)[0]
    prev, dist = gr.dag_shortest_paths(G, G.keys(), source_nodes)

    path_matrix, path_dist = paths_to_foot(prev, dist, labels)

    filtered_score_matrix = filter_by_path(score_matrix, path_matrix,\
                                              expected_lengths_simple)

    foot_A, foot_B = select_best_feet(dist_matrix, filtered_score_matrix,\
                                      path_matrix, radii)

    path_A, path_B  = path_matrix[foot_A, :], path_matrix[foot_B, :]
    pop_A, pop_B    = population[path_A, :], population[path_B, :]

    # Select the head along the minimum shortest path
    min_path = path_matrix[np.argmin(path_dist), :]
    head_pos = population[min_path[0], :]

    pop_A[0, :], pop_B[0, :] = head_pos, head_pos

    return pop_A, pop_B
