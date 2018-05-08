
import numpy as np


def get_population(population_dict, part_types):

    population_list, labels = [], []

    for i, part_type in enumerate(part_types):
        for full_part_name in population_dict:

            if part_type in full_part_name:
                points = population_dict[full_part_name]
                n_points, _ = points.shape

                population_list.append(points)
                labels.extend([i for _ in range(n_points)])

    # Convert list to numpy matrix
    population = np.concatenate(population_list)

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


def distances_to_adj_matrix(dist_matrix, labels, expected_lengths, cost_func):

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


def paths_to_foot(prev, labels):

    n_nodes = len(prev);
    max_label = max(labels)

    foot_index = np.where(labels == max_label)

    for i, foot in enumerate(foot_index):
        path_matrix[i, :] = trace_path(prev, foot)

    return path_matrix
