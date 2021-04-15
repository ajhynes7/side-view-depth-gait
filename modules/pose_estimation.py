"""
Module for estimating the pose of a walking person.

The pose is estimated by selecting body parts from a set of hypotheses.

"""
import itertools
from typing import Mapping, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.spatial.distance import cdist

import modules.graphs as gr
import modules.math_funcs as mf
from modules.constants import PART_CONNECTIONS, PART_TYPES, TYPE_CONNECTIONS
from modules.typing import adj_list, array_like, func_ab


def cost_func(a: float, b: float) -> float:
    """Cost function for weighting edges of graph."""
    return (a - b) ** 2


def score_func(a: float, b: float) -> float:
    """Score function for scoring links between body parts."""
    x = 1 / mf.norm_ratio(a, b)
    return -((x - 1) ** 2) + 1


def measure_min_path(population: ndarray, labels: ndarray, label_adj_list: adj_list) -> ndarray:
    """
    Measure lengths along the minimum shortest path.

    Parameters
    ----------
    population : (N, 3) ndarray
        All position hypotheses on a frame.
    labels : (N,) ndarray
        Array of labels for N positions.
        The labels correspond to body part types (e.g., foot).
        They are sorted in ascending order.
    label_adj_list : dict
        Adjacency list for the labels.
        label_adj_list[A][B] is the expected distance between
        a point with label A and a point with label B.

    Returns
    -------
    lengths_measured : (N_lengths,) ndarray
        Lengths on the minimum shortest path.

    """
    dist_matrix = cdist(population, population)
    prev, dist = pop_shortest_paths(dist_matrix, labels, label_adj_list, cost_func)

    # Get shortest path to each foot
    paths, path_dist = paths_to_foot(prev, dist, labels)
    path_minimum = paths[np.argmin(path_dist)]

    lengths_measured = dist_matrix[path_minimum[:-1], path_minimum[1:]]

    return lengths_measured


def estimate_lengths(df_hypo_trial: pd.DataFrame, **kwargs) -> ndarray:
    """
    Estimate the lengths between adjacent body parts in a walking trial.

    Parameters
    ----------
    df_hypo_trial : DataFrame
        Dataframe of position hypotheses for a walking trial.
        Columns include 'population' and 'labels'.
    kwargs : dict, optional
        Keyword arguments passed to `np.allclose`.

    Returns
    -------
    lengths_estimated: ndarray
        Array of estimated lengths.
        These are the expected lengths for the walking trial.

    """
    n_frames = df_hypo_trial.shape[0]
    n_lengths = len(PART_TYPES) - 1

    matrix_lengths_measured = np.full((n_frames, n_lengths), np.nan)

    lengths_estimated = np.zeros(n_lengths)
    lengths_prev = np.full(n_lengths, np.inf)

    # Use a for loop so algorithm will terminate if convergence does not occur.
    for _ in range(10):

        label_adj_list_types = lengths_to_adj_list(TYPE_CONNECTIONS, lengths_estimated)

        medians_prev = np.full(n_lengths, np.inf)  # Initiate medians.
        lengths_prev = np.copy(lengths_estimated)  # Record previous lengths.

        for i, tuple_frame in enumerate(df_hypo_trial.itertuples()):

            population, labels = tuple_frame.population, tuple_frame.labels

            lengths_measured = measure_min_path(population, labels, label_adj_list_types)

            matrix_lengths_measured[i] = lengths_measured
            matrix_lengths_so_far = matrix_lengths_measured[: i + 1]

            medians = np.median(matrix_lengths_so_far, axis=0)

            if np.allclose(medians, medians_prev, **kwargs):
                lengths_estimated = np.copy(medians)
                break

            medians_prev = np.copy(medians)

        if np.allclose(lengths_estimated, lengths_prev, **kwargs):
            break

    return lengths_estimated


def get_population(list_frame_points: Sequence, part_labels: array_like) -> Tuple[ndarray, ndarray]:
    """
    Return the population of part hypotheses from one image frame.

    Parameters
    ----------
    list_frame_points : Sequence
        Sequence of points on a frame.
        Each element is an array of points for a body part.
    part_labels : array_like
        Corresponding label for each body part.
        e.g. L_FOOT and R_FOOT both have the label 5.

    Returns
    -------
    population : (N, 3) ndarray
        All position hypotheses on a frame.
    labels : (N,) ndarray
        Array of labels for N positions.
        The labels correspond to body part types (e.g., foot).
        They are sorted in ascending order.

    Examples
    --------
    >>> head_points = [[-45, 66, 238]]
    >>> foot_points = [[-26., -57, 249], [-74, -58, 260]]

    >>> population, labels = get_population([foot_points, head_points], [5, 0])

    >>> population
    array([[-45.,  66., 238.],
           [-26., -57., 249.],
           [-74., -58., 260.]])

    >>> labels
    array([0, 5, 5])

    """
    pop_list, label_list = [], []

    for index_points, label in zip(list_frame_points, part_labels):

        for point in index_points:

            pop_list.append(point)
            label_list.append(label)

    population, labels = np.array(pop_list), np.array(label_list)

    # Sort the labels and apply the sorting to the points
    sort_index = np.argsort(labels)
    population, labels = population[sort_index], labels[sort_index]

    return population, labels


def lengths_to_adj_list(label_connections: ndarray, lengths: array_like) -> adj_list:
    """
    Convert a array_like of lengths between body parts to an adjacency list.

    Parameters
    ----------
    label_connections : ndarray
        Each row is a connection from label A to label B.
        Column 1 is label A, column 2 is label B.
    lengths : array_like
        List of lengths between consecutive body parts
        (e.g., calf to foot).

    Returns
    -------
    label_adj_list : dict
        Adjacency list for the labels.
        label_adj_list[A][B] is the expected distance between
        a point with label A and a point with label B.

    Examples
    --------
    >>> label_connections = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [3, 5]])
    >>> lengths = [62, 20, 14, 19, 20]

    >>> lengths_to_adj_list(label_connections, lengths)
    {0: {}, 1: {2: 20}, 2: {3: 14}, 3: {4: 19, 5: 39}, 4: {5: 20}, 5: {}}

    """
    last_part = int(label_connections.max())
    label_adj_list: dict = {i: {} for i in range(last_part + 1)}

    n_rows = len(label_connections)

    for i in range(n_rows):
        u, v = label_connections[i, 0], label_connections[i, 1]

        label_adj_list[u][v] = sum(lengths[u:v])

    return label_adj_list


def pop_shortest_paths(
    dist_matrix: ndarray, labels: ndarray, label_adj_list: adj_list, weight_func: func_ab
) -> Tuple[Mapping[int, int], Mapping[int, float]]:
    """
    Calculate shortest paths on the population of body parts.

    Parameters
    ----------
    dist_matrix : (N, N) ndarray
        Distance matrix of the points.
    labels : (N,) ndarray
        Array of labels for N positions.
        The labels correspond to body part types (e.g., foot).
    label_adj_list : dict
        Adjacency list for the labels.
        label_adj_list[A][B] is the expected distance between
        a point with label A and a point with label B.
    weight_func : function
        Function used to weight edges of the graph.

    Returns
    -------
    prev : dict
        For each node u in the graph, prev[u] is the previous node
        on the shortest path to u.
    dist : dict
        For each node u in the graph, dist[u] is the total distance (weight)
        of the shortest path to u.

    """
    # Represent population as a weighted directed acyclic graph
    pop_graph = gr.points_to_graph(dist_matrix, labels, label_adj_list, weight_func)

    # Run shortest path algorithm
    head_nodes = np.where(labels == 0)[0]  # Source nodes
    order = list(pop_graph.keys())  # Topological ordering of the nodes

    prev, dist = gr.dag_shortest_paths(pop_graph, order, head_nodes)

    return prev, dist


def paths_to_foot(prev: Mapping[int, int], dist: Mapping[int, float], labels: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Retrieve the shortest path to each foot position.

    Parameters
    ----------
    prev : dict
        For each node u in the graph, prev[u] is the previous node
        on the shortest path to u.
    dist : dict
        For each node u in the graph, dist[u] is the total distance (weight)
        of the shortest path to u.
    labels : ndarray
        Label of each node.

    Returns
    -------
    paths : ndarray
        One row for each foot position.
        Each row is a shortest path from head to foot.
    path_dist : ndarray
        Total distance of the path to each foot.

    Examples
    --------
    >>> prev = {0: np.nan, 1: 0, 2: 1, 3: 2, 4: 3, 5: 3}
    >>> dist = {0: 0, 1: 0, 2: 20, 3: 5, 4: 11, 5: 10}

    >>> labels = np.array([0, 1, 2, 3, 4, 4])

    >>> paths, path_dist = paths_to_foot(prev, dist, labels)

    >>> paths
    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 5]])

    >>> path_dist
    array([11., 10.])

    """
    max_label = max(labels)

    foot_index = np.where(labels == max_label)[0]
    n_feet = len(foot_index)

    paths = np.full((n_feet, max_label + 1), np.nan)
    path_dist = np.full(n_feet, np.nan)

    for i, foot in enumerate(foot_index):

        paths[i, :] = gr.trace_path(prev, foot)
        path_dist[i] = dist[foot]

    return paths.astype(int), path_dist


def get_scores(dist_matrix: ndarray, paths: ndarray, label_adj_list: adj_list, score_func: func_ab) -> ndarray:
    """
    Compute a score matrix from a set of body part positions.

    Compares measured distance between points to the expected distance.
    Only the connections along the set of shortest paths are computed.

    Parameters
    ----------
    dist_matrix : (N, N) ndarray
        Distance matrix for the N position hypotheses.
    paths : (N_paths, N_types) ndarray
        Each row lists the nodes on a shortest path through the body part
        types, i.e., from head to foot.
    label_adj_list : dict
        Adjacency list for the labels.
        label_adj_list[A][B] is the expected distance between
        a point with label A and a point with label B.
    score_func : function
        Function of form f(a, b) -> c.
        Outputs a score given a measured distance and an expected distance.

    Returns
    -------
    score_matrix : (N, N) ndarray
       Array of scores.

    """
    score_matrix = np.zeros(dist_matrix.shape)
    n_paths, n_path_nodes = paths.shape

    for i in range(n_paths):
        for j in range(n_path_nodes):
            for k in range(j, n_path_nodes):

                if k in label_adj_list[j]:
                    # These vertices are connected by a body link and
                    # are in the same shortest path
                    u, v = paths[i, j], paths[i, k]

                    length_expected = label_adj_list[j][k]
                    length_measured = dist_matrix[u, v]

                    score_matrix[u, v] = score_func(length_measured, length_expected)

    # Ensure that all values are finite so the elements can be summed
    score_matrix[~np.isfinite(score_matrix)] = 0

    return score_matrix


def reduce_population(population: ndarray, paths: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Reduce the population of a frame to only the points on the shortest paths.

    Parameters
    ----------
    population : (N, 3) ndarray
        All position hypotheses on a frame.
    paths : ndarray
        One row for each foot position.
        Each row is a shortest path from head to foot.

    Returns
    -------
    pop_reduced : ndarray
        (n_reduced, 3) array of positions.
    paths_reduced : ndarray
        Shortest paths with new values for the reduced population.

    """
    path_nums = np.unique(paths)

    # Population along the shortest paths
    pop_reduced = population[path_nums, :]

    n_pop = len(path_nums)
    mapping = {k: v for k, v in zip(path_nums, range(n_pop))}

    paths_reduced = np.zeros(paths.shape, dtype=int)
    n_paths, n_types = paths.shape

    for i in range(n_paths):
        for j in range(n_types):
            paths_reduced[i, j] = mapping[paths[i, j]]

    return pop_reduced, paths_reduced


def get_path_vectors(paths: ndarray, n_pop: int) -> ndarray:
    """
    Convert the paths to boolean vectors.

    Parameters
    ----------
    paths : ndarray
        One row for each foot position.
        Each row is a shortest path from head to foot.
    n_pop : int
        Total number of positions in the population.

    Returns
    -------
    path_vectors : ndarray
        (n_paths, n_pop) array.
        Each row is a boolean vector.
        Element i is True if position i is in the path.

    """
    n_paths = paths.shape[0]
    path_vectors = np.full((n_paths, n_pop), False)

    all_nums = [i for i in range(n_pop)]
    for i, path in enumerate(paths):
        path_vectors[i, :] = np.in1d(all_nums, path)

    return path_vectors


def in_spheres(within_radius: ndarray, has_sphere: ndarray) -> ndarray:
    """
    Return a boolean vector for the positions in the combined sphere volume.

    Parameters
    ----------
    within_radius : (N, N) ndarray
        Boolean array.
        Element (i, j) is True if i is within a given radius from j.
    has_sphere : (N,) ndarray
        Boolean array.
        Element i is True if position i is the centre of a sphere.

    Returns
    -------
    ndarray
        Boolean array.
        Element i is True if position i is within the combined sphere volume.

    """
    n = len(has_sphere)
    tiled = np.tile(has_sphere, (n, 1))

    # Cast to np.ndarray to satisfy mypy.
    return cast(np.ndarray, np.any(tiled * within_radius, 1))


def select_best_feet(
    dist_matrix: ndarray, score_matrix: ndarray, path_vectors: ndarray, radii: array_like
) -> Tuple[int, int]:
    """
    Select the best two feet from multiple hypotheses.

    Parameters
    ----------
    dist_matrix : (N, N) ndarray
        Distance matrix for N position hypotheses.
    score_matrix : (N, N) ndarray
        Score matrix.
        The scores depend on the expected and actual lengths between positions.
    path_vectors : (N_paths, N) ndarray
        Each row is a boolean vector.
        Element i is True if position i is in the path.
    radii : array_like
        List of radii for the spheres, e.g. [0, 5, 10, 15, 20].

    Returns
    -------
    foot_1, foot_2 : int
        Numbers of the best two feet.

    """
    n_paths = path_vectors.shape[0]

    pairs = [*itertools.combinations(range(n_paths), 2)]
    n_pairs = len(pairs)

    votes, pair_scores = np.zeros(n_pairs), np.zeros(n_pairs)

    for r in radii:

        within_radius = dist_matrix < r

        for i in range(n_pairs):

            a, b = pairs[i]

            has_sphere_1 = path_vectors[a, :]
            has_sphere_2 = path_vectors[b, :]

            # Element i is True if i is the centre of a sphere
            has_sphere = np.logical_or(has_sphere_1, has_sphere_2)

            # For type checker: cast Union[generic, ndarray] to ndarray.
            has_sphere = cast(ndarray, has_sphere)

            # Element i is True if i is inside the combined sphere volume
            inside_spheres = in_spheres(within_radius, has_sphere)

            in_spheres_col = inside_spheres.reshape(-1, 1)
            score_included = in_spheres_col @ in_spheres_col.T
            pair_scores[i] = np.sum(score_matrix[score_included])

        max_score = max(pair_scores)

        # Winning pairs for this radius
        radius_winners = pair_scores == max_score

        # Votes go to the winners
        votes += radius_winners

    winning_pair = int(np.argmax(votes))
    foot_1, foot_2 = pairs[winning_pair]

    return foot_1, foot_2


def foot_to_pop(
    population: ndarray, paths: ndarray, path_dist: ndarray, foot_num_1: int, foot_num_2: int
) -> Tuple[ndarray, ndarray]:
    """
    Return the positions on the shortest paths to the two selected feet.

    For consistency, the two paths receive the same head position,
    which is the head along the minimum shortest path.

    Parameters
    ----------
    population : (N, 3) ndarray
        All position hypotheses on a frame.
    paths : ndarray
        One row for each foot position.
        Each row is a shortest path from head to foot.
    path_dist : ndarray
        Total distance of the path to each foot.
    foot_num_1, foot_num_2 : int
        Numbers of foot 1 and 2 (out of all foot positions)

    Returns
    -------
    pop_1, pop_2 : ndarray
        (n_labels, 3) array of chosen points from the input population.
        One point for each label (i.e., each body part type).

    """
    path_1, path_2 = paths[foot_num_1, :], paths[foot_num_2, :]
    pop_1, pop_2 = population[path_1, :], population[path_2, :]

    # Select the head on the shorter of the two selected shortest paths

    foot_distances = path_dist[[foot_num_1, foot_num_2]]
    head_points = (pop_1[0, :], pop_2[0, :])

    # Wrap in int() to satisfy mypy.
    head_selected = head_points[int(np.argmin(foot_distances))]

    pop_1[0, :], pop_2[0, :] = head_selected, head_selected

    return pop_1, pop_2


def process_frame(
    population: ndarray, labels: ndarray, lengths: ndarray, radii: array_like, cost_func: func_ab, score_func: func_ab
) -> Tuple[ndarray, ndarray]:
    """
    Return chosen body part positions from an input set of position hypotheses.

    Uses a score function to select the best foot positions and return the
    shortest paths to these positions.

    Parameters
    ----------
    population : (N, 3) ndarray
        All position hypotheses on a frame.
    labels : (N,) ndarray
        Array of labels for N positions.
        The labels correspond to body part types (e.g., foot).
    lengths : (N_lengths,) ndarray
        Lengths between adjacent body parts.
    radii : array_like
        List of radii used to select the best feet.
    cost_func : function
        Cost function used to weight the body part graph.
    score_func : function
        Score function used to assign scores to connections between body parts.

    Returns
    -------
    pop_1, pop_2 : ndarray
        (n_labels, 3) array of chosen points from the input population.
        One point for each label (i.e., each body part type).

    """
    dist_matrix = cdist(population, population)

    label_adj_list_types = lengths_to_adj_list(TYPE_CONNECTIONS, lengths)
    label_adj_list_parts = lengths_to_adj_list(PART_CONNECTIONS, lengths)

    # Run shortest path algorithm on the body graph
    prev, dist = pop_shortest_paths(dist_matrix, labels, label_adj_list_types, cost_func)

    # Get shortest path to each foot
    paths, path_dist = paths_to_foot(prev, dist, labels)

    pop_reduced, paths_reduced = reduce_population(population, paths)

    dist_matrix_reduced = cdist(pop_reduced, pop_reduced)
    score_matrix = get_scores(dist_matrix_reduced, paths_reduced, label_adj_list_parts, score_func)

    path_vectors = get_path_vectors(paths_reduced, pop_reduced.shape[0])

    foot_1, foot_2 = select_best_feet(dist_matrix_reduced, score_matrix, path_vectors, radii)

    pop_1, pop_2 = foot_to_pop(population, paths, path_dist, foot_1, foot_2)

    return pop_1, pop_2
