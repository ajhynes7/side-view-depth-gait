"""
Module for estimating the pose of a walking person.

The pose is estimated by selecting body parts from a set of hypotheses.

"""
import itertools

import numpy as np
from scipy.spatial.distance import cdist

import modules.graphs as gr
import modules.iterable_funcs as itf
import modules.point_processing as pp


def only_consecutive_labels(label_adj_list):
    """
    Return a label adjacency list with only consecutive labels.

    For example, if the original adjacency list includes 2->3, 3->4, and 2->4,
    the returned adjacency list will only have 2->3 and 3->4.

    Parameters
    ----------
    label_adj_list : dict
        Adjacency list for the labels.
        label_adj_list[A][B] is the expected distance between
        a point with label A and a point with label B.

    Returns
    -------
    consecutive_adj_list : dict
        Adjacency list for consecutive labels only.
        Every key in original label_adj_list is included.

    Examples
    --------
    >>> label_adj_list = {0: {1: 6}, 1: {2: 1, 3: 3}, 3: {4: 20}, 4: {5: 20}}

    >>> only_consecutive_labels(label_adj_list)
    {0: {1: 6}, 1: {2: 1}, 3: {4: 20}, 4: {5: 20}}

    """
    consecutive_adj_list = {k: {} for k in label_adj_list}

    for key_1 in label_adj_list:
        for key_2 in label_adj_list[key_1]:

            if key_2 - key_1 == 1:

                consecutive_adj_list[key_1] = {
                    key_2: label_adj_list[key_1][key_2]
                }

    return consecutive_adj_list


def estimate_lengths(pop_series, label_series, cost_func, n_frames, eps=0.01):
    """
    Estimate the lengths between consecutive body parts (e.g., calf to foot).

    Starting with an initial estimate of zeros, the lengths are updated by
    iterating over a number of frames and taking the median of the results.

    Parameters
    ----------
    pop_series : Series
        Index of the series is image frame numbers.
        Value at each frame is the population of body part hypotheses.
    label_series : Series
        Index of the series is image frame numbers.
        Value at each frame is the labels of the body part hypotheses.
    cost_func : function
        Cost function for creating the weighted graph.
    n_frames : int
        Number of frames used to estimate the lengths.
    eps : float, optional
        The convergence criterion epsilon (the default is 0.01).
        When all lengths have changed by less
        than epsilon from the previous iteration,
        the iterative process ends.

    Returns
    -------
    lengths : ndarray
        1-D array of estimated lengths between adjacent parts.

    """
    n_lengths = label_series.iloc[0].max()
    lengths = np.zeros(n_lengths)  # Initial estimate of lengths

    frames = pop_series.index.values  # List of image frames with data

    while True:

        prev_lengths = lengths

        length_dict = {i: {i + 1: length} for i, length in enumerate(lengths)}
        length_dict[n_lengths] = {}

        length_array = np.full((n_frames, n_lengths), np.nan)

        for i, f in enumerate(frames[:n_frames]):

            population = pop_series.loc[f]
            labels = label_series.loc[f]

            prev, dist = pop_shortest_paths(population, labels, length_dict,
                                            cost_func)

            label_dict = itf.iterable_to_dict(labels)
            min_path = gr.min_shortest_path(prev, dist, label_dict, n_lengths)

            min_pop = population[min_path]

            length_array[i, :] = [*pp.consecutive_dist(min_pop)]

        lengths = np.median(length_array, axis=0)  # Update lengths

        if np.all(abs(lengths - prev_lengths)) < eps:
            break

    return lengths


def get_population(frame_series, part_labels):
    """
    Return the population of part hypotheses from one image frame.

    Parameters
    ----------
    frame_series : Series
        Index of the series is body parts.
        Values of the series are part hypotheses.
    part_labels : array_like
        Label for each body part in the series.
        e.g. L_FOOT and R_FOOT both have the label 5.

    Returns
    -------
    population : ndarray
        (n, 3) array of n positions.
    labels : ndarray
        (n,) array of labels for n positions.
        The labels correspond to body part types (e.g., foot).
        They are sorted in ascending order.

    Examples
    --------
    >>> import pandas as pd
    >>> head_points = np.array([-45, 66, 238]).reshape(-1, 3)
    >>> foot_points = np.array([[-26., -57, 249], [-74, -58, 260]])

    >>> frame_series = pd.Series({'L_FOOT': foot_points, 'HEAD': head_points})
    >>> part_labels = [5, 0]

    >>> population, labels = get_population(frame_series, part_labels)

    >>> population
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


def lengths_to_adj_list(label_connections, lengths):
    """
    Convert a sequence of lengths between body parts to an adjacency list.

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
    last_part = label_connections.max()
    label_adj_list = {i: {} for i in range(last_part + 1)}

    n_rows = len(label_connections)

    for i in range(n_rows):
        u, v = label_connections[i, 0], label_connections[i, 1]

        label_adj_list[u][v] = sum(lengths[u:v])

    return label_adj_list


def pop_shortest_paths(population, labels, label_adj_list, weight_func):
    """
    Calculate shortest paths on the population of body parts.

    Parameters
    ----------
    population : ndarray
        (n, 3) array of n positions.
    labels : ndarray
        (n,) array of labels for n positions.
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
    pop_graph = gr.points_to_graph(population, labels, label_adj_list,
                                   weight_func)

    # Run shortest path algorithm
    head_nodes = np.where(labels == 0)[0]  # Source nodes
    order = pop_graph.keys()  # Topological ordering of the nodes
    prev, dist = gr.dag_shortest_paths(pop_graph, order, head_nodes)

    return prev, dist


def paths_to_foot(prev, dist, labels):
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


def get_scores(dist_matrix, paths, label_adj_list, score_func):
    """
    Compute a score matrix from a set of body part positions.

    Compares measured distance between points to the expected distance.
    Only the connections along the set of shortest paths are computed.

    Parameters
    ----------
    dist_matrix : ndarray
        (n, n) matrix for the n position hypotheses.
    paths : ndarray
        (n_paths, n_types) array.
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
    score_matrix : ndarray
       (n, n) array of scores.

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

                    score_matrix[u, v] = score_func(length_measured,
                                                    length_expected)

    # Ensure that all values are finite so the elements can be summed
    score_matrix[~np.isfinite(score_matrix)] = 0

    return score_matrix


def reduce_population(population, paths):
    """
    Reduce the population of a frame to only the points on the shortest paths.

    Parameters
    ----------
    population : ndarray
        (n, 3) array of n positions.
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


def get_path_vectors(paths, n_pop):
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
    -----
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


def in_spheres(within_radius, has_sphere):
    """
    Return a boolean vector for the positions in the combined sphere volume.

    Parameters
    ----------
    within_radius : ndarray
        (n, n) boolean array.
        Element (i, j) is true if i is within a given radius from j
    has_sphere : ndarray
        (n,) boolean array.
        Element i is true if position i is the centre of a sphere.

    Returns
    -------
    ndarray
        Boolean array.
        Element i is true if position i is within the combined sphere volume.

    """
    n = len(has_sphere)
    tiled = np.tile(has_sphere, (n, 1))

    return np.any(tiled * within_radius, 1)


def select_best_feet(dist_matrix, score_matrix, path_vectors, radii):
    """
    Select the best two feet from multiple hypotheses.

    Parameters
    ----------
    dist_matrix : ndarray
        (n, n) distance matrix for n position hypotheses.
    score_matrix : ndarray
        (n, n) score matrix.
        The scores depend on the expected and actual lengths between positions.
    path_vectors : ndarray
        (n_paths, n) array.
        Each row is a boolean vector.
        Element i is true if position i is in the path.
    radii : list
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

            # Element i is true if i is the centre of a sphere
            has_sphere = np.logical_or(has_sphere_1, has_sphere_2)

            # Element i is true if i is inside the combined sphere volume
            inside_spheres = in_spheres(within_radius, has_sphere)

            in_spheres_col = inside_spheres.reshape(-1, 1)
            score_included = in_spheres_col @ in_spheres_col.T
            pair_scores[i] = np.sum(score_matrix[score_included])

        max_score = max(pair_scores)

        # Winning pairs for this radius
        radius_winners = pair_scores == max_score

        # Votes go to the winners
        votes += radius_winners

    winning_pair = np.argmax(votes)
    foot_1, foot_2 = pairs[winning_pair]

    return foot_1, foot_2


def foot_to_pop(population, paths, path_dist, foot_num_1, foot_num_2):
    """
    Return the positions comprising the shortest path to each chosen foot.

    For consistency, the two paths receive the same head position,
    which is the head along the minimum shortest path.

    Parameters
    ----------
    population : ndarray
        (n, 3) array of n positions.
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

    # Select the head along the minimum shortest path
    min_path = paths[np.argmin(path_dist), :]
    head_pos = population[min_path[0], :]

    pop_1[0, :], pop_2[0, :] = head_pos, head_pos

    return pop_1, pop_2


def process_frame(population, labels, label_adj_list, radii, cost_func,
                  score_func):
    """
    Return chosen body part positions from an input set of position hypotheses.

    Uses a score function to select the best foot positions and return the
    shortest paths to these positions.

    Parameters
    ----------
    population : ndarray
        (n, 3) array of n positions.
    labels : ndarray
        (n,) array of labels for n positions.
        The labels correspond to body part types (e.g., foot).
    label_adj_list : dict
        Adjacency list for the labels.
        label_adj_list[A][B] is the expected distance between
        a point with label A and a point with label B.
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
    # Define a graph with edges between consecutive parts
    # (e.g. knee to calf, not knee to foot)
    cons_label_adj_list = only_consecutive_labels(label_adj_list)

    # Run shortest path algorithm on the body graph
    prev, dist = pop_shortest_paths(population, labels, cons_label_adj_list,
                                    cost_func)

    # Get shortest path to each foot
    paths, path_dist = paths_to_foot(prev, dist, labels)

    pop_reduced, paths_reduced = reduce_population(population, paths)

    dist_matrix = cdist(pop_reduced, pop_reduced)
    score_matrix = get_scores(dist_matrix, paths_reduced, label_adj_list,
                              score_func)

    path_vectors = get_path_vectors(paths_reduced, pop_reduced.shape[0])

    foot_1, foot_2 = select_best_feet(dist_matrix, score_matrix, path_vectors,
                                      radii)

    pop_1, pop_2 = foot_to_pop(population, paths, path_dist, foot_1, foot_2)

    return pop_1, pop_2
