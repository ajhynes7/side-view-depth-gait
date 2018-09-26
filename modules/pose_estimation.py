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
    path_matrix : ndarray
        One row for each foot position.
        Each row is a shortest path from head to foot.
    path_dist : ndarray
        Total distance of the path to each foot.

    Examples
    --------
    >>> prev = {0: np.nan, 1: 0, 2: 1, 3: 2, 4: 3, 5: 3}
    >>> dist = {0: 0, 1: 0, 2: 20, 3: 5, 4: 11, 5: 10}

    >>> labels = np.array([0, 1, 2, 3, 4, 4])

    >>> path_matrix, path_dist = paths_to_foot(prev, dist, labels)

    >>> path_matrix
    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 5]])

    >>> path_dist
    array([11., 10.])

    """
    max_label = max(labels)

    foot_index = np.where(labels == max_label)[0]
    n_feet = len(foot_index)

    path_matrix = np.full((n_feet, max_label + 1), np.nan)
    path_dist = np.full(n_feet, np.nan)

    for i, foot in enumerate(foot_index):

        path_matrix[i, :] = gr.trace_path(prev, foot)
        path_dist[i] = dist[foot]

    return path_matrix.astype(int), path_dist


def get_score_matrix(population, labels, label_adj_list, score_func):
    """
    Compute a score matrix from a set of body part positions.

    Compares measured distance between points to the expected distances.

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
    score_func : function
        Function of form f(a, b) -> c.
        Outputs a score given a measured distance and an expected distance.

    Returns
    -------
    score_matrix : ndarray
       (n, n) array of scores.

    dist_matrix : ndarray
        (n, n) array of measured distances between the n points.

    """
    # Matrix of measured distances between all n points
    dist_matrix = cdist(population, population)

    # Adjacency list of all n nodes in the graph
    # Edge weights are the expected distances between points
    label_dict = itf.iterable_to_dict(labels)
    expected_adj_list = gr.labelled_nodes_to_graph(label_dict, label_adj_list)

    # Convert adj list to a matrix so it can be compared to the
    # actual distance matrix
    expected_dist_matrix = gr.adj_list_to_matrix(expected_adj_list)

    vectorized_score_func = np.vectorize(score_func)

    # Score is high if measured distance is close to expected distance
    score_matrix = vectorized_score_func(dist_matrix, expected_dist_matrix)
    score_matrix[np.isnan(score_matrix)] = 0

    return score_matrix, dist_matrix


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


def filter_by_path(input_matrix, path_matrix, part_connections):
    """
    Filter values in a matrix using the shortest paths.

    Only the connections along the set of shortest paths are kept.

    Parameters
    ----------
    input_matrix : ndarray
        (n, n) matrix for the n position hypotheses.
    path_matrix : ndarray
        (n_paths, n_types) array.
        Each row lists the nodes on a shortest path through the body part
        types, i.e., from head to foot.
    part_connections : dict
        part_connections[i][j] is the expected value
        between parts of type i to parts of type j.

    Returns
    -------
    filtered_matrix : ndarray
        (n, n) array with a subset of values from the input matrix.
        The filtered values are set to NaN.

    """
    filtered_matrix = np.zeros(input_matrix.shape)
    n_paths, n_path_nodes = path_matrix.shape

    for i in range(n_paths):
        for j in range(n_path_nodes):
            for k in range(n_path_nodes):

                if k in part_connections[j]:
                    # These nodes in the path are connected
                    # in the body part graph
                    a, b = path_matrix[i, j], path_matrix[i, k]
                    filtered_matrix[a, b] = input_matrix[a, b]

    return filtered_matrix


def get_scores(dist_matrix, path_matrix, label_adj_list, score_func):

    score_matrix = np.zeros(dist_matrix.shape)
    n_paths, n_path_nodes = path_matrix.shape

    for i in range(n_paths):
        for j in range(n_path_nodes):
            for k in range(j, n_path_nodes):

                if k in label_adj_list[j]:
                    # These vertices are connected by a body link and
                    # are in the same shortest path
                    u, v = path_matrix[i, j], path_matrix[i, k]

                    length_expected = label_adj_list[j][k]
                    length_measured = dist_matrix[u, v]

                    score_matrix[u, v] = score_func(length_measured,
                                                    length_expected)

    # Ensure that all values are finite so the elements can be summed
    score_matrix[~np.isfinite(score_matrix)] = 0

    return score_matrix


def inside_spheres(dist_matrix, point_nums, r):
    """
    Calculate which of n points are contained inside m spheres.

    Parameters
    ----------
    dist_matrix : ndarray
        (n, n) distance matrix.
        Element (i, j) is distance from point i to point j.
    point_nums : array_like
        (m, ) list of points that are the sphere centres.
        Numbers are between 1 and n.
    r : float
        Radius of spheres.

    Returns
    -------
    in_spheres : ndarray
        (n, ) array of bools.
        Element i is true if point i is in the set of spheres.

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
    Find the positions inside a set of spheres at various radii.

    Parameters
    ----------
    dist_matrix : ndarray
        (n, n) distance matrix for n position hypotheses.
    path_matrix : ndarray
        (n_paths, n_types) array.
        Each row lists the nodes on a shortest path through the body part
        types, i.e., from head to foot.
    radii : list
        List of radii for the spheres, e.g. [0, 5, 10, 15, 20].

    Returns
    -------
    in_spheres_list
        Each element is a list of length n wih boolean values,
        corresponding to a sphere radius.
        If element (i, j) is true, the position j is
        within the sphere space of radius i.

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
    Select the best two feet from multiple hypotheses.

    Parameters
    ----------
    dist_matrix : ndarray
        (n, n) distance matrix for n position hypotheses.
    score_matrix : ndarray
        (n, n) score matrix.
        The scores depend on the expected and actual lengths between positions.
    path_matrix : ndarray
        (n_paths, n_types) array.
        Each row lists the nodes on a shortest path through the body part
        types, i.e., from head to foot.
    radii : list
        List of radii for the spheres, e.g. [0, 5, 10, 15, 20].

    Returns
    -------
    foot_1, foot_2 : int
        Numbers of the best two feet.

    """
    n_paths = len(path_matrix)
    n_radii = len(radii)

    in_spheres_list = inside_radii(dist_matrix, path_matrix, radii)

    # All possible pairs of paths
    combos = [*itertools.combinations(range(n_paths), 2)]

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
        votes += radius_winners

    winning_combo = np.argmax(votes)
    foot_1, foot_2 = combos[winning_combo]

    return foot_1, foot_2


def foot_to_pop(population, path_matrix, path_dist, foot_num_1, foot_num_2):
    """
    Return the positions comprising the shortest path to each chosen foot.

    For consistency, the two paths receive the same head position,
    which is the head along the minimum shortest path.

    Parameters
    ----------
    population : ndarray
        (n, 3) array of n positions.
    path_matrix : ndarray
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
    path_1, path_2 = path_matrix[foot_num_1, :], path_matrix[foot_num_2, :]
    pop_1, pop_2 = population[path_1, :], population[path_2, :]

    # Select the head along the minimum shortest path
    min_path = path_matrix[np.argmin(path_dist), :]
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
    path_matrix, path_dist = paths_to_foot(prev, dist, labels)

    # Matrix of measured distances between all n points
    dist_matrix = cdist(population, population)

    # Compute scores for every edge between body parts
    score_matrix, dist_matrix = get_score_matrix(population, labels,
                                                 label_adj_list, score_func)

    # Keep only scores of edges along the shortest paths to the feet
    filtered_score_matrix = filter_by_path(score_matrix, path_matrix,
                                           label_adj_list)

    filtered_score_matrix2 = get_scores(dist_matrix, path_matrix, label_adj_list,
                            score_func)

    assert np.array_equal(filtered_score_matrix2, filtered_score_matrix)

    foot_1, foot_2 = select_best_feet(dist_matrix, filtered_score_matrix,
                                      path_matrix, radii)

    pop_1, pop_2 = foot_to_pop(population, path_matrix, path_dist, foot_1,
                               foot_2)

    return pop_1, pop_2
