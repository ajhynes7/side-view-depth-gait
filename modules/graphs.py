import numpy as np
from modules.general import pairwise


def adj_list_to_matrix(G, n_nodes):
    """
    Convert an adjacency list to an adjacency matrix.

    Parameters
    ----------
    G : dict
        Adjacency list in the form of a dictionary
        G[u][v] is the weight from node u to node v

    n_nodes : int
        Number of nodes in the graph

    Returns
    -------
    M : ndarray
        Adjacency matrix

    Examples
    --------
    >>> G = {2: {1: 10}}

    >>> adj_list_to_matrix(G, 3)
    array([[nan, nan, nan],
           [nan, nan, nan],
           [nan, 10., nan]])
    """
    M = np.full((n_nodes, n_nodes), np.nan)

    for u in G:
        for v in G[u]:
            M[u, v] = G[u][v]

    return M


def adj_matrix_to_list(M):
    """


    Parameters
    ----------


    Returns
    -------

    """
    n_nodes = len(M)
    G = {i: {} for i in range(n_nodes)}

    for u in range(n_nodes):
        for v in range(n_nodes):
            element = M[u, v]

            if ~np.isnan(element):
                G[u][v] = element

    return G


def dag_shortest_paths(G, V, source_nodes):
    """


    Parameters
    ----------


    Returns
    -------

    """
    dist = {i: np.inf for i in G}
    prev = {i: np.nan for i in G}

    for v in source_nodes:
        dist[v] = 0

    for u in V:
        for v in G[u]:
            weight = G[u][v]
            if dist[v] > dist[u] + weight:
                # Relax the edge
                dist[v] = dist[u] + weight
                prev[v] = u

    return prev, dist


def trace_path(prev, last_node):
    """


    Parameters
    ----------


    Returns
    -------

    """
    u, path = last_node, [last_node]

    while ~np.isnan(u):

        u = prev[u]
        path.append(u)

    path = [x for x in path if ~np.isnan(x)]
    return path[::-1]  # Reverse the path


def weight_along_path(G, path):

    total_weight = 0

    for a, b in pairwise(path):
        total_weight += G[a][b]

    return total_weight


if __name__ == "__main__":

    import doctest
    doctest.testmod()

    print(pairwise([1, 2, 3]))
