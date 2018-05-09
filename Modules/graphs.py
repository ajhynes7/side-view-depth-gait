import numpy as np

def adj_list_to_matrix(G):

    n_nodes = len(G)
    M = np.full((n_nodes, n_nodes), np.nan)

    for u in G:
        for v in G[u]:
            M[u, v] = G[u][v]

    return M


def adj_matrix_to_list(M):

    n_nodes = len(M)
    G = {i: {} for i in range(n_nodes)}

    for u in range(n_nodes):
        for v in range(n_nodes):
            G[u][v] = M[u, v]

    return G


def dag_shortest_paths(G, V, source_nodes):
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
	#u, path = target, [target]
    u, path = last_node, [last_node]

    while ~np.isnan(u):

        u = prev[u]
        path.append(u)

    path = [x for x in path if ~np.isnan(x)]
    return path[::-1]  # Reverse the path
