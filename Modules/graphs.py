
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


def dag_shortest_paths(G, V, source):

	dist = {i: np.inf for i in G}
	prev = {i: np.nan for i in G}
	dist[source] = 0

	for u in V:
	    for v in G[u]:
	        weight = G[u][v]
	        if dist[v] > dist[u] + weight:
	            # Relax the edge
	            dist[v] = dist[u] + weight
	            prev[v] = u

	return prev, dist


def trace_path(prev, source, target):
	u, path = target, [target]
	while u != source:
		u = prev[u]
		path.append(u)
	return path[::-1]  # Reverse the path
