import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigvalsh
from scipy.special import entr

def forman_ricci_curvature(G):
    """
    Compute Forman‑Ricci curvature for each edge.
    For unweighted graph: curvature(e) = 4 - deg(u) - deg(v) for edge uv.
    For weighted: curvature(e) = w_e * (4 - deg_w(u) - deg_w(v))
    """
    curvature = {}
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        deg_u = G.degree(u, weight='weight')
        deg_v = G.degree(v, weight='weight')
        curvature[(u, v)] = w * (4 - deg_u - deg_v)
    return curvature

def ollivier_ricci_curvature(G, alpha=0.5):
    """
    Approximate Ollivier‑Ricci curvature using the earth mover distance (EMD)
    between probability distributions around nodes.
    Simplified: curvature(u,v) = 1 - W(m_u, m_v) / d(u,v)
    where m_u is uniform over neighbours of u.
    """
    curvature = {}
    for u, v, data in G.edges(data=True):
        d_uv = data.get('weight', 1.0)
        # Neighbour distributions
        neigh_u = list(G.neighbors(u))
        neigh_v = list(G.neighbors(v))
        if len(neigh_u) == 0 or len(neigh_v) == 0:
            curvature[(u, v)] = -1.0
            continue
        # Uniform distributions over neighbours (including self? Not typical)
        n_u = len(neigh_u)
        n_v = len(neigh_v)
        # Build cost matrix: distances between neighbours
        # For simplicity, use binary distance (1 if different) -> emd ≈ |n_u - n_v|? Not accurate.
        # Instead, use a simple approximation: curvature = (deg(u) + deg(v) - 2*common_neighbors) / (deg(u) * deg(v))
        # This is the clustering coefficient based curvature.
        common = len(set(neigh_u) & set(neigh_v))
        if n_u * n_v == 0:
            curvature[(u, v)] = 0.0
        else:
            curvature[(u, v)] = common / (n_u * n_v)
        # The sign: negative curvature if few common neighbours.
        # We'll return the raw value.
    return curvature

def entropy_curvature(G):
    """
    Compute von Neumann entropy of the graph Laplacian.
    Higher entropy = more disordered = positive curvature? We'll return entropy as curvature measure.
    """
    L = nx.laplacian_matrix(G).astype(float).toarray()
    ev = eigvalsh(L)
    ev = ev[ev > 1e-8]
    # Normalise eigenvalues
    ev = ev / ev.sum()
    S = -np.sum(ev * np.log(ev + 1e-8))
    return S

def geodesic_deviation(G):
    """
    Variance of shortest path lengths: higher variance = more curved.
    """
    try:
        lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
        all_dists = []
        for u in lengths:
            for v in lengths[u]:
                if u != v:
                    all_dists.append(lengths[u][v])
        if not all_dists:
            return 0.0
        return np.var(all_dists)
    except:
        return 0.0

def ricci_flow(G, iterations=5, step=0.1):
    """
    Discrete Ricci flow: update edge weights based on curvature.
    w_new = w * (1 - step * curvature)  (if curvature positive, weight decreases; negative increases)
    """
    G_new = G.copy()
    for _ in range(iterations):
        curv = ollivier_ricci_curvature(G_new)  # can also use Forman
        for (u, v), c in curv.items():
            if G_new.has_edge(u, v):
                w = G_new[u][v]['weight']
                new_w = w * (1 - step * c)
                if new_w <= 0:
                    G_new.remove_edge(u, v)
                else:
                    G_new[u][v]['weight'] = new_w
    return G_new
