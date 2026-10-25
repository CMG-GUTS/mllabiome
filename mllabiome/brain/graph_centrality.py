from __future__ import annotations

from math import ceil

import networkx as nx
import numpy as np

from .handcrafted import _check_ts


def graph_centrality(time_series: np.ndarray, *, density: float = 0.15) -> np.ndarray:
    if not np.isfinite(density) or not 0 < density <= 1:
        raise ValueError("Graph density must be finite and in (0, 1].")
    x = _check_ts(time_series, tr=1.0)
    n = x.shape[1]
    target = ceil(density * n * (n - 1) / 2)
    if target < n - 1:
        raise ValueError(
            f"Graph density {density} is too low for a connected {n}-node graph."
        )
    corr = np.corrcoef(x, rowvar=False)
    if not np.isfinite(corr).all():
        raise ValueError("Non-finite correlation matrix.")
    positive = np.maximum(corr, 0.0)
    np.fill_diagonal(positive, 0.0)
    positive_graph = nx.from_numpy_array(positive)
    if not nx.is_connected(positive_graph):
        raise ValueError(
            "Positive correlation graph is disconnected; centrality is undefined under this protocol."
        )
    if positive_graph.number_of_edges() < target:
        raise ValueError("Insufficient positive edges for the requested graph density.")
    tree = nx.maximum_spanning_tree(positive_graph, weight="weight")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(tree.edges(data=True))
    i, j = np.triu_indices(n, k=1)
    order = np.lexsort((j, i, -positive[i, j]))
    for idx in order:
        if graph.number_of_edges() >= target:
            break
        a, b = int(i[idx]), int(j[idx])
        weight = float(positive[a, b])
        if weight <= 0:
            break
        if not graph.has_edge(a, b):
            graph.add_edge(a, b, weight=weight)
    if graph.number_of_edges() != target or not nx.is_connected(graph):
        raise RuntimeError(
            "Unable to construct the requested connected correlation graph."
        )
    for _, _, attributes in graph.edges(data=True):
        attributes["distance"] = 1.0 / attributes["weight"]
    strength = np.array(
        [graph.degree(i, weight="weight") for i in range(n)], dtype=float
    )
    betweenness_map = nx.betweenness_centrality(
        graph, normalized=True, weight="distance"
    )
    betweenness = np.array([betweenness_map[i] for i in range(n)], dtype=float)
    eigenvector_map = nx.eigenvector_centrality_numpy(
        graph, weight="weight", max_iter=1000, tol=1e-8
    )
    eigenvector = np.array([eigenvector_map[i] for i in range(n)], dtype=float)
    features = np.concatenate((strength, betweenness, eigenvector))
    if (
        features.shape != (3 * n,)
        or not np.isfinite(features).all()
        or np.min(features) < -1e-10
    ):
        raise ValueError("Invalid graph centrality features.")
    return np.maximum(features, 0.0)
