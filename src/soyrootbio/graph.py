from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def build_local_graph(points: np.ndarray, k: int = 12, radius: float | None = None) -> nx.Graph:
    points = np.asarray(points, dtype=float)
    tree = cKDTree(points)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(points)))
    k = max(2, min(int(k), len(points)))
    distances, indices = tree.query(points, k=k + 1, workers=-1)
    for i in range(len(points)):
        for distance, j in zip(distances[i, 1:], indices[i, 1:]):
            if j == i:
                continue
            if radius is not None and distance > radius:
                continue
            graph.add_edge(int(i), int(j), weight=float(distance))
    return graph


def dijkstra_path_between_points(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    k: int = 14,
    radius: float | None = None,
    max_retries: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(points)
    _, start_idx = tree.query(start, k=1)
    _, end_idx = tree.query(end, k=1)
    current_k = k
    current_radius = radius
    last_error: Exception | None = None
    for _ in range(max_retries):
        graph = build_local_graph(points, k=current_k, radius=current_radius)
        try:
            node_indices = nx.dijkstra_path(graph, int(start_idx), int(end_idx), weight="weight")
            node_indices = np.asarray(node_indices, dtype=int)
            return points[node_indices], node_indices
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            last_error = exc
            current_k = min(len(points) - 1, current_k * 2)
            current_radius = None if current_radius is None else current_radius * 1.75
    raise RuntimeError("Could not find a connected Dijkstra path between primary-root endpoints.") from last_error

