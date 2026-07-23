from __future__ import annotations

import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra
from scipy.spatial import cKDTree

from .runtime import worker_threads


def build_local_graph(points: np.ndarray, k: int = 12, radius: float | None = None) -> nx.Graph:
    points = np.asarray(points, dtype=float)
    tree = cKDTree(points)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(points)))
    k = max(2, min(int(k), len(points)))
    distances, indices = tree.query(points, k=k + 1, workers=worker_threads())
    for i in range(len(points)):
        for distance, j in zip(distances[i, 1:], indices[i, 1:]):
            if j == i:
                continue
            if radius is not None and distance > radius:
                continue
            graph.add_edge(int(i), int(j), weight=float(distance))
    return graph


def build_sparse_local_graph(
    points: np.ndarray,
    k: int = 12,
    radius: float | None = None,
) -> sparse.csr_matrix:
    """Build a symmetric weighted k-nearest-neighbour graph.

    The original MVP constructed a Python/NetworkX graph with one object per
    point and edge.  That becomes impractical for the supplied 100k--526k
    vertex meshes.  This CSR representation keeps the same Euclidean edge
    weights while allowing SciPy's compiled shortest-path implementation to be
    used for primary-root tracing and candidate scoring.
    """

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if len(points) < 2:
        raise ValueError("at least two points are required to build a graph")
    neighbour_count = max(1, min(int(k), len(points) - 1))
    query_count = neighbour_count + 1
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=query_count, workers=worker_threads())
    rows = np.repeat(np.arange(len(points), dtype=np.int64), neighbour_count)
    cols = np.asarray(indices[:, 1:], dtype=np.int64).reshape(-1)
    weights = np.asarray(distances[:, 1:], dtype=float).reshape(-1)
    valid = np.isfinite(weights) & (cols >= 0) & (cols < len(points)) & (cols != rows)
    if radius is not None:
        valid &= weights <= float(radius)
    directed = sparse.csr_matrix(
        (weights[valid], (rows[valid], cols[valid])),
        shape=(len(points), len(points)),
    )
    graph = directed.maximum(directed.T).tocsr()
    graph.eliminate_zeros()
    return graph


def shortest_path_indices(
    graph: sparse.csr_matrix,
    start_index: int,
    end_index: int,
) -> tuple[np.ndarray, float]:
    """Return one shortest path and its distance from an existing CSR graph."""

    distances, predecessors = sparse_dijkstra(
        graph,
        directed=False,
        indices=int(start_index),
        return_predecessors=True,
    )
    end_index = int(end_index)
    if not np.isfinite(distances[end_index]):
        raise RuntimeError("No connected graph path exists between the requested points.")
    path = [end_index]
    current = end_index
    while current != int(start_index):
        current = int(predecessors[current])
        if current < 0:
            raise RuntimeError("Shortest-path predecessor chain is incomplete.")
        path.append(current)
    path.reverse()
    return np.asarray(path, dtype=int), float(distances[end_index])


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
        graph = build_sparse_local_graph(points, k=current_k, radius=current_radius)
        try:
            node_indices, _ = shortest_path_indices(graph, int(start_idx), int(end_idx))
            return points[node_indices], node_indices
        except RuntimeError as exc:
            last_error = exc
            current_k = min(len(points) - 1, current_k * 2)
            current_radius = None if current_radius is None else current_radius * 1.75
    raise RuntimeError("Could not find a connected Dijkstra path between primary-root endpoints.") from last_error

