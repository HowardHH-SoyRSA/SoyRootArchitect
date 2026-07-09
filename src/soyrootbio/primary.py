from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .geometry import resample_polyline, tangent_vectors
from .graph import dijkstra_path_between_points
from .types import RootPath


def cluster_hdbscan(data: np.ndarray, min_cluster_size: int, min_samples: int | None = None) -> np.ndarray:
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError("hdbscan is required for primary and lateral clustering.") from exc
    if len(data) < max(2, min_cluster_size):
        return np.zeros(len(data), dtype=int)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, int(min_cluster_size)),
        min_samples=min_samples,
        allow_single_cluster=True,
    )
    return clusterer.fit_predict(data)


def estimate_primary_path(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    d_bar: float,
    graph_k: int = 14,
) -> RootPath:
    radius = max(4.0 * d_bar, 1e-4)
    path, node_indices = dijkstra_path_between_points(points, start, end, k=graph_k, radius=radius)
    path = resample_polyline(path, spacing=max(2.0 * d_bar, 1e-4))
    return RootPath(root_id="primary", points=path, node_indices=node_indices)


def tangent_plane_primary_segmentation(
    points: np.ndarray,
    primary_path: np.ndarray,
    d_bar: float,
    plane_radius: float | None = None,
    slab_half_thickness: float | None = None,
    min_cluster_size: int = 12,
) -> np.ndarray:
    plane_radius = float(plane_radius or max(8.0 * d_bar, 0.01))
    slab_half_thickness = float(slab_half_thickness or max(2.5 * d_bar, 0.003))
    tree = cKDTree(points)
    tangents = tangent_vectors(primary_path)
    primary_indices: set[int] = set()
    for center, tangent in zip(primary_path, tangents):
        local_idx = tree.query_ball_point(center, r=plane_radius, workers=-1)
        if len(local_idx) < 3:
            continue
        local_idx = np.asarray(local_idx, dtype=int)
        vectors = points[local_idx] - center
        axial = np.abs(vectors @ tangent)
        in_slab = axial <= slab_half_thickness
        if not np.any(in_slab):
            continue
        slab_idx = local_idx[in_slab]
        slab_vectors = points[slab_idx] - center
        basis = _plane_basis(tangent)
        projected = slab_vectors @ basis.T
        labels = cluster_hdbscan(projected, min_cluster_size=min(min_cluster_size, max(2, len(projected) // 2)))
        valid_labels = [label for label in np.unique(labels) if label >= 0]
        if not valid_labels:
            close = np.linalg.norm(projected, axis=1) <= max(3.0 * d_bar, plane_radius * 0.25)
            primary_indices.update(slab_idx[close].tolist())
            continue
        best_label = min(
            valid_labels,
            key=lambda label: float(np.linalg.norm(projected[labels == label], axis=1).mean()),
        )
        primary_indices.update(slab_idx[labels == best_label].tolist())

    distances, _ = cKDTree(primary_path).query(points, k=1, workers=-1)
    primary_indices.update(np.flatnonzero(distances <= max(2.5 * d_bar, 0.004)).tolist())
    mask = np.zeros(len(points), dtype=bool)
    mask[list(primary_indices)] = True
    return mask


def _plane_basis(normal: np.ndarray) -> np.ndarray:
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, normal)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = helper - np.dot(helper, normal) * normal
    u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(normal, u)
    v /= max(np.linalg.norm(v), 1e-12)
    return np.vstack([u, v])

