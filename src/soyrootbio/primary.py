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


def refine_primary_centerline(
    points: np.ndarray,
    primary_mask: np.ndarray,
    primary_path: np.ndarray,
    d_bar: float,
    max_stations: int = 400,
    min_slice_points: int = 12,
) -> np.ndarray:
    """Recenter a coarse surface path using primary-root cross sections."""
    points = np.asarray(points, dtype=float)
    primary_path = np.asarray(primary_path, dtype=float)
    primary_mask = np.asarray(primary_mask, dtype=bool)
    if len(primary_path) < 3 or primary_mask.shape != (len(points),):
        return primary_path.copy()

    primary_points = points[primary_mask]
    if len(primary_points) < max(3 * min_slice_points, 30):
        return primary_path.copy()

    total_length = float(np.linalg.norm(np.diff(primary_path, axis=0), axis=1).sum())
    if total_length <= 0:
        return primary_path.copy()

    station_spacing = max(6.0 * d_bar, total_length / max(3, int(max_stations)), 1e-4)
    stations = resample_polyline(primary_path, station_spacing)
    if len(stations) < 3:
        return primary_path.copy()

    tangents = tangent_vectors(stations)
    tree = cKDTree(primary_points)
    search_radius = max(18.0 * d_bar, 3.0 * station_spacing, 0.008)
    slab_half_thickness = max(2.5 * d_bar, 0.75 * station_spacing)
    centers = np.full_like(stations, np.nan)
    valid = np.zeros(len(stations), dtype=bool)

    for index, (station, tangent) in enumerate(zip(stations, tangents)):
        local_indices = tree.query_ball_point(station, r=search_radius)
        if len(local_indices) < min_slice_points:
            continue
        offsets = primary_points[np.asarray(local_indices, dtype=int)] - station
        axial = offsets @ tangent
        in_slab = np.abs(axial) <= slab_half_thickness
        if int(in_slab.sum()) < min_slice_points:
            continue
        offsets = offsets[in_slab]
        axial = axial[in_slab]
        basis = _plane_basis(tangent)
        radial = offsets @ basis.T
        radial_center = _robust_cross_section_center(radial)
        center_offset = np.median(axial) * tangent + radial_center @ basis
        if np.linalg.norm(center_offset) > 0.9 * search_radius:
            continue
        centers[index] = station + center_offset
        valid[index] = True

    if int(valid.sum()) < max(3, int(np.ceil(0.25 * len(stations)))):
        return primary_path.copy()

    station_indices = np.arange(len(stations))
    valid_indices = station_indices[valid]
    for axis in range(3):
        centers[:, axis] = np.interp(station_indices, valid_indices, centers[valid, axis])

    for _ in range(3):
        smoothed = centers.copy()
        smoothed[1:-1] = (centers[:-2] + 2.0 * centers[1:-1] + centers[2:]) / 4.0
        centers = smoothed

    centers[0] = primary_path[0]
    centers[-1] = primary_path[-1]
    return resample_polyline(centers, spacing=max(2.0 * d_bar, 1e-4))


def _robust_cross_section_center(projected: np.ndarray) -> np.ndarray:
    """Estimate the midpoint of a possibly unevenly sampled 2D boundary."""
    projected = np.asarray(projected, dtype=float)
    median = np.median(projected, axis=0)
    if len(projected) < 8:
        return median
    centered = projected - median
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ axes.T
    lower, upper = np.percentile(rotated, [10.0, 90.0], axis=0)
    return median + ((lower + upper) * 0.5) @ axes


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

