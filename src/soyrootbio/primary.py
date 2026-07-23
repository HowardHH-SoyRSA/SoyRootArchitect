from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra
from scipy.spatial import cKDTree

from .geometry import resample_polyline, tangent_vectors
from .graph import build_sparse_local_graph, dijkstra_path_between_points, shortest_path_indices
from .types import PrimaryCandidate, RootPath
from .runtime import worker_threads


GRAVITY = np.array([0.0, 0.0, -1.0])
COLLAR_COMPLETION_FRACTION = 0.05


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
        # GUI batches are already parallel at the sample level.  HDBSCAN's
        # default nested workers multiply outside that resource allocation
        # when several large samples reach lateral tracing together.
        core_dist_n_jobs=1,
    )
    return clusterer.fit_predict(data)


def estimate_primary_path(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    d_bar: float,
    graph_k: int = 14,
    waypoints: np.ndarray | None = None,
    cooperate: Callable[[], None] | None = None,
) -> RootPath:
    if cooperate is not None:
        cooperate()
    if waypoints is not None and len(waypoints):
        return estimate_primary_path_through_points(
            points,
            start,
            end,
            np.asarray(waypoints, dtype=float),
            d_bar=d_bar,
            graph_k=graph_k,
            cooperate=cooperate,
        )
    radius = max(4.0 * d_bar, 1e-4)
    path, node_indices = dijkstra_path_between_points(points, start, end, k=graph_k, radius=radius)
    if cooperate is not None:
        cooperate()
    path = resample_polyline(path, spacing=max(2.0 * d_bar, 1e-4))
    return RootPath(root_id="primary", points=path, node_indices=node_indices, order=0, parent_id="")


def estimate_primary_path_through_points(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    waypoints: np.ndarray,
    *,
    d_bar: float,
    graph_k: int = 14,
    cooperate: Callable[[], None] | None = None,
) -> RootPath:
    """Trace a primary centerline constrained by manually selected sections."""

    points = np.asarray(points, dtype=float)
    gravity = GRAVITY
    controls = np.vstack([start, np.asarray(waypoints, dtype=float), end])
    # Manual section clicks are allowed in any order.  Root paths are always
    # stored collar-to-tip, so sort intermediate guides along gravity.
    guides = controls[1:-1]
    if len(guides):
        guide_order = np.argsort(guides @ gravity)
        controls = np.vstack([controls[0], guides[guide_order], controls[-1]])
    tree = cKDTree(points)
    control_indices = np.asarray(tree.query(controls, k=1)[1], dtype=int)
    graph = build_sparse_local_graph(points, k=graph_k, radius=max(4.0 * d_bar, 1e-4))
    paths: list[np.ndarray] = []
    try:
        for a, b in zip(control_indices[:-1], control_indices[1:]):
            if cooperate is not None:
                cooperate()
            section, _ = shortest_path_indices(graph, int(a), int(b))
            paths.append(section)
    except RuntimeError:
        graph = build_sparse_local_graph(points, k=max(graph_k * 2, 20), radius=None)
        paths = []
        for a, b in zip(control_indices[:-1], control_indices[1:]):
            if cooperate is not None:
                cooperate()
            section, _ = shortest_path_indices(graph, int(a), int(b))
            paths.append(section)
    node_indices = np.concatenate([part if index == 0 else part[1:] for index, part in enumerate(paths)])
    path = resample_polyline(points[node_indices], spacing=max(2.0 * d_bar, 1e-4))
    return RootPath(
        root_id="primary",
        points=path,
        node_indices=node_indices,
        order=0,
        parent_id="",
        confidence=1.0,
        score_components={"manual_section_constraint": 1.0},
    )


def rank_primary_candidates(
    points: np.ndarray,
    d_bar: float,
    *,
    gravity: np.ndarray = GRAVITY,
    soil_level: float | None = None,
    graph_k: int = 14,
    max_candidates: int = 5,
    collar_seed_count: int = 10,
    cooperate: Callable[[], None] | None = None,
) -> list[PrimaryCandidate]:
    """Rank automatic collar/taproot paths using biological evidence.

    Every requested score term is exposed in ``components``: basal location,
    local radius/thickness continuity, downward extent, path length, and graph
    centrality.  The result is intentionally ranked rather than returning a
    hidden single guess so the GUI and validation files can retain alternatives.
    """

    points = np.asarray(points, dtype=float)
    if cooperate is not None:
        cooperate()
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 20:
        raise ValueError("Automatic primary detection requires at least 20 XYZ points.")
    gravity = np.asarray(gravity, dtype=float)
    gravity /= max(np.linalg.norm(gravity), 1e-12)
    up = -gravity
    height = points @ up
    height_span = max(float(np.ptp(height)), 1e-12)
    height01 = (height - float(height.min())) / height_span
    if soil_level is None:
        basal = height01
        collar_pool = np.flatnonzero(height01 >= np.quantile(height01, 0.90))
    else:
        band = max(0.025 * height_span, 8.0 * d_bar, 1e-4)
        basal = np.exp(-np.abs(height - float(soil_level)) / band)
        collar_pool = np.flatnonzero(np.abs(height - float(soil_level)) <= 2.5 * band)
        if len(collar_pool) < 8:
            collar_pool = np.argsort(np.abs(height - float(soil_level)))[: min(len(points), 64)]

    # Measure a bounded set of basal candidates with geometric cross-sections.
    # Fixed-k covariance confounds tessellation density with root radius; the
    # percentile spans below use a fixed spatial slab and are stable when the
    # same surface is sampled more densely.
    basal_order = collar_pool[np.argsort(basal[collar_pool])[::-1]]
    measured_pool = basal_order[: min(len(basal_order), 256)]
    collar_axes = np.tile(gravity, (len(measured_pool), 1))
    collar_radii = _cross_section_radii(
        points,
        points[measured_pool],
        collar_axes,
        search_radius=0.075,
        slab_half_thickness=max(0.010, 4.0 * d_bar),
    )
    thickness01 = _normalize_feature(collar_radii)
    seed_priority = 0.62 * basal[measured_pool] + 0.38 * thickness01
    ordered_pool = measured_pool[np.argsort(seed_priority)[::-1]]
    seed_indices = _spatially_distinct_indices(
        points,
        ordered_pool,
        count=max(2, int(collar_seed_count)),
        separation=max(10.0 * d_bar, 0.015),
    )
    if len(seed_indices) < 2:
        seed_indices = ordered_pool[: min(len(ordered_pool), max(2, collar_seed_count))]

    tip_pool = np.flatnonzero(height01 <= np.quantile(height01, 0.10))
    if not len(tip_pool):
        tip_pool = np.argsort(height)[: min(32, len(points))]
    radius = max(5.0 * d_bar, 0.006)
    graph = build_sparse_local_graph(points, k=graph_k, radius=radius)
    if cooperate is not None:
        cooperate()
    distances, predecessors = sparse_dijkstra(
        graph,
        directed=False,
        indices=np.asarray(seed_indices, dtype=int),
        return_predecessors=True,
    )
    if distances.ndim == 1:
        distances = distances[None, :]
        predecessors = predecessors[None, :]
    if not np.any(np.isfinite(distances[:, tip_pool])):
        graph = build_sparse_local_graph(points, k=max(20, graph_k * 2), radius=None)
        distances, predecessors = sparse_dijkstra(
            graph,
            directed=False,
            indices=np.asarray(seed_indices, dtype=int),
            return_predecessors=True,
        )
        if distances.ndim == 1:
            distances = distances[None, :]
            predecessors = predecessors[None, :]

    raw: list[dict] = []
    landmark_indices = _spatial_landmark_indices(points, min(96, len(points)))
    for row, start_index in enumerate(np.asarray(seed_indices, dtype=int)):
        if cooperate is not None:
            cooperate()
        tip_distances = distances[row, tip_pool]
        finite = np.isfinite(tip_distances)
        if not np.any(finite):
            continue
        available_tips = tip_pool[finite]
        available_distances = tip_distances[finite]
        downward = (points[available_tips] - points[start_index]) @ gravity
        downward01 = np.clip(downward / height_span, 0.0, 1.5)
        distance01 = available_distances / max(float(np.nanmax(available_distances)), 1e-12)
        end_position = int(np.argmax(0.58 * downward01 + 0.42 * distance01))
        end_index = int(available_tips[end_position])
        node_indices = _predecessor_path(predecessors[row], int(start_index), end_index)
        if len(node_indices) < 3:
            continue
        path = points[node_indices]
        path_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        downward_extent = float(max(0.0, np.dot(path[-1] - path[0], gravity)))
        sampled_positions = np.linspace(0, len(path) - 1, min(20, len(path)), dtype=int)
        sampled_path = path[sampled_positions]
        sampled_tangents = tangent_vectors(path)[sampled_positions]
        path_thickness = _cross_section_radii(
            points,
            sampled_path,
            sampled_tangents,
            search_radius=0.065,
            slab_half_thickness=max(0.009, 4.0 * d_bar),
        )
        positive_thickness = path_thickness[path_thickness > 0]
        if len(positive_thickness) > 1:
            cv = float(np.std(positive_thickness) / max(np.mean(positive_thickness), 1e-12))
            jumps = np.abs(np.diff(positive_thickness)) / np.maximum(positive_thickness[:-1], 1e-12)
            continuity = float(np.exp(-cv) * np.exp(-np.percentile(jumps, 80)))
        else:
            continuity = 0.0
        path_sources = node_indices[
            np.linspace(0, len(node_indices) - 1, min(24, len(node_indices)), dtype=int)
        ]
        distance_to_path = sparse_dijkstra(
            graph,
            directed=False,
            indices=np.unique(path_sources),
            min_only=True,
        )
        landmark_distances = np.asarray(distance_to_path)[landmark_indices]
        finite_landmarks = landmark_distances[np.isfinite(landmark_distances)]
        reach = len(finite_landmarks) / max(1, len(landmark_indices))
        closeness = 0.0 if not len(finite_landmarks) else 1.0 / max(float(np.mean(finite_landmarks)), 1e-12)
        raw.append(
            {
                "start_index": int(start_index),
                "end_index": end_index,
                "node_indices": node_indices,
                "path": path,
                "basal_location": float(basal[start_index]),
                "local_radius": float(np.median(positive_thickness[: max(1, len(positive_thickness) // 3)]))
                if len(positive_thickness)
                else 0.0,
                "radius_continuity": continuity,
                "downward_extent": downward_extent,
                "path_length": path_length,
                "centrality_raw": float(reach * closeness),
            }
        )
    if not raw:
        raise RuntimeError("Automatic primary scorer could not find a connected collar-to-tip candidate.")

    for key in ("local_radius", "downward_extent", "path_length", "centrality_raw"):
        scaled = _normalize_feature(np.asarray([item[key] for item in raw], dtype=float))
        for item, value in zip(raw, scaled):
            item[f"{key}_score"] = float(value)
    for item in raw:
        thickness_continuity = 0.45 * item["local_radius_score"] + 0.55 * item["radius_continuity"]
        item["thickness_continuity"] = float(thickness_continuity)
        item["score"] = float(
            0.20 * item["basal_location"]
            + 0.22 * thickness_continuity
            + 0.24 * item["downward_extent_score"]
            + 0.19 * item["path_length_score"]
            + 0.15 * item["centrality_raw_score"]
        )
    raw.sort(key=lambda item: item["score"], reverse=True)
    results: list[PrimaryCandidate] = []
    previous_confidence = 1.0
    for rank, item in enumerate(raw[: max(1, int(max_candidates))], start=1):
        next_score = raw[rank]["score"] if rank < len(raw) else 0.0
        margin = max(0.0, item["score"] - next_score)
        confidence = float(
            np.clip(
                0.75 * item["score"]
                + 0.25 * min(1.0, 4.0 * margin)
                - 0.04 * (rank - 1),
                0.0,
                1.0,
            )
        )
        if rank > 1:
            confidence = min(confidence, max(0.0, previous_confidence - 0.02))
        previous_confidence = confidence
        flags: list[str] = []
        if item["radius_continuity"] < 0.35:
            flags.append("abrupt_thickness_change")
        if item["downward_extent_score"] < 0.45:
            flags.append("limited_downward_extent")
        if confidence < 0.55:
            flags.append("low_primary_confidence")
        components = {
            "basal_location": item["basal_location"],
            "local_radius_thickness_continuity": item["thickness_continuity"],
            "downward_extent": item["downward_extent_score"],
            "path_length": item["path_length_score"],
            "graph_centrality": item["centrality_raw_score"],
        }
        results.append(
            PrimaryCandidate(
                rank=rank,
                start_index=item["start_index"],
                end_index=item["end_index"],
                start=points[item["start_index"]].copy(),
                end=points[item["end_index"]].copy(),
                path=resample_polyline(item["path"], spacing=max(2.0 * d_bar, 1e-4)),
                score=item["score"],
                confidence=confidence,
                components=components,
                qc_flags=flags,
            )
        )
    return results


def _predecessor_path(predecessors: np.ndarray, start: int, end: int) -> np.ndarray:
    path = [int(end)]
    current = int(end)
    while current != int(start):
        current = int(predecessors[current])
        if current < 0:
            return np.empty(0, dtype=int)
        path.append(current)
    path.reverse()
    return np.asarray(path, dtype=int)


def _local_thickness(points: np.ndarray, indices: np.ndarray, neighbour_count: int = 32) -> np.ndarray:
    """Legacy density-robust local surface scale (not used as root radius)."""

    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return np.empty(0, dtype=float)
    tree = cKDTree(points)
    result = np.zeros(len(indices), dtype=float)
    spatial_radius = max(0.025 * float(np.max(np.ptp(points, axis=0))), 1e-6)
    for row, index in enumerate(indices):
        member_indices = tree.query_ball_point(points[int(index)], r=spatial_radius)
        if len(member_indices) < 4:
            continue
        local = points[np.asarray(member_indices, dtype=int)]
        centered = local - np.mean(local, axis=0)
        covariance = centered.T @ centered / max(1, len(local) - 1)
        eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
        result[row] = float(np.sqrt(eigenvalues[1] + eigenvalues[2])) if len(eigenvalues) >= 3 else 0.0
    return result


def _cross_section_radii(
    points: np.ndarray,
    centers: np.ndarray,
    axes: np.ndarray,
    *,
    search_radius: float,
    slab_half_thickness: float,
) -> np.ndarray:
    """Estimate radii from fixed-spatial tangent-normal percentile spans."""

    points = np.asarray(points, dtype=float)
    centers = np.asarray(centers, dtype=float)
    axes = np.asarray(axes, dtype=float)
    if len(centers) == 0:
        return np.empty(0, dtype=float)
    tree = cKDTree(points)
    result = np.zeros(len(centers), dtype=float)
    for row, (center, axis) in enumerate(zip(centers, axes)):
        member_indices = tree.query_ball_point(
            center,
            r=float(search_radius),
            workers=worker_threads(),
        )
        if len(member_indices) < 8:
            continue
        local = points[np.asarray(member_indices, dtype=int)] - center
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        local = local[np.abs(local @ axis) <= float(slab_half_thickness)]
        if len(local) < 8:
            continue
        projected = local @ _plane_basis(axis).T
        lower, upper = np.percentile(projected, [5.0, 95.0], axis=0)
        widths = np.maximum(upper - lower, 0.0)
        # Half the mean of the two robust diameters.
        result[row] = float(0.25 * np.sum(widths))
    return result


def _spatial_landmark_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Return deterministic farthest-point landmarks independent of file order."""

    points = np.asarray(points, dtype=float)
    count = max(1, min(int(count), len(points)))
    selected = list(
        np.unique(np.concatenate([np.argmin(points, axis=0), np.argmax(points, axis=0)])).astype(int)
    )
    selected = selected[:count]
    squared_distance = np.full(len(points), np.inf, dtype=float)
    for index in selected:
        squared_distance = np.minimum(
            squared_distance,
            np.sum((points - points[index]) ** 2, axis=1),
        )
    while len(selected) < count:
        index = int(np.argmax(squared_distance))
        if squared_distance[index] <= 1e-20:
            break
        selected.append(index)
        squared_distance = np.minimum(
            squared_distance,
            np.sum((points - points[index]) ** 2, axis=1),
        )
    return np.asarray(selected, dtype=int)


def _normalize_feature(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if not np.isfinite(low) or not np.isfinite(high) or high - low <= 1e-12:
        return np.ones_like(values, dtype=float) * 0.5
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _spatially_distinct_indices(
    points: np.ndarray,
    ordered_indices: np.ndarray,
    *,
    count: int,
    separation: float,
) -> np.ndarray:
    selected: list[int] = []
    for index in np.asarray(ordered_indices, dtype=int):
        if not selected or np.min(np.linalg.norm(points[np.asarray(selected)] - points[index], axis=1)) >= separation:
            selected.append(int(index))
        if len(selected) >= count:
            break
    return np.asarray(selected, dtype=int)


def tangent_plane_primary_segmentation(
    points: np.ndarray,
    primary_path: np.ndarray,
    d_bar: float,
    plane_radius: float | None = None,
    slab_half_thickness: float | None = None,
    min_cluster_size: int = 12,
    complete_cross_section: bool = False,
    cooperate: Callable[[], None] | None = None,
) -> np.ndarray:
    plane_radius = float(plane_radius or max(8.0 * d_bar, 0.01))
    slab_half_thickness = float(slab_half_thickness or max(2.5 * d_bar, 0.003))
    tree = cKDTree(points)
    tangents = tangent_vectors(primary_path)
    primary_indices: set[int] = set()
    collar_station_count = max(
        1,
        int(np.ceil(COLLAR_COMPLETION_FRACTION * len(primary_path))),
    )
    for station_index, (center, tangent) in enumerate(zip(primary_path, tangents)):
        if cooperate is not None:
            cooperate()
        collar_completion = complete_cross_section and station_index < collar_station_count
        # The first refined path can still be nearer one wall than the other.
        # Widen only the collar completion query so the opposite wall is
        # visible; the ordinary radius remains unchanged along the rest of the
        # primary where a broader tube could absorb adjacent laterals.
        local_plane_radius = plane_radius * (1.5 if collar_completion else 1.0)
        local_idx = tree.query_ball_point(
            center,
            r=local_plane_radius,
            workers=worker_threads(),
        )
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
            close = np.linalg.norm(projected, axis=1) <= max(
                3.0 * d_bar,
                local_plane_radius * 0.25,
            )
            primary_indices.update(slab_idx[close].tolist())
            continue
        best_label = min(
            valid_labels,
            key=lambda label: float(np.linalg.norm(projected[labels == label], axis=1).mean()),
        )
        selected = labels == best_label
        if collar_completion:
            radial = np.linalg.norm(projected, axis=1)
            selected_radial = radial[selected]
            # HDBSCAN can split a sampled cylindrical ring into two or more
            # arcs.  Once the path has been recentered, all arcs at the same
            # local radius belong to the primary tube; only radially escaping
            # surfaces should remain available to lateral tracing.
            radial_margin = max(2.5 * d_bar, 0.0015)
            outer_radius = min(
                local_plane_radius,
                float(np.quantile(selected_radial, 0.90)) + radial_margin,
            )
            selected = radial <= outer_radius
        primary_indices.update(slab_idx[selected].tolist())

    distances, _ = cKDTree(primary_path).query(points, k=1, workers=worker_threads())
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
    fit_circular_cross_sections: bool = False,
    cooperate: Callable[[], None] | None = None,
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
        if cooperate is not None:
            cooperate()
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
        radial_center = _robust_cross_section_center(
            radial,
            fit_circle=fit_circular_cross_sections,
        )
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

    # The selected collar endpoint is a surface click, not a biological
    # centreline point.  Pinning the first refined station back to that click
    # keeps the path on one wall of the collar and makes cross-section
    # segmentation return an open half-cylinder.  Preserve the measured
    # cross-section centre at the collar; the original click is still retained
    # separately by the pipeline as the above-base exclusion plane.
    centers[-1] = primary_path[-1]
    return resample_polyline(centers, spacing=max(2.0 * d_bar, 1e-4))


def _robust_cross_section_center(
    projected: np.ndarray,
    *,
    fit_circle: bool = False,
) -> np.ndarray:
    """Estimate the midpoint of a possibly unevenly sampled 2D boundary."""
    projected = np.asarray(projected, dtype=float)
    median = np.median(projected, axis=0)
    if len(projected) < 8:
        return median
    centered = projected - median
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ axes.T
    lower, upper = np.percentile(rotated, [10.0, 90.0], axis=0)
    fallback = median + ((lower + upper) * 0.5) @ axes

    if not fit_circle:
        return fallback

    # A surface-derived path often exposes only one arc of the primary cross
    # section.  The bounding-box midpoint of that arc remains on the same wall;
    # a robust circle fit recovers the tube axis from its curvature.  Iterative
    # residual trimming prevents nearby lateral surfaces from pulling the fit.
    retained = projected.copy()
    circle_center: np.ndarray | None = None
    for _ in range(3):
        design = np.column_stack(
            [2.0 * retained[:, 0], 2.0 * retained[:, 1], np.ones(len(retained))]
        )
        target = np.sum(retained**2, axis=1)
        if np.linalg.matrix_rank(design) < 3:
            return fallback
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        circle_center = np.asarray(coefficients[:2], dtype=float)
        radii = np.linalg.norm(retained - circle_center, axis=1)
        residual = np.abs(radii - np.median(radii))
        cutoff = float(np.quantile(residual, 0.80))
        keep = residual <= max(cutoff, 1e-12)
        if int(np.count_nonzero(keep)) < 8 or np.all(keep):
            break
        retained = retained[keep]

    if circle_center is None or not np.all(np.isfinite(circle_center)):
        return fallback
    span = max(float(np.linalg.norm(np.ptp(projected, axis=0))), 1e-12)
    if np.linalg.norm(circle_center - fallback) > 2.0 * span:
        return fallback
    return circle_center


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
