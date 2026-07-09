from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import SpectralClustering

from .geometry import nearest_path_tangent, point_to_polyline_distance, resample_polyline, tangent_vectors
from .primary import cluster_hdbscan
from .types import RootPath


@dataclass
class LateralStart:
    start_id: int
    point: np.ndarray
    primary_point: np.ndarray
    primary_index: int
    member_indices: np.ndarray


def find_lateral_starting_points(
    points: np.ndarray,
    primary_mask: np.ndarray,
    primary_path: np.ndarray,
    closest_fraction: float = 0.03,
    min_cluster_size: int = 8,
) -> list[LateralStart]:
    """Cluster non-primary points closest to the primary root as branch starts.

    The closest_fraction parameter is interpreted as a percentile distance
    threshold, matching the paper-inspired nearest-boundary seed step.
    """
    non_primary = np.flatnonzero(~primary_mask)
    if len(non_primary) == 0:
        return []
    distances, primary_indices = point_to_polyline_distance(points[non_primary], primary_path)
    percentile = float(np.clip(closest_fraction, 0.001, 1.0) * 100.0)
    threshold = float(np.percentile(distances, percentile))
    seed_local = np.flatnonzero(distances <= threshold)
    if len(seed_local) == 0:
        seed_local = np.argsort(distances)[:1]
    seed_indices = non_primary[seed_local]
    if len(seed_indices) < 2:
        return []
    labels = cluster_hdbscan(points[seed_indices], min_cluster_size=min(min_cluster_size, max(2, len(seed_indices) // 2)))
    starts: list[LateralStart] = []
    primary_tree = cKDTree(primary_path)
    for label in sorted(label for label in np.unique(labels) if label >= 0):
        members = seed_indices[labels == label]
        if len(members) == 0:
            continue
        cluster_points = points[members]
        distances_to_primary, primary_matches = primary_tree.query(cluster_points, k=1)
        best_member = int(np.argmin(distances_to_primary))
        start_point = cluster_points[best_member]
        primary_idx = int(primary_matches[best_member])
        starts.append(
            LateralStart(
                start_id=len(starts),
                point=start_point,
                primary_point=primary_path[primary_idx],
                primary_index=primary_idx,
                member_indices=members,
            )
        )
    if not starts and len(seed_indices):
        seed_points = points[seed_indices]
        distances_to_primary, primary_matches = primary_tree.query(seed_points, k=1)
        best_member = int(np.argmin(distances_to_primary))
        primary_idx = int(primary_matches[best_member])
        starts.append(LateralStart(0, seed_points[best_member], primary_path[primary_idx], primary_idx, seed_indices))
    return starts


def grow_lateral_candidates(
    points: np.ndarray,
    starts: list[LateralStart],
    primary_path: np.ndarray,
    primary_mask: np.ndarray,
    d_bar: float,
    step_multipliers: tuple[float, ...] = (2.5, 4.0, 6.0),
    open_angles: tuple[float, ...] = (35.0, 55.0, 75.0),
    max_steps: int = 80,
    search_radius_factor: float = 2.2,
) -> list[RootPath]:
    point_tree = cKDTree(points)
    primary_tangents = tangent_vectors(primary_path)
    non_primary = np.flatnonzero(~primary_mask)
    candidates: list[RootPath] = []
    for start in starts:
        outward = start.point - start.primary_point
        if np.linalg.norm(outward) < 1e-9:
            tangent = primary_tangents[start.primary_index]
            outward = _perpendicular_vector(tangent)
        outward = outward / np.linalg.norm(outward)
        primary_tangent = primary_tangents[start.primary_index]
        for multiplier in step_multipliers:
            step_length = max(multiplier * d_bar, 0.004)
            for open_angle in open_angles:
                path = _grow_one_candidate(
                    points=points,
                    point_tree=point_tree,
                    allowed_indices=set(non_primary.tolist()),
                    start=start,
                    initial_direction=outward,
                    primary_tangent=primary_tangent,
                    step_length=step_length,
                    open_angle=open_angle,
                    max_steps=max_steps,
                    search_radius=search_radius_factor * step_length,
                )
                if len(path.points) >= 3:
                    path.root_id = f"lateral_{start.start_id}_s{multiplier:g}_a{int(open_angle)}"
                    path.score = _path_density_score(points, point_tree, path.points, radius=max(2.0 * d_bar, 0.004))
                    path.start_index = start.start_id
                    candidates.append(path)
    return candidates


def _grow_one_candidate(
    points: np.ndarray,
    point_tree: cKDTree,
    allowed_indices: set[int],
    start: LateralStart,
    initial_direction: np.ndarray,
    primary_tangent: np.ndarray,
    step_length: float,
    open_angle: float,
    max_steps: int,
    search_radius: float,
) -> RootPath:
    nodes = [start.primary_point, start.point]
    current = start.point.copy()
    direction = initial_direction.copy()
    covered: set[int] = set()
    for _ in range(max_steps):
        local = point_tree.query_ball_point(current, r=search_radius, workers=-1)
        local = [idx for idx in local if idx in allowed_indices and idx not in covered]
        if not local:
            break
        vectors = points[local] - current
        distances = np.linalg.norm(vectors, axis=1)
        valid = distances >= 0.35 * step_length
        if not np.any(valid):
            break
        local = np.asarray(local, dtype=int)[valid]
        vectors = vectors[valid]
        distances = distances[valid]
        unit = vectors / distances[:, None]
        turn_cos = unit @ direction
        turn_ok = turn_cos >= np.cos(np.radians(70.0))
        open_angle_to_primary = np.degrees(np.arccos(np.clip(np.abs(unit @ primary_tangent), -1.0, 1.0)))
        open_ok = open_angle_to_primary <= open_angle
        if not np.any(turn_ok & open_ok):
            ok = turn_ok
        else:
            ok = turn_ok & open_ok
        if not np.any(ok):
            break
        local = local[ok]
        unit = unit[ok]
        distances = distances[ok]
        density = np.asarray([len(point_tree.query_ball_point(points[idx], r=0.75 * search_radius)) for idx in local], dtype=float)
        distance_score = -np.abs(distances - step_length) / max(step_length, 1e-12)
        direction_score = unit @ direction
        score = 0.55 * direction_score + 0.30 * _normalize(density) + 0.15 * distance_score
        best_pos = int(np.argmax(score))
        next_idx = int(local[best_pos])
        next_point = points[next_idx]
        new_direction = next_point - current
        new_direction /= max(np.linalg.norm(new_direction), 1e-12)
        direction = 0.65 * direction + 0.35 * new_direction
        direction /= max(np.linalg.norm(direction), 1e-12)
        current = next_point
        nodes.append(current)
        nearby = point_tree.query_ball_point(current, r=0.9 * search_radius, workers=-1)
        covered.update(idx for idx in nearby if idx in allowed_indices)
    return RootPath(root_id="candidate", points=np.asarray(nodes), covered_indices=covered)


def reduce_similar_paths(candidates: list[RootPath], n_clusters: int | None = None) -> list[RootPath]:
    if len(candidates) <= 1:
        return candidates
    n_clusters = n_clusters or max(1, int(np.sqrt(len(candidates))))
    n_clusters = min(n_clusters, len(candidates))
    distance = np.zeros((len(candidates), len(candidates)), dtype=float)
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            value = _path_distance(candidates[i].points, candidates[j].points)
            distance[i, j] = distance[j, i] = value
    sigma = np.median(distance[distance > 0]) if np.any(distance > 0) else 1.0
    affinity = np.exp(-(distance**2) / (2.0 * sigma**2 + 1e-12))
    labels = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="discretize",
        random_state=42,
    ).fit_predict(affinity)
    reduced: list[RootPath] = []
    for label in sorted(np.unique(labels)):
        members = [candidates[i] for i in np.flatnonzero(labels == label)]
        reduced.append(max(members, key=lambda path: (path.score, path.length, len(path.covered_indices))))
    return reduced


def select_non_overlapping_paths(
    candidates: list[RootPath],
    points: np.ndarray,
    d_bar: float,
    max_paths: int | None = None,
    overlap_penalty: float = 1.25,
) -> list[RootPath]:
    if not candidates:
        return []
    tree = cKDTree(points)
    radius = max(2.5 * d_bar, 0.004)
    for candidate in candidates:
        if not candidate.covered_indices:
            covered = set()
            for node in candidate.points:
                covered.update(tree.query_ball_point(node, r=radius, workers=-1))
            candidate.covered_indices = covered
    selected: list[RootPath] = []
    used: set[int] = set()
    pool = sorted(candidates, key=lambda p: (p.score, p.length), reverse=True)
    while pool:
        best_path = None
        best_value = 0.0
        for path in pool:
            covered = path.covered_indices
            overlap = len(covered & used)
            novel = len(covered - used)
            value = novel - overlap_penalty * overlap + 10.0 * path.length
            if value > best_value:
                best_value = value
                best_path = path
        if best_path is None:
            break
        selected.append(best_path)
        used.update(best_path.covered_indices)
        pool = [path for path in pool if path is not best_path]
        if max_paths is not None and len(selected) >= max_paths:
            break
    for idx, path in enumerate(selected, start=1):
        path.root_id = f"lateral_{idx:03d}"
    return selected


def backtrace_to_primary(paths: list[RootPath], primary_path: np.ndarray, primary_points: np.ndarray | None = None) -> list[RootPath]:
    target = primary_points if primary_points is not None and len(primary_points) else primary_path
    tree = cKDTree(target)
    refined: list[RootPath] = []
    for path in paths:
        if len(path.points) < 2:
            refined.append(path)
            continue
        _, idx = tree.query(path.points[1], k=1)
        junction = target[int(idx)]
        new_points = path.points.copy()
        new_points[0] = junction
        if np.linalg.norm(new_points[1] - junction) > np.linalg.norm(new_points[1] - new_points[0]) * 2.5:
            new_points = np.vstack([junction, new_points[1:]])
        path.points = resample_polyline(new_points, spacing=max(path.length / max(len(new_points), 2), 1e-5))
        refined.append(path)
    return refined


def _path_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = resample_polyline(a, spacing=max(np.linalg.norm(a[-1] - a[0]) / 20.0, 1e-4))
    b = resample_polyline(b, spacing=max(np.linalg.norm(b[-1] - b[0]) / 20.0, 1e-4))
    tree_b = cKDTree(b)
    tree_a = cKDTree(a)
    dab = tree_b.query(a, k=1, workers=-1)[0].mean()
    dba = tree_a.query(b, k=1, workers=-1)[0].mean()
    return float((dab + dba) / 2.0)


def _path_density_score(points: np.ndarray, tree: cKDTree, path: np.ndarray, radius: float) -> float:
    covered = set()
    for node in path:
        covered.update(tree.query_ball_point(node, r=radius, workers=-1))
    return float(len(covered) + 20.0 * np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    span = values.max() - values.min()
    if span <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - values.min()) / span


def _perpendicular_vector(vector: np.ndarray) -> np.ndarray:
    vector = vector / max(np.linalg.norm(vector), 1e-12)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, vector)) > 0.8:
        helper = np.array([0.0, 1.0, 0.0])
    perp = helper - np.dot(helper, vector) * vector
    return perp / max(np.linalg.norm(perp), 1e-12)




