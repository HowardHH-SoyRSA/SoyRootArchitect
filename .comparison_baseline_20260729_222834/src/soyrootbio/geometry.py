from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .runtime import worker_threads

from .types import Normalization


def normalize_unit_box(points: np.ndarray) -> tuple[np.ndarray, Normalization]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    minimum = points.min(axis=0)
    extent = points.max(axis=0) - minimum
    scale = float(np.max(extent))
    if scale <= 0:
        raise ValueError("point cloud has zero bounding-box scale")
    return (points - minimum) / scale, Normalization(minimum=minimum, scale=scale)


def mean_nearest_neighbor_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2, workers=worker_threads())
    return float(np.mean(distances[:, 1]))


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) <= 1:
        return points.copy()
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(segment_lengths.sum())
    if total == 0:
        return points[:1].copy()
    spacing = max(float(spacing), total / 1000.0)
    targets = np.arange(0.0, total, spacing)
    if targets[-1] < total:
        targets = np.append(targets, total)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    output = []
    for target in targets:
        idx = int(np.searchsorted(cumulative, target, side="right") - 1)
        idx = min(idx, len(segment_lengths) - 1)
        denom = segment_lengths[idx]
        alpha = 0.0 if denom == 0 else (target - cumulative[idx]) / denom
        output.append(points[idx] * (1.0 - alpha) + points[idx + 1] * alpha)
    return np.asarray(output)


def tangent_vectors(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    tangents = np.zeros_like(path)
    if len(path) == 1:
        tangents[0] = np.array([0.0, 0.0, 1.0])
        return tangents
    tangents[0] = path[1] - path[0]
    tangents[-1] = path[-1] - path[-2]
    if len(path) > 2:
        tangents[1:-1] = path[2:] - path[:-2]
    norms = np.linalg.norm(tangents, axis=1)
    zero = norms <= 1e-12
    tangents[~zero] /= norms[~zero, None]
    tangents[zero] = np.array([0.0, 0.0, 1.0])
    return tangents


def point_to_polyline_distance(points: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(path) == 0:
        raise ValueError("path cannot be empty")
    if len(path) == 1:
        distances = np.linalg.norm(points - path[0], axis=1)
        return distances, np.zeros(len(points), dtype=int)
    tree = cKDTree(path)
    distances, nearest = tree.query(points, k=1, workers=worker_threads())
    return distances.astype(float), nearest.astype(int)


def nearest_path_tangent(point: np.ndarray, path: np.ndarray, tangents: np.ndarray) -> tuple[np.ndarray, int]:
    tree = cKDTree(path)
    _, idx = tree.query(point, k=1)
    return tangents[int(idx)], int(idx)


def vector_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    """Return the directional angle between two vectors in the range 0--180°."""

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return float("nan")
    cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    """Return the undirected axial angle between two lines in the range 0--90°.

    This compatibility helper is appropriate for a line whose orientation is
    unknown.  Gravity and ordered root vectors must use
    :func:`vector_angle_degrees` so upward and downward directions are not
    collapsed together.
    """

    angle = vector_angle_degrees(a, b)
    return min(angle, 180.0 - angle)

