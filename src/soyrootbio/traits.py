from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geometry import angle_degrees, path_length, tangent_vectors
from .types import Normalization, RootPath


Z_AXIS = np.array([0.0, 0.0, 1.0])


def compute_traits(
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    points: np.ndarray,
    primary_mask: np.ndarray,
    lateral_labels: np.ndarray,
    normalization: Normalization,
    lateral_start_count: int = 0,
) -> pd.DataFrame:
    """Compute MVP root traits in source units, optionally scaled by unit_scale.

    For every non-primary skeleton, `angle_deg` is the base angle against its
    parent skeleton. `tip_angle_parent_deg` is the angle between the lateral tip
    tangent and the nearest parent tangent; `tip_angle_z_deg` is the angle between
    the lateral tip tangent and the global z-axis.
    """
    primary_original = normalization.inverse_points(primary_path)
    records = [
        {
            "root_id": "primary",
            "parent_id": "",
            "root_order": 0,
            "length": path_length(primary_original) * normalization.unit_scale,
            "angle_deg": np.nan,
            "tip_angle_parent_deg": np.nan,
            "tip_angle_primary_deg": np.nan,
            "tip_angle_z_deg": np.nan,
            "lateral_start_count": int(lateral_start_count),
            "selected_lateral_count": int(len(lateral_paths)),
            "point_count": int(primary_mask.sum()),
            "mean_radius": _estimate_radius(points[primary_mask], primary_path, normalization),
            "surface_area": np.nan,
            "volume": np.nan,
        }
    ]
    for lateral_idx, lateral in enumerate(lateral_paths, start=1):
        parent_path = lateral.parent_points if lateral.parent_points is not None and len(lateral.parent_points) else primary_path
        parent_tangents = tangent_vectors(parent_path)
        parent_tree = cKDTree(parent_path)
        primary_tangents = tangent_vectors(primary_path)
        primary_tree = cKDTree(primary_path)

        original = normalization.inverse_points(lateral.points)
        label_mask = lateral_labels == lateral_idx
        radius = _estimate_radius(points[label_mask], lateral.points, normalization)
        _, base_parent_idx = parent_tree.query(lateral.points[0], k=1)
        base_vector = lateral.points[min(2, len(lateral.points) - 1)] - lateral.points[0]
        base_angle = angle_degrees(base_vector, parent_tangents[int(base_parent_idx)])
        tip_vector = _tip_vector(lateral.points)
        _, tip_parent_idx = parent_tree.query(lateral.points[-1], k=1)
        tip_parent_angle = angle_degrees(tip_vector, parent_tangents[int(tip_parent_idx)])
        _, tip_primary_idx = primary_tree.query(lateral.points[-1], k=1)
        tip_primary_angle = angle_degrees(tip_vector, primary_tangents[int(tip_primary_idx)])
        tip_z_angle = angle_degrees(tip_vector, Z_AXIS)
        length = path_length(original) * normalization.unit_scale
        records.append(
            {
                "root_id": lateral.root_id,
                "parent_id": lateral.parent_id,
                "root_order": int(lateral.order),
                "length": length,
                "angle_deg": base_angle,
                "tip_angle_parent_deg": tip_parent_angle,
                "tip_angle_primary_deg": tip_primary_angle,
                "tip_angle_z_deg": tip_z_angle,
                "lateral_start_count": np.nan,
                "selected_lateral_count": np.nan,
                "point_count": int(label_mask.sum()),
                "mean_radius": radius,
                "surface_area": float(2.0 * np.pi * radius * length) if radius > 0 else np.nan,
                "volume": float(np.pi * radius**2 * length) if radius > 0 else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def _tip_vector(path: np.ndarray) -> np.ndarray:
    if len(path) < 2:
        return np.array([0.0, 0.0, 1.0])
    span = min(4, len(path) - 1)
    vector = path[-1] - path[-1 - span]
    if np.linalg.norm(vector) <= 1e-12:
        vector = path[-1] - path[-2]
    return vector


def _estimate_radius(points: np.ndarray, path: np.ndarray, normalization: Normalization) -> float:
    if len(points) == 0 or len(path) == 0:
        return 0.0
    tree = cKDTree(path)
    distances, _ = tree.query(points, k=1, workers=-1)
    if len(distances) == 0:
        return 0.0
    return normalization.inverse_length(float(np.percentile(distances, 65)))
