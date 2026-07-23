from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geometry import angle_degrees, path_length, vector_angle_degrees
from .types import Normalization, RootPath
from .runtime import worker_threads


GRAVITY = np.array([0.0, 0.0, -1.0])


def compute_traits(
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    points: np.ndarray,
    primary_mask: np.ndarray,
    lateral_labels: np.ndarray,
    normalization: Normalization,
    lateral_start_count: int = 0,
    *,
    gravity: np.ndarray = GRAVITY,
    full_points: np.ndarray | None = None,
    triangles: np.ndarray | None = None,
    full_root_labels: np.ndarray | None = None,
    mesh_metadata: dict | None = None,
    primary_confidence: float = 1.0,
    primary_qc_flags: list[str] | None = None,
    tip_vector_window: float = 2.0,
) -> pd.DataFrame:
    """Measure per-root geometry in source mesh units and requested angles.

    Ordered paths are stored collar/insertion-to-tip.  The three user-requested
    directional angles are therefore unambiguous and remain in the full
    0--180° range:

    * ``tip_gravity_angle_deg``: lateral tip tangent versus ``(0, 0, -1)``;
    * ``tip_start_gravity_angle_deg``: start-to-tip vector versus gravity;
    * ``tip_primary_angle_deg``: lateral tip tangent versus the ordered primary
      tangent at the lateral insertion location.
    """

    gravity = np.asarray(gravity, dtype=float)
    gravity /= max(np.linalg.norm(gravity), 1e-12)
    mesh_areas = _partition_mesh_surface_areas(
        full_points,
        triangles,
        full_root_labels,
    )
    primary_original = normalization.inverse_points(primary_path)
    primary_radii = _radius_profile(points[primary_mask], primary_path, normalization)
    primary_length = path_length(primary_original)
    primary_frustum_area, primary_frustum_volume = _frustum_measurements(
        primary_original,
        primary_radii,
    )
    primary_chord = _chord_length(primary_original)
    records = [
        _base_record(
            root_id="primary",
            parent_id="",
            order=0,
            length=primary_length,
            chord=primary_chord,
            radii=primary_radii,
            point_count=int(primary_mask.sum()),
            confidence=float(primary_confidence),
            qc_flags=list(primary_qc_flags or []),
            surface_area=mesh_areas.get(0, primary_frustum_area),
            surface_method="partitioned_mesh_triangles" if 0 in mesh_areas else "centerline_frustum_estimate",
            volume=primary_frustum_volume,
            volume_method="centerline_frustum_estimate",
        )
    ]
    records[0].update(
        {
            "lateral_start_count": int(lateral_start_count),
            "selected_lateral_count": int(len(lateral_paths)),
            "angle_deg": np.nan,
            "base_parent_angle_deg": np.nan,
            "tip_angle_parent_deg": np.nan,
            "tip_angle_primary_deg": np.nan,
            "tip_angle_z_deg": np.nan,
            "tip_gravity_angle_deg": np.nan,
            "tip_start_gravity_angle_deg": np.nan,
            "tip_primary_angle_deg": np.nan,
            **_point_columns(
                "root_start",
                primary_original[0],
            ),
            **_point_columns(
                "root_tip",
                primary_original[-1],
            ),
            "gravity_dx": float(gravity[0]),
            "gravity_dy": float(gravity[1]),
            "gravity_dz": float(gravity[2]),
        }
    )

    primary_tree = cKDTree(primary_path)
    for lateral_index, lateral in enumerate(lateral_paths, start=1):
        parent_path = lateral.parent_points if lateral.parent_points is not None and len(lateral.parent_points) else primary_path
        parent_tree = cKDTree(parent_path)
        original = normalization.inverse_points(lateral.points)
        parent_original = normalization.inverse_points(parent_path)
        label_mask = lateral_labels == lateral_index
        radii = _radius_profile(points[label_mask], lateral.points, normalization)
        length = path_length(original)
        chord = _chord_length(original)
        frustum_area, frustum_volume = _frustum_measurements(original, radii)

        _, parent_index = parent_tree.query(lateral.points[0], k=1)
        _, primary_index = primary_tree.query(lateral.points[0], k=1)
        _, _, base_vector, _ = _arc_window_vector(
            original,
            anchor_index=0,
            requested_window=tip_vector_window,
            from_start=True,
        )
        tip_start_point, tip_point, tip_vector, actual_tip_window = _arc_window_vector(
            original,
            anchor_index=len(original) - 1,
            requested_window=tip_vector_window,
            from_start=False,
        )
        root_start = original[0]
        root_tip = original[-1]
        tip_start_vector = root_tip - root_start
        _, _, parent_vector, _ = _arc_window_vector(
            parent_original,
            anchor_index=int(parent_index),
            requested_window=tip_vector_window,
        )
        primary_vector_start, primary_vector_end, primary_vector, _ = _arc_window_vector(
            primary_original,
            anchor_index=int(primary_index),
            requested_window=tip_vector_window,
        )
        base_parent_angle = angle_degrees(base_vector, parent_vector)
        tip_parent_angle = vector_angle_degrees(tip_vector, parent_vector)
        tip_primary_angle = vector_angle_degrees(tip_vector, primary_vector)
        tip_gravity_angle = vector_angle_degrees(tip_vector, gravity)
        tip_start_gravity_angle = vector_angle_degrees(tip_start_vector, gravity)

        qc_flags = list(lateral.qc_flags)
        if np.linalg.norm(tip_vector) <= 1e-12:
            qc_flags.append("undefined_tip_vector")
        if np.linalg.norm(tip_start_vector) <= 1e-12:
            qc_flags.append("undefined_tip_start_vector")
        if np.linalg.norm(primary_vector) <= 1e-12:
            qc_flags.append("undefined_primary_vector")
        qc_flags = list(dict.fromkeys(qc_flags))
        record = _base_record(
            root_id=lateral.root_id,
            parent_id=lateral.parent_id,
            order=int(lateral.order),
            length=length,
            chord=chord,
            radii=radii,
            point_count=int(label_mask.sum()),
            confidence=float(lateral.confidence),
            qc_flags=qc_flags,
            surface_area=mesh_areas.get(lateral_index, frustum_area),
            surface_method="partitioned_mesh_triangles" if lateral_index in mesh_areas else "centerline_frustum_estimate",
            volume=frustum_volume,
            volume_method="centerline_frustum_estimate",
        )
        record.update(
            {
                "angle_deg": base_parent_angle,
                "base_parent_angle_deg": base_parent_angle,
                "tip_angle_parent_deg": tip_parent_angle,
                "tip_angle_primary_deg": tip_primary_angle,
                # Backwards-compatible name now follows the requested gravity
                # definition rather than the old +Z/acute-axis calculation.
                "tip_angle_z_deg": tip_gravity_angle,
                "tip_gravity_angle_deg": tip_gravity_angle,
                "tip_start_gravity_angle_deg": tip_start_gravity_angle,
                "tip_primary_angle_deg": tip_primary_angle,
                "tip_vector_requested_window": float(tip_vector_window),
                "tip_vector_arc_window": float(actual_tip_window),
                "tip_vector_window_unit": "mesh_unit",
                "lateral_start_count": np.nan,
                "selected_lateral_count": np.nan,
                **_point_columns(
                    "root_start",
                    root_start,
                ),
                **_point_columns(
                    "root_tip",
                    root_tip,
                ),
                **_vector_columns(
                    "tip_vector",
                    tip_start_point,
                    tip_point,
                ),
                **_vector_columns(
                    "tip_start_vector",
                    root_start,
                    root_tip,
                ),
                **_vector_columns(
                    "primary_vector",
                    primary_vector_start,
                    primary_vector_end,
                ),
                "gravity_dx": float(gravity[0]),
                "gravity_dy": float(gravity[1]),
                "gravity_dz": float(gravity[2]),
            }
        )
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    frame["coordinate_unit"] = "mesh_unit"
    frame.attrs["system_summary"] = _system_summary(
        frame,
        mesh_metadata=mesh_metadata or {},
        full_root_labels=full_root_labels,
    )
    return frame


def trait_summary_frame(traits: pd.DataFrame) -> pd.DataFrame:
    summary = dict(traits.attrs.get("system_summary", {}))
    if not summary:
        summary = _system_summary(traits, mesh_metadata={}, full_root_labels=None)
    return pd.DataFrame(
        [{"trait": key, "value": value} for key, value in summary.items()]
    )


def lateral_counts_frame(traits: pd.DataFrame) -> pd.DataFrame:
    lateral = traits[traits["root_order"] > 0]
    if lateral.empty:
        return pd.DataFrame(columns=["root_order", "lateral_root_count"])
    return (
        lateral.groupby("root_order", as_index=False)["root_id"]
        .count()
        .rename(columns={"root_id": "lateral_root_count"})
        .sort_values("root_order")
        .reset_index(drop=True)
    )


def angle_vectors_frame(traits: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "root_id",
        "parent_id",
        "root_order",
        "tip_gravity_angle_deg",
        "tip_start_gravity_angle_deg",
        "tip_primary_angle_deg",
    ]
    columns.extend(
        column
        for column in ("coordinate_unit",)
        if column in traits.columns
    )
    vector_prefixes = ("root_start_", "root_tip_", "tip_vector_", "tip_start_vector_", "primary_vector_", "gravity_")
    columns.extend(
        column
        for column in traits.columns
        if column not in columns and column.startswith(vector_prefixes)
    )
    return traits.loc[traits["root_order"] > 0, columns].copy()


def _base_record(
    *,
    root_id: str,
    parent_id: str,
    order: int,
    length: float,
    chord: float,
    radii: np.ndarray,
    point_count: int,
    confidence: float,
    qc_flags: list[str],
    surface_area: float,
    surface_method: str,
    volume: float,
    volume_method: str,
) -> dict:
    radii = np.asarray(radii, dtype=float)
    valid = radii[np.isfinite(radii) & (radii >= 0)]
    mean_radius = float(np.mean(valid)) if len(valid) else 0.0
    median_radius = float(np.median(valid)) if len(valid) else 0.0
    return {
        "root_id": root_id,
        "parent_id": parent_id,
        "root_order": int(order),
        "length": float(length),
        "chord_length": float(chord),
        "tortuosity": float(length / chord) if chord > 1e-12 else np.nan,
        "mean_radius": mean_radius,
        "mean_diameter": 2.0 * mean_radius,
        "median_diameter": 2.0 * median_radius,
        "minimum_diameter": 2.0 * float(np.min(valid)) if len(valid) else 0.0,
        "maximum_diameter": 2.0 * float(np.max(valid)) if len(valid) else 0.0,
        "surface_area": float(surface_area),
        "surface_area_method": surface_method,
        "volume": float(volume),
        "volume_method": volume_method,
        "point_count": int(point_count),
        "confidence": float(confidence),
        "qc_flags": ";".join(qc_flags),
        "length_unit": "mesh_unit",
        "area_unit": "mesh_unit^2",
        "volume_unit": "mesh_unit^3",
    }


def _radius_profile(
    assigned_points: np.ndarray,
    path: np.ndarray,
    normalization: Normalization,
) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if len(path) == 0:
        return np.empty(0, dtype=float)
    if len(assigned_points) == 0:
        return np.zeros(len(path), dtype=float)
    if len(path) == 1:
        distances = np.linalg.norm(np.asarray(assigned_points, dtype=float) - path[0], axis=1)
        nearest = np.zeros(len(distances), dtype=int)
    else:
        segment_start = path[:-1]
        segment_vectors = path[1:] - path[:-1]
        segment_length2 = np.sum(segment_vectors**2, axis=1)
        segment_midpoints = 0.5 * (path[:-1] + path[1:])
        tree = cKDTree(segment_midpoints)
        query_k = min(8, len(segment_midpoints))
        all_distances: list[np.ndarray] = []
        all_nearest: list[np.ndarray] = []
        assigned_points = np.asarray(assigned_points, dtype=float)
        for chunk_start in range(0, len(assigned_points), 50000):
            chunk = assigned_points[chunk_start : chunk_start + 50000]
            _, candidates = tree.query(chunk, k=query_k, workers=worker_threads())
            if query_k == 1:
                candidates = np.asarray(candidates, dtype=int)[:, None]
            candidate_start = segment_start[candidates]
            candidate_vectors = segment_vectors[candidates]
            denominator = np.maximum(segment_length2[candidates], 1e-20)
            offsets = chunk[:, None, :] - candidate_start
            parameters = np.clip(
                np.einsum("nkj,nkj->nk", offsets, candidate_vectors) / denominator,
                0.0,
                1.0,
            )
            projections = candidate_start + parameters[..., None] * candidate_vectors
            candidate_distances = np.linalg.norm(chunk[:, None, :] - projections, axis=2)
            best_position = np.argmin(candidate_distances, axis=1)
            rows = np.arange(len(chunk))
            best_segments = candidates[rows, best_position]
            best_parameters = parameters[rows, best_position]
            all_distances.append(candidate_distances[rows, best_position])
            all_nearest.append(best_segments + (best_parameters >= 0.5).astype(int))
        distances = np.concatenate(all_distances)
        nearest = np.concatenate(all_nearest)
    global_radius = float(np.percentile(distances, 65)) if len(distances) else 0.0
    radii = np.full(len(path), global_radius, dtype=float)
    for index in np.unique(nearest):
        local = distances[nearest == index]
        if len(local) >= 3:
            radii[int(index)] = float(np.percentile(local, 65))
    if len(radii) > 2:
        for _ in range(2):
            smoothed = radii.copy()
            smoothed[1:-1] = (radii[:-2] + 2.0 * radii[1:-1] + radii[2:]) / 4.0
            radii = smoothed
    return radii * normalization.scale


def _frustum_measurements(points: np.ndarray, radii: np.ndarray) -> tuple[float, float]:
    if len(points) < 2:
        return 0.0, 0.0
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    radii = np.asarray(radii, dtype=float)
    if len(radii) != len(points):
        radius = float(np.mean(radii)) if len(radii) else 0.0
        radii = np.full(len(points), radius, dtype=float)
    r0 = radii[:-1]
    r1 = radii[1:]
    slant = np.sqrt(segment_lengths**2 + (r1 - r0) ** 2)
    area = np.pi * (r0 + r1) * slant
    volume = np.pi * segment_lengths * (r0**2 + r0 * r1 + r1**2) / 3.0
    return float(np.sum(area)), float(np.sum(volume))


def _partition_mesh_surface_areas(
    vertices: np.ndarray | None,
    triangles: np.ndarray | None,
    labels: np.ndarray | None,
) -> dict[int, float]:
    if vertices is None or triangles is None or labels is None:
        return {}
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    labels = np.asarray(labels, dtype=int)
    if len(labels) != len(vertices) or not len(triangles):
        return {}
    a = labels[triangles[:, 0]]
    b = labels[triangles[:, 1]]
    c = labels[triangles[:, 2]]
    face_labels = np.full(len(triangles), -2, dtype=int)
    face_labels[a == b] = a[a == b]
    remaining = face_labels == -2
    face_labels[remaining & (a == c)] = a[remaining & (a == c)]
    remaining = face_labels == -2
    face_labels[remaining & (b == c)] = b[remaining & (b == c)]
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    result: dict[int, float] = {}
    for label in np.unique(face_labels):
        if label >= 0:
            result[int(label)] = float(np.sum(areas[face_labels == label]))
    return result


def _system_summary(
    traits: pd.DataFrame,
    *,
    mesh_metadata: dict,
    full_root_labels: np.ndarray | None,
) -> dict:
    laterals = traits[traits["root_order"] > 0]
    primary = traits[traits["root_order"] == 0]
    mesh_area = mesh_metadata.get("surface_area_source_units2")
    mesh_volume = mesh_metadata.get("absolute_volume_source_units3")
    exact_surface_available = mesh_area is not None
    whole_area = (
        float(mesh_area)
        if exact_surface_available
        else float(traits["surface_area"].sum())
    )
    exact_volume_available = bool(mesh_metadata.get("volume_reliable", False)) and mesh_volume is not None
    whole_volume = (
        float(mesh_volume)
        if exact_volume_available
        else float(traits["volume"].sum())
    )
    labels = None if full_root_labels is None else np.asarray(full_root_labels, dtype=int)
    assigned_fraction = np.nan if labels is None or not len(labels) else float(np.mean(labels >= 0))
    uncertain_fraction = np.nan if labels is None or not len(labels) else float(np.mean(labels == -2))
    unassigned_fraction = np.nan if labels is None or not len(labels) else float(np.mean(labels == -1))
    return {
        "root_count_total": int(len(traits)),
        "lateral_root_count_total": int(len(laterals)),
        "maximum_root_order": int(laterals["root_order"].max()) if len(laterals) else 0,
        "length_unit": "mesh_unit",
        "area_unit": "mesh_unit^2",
        "volume_unit": "mesh_unit^3",
        "root_system_length": float(traits["length"].sum()),
        "primary_root_length": float(primary["length"].sum()),
        "lateral_root_length_sum": float(laterals["length"].sum()),
        "root_system_surface_area": whole_area,
        "root_system_surface_area_method": (
            "full_mesh_triangle_area" if exact_surface_available else "sum_per_root_surface_estimates"
        ),
        "primary_root_surface_area": float(primary["surface_area"].sum()),
        "lateral_root_surface_area_sum": float(laterals["surface_area"].sum()),
        "root_system_volume": whole_volume,
        "root_system_volume_method": (
            "closed_manifold_component_volume" if exact_volume_available else "sum_centerline_frustum_estimates"
        ),
        "root_system_volume_reliable": bool(exact_volume_available),
        "root_system_volume_reliability_reason": mesh_metadata.get("volume_reliability_reason", "frustum_estimate"),
        "primary_root_volume_estimate": float(primary["volume"].sum()),
        "lateral_root_volume_estimate_sum": float(laterals["volume"].sum()),
        "assigned_vertex_fraction": assigned_fraction,
        "uncertain_vertex_fraction": uncertain_fraction,
        "unassigned_vertex_fraction": unassigned_fraction,
    }


def _chord_length(points: np.ndarray) -> float:
    return 0.0 if len(points) < 2 else float(np.linalg.norm(points[-1] - points[0]))


def _arc_window_vector(
    path: np.ndarray,
    *,
    anchor_index: int,
    requested_window: float,
    from_start: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return a sampling-invariant vector over a source mesh-unit window.

    Long roots use the requested mesh-unit window. For a root shorter than four
    windows, the final quarter of its arc is used so a local tip direction is
    not silently replaced by the whole-root chord.
    """

    path = np.asarray(path, dtype=float)
    if len(path) == 0:
        zero = np.zeros(3, dtype=float)
        return zero, zero, zero, 0.0
    if len(path) == 1:
        return path[0].copy(), path[0].copy(), np.zeros(3, dtype=float), 0.0
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(cumulative[-1])
    if total <= 1e-12:
        return path[0].copy(), path[-1].copy(), np.zeros(3, dtype=float), 0.0
    effective_window = min(float(requested_window), 0.25 * total)
    effective_window = max(0.0, effective_window)
    if from_start is True:
        start_arc, end_arc = 0.0, effective_window
    elif from_start is False:
        start_arc, end_arc = total - effective_window, total
    else:
        anchor_index = int(np.clip(anchor_index, 0, len(path) - 1))
        center_arc = float(cumulative[anchor_index])
        start_arc = float(np.clip(center_arc - 0.5 * effective_window, 0.0, total - effective_window))
        end_arc = start_arc + effective_window
    unique_arc, unique_indices = np.unique(cumulative, return_index=True)
    unique_points = path[unique_indices]
    start = np.array(
        [np.interp(start_arc, unique_arc, unique_points[:, axis]) for axis in range(3)],
        dtype=float,
    )
    end = np.array(
        [np.interp(end_arc, unique_arc, unique_points[:, axis]) for axis in range(3)],
        dtype=float,
    )
    return start, end, end - start, float(end_arc - start_arc)


def _point_columns(prefix: str, point: np.ndarray) -> dict[str, float]:
    point = np.asarray(point, dtype=float)
    return {
        f"{prefix}_x": float(point[0]),
        f"{prefix}_y": float(point[1]),
        f"{prefix}_z": float(point[2]),
    }


def _vector_columns(prefix: str, start: np.ndarray, end: np.ndarray) -> dict[str, float]:
    vector = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    return {
        **_point_columns(f"{prefix}_start", start),
        **_point_columns(f"{prefix}_end", end),
        f"{prefix}_dx": float(vector[0]),
        f"{prefix}_dy": float(vector[1]),
        f"{prefix}_dz": float(vector[2]),
    }
