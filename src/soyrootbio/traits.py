from __future__ import annotations

from collections import defaultdict
from hashlib import sha256

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geometry import angle_degrees, path_length, vector_angle_degrees
from .types import Normalization, RootPath
from .runtime import worker_threads


GRAVITY = np.array([0.0, 0.0, -1.0])
MEASUREMENT_METRICS = ("length", "mean_diameter", "volume", "surface_area",
                       "base_parent_angle_deg", "tip_angle_parent_deg", "tip_primary_angle_deg",
                       "tip_gravity_angle_deg", "tip_start_gravity_angle_deg")


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
    primary_centerline_assessment: dict | None = None,
    tip_vector_window: float = 2.0,
    noise_mask: np.ndarray | None = None,
    primary_top_reference: np.ndarray | None = None,
) -> pd.DataFrame:
    """Measure per-root geometry in calibrated mm (legacy mesh units) and angles.

    Ordered paths are stored collar/insertion-to-tip.  The three user-requested
    directional angles are therefore unambiguous and remain in the full
    0--180° range:

    * ``tip_gravity_angle_deg``: lateral tip tangent versus ``(0, 0, -1)``;
    * ``tip_start_gravity_angle_deg``: start-to-tip vector versus gravity;
    * ``tip_primary_angle_deg``: lateral start tangent versus the ordered
      primary tangent at the lateral insertion location.  The historical
      column name is retained for export compatibility.
    """
    evidence_triangles = triangles
    if noise_mask is not None:
        noise_mask = np.asarray(noise_mask, dtype=bool)
        if full_points is None or full_root_labels is None or noise_mask.shape != (len(full_points),):
            raise ValueError('Noise exclusion requires matching full geometry and final labels')
        if np.any(np.asarray(full_root_labels)[noise_mask] != -1):
            raise ValueError('Noise-excluded vertices cannot contribute to root measurements')
        if noise_mask.any() and triangles is not None:
            from .io import _mesh_audit
            triangles = np.asarray(triangles)[~np.any(noise_mask[np.asarray(triangles)], axis=1)]
            mesh_metadata = {**(mesh_metadata or {}), **_mesh_audit(full_points, triangles),
                             'noise_excluded_from_totals': True}
    gravity = np.asarray(gravity, dtype=float).copy()
    gravity /= max(np.linalg.norm(gravity), 1e-12)
    mesh_areas = _partition_mesh_surface_areas(
        full_points,
        triangles,
        full_root_labels,
    )
    factor = normalization.mm_per_mesh_unit if normalization.mm_per_mesh_unit is not None else 1.
    mesh_areas = {label: area * factor**2 for label, area in mesh_areas.items()}
    mesh_metadata = dict(mesh_metadata or {})
    for key, exponent in (("surface_area_source_units2", 2), ("signed_volume_source_units3", 3),
                          ("absolute_volume_source_units3", 3)):
        if mesh_metadata.get(key) is not None:
            mesh_metadata[key] = mesh_metadata[key] * factor**exponent
    primary_original = normalization.to_output_points(primary_path)
    # Use the same final full-resolution ownership as the fitting pass, even
    # when analysis was reduced and a small root has no sampled vertices.
    support_points = points
    support_labels = None
    if full_points is not None and full_root_labels is not None:
        support_points = normalization.transform_points(np.asarray(full_points))
        support_labels = np.asarray(full_root_labels)
    if support_labels is None:
        support_labels = np.where(primary_mask, 0, np.where(np.asarray(lateral_labels) > 0, lateral_labels, -1))
    from .measurement_evidence import make_evidence_context
    evidence_context = make_evidence_context(support_points, evidence_triangles)
    primary_support = support_points[support_labels == 0] if support_labels is not None else points[primary_mask]
    primary_radius_assessment = {}
    primary_measurement_support, primary_evidence = _measurement_support(
        support_points, support_labels, 0, primary_path, primary_centerline_assessment or {},
        evidence_triangles, evidence_context, primary_radius_assessment, origin_reference=primary_top_reference)
    primary_radii = _radius_profile(primary_measurement_support, primary_path, normalization,
                                    assessment=primary_radius_assessment, support_evidence=primary_evidence)
    primary_length = path_length(primary_original)
    primary_frustum_area, primary_frustum_volume = _frustum_measurements(
        primary_original,
        primary_radii,
        primary_radius_assessment.get("radius_interval_mask", []),
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
            radius_segment_lengths=np.linalg.norm(np.diff(primary_original, axis=0), axis=1),
            radius_interval_mask=primary_radius_assessment.get("radius_interval_mask", []),
            point_count=int(len(primary_support)),
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
            "selected_lateral_count": sum('noise_excluded_root' not in root.qc_flags for root in lateral_paths),
            "angle_deg": np.nan,
            "base_parent_angle_deg": np.nan,
            "tip_angle_parent_deg": np.nan,
            "tip_angle_primary_deg": np.nan,
            "tip_angle_z_deg": np.nan,
            "tip_gravity_angle_deg": np.nan,
            "tip_start_gravity_angle_deg": np.nan,
            "tip_primary_angle_deg": np.nan,
            "parent_connector_mode": "none",
            "preserved_topology_length": 0.0,
            "topology_path_length": primary_length,
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
    p_tip_start, p_tip_end, p_tip_vector, p_window = _arc_window_vector(
        primary_original, anchor_index=len(primary_original)-1, requested_window=tip_vector_window, from_start=False)
    records[0].update(tip_gravity_angle_deg=vector_angle_degrees(p_tip_vector, gravity),
                      tip_angle_z_deg=vector_angle_degrees(p_tip_vector, gravity),
                      tip_start_gravity_angle_deg=vector_angle_degrees(primary_original[-1]-primary_original[0], gravity),
                      tip_vector_requested_window=float(tip_vector_window), tip_vector_arc_window=float(p_window),
                      tip_vector_window_unit=normalization.coordinate_unit,
                      **_vector_columns("tip_vector", p_tip_start, p_tip_end),
                      **_vector_columns("tip_start_vector", primary_original[0], primary_original[-1]))
    primary_unmeasurable = any(flag in (primary_qc_flags or []) for flag in ("centerline_no_support", "centerline_insufficient_support", "centerline_nodule_obscured"))
    records[0]["centerline_fit_status"] = (primary_centerline_assessment or {}).get("status", "insufficient_support" if primary_unmeasurable else "not_assessed")
    if primary_unmeasurable:
        _mark_unmeasurable(records[0])
    _apply_supported_length(records[0], primary_path, primary_evidence, normalization,
                            primary_centerline_assessment or {}, primary_unmeasurable)
    _record_radius_quality(records[0], primary_radius_assessment, normalization, primary_centerline_assessment)

    primary_tree = cKDTree(primary_path)
    for lateral_index, lateral in enumerate(lateral_paths, start=1):
        parent_path = lateral.parent_points if lateral.parent_points is not None and len(lateral.parent_points) else primary_path
        parent_tree = cKDTree(parent_path)
        original = normalization.to_output_points(lateral.points)
        body_index = int(lateral.body_start_index)
        body = lateral.points[body_index:]
        body_original = original[body_index:]
        parent_original = normalization.to_output_points(parent_path)
        label_mask = lateral_labels == lateral_index
        support = support_points[support_labels == lateral_index] if support_labels is not None else points[label_mask]
        radius_assessment = {}
        measurement_support, evidence = _measurement_support(
            support_points, support_labels, lateral_index, lateral.points, lateral.centerline_assessment,
            evidence_triangles, evidence_context, radius_assessment, body_start_index=body_index,
            origin_reference=lateral.insertion_point)
        radii = _radius_profile(measurement_support, body, normalization, assessment=radius_assessment,
                               support_evidence=evidence)
        length = path_length(original)
        chord = _chord_length(original)
        frustum_area, frustum_volume = _frustum_measurements(body_original, radii,
                                                           radius_assessment.get("radius_interval_mask", []))

        _, parent_index = parent_tree.query(lateral.points[0], k=1)
        _, primary_index = primary_tree.query(lateral.points[0], k=1)
        (
            base_vector_start,
            base_vector_end,
            base_vector,
            actual_base_window,
        ) = _arc_window_vector(
            body_original,
            anchor_index=0,
            requested_window=tip_vector_window,
            from_start=True,
        )
        tip_start_point, tip_point, tip_vector, actual_tip_window = _arc_window_vector(
            body_original,
            anchor_index=len(body_original) - 1,
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
        tip_primary_angle = vector_angle_degrees(base_vector, primary_vector)
        tip_gravity_angle = vector_angle_degrees(tip_vector, gravity)
        tip_start_gravity_angle = vector_angle_degrees(tip_start_vector, gravity)

        qc_flags = list(lateral.qc_flags)
        if np.linalg.norm(tip_vector) <= 1e-12:
            qc_flags.append("undefined_tip_vector")
        if np.linalg.norm(tip_start_vector) <= 1e-12:
            qc_flags.append("undefined_tip_start_vector")
        if np.linalg.norm(base_vector) <= 1e-12:
            qc_flags.append("undefined_base_vector")
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
            radius_segment_lengths=np.linalg.norm(np.diff(body_original, axis=0), axis=1),
            radius_interval_mask=radius_assessment.get("radius_interval_mask", []),
            point_count=int(len(support)),
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
                "exposed_body_length": path_length(body_original),
                "parent_connector_length": path_length(original[:body_index + 1]) if body_index else 0.0,
                "parent_connector_mode": lateral.centerline_assessment.get("parent_connector_mode", "not_assessed"),
                "preserved_topology_length": (
                    path_length(original[:body_index + 1])
                    if lateral.centerline_assessment.get("parent_connector_mode") == "preserved_topology"
                    else 0.0
                ),
                "topology_path_length": length,
                "centerline_fit_status": lateral.centerline_assessment.get("status", "not_assessed"),
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
                    "base_vector",
                    base_vector_start,
                    base_vector_end,
                ),
                **_vector_columns(
                    "primary_vector",
                    primary_vector_start,
                    primary_vector_end,
                ),
                "base_vector_requested_window": float(tip_vector_window),
                "base_vector_arc_window": float(actual_base_window),
                "base_vector_window_unit": "mesh_unit",
                "gravity_dx": float(gravity[0]),
                "gravity_dy": float(gravity[1]),
                "gravity_dz": float(gravity[2]),
            }
        )
        connector_mode = lateral.centerline_assessment.get("parent_connector_mode")
        if connector_mode in {"preserved_topology", "retained_prior"}:
            # Final ownership did not establish a measurable centreline for
            # the preserved basal path (or any of the retained prior path).
            record["length"] = np.nan
            record["tortuosity"] = np.nan
        if connector_mode == "retained_prior":
            record["chord_length"] = np.nan
            record["exposed_body_length"] = np.nan
            record["volume"] = np.nan
            record["volume_method"] = "unmeasurable_retained_prior_path"
        if lateral.centerline_assessment.get("status") in {"no_support", "insufficient_support", "nodule_obscured"}:
            _mark_unmeasurable(record)
        _apply_supported_length(record, lateral.points, evidence, normalization, lateral.centerline_assessment,
                                lateral.centerline_assessment.get("status") in {"no_support", "insufficient_support", "nodule_obscured"})
        _record_radius_quality(record, radius_assessment, normalization, lateral.centerline_assessment)
        if 'noise_excluded_root' in lateral.qc_flags:
            if len(support):
                raise ValueError('A noise-only topology placeholder must have no assigned support')
            for key, value in list(record.items()):
                if isinstance(value, (int, float, np.number)) and key not in {'root_order', 'point_count'}:
                    record[key] = np.nan
                elif key.endswith('_method'):
                    record[key] = 'excluded_noise'
            record['point_count'] = 0
            record['measurement_excluded'] = True
            for metric in MEASUREMENT_METRICS:
                _set_metric_status(record, metric, "excluded", "excluded_noise")
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    if 'measurement_excluded' in frame:
        frame['measurement_excluded'] = frame['measurement_excluded'].eq(True)
    _annotate_measured_length_comparisons(frame)
    unit = normalization.coordinate_unit
    frame["coordinate_unit"] = unit
    frame["length_unit"] = unit
    frame["area_unit"] = f"{unit}^2"
    frame["volume_unit"] = f"{unit}^3"
    for column in ("tip_vector_window_unit", "base_vector_window_unit"):
        if column in frame:
            frame[column] = unit
    from .architecture_traits import eligible_mask
    frame["measurement_eligible"] = eligible_mask(frame)
    frame.attrs["system_summary"] = _system_summary(
        frame,
        mesh_metadata=mesh_metadata or {},
        full_root_labels=(np.asarray(full_root_labels)[~noise_mask]
                          if noise_mask is not None else full_root_labels),
    )
    frame.attrs["system_summary"].update(length_unit=unit, area_unit=f"{unit}^2", volume_unit=f"{unit}^3")
    from dataclasses import replace
    from .architecture_traits import add_architecture_traits
    output_paths = [replace(root, points=normalization.to_output_points(root.points),
                            insertion_point=None if root.insertion_point is None else
                            normalization.to_output_points(np.asarray(root.insertion_point)[None])[0])
                    for root in lateral_paths]
    output_points = (normalization.source_to_output_points(full_points) if full_points is not None
                     else normalization.to_output_points(points))
    output_labels = (np.asarray(full_root_labels) if full_root_labels is not None else
                     np.where(primary_mask, 0, np.where(lateral_labels > 0, lateral_labels, -1)))
    top = normalization.to_output_points(np.asarray(primary_top_reference if primary_top_reference is not None
                                                     else primary_path[0])[None])[0]
    add_architecture_traits(frame, primary_original, output_paths, output_points, output_labels,
                            triangles, top, gravity, unit)
    if primary_top_reference is None:
        frame.attrs["measurement_policy"]["primary_top_source"] = "path_start_fallback_no_explicit_reference"
    system = frame.attrs["architecture_summary"][0]
    measured = frame[frame.measurement_eligible]
    frame.attrs["system_summary"].update(
        root_count_total=len(measured), lateral_root_count_total=int((measured.root_order > 0).sum()),
        excluded_root_count=len(frame)-len(measured), root_system_length=system["length_sum"],
        root_system_surface_area=system["mesh_surface_area"],
        root_system_surface_area_method="eligible_segmented_mesh_triangles",
        root_system_volume=system["volume_frustum"], root_system_volume_method="eligible_centerline_frustum_estimates",
        root_system_mesh_volume=system["mesh_volume"], root_system_frustum_volume=system["volume_frustum"],
        root_system_mesh_volume_reliable=system["mesh_volume_reliable"],
        root_system_volume_reliable=False)
    for name in ("source_surface_area_including_nodules", "source_volume_including_nodules"):
        frame.attrs["system_summary"].pop(name, None)
    if noise_mask is not None:
        frame.attrs['noise_exclusion_sha256'] = sha256(noise_mask.tobytes()).hexdigest()
        frame.attrs['noise_excluded_vertex_count'] = int(noise_mask.sum())
        if noise_mask.any():
            frame.attrs['system_summary']['measurement_vertex_count'] = int(np.sum(~noise_mask))
            frame.attrs['system_summary']['noise_excluded_vertex_count'] = int(noise_mask.sum())
    return frame


def _mark_unmeasurable(record: dict) -> None:
    """A stored support location is not a measured zero-length biological root."""
    for key in ("length", "chord_length", "tortuosity", "mean_radius", "mean_diameter", "median_diameter",
                "minimum_diameter", "maximum_diameter", "volume", "exposed_body_length"):
        record[key] = np.nan
    for key in ("root_tip_x", "root_tip_y", "root_tip_z"):
        record[key] = np.nan
    if record["surface_area_method"] == "centerline_frustum_estimate":
        record["surface_area"] = np.nan


def trait_summary_frame(traits: pd.DataFrame) -> pd.DataFrame:
    summary = dict(traits.attrs.get("system_summary", {}))
    if not summary:
        summary = _system_summary(traits, mesh_metadata={}, full_root_labels=None)
    return pd.DataFrame(
        [{"trait": key, "value": value} for key, value in summary.items()]
    )


def lateral_counts_frame(traits: pd.DataFrame) -> pd.DataFrame:
    traits = _measured_rows(traits)
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
    vector_prefixes = (
        "root_start_",
        "root_tip_",
        "tip_vector_",
        "tip_start_vector_",
        "base_vector_",
        "primary_vector_",
        "gravity_",
    )
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
    radius_segment_lengths: np.ndarray | None = None,
    radius_interval_mask: np.ndarray | None = None,
) -> dict:
    radii = np.asarray(radii, dtype=float)
    valid = radii[np.isfinite(radii) & (radii >= 0)]
    mean_radius = float(np.mean(valid)) if len(valid) else np.nan
    median_radius = float(np.median(valid)) if len(valid) else np.nan
    statistics_method = "station_statistics_insufficient_arc_support"
    if radius_segment_lengths is not None and len(radii) >= 2:
        lengths = np.asarray(radius_segment_lengths, dtype=float)
        supported = (np.isfinite(radii[:-1]) & np.isfinite(radii[1:]) & (radii[:-1] >= 0)
                     & (radii[1:] >= 0) & np.isfinite(lengths) & (lengths > 0))
        if radius_interval_mask is not None:
            mask = np.asarray(radius_interval_mask, bool)
            supported &= mask if mask.shape == supported.shape else False
        weights = lengths[supported]
        if len(weights):
            first, second = radii[:-1][supported], radii[1:][supported]
            mean_radius = float(np.dot(weights, .5*(first+second))/weights.sum())
            low, high = np.minimum(first,second), np.maximum(first,second)
            left,right = float(low.min()),float(high.max())
            for _ in range(64):
                middle = .5*(left+right)
                fraction = np.where(high > low, np.clip((middle-low)/np.maximum(high-low,1e-30),0,1), middle >= low)
                if np.dot(weights,fraction) < .5*weights.sum():
                    left=middle
                else:
                    right=middle
            median_radius = .5*(left+right)
            statistics_method = "arc-weighted-piecewise-linear-radius-v1"
    return {
        "root_id": root_id,
        "parent_id": parent_id,
        "root_order": int(order),
        "length": float(length),
        "chord_length": float(chord),
        "tortuosity": float(length / chord) if chord > 1e-12 else np.nan,
        "mean_radius": mean_radius,
        "diameter_statistics_method": statistics_method,
        "mean_diameter": 2.0 * mean_radius,
        "median_diameter": 2.0 * median_radius,
        "minimum_diameter": 2.0 * float(np.min(valid)) if len(valid) else np.nan,
        "maximum_diameter": 2.0 * float(np.max(valid)) if len(valid) else np.nan,
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


def _positive_support_spacing(points: np.ndarray) -> float | None:
    """Estimate local positive sampling on a private evidence copy."""
    if len(points) < 2:
        return None
    tree = cKDTree(points)
    sample = points[np.linspace(0, len(points)-1, min(4096, len(points)), dtype=int)]
    distances, _ = tree.query(sample, k=min(8, len(points)), workers=worker_threads())
    positive = np.where(np.asarray(distances) > 0., distances, np.inf).min(axis=1)
    if np.count_nonzero(np.isfinite(positive)) < .5 * len(sample):
        # Duplicate facet positions can exhaust k8; source rows stay untouched.
        distinct = np.unique(points, axis=0)
        if len(distinct) < 2:
            return None
        sample = distinct[np.linspace(0, len(distinct)-1, min(4096,len(distinct)), dtype=int)]
        distances, _ = cKDTree(distinct).query(sample, k=2, workers=worker_threads())
        positive = np.asarray(distances)[:,1]
    positive = positive[np.isfinite(positive) & (positive > 0.)]
    return float(np.median(positive)) if len(positive) else None


def _measurement_support(points, labels, label, path, fitting, triangles, context, assessment,
                         *, body_start_index=0, origin_reference=None):
    """Consume current evidence, or conservatively import a legacy native component.

    An invalid persisted record is never refreshed against edited geometry. Legacy
    imports retain useful fragments but cannot certify the biological endpoints.
    """
    from .measurement_evidence import (assess_origin_coverage, build_measurement_evidence, infer_component_indices,
                                       upgrade_measurement_evidence,
                                       resolve_measurement_support)
    owned = np.flatnonzero(labels == label)
    fallback_spacing = fitting.get("fit_spacing")
    fallback_spacing = float(fallback_spacing) if fallback_spacing is not None and np.isfinite(fallback_spacing) and fallback_spacing > 0 else 1e-6
    spacing = _positive_support_spacing(points[owned]) or fallback_spacing
    evidence = fitting.get("measurement_evidence")
    reason = "current"
    unsupported = (fitting.get("status") in {"no_support", "insufficient_support", "nodule_obscured",
                                           "qc_rejected_prior_retained", "attachment_preserving_prior_retained"}
                   or fitting.get("parent_connector_mode") == "retained_prior"
                   or str(fitting.get("traits_geometry_source", "")).startswith(("retained_prior", "unmeasurable_support")))
    if evidence is None:
        if triangles is None or not len(triangles):
            ids, reason = owned, "legacy_point_support_unverified_identity"
        else:
            ids, reason = infer_component_indices(points, labels, label, triangles, fitting,
                                                  spacing=spacing, context=context)
        if ids is not None:
            evidence = build_measurement_evidence(points, labels, label, path, triangles=triangles,
                selected_vertex_indices=ids, spacing=spacing, body_start_index=body_start_index,
                identity_resolved=False, attachment_supported=False, tip_supported=False,
                source="legacy_component_measurement", context=context)
    if evidence is not None:
        upgraded, upgrade_reason = upgrade_measurement_evidence(evidence, points, labels, label, path,
            triangles=triangles, body_start_index=body_start_index, context=context)
        if upgraded is None:
            evidence = dict(evidence, status="invalidated", invalid_reason=upgrade_reason)
        else:
            evidence = upgraded
            if reason == "current":
                reason = upgrade_reason
    ids, validation = resolve_measurement_support(evidence, points, labels, label, path,
        triangles=triangles, body_start_index=body_start_index, context=context)
    valid = ids is not None and not unsupported
    if ids is None:
        reason = validation if evidence is not None else reason
    if unsupported:
        reason = "retained_or_unsupported_path_is_not_a_measurement"
    if not valid:
        ids = np.empty(0, dtype=int)
        evidence = dict(native_vertex_indices=[], support_station_indices=[], support_arc_positions=[],
                        supported_station_mask=[False] * len(path),
                        supported_interval_mask=[False] * max(0, len(path)-1),
                        body_start_index=body_start_index, spacing=spacing,
                        identity_resolved=False, attachment_supported=False, biological_tip_verified=False)
    evidence = dict(evidence, measurement_evidence_valid=valid, measurement_evidence_reason=reason)
    if valid and fitting.get("attachment_context_changed_since_fit"):
        # Parent edits invalidate the anatomical attachment reference without
        # discarding independently current child-body geometry or a real tip.
        evidence["attachment_supported"] = False
        evidence["measurement_evidence_reason"] = reason = "attachment_context_changed_since_fit"
    independent = fitting.get("prefit_identity_evidence") or {}
    if valid and independent.get("reference_valid"):
        reference_origin = origin_reference if origin_reference is not None else independent.get("origin")
        origin_evidence = assess_origin_coverage(path, reference_origin,
            reference_path=independent.get("prior_path"), spacing=float(evidence.get("spacing", spacing)),
            attachment_supported=bool(evidence.get("attachment_supported", False)))
    else:
        origin_evidence = evidence.get("origin_coverage_evidence") or {
            "origin_coverage_complete": bool(valid and evidence.get("origin_coverage_complete", False)),
            "reason": "explicit_origin_certificate" if valid and evidence.get("origin_coverage_complete") else "independent_origin_reference_unavailable"}
    if fitting.get("attachment_context_changed_since_fit"):
        origin_evidence = dict(origin_evidence, origin_coverage_complete=False,
                               reason="attachment_context_changed_since_fit")
    evidence["origin_coverage_evidence"] = origin_evidence
    evidence["origin_coverage_complete"] = bool(origin_evidence.get("origin_coverage_complete", False))
    body_mask = np.asarray(evidence["supported_interval_mask"], bool)[body_start_index:]
    assessment.update(measurement_evidence_valid=valid, measurement_evidence_reason=reason,
                      measurement_identity_resolved=bool(evidence.get("identity_resolved", False)),
                      attachment_supported=bool(evidence.get("attachment_supported", False)),
                      biological_tip_verified=bool(evidence.get("biological_tip_verified", False)),
                      origin_coverage_complete=evidence["origin_coverage_complete"],
                      origin_coverage_evidence=origin_evidence,
                      centerline_intervals_complete=bool(len(body_mask) and body_mask.all()))
    return points[ids], evidence


def _apply_supported_length(record, path, evidence, normalization, fitting, unavailable=False):
    lengths = np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1)
    mask = np.asarray(evidence.get("supported_interval_mask", []), bool)
    record.setdefault("topology_path_length", normalization.to_output_length(float(lengths.sum())))
    if len(mask) != len(lengths) or unavailable or not evidence.get("measurement_evidence_valid", False):
        record["measurement_supported_length"] = np.nan
        record["length"] = np.nan
        return
    mask = mask.copy()
    body_start = int(evidence.get("body_start_index", 0))
    if body_start and evidence.get("attachment_supported") and fitting.get("parent_connector_supported"):
        mask[:body_start] = True
    supported = mask & np.isfinite(lengths) & (lengths > 0)
    amount = normalization.to_output_length(float(lengths[supported].sum())) if supported.any() else np.nan
    record["measurement_supported_length"] = amount
    record["length"] = amount
    if not supported.all():
        # The chord of disconnected measured pieces is not their tortuosity.
        record["chord_length"] = np.nan
        record["tortuosity"] = np.nan


def _record_radius_quality(record: dict, assessment: dict, normalization: Normalization,
                           centerline_assessment: dict | None = None) -> None:
    coverage = bool(assessment.get("body_coverage_complete", False))
    radius_complete = (coverage and assessment.get("outside_body_support_count",0) == 0
                and assessment.get("admissible_support_count",0) >= 3
                and assessment.get("positive_radius_support_count",0) >= 3
                and not assessment.get("global_radius_fallback", True))
    fitting = centerline_assessment or {}
    evidence_valid = bool(assessment.get("measurement_evidence_valid", False))
    identity_resolved = bool(assessment.get("measurement_identity_resolved", False))
    attachment_supported = bool(assessment.get("attachment_supported", False))
    tip_supported = bool(assessment.get("biological_tip_verified", False))
    origin_supported = bool(assessment.get("origin_coverage_complete", False))
    root_flags = set(str(record.get("qc_flags", "")).split(";"))
    identity_resolved = identity_resolved and fitting.get("identity_qc_passed") is not False
    identity_resolved = identity_resolved and not bool(root_flags & {
        "discrete_child_patch_unresolved", "native_body_identity_unresolved",
        "primary_contact_unresolved", "centerline_disconnected_distal_identity_review"})
    attachment_supported = attachment_supported and not bool(root_flags & {
        "attachment_unresolved", "attachment_parent_unresolved", "attachment_parent_component_unresolved",
        "attachment_footprint_rejected", "parent_contact_unresolved", "primary_contact_unresolved"})
    fit_incomplete = (fitting.get("status") in {"partial_support", "qc_rejected_prior_retained", "attachment_preserving_prior_retained",
                         "no_support", "insufficient_support", "nodule_obscured"}
                      or fitting.get("parent_connector_mode") in {"unsupported_gap", "preserved_topology", "retained_prior"}
                      or fitting.get("exposed_body_supported") is False or fitting.get("fit_qc_passed") is False
                      or fitting.get("identity_qc_passed") is False
                      or str(fitting.get("traits_geometry_source", "")).startswith(("retained_prior", "unmeasurable_support")))
    path_complete = (evidence_valid and identity_resolved and attachment_supported and origin_supported and tip_supported
                     and bool(assessment.get("centerline_intervals_complete", False)) and not fit_incomplete)
    radius_complete = radius_complete and path_complete and bool(assessment.get("radius_intervals_complete", False))
    radius_complete = radius_complete and np.isfinite(record["mean_radius"]) and np.isfinite(record["mean_diameter"])
    flags = [flag for flag in str(record.get("qc_flags", "")).split(";") if flag]
    if assessment.get("outside_body_support_count",0):
        flags.append("radius_outside_fitted_body_support_excluded")
    if assessment.get("admissible_support_count",0) < 3 or assessment.get("positive_radius_support_count",0) < 3:
        flags.append("radius_insufficient_body_support")
    if not coverage:
        flags.append("radius_endpoint_support_incomplete")
    if assessment.get("undersampled_station_count",0):
        flags.append("radius_profile_interpolated_stations")
    if assessment.get("global_radius_fallback", True):
        flags.append("radius_global_fallback_insufficient_local_stations")
    if fit_incomplete:
        flags.append("final_centerline_measurement_support_incomplete")
    record.update(radius_profile_method="native-interval-transverse-support-v2",
                  radius_admissible_support_count=assessment.get("admissible_support_count",0),
                  radius_outside_body_support_count=assessment.get("outside_body_support_count",0),
                  radius_undersampled_station_count=assessment.get("undersampled_station_count",0),
                  radius_sampling_spacing=(normalization.to_output_length(assessment["sampling_spacing"])
                                           if assessment.get("sampling_spacing") is not None else np.nan),
                  radius_sampling_spacing_unit=normalization.coordinate_unit,
                  body_coverage_complete=coverage,
                  radius_profile_complete=radius_complete,
                  radius_global_fallback=assessment.get("global_radius_fallback", True),
                  centerline_measurement_support_complete=path_complete,
                  length_estimate_complete=path_complete and np.isfinite(record["length"]) and record["length"] > 0,
                  volume_estimate_complete=radius_complete and np.isfinite(record["volume"]) and record["volume"] > 0,
                  tracing_confidence=record.get("confidence", np.nan),
                  measurement_evidence_valid=evidence_valid,
                  measurement_evidence_reason=assessment.get("measurement_evidence_reason", "legacy_unverified_support"),
                  measurement_identity_resolved=identity_resolved,
                  origin_coverage_complete=origin_supported,
                  origin_coverage_reason=assessment.get("origin_coverage_evidence", {}).get("reason", "independent_origin_reference_unavailable"),
                  radius_supported_length=normalization.to_output_length(assessment.get("radius_supported_length", 0.)),
                  qc_flags=";".join(dict.fromkeys(flags)))
    for field in ("axial_loss", "axial_displacement", "radial_offset", "tolerance"):
        value = assessment.get("origin_coverage_evidence", {}).get(field)
        record[f"origin_{field}"] = normalization.to_output_length(value) if value is not None and np.isfinite(value) else np.nan
    reason = assessment.get("measurement_evidence_reason", "legacy_unverified_support")
    for metric, complete in (("length", record["length_estimate_complete"]),
                             ("mean_diameter", radius_complete), ("volume", record["volume_estimate_complete"]),
                             ("surface_area", identity_resolved and record["surface_area_method"] == "partitioned_mesh_triangles")):
        value = record.get(metric, np.nan)
        available = bool(np.isfinite(value) and value > 0)
        unavailable_reason = reason if not evidence_valid else {
            "length": "no_supported_positive_centerline_intervals",
            "mean_diameter": "insufficient_local_transverse_support",
            "volume": "no_supported_radius_intervals", "surface_area": "native_surface_unavailable"}[metric]
        partial_reason = "supported_partial_extent"
        if not identity_resolved:
            partial_reason += ";unresolved_root_identity"
        if metric != "surface_area" and not attachment_supported:
            partial_reason += ";unverified_attachment"
        if metric != "surface_area" and not origin_supported:
            partial_reason += ";unverified_origin_coverage"
        _set_metric_status(record, metric, "complete" if available and complete else "partial" if available else "unavailable",
                           "supported_complete_inventory_root" if available and complete else partial_reason if available else unavailable_reason)
    record["surface_area_estimate_complete"] = record["surface_area_measurement_status"] == "complete"
    # Endpoint and junction angles need the anatomical reference, not merely a
    # finite tangent at a cropped or uncertain fragment boundary.
    for metric in ("tip_gravity_angle_deg", "tip_start_gravity_angle_deg", "base_parent_angle_deg",
                   "tip_angle_parent_deg", "tip_primary_angle_deg"):
        valid_reference = evidence_valid and identity_resolved
        if metric in {"tip_gravity_angle_deg", "tip_angle_parent_deg", "tip_start_gravity_angle_deg"}:
            valid_reference &= tip_supported
        if metric != "tip_gravity_angle_deg":
            valid_reference &= attachment_supported and origin_supported
        available = valid_reference and np.isfinite(record.get(metric, np.nan))
        if not available:
            record[metric] = np.nan
        _set_metric_status(record, metric, "complete" if available else "unavailable",
                           "supported_anatomical_reference" if available else "unverified_tip_or_attachment_identity")
    record["angle_deg"] = record.get("base_parent_angle_deg", np.nan)
    record["tip_angle_primary_deg"] = record.get("tip_primary_angle_deg", np.nan)
    record["tip_angle_z_deg"] = record.get("tip_gravity_angle_deg", np.nan)


def _set_metric_status(record: dict, metric: str, status: str, reason: str) -> None:
    record[f"{metric}_measurement_status"] = status
    record[f"{metric}_measurement_eligible"] = status in {"complete", "partial"}
    record[f"{metric}_measurement_reason"] = reason


def _annotate_measured_length_comparisons(frame: pd.DataFrame) -> None:
    """Flag measured excess for junction review without changing any geometry.

    The fitting pass compares topology paths; supported measured lengths can
    differ. A ratio of partial extents is explicitly not a whole-root ratio.
    """
    from .architecture_traits import metric_eligible_mask
    frame["measured_child_parent_length_ratio"] = np.nan
    frame["measured_child_parent_length_excess"] = np.nan
    frame["measured_parent_length_measurement_status"] = ""
    frame["measured_child_parent_length_comparison_basis"] = ""
    frame["measured_child_parent_length_review"] = False
    review_flag = "measured_child_longer_than_parent"
    frame["qc_flags"] = frame.qc_flags.fillna("").map(
        lambda value: ";".join(flag for flag in str(value).split(";") if flag and flag != review_flag))
    rows = frame.set_index("root_id")
    available = set(frame.loc[metric_eligible_mask(frame, "length"), "root_id"])
    reviews = []
    for index, row in frame.iterrows():
        parent_id = row.get("parent_id")
        if row.root_id not in available or parent_id not in available:
            continue
        parent = rows.loc[parent_id]
        child_length, parent_length = float(row.length), float(parent.length)
        ratio = child_length / parent_length
        excess = max(0., child_length-parent_length)
        parent_status = parent.get("length_measurement_status", "partial")
        child_status = row.get("length_measurement_status", "partial")
        basis = "complete_root_lengths" if parent_status == child_status == "complete" else "supported_partial_extents"
        review = excess > 1e-12 * max(child_length, parent_length)
        frame.loc[index, "measured_child_parent_length_ratio"] = ratio
        frame.loc[index, "measured_child_parent_length_excess"] = excess
        frame.loc[index, "measured_parent_length_measurement_status"] = parent_status
        frame.loc[index, "measured_child_parent_length_comparison_basis"] = basis
        frame.loc[index, "measured_child_parent_length_review"] = review
        if review:
            frame.loc[index, "qc_flags"] = ";".join(filter(None, [frame.loc[index, "qc_flags"], review_flag]))
            reviews.append(dict(root_id=row.root_id, parent_id=parent_id,
                child_measured_length=child_length, parent_measured_length=parent_length,
                measured_length_ratio=ratio, measured_length_excess=excess,
                child_length_measurement_status=child_status, parent_length_measurement_status=parent_status,
                comparison_basis=basis, policy="review_only_no_topology_or_eligibility_change"))
    frame.attrs["measured_child_parent_length_reviews"] = reviews


def _radius_profile(assigned_points: np.ndarray, path: np.ndarray,
                    normalization: Normalization, *, assessment: dict | None = None,
                    support_evidence: dict | None = None) -> np.ndarray:
    """Measure transverse radius on supported fitted body, excluding axial overflow.

    A two-local-spacing terminal slab is independent of centerline node density.
    Global quantile fallback is restricted to admissible body support and never
    certifies unsupported centerline gaps or a cropped full-root measurement.
    """
    if support_evidence is not None:
        return _supported_radius_profile(assigned_points, path, normalization, support_evidence,
                                         {} if assessment is None else assessment)
    path = np.asarray(path, dtype=float)
    assigned_points = np.asarray(assigned_points, dtype=float)
    spacing = _positive_support_spacing(assigned_points)
    if assessment is None:
        assessment = {}
    assessment.update(sampling_spacing=spacing, assigned_support_count=len(assigned_points),
                      admissible_support_count=0, outside_body_support_count=0,
                      undersampled_station_count=len(path), positive_radius_support_count=0,
                      body_coverage_complete=False, global_radius_fallback=True)
    if len(path) == 0:
        return np.empty(0, dtype=float)
    if len(assigned_points) == 0:
        return np.full(len(path), np.nan, dtype=float)
    if len(path) == 1:
        distances = np.linalg.norm(assigned_points-path[0], axis=1)
        nearest = np.zeros(len(distances), dtype=int)
    else:
        segment_start = path[:-1]
        segment_vectors = np.diff(path, axis=0)
        segment_length2 = np.sum(segment_vectors**2, axis=1)
        segment_lengths = np.sqrt(segment_length2)
        cumulative = np.r_[0., np.cumsum(segment_lengths)]
        tree = cKDTree(.5*(path[:-1]+path[1:]))
        query_k = min(8, len(segment_start))
        all_distances, all_nearest, all_arcs = [], [], []
        for chunk_start in range(0,len(assigned_points),50000):
            chunk = assigned_points[chunk_start:chunk_start+50000]
            _, candidates = tree.query(chunk,k=query_k,workers=worker_threads())
            if query_k == 1:
                candidates = np.asarray(candidates,dtype=int)[:,None]
            offsets = chunk[:,None,:]-segment_start[candidates]
            vectors = segment_vectors[candidates]
            unbounded_parameters = np.einsum("nkj,nkj->nk",offsets,vectors)/np.maximum(segment_length2[candidates],1e-20)
            parameters = np.clip(unbounded_parameters,0.,1.)
            projections = segment_start[candidates]+parameters[...,None]*vectors
            candidate_distances = np.linalg.norm(chunk[:,None,:]-projections,axis=2)
            best = np.argmin(candidate_distances,axis=1)
            rows = np.arange(len(chunk))
            segments = candidates[rows,best]
            clipped = parameters[rows,best]
            unbounded = unbounded_parameters[rows,best]
            terminal = ((segments==0)&(unbounded<0.))|((segments==len(segment_start)-1)&(unbounded>1.))
            overflow = np.abs(unbounded-clipped)*segment_lengths[segments]
            admissible = ~terminal | (overflow <= (2.*spacing if spacing is not None else 0.))
            best_distances = candidate_distances[rows,best].copy()
            transverse = offsets[rows,best]-unbounded[:,None]*segment_vectors[segments]
            best_distances[terminal&admissible] = np.linalg.norm(transverse[terminal&admissible],axis=1)
            assessment["outside_body_support_count"] += int(np.count_nonzero(~admissible))
            all_distances.append(best_distances[admissible])
            all_nearest.append((segments+(clipped>=.5).astype(int))[admissible])
            all_arcs.append((cumulative[segments]+clipped*segment_lengths[segments])[admissible])
        distances = np.concatenate(all_distances)
        nearest = np.concatenate(all_nearest)
        arcs = np.concatenate(all_arcs)
        assessment["body_coverage_complete"] = bool(len(arcs) and spacing is not None
            and arcs.min() <= 2.*spacing and cumulative[-1]-arcs.max() <= 2.*spacing)
        assessment["start_coverage_complete"] = bool(len(arcs) and spacing is not None and arcs.min() <= 2.*spacing)
        assessment["end_coverage_complete"] = bool(len(arcs) and spacing is not None and cumulative[-1]-arcs.max() <= 2.*spacing)
        assessment["minimum_supported_arc"] = float(arcs.min()) if len(arcs) else None
        assessment["maximum_supported_arc"] = float(arcs.max()) if len(arcs) else None
    assessment["admissible_support_count"] = len(distances)
    assessment["positive_radius_support_count"] = int(np.count_nonzero(distances>0.))
    radii = np.full(len(path),np.nan,dtype=float)
    measured_stations = []
    for index in np.unique(nearest):
        local = distances[nearest==index]
        if len(local)>=3:
            measured_stations.append(int(index))
            radii[int(index)] = float(np.percentile(local,65))
    assessment["undersampled_station_count"] = len(path)-len(measured_stations)
    if len(measured_stations) >= 2:
        station_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        radii = np.interp(station_arc,station_arc[measured_stations],radii[measured_stations])
        if not assessment.get("start_coverage_complete",False):
            radii[station_arc < station_arc[measured_stations[0]]] = np.nan
        if not assessment.get("end_coverage_complete",False):
            radii[station_arc > station_arc[measured_stations[-1]]] = np.nan
        assessment["global_radius_fallback"] = False
    else:
        radii[:] = float(np.percentile(distances,65)) if len(distances) else np.nan
    if len(radii)>2:
        for _ in range(2):
            smoothed = radii.copy()
            local = (radii[:-2]+2.*radii[1:-1]+radii[2:])/4.
            supported = np.isfinite(local)
            smoothed[1:-1][supported] = local[supported]
            radii=smoothed
    return radii*normalization.output_scale


def _supported_radius_profile(points, path, normalization, evidence, assessment):
    """Measure local transverse sections from the exact accepted native support.

    A remote component cannot enter this function. Local longitudinal ownership
    and angular coverage prevent a nearby turn or one-sided wall being treated
    as a circular root. Missing sections never trigger whole-path extrapolation.
    """
    points, path = np.asarray(points, float), np.asarray(path, float)
    count = len(path)
    radii = np.full(count, np.nan)
    spacing = _positive_support_spacing(points)
    start = int(evidence.get("body_start_index", 0))
    interval_mask = np.asarray(evidence.get("supported_interval_mask", []), bool)[start:]
    station_mask = np.asarray(evidence.get("supported_station_mask", []), bool)[start:]
    arc_positions = np.asarray(evidence.get("support_arc_positions", []), float)
    station_indices = np.asarray(evidence.get("support_station_indices", []), int) - start
    assessment.update(sampling_spacing=spacing, assigned_support_count=len(points),
                      admissible_support_count=0, outside_body_support_count=0,
                      positive_radius_support_count=0, undersampled_station_count=count,
                      body_coverage_complete=False, global_radius_fallback=False,
                      radius_intervals_complete=False, radius_supported_length=0.,
                      radius_rejected_angular_station_count=0)
    if count < 2 or not len(points) or len(arc_positions) != len(points) or len(station_indices) != len(points):
        return radii
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    admissible = np.zeros(len(points), bool)
    positive = np.zeros(len(points), bool)
    for i in range(count):
        if len(station_mask) != count or not station_mask[i]:
            continue
        before, after = max(0, i-1), min(count-1, i+1)
        tangent = path[after] - path[before]
        norm = np.linalg.norm(tangent)
        if norm <= 1e-15:
            continue
        tangent /= norm
        # The transverse window follows native sampling, not the spacing of
        # stored centerline nodes. A coarse node at a bend must not collect
        # distant sections merely because its neighboring nodes are far away.
        half_width = 2.*(spacing or float(evidence.get("spacing", 0.)))
        local_ids = np.flatnonzero((np.abs(arc_positions-arc[i]) <= half_width)
                                   & (station_indices >= 0) & (station_indices < count))
        if not len(local_ids):
            continue
        delta = points[local_ids] - path[i]
        axial = delta @ tangent
        local_ids = local_ids[np.abs(axial) <= half_width]
        if len(local_ids) < 3:
            continue
        delta = points[local_ids] - path[i]
        transverse = delta - (delta @ tangent)[:, None]*tangent
        distance = np.linalg.norm(transverse, axis=1)
        nonzero = distance > 1e-15
        # Three points on one wall do not establish a transverse radius.
        helper = np.eye(3)[int(np.argmin(np.abs(tangent)))]
        u = np.cross(tangent, helper); u /= np.linalg.norm(u)
        v = np.cross(tangent, u)
        angles = np.sort(np.mod(np.arctan2(transverse[nonzero] @ v, transverse[nonzero] @ u), 2*np.pi))
        if len(angles) < 3 or np.max(np.diff(np.r_[angles, angles[0]+2*np.pi])) > np.pi + 1e-8:
            assessment["radius_rejected_angular_station_count"] += 1
            continue
        radii[i] = float(np.percentile(distance, 65))
        admissible[local_ids] = True
        positive[local_ids[nonzero]] = True
    measured = np.flatnonzero(np.isfinite(radii))
    assessment["undersampled_station_count"] = count-len(measured)
    # Fill only short sampling gaps inside intervals already supported by the
    # native evidence. No global fallback, endpoint extension, or gap bridge.
    for left, right in zip(measured[:-1], measured[1:]):
        if right-left <= 1:
            continue
        if (spacing is not None and arc[right]-arc[left] <= 4.*spacing
                and len(interval_mask) == count-1 and interval_mask[left:right].all()):
            radii[left:right+1] = np.interp(arc[left:right+1], arc[[left,right]], radii[[left,right]])
    radius_intervals = np.isfinite(radii[:-1]) & np.isfinite(radii[1:]) & (lengths > 0)
    if len(interval_mask) != len(radius_intervals):
        radius_intervals[:] = False
    else:
        radius_intervals &= interval_mask
    # The integration mask is separate from finite station values: two valid
    # sections on opposite sides of a native gap must not form a frustum.
    assessment["radius_interval_mask"] = radius_intervals.tolist()
    assessment["radius_supported_length"] = float(lengths[radius_intervals].sum())
    assessment["radius_intervals_complete"] = bool(len(radius_intervals) and radius_intervals.all())
    assessment["admissible_support_count"] = int(admissible.sum())
    assessment["positive_radius_support_count"] = int(positive.sum())
    assessment["outside_body_support_count"] = int((~admissible).sum())
    assessment["body_coverage_complete"] = bool(len(measured) and np.isfinite(radii[0]) and np.isfinite(radii[-1]))
    return radii*normalization.output_scale


def _frustum_measurements(points: np.ndarray, radii: np.ndarray,
                          interval_mask: np.ndarray | None = None) -> tuple[float, float]:
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
    supported = np.isfinite(area) & np.isfinite(volume)
    if interval_mask is not None:
        interval_mask = np.asarray(interval_mask, bool)
        if interval_mask.shape != supported.shape:
            return np.nan, np.nan
        supported &= interval_mask
    return (float(np.sum(area[supported])),float(np.sum(volume[supported]))) if np.any(supported) else (np.nan,np.nan)


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
    if np.any(labels <= -3):
        # Partition interface faces once across roots and nonroot objects.
        # This matches the separately exported nodule surface measurements.
        vertex_area = np.bincount(triangles.ravel(), weights=np.repeat(areas / 3, 3), minlength=len(vertices))
        return {int(label): float(vertex_area[labels == label].sum()) for label in np.unique(labels) if label >= 0}
    result: dict[int, float] = {}
    for label in np.unique(face_labels):
        if label >= 0:
            result[int(label)] = float(np.sum(areas[face_labels == label]))
    return result


def _measured_rows(traits: pd.DataFrame) -> pd.DataFrame:
    if "measurement_eligible" in traits:
        return traits[traits.measurement_eligible.eq(True)]
    return traits[~traits['measurement_excluded'].eq(True)] if 'measurement_excluded' in traits else traits


def _system_summary(
    traits: pd.DataFrame,
    *,
    mesh_metadata: dict,
    full_root_labels: np.ndarray | None,
) -> dict:
    from .architecture_traits import metric_eligible_mask, metric_status_counts
    inventory = traits
    quality_rows = traits[~traits.measurement_excluded.eq(True)] if "measurement_excluded" in traits else traits
    traits = _measured_rows(traits)
    laterals = traits[traits["root_order"] > 0]
    primary = traits[traits["root_order"] == 0]
    mesh_area = mesh_metadata.get("surface_area_source_units2")
    mesh_volume = mesh_metadata.get("absolute_volume_source_units3")
    exact_surface_available = mesh_area is not None
    whole_area = (
        float(mesh_area)
        if exact_surface_available
        else float(traits["surface_area"].sum(min_count=1))
    )
    exact_volume_available = bool(mesh_metadata.get("volume_reliable", False)) and mesh_volume is not None
    whole_volume = (
        float(mesh_volume)
        if exact_volume_available
        else float(traits["volume"].sum(min_count=1))
    )
    labels = None if full_root_labels is None else np.asarray(full_root_labels, dtype=int)
    contains_nodules = labels is not None and bool(np.any(labels <= -3))
    if contains_nodules:
        # Source-mesh totals include nodules and cannot be presented as root
        # totals after semantic separation. Retain the source in metadata only.
        whole_area = float(traits["surface_area"].sum(min_count=1))
        whole_volume = float(traits["volume"].sum(min_count=1))
        exact_surface_available = False
        exact_volume_available = False
    assigned_fraction = np.nan if labels is None or not len(labels) else float(np.mean(labels >= 0))
    uncertain_fraction = np.nan if labels is None or not len(labels) else float(np.mean(labels == -2))
    unassigned_fraction = np.nan if labels is None or not len(labels) else float(np.mean(labels == -1))
    length_counts = metric_status_counts(inventory, "length")
    volume_counts = metric_status_counts(inventory, "volume")
    unavailable_length_count, partial_length_count = length_counts["unavailable"], length_counts["partial"]
    unavailable_volume_count, partial_volume_count = volume_counts["unavailable"], volume_counts["partial"]
    def metric_sum(metric, order=None):
        rows = inventory if order is None else inventory[inventory.root_order.eq(0) if order == 0 else inventory.root_order.gt(0)]
        selected = rows[metric_eligible_mask(rows, metric)]
        return float(selected[metric].sum(min_count=1)) if len(rows) else 0.
    if "volume_measurement_status" in inventory:
        whole_volume = metric_sum("volume")
        whole_area = metric_sum("surface_area")
        exact_volume_available = False
    result = {
        "root_count_total": int(len(traits)),
        "root_inventory_count": length_counts["inventory"],
        "noise_excluded_root_count": length_counts["excluded"],
        **({"noise_excluded_from_root_totals": True} if mesh_metadata.get("noise_excluded_from_totals") else {}),
        **({"nodule_vertex_fraction": float(np.mean(labels <= -3)),
            "nodule_excluded_from_root_totals": True,
            "source_surface_area_including_nodules": mesh_area,
            "source_volume_including_nodules": mesh_volume} if contains_nodules else {}),
        "lateral_root_count_total": int(len(laterals)),
        "maximum_root_order": int(laterals["root_order"].max()) if len(laterals) else 0,
        "length_unit": "mesh_unit",
        "area_unit": "mesh_unit^2",
        "volume_unit": "mesh_unit^3",
        "root_system_length": float(traits["length"].sum(min_count=1)),
        "root_system_length_complete": unavailable_length_count == 0 and partial_length_count == 0,
        "root_system_length_unavailable_root_count": unavailable_length_count,
        "root_system_length_partial_root_count": partial_length_count,
        "primary_root_length": metric_sum("length", 0),
        "lateral_root_length_sum": metric_sum("length", 1),
        "root_system_surface_area": whole_area,
        "root_system_surface_area_method": (
            ("noise_filtered_mesh_triangle_area" if mesh_metadata.get("noise_excluded_from_totals")
             else "full_mesh_triangle_area") if exact_surface_available else "sum_per_root_surface_estimates"
        ),
        "primary_root_surface_area": metric_sum("surface_area", 0),
        "lateral_root_surface_area_sum": metric_sum("surface_area", 1),
        "root_system_volume": whole_volume,
        "root_system_volume_estimate_complete": unavailable_volume_count == 0 and partial_volume_count == 0,
        "root_system_volume_estimate_unavailable_root_count": unavailable_volume_count,
        "root_system_volume_estimate_partial_root_count": partial_volume_count,
        "radius_profile_method": "native-interval-transverse-support-v2" if "radius_profile_method" in traits else "legacy_segment_distance",
        "root_system_volume_method": (
            "closed_manifold_component_volume" if exact_volume_available else "sum_centerline_frustum_estimates"
        ),
        "root_system_volume_reliable": bool(exact_volume_available),
        "root_system_volume_reliability_reason": mesh_metadata.get("volume_reliability_reason", "frustum_estimate"),
        "primary_root_volume_estimate": metric_sum("volume", 0),
        "lateral_root_volume_estimate_sum": metric_sum("volume", 1),
        "assigned_vertex_fraction": assigned_fraction,
        "uncertain_vertex_fraction": uncertain_fraction,
        "unassigned_vertex_fraction": unassigned_fraction,
    }
    for metric in MEASUREMENT_METRICS:
        if metric not in inventory:
            continue
        counts = metric_status_counts(inventory, metric)
        for status in ("complete", "partial", "unavailable"):
            result[f"{metric}_{status}_root_count"] = counts[status]
        result[f"{metric}_measured_root_count"] = counts["complete"] + counts["partial"]
    return result


def _chord_length(points: np.ndarray) -> float:
    return 0.0 if len(points) < 2 else float(np.linalg.norm(points[-1] - points[0]))


def _arc_window_vector(
    path: np.ndarray,
    *,
    anchor_index: int,
    requested_window: float,
    from_start: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return a sampling-invariant vector over a window in output coordinates.

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
