from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .export import export_results
from .geometry import mean_nearest_neighbor_distance, normalize_unit_box
from .io import load_root_geometry
from .lateral import (
    backtrace_to_primary,
    estimate_parent_radius_profile,
    extend_lateral_tip,
    find_lateral_starting_points,
    grow_lateral_candidates,
    is_parent_tracking_candidate,
    reduce_similar_paths,
    select_non_overlapping_paths,
)
from .primary import (
    GRAVITY,
    estimate_primary_path,
    rank_primary_candidates,
    refine_primary_centerline,
    tangent_plane_primary_segmentation,
)
from .topology import apply_hierarchy_corrections, repair_root_hierarchy, validate_root_tree
from .traits import compute_traits
from .types import Normalization, PointCloudData, PrimaryCandidate, RootPath, TopologyReport
from .runtime import worker_thread_limit, worker_threads
from .visualize import save_angle_front_views, save_overview_plot


LOGGER = logging.getLogger(__name__)
MIN_PIPELINE_POINTS = 20
ABOVE_BASE_TOLERANCE_NORMALIZED = 1e-9


class AnalysisCancelled(RuntimeError):
    """Raised when the desktop GUI requests cooperative cancellation."""


def _lateral_start_distance_limits(
    parent_radius_profile: np.ndarray,
    d_bar: float,
) -> np.ndarray:
    """Build a sampling-scaled seed envelope around one parent centerline.

    The diameter bridge recovers lateral stems hidden by collar completion on
    ordinary-width parents.  A strongly flared parent disables that bridge as
    a whole because isolated node-by-node decisions can still join unrelated
    collar surfaces into a competing seed cluster.
    """

    radii = np.asarray(parent_radius_profile, dtype=float)
    if radii.ndim != 1 or len(radii) == 0 or not np.all(np.isfinite(radii)):
        raise ValueError("parent_radius_profile must be a non-empty finite vector")
    spacing = float(d_bar)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("d_bar must be positive and finite")
    base_distance_limit = np.maximum(
        np.full(len(radii), max(14.0 * spacing, 0.012)),
        radii + 9.0 * spacing,
    )
    diameter_bridge = 2.0 * radii + 2.0 * spacing
    strongly_flared_parent = bool(np.quantile(radii, 0.99) > 18.0 * spacing)
    ordinary_width = (
        radii <= 16.0 * spacing
        if not strongly_flared_parent
        else np.zeros(len(radii), dtype=bool)
    )
    return np.where(
        ordinary_width,
        np.maximum(base_distance_limit, diameter_bridge),
        base_distance_limit,
    )


@dataclass
class PipelineConfig:
    input_path: Path
    output_dir: Path
    start: tuple[float, float, float] | None = None
    end: tuple[float, float, float] | None = None
    endpoint_file: Path | None = None
    auto_endpoints: str | None = None
    soil_z: float | None = None
    primary_guides: tuple[tuple[float, float, float], ...] = ()
    guide_file: Path | None = None
    correction_file: Path | None = None
    sample_points: int | None = None
    graph_k: int = 14
    lateral_max_paths: int | None = None
    max_root_order: int = 3
    gravity: tuple[float, float, float] = (0.0, 0.0, -1.0)
    runtime_limit_minutes: float = 30.0
    minimum_retained_fraction: float = 0.25
    tip_vector_window_mesh_units: float = 2.0
    worker_threads: int | None = None
    random_seed: int = 42


@dataclass
class PipelineResult:
    output_dir: Path
    point_count: int
    d_bar: float
    primary_path: np.ndarray
    lateral_paths: list[RootPath]
    primary_mask: np.ndarray
    lateral_labels: np.ndarray
    normalization: Normalization
    lateral_start_count: int
    full_root_labels: np.ndarray | None = None
    above_base_mask: np.ndarray | None = None
    full_above_base_mask: np.ndarray | None = None
    primary_candidates: list[PrimaryCandidate] | None = None
    topology_report: TopologyReport | None = None
    traits: pd.DataFrame | None = None


def run_pipeline(
    config: PipelineConfig,
    *,
    preloaded_cloud: PointCloudData | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    pause_check: Callable[[], bool] | None = None,
) -> PipelineResult:
    """Run one analysis with an isolated per-job SciPy worker limit."""

    with worker_thread_limit(config.worker_threads):
        return _run_pipeline_impl(
            config,
            preloaded_cloud=preloaded_cloud,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            pause_check=pause_check,
        )


def _run_pipeline_impl(
    config: PipelineConfig,
    *,
    preloaded_cloud: PointCloudData | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    pause_check: Callable[[], bool] | None = None,
) -> PipelineResult:
    """Run the soybean root skeletonization and trait workflow.

    The selected primary is always order 0.  Candidate lateral paths are repaired
    into a rooted acyclic hierarchy before orders and traits are calculated.
    """
    _validate_config(config)
    cooperate = lambda: _cooperate(cancel_check, pause_check)
    timings: dict[str, float] = {}
    pipeline_started = time.perf_counter()
    stage_started = pipeline_started

    def checkpoint(completed_stage: str, next_stage: str, fraction: float) -> None:
        nonlocal stage_started
        now = time.perf_counter()
        timings[completed_stage] = float(now - stage_started)
        stage_started = now
        cooperate()
        _report_progress(progress_callback, next_stage, fraction)

    cooperate()
    _report_progress(progress_callback, "Preparing analysis", 0.02)
    np.random.seed(config.random_seed)
    LOGGER.info("Loading input geometry: %s", config.input_path)
    if preloaded_cloud is None:
        cloud = load_root_geometry(
            config.input_path,
            sample_points=config.sample_points,
            random_seed=config.random_seed,
            runtime_limit_seconds=config.runtime_limit_minutes * 60.0,
            minimum_retained_fraction=config.minimum_retained_fraction,
        )
    else:
        cloud = preloaded_cloud
        LOGGER.info("Using %d points already loaded by the desktop GUI", len(cloud.points))
    checkpoint("load_geometry", "Point cloud ready", 0.12)
    if len(cloud.points) < MIN_PIPELINE_POINTS:
        raise ValueError(f"Too few points for skeletonization: found {len(cloud.points)}, need at least {MIN_PIPELINE_POINTS}")
    normalized, base_normalization = normalize_unit_box(cloud.points)
    # Geometry is normalized internally for numerical stability, then restored
    # to the source mesh coordinate system for every reported trait.  Physical
    # calibration is intentionally disabled until its contract is reinstated.
    normalization = Normalization(base_normalization.minimum, base_normalization.scale)
    d_bar = mean_nearest_neighbor_distance(normalized)
    LOGGER.info("Loaded %d points; d_bar=%g", len(normalized), d_bar)
    checkpoint("normalization", "Detecting primary root", 0.20)

    coarse_primary, primary_candidates = _resolve_primary_path(
        cloud.points,
        normalized,
        normalization,
        d_bar,
        config,
        cooperate=cooperate,
    )
    selected_base = np.asarray(coarse_primary.points[0], dtype=float).copy()
    direction_index = max(
        1,
        min(len(coarse_primary.points) - 1, int(np.ceil(0.02 * len(coarse_primary.points)))),
    )
    base_tipward_direction = coarse_primary.points[direction_index] - selected_base
    base_tolerance = ABOVE_BASE_TOLERANCE_NORMALIZED
    base_collar_neighborhood_radius = 36.0 * d_bar
    above_base_mask = _selected_base_exclusion_mask(
        normalized,
        selected_base,
        base_tipward_direction,
        gravity=np.asarray(config.gravity, dtype=float),
        collar_neighborhood_radius=base_collar_neighborhood_radius,
        tolerance=base_tolerance,
    )
    checkpoint("primary_detection", "Estimated primary-root path", 0.32)
    primary_mask = tangent_plane_primary_segmentation(
        normalized,
        coarse_primary.points,
        d_bar=d_bar,
        cooperate=cooperate,
    )
    primary_mask[above_base_mask] = False
    refined_primary_points = refine_primary_centerline(
        normalized,
        primary_mask,
        coarse_primary.points,
        d_bar=d_bar,
        fit_circular_cross_sections=True,
        cooperate=cooperate,
    )
    # Re-segment around the centred path.  The first pass starts from a path on
    # the mesh surface, so it can only see the near wall of a cylindrical
    # collar.  A centred second pass restores the opposite wall before lateral
    # tracing and prevents those primary points from being claimed as laterals.
    primary_mask = tangent_plane_primary_segmentation(
        normalized,
        refined_primary_points,
        d_bar=d_bar,
        complete_cross_section=True,
        cooperate=cooperate,
    )
    primary_mask[above_base_mask] = False
    refined_primary_points = refine_primary_centerline(
        normalized,
        primary_mask,
        refined_primary_points,
        d_bar=d_bar,
        fit_circular_cross_sections=True,
        cooperate=cooperate,
    )
    primary = RootPath(
        root_id="primary",
        points=refined_primary_points,
        node_indices=coarse_primary.node_indices,
        order=0,
        parent_id="",
        confidence=coarse_primary.confidence,
        qc_flags=list(coarse_primary.qc_flags),
        score_components=dict(coarse_primary.score_components),
    )
    LOGGER.info(
        "Excluded %d/%d analysis points above the selected base",
        int(np.count_nonzero(above_base_mask)),
        len(above_base_mask),
    )
    LOGGER.info("Primary segmentation assigned %d/%d points", int(primary_mask.sum()), len(primary_mask))
    checkpoint("primary_segmentation", "Segmented and refined primary root", 0.46)

    selected, lateral_start_count, candidate_count, order_counts = _trace_lateral_orders(
        normalized,
        primary.points,
        primary_mask,
        d_bar,
        max_root_order=config.max_root_order,
        max_paths=config.lateral_max_paths,
        excluded_mask=above_base_mask,
        cooperate=cooperate,
    )
    checkpoint("lateral_tracing", "Repairing root topology", 0.70)
    if lateral_start_count == 0:
        LOGGER.warning("No lateral root starting points detected; exporting primary-root-only results.")
    selected, topology_report = repair_root_hierarchy(primary.points, selected, d_bar=d_bar)
    if config.correction_file is not None:
        selected = apply_hierarchy_corrections(
            primary.points,
            selected,
            config.correction_file,
            normalization=normalization,
        )
    topology_errors = validate_root_tree(selected)
    if topology_errors:
        raise RuntimeError("Root topology validation failed: " + "; ".join(topology_errors))
    checkpoint("topology_repair", "Assigning root vertices", 0.76)
    lateral_labels = _assign_lateral_points(
        normalized,
        selected,
        primary_mask,
        d_bar,
        excluded_mask=above_base_mask,
    )
    full_normalized = normalization.transform_points(cloud.export_points)
    full_above_base_mask = _selected_base_exclusion_mask(
        full_normalized,
        selected_base,
        base_tipward_direction,
        gravity=np.asarray(config.gravity, dtype=float),
        collar_neighborhood_radius=base_collar_neighborhood_radius,
        tolerance=base_tolerance,
    )
    full_root_labels = _assign_full_root_labels(
        full_normalized,
        primary.points,
        selected,
        d_bar=d_bar,
        excluded_mask=full_above_base_mask,
    )
    if cloud.analysis_indices is not None and len(cloud.analysis_indices) == len(normalized):
        analysis_root_labels = _analysis_root_labels(primary_mask, lateral_labels)
        full_root_labels[np.asarray(cloud.analysis_indices, dtype=int)] = analysis_root_labels
    full_root_labels[full_above_base_mask] = -1
    checkpoint("point_assignment", "Computing root traits", 0.82)
    traits = compute_traits(
        primary.points,
        selected,
        normalized,
        primary_mask,
        lateral_labels,
        normalization,
        lateral_start_count=lateral_start_count,
        gravity=np.asarray(config.gravity, dtype=float),
        full_points=cloud.export_points,
        triangles=cloud.triangles,
        full_root_labels=full_root_labels,
        mesh_metadata=cloud.source_metadata,
        primary_confidence=primary.confidence,
        primary_qc_flags=primary.qc_flags,
        tip_vector_window=config.tip_vector_window_mesh_units,
    )
    checkpoint("trait_measurement", "Rendering validation figures", 0.87)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_overview_plot(config.output_dir / "overview.png", normalized, primary_mask, lateral_labels, primary.points, selected)
    save_angle_front_views(
        config.output_dir,
        normalized,
        primary_mask,
        lateral_labels,
        primary.points,
        selected,
        traits,
        gravity=np.asarray(config.gravity, dtype=float),
    )
    checkpoint("validation_figures", "Exporting results", 0.92)
    order_counts = {
        int(order): int(count)
        for order, count in traits.loc[traits["root_order"] > 0, "root_order"].value_counts().sort_index().items()
    }
    metadata = {
        "source": str(config.input_path),
        "algorithm_reference": "Zhou et al. 2025, Computers and Electronics in Agriculture, DOI 10.1016/j.compag.2025.110890",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "point_count": len(normalized),
        "full_resolution_point_count": len(cloud.export_points),
        "d_bar_normalized": d_bar,
        "normalization_minimum": normalization.minimum,
        "normalization_scale": normalization.scale,
        "coordinate_unit": "mesh_unit",
        "output_length_unit": "mesh_unit",
        "output_area_unit": "mesh_unit^2",
        "output_volume_unit": "mesh_unit^3",
        "physical_unit_conversion_applied": False,
        "gravity_vector": list(config.gravity),
        "source_geometry": cloud.source_metadata,
        "lateral_start_count": lateral_start_count,
        "candidate_lateral_count": candidate_count,
        "selected_lateral_count": len(selected),
        "selected_order_counts": order_counts,
        "primary_detection_method": _primary_method(config),
        "point_assignment": _point_assignment_summary(
            full_root_labels,
            full_above_base_mask,
            base_point_source=normalization.inverse_points(selected_base[None, :])[0],
            base_tipward_direction=base_tipward_direction,
            gravity=np.asarray(config.gravity, dtype=float),
            collar_neighborhood_radius=base_collar_neighborhood_radius,
            base_tolerance=base_tolerance,
            d_bar=d_bar,
            analysis_above_base_count=int(np.count_nonzero(above_base_mask)),
        ),
        "primary_candidates": [_candidate_metadata(candidate, normalization) for candidate in primary_candidates],
        "topology_report": topology_report.__dict__,
        "stage_timings_seconds": timings,
    }
    if config.correction_file is not None:
        correction_payload = json.loads(Path(config.correction_file).read_text(encoding="utf-8"))
        correction_rows = correction_payload.get("roots", [])
        metadata["hierarchy_correction"] = {
            "applied": True,
            "file": str(config.correction_file),
            "root_ids": [str(row.get("root_id")) for row in correction_rows],
            "removed_root_ids": [
                str(row.get("root_id"))
                for row in correction_rows
                if row.get("valid", True) is False
            ],
            "manually_changed_root_ids": [
                path.root_id
                for path in selected
                if "manual_correction" in path.qc_flags
            ],
        }
    export_results(
        config.output_dir,
        cloud.points,
        primary.points,
        selected,
        primary_mask,
        lateral_labels,
        traits,
        normalization,
        metadata,
        full_points=cloud.export_points,
        triangles=cloud.triangles,
        full_root_labels=full_root_labels,
        topology_report=topology_report,
    )
    checkpoint("export", "Finalizing metadata", 0.98)
    timings["total"] = float(time.perf_counter() - pipeline_started)
    _update_metadata_timings(config.output_dir / "metadata.json", timings)
    cooperate()
    _report_progress(progress_callback, "Analysis complete", 1.0)
    return PipelineResult(
        output_dir=config.output_dir,
        point_count=len(normalized),
        d_bar=d_bar,
        primary_path=primary.points,
        lateral_paths=selected,
        primary_mask=primary_mask,
        lateral_labels=lateral_labels,
        normalization=normalization,
        lateral_start_count=lateral_start_count,
        full_root_labels=full_root_labels,
        above_base_mask=above_base_mask,
        full_above_base_mask=full_above_base_mask,
        primary_candidates=primary_candidates,
        topology_report=topology_report,
        traits=traits,
    )


def _report_progress(
    callback: Callable[[str, float], None] | None,
    stage: str,
    fraction: float,
) -> None:
    if callback is not None:
        callback(stage, float(np.clip(fraction, 0.0, 1.0)))


def _raise_if_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise AnalysisCancelled("Analysis cancelled by the user.")


def _cooperate(
    cancel_check: Callable[[], bool] | None,
    pause_check: Callable[[], bool] | None,
) -> None:
    _raise_if_cancelled(cancel_check)
    while pause_check is not None and pause_check():
        time.sleep(0.10)
        _raise_if_cancelled(cancel_check)


def _trace_lateral_orders(
    points: np.ndarray,
    primary_path: np.ndarray,
    primary_mask: np.ndarray,
    d_bar: float,
    max_root_order: int,
    max_paths: int | None,
    excluded_mask: np.ndarray | None = None,
    cooperate: Callable[[], None] | None = None,
) -> tuple[list[RootPath], int, int, dict[int, int]]:
    """Trace lateral roots iteratively from parent skeletons.

    Order 1 uses a broad paper-inspired junction search. Higher orders use a
    tighter percentile threshold, but every detected start is retained until
    parameter variants have been collapsed by parent/start identity.  Any root
    count limit is therefore explicit through ``max_paths`` rather than a
    hidden per-parent or pre-reduction truncation.
    """
    selected_all: list[RootPath] = []
    excluded = _coerce_exclusion_mask(excluded_mask, len(points))
    occupied_mask = np.asarray(primary_mask, dtype=bool) | excluded
    parent_paths: list[tuple[str, np.ndarray]] = [("primary", primary_path)]
    labels = np.zeros(len(points), dtype=int)
    extension_tree = cKDTree(points)
    total_starts = 0
    total_candidates = 0
    order_counts: dict[int, int] = {}

    for order in range(1, max(1, int(max_root_order)) + 1):
        if cooperate is not None:
            cooperate()
        if max_paths is not None and len(selected_all) >= max_paths:
            break
        order_candidates: list[RootPath] = []
        order_starts = 0
        order_candidate_count = 0
        order_parent_tracking_rejections = 0
        for parent_id, parent_path in parent_paths:
            if cooperate is not None:
                cooperate()
            closest_fraction = 0.03 if order == 1 else 0.01
            if parent_id == "primary":
                parent_support_mask = np.asarray(primary_mask, dtype=bool) & ~excluded
            else:
                parent_position = next(
                    (index for index, selected_path in enumerate(selected_all) if selected_path.root_id == parent_id),
                    None,
                )
                parent_support_mask = (
                    labels == int(parent_position) + 1
                    if parent_position is not None
                    else np.zeros(len(points), dtype=bool)
                )
            parent_support_points = points[parent_support_mask]
            if len(parent_support_points) < 3:
                parent_support_points = np.asarray(parent_path, dtype=float)
            parent_radius_profile = estimate_parent_radius_profile(
                parent_path,
                parent_support_points,
                d_bar,
            )
            # Attachment distance must follow the local parent thickness.  A
            # fixed centreline gate rejects valid branches on a thick/flared
            # collar even when their surface touches the parent.  Add a
            # sampling-scale junction margin to the measured radius.  A local
            # diameter envelope is also retained to bridge a proximal lateral
            # stem absorbed by the completed collar mask.  Restrict that bridge
            # to ordinary-width sections: on a strongly flared collar a full
            # diameter reaches unrelated surfaces and creates competing seeds.
            # Every term is expressed in normalized mesh geometry or sampling
            # spacing; none is a calibrated physical distance.
            parent_distance_limit = _lateral_start_distance_limits(
                parent_radius_profile,
                d_bar,
            )
            starts = find_lateral_starting_points(
                points,
                occupied_mask,
                parent_path,
                closest_fraction=closest_fraction,
                min_cluster_size=4,
                max_parent_distance=parent_distance_limit,
                minimum_branch_angle_degrees=18.0,
                exclude_parent_tip_fraction=0.12 if order > 1 else 0.0,
            )
            order_starts += len(starts)
            max_steps = 80 if order == 1 else 35
            candidates = grow_lateral_candidates(
                points,
                starts,
                parent_path,
                occupied_mask,
                d_bar=d_bar,
                max_steps=max_steps,
                parent_radius_profile=parent_radius_profile,
                cooperate=cooperate,
            )
            order_candidate_count += len(candidates)
            evaluated_groups: dict[int | str, list[tuple[RootPath, bool]]] = {}
            for candidate in candidates:
                candidate.order = order
                candidate.parent_id = parent_id
                candidate.parent_points = parent_path
                rejected, tracking_metrics = is_parent_tracking_candidate(
                    candidate,
                    parent_path,
                    parent_radius_profile,
                    d_bar,
                )
                candidate.score_components.update(tracking_metrics)
                if rejected:
                    order_parent_tracking_rejections += 1
                start_key: int | str = (
                    candidate.start_index
                    if candidate.start_index is not None
                    else candidate.root_id
                )
                evaluated_groups.setdefault(start_key, []).append((candidate, rejected))
            accepted_candidates: list[RootPath] = []
            for group in evaluated_groups.values():
                escaping = [candidate for candidate, rejected in group if not rejected]
                if escaping:
                    accepted_candidates.extend(escaping)
                else:
                    # Keep an all-tracking start provisionally so genuine roots
                    # joined at that collar region can still be discovered in
                    # the next pass.  The selected tracking path is removed and
                    # its children promoted after tracing all requested orders.
                    accepted_candidates.extend(candidate for candidate, _ in group)
            order_candidates.extend(accepted_candidates)
        total_starts += order_starts
        total_candidates += order_candidate_count
        if not order_candidates:
            LOGGER.info(
                "No candidate paths for root order %d after rejecting %d parent-tracking variants",
                order,
                order_parent_tracking_rejections,
            )
            break
        reduced = reduce_similar_paths(order_candidates)
        remaining = None if max_paths is None else max(0, max_paths - len(selected_all))
        selected = select_non_overlapping_paths(reduced, points, d_bar=d_bar, max_paths=remaining)
        refined: list[RootPath] = []
        for path in selected:
            path.order = order
            path.root_id = f"order{order}_{len(selected_all) + len(refined) + 1:03d}"
            path.parent_points = path.parent_points if path.parent_points is not None else primary_path
            path.parent_id = path.parent_id or "primary"
            traced_paths = backtrace_to_primary(
                [path],
                path.parent_points,
                primary_points=path.parent_points,
            )
            for traced in traced_paths:
                local_mask = np.zeros(len(points), dtype=bool)
                covered = np.asarray(sorted(traced.covered_indices), dtype=int)
                covered = covered[(covered >= 0) & (covered < len(points))]
                local_mask[covered] = True
                if np.count_nonzero(local_mask) >= 30:
                    centered = refine_primary_centerline(
                        points,
                        local_mask,
                        traced.points,
                        d_bar=d_bar,
                        max_stations=240,
                        min_slice_points=6,
                        cooperate=cooperate,
                    )
                    if len(centered) >= 2:
                        traced.points = centered
                refined.append(traced)
        if not refined:
            break
        provisional_labels = _assign_lateral_points(
            points,
            selected_all + refined,
            primary_mask,
            d_bar,
            excluded_mask=excluded,
        )
        continuation_blocked = (
            np.asarray(primary_mask, dtype=bool)
            | excluded
            | (provisional_labels != 0)
        )
        extended_count = 0
        for traced in refined:
            if traced.score_components.get("parent_tracking_rejected", 0.0) > 0.0:
                continue
            extend_lateral_tip(
                points,
                traced,
                continuation_blocked,
                d_bar,
                point_tree=extension_tree,
                cooperate=cooperate,
            )
            if traced.score_components.get("tip_continuation_accepted", 0.0) <= 0.0:
                continue
            extended_count += 1
            covered = np.asarray(sorted(traced.covered_indices), dtype=int)
            covered = covered[(covered >= 0) & (covered < len(points))]
            continuation_blocked[covered] = True
            local_mask = np.zeros(len(points), dtype=bool)
            local_mask[covered] = True
            if np.count_nonzero(local_mask) >= 30:
                centered = refine_primary_centerline(
                    points,
                    local_mask,
                    traced.points,
                    d_bar=d_bar,
                    max_stations=240,
                    min_slice_points=6,
                    cooperate=cooperate,
                )
                if len(centered) >= 2:
                    traced.points = centered
        selected_all.extend(refined)
        order_counts[order] = len(refined)
        labels = _assign_lateral_points(
            points,
            selected_all,
            primary_mask,
            d_bar,
            excluded_mask=excluded,
        )
        occupied_mask = np.asarray(primary_mask, dtype=bool) | excluded | (labels > 0)
        parent_paths = [(path.root_id, path.points) for path in refined]
        LOGGER.info(
            "Selected %d order-%d lateral paths (%d tip-extended) from %d starts and %d candidates; flagged %d parent-tracking variants",
            len(refined),
            order,
            extended_count,
            order_starts,
            order_candidate_count,
            order_parent_tracking_rejections,
        )
    selected_all = _prune_parent_tracking_paths(selected_all)
    order_counts = {
        order: sum(int(path.order) == order for path in selected_all)
        for order in range(1, max(1, int(max_root_order)) + 1)
    }
    order_counts = {order: count for order, count in order_counts.items() if count > 0}
    return selected_all, total_starts, total_candidates, order_counts


def _prune_parent_tracking_paths(paths: list[RootPath]) -> list[RootPath]:
    rejected = {
        path.root_id: path
        for path in paths
        if path.score_components.get("parent_tracking_rejected", 0.0) > 0.0
    }
    if not rejected:
        return paths
    kept = [path for path in paths if path.root_id not in rejected]
    for path in kept:
        promoted = 0
        seen: set[str] = set()
        while path.parent_id in rejected and path.parent_id not in seen:
            seen.add(path.parent_id)
            removed_parent = rejected[path.parent_id]
            path.parent_id = removed_parent.parent_id
            path.parent_points = removed_parent.parent_points
            path.order = max(1, int(path.order) - 1)
            promoted += 1
        if promoted:
            path.score_components["parent_tracking_ancestor_promotions"] = float(promoted)
            if "parent_tracking_parent_removed" not in path.qc_flags:
                path.qc_flags.append("parent_tracking_parent_removed")
    return kept

def _validate_config(config: PipelineConfig) -> None:
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {output_dir}. Choose a fresh directory to avoid stale or mixed results."
        )
    if config.max_root_order < 1:
        raise ValueError(f"max_root_order must be at least 1; got {config.max_root_order}")
    if config.sample_points not in (None, 0) and int(config.sample_points) < MIN_PIPELINE_POINTS:
        raise ValueError(f"sample_points must be at least {MIN_PIPELINE_POINTS} or 0/None; got {config.sample_points}")
    if config.runtime_limit_minutes <= 0:
        raise ValueError("runtime_limit_minutes must be positive")
    if not 0 < config.minimum_retained_fraction <= 1:
        raise ValueError("minimum_retained_fraction must be in (0, 1]")
    if (
        config.tip_vector_window_mesh_units <= 0
        or not np.isfinite(config.tip_vector_window_mesh_units)
    ):
        raise ValueError("tip_vector_window_mesh_units must be a positive finite number")
    gravity = np.asarray(config.gravity, dtype=float)
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)) or np.linalg.norm(gravity) <= 1e-12:
        raise ValueError("gravity must contain three finite values and have non-zero length")
    if config.auto_endpoints not in (None, "scored", "z", "pca"):
        raise ValueError("auto_endpoints must be one of: scored, z, pca")
    if config.start is not None and config.end is None:
        raise ValueError("Both --start and --end must be provided together.")
    if config.end is not None and config.start is None:
        raise ValueError("Both --start and --end must be provided together.")
    if config.worker_threads is not None and int(config.worker_threads) < 1:
        raise ValueError("worker_threads must be a positive integer when provided")


def _resolve_primary_path(
    original_points: np.ndarray,
    normalized_points: np.ndarray,
    normalization: Normalization,
    d_bar: float,
    config: PipelineConfig,
    *,
    cooperate: Callable[[], None] | None = None,
) -> tuple[RootPath, list[PrimaryCandidate]]:
    gravity = np.asarray(config.gravity, dtype=float)
    gravity /= np.linalg.norm(gravity)
    guides = _read_primary_guides(config)
    normalized_guides = (
        normalization.transform_points(guides) if len(guides) else np.empty((0, 3), dtype=float)
    )
    manual = config.endpoint_file is not None or (config.start is not None and config.end is not None)
    candidates: list[PrimaryCandidate] = []
    if manual:
        if config.endpoint_file is not None:
            start_original, end_original = read_endpoint_file(config.endpoint_file)
        else:
            start_original = np.asarray(config.start, dtype=float)
            end_original = np.asarray(config.end, dtype=float)
        start, end = _validate_and_transform_endpoints(start_original, end_original, normalization)
        start, end = _orient_collar_to_tip(start, end, gravity)
        path = estimate_primary_path(
            normalized_points,
            start,
            end,
            d_bar=d_bar,
            graph_k=config.graph_k,
            waypoints=normalized_guides,
            cooperate=cooperate,
        )
        path.confidence = 1.0
        path.score_components = {
            "manual_endpoints": 1.0,
            "manual_section_constraints": float(bool(len(normalized_guides))),
        }
        return path, candidates

    method = config.auto_endpoints or "scored"
    if method == "scored":
        up = -gravity
        soil_level = None
        if config.soil_z is not None:
            soil_point = np.array([0.0, 0.0, float(config.soil_z)])
            soil_level = float(normalization.transform_points(soil_point[None, :])[0] @ up)
        candidates = rank_primary_candidates(
            normalized_points,
            d_bar,
            gravity=gravity,
            soil_level=soil_level,
            graph_k=config.graph_k,
            cooperate=cooperate,
        )
        best = candidates[0]
        if len(normalized_guides):
            path = estimate_primary_path(
                normalized_points,
                best.start,
                best.end,
                d_bar=d_bar,
                graph_k=config.graph_k,
                waypoints=normalized_guides,
                cooperate=cooperate,
            )
        else:
            path = RootPath(
                root_id="primary",
                points=best.path.copy(),
                order=0,
                parent_id="",
            )
        path.confidence = best.confidence
        path.qc_flags = list(best.qc_flags)
        path.score_components = dict(best.components)
        return path, candidates

    if method == "z":
        height = normalized_points @ (-gravity)
        start = normalized_points[int(np.argmax(height))]
        end = normalized_points[int(np.argmin(height))]
    else:
        centered = normalized_points - normalized_points.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
        scores = centered @ axis
        first = normalized_points[int(np.argmin(scores))]
        second = normalized_points[int(np.argmax(scores))]
        start, end = _orient_collar_to_tip(first, second, gravity)
    path = estimate_primary_path(
        normalized_points,
        start,
        end,
        d_bar=d_bar,
        graph_k=config.graph_k,
        waypoints=normalized_guides,
        cooperate=cooperate,
    )
    path.confidence = 0.35
    path.qc_flags = ["unscored_automatic_primary"]
    path.score_components = {f"{method}_extrema": 1.0}
    return path, candidates


def _read_primary_guides(config: PipelineConfig) -> np.ndarray:
    rows = [np.asarray(row, dtype=float) for row in config.primary_guides]
    if config.guide_file is not None:
        path = Path(config.guide_file)
        if not path.exists():
            raise FileNotFoundError(f"Primary guide file does not exist: {path}")
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload = payload.get("guides", payload) if isinstance(payload, dict) else payload
            rows.extend(np.asarray(payload, dtype=float))
        elif path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            lower = {column.lower(): column for column in frame.columns}
            if not {"x", "y", "z"}.issubset(lower):
                raise ValueError("Guide CSV must contain x, y, z columns")
            rows.extend(frame[[lower["x"], lower["y"], lower["z"]]].to_numpy(float))
        else:
            values = []
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                numbers = []
                for token in line.replace(",", " ").split():
                    try:
                        numbers.append(float(token))
                    except ValueError:
                        continue
                if len(numbers) >= 3:
                    values.append(numbers[:3])
            rows.extend(values)
    if not rows:
        return np.empty((0, 3), dtype=float)
    guides = np.asarray(rows, dtype=float)
    if guides.ndim != 2 or guides.shape[1] != 3 or not np.all(np.isfinite(guides)):
        raise ValueError("Primary guide points must be finite XYZ triples")
    return guides


def _orient_collar_to_tip(
    start: np.ndarray,
    end: np.ndarray,
    gravity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Collar-to-tip should have a positive projection along gravity.
    return (start, end) if np.dot(end - start, gravity) >= 0 else (end, start)


def _primary_method(config: PipelineConfig) -> str:
    if config.endpoint_file is not None or (config.start is not None and config.end is not None):
        return "manual_endpoints_with_optional_sections"
    if config.soil_z is not None:
        return "scored_candidates_with_manual_soil_line"
    return config.auto_endpoints or "scored"


def _candidate_metadata(candidate: PrimaryCandidate, normalization: Normalization) -> dict:
    return {
        "rank": candidate.rank,
        "score": candidate.score,
        "confidence": candidate.confidence,
        "start": normalization.inverse_points(candidate.start[None, :])[0],
        "end": normalization.inverse_points(candidate.end[None, :])[0],
        "components": candidate.components,
        "qc_flags": candidate.qc_flags,
    }


def _resolve_endpoints(
    original_points: np.ndarray,
    normalized_points: np.ndarray,
    normalization: Normalization,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if config.endpoint_file is not None:
        start, end = read_endpoint_file(config.endpoint_file)
        transformed = _validate_and_transform_endpoints(start, end, normalization)
        return _orient_collar_to_tip(*transformed, np.asarray(config.gravity, dtype=float))
    if config.start is not None and config.end is not None:
        transformed = _validate_and_transform_endpoints(np.asarray(config.start, dtype=float), np.asarray(config.end, dtype=float), normalization)
        return _orient_collar_to_tip(*transformed, np.asarray(config.gravity, dtype=float))
    if config.auto_endpoints == "z":
        start_idx = int(np.argmax(original_points[:, 2]))
        end_idx = int(np.argmin(original_points[:, 2]))
        return normalized_points[start_idx], normalized_points[end_idx]
    if config.auto_endpoints == "pca":
        centered = normalized_points - normalized_points.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
        scores = centered @ axis
        return normalized_points[int(np.argmin(scores))], normalized_points[int(np.argmax(scores))]
    raise ValueError("Provide --start/--end, --endpoint-file, or use --auto-endpoints z|pca.")


def read_endpoint_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read two primary-root endpoints from JSON, CSV, or whitespace text."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Endpoint file does not exist: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return np.asarray(data["start"], dtype=float), np.asarray(data["end"], dtype=float)
        arr = np.asarray(data, dtype=float)
        if arr.shape == (2, 3):
            return arr[0], arr[1]
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else None
    if frame is not None:
        lower_cols = {col.lower(): col for col in frame.columns}
        if {"x", "y", "z"}.issubset(lower_cols):
            xyz = frame[[lower_cols["x"], lower_cols["y"], lower_cols["z"]]].to_numpy(float)
            if len(xyz) >= 2:
                if "name" in lower_cols:
                    names = frame[lower_cols["name"]].astype(str).str.lower().to_numpy()
                    if "start" in names and "end" in names:
                        return xyz[np.where(names == "start")[0][0]], xyz[np.where(names == "end")[0][0]]
                return xyz[0], xyz[1]
    numbers: list[float] = []
    for token in path.read_text(encoding="utf-8", errors="ignore").replace(",", " ").split():
        try:
            numbers.append(float(token))
        except ValueError:
            continue
    if len(numbers) >= 6:
        arr = np.asarray(numbers[:6], dtype=float).reshape(2, 3)
        return arr[0], arr[1]
    raise ValueError(f"Could not parse two endpoint coordinates from {path}")


def _validate_and_transform_endpoints(start: np.ndarray, end: np.ndarray, normalization: Normalization) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError(f"Endpoint coordinates must each contain exactly 3 values; got {start.shape} and {end.shape}")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise ValueError("Endpoint coordinates must be finite numbers.")
    if np.linalg.norm(start - end) <= 1e-12:
        raise ValueError("Primary-root endpoint coordinates are identical; choose two distinct endpoints.")
    return normalization.transform_points(start), normalization.transform_points(end)


def _coerce_exclusion_mask(mask: np.ndarray | None, point_count: int) -> np.ndarray:
    if mask is None:
        return np.zeros(point_count, dtype=bool)
    result = np.asarray(mask, dtype=bool)
    if result.shape != (point_count,):
        raise ValueError("excluded point mask must match the point count")
    return result


def _points_above_base_mask(
    points: np.ndarray,
    base_point: np.ndarray,
    tipward_direction: tuple[float, float, float] | np.ndarray,
    *,
    tolerance: float = ABOVE_BASE_TOLERANCE_NORMALIZED,
) -> np.ndarray:
    """Return points lying collarward of the selected cross-section.

    The base click is normally a surface vertex.  A horizontal gravity plane
    through that vertex cuts away the opposite half of a tilted cylindrical
    collar.  Using the local tipward primary direction instead keeps the whole
    selected cross-section while excluding geometry longitudinally above it.
    """

    points = np.asarray(points, dtype=float)
    base_point = np.asarray(base_point, dtype=float)
    tipward = np.asarray(tipward_direction, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if base_point.shape != (3,):
        raise ValueError("base point must contain three coordinates")
    direction_norm = float(np.linalg.norm(tipward))
    if tipward.shape != (3,) or not np.isfinite(direction_norm) or direction_norm <= 1e-12:
        raise ValueError("tipward_direction must contain three finite values and have non-zero length")
    tipward /= direction_norm
    return ((points - base_point) @ tipward) < -float(tolerance)


def _selected_base_exclusion_mask(
    points: np.ndarray,
    base_point: np.ndarray,
    tipward_direction: tuple[float, float, float] | np.ndarray,
    *,
    gravity: tuple[float, float, float] | np.ndarray,
    collar_neighborhood_radius: float,
    tolerance: float = ABOVE_BASE_TOLERANCE_NORMALIZED,
) -> np.ndarray:
    """Exclude shoot-side points without extending an oblique plane forever.

    Close to the selected surface vertex, the boundary is the local primary
    cross-section so both walls of a tilted collar remain available.  Outside
    that sampling-scaled collar neighbourhood, "above" follows gravity; this
    prevents a long lateral growing sideways from crossing an infinite oblique
    plane and being incorrectly removed downstream.
    """

    points = np.asarray(points, dtype=float)
    base = np.asarray(base_point, dtype=float)
    local_above = _points_above_base_mask(
        points,
        base,
        tipward_direction,
        tolerance=tolerance,
    )
    gravity_above = _points_above_base_mask(
        points,
        base,
        gravity,
        tolerance=tolerance,
    )
    radius = float(collar_neighborhood_radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("collar_neighborhood_radius must be a positive finite number")
    near_collar = np.linalg.norm(points - base, axis=1) <= radius
    return np.where(near_collar, local_above, gravity_above)


def _assign_lateral_points(
    points: np.ndarray,
    paths: list[RootPath],
    primary_mask: np.ndarray,
    d_bar: float,
    *,
    excluded_mask: np.ndarray | None = None,
) -> np.ndarray:
    labels = np.zeros(len(points), dtype=int)
    if not paths:
        return labels
    excluded = _coerce_exclusion_mask(excluded_mask, len(points))
    non_primary = np.flatnonzero(~np.asarray(primary_mask, dtype=bool) & ~excluded)
    if len(non_primary) == 0:
        return labels
    path_nodes = np.vstack([path.points for path in paths])
    node_to_label = np.concatenate([np.full(len(path.points), idx, dtype=int) for idx, path in enumerate(paths, start=1)])
    tree = cKDTree(path_nodes)
    query_k = 2 if len(path_nodes) > 1 else 1
    distances, node_idx = tree.query(points[non_primary], k=query_k, workers=worker_threads())
    if query_k == 1:
        distances = distances[:, None]
        node_idx = node_idx[:, None]
    radius = max(4.0 * d_bar, 0.006)
    assigned = distances[:, 0] <= radius
    nearest_labels = node_to_label[node_idx[:, 0]]
    labels[non_primary[assigned]] = nearest_labels[assigned]
    if query_k > 1:
        second_labels = node_to_label[node_idx[:, 1]]
        ambiguous = (
            assigned
            & (nearest_labels != second_labels)
            & ((distances[:, 1] - distances[:, 0]) <= max(0.75 * d_bar, 0.001))
        )
        labels[non_primary[ambiguous]] = -1
    return labels


def _analysis_root_labels(primary_mask: np.ndarray, lateral_labels: np.ndarray) -> np.ndarray:
    labels = np.full(len(primary_mask), -1, dtype=int)
    labels[np.asarray(primary_mask, dtype=bool)] = 0
    lateral_labels = np.asarray(lateral_labels, dtype=int)
    labels[lateral_labels > 0] = lateral_labels[lateral_labels > 0]
    labels[lateral_labels < 0] = -2
    return labels


def _point_assignment_summary(
    labels: np.ndarray,
    above_base_mask: np.ndarray,
    *,
    base_point_source: np.ndarray,
    base_tipward_direction: np.ndarray,
    gravity: np.ndarray,
    collar_neighborhood_radius: float,
    base_tolerance: float,
    d_bar: float,
    analysis_above_base_count: int,
) -> dict:
    labels = np.asarray(labels, dtype=int)
    above_base = _coerce_exclusion_mask(above_base_mask, len(labels))
    unassigned = labels == -1
    uncertain = labels == -2
    assigned = labels >= 0
    tipward = np.asarray(base_tipward_direction, dtype=float)
    tipward /= max(float(np.linalg.norm(tipward)), 1e-12)
    gravity_direction = np.asarray(gravity, dtype=float)
    gravity_direction /= max(float(np.linalg.norm(gravity_direction)), 1e-12)
    return {
        "total_vertex_count": int(len(labels)),
        "primary_assigned_vertex_count": int(np.count_nonzero(labels == 0)),
        "lateral_assigned_vertex_count": int(np.count_nonzero(labels > 0)),
        "base_point_source_coordinates": np.asarray(base_point_source, dtype=float),
        "base_tipward_direction": tipward,
        "gravity_direction": gravity_direction,
        "base_collar_neighborhood_radius_normalized": float(collar_neighborhood_radius),
        "above_base_tolerance_normalized": float(base_tolerance),
        "rule": "inside the collar neighbourhood, points collarward of the local primary cross-section remain unassigned; outside it, points above the selected base along gravity remain unassigned",
        "analysis_lateral_assignment_radius_normalized": max(4.0 * float(d_bar), 0.006),
        "full_resolution_assignment_radius_normalized": max(5.0 * float(d_bar), 0.008),
        "ambiguity_margin_normalized": max(0.75 * float(d_bar), 0.001),
        "assigned_vertex_count": int(np.count_nonzero(assigned)),
        "uncertain_vertex_count": int(np.count_nonzero(uncertain)),
        "unassigned_vertex_count": int(np.count_nonzero(unassigned)),
        "analysis_above_base_point_count": int(analysis_above_base_count),
        "full_resolution_above_base_point_count": int(np.count_nonzero(above_base)),
        "unassigned_reason_counts": {
            "above_selected_base": int(np.count_nonzero(unassigned & above_base)),
            "not_claimed_by_primary_or_selected_lateral": int(np.count_nonzero(unassigned & ~above_base)),
        },
        "unassigned_reason_descriptions": {
            "above_selected_base": "The point is shoot-side of the selected base under the local-collar/gravity hybrid boundary.",
            "not_claimed_by_primary_or_selected_lateral": "The point was not in the segmented primary and was not within the assignment support of a selected lateral root.",
        },
        "uncertain_description": "The point is close enough to competing selected roots that ownership is ambiguous.",
    }


def _assign_full_root_labels(
    points: np.ndarray,
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    *,
    d_bar: float,
    excluded_mask: np.ndarray | None = None,
) -> np.ndarray:
    all_paths = [np.asarray(primary_path, dtype=float)] + [np.asarray(path.points, dtype=float) for path in lateral_paths]
    path_nodes = np.vstack(all_paths)
    node_labels = np.concatenate(
        [np.full(len(path), label, dtype=int) for label, path in enumerate(all_paths)]
    )
    tree = cKDTree(path_nodes)
    query_k = 2 if len(path_nodes) > 1 else 1
    distances, node_indices = tree.query(points, k=query_k, workers=worker_threads())
    if query_k == 1:
        distances = distances[:, None]
        node_indices = node_indices[:, None]
    labels = np.full(len(points), -1, dtype=int)
    radius = max(5.0 * d_bar, 0.008)
    excluded = _coerce_exclusion_mask(excluded_mask, len(points))
    assigned = (distances[:, 0] <= radius) & ~excluded
    nearest = node_labels[node_indices[:, 0]]
    labels[assigned] = nearest[assigned]
    if query_k > 1:
        second = node_labels[node_indices[:, 1]]
        ambiguous = (
            assigned
            & (nearest != second)
            & ((distances[:, 1] - distances[:, 0]) <= max(0.75 * d_bar, 0.001))
        )
        labels[ambiguous] = -2
    return labels


def _update_metadata_timings(path: Path, timings: dict[str, float]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["stage_timings_seconds"] = timings
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
