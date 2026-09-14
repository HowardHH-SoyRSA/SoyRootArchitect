from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .competition import RootSegmentIndex, TIE_TOLERANCE
from .export import export_results
from .centerline import refit_final_centerlines
from .primary_guidance import (
    PRIMARY_GUIDANCE_FILENAME,
    PrimaryGuidance,
    write_primary_guidance,
)
from .geometry import mean_nearest_neighbor_distance, normalize_unit_box
from .io import load_root_geometry
from .lateral import (
    MAIN_TRACER_LOCAL_DENSITY_WEIGHT,
    MAIN_TRACER_MAX_TURN_DEGREES,
    MAIN_TRACER_NEW_DIRECTION_WEIGHT,
    MAIN_TRACER_OLD_DIRECTION_WEIGHT,
    MAIN_TRACER_RADIUS_CONTINUITY_WEIGHT,
    MAIN_TRACER_STEP_DISTANCE_WEIGHT,
    MAIN_TRACER_TURN_ALIGNMENT_WEIGHT,
    TIP_EXTENSION_MAX_STEPS,
    backtrace_to_primary,
    estimate_parent_radius_profile,
    extend_lateral_tip,
    find_lateral_starting_points,
    grow_lateral_candidates,
    is_ancestor_inward_candidate,
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
from .topology import (
    _polyline_fingerprint,
    apply_hierarchy_corrections,
    repair_root_hierarchy,
    uncross_internal_primary_sibling_contacts,
    validate_root_tree,
)
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


def _ancestor_tube_mask(
    points: np.ndarray,
    ancestor_path: np.ndarray,
    ancestor_radius_profile: np.ndarray,
    d_bar: float,
    *,
    ancestor_tree: cKDTree | None = None,
) -> np.ndarray:
    """Return vertices inside a sampling-scaled ancestor surface envelope."""

    source = np.asarray(points, dtype=float)
    ancestor = np.asarray(ancestor_path, dtype=float)
    radii = np.asarray(ancestor_radius_profile, dtype=float)
    if radii.shape != (len(ancestor),):
        raise ValueError(
            "ancestor_radius_profile must contain one value per path node"
        )
    tree = ancestor_tree if ancestor_tree is not None else cKDTree(ancestor)
    distances, nearest = tree.query(
        source,
        k=1,
        workers=worker_threads(),
    )
    envelope = radii[np.asarray(nearest, dtype=int)] + 2.0 * float(d_bar)
    return np.asarray(distances, dtype=float) <= envelope


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
    # Snapshot external selection files once, then save and use exactly those
    # same source-coordinate values even if the original files later move.
    manual_guidance = _manual_primary_guidance(config)
    if manual_guidance is not None:
        write_primary_guidance(
            config.output_dir / PRIMARY_GUIDANCE_FILENAME,
            manual_guidance,
            input_path=config.input_path,
        )
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
        manual_guidance=manual_guidance,
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
    selected, topology_report = repair_root_hierarchy(
        primary.points,
        selected,
        d_bar=d_bar,
        primary_surface_points=normalized[
            np.asarray(primary_mask, dtype=bool) & ~above_base_mask
        ],
    )
    correction_input_fingerprints = {
        "primary": _polyline_fingerprint(primary.points),
        **{path.root_id: _polyline_fingerprint(path.points) for path in selected},
    }
    if config.correction_file is not None:
        selected = apply_hierarchy_corrections(
            primary.points,
            selected,
            config.correction_file,
            normalization=normalization,
        )
    topology_errors = validate_root_tree(
        selected,
        primary_path=primary.points,
    )
    if topology_errors:
        raise RuntimeError("Root topology validation failed: " + "; ".join(topology_errors))
    checkpoint("topology_repair", "Assigning root vertices", 0.76)
    segmented_primary_mask = np.asarray(primary_mask, dtype=bool).copy()
    assignment_radii = _assignment_radius_profiles(
        normalized, primary.points, selected, segmented_primary_mask & ~above_base_mask, d_bar,
    )
    lateral_labels, analysis_competing_labels = _assign_lateral_points(
        normalized,
        selected,
        segmented_primary_mask,
        d_bar,
        primary_path=primary.points,
        return_root_labels=True,
        excluded_mask=above_base_mask,
        return_competing_labels=True,
        root_radii=assignment_radii,
    )
    analysis_root_labels = lateral_labels.copy()
    analysis_root_labels, analysis_junction_report = _resolve_parent_owned_junctions(
        normalized,
        analysis_root_labels,
        primary.points,
        selected,
        d_bar=d_bar,
        assignment_radius=max(4.0 * d_bar, 0.006),
        ambiguity_margin=max(0.75 * d_bar, 0.001),
        competing_labels=analysis_competing_labels,
        root_radii=assignment_radii,
    )
    # The lateral-label representation reserves zero for "not lateral", so
    # reconstruct both masks from unified labels after resolving primary-owned
    # junctions.  This prevents the sampled-point overwrite below from
    # reintroducing orange uncertainty bands into full-resolution exports.
    primary_mask = analysis_root_labels == 0
    lateral_labels = np.zeros(len(analysis_root_labels), dtype=int)
    lateral_labels[analysis_root_labels > 0] = analysis_root_labels[
        analysis_root_labels > 0
    ]
    lateral_labels[analysis_root_labels == -2] = -1
    full_normalized = normalization.transform_points(cloud.export_points)
    full_above_base_mask = _selected_base_exclusion_mask(
        full_normalized,
        selected_base,
        base_tipward_direction,
        gravity=np.asarray(config.gravity, dtype=float),
        collar_neighborhood_radius=base_collar_neighborhood_radius,
        tolerance=base_tolerance,
    )
    full_root_labels, full_competing_labels = _assign_full_root_labels(
        full_normalized,
        primary.points,
        selected,
        d_bar=d_bar,
        excluded_mask=full_above_base_mask,
        return_competing_labels=True,
        root_radii=assignment_radii,
    )
    (
        internal_o1_contact_changed_ids,
        internal_o1_contact_decisions,
    ) = uncross_internal_primary_sibling_contacts(
        full_normalized,
        full_root_labels,
        selected,
        d_bar=d_bar,
    )
    internal_o1_contact_changed_ids = {
        str(root_id) for root_id in internal_o1_contact_changed_ids
    }
    if internal_o1_contact_changed_ids:
        topology_errors = validate_root_tree(
            selected,
            primary_path=primary.points,
        )
        if topology_errors:
            raise RuntimeError(
                "Root topology validation failed after internal O1 contact "
                "uncrossing: "
                + "; ".join(topology_errors)
            )
        LOGGER.info(
            "Internal O1 contact uncrossing changed %d roots; "
            "reassigning analysis and full-resolution vertices",
            len(internal_o1_contact_changed_ids),
        )
        assignment_radii = _assignment_radius_profiles(
            normalized, primary.points, selected, segmented_primary_mask & ~above_base_mask, d_bar,
        )
        lateral_labels, analysis_competing_labels = _assign_lateral_points(
            normalized,
            selected,
            segmented_primary_mask,
            d_bar,
            primary_path=primary.points,
            return_root_labels=True,
            excluded_mask=above_base_mask,
            return_competing_labels=True,
            root_radii=assignment_radii,
        )
        analysis_root_labels = lateral_labels.copy()
        (
            analysis_root_labels,
            analysis_junction_report,
        ) = _resolve_parent_owned_junctions(
            normalized,
            analysis_root_labels,
            primary.points,
            selected,
            d_bar=d_bar,
            assignment_radius=max(4.0 * d_bar, 0.006),
            ambiguity_margin=max(0.75 * d_bar, 0.001),
            competing_labels=analysis_competing_labels,
            root_radii=assignment_radii,
        )
        primary_mask = analysis_root_labels == 0
        lateral_labels = np.zeros(len(analysis_root_labels), dtype=int)
        lateral_labels[analysis_root_labels > 0] = analysis_root_labels[
            analysis_root_labels > 0
        ]
        lateral_labels[analysis_root_labels == -2] = -1
        full_root_labels, full_competing_labels = _assign_full_root_labels(
            full_normalized,
            primary.points,
            selected,
            d_bar=d_bar,
            excluded_mask=full_above_base_mask,
            return_competing_labels=True,
            root_radii=assignment_radii,
        )
    analysis_to_full: np.ndarray | None = None
    if cloud.analysis_indices is not None and len(cloud.analysis_indices) == len(normalized):
        analysis_to_full = np.asarray(cloud.analysis_indices, dtype=int)
        full_root_labels[analysis_to_full] = analysis_root_labels
        # Keep evidence synchronized with the labels being restored.
        for vertex_index in analysis_to_full:
            full_competing_labels.pop(int(vertex_index), None)
        for analysis_index, pair in analysis_competing_labels.items():
            full_competing_labels[int(analysis_to_full[int(analysis_index)])] = pair
    full_root_labels[full_above_base_mask] = -1
    full_root_labels, full_junction_report = _resolve_parent_owned_junctions(
        full_normalized,
        full_root_labels,
        primary.points,
        selected,
        d_bar=d_bar,
        assignment_radius=max(5.0 * d_bar, 0.008),
        ambiguity_margin=max(0.75 * d_bar, 0.001),
        competing_labels=full_competing_labels,
        root_radii=assignment_radii,
    )
    full_root_labels, primary_surface_patch_report = (
        _absorb_small_primary_surface_patches(
            full_normalized,
            full_root_labels,
            primary.points,
            selected,
            d_bar=d_bar,
            triangles=cloud.triangles,
            excluded_mask=full_above_base_mask,
            competing_labels=full_competing_labels,
            root_radii=assignment_radii,
            primary_support_points=normalized[
                segmented_primary_mask & ~above_base_mask
            ],
        )
    )
    # Run after sampled-mask restoration AND parent patch cleanup. Strong,
    # connected child support must have the final say over a primary mask.
    full_root_labels, primary_o1_ownership_report = _resolve_primary_o1_ownership(
        full_normalized, full_root_labels, primary.points, selected,
        d_bar=d_bar, triangles=cloud.triangles,
        excluded_mask=full_above_base_mask,
        competing_labels=full_competing_labels,
    )
    cleaned_analysis_labels: np.ndarray | None = None
    if analysis_to_full is not None:
        cleaned_analysis_labels = full_root_labels[analysis_to_full]
    elif len(full_root_labels) == len(analysis_root_labels):
        cleaned_analysis_labels = full_root_labels
    if cleaned_analysis_labels is not None:
        analysis_root_labels = np.asarray(cleaned_analysis_labels, dtype=int).copy()
        primary_mask = analysis_root_labels == 0
        lateral_labels = np.zeros(len(analysis_root_labels), dtype=int)
        lateral_labels[analysis_root_labels > 0] = analysis_root_labels[
            analysis_root_labels > 0
        ]
        lateral_labels[analysis_root_labels == -2] = -1
    LOGGER.info(
        "Assigned %d analysis and %d full-resolution uncertain junction points to their parents",
        int(analysis_junction_report["resolved_vertex_count"]),
        int(full_junction_report["resolved_vertex_count"]),
    )
    LOGGER.info(
        "Absorbed %d discrete primary-surface patches containing %d vertices",
        int(primary_surface_patch_report["absorbed_patch_count"]),
        int(primary_surface_patch_report["absorbed_vertex_count"]),
    )
    competition_indices = np.asarray(sorted(full_competing_labels), dtype=int)
    collar_indices = competition_indices[
        np.linalg.norm(full_normalized[competition_indices] - selected_base, axis=1)
        <= base_collar_neighborhood_radius
    ]
    competition_report = {
        "policy": "distinct-root-segments-v1",
        "metric": "absolute(centerline segment distance - interpolated local radius)",
        "evidence_file": "root_competition.npz",
        "radius_source": "frozen unshared traced support; primary segmented support; missing radius zero",
        "radius_profiles_normalized": {str(k): v for k, v in assignment_radii.items()},
        "tie_tolerance_normalized": TIE_TOLERANCE,
        "index": "length-bucketed segment midpoint KD trees; conservative radius balls; chunked exact projection",
        "competing_vertex_count": len(full_competing_labels),
        "remaining_uncertain_competing_vertex_count": int(np.sum(full_root_labels[competition_indices] == -2)),
        "collar_competing_vertex_count": int(len(collar_indices)),
        "collar_uncertain_vertex_count": int(np.sum(full_root_labels[collar_indices] == -2)),
        "collar_excluded_vertex_count": int(np.sum(full_above_base_mask)),
        "evidence_geometry": "before final assignment-supported centerline fitting",
    }
    checkpoint("point_assignment", "Fitting final assigned root centerlines", 0.80)
    primary.points, final_centerline_report = refit_final_centerlines(
        full_normalized,
        full_root_labels,
        primary.points,
        selected,
        d_bar=d_bar,
        triangles=cloud.triangles,
        cooperate=cooperate,
    )
    primary.qc_flags = list(dict.fromkeys([
        *primary.qc_flags, *final_centerline_report["primary_qc_flags"],
    ]))
    for assessment in final_centerline_report["roots"]:
        assessment["correction_input_geometry_fingerprint"] = correction_input_fingerprints[assessment["root_id"]]
    checkpoint("final_centerline_fitting", "Computing root traits", 0.84)
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
        primary_centerline_assessment=final_centerline_report["roots"][0],
        tip_vector_window=config.tip_vector_window_mesh_units,
    )
    checkpoint("trait_measurement", "Rendering validation figures", 0.87)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output_dir / "root_competition.npz",
        vertex_indices=competition_indices,
        root_labels=np.asarray(
            [full_competing_labels[int(i)] for i in competition_indices], dtype=np.int64,
        ).reshape(-1, 2),
    )
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
        "distinct_root_competition": competition_report,
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
        "lateral_tracing_policy": {
            "hypotheses_per_parameter_variant": 1,
            "fork_hypothesis_promotion": False,
            "surface_aware_order1_seeding": True,
            "surface_contact_distance_d_bar": 2.5,
            "surface_contact_minimum_points": 3,
            "minimum_travel_fraction_insertion": 0.30,
            "minimum_travel_fraction_stable": 0.375,
            "minimum_travel_fraction_high_curvature": 0.30,
            "minimum_travel_fraction_fallback": 0.30,
            "accepted_node_cross_section_clearance": (
                "0.90 * search_radius with axial progress <= "
                "0.29 * step_length"
            ),
            "covered_forward_recovery": True,
            "main_tracer_max_turn_degrees": MAIN_TRACER_MAX_TURN_DEGREES,
            "main_tracer_step_score_turn_alignment_weight": (
                MAIN_TRACER_TURN_ALIGNMENT_WEIGHT
            ),
            "main_tracer_step_score_local_density_weight": (
                MAIN_TRACER_LOCAL_DENSITY_WEIGHT
            ),
            "main_tracer_step_score_distance_weight": (
                MAIN_TRACER_STEP_DISTANCE_WEIGHT
            ),
            "main_tracer_step_score_radius_continuity_weight": (
                MAIN_TRACER_RADIUS_CONTINUITY_WEIGHT
            ),
            "main_tracer_old_direction_weight": (
                MAIN_TRACER_OLD_DIRECTION_WEIGHT
            ),
            "main_tracer_new_direction_weight": (
                MAIN_TRACER_NEW_DIRECTION_WEIGHT
            ),
            "tip_extension_max_steps": TIP_EXTENSION_MAX_STEPS,
            "local_radius_continuity_reward": (
                MAIN_TRACER_RADIUS_CONTINUITY_WEIGHT
            ),
            "longest_path_reward_per_d_bar": 0.35,
            "rms_curvature_penalty": 0.25,
            "cumulative_turn_penalty": 0.06,
            "same_insertion_divergence_reparenting": True,
            "same_insertion_o1_duplicate_merge": True,
            "contacted_primary_sibling_suffix_crop": True,
            "contacted_sibling_child_continuation_join": True,
            "internal_o1_contact_uncrossing": True,
            "internal_o1_contact_uncrossing_labels": (
                "pre-junction full-resolution root labels"
            ),
            "ancestor_inward_terminal_rejection": True,
            "child_length_may_not_exceed_parent": True,
            "child_length_control_stage": "topology_repair_before_final_support_fitting",
            "post_fit_length_violation_action": "QC_flag_preserve_final_ownership_and_hierarchy",
            "overlong_child_alternative_parent_resurvey": True,
            "overlong_child_action": (
                "at a supported internal fork, resurvey the overlong "
                "child arm as the alternative parent continuation and retain "
                "both arms only when the resulting parent and child satisfy "
                "the same length control; otherwise remove the violating "
                "automatic child and its descendant subtree; reject manual "
                "hierarchy edits"
            ),
        },
        "internal_o1_contact_changed_root_ids": sorted(
            internal_o1_contact_changed_ids
        ),
        "internal_o1_contact_decisions": internal_o1_contact_decisions,
        "final_centerline_fitting": final_centerline_report,
        "primary_o1_ownership": primary_o1_ownership_report,
        "primary_detection_method": _primary_method(config),
        "primary_guidance_file": (
            PRIMARY_GUIDANCE_FILENAME if manual_guidance is not None else None
        ),
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
            analysis_junction_report=analysis_junction_report,
            full_junction_report=full_junction_report,
            primary_surface_patch_report=primary_surface_patch_report,
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
    primary_tree = cKDTree(primary_path)
    parent_tree_cache: dict[str, cKDTree] = {"primary": primary_tree}
    total_starts = 0
    total_candidates = 0
    order_counts: dict[int, int] = {}
    primary_support_points = points[
        np.asarray(primary_mask, dtype=bool) & ~excluded
    ]
    if len(primary_support_points) < 3:
        primary_support_points = np.asarray(primary_path, dtype=float)
    primary_ancestor_radius_profile = estimate_parent_radius_profile(
        primary_path,
        primary_support_points,
        d_bar,
        parent_tree=primary_tree,
    )
    parent_radius_cache: dict[str, np.ndarray] = {
        "primary": primary_ancestor_radius_profile,
    }
    primary_ancestor_mask = _ancestor_tube_mask(
        points,
        primary_path,
        primary_ancestor_radius_profile,
        d_bar,
        ancestor_tree=primary_tree,
    )

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
            parent_position: int | None = None
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
            parent_tree = parent_tree_cache.get(parent_id)
            if parent_tree is None:
                parent_tree = cKDTree(parent_path)
                parent_tree_cache[parent_id] = parent_tree
            parent_radius_profile = parent_radius_cache.get(parent_id)
            if parent_radius_profile is None:
                parent_radius_profile = estimate_parent_radius_profile(
                    parent_path,
                    parent_support_points,
                    d_bar,
                    parent_tree=parent_tree,
                )
                parent_radius_cache[parent_id] = parent_radius_profile
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
                parent_surface_points=(
                    parent_support_points
                    if order == 1 and parent_id == "primary"
                    else None
                ),
                surface_contact_distance=(
                    2.5 * float(d_bar)
                    if order == 1 and parent_id == "primary"
                    else None
                ),
                parent_tree=parent_tree,
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
                ancestor_exclusion_mask=(
                    primary_ancestor_mask if order > 1 else None
                ),
                cooperate=cooperate,
                point_tree=extension_tree,
                parent_tree=parent_tree,
            )
            order_candidate_count += len(candidates)
            evaluated_groups: dict[int | str, list[tuple[RootPath, bool]]] = {}
            for candidate in candidates:
                candidate.order = order
                candidate.parent_id = parent_id
                candidate.parent_points = parent_path
                if order > 1:
                    ancestor_rejected, ancestor_metrics = (
                        is_ancestor_inward_candidate(
                            candidate,
                            primary_path,
                            primary_ancestor_radius_profile,
                            d_bar,
                            ancestor_tree=primary_tree,
                        )
                    )
                    candidate.score_components.update(ancestor_metrics)
                    if ancestor_rejected:
                        order_parent_tracking_rejections += 1
                        continue
                rejected, tracking_metrics = is_parent_tracking_candidate(
                    candidate,
                    parent_path,
                    parent_radius_profile,
                    d_bar,
                    parent_tree=parent_tree,
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
        existing_order_support: set[int] = set()
        for existing in selected_all:
            if int(existing.order) != order:
                continue
            existing_order_support.update(
                existing.novel_support_indices
                if existing.novel_support_indices is not None
                else existing.covered_indices
            )
        selected = select_non_overlapping_paths(
            reduced,
            points,
            d_bar=d_bar,
            max_paths=remaining,
            rename_selected=False,
            initial_used=existing_order_support,
            point_tree=extension_tree,
        )
        connectors_by_parent: dict[str, list[RootPath]] = {}
        ordinary_selected: list[RootPath] = []
        selected_parent_by_id = {
            str(path.root_id): path
            for path in selected_all
        }
        for path in selected:
            parent_root = selected_parent_by_id.get(str(path.parent_id))
            if (
                parent_root is not None
                and _is_parent_owned_basal_connector(
                    path,
                    parent_root,
                    d_bar=d_bar,
                )
            ):
                connectors_by_parent.setdefault(
                    str(parent_root.root_id),
                    [],
                ).append(path)
            else:
                ordinary_selected.append(path)
        for parent_id, connectors in connectors_by_parent.items():
            _merge_parent_owned_basal_connectors(
                selected_parent_by_id[parent_id],
                connectors,
                d_bar=d_bar,
            )
            order_parent_tracking_rejections += len(connectors)
        selected = ordinary_selected
        sequence_base = len(selected_all)
        for sequence_index, path in enumerate(selected, start=1):
            path.order = order
            path.root_id = (
                f"order{path.order}_{sequence_base + sequence_index:03d}"
            )
        refined: list[RootPath] = []
        for path in selected:
            path.parent_points = (
                path.parent_points
                if path.parent_points is not None
                else primary_path
            )
            path.parent_id = path.parent_id or "primary"
            traced_paths = backtrace_to_primary(
                [path],
                path.parent_points,
                primary_points=path.parent_points,
                target_tree=parent_tree_cache.get(str(path.parent_id)),
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
        if order > 1:
            final_refined: list[RootPath] = []
            for traced in refined:
                ancestor_rejected, ancestor_metrics = (
                    is_ancestor_inward_candidate(
                        traced,
                        primary_path,
                        primary_ancestor_radius_profile,
                        d_bar,
                        ancestor_tree=primary_tree,
                    )
                )
                traced.score_components.update(ancestor_metrics)
                traced.score_components[
                    "ancestor_inward_final_check"
                ] = 1.0
                if ancestor_rejected:
                    order_parent_tracking_rejections += 1
                    continue
                final_refined.append(traced)
            refined = final_refined
        if not refined:
            break
        selected_all.extend(refined)
        for path in refined:
            parent_tree_cache[str(path.root_id)] = cKDTree(path.points)
        order_counts[order] = sum(int(path.order) == order for path in selected_all)
        labels = _assign_lateral_points(
            points,
            selected_all,
            primary_mask,
            d_bar,
            excluded_mask=excluded,
        )
        # Competing selected roots already claim these surfaces. Uncertain
        # ownership must not turn them into novel support for a later order.
        occupied_mask = np.asarray(primary_mask, dtype=bool) | excluded | (labels != 0)
        parent_paths = [
            (path.root_id, path.points)
            for path in selected_all
            if int(path.order) == order
        ]
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


def _is_parent_owned_basal_connector(
    candidate: RootPath,
    parent: RootPath,
    *,
    d_bar: float,
) -> bool:
    """Identify a reverse trace that completes its parent's ancestor junction."""

    if (
        len(candidate.points) < 3
        or len(parent.points) < 2
        or parent.parent_points is None
        or len(parent.parent_points) < 2
        or candidate.score_components.get(
            "parent_short_contained_without_escape",
            0.0,
        )
        <= 0.0
        or candidate.score_components.get(
            "parent_terminal_outside_fraction",
            1.0,
        )
        > 0.10
    ):
        return False

    spacing = float(d_bar)
    attachment_radius = float(
        candidate.score_components.get(
            "parent_attachment_radius",
            0.0,
        )
    )
    if (
        not np.isfinite(spacing)
        or spacing <= 0.0
        or not np.isfinite(attachment_radius)
        or attachment_radius <= 0.0
    ):
        return False

    child = np.asarray(candidate.points, dtype=float)
    parent_path = np.asarray(parent.points, dtype=float)
    ancestor = np.asarray(parent.parent_points, dtype=float)
    parent_start = parent_path[0]
    start_limit = max(2.0 * spacing, 0.35 * attachment_radius)
    if float(np.linalg.norm(child[0] - parent_start)) > start_limit:
        return False

    ancestor_tree = cKDTree(ancestor)
    ancestor_distances, ancestor_indices = ancestor_tree.query(
        child,
        k=1,
        workers=worker_threads(),
    )
    ancestor_distances = np.asarray(ancestor_distances, dtype=float)
    # The terminal point supplies the intended ancestor insertion.  Projecting
    # the parent-side start instead can choose a different ancestor station and
    # turn a valid diagonal connector into an apparent corridor excursion.
    ancestor_projection = ancestor[int(ancestor_indices[-1])]
    ancestor_gap = float(np.linalg.norm(parent_start - ancestor_projection))
    if ancestor_gap <= max(2.0 * spacing, 0.50 * attachment_radius):
        return False
    approach = float(ancestor_distances[0] - ancestor_distances[-1])
    minimum_approach = max(2.0 * spacing, 0.50 * attachment_radius)
    terminal_limit = max(4.0 * spacing, 0.75 * attachment_radius)
    if approach < minimum_approach or float(ancestor_distances[-1]) > terminal_limit:
        return False

    parent_segments = np.linalg.norm(np.diff(parent_path, axis=0), axis=1)
    parent_arc = np.concatenate([[0.0], np.cumsum(parent_segments)])
    _, nearest_parent = cKDTree(parent_path).query(
        child,
        k=1,
        workers=worker_threads(),
    )
    basal_limit = max(8.0 * spacing, 1.5 * attachment_radius)
    if float(np.max(parent_arc[np.asarray(nearest_parent, dtype=int)])) > basal_limit:
        return False

    corridor = parent_start - ancestor_projection
    corridor_squared = float(np.dot(corridor, corridor))
    if corridor_squared <= 1e-20:
        return False
    parameters = np.clip(
        ((child - ancestor_projection) @ corridor) / corridor_squared,
        0.0,
        1.0,
    )
    corridor_projection = (
        ancestor_projection
        + parameters[:, None] * corridor
    )
    corridor_deviation = np.linalg.norm(
        child - corridor_projection,
        axis=1,
    )
    corridor_limit = max(2.0 * spacing, 0.60 * attachment_radius)
    return bool(float(np.max(corridor_deviation)) <= corridor_limit)


def _merge_parent_owned_basal_connectors(
    parent: RootPath,
    connectors: list[RootPath],
    *,
    d_bar: float,
) -> None:
    """Prepend the strongest ancestor connector and retain all its support."""

    if not connectors or parent.parent_points is None:
        return
    representative = max(
        connectors,
        key=lambda path: (
            len(
                path.novel_support_indices
                if path.novel_support_indices is not None
                else path.covered_indices
            ),
            float(path.score),
            float(path.length),
            str(path.root_id),
        ),
    )
    ancestor = np.asarray(parent.parent_points, dtype=float)
    terminal = np.asarray(representative.points[-1], dtype=float)
    _, ancestor_index = cKDTree(ancestor).query(terminal, k=1)
    ancestor_projection = ancestor[int(ancestor_index)].copy()
    reversed_connector = np.asarray(
        representative.points[::-1],
        dtype=float,
    )
    joined = np.vstack(
        [
            ancestor_projection,
            reversed_connector,
            np.asarray(parent.points[1:], dtype=float),
        ]
    )
    keep = np.concatenate(
        [
            [True],
            np.linalg.norm(np.diff(joined, axis=0), axis=1) > 1e-12,
        ]
    )
    parent.points = joined[keep]
    covered_support: set[int] = set()
    novel_support: set[int] = set()
    for connector in connectors:
        covered_support.update(connector.covered_indices)
        novel_support.update(
            connector.novel_support_indices
            if connector.novel_support_indices is not None
            else connector.covered_indices
        )
    parent.covered_indices.update(covered_support)
    if parent.novel_support_indices is not None or novel_support:
        parent.novel_support_indices = set(
            parent.novel_support_indices or ()
        ) | novel_support
    parent.raw_start_point = terminal.copy()
    parent.node_indices = None
    parent.score_components["parent_owned_basal_connector_merged"] = (
        parent.score_components.get(
            "parent_owned_basal_connector_merged",
            0.0,
        )
        + 1.0
    )
    parent.score_components["parent_owned_basal_connector_variants"] = (
        parent.score_components.get(
            "parent_owned_basal_connector_variants",
            0.0,
        )
        + float(len(connectors))
    )
    parent.score_components["parent_owned_basal_connector_support_added"] = (
        parent.score_components.get(
            "parent_owned_basal_connector_support_added",
            0.0,
        )
        + float(len(covered_support))
    )
    parent.score_components["parent_owned_basal_connector_spacing"] = float(
        d_bar
    )


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


def _manual_primary_guidance(config: PipelineConfig) -> PrimaryGuidance | None:
    if config.endpoint_file is not None:
        start, end = read_endpoint_file(config.endpoint_file)
    elif config.start is not None and config.end is not None:
        start, end = np.asarray(config.start, dtype=float), np.asarray(config.end, dtype=float)
    else:
        return None
    return PrimaryGuidance(start, end, config.soil_z, _read_primary_guides(config))


def _resolve_primary_path(
    original_points: np.ndarray,
    normalized_points: np.ndarray,
    normalization: Normalization,
    d_bar: float,
    config: PipelineConfig,
    *,
    cooperate: Callable[[], None] | None = None,
    manual_guidance: PrimaryGuidance | None = None,
) -> tuple[RootPath, list[PrimaryCandidate]]:
    gravity = np.asarray(config.gravity, dtype=float)
    gravity /= np.linalg.norm(gravity)
    guidance = manual_guidance if manual_guidance is not None else _manual_primary_guidance(config)
    guides = guidance.guides if guidance is not None else _read_primary_guides(config)
    normalized_guides = (
        normalization.transform_points(guides) if len(guides) else np.empty((0, 3), dtype=float)
    )
    candidates: list[PrimaryCandidate] = []
    if guidance is not None:
        start, end = _validate_and_transform_endpoints(guidance.start, guidance.end, normalization)
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


def _assignment_radius_profiles(points, primary, paths, primary_mask, d_bar):
    """Freeze segment-derived radii from unshared traced surface support.

    Shared coverage and primary-owned vertices cannot inflate lateral radii.
    Radius measurement is separate from the assignment it will influence.
    """
    coverage = np.zeros(len(points), dtype=int)
    supports = []
    for path in paths:
        ids = np.asarray(sorted(path.covered_indices), dtype=int)
        ids = ids[(ids >= 0) & (ids < len(points))]
        supports.append(ids)
        coverage[ids] += 1
    profiles = {}
    for label, path in enumerate([primary] + [p.points for p in paths]):
        if not len(path):
            profiles[label] = np.empty(0)
            continue
        if label == 0:
            support = points[primary_mask]
        else:
            ids = supports[label - 1]
            ids = ids[(coverage[ids] == 1) & ~primary_mask[ids]]
            support = points[ids]
        if len(support) >= 3:
            stations, values = _segment_radius_profile(support, path, d_bar)
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            profiles[label] = np.interp(arc, stations, values)
        else:
            value = paths[label - 1].mean_radius if label else None
            profiles[label] = np.full(len(path), _valid_root_radius(value))
    return profiles


def _valid_root_radius(value):
    return float(value) if value is not None and np.isfinite(value) and value >= 0 else 0.0


def _root_segment_index(primary_path, paths, root_radii=None):
    root_ids = (["primary"] if primary_path is not None else []) + [str(p.root_id) for p in paths]
    if len(root_ids) != len(set(root_ids)):
        raise ValueError("root identities must be unique")
    polylines = ([] if primary_path is None else [primary_path]) + [p.points for p in paths]
    labels = list(range(0 if primary_path is not None else 1, len(paths) + 1))
    defaults = ([] if primary_path is None else [0.0]) + [_valid_root_radius(p.mean_radius) for p in paths]
    profiles = [(root_radii or {}).get(label, fallback) for label, fallback in zip(labels, defaults, strict=True)]
    return RootSegmentIndex(polylines, labels=labels, radii=profiles)


def _assign_lateral_points(
    points: np.ndarray,
    paths: list[RootPath],
    primary_mask: np.ndarray,
    d_bar: float,
    *,
    excluded_mask: np.ndarray | None = None,
    return_competing_labels: bool = False,
    primary_path: np.ndarray | None = None,
    root_radii: dict | None = None,
    return_root_labels: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[int, tuple[int, int]]]:
    labels = np.zeros(len(points), dtype=int)
    if return_root_labels:
        labels[:] = -1
        labels[np.asarray(primary_mask, dtype=bool)] = 0
    excluded = _coerce_exclusion_mask(excluded_mask, len(points))
    # With primary geometry available, measure competition even on the
    # segmented primary mask; mask restoration must not erase this evidence.
    ids = np.flatnonzero(~excluded & (True if primary_path is not None else ~np.asarray(primary_mask, dtype=bool)))
    radius, margin = max(4.0 * d_bar, 0.006), max(0.75 * d_bar, 0.001)
    result = _root_segment_index(primary_path, paths, root_radii).query(points[ids], max_distance=radius + margin)
    assigned = result.distances[:, 0] <= radius + TIE_TOLERANCE
    labels[ids[assigned]] = result.labels[assigned, 0]
    ambiguous = assigned & result.ambiguous(margin)
    labels[ids[ambiguous]] = -2 if return_root_labels else -1
    if return_root_labels:
        labels[excluded] = -1
    pairs = {int(i): tuple(map(int, pair)) for i, pair in zip(ids[ambiguous], result.labels[ambiguous], strict=True)}
    return (labels, pairs) if return_competing_labels else labels


def _analysis_root_labels(primary_mask: np.ndarray, lateral_labels: np.ndarray) -> np.ndarray:
    labels = np.full(len(primary_mask), -1, dtype=int)
    labels[np.asarray(primary_mask, dtype=bool)] = 0
    lateral_labels = np.asarray(lateral_labels, dtype=int)
    labels[lateral_labels > 0] = lateral_labels[lateral_labels > 0]
    labels[lateral_labels < 0] = -2
    return labels


def _group_indices_by_label(
    labels: np.ndarray,
    included_labels: set[int],
) -> dict[int, np.ndarray]:
    """Group selected label indices with one full label-array pass."""

    values = np.asarray(labels, dtype=int)
    if values.ndim != 1 or not included_labels:
        return {}
    selected = np.asarray(sorted(int(label) for label in included_labels), dtype=int)
    included_indices = np.flatnonzero(np.isin(values, selected))
    if not len(included_indices):
        return {}
    included_values = values[included_indices]
    order = np.argsort(included_values, kind="stable")
    sorted_indices = included_indices[order]
    sorted_values = included_values[order]
    starts = np.concatenate(
        [[0], np.flatnonzero(np.diff(sorted_values) != 0) + 1]
    )
    ends = np.concatenate([starts[1:], [len(sorted_values)]])
    return {
        int(sorted_values[start]): sorted_indices[start:end]
        for start, end in zip(starts, ends, strict=True)
    }


def _surface_connectivity_edges(
    points: np.ndarray,
    triangles: np.ndarray | None,
    d_bar: float,
) -> tuple[np.ndarray, str]:
    """Return deterministic undirected surface edges for patch detection."""

    source = np.asarray(points, dtype=float)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    faces = (
        np.asarray(triangles, dtype=np.int64)
        if triangles is not None
        else np.empty((0, 3), dtype=np.int64)
    )
    if len(faces):
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("triangles must have shape (m, 3)")
        if int(faces.min()) < 0 or int(faces.max()) >= len(source):
            raise ValueError("triangles contain an out-of-range vertex index")
        edges = np.vstack(
            [
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ]
        )
        mode = "triangle_edges"
    else:
        if len(source) < 2:
            return np.empty((0, 2), dtype=np.int64), "no_connectivity"
        spacing = float(d_bar)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("d_bar must be positive and finite")
        neighbour_count = min(8, len(source))
        distances, neighbours = cKDTree(source).query(
            source,
            k=neighbour_count,
            distance_upper_bound=max(3.0 * spacing, 1e-6),
            workers=worker_threads(),
        )
        if neighbour_count == 1:
            distances = distances[:, None]
            neighbours = neighbours[:, None]
        starts = np.repeat(np.arange(len(source), dtype=np.int64), neighbour_count)
        ends = np.asarray(neighbours, dtype=np.int64).reshape(-1)
        finite = np.isfinite(np.asarray(distances, dtype=float).reshape(-1))
        valid = finite & (ends >= 0) & (ends < len(source)) & (starts != ends)
        edges = np.column_stack([starts[valid], ends[valid]])
        mode = "radius_knn"
    if not len(edges):
        return np.empty((0, 2), dtype=np.int64), mode
    edges = np.sort(np.asarray(edges, dtype=np.int64), axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return np.unique(edges, axis=0), mode


def _segment_radius_profile(points, path, spacing):
    """Measure radii against segments, robust to sparse skeleton stations.

    Median radial support suppresses short primary-labelled side protrusions.
    Bins have a physical arc width; adding collinear path nodes changes neither
    the distances nor the radius estimate.
    """
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    stations = np.linspace(0.0, arc[-1], max(2, int(np.ceil(arc[-1] / (4 * spacing))) + 1))
    distances, projected = _polyline_projection_distance_and_arc(points, path)
    bins = np.clip(np.searchsorted((stations[:-1] + stations[1:]) / 2, projected), 0, len(stations) - 1)
    values = np.full(len(stations), np.nan)
    for index in np.unique(bins):
        support = distances[bins == index]
        if len(support) >= 3:
            values[index] = np.median(support)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        values[:] = np.median(distances) if len(distances) else spacing
    else:
        values = np.interp(stations, stations[valid], values[valid])
    padded = np.pad(values, (2, 2), mode="edge")
    values = np.median(np.lib.stride_tricks.sliding_window_view(padded, 5), axis=1)
    return stations, np.maximum(values, spacing)


def _resolve_primary_o1_ownership(
    points: np.ndarray,
    labels: np.ndarray,
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    *,
    d_bar: float,
    triangles: np.ndarray | None = None,
    excluded_mask: np.ndarray | None = None,
    competing_labels: dict[int, tuple[int, int]] | None = None,
) -> tuple[np.ndarray, dict]:
    """Trim short primary protrusions continuous with exposed O1 surfaces.

    All proposals use frozen ownership. Internal connectors are geometric
    paths, not surface seeds: only existing, exposed child support can anchor
    a claim. No other lateral's ownership or centerline is modified. Distances
    and all bounds use the same coordinate system as points and d_bar.
    """
    from scipy.sparse.csgraph import dijkstra

    source = np.asarray(points, dtype=float)
    before = np.asarray(labels, dtype=int)
    primary = np.asarray(primary_path, dtype=float)
    spacing = float(d_bar)
    if source.ndim != 2 or source.shape[1] != 3 or not np.all(np.isfinite(source)):
        raise ValueError("points must contain finite XYZ coordinates")
    if before.shape != (len(source),):
        raise ValueError("labels must contain one value per point")
    if primary.ndim != 2 or primary.shape[1] != 3 or not np.all(np.isfinite(primary)):
        raise ValueError("primary_path must contain finite XYZ coordinates")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("d_bar must be positive and finite")
    excluded = _coerce_exclusion_mask(excluded_mask, len(source))
    resolved = before.copy()
    # Existing assigned editor surfaces above the collar are not an invitation
    # to grow there; uncertain shoot-side points must become unassigned.
    resolved[excluded & (before == -2)] = -1
    report = {
        "policy": "primary-o1-exposed-surface-competition-v1",
        "radius_estimator": "median exact-segment distance in 4*d_bar arc bins; five-bin median",
        "region_radius_rule": "4*parent_radius + 4*child_radius + 8*d_bar",
        "child_arc_bound_rule": "max(6*parent_radius, 4*child_radius, 12*d_bar)",
        "protrusion_span_rule": "max(2*parent_radius, 4*child_radius, 8*d_bar)",
        "geodesic_limit_rule": "max(2*parent_radius, 3*child_radius, 6*d_bar)",
        "primary_mask_lock": False,
        "transferred_vertex_count": 0,
        "excluded_uncertain_to_unassigned_count": int(np.sum(excluded & (before == -2))),
        "multi_claim_vertex_count": 0,
        "ambiguous_claim_vertex_count": 0,
        "junctions": [],
    }
    if len(primary) < 2 or not len(source) or np.sum((before == 0) & ~excluded) < 3:
        return resolved, report
    edges, mode = _surface_connectivity_edges(source, triangles, spacing)
    report["connectivity"] = mode
    if not len(edges):
        return resolved, report
    tree = cKDTree(source)
    parent_stations, parent_radii = _segment_radius_profile(source[(before == 0) & ~excluded], primary, spacing)
    primary_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(primary, axis=0), axis=1))]
    claims = []
    for label, child in enumerate(lateral_paths, start=1):
        if child.order != 1 or child.parent_id != "primary":
            continue
        path = np.asarray(child.points, dtype=float)
        row = {"root_id": str(child.root_id), "label": label, "transferred_vertex_count": 0}
        report["junctions"].append(row)
        support = np.flatnonzero((before == label) & ~excluded)
        if len(path) < 2 or not np.all(np.isfinite(path)) or len(support) < 3:
            row["status"] = "insufficient_child_support"
            continue
        _, insertion_arc = _polyline_projection_distance_and_arc(path[:1], primary)
        rp = float(np.interp(insertion_arc[0], parent_stations, parent_radii))
        child_stations, child_radii = _segment_radius_profile(source[support], path, spacing)
        rc = float(np.median(child_radii[child_stations <= max(6 * rp, 12 * spacing)]))
        region_radius = 4 * rp + 4 * rc + 8 * spacing
        max_arc = max(6 * rp, 4 * rc, 12 * spacing)
        ids = np.asarray(sorted(tree.query_ball_point(path[0], region_radius)), dtype=int)
        ids = ids[~excluded[ids] & np.isin(before[ids], [0, label])]
        row.update(parent_radius=rp, child_radius=rc, region_center=path[0].tolist(),
                   region_radius=region_radius, maximum_child_arc=max_arc,
                   competition_vertex_count=int(len(ids)))
        if not len(ids):
            row["status"] = "empty_region"
            continue
        p = source[ids]
        pd, pa = _polyline_projection_distance_and_arc(p, primary)
        cd, ca = _polyline_projection_distance_and_arc(p, path)
        pr = np.interp(pa, parent_stations, parent_radii)
        cr = np.interp(ca, child_stations, child_radii)
        child_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        child_center = np.column_stack([np.interp(ca, child_arc, path[:, axis]) for axis in range(3)])
        parent_center = np.column_stack([np.interp(pa, primary_arc, primary[:, axis]) for axis in range(3)])
        ci = np.clip(np.searchsorted(child_arc, ca, side="right") - 1, 0, len(path) - 2)
        tangent = path[ci + 1] - path[ci]
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-12)
        radial = p - parent_center
        radial /= np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-12)
        outward = np.einsum("ij,ij->i", tangent, radial)
        axis_distance, _ = _polyline_projection_distance_and_arc(child_center, primary)
        # Tangential emergence is accepted once the child axis has separated;
        # near the parent wall it must point outward from that wall.
        exposed = (axis_distance >= pr - 0.5 * cr) & ((outward >= 0.20) | (axis_distance >= pr + cr))
        geometric = ((ca <= max_arc) & (pd >= pr + 0.25 * spacing)
                     & (cd <= 1.5 * cr + spacing) & exposed)
        strength = pd / (pr + spacing) - cd / (cr + spacing)
        unrelated = np.asarray([
            any(other not in (0, label) for other in (competing_labels or {}).get(int(i), ()))
            for i in ids
        ], dtype=bool)
        row["competition_protected_vertex_count"] = int(np.sum(unrelated & (before[ids] == 0)))
        eligible = geometric & (strength >= 0.20) & ~unrelated
        seeds = (before[ids] == label) & geometric
        row["supported_seed_count"] = int(seeds.sum())
        if seeds.sum() < 3:
            row["status"] = "no_exposed_child_anchor"
            continue
        allowed = seeds | ((before[ids] == 0) & eligible)
        local = np.full(len(source), -1, dtype=int)
        local[ids[allowed]] = np.flatnonzero(allowed)
        e = edges[(local[edges[:, 0]] >= 0) & (local[edges[:, 1]] >= 0)]
        u, v = local[e[:, 0]], local[e[:, 1]]
        lengths = np.linalg.norm(source[e[:, 0]] - source[e[:, 1]], axis=1)
        graph = coo_matrix((np.r_[lengths, lengths], (np.r_[u, v], np.r_[v, u])), shape=(len(ids), len(ids))).tocsr()
        limit = max(2 * rp, 3 * rc, 6 * spacing)
        distance = dijkstra(graph, directed=False, indices=np.flatnonzero(seeds), min_only=True, limit=limit)
        candidates = allowed & (before[ids] == 0) & np.isfinite(distance)
        # Limit each primary protrusion's axial extent, not the entire child's
        # supported body. Large ambiguous strips are retained for review.
        ce = candidates[u] & candidates[v]
        cg = coo_matrix((np.ones(2 * int(ce.sum())), (np.r_[u[ce], v[ce]], np.r_[v[ce], u[ce]])), shape=graph.shape).tocsr()
        _, components = connected_components(cg, directed=False)
        for component in np.unique(components[candidates]):
            members = candidates & (components == component)
            if np.ptp(ca[members]) > max(2 * rp, 4 * rc, 8 * spacing):
                candidates[members] = False
        row["candidate_vertex_count"] = int(candidates.sum())
        row["status"] = "evaluated"
        claims.append((label, ids, candidates, seeds, graph, strength, row))

    best = np.full(len(source), -np.inf)
    second = best.copy()
    winners = np.zeros(len(source), dtype=int)
    counts = np.zeros(len(source), dtype=int)
    for label, ids, candidates, _, _, strength, _ in claims:
        ix = ids[candidates]
        score = strength[candidates]
        better = score > best[ix]
        second[ix] = np.where(better, best[ix], np.maximum(second[ix], score))
        best[ix] = np.maximum(best[ix], score)
        winners[ix[better]] = label
        counts[ix] += 1
    ambiguous = (counts > 1) & ((best - np.where(np.isfinite(second), second, 0)) < 0.15)
    winners[ambiguous] = 0
    report["multi_claim_vertex_count"] = int(np.sum(counts > 1))
    report["ambiguous_claim_vertex_count"] = int(ambiguous.sum())
    for label, ids, candidates, seeds, graph, _, row in claims:
        # Arbitration can remove an intermediate bridge. Recheck connectivity
        # using only this winner's vertices and its original surface anchors.
        allowed = seeds | (candidates & (winners[ids] == label))
        g = graph.tocoo()
        keep = allowed[g.row] & allowed[g.col]
        restricted = coo_matrix((g.data[keep], (g.row[keep], g.col[keep])), shape=g.shape).tocsr()
        _, components = connected_components(restricted, directed=False)
        reachable = np.isin(components, np.unique(components[seeds]))
        transfer = ids[candidates & (winners[ids] == label) & reachable]
        resolved[transfer] = label
        row["transferred_vertex_count"] = int(len(transfer))
    report["transferred_vertex_count"] = int(np.sum((before == 0) & (resolved > 0)))
    return resolved, report


def _absorb_small_primary_surface_patches(
    points: np.ndarray,
    labels: np.ndarray,
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    *,
    d_bar: float,
    triangles: np.ndarray | None = None,
    excluded_mask: np.ndarray | None = None,
    primary_support_points: np.ndarray | None = None,
    competing_labels: dict[int, tuple[int, int]] | None = None,
    root_radii: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Absorb discrete non-primary islands embedded in the primary surface.

    A patch must be a small same-label surface component, be surrounded mostly
    by primary-labelled neighbours, and lie inside the measured primary tube.
    The largest surface component of every order-1 root is always retained, as
    are all explicitly excluded points above the selected base.
    """

    source = np.asarray(points, dtype=float)
    resolved = np.asarray(labels, dtype=int).copy()
    primary = np.asarray(primary_path, dtype=float)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if resolved.shape != (len(source),):
        raise ValueError("labels must contain one value per point")
    if primary.ndim != 2 or primary.shape[1] != 3:
        raise ValueError("primary_path must contain XYZ points")
    spacing = float(d_bar)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("d_bar must be positive and finite")
    excluded = _coerce_exclusion_mask(excluded_mask, len(source))
    protected = np.zeros(len(source), dtype=bool)
    for vertex, pair in (competing_labels or {}).items():
        if resolved[vertex] == -2 and pair[0] != pair[1]:
            protected[vertex] = True
    order_one_labels = {
        index
        for index, path in enumerate(lateral_paths, start=1)
        if int(path.order) == 1
    }
    target_labels = {-2, -1, *order_one_labels}
    edges, connectivity = _surface_connectivity_edges(source, triangles, spacing)
    report = {
        "policy": "primary-surface-small-patch-cleanup-v2",
        "rule": (
            "A non-main order-1, uncertain, or unassigned same-label surface "
            "component is assigned to primary only when at least 75% of its "
            "boundary edges meet primary, at least 90% of its vertices lie "
            "inside the local primary-radius envelope, and its spatial span is "
            "no larger than max(8*d_bar, 0.75*local_primary_radius, 0.006). "
            "The largest component of each order-1 root and every explicitly "
            "excluded above-base vertex are retained. Unresolved distinct-root competition is protected."
        ),
        "connectivity": connectivity,
        "absorbed_patch_count": 0,
        "absorbed_vertex_count": 0,
        "passes": 0,
        "competition_protected_vertex_count": int(protected.sum()),
        "per_source_label": {},
    }
    if not len(source) or not len(primary) or not len(edges):
        return resolved, report

    support = (
        np.asarray(primary_support_points, dtype=float)
        if primary_support_points is not None
        else source[(resolved == 0) & ~excluded]
    )
    if support.ndim != 2 or support.shape[1] != 3:
        raise ValueError("primary_support_points must contain XYZ points")
    if root_radii is not None and 0 in root_radii:
        primary_radii = root_radii[0]
        stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(primary, axis=0), axis=1))]
    else:
        stations, primary_radii = _segment_radius_profile(support, primary, spacing)
    root_ids = ["primary"] + [str(path.root_id) for path in lateral_paths]

    for pass_index in range(3):
        eligible = ~excluded & ~protected & np.isin(
            resolved,
            np.asarray(sorted(target_labels), dtype=int),
        )
        eligible_indices = np.flatnonzero(eligible)
        if not len(eligible_indices):
            break
        local_index = np.full(len(source), -1, dtype=np.int64)
        local_index[eligible_indices] = np.arange(len(eligible_indices), dtype=np.int64)
        edge_starts = edges[:, 0]
        edge_ends = edges[:, 1]
        same_label_edges = (
            eligible[edge_starts]
            & eligible[edge_ends]
            & (resolved[edge_starts] == resolved[edge_ends])
        )
        rows = local_index[edge_starts[same_label_edges]]
        columns = local_index[edge_ends[same_label_edges]]
        if len(rows):
            graph_rows = np.concatenate([rows, columns])
            graph_columns = np.concatenate([columns, rows])
            graph = coo_matrix(
                (
                    np.ones(len(graph_rows), dtype=np.uint8),
                    (graph_rows, graph_columns),
                ),
                shape=(len(eligible_indices), len(eligible_indices)),
            ).tocsr()
            component_count, component_labels = connected_components(
                graph,
                directed=False,
                return_labels=True,
            )
        else:
            component_count = len(eligible_indices)
            component_labels = np.arange(len(eligible_indices), dtype=np.int32)
        counts = np.bincount(
            component_labels,
            minlength=component_count,
        ).astype(np.int64, copy=False)
        component_source_labels = np.zeros(component_count, dtype=int)
        component_source_labels[component_labels] = resolved[eligible_indices]
        component_by_vertex = np.full(len(source), -1, dtype=np.int64)
        component_by_vertex[eligible_indices] = component_labels

        component_start = component_by_vertex[edge_starts]
        component_end = component_by_vertex[edge_ends]
        start_boundary = (component_start >= 0) & (component_start != component_end)
        end_boundary = (component_end >= 0) & (component_start != component_end)
        boundary_components = np.concatenate(
            [component_start[start_boundary], component_end[end_boundary]]
        )
        boundary_neighbour_labels = np.concatenate(
            [resolved[edge_ends[start_boundary]], resolved[edge_starts[end_boundary]]]
        )
        boundary_counts = np.bincount(
            boundary_components,
            minlength=component_count,
        )
        primary_boundary_counts = np.bincount(
            boundary_components,
            weights=(boundary_neighbour_labels == 0).astype(float),
            minlength=component_count,
        )

        largest_order_one_component: dict[int, int] = {}
        for component in range(component_count):
            source_label = int(component_source_labels[component])
            if source_label not in order_one_labels:
                continue
            previous = largest_order_one_component.get(source_label)
            if previous is None or counts[component] > counts[previous]:
                largest_order_one_component[source_label] = component

        membership_order = np.argsort(component_labels, kind="stable")
        offsets = np.concatenate(
            [np.array([0], dtype=np.int64), np.cumsum(counts)]
        )
        absorb: list[tuple[int, np.ndarray]] = []
        for component in range(component_count):
            source_label = int(component_source_labels[component])
            if (
                source_label in order_one_labels
                and largest_order_one_component.get(source_label) == component
            ):
                continue
            boundary_count = int(boundary_counts[component])
            if boundary_count < 2:
                continue
            primary_boundary_fraction = float(
                primary_boundary_counts[component] / boundary_count
            )
            if primary_boundary_fraction < 0.75:
                continue
            members = eligible_indices[
                membership_order[offsets[component] : offsets[component + 1]]
            ]
            positions = source[members]
            patch_span = float(np.linalg.norm(np.ptp(positions, axis=0)))
            distances, arcs = _polyline_projection_distance_and_arc(positions, primary)
            local_radii = np.interp(arcs, stations, primary_radii)
            local_radius = float(np.median(local_radii))
            maximum_span = max(8.0 * spacing, 0.75 * local_radius, 0.006)
            if patch_span > maximum_span:
                continue
            inside_fraction = float(
                np.mean(
                    np.asarray(distances, dtype=float)
                    <= local_radii + max(2.0 * spacing, 0.002)
                )
            )
            if inside_fraction < 0.90:
                continue
            absorb.append((source_label, members))

        if not absorb:
            break
        report["passes"] = pass_index + 1
        for source_label, members in absorb:
            resolved[members] = 0
            if source_label == -2:
                source_name = "uncertain"
            elif source_label == -1:
                source_name = "unassigned"
            elif 0 < source_label < len(root_ids):
                source_name = root_ids[source_label]
            else:
                source_name = str(source_label)
            entry = report["per_source_label"].setdefault(
                source_name,
                {"patch_count": 0, "vertex_count": 0},
            )
            entry["patch_count"] += 1
            entry["vertex_count"] += int(len(members))
            report["absorbed_patch_count"] += 1
            report["absorbed_vertex_count"] += int(len(members))
    report["per_source_label"] = dict(sorted(report["per_source_label"].items()))
    return resolved, report


def _competing_pair_supports_parent_claim(
    pair: tuple[int, int],
    *,
    parent_label: int,
    child_label: int,
    parent_by_label: dict[int, int],
) -> bool:
    """Return whether the observed competitors belong to this junction."""

    first, second = (int(pair[0]), int(pair[1]))
    if first == second:
        return False
    if first == child_label:
        other = second
    elif second == child_label:
        other = first
    else:
        other = None
    if other is not None:
        return (
            other == parent_label
            or parent_by_label.get(other) == parent_label
            or _root_label_is_ancestor(other, parent_label, parent_by_label)
            or _root_label_is_ancestor(child_label, other, parent_by_label)
        )

    if first == parent_label:
        other = second
    elif second == parent_label:
        other = first
    else:
        return False
    return (
        other == child_label
        or parent_by_label.get(other) == parent_label
        or _root_label_is_ancestor(child_label, other, parent_by_label)
    )


def _root_label_is_ancestor(
    possible_ancestor: int,
    descendant: int,
    parent_by_label: dict[int, int],
) -> bool:
    current = int(descendant)
    seen: set[int] = set()
    while current in parent_by_label and current not in seen:
        seen.add(current)
        current = int(parent_by_label[current])
        if current == int(possible_ancestor):
            return True
    return False


def _resolve_parent_owned_junctions(
    points: np.ndarray,
    labels: np.ndarray,
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    *,
    d_bar: float,
    assignment_radius: float,
    ambiguity_margin: float,
    competing_labels: dict[int, tuple[int, int]] | None = None,
    root_radii: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Assign only branch-site uncertainty to the biological parent root.

    Parent and child skeletons intentionally share their insertion coordinate.
    A distinct-root ambiguity rule therefore creates a short orange band even
    when the topology is known.  This postprocessor uses the observed competing
    root labels, repaired topology, and exact point-to-segment projections to
    resolve only uncertain vertices within a sampling-scaled basal child
    prefix. Unassigned and excluded vertices are never reclaimed.
    """

    source = np.asarray(points, dtype=float)
    resolved = np.asarray(labels, dtype=int).copy()
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if resolved.shape != (len(source),):
        raise ValueError("labels must contain one value per point")
    spacing = float(d_bar)
    radius = float(assignment_radius)
    margin = float(ambiguity_margin)
    if (
        not np.isfinite(spacing)
        or not np.isfinite(radius)
        or not np.isfinite(margin)
        or spacing <= 0.0
        or radius <= 0.0
        or margin < 0.0
    ):
        raise ValueError("junction assignment scales must be non-negative and finite")

    root_ids = ["primary"] + [str(path.root_id) for path in lateral_paths]
    root_id_to_label = {root_id: index for index, root_id in enumerate(root_ids)}
    root_paths = [np.asarray(primary_path, dtype=float)] + [
        np.asarray(path.points, dtype=float) for path in lateral_paths
    ]
    parent_by_label: dict[int, int] = {}
    order_by_label: dict[int, int] = {0: 0}
    for child_label, path in enumerate(lateral_paths, start=1):
        parent_label = root_id_to_label.get(str(path.parent_id))
        if parent_label is not None:
            parent_by_label[child_label] = int(parent_label)
        order_by_label[child_label] = int(path.order)

    uncertain_indices = np.flatnonzero(resolved == -2)
    report = {
        "policy": "topology-aware-parent-owned-junction-v3",
        "rule": (
            "Only uncertain vertices whose observed competing roots belong to "
            "the repaired parent-child junction, and whose segment distances "
            "remain ambiguous relative to local radii, are assigned to the parent. The prefix length is "
            "the larger of four mean point spacings and the local parent radius, "
            "capped by child length; no physical-distance threshold is used."
        ),
        "resolved_vertex_count": 0,
        "remaining_uncertain_vertex_count": int(len(uncertain_indices)),
        "per_parent_vertex_count": {},
        "junction_arc_count": 0,
        "junction_arc_min_normalized": 0.0,
        "junction_arc_median_normalized": 0.0,
        "junction_arc_max_normalized": 0.0,
        "multi_claim_vertex_count": 0,
    }
    if not len(uncertain_indices) or not lateral_paths:
        return resolved, report

    observed_competitors = competing_labels or {}
    uncertain_tree = cKDTree(source[uncertain_indices])
    parent_radius_cache: dict[int, np.ndarray] = {}
    parent_support_indices = _group_indices_by_label(
        resolved,
        set(parent_by_label.values()),
    )
    empty_indices = np.empty(0, dtype=int)
    claims: dict[int, list[tuple[int, int, float, float, float]]] = {}
    junction_arcs: list[float] = []

    for child_label, child in enumerate(lateral_paths, start=1):
        parent_label = parent_by_label.get(child_label)
        if parent_label is None:
            continue
        parent = root_paths[parent_label]
        child_points = root_paths[child_label]
        if len(parent) == 0 or len(child_points) < 2:
            continue
        insertion = (
            np.asarray(child.insertion_point, dtype=float)
            if child.insertion_point is not None
            else child_points[0]
        )
        if insertion.shape != (3,) or not np.all(np.isfinite(insertion)):
            continue
        if parent_label not in parent_radius_cache:
            parent_support = source[parent_support_indices.get(parent_label, empty_indices)]
            stations, values = _segment_radius_profile(parent_support, parent, spacing)
            parent_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(parent, axis=0), axis=1))]
            parent_radius_cache[parent_label] = np.interp(parent_arc, stations, values)
        parent_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(parent, axis=0), axis=1))]
        parent_radii = (root_radii or {}).get(parent_label, parent_radius_cache[parent_label])
        _, insertion_arc = _polyline_projection_distance_and_arc(insertion[None], parent)
        local_parent_radius = float(np.interp(insertion_arc[0], parent_arc, parent_radii))
        # Missing profiles follow assignment's zero/mean-radius fallback.
        parent_metric_radii = (root_radii or {}).get(
            parent_label, 0.0 if parent_label == 0 else _valid_root_radius(lateral_paths[parent_label - 1].mean_radius),
        )
        child_metric_radii = (root_radii or {}).get(child_label, _valid_root_radius(child.mean_radius))
        child_length = float(
            np.linalg.norm(np.diff(child_points, axis=0), axis=1).sum()
        )
        junction_arc = min(
            child_length,
            max(4.0 * spacing, local_parent_radius),
        )
        if not np.isfinite(junction_arc) or junction_arc <= 1e-12:
            continue
        junction_arcs.append(float(junction_arc))

        prefilter_radius = float(
            junction_arc + radius + margin + max(
                float(np.max(parent_metric_radii)),
                float(np.max(child_metric_radii)),
            )
        )
        nearby_positions = uncertain_tree.query_ball_point(
            insertion,
            r=prefilter_radius,
            workers=worker_threads(),
        )
        if not nearby_positions:
            continue
        candidate_indices = uncertain_indices[np.asarray(nearby_positions, dtype=int)]
        candidate_points = source[candidate_indices]
        child_distance, child_arc = _polyline_projection_distance_and_arc(candidate_points, child_points)
        parent_distance, _ = _polyline_projection_distance_and_arc(candidate_points, parent)
        # Use the same frozen surface metric that produced the evidence.
        metric = RootSegmentIndex(
            [parent, child_points], labels=[parent_label, child_label],
            radii=[parent_metric_radii, child_metric_radii],
        ).query(candidate_points, max_distance=radius + margin)
        surface_ambiguous = metric.ambiguous(margin)
        eligible = (
            (child_distance <= radius + float(np.max(child_metric_radii)))
            & (child_arc <= junction_arc + 1e-12)
            & (parent_distance <= radius + margin + float(np.max(parent_metric_radii)))
            & surface_ambiguous
        )
        for candidate_position in np.flatnonzero(eligible):
            vertex_index = int(candidate_indices[candidate_position])
            pair = observed_competitors.get(vertex_index)
            if pair is None or not _competing_pair_supports_parent_claim(
                pair,
                parent_label=int(parent_label),
                child_label=int(child_label),
                parent_by_label=parent_by_label,
            ):
                continue
            insertion_distance = float(
                np.linalg.norm(candidate_points[candidate_position] - insertion)
            )
            distance_mismatch = float(
                abs(
                    parent_distance[candidate_position]
                    - child_distance[candidate_position]
                )
            )
            combined_distance = float(
                parent_distance[candidate_position]
                + child_distance[candidate_position]
            )
            claims.setdefault(vertex_index, []).append(
                (
                    int(parent_label),
                    int(child_label),
                    insertion_distance,
                    distance_mismatch,
                    combined_distance,
                )
            )

    assigned_parent: dict[int, int] = {}
    multi_claim_vertex_count = 0
    for vertex_index, vertex_claims in claims.items():
        if len(vertex_claims) > 1:
            multi_claim_vertex_count += 1
        best_claim = min(
            vertex_claims,
            key=lambda claim: (
                -order_by_label.get(claim[0], -1),
                claim[2],
                claim[3],
                claim[4],
                claim[0],
                claim[1],
            ),
        )
        assigned_parent[vertex_index] = int(best_claim[0])

    per_parent: dict[str, int] = {}
    for vertex_index, parent_label in assigned_parent.items():
        resolved[vertex_index] = int(parent_label)
        parent_id = root_ids[parent_label]
        per_parent[parent_id] = per_parent.get(parent_id, 0) + 1

    report["resolved_vertex_count"] = int(len(assigned_parent))
    report["remaining_uncertain_vertex_count"] = int(np.count_nonzero(resolved == -2))
    report["per_parent_vertex_count"] = dict(sorted(per_parent.items()))
    report["multi_claim_vertex_count"] = int(multi_claim_vertex_count)
    report["junction_arc_count"] = int(len(junction_arcs))
    if junction_arcs:
        arcs = np.asarray(junction_arcs, dtype=float)
        report["junction_arc_min_normalized"] = float(np.min(arcs))
        report["junction_arc_median_normalized"] = float(np.median(arcs))
        report["junction_arc_max_normalized"] = float(np.max(arcs))
    return resolved, report


def _polyline_prefix(path: np.ndarray, maximum_arc: float) -> np.ndarray:
    """Return a polyline ending exactly at a requested positive arc length."""

    polyline = np.asarray(path, dtype=float)
    if len(polyline) <= 1:
        return polyline.copy()
    keep = np.concatenate(
        [[True], np.linalg.norm(np.diff(polyline, axis=0), axis=1) > 1e-12]
    )
    polyline = polyline[keep]
    if len(polyline) <= 1:
        return polyline.copy()
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    target = min(max(float(maximum_arc), 0.0), float(cumulative[-1]))
    if target >= cumulative[-1] - 1e-12:
        return polyline.copy()
    segment_index = int(np.searchsorted(cumulative, target, side="right") - 1)
    segment_index = min(max(segment_index, 0), len(lengths) - 1)
    segment_length = float(lengths[segment_index])
    if segment_length <= 1e-12:
        return polyline[: segment_index + 2].copy()
    fraction = (target - float(cumulative[segment_index])) / segment_length
    endpoint = (
        polyline[segment_index]
        + float(np.clip(fraction, 0.0, 1.0))
        * (polyline[segment_index + 1] - polyline[segment_index])
    )
    return np.vstack([polyline[: segment_index + 1], endpoint])


def _polyline_projection_distance_and_arc(
    query_points: np.ndarray,
    path: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact nearest-segment distance and projected path arc."""

    query = np.asarray(query_points, dtype=float)
    polyline = np.asarray(path, dtype=float)
    if query.ndim != 2 or query.shape[1] != 3:
        raise ValueError("query_points must have shape (n, 3)")
    if len(polyline) == 0:
        return (
            np.full(len(query), np.inf, dtype=float),
            np.zeros(len(query), dtype=float),
        )
    if len(polyline) == 1:
        return (
            np.linalg.norm(query - polyline[0], axis=1),
            np.zeros(len(query), dtype=float),
        )
    starts = polyline[:-1]
    vectors = np.diff(polyline, axis=0)
    squared_lengths = np.sum(vectors * vectors, axis=1)
    segment_lengths = np.sqrt(squared_lengths)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    distances = np.full(len(query), np.inf, dtype=float)
    arcs = np.zeros(len(query), dtype=float)
    chunk_size = max(128, min(8192, 1_000_000 // max(1, len(starts))))
    for chunk_start in range(0, len(query), chunk_size):
        chunk = query[chunk_start : chunk_start + chunk_size]
        offsets = chunk[:, None, :] - starts[None, :, :]
        parameters = np.zeros((len(chunk), len(starts)), dtype=float)
        valid = squared_lengths > 1e-20
        if np.any(valid):
            parameters[:, valid] = np.clip(
                np.einsum(
                    "nkj,kj->nk",
                    offsets[:, valid, :],
                    vectors[valid],
                )
                / squared_lengths[valid][None, :],
                0.0,
                1.0,
            )
        projections = starts[None, :, :] + parameters[:, :, None] * vectors[None, :, :]
        candidate_distances = np.linalg.norm(
            chunk[:, None, :] - projections,
            axis=2,
        )
        best_segment = np.argmin(candidate_distances, axis=1)
        rows = np.arange(len(chunk))
        best_distance = candidate_distances[rows, best_segment]
        best_arc = (
            cumulative[best_segment]
            + parameters[rows, best_segment] * segment_lengths[best_segment]
        )
        distances[chunk_start : chunk_start + len(chunk)] = best_distance
        arcs[chunk_start : chunk_start + len(chunk)] = best_arc
    return distances, arcs


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
    analysis_junction_report: dict,
    full_junction_report: dict,
    primary_surface_patch_report: dict,
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
        "parent_owned_junction_assignment": {
            "analysis": analysis_junction_report,
            "full_resolution": full_junction_report,
        },
        "primary_surface_patch_cleanup": primary_surface_patch_report,
    }


def _assign_full_root_labels(
    points: np.ndarray,
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    *,
    d_bar: float,
    excluded_mask: np.ndarray | None = None,
    return_competing_labels: bool = False,
    root_radii: dict | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[int, tuple[int, int]]]:
    radius, margin = max(5.0 * d_bar, 0.008), max(0.75 * d_bar, 0.001)
    excluded = _coerce_exclusion_mask(excluded_mask, len(points))
    ids = np.flatnonzero(~excluded)
    result = _root_segment_index(primary_path, lateral_paths, root_radii).query(points[ids], max_distance=radius + margin)
    labels = np.full(len(points), -1, dtype=int)
    assigned = result.distances[:, 0] <= radius + TIE_TOLERANCE
    labels[ids[assigned]] = result.labels[assigned, 0]
    ambiguous = assigned & result.ambiguous(margin)
    labels[ids[ambiguous]] = -2
    pairs = {int(i): tuple(map(int, pair)) for i, pair in zip(ids[ambiguous], result.labels[ambiguous], strict=True)}
    return (labels, pairs) if return_competing_labels else labels


def _update_metadata_timings(path: Path, timings: dict[str, float]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["stage_timings_seconds"] = timings
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


