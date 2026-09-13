from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from .geometry import nearest_path_tangent, point_to_polyline_distance, resample_polyline, tangent_vectors
from .primary import cluster_hdbscan
from .types import RootPath
from .runtime import worker_threads


@dataclass
class LateralStart:
    start_id: int
    point: np.ndarray
    primary_point: np.ndarray
    primary_index: int
    member_indices: np.ndarray
    direction: np.ndarray | None = None
    radial_direction: np.ndarray | None = None
    extent_direction: np.ndarray | None = None


@dataclass
class _TraceState:
    """One live lateral-tracing hypothesis retained by the bounded beam."""

    nodes: list[np.ndarray]
    current: np.ndarray
    direction: np.ndarray
    covered: set[int]
    growth_arc: float = 0.0
    supported_arc: float = 0.0
    support_sum: float = 0.0
    turn_squared_sum: float = 0.0
    cumulative_turn_degrees: float = 0.0
    cumulative_score: float = 0.0
    local_radius: float = np.nan
    radius_similarity_sum: float = 0.0
    radius_observations: int = 0
    steps: int = 0


def estimate_parent_radius_profile(
    parent_path: np.ndarray,
    parent_support_points: np.ndarray,
    d_bar: float,
) -> np.ndarray:
    """Estimate a robust local surface radius at every parent-path node.

    The profile is intentionally based only on points already owned by the
    parent.  A moderate quantile is used so a child surface at a junction does
    not inflate the envelope, while interpolation and a short median smooth
    make the estimate usable at sparsely sampled stations.
    """

    parent_path = np.asarray(parent_path, dtype=float)
    support = np.asarray(parent_support_points, dtype=float)
    if len(parent_path) == 0:
        return np.empty(0, dtype=float)
    radius_floor = max(2.5 * float(d_bar), 0.002)
    if support.ndim != 2 or support.shape[1] != 3 or len(support) < 3:
        return np.full(len(parent_path), radius_floor, dtype=float)

    distances, nearest = cKDTree(parent_path).query(support, k=1, workers=worker_threads())
    profile = np.full(len(parent_path), np.nan, dtype=float)
    for node_index in np.unique(nearest):
        values = distances[nearest == node_index]
        if len(values) >= 3:
            profile[int(node_index)] = float(np.quantile(values, 0.70))

    valid = np.flatnonzero(np.isfinite(profile))
    if not len(valid):
        fallback = max(radius_floor, float(np.quantile(distances, 0.70)))
        return np.full(len(parent_path), fallback, dtype=float)
    if len(valid) == 1:
        profile[:] = profile[valid[0]]
    else:
        node_axis = np.arange(len(parent_path), dtype=float)
        profile = np.interp(node_axis, valid.astype(float), profile[valid])

    padded = np.pad(profile, (2, 2), mode="edge")
    smoothed = np.asarray([np.median(padded[index : index + 5]) for index in range(len(profile))])
    return np.maximum(smoothed, radius_floor)


def is_parent_tracking_candidate(
    path: RootPath,
    parent_path: np.ndarray,
    parent_radius_profile: np.ndarray,
    d_bar: float,
) -> tuple[bool, dict[str, float]]:
    """Return whether a candidate is an offset trace of its parent surface.

    Real laterals may share a basal insertion and may briefly curl toward the
    collar, so insertion height alone is never a rejection reason.  A path is
    rejected only after a parent-radius-sized attachment region when it keeps
    tracking the parent envelope and either runs collarward or remains strongly
    parallel.  Sustained terminal escape always preserves the candidate.
    """

    parent_path = np.asarray(parent_path, dtype=float)
    radii = np.asarray(parent_radius_profile, dtype=float)
    if len(path.points) < 3 or len(parent_path) < 2 or radii.shape != (len(parent_path),):
        return False, {}

    spacing = max(4.0 * float(d_bar), 0.002)
    sampled = resample_polyline(path.points, spacing=spacing)
    if len(sampled) < 3:
        return False, {}
    parent_tree = cKDTree(parent_path)
    distances, nearest = parent_tree.query(sampled, k=1, workers=worker_threads())
    nearest = np.asarray(nearest, dtype=int)

    segment_lengths = np.linalg.norm(np.diff(sampled, axis=0), axis=1)
    child_arc = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(child_arc[-1])
    attachment_radius = float(radii[nearest[0]])
    attachment_arc = max(8.0 * float(d_bar), 4.0 * attachment_radius)
    evidence_start = int(np.searchsorted(child_arc, attachment_arc, side="left"))
    evidence_start = min(evidence_start, len(sampled) - 1)
    evidence_length = max(0.0, total_length - float(child_arc[evidence_start]))
    minimum_evidence = max(8.0 * float(d_bar), 4.0 * attachment_radius)

    envelope = np.maximum(
        2.0 * radii[nearest] + 2.0 * float(d_bar),
        max(8.0 * float(d_bar), 0.006),
    )
    evidence = np.arange(evidence_start, len(sampled), dtype=int)
    inside = distances[evidence] <= envelope[evidence]
    inside_fraction = float(np.mean(inside)) if len(inside) else 0.0

    child_tangents = tangent_vectors(sampled)
    parent_tangents = tangent_vectors(parent_path)
    alignment = np.abs(np.sum(child_tangents[evidence] * parent_tangents[nearest[evidence]], axis=1))
    parallel_inside = inside & (alignment >= np.cos(np.radians(30.0)))
    parallel_fraction = float(np.count_nonzero(parallel_inside) / max(1, np.count_nonzero(inside)))

    parent_segments = np.linalg.norm(np.diff(parent_path, axis=0), axis=1)
    parent_arc = np.concatenate([[0.0], np.cumsum(parent_segments)])
    tail_count = max(3, int(np.ceil(0.20 * len(evidence))))
    terminal_parent_arc = float(np.median(parent_arc[nearest[evidence[-tail_count:]]]))
    collarward_progress = max(0.0, float(parent_arc[nearest[evidence[0]]] - terminal_parent_arc))
    collarward_threshold = max(4.0 * float(d_bar), 0.20 * evidence_length)

    terminal_window = max(8.0 * float(d_bar), 4.0 * attachment_radius)
    terminal_start_arc = max(float(child_arc[evidence_start]), total_length - terminal_window)
    terminal = np.flatnonzero(child_arc >= terminal_start_arc)
    terminal_outside = distances[terminal] > envelope[terminal]
    terminal_outside_fraction = float(np.mean(terminal_outside)) if len(terminal) else 0.0
    split = max(1, len(terminal) // 2)
    early_terminal_distance = float(np.median(distances[terminal[:split]]))
    late_terminal_distance = float(np.median(distances[terminal[split:]])) if len(terminal[split:]) else early_terminal_distance
    terminal_separation_gain = late_terminal_distance - early_terminal_distance
    sustained_terminal_escape = bool(
        terminal_outside_fraction >= 0.70
        and terminal_separation_gain >= 2.0 * float(d_bar)
    )

    enough_evidence = evidence_length >= minimum_evidence
    net_direction = sampled[-1] - sampled[0]
    net_direction /= max(float(np.linalg.norm(net_direction)), 1e-12)
    signed_parent_alignment = float(
        np.dot(
            net_direction,
            parent_tangents[int(nearest[0])],
        )
    )
    basal_attachment_limit = max(
        8.0 * float(d_bar),
        1.5 * attachment_radius,
    )
    basal_reverse_stub = bool(
        float(parent_arc[int(nearest[0])]) <= basal_attachment_limit
        and signed_parent_alignment
        <= -float(np.cos(np.radians(40.0)))
    )
    # A short trace is conclusive only when it begins at the parent's basal
    # junction and runs backward toward the ancestor while remaining inside the
    # parent envelope.  The signed reverse-direction requirement protects a
    # genuine short orthogonal child that has not yet travelled a full parent
    # radius.
    contained_without_escape = bool(
        not enough_evidence
        and inside_fraction >= 0.90
        and terminal_outside_fraction <= 0.10
        and terminal_separation_gain <= 2.0 * float(d_bar)
        and basal_reverse_stub
    )
    # Collar-returning surface traces can briefly ride just outside the robust
    # diameter envelope at a flared crown.  The strong directed return along
    # the parent supplies the additional evidence, so this arm is deliberately
    # a little more permissive than the purely parallel-tracking arm below.
    collar_tracking = inside_fraction >= 0.65 and collarward_progress >= collarward_threshold
    parallel_tracking = inside_fraction >= 0.90 and parallel_fraction >= 0.65
    rejected = bool(
        enough_evidence
        and not sustained_terminal_escape
        and (collar_tracking or parallel_tracking)
    )
    metrics = {
        "parent_attachment_radius": attachment_radius,
        "parent_envelope_fraction": inside_fraction,
        "parent_parallel_fraction": parallel_fraction,
        "parent_collarward_progress": collarward_progress,
        "parent_terminal_outside_fraction": terminal_outside_fraction,
        "parent_terminal_separation_gain": terminal_separation_gain,
        "parent_signed_basal_alignment": signed_parent_alignment,
        "parent_basal_reverse_stub": float(basal_reverse_stub),
        "parent_short_contained_without_escape": float(
            contained_without_escape
        ),
        "parent_tracking_rejected": float(rejected),
    }
    return rejected, metrics


def find_lateral_starting_points(
    points: np.ndarray,
    primary_mask: np.ndarray,
    primary_path: np.ndarray,
    closest_fraction: float = 0.03,
    min_cluster_size: int = 8,
    max_parent_distance: float | np.ndarray | None = None,
    minimum_branch_angle_degrees: float = 0.0,
    exclude_parent_tip_fraction: float = 0.0,
) -> list[LateralStart]:
    """Cluster non-primary points closest to the primary root as branch starts.

    The closest_fraction parameter is interpreted as a percentile distance
    threshold, matching the paper-inspired nearest-boundary seed step.
    """
    non_primary = np.flatnonzero(~primary_mask)
    if len(non_primary) == 0:
        return []
    distances, primary_indices = point_to_polyline_distance(points[non_primary], primary_path)
    eligible = np.ones(len(distances), dtype=bool)
    if max_parent_distance is not None:
        parent_distance_limit = np.asarray(max_parent_distance, dtype=float)
        if parent_distance_limit.ndim == 0:
            limits = np.full(len(distances), float(parent_distance_limit), dtype=float)
        elif parent_distance_limit.shape == (len(primary_path),):
            limits = parent_distance_limit[np.asarray(primary_indices, dtype=int)]
        else:
            raise ValueError(
                "max_parent_distance must be a scalar or contain one value per parent-path node"
            )
        eligible &= distances <= limits
        if np.count_nonzero(eligible) < max(2, int(min_cluster_size)):
            return []
    if max_parent_distance is not None:
        # The absolute biological attachment gate already removes remote
        # residual roots.  Keep every supported junction so percentile ranking
        # cannot erase laterals whose surface happens to lie slightly farther
        # from the parent centerline.
        seed_local = np.flatnonzero(eligible)
    else:
        percentile = float(np.clip(closest_fraction, 0.001, 1.0) * 100.0)
        threshold = float(np.percentile(distances[eligible], percentile))
        seed_local = np.flatnonzero(eligible & (distances <= threshold))
        minimum_seed_support = min(
            int(np.count_nonzero(eligible)),
            max(2, 3 * int(min_cluster_size)),
        )
        if len(seed_local) < minimum_seed_support:
            eligible_indices = np.flatnonzero(eligible)
            nearest = np.argsort(distances[eligible_indices])[:minimum_seed_support]
            seed_local = eligible_indices[nearest]
    seed_indices = non_primary[seed_local]
    if len(seed_indices) < 2:
        return []
    labels = cluster_hdbscan(points[seed_indices], min_cluster_size=min(min_cluster_size, max(2, len(seed_indices) // 2)))
    starts: list[LateralStart] = []
    primary_tree = cKDTree(primary_path)
    primary_tangents = tangent_vectors(primary_path)
    for label in sorted(label for label in np.unique(labels) if label >= 0):
        members = seed_indices[labels == label]
        if len(members) == 0:
            continue
        cluster_points = points[members]
        distances_to_primary, primary_matches = primary_tree.query(cluster_points, k=1)
        best_member = int(np.argmin(distances_to_primary))
        start_point = cluster_points[best_member]
        primary_idx = int(primary_matches[best_member])
        tip_guard_nodes = int(np.ceil(float(exclude_parent_tip_fraction) * len(primary_path)))
        if tip_guard_nodes > 0 and primary_idx >= len(primary_path) - tip_guard_nodes:
            continue
        radial_direction = start_point - primary_path[primary_idx]
        radial_norm = float(np.linalg.norm(radial_direction))
        radial_unit = (
            radial_direction / radial_norm
            if radial_norm > 1e-12
            else np.zeros(3, dtype=float)
        )
        farthest = cluster_points[
            int(np.argmax(np.linalg.norm(cluster_points - primary_path[primary_idx], axis=1)))
        ]
        extent_direction = farthest - primary_path[primary_idx]
        extent_norm = float(np.linalg.norm(extent_direction))
        extent_unit = (
            extent_direction / extent_norm
            if extent_norm > 1e-12
            else np.zeros(3, dtype=float)
        )
        centered = cluster_points - np.mean(cluster_points, axis=0)
        if len(cluster_points) >= 3 and np.any(np.ptp(cluster_points, axis=0) > 1e-12):
            _, _, axes = np.linalg.svd(centered, full_matrices=False)
            direction = axes[0]
            # PCA axes have arbitrary sign.  Orient them with the local parent
            # surface normal, not the farthest cluster member: a large merged
            # junction cluster can extend farther on the opposite branch and
            # otherwise point every trace back into the collar.
            if radial_norm > 1e-12 and np.dot(direction, radial_unit) < 0:
                direction = -direction
        else:
            direction = radial_direction
        direction_norm = max(float(np.linalg.norm(direction)), 1e-12)
        pca_branch_angle = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        abs(np.dot(direction / direction_norm, primary_tangents[primary_idx])),
                        0.0,
                        1.0,
                    )
                )
            )
        )
        radial_branch_angle = (
            float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            abs(np.dot(radial_unit, primary_tangents[primary_idx])),
                            0.0,
                            1.0,
                        )
                    )
                )
            )
            if radial_norm > 1e-12
            else 0.0
        )
        extent_branch_angle = (
            float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            abs(np.dot(extent_unit, primary_tangents[primary_idx])),
                            0.0,
                            1.0,
                        )
                    )
                )
            )
            if extent_norm > 1e-12
            else 0.0
        )
        branch_angle = max(pca_branch_angle, radial_branch_angle, extent_branch_angle)
        if branch_angle < float(minimum_branch_angle_degrees):
            continue
        starts.append(
            LateralStart(
                start_id=len(starts),
                point=start_point,
                primary_point=primary_path[primary_idx],
                primary_index=primary_idx,
                member_indices=members,
                direction=direction,
                radial_direction=radial_unit if radial_norm > 1e-12 else None,
                extent_direction=extent_unit if extent_norm > 1e-12 else None,
            )
        )
    valid_labels = [label for label in np.unique(labels) if label >= 0]
    if not starts and not valid_labels and len(seed_indices) >= max(2, int(min_cluster_size)):
        seed_points = points[seed_indices]
        distances_to_primary, primary_matches = primary_tree.query(seed_points, k=1)
        best_member = int(np.argmin(distances_to_primary))
        primary_idx = int(primary_matches[best_member])
        tip_guard_nodes = int(np.ceil(float(exclude_parent_tip_fraction) * len(primary_path)))
        if tip_guard_nodes > 0 and primary_idx >= len(primary_path) - tip_guard_nodes:
            return starts
        direction = seed_points[best_member] - primary_path[primary_idx]
        direction_norm = max(float(np.linalg.norm(direction)), 1e-12)
        branch_angle = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        abs(np.dot(direction / direction_norm, primary_tangents[primary_idx])),
                        0.0,
                        1.0,
                    )
                )
            )
        )
        if branch_angle < float(minimum_branch_angle_degrees):
            return starts
        starts.append(
            LateralStart(
                0,
                seed_points[best_member],
                primary_path[primary_idx],
                primary_idx,
                seed_indices,
                direction,
            )
        )
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
    cooperate: Callable[[], None] | None = None,
    parent_radius_profile: np.ndarray | None = None,
) -> list[RootPath]:
    point_tree = cKDTree(points)
    primary_tangents = tangent_vectors(primary_path)
    non_primary = np.flatnonzero(~primary_mask)
    allowed_indices = set(non_primary.tolist())
    novel_support_mask = _novel_support_mask(
        points,
        occupied_mask=primary_mask,
        parent_path=primary_path,
        parent_radius_profile=parent_radius_profile,
        d_bar=d_bar,
    )
    candidates: list[RootPath] = []
    for start in starts:
        if cooperate is not None:
            cooperate()
        outward = (
            np.asarray(start.direction, dtype=float)
            if start.direction is not None
            else start.point - start.primary_point
        )
        if np.linalg.norm(outward) < 1e-9:
            tangent = primary_tangents[start.primary_index]
            outward = _perpendicular_vector(tangent)
        outward = outward / np.linalg.norm(outward)
        outward_directions: list[tuple[str, np.ndarray]] = [("pca", outward)]
        if start.radial_direction is not None:
            radial = np.asarray(start.radial_direction, dtype=float)
            radial /= max(float(np.linalg.norm(radial)), 1e-12)
            separation = float(
                np.degrees(np.arccos(np.clip(np.dot(outward, radial), -1.0, 1.0)))
            )
            if separation >= 20.0:
                outward_directions.append(("radial", radial))
        if start.extent_direction is not None:
            extent = np.asarray(start.extent_direction, dtype=float)
            extent /= max(float(np.linalg.norm(extent)), 1e-12)
            if all(
                float(
                    np.degrees(
                        np.arccos(np.clip(np.dot(existing, extent), -1.0, 1.0))
                    )
                )
                >= 20.0
                for _, existing in outward_directions
            ):
                outward_directions.append(("extent", extent))
        primary_tangent = primary_tangents[start.primary_index]
        candidate_count_before = len(candidates)
        trace_family_counter = 0

        def trace_variants(
            initial_direction: np.ndarray,
            direction_label: str,
            multipliers: tuple[float, ...],
            angles: tuple[float, ...],
        ) -> None:
            nonlocal trace_family_counter
            for multiplier in multipliers:
                step_length = max(multiplier * d_bar, 0.004)
                for open_angle in angles:
                    trace_family = trace_family_counter
                    trace_family_counter += 1
                    hypotheses = _grow_candidate_hypotheses(
                        points=points,
                        point_tree=point_tree,
                        allowed_indices=allowed_indices,
                        start=start,
                        initial_direction=initial_direction,
                        primary_tangent=primary_tangent,
                        step_length=step_length,
                        open_angle=open_angle,
                        max_steps=max_steps,
                        search_radius=search_radius_factor * step_length,
                        limit_primary_angle_to_insertion=True,
                        density_support_mask=novel_support_mask,
                        cooperate=cooperate,
                        beam_width=3,
                    )
                    for hypothesis_rank, path in enumerate(hypotheses):
                        if len(path.points) < 3:
                            continue
                        path.root_id = (
                            f"lateral_{start.start_id}_{direction_label}"
                            f"_s{multiplier:g}_a{int(open_angle)}"
                            f"_h{hypothesis_rank + 1}"
                        )
                        path.novel_support_indices = _path_support_indices(
                            point_tree,
                            path.points,
                            radius=max(2.0 * d_bar, 0.004),
                            support_mask=novel_support_mask,
                        )
                        path.score_components["novel_density_support"] = float(
                            len(path.novel_support_indices)
                        )
                        path.score_components["trace_hypothesis_rank"] = float(
                            hypothesis_rank
                        )
                        path.score_components["trace_parameter_family"] = float(
                            trace_family
                        )
                        # Novel support remains the dominant biological
                        # evidence.  Trace quality only breaks close variants,
                        # so radius continuity cannot suppress a genuinely
                        # supported taper or fork.
                        path.score = float(
                            len(path.novel_support_indices)
                            + 20.0 * path.length
                            + path.score_components.get("trace_rank_score", 0.0)
                        )
                        path.start_index = start.start_id
                        candidates.append(path)

        for direction_label, initial_direction in outward_directions:
            trace_variants(
                initial_direction,
                direction_label,
                step_multipliers,
                open_angles,
            )
        if len(candidates) == candidate_count_before:
            # A primary segmentation can legitimately absorb the first few
            # mesh units of a junction.  Retry only failed starts with longer
            # bridge steps rather than paying this cost for every candidate.
            for direction_label, initial_direction in outward_directions:
                trace_variants(
                    initial_direction,
                    direction_label,
                    (8.0, 10.0),
                    (55.0, 75.0),
                )
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
    limit_primary_angle_to_insertion: bool = False,
    max_turn_degrees: float = 70.0,
    minimum_local_support: int = 1,
    cooperate: Callable[[], None] | None = None,
    density_support_mask: np.ndarray | None = None,
) -> RootPath:
    """Return the strongest trace while retaining the historical one-path API."""

    hypotheses = _grow_candidate_hypotheses(
        points=points,
        point_tree=point_tree,
        allowed_indices=allowed_indices,
        start=start,
        initial_direction=initial_direction,
        primary_tangent=primary_tangent,
        step_length=step_length,
        open_angle=open_angle,
        max_steps=max_steps,
        search_radius=search_radius,
        limit_primary_angle_to_insertion=limit_primary_angle_to_insertion,
        max_turn_degrees=max_turn_degrees,
        minimum_local_support=minimum_local_support,
        cooperate=cooperate,
        density_support_mask=density_support_mask,
        beam_width=1,
    )
    if hypotheses:
        return hypotheses[0]
    return RootPath(
        root_id="candidate",
        points=np.asarray([start.primary_point, start.point], dtype=float),
        raw_start_point=np.asarray(start.point, dtype=float).copy(),
    )


def _grow_candidate_hypotheses(
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
    limit_primary_angle_to_insertion: bool = False,
    max_turn_degrees: float = 70.0,
    minimum_local_support: int = 1,
    cooperate: Callable[[], None] | None = None,
    density_support_mask: np.ndarray | None = None,
    beam_width: int = 3,
) -> list[RootPath]:
    """Trace a bounded beam of directionally distinct fork hypotheses.

    The biological lateral tracer applies the parent-relative angle gate only
    to the insertion step, then follows each hypothesis' evolving local
    tangent.  Conservative callers can retain a fixed reference cone by
    leaving ``limit_primary_angle_to_insertion`` false.

    Local direction remains important, but its proposal weight is reduced from
    the former 0.55 to 0.47 and the beam-level curvature penalty is gentler.
    A small, soft local-radius continuity reward helps distinguish a sustained
    root from an abrupt change in tubular scale without rejecting natural
    tapering.
    """

    initial = np.asarray(initial_direction, dtype=float).copy()
    initial /= max(float(np.linalg.norm(initial)), 1e-12)
    parent_tangent = np.asarray(primary_tangent, dtype=float).copy()
    parent_tangent /= max(float(np.linalg.norm(parent_tangent)), 1e-12)
    initial_radius = _estimate_local_radius(
        point_tree,
        points,
        np.asarray(start.point, dtype=float),
        initial,
        radius=0.75 * float(search_radius),
        support_mask=density_support_mask,
    )
    active = [
        _TraceState(
            nodes=[
                np.asarray(start.primary_point, dtype=float).copy(),
                np.asarray(start.point, dtype=float).copy(),
            ],
            current=np.asarray(start.point, dtype=float).copy(),
            direction=initial,
            covered=set(),
            local_radius=initial_radius,
        )
    ]
    completed: list[_TraceState] = []
    width = max(1, int(beam_width))

    for step_index in range(max_steps):
        if cooperate is not None and step_index % 8 == 0:
            cooperate()
        expanded: list[_TraceState] = []
        for state in active:
            local = point_tree.query_ball_point(
                state.current,
                r=search_radius,
                workers=worker_threads(),
            )
            local = [
                idx
                for idx in local
                if idx in allowed_indices and idx not in state.covered
            ]
            if len(local) < max(1, int(minimum_local_support)):
                completed.append(state)
                continue
            vectors = points[local] - state.current
            distances = np.linalg.norm(vectors, axis=1)
            valid = distances >= 0.35 * step_length
            if not np.any(valid):
                completed.append(state)
                continue
            local_array = np.asarray(local, dtype=int)[valid]
            vectors = vectors[valid]
            distances = distances[valid]
            unit = vectors / np.maximum(distances[:, None], 1e-12)
            turn_cos = unit @ state.direction
            turn_ok = turn_cos >= np.cos(np.radians(float(max_turn_degrees)))
            if step_index == 0 or not limit_primary_angle_to_insertion:
                open_angle_to_primary = np.degrees(
                    np.arccos(
                        np.clip(
                            np.abs(unit @ parent_tangent),
                            -1.0,
                            1.0,
                        )
                    )
                )
                open_ok = open_angle_to_primary <= open_angle
                ok = (
                    turn_ok & open_ok
                    if np.any(turn_ok & open_ok)
                    else turn_ok
                )
            else:
                ok = turn_ok
            if not np.any(ok):
                completed.append(state)
                continue
            local_array = local_array[ok]
            unit = unit[ok]
            distances = distances[ok]
            turn_cos = turn_cos[ok]
            density = _local_support_counts(
                point_tree,
                points[local_array],
                radius=0.75 * search_radius,
                support_mask=density_support_mask,
            )
            distance_score = -np.abs(distances - step_length) / max(
                step_length,
                1e-12,
            )
            base_score = (
                0.47 * turn_cos
                + 0.30 * _normalize(density)
                + 0.15 * distance_score
            )
            # Radius estimation needs a local cross-section and is therefore
            # more expensive than the batched density query.  Evaluate it only
            # for the strongest direction-diverse shortlist, not every surface
            # vertex in the search ball.
            radius_shortlist = _distinct_direction_proposals(
                unit,
                base_score,
                limit=max(2 * width, 4),
                minimum_separation_degrees=12.0,
            )
            shortlist_array = np.asarray(radius_shortlist, dtype=int)
            local_radii = _local_radius_estimates(
                point_tree,
                points,
                points[local_array[shortlist_array]],
                unit[shortlist_array],
                radius=0.75 * search_radius,
                support_mask=density_support_mask,
            )
            radius_similarity = _radius_continuity_scores(
                local_radii,
                state.local_radius,
            )
            shortlist_score = (
                base_score[shortlist_array]
                + 0.08 * radius_similarity
            )
            selected_shortlist_positions = _distinct_direction_proposals(
                unit[shortlist_array],
                shortlist_score,
                limit=max(width, 2),
            )
            proposal_positions = [
                int(shortlist_array[position])
                for position in selected_shortlist_positions
            ]
            for position in proposal_positions:
                next_point = np.asarray(
                    points[int(local_array[position])],
                    dtype=float,
                )
                segment = next_point - state.current
                segment_length = float(np.linalg.norm(segment))
                new_direction = segment / max(segment_length, 1e-12)
                turn_angle = float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                np.dot(state.direction, new_direction),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                evolved_direction = 0.65 * state.direction + 0.35 * new_direction
                evolved_direction /= max(
                    float(np.linalg.norm(evolved_direction)),
                    1e-12,
                )
                nearby = point_tree.query_ball_point(
                    next_point,
                    # Suppress local revisits without consuming the whole
                    # forward search shell needed by a dense curved tube.
                    r=min(
                        0.85 * float(step_length),
                        0.65 * float(search_radius),
                    ),
                    workers=worker_threads(),
                )
                covered = set(state.covered)
                covered.update(idx for idx in nearby if idx in allowed_indices)
                support_value = float(density[position])
                shortlist_position = radius_shortlist.index(position)
                proposed_radius = float(local_radii[shortlist_position])
                if np.isfinite(proposed_radius) and proposed_radius > 0.0:
                    if np.isfinite(state.local_radius) and state.local_radius > 0.0:
                        evolved_radius = (
                            0.65 * float(state.local_radius)
                            + 0.35 * proposed_radius
                        )
                    else:
                        evolved_radius = proposed_radius
                    radius_observation = 1
                else:
                    evolved_radius = float(state.local_radius)
                    radius_observation = 0
                expanded.append(
                    _TraceState(
                        nodes=state.nodes + [next_point.copy()],
                        current=next_point.copy(),
                        direction=evolved_direction,
                        covered=covered,
                        growth_arc=state.growth_arc + segment_length,
                        supported_arc=state.supported_arc
                        + (
                            segment_length
                            if support_value >= max(1, int(minimum_local_support))
                            else 0.0
                        ),
                        support_sum=state.support_sum + support_value,
                        turn_squared_sum=state.turn_squared_sum + turn_angle**2,
                        cumulative_turn_degrees=(
                            state.cumulative_turn_degrees + turn_angle
                        ),
                        cumulative_score=(
                            state.cumulative_score
                            + float(shortlist_score[shortlist_position])
                        ),
                        local_radius=evolved_radius,
                        radius_similarity_sum=(
                            state.radius_similarity_sum
                            + (
                                float(radius_similarity[shortlist_position])
                                if radius_observation
                                else 0.0
                            )
                        ),
                        radius_observations=(
                            state.radius_observations + radius_observation
                        ),
                        steps=state.steps + 1,
                    )
                )
        if not expanded:
            active = []
            break
        active = _prune_trace_beam(
            expanded,
            beam_width=width,
            step_length=step_length,
            max_turn_degrees=max_turn_degrees,
        )

    completed.extend(active)
    if not completed:
        return []
    terminal = _prune_trace_beam(
        completed,
        beam_width=width,
        step_length=step_length,
        max_turn_degrees=max_turn_degrees,
    )
    paths: list[RootPath] = []
    for state in terminal:
        turn_rms = float(
            np.sqrt(state.turn_squared_sum / max(1, state.steps))
        )
        mean_support = float(state.support_sum / max(1, state.steps))
        supported_fraction = float(
            state.supported_arc / max(state.growth_arc, 1e-12)
        )
        radius_similarity = _mean_radius_similarity(state)
        path = RootPath(
            root_id="candidate",
            points=np.asarray(state.nodes, dtype=float),
            raw_start_point=np.asarray(start.point, dtype=float).copy(),
            covered_indices=set(state.covered),
        )
        path.score_components.update(
            {
                "trace_supported_arc": float(state.supported_arc),
                "trace_supported_fraction": supported_fraction,
                "trace_mean_support": mean_support,
                "trace_turn_rms_degrees": turn_rms,
                "trace_cumulative_turn_degrees": float(
                    state.cumulative_turn_degrees
                ),
                "trace_smoothness": float(
                    np.exp(
                        -(
                            turn_rms
                            / max(float(max_turn_degrees), 1.0)
                        )
                        ** 2
                    )
                ),
                "trace_local_radius": (
                    float(state.local_radius)
                    if np.isfinite(state.local_radius)
                    else 0.0
                ),
                "trace_radius_similarity": radius_similarity,
                "trace_radius_observations": float(state.radius_observations),
                "trace_cumulative_score": float(state.cumulative_score),
                "trace_rank_score": _trace_state_rank(
                    state,
                    step_length=step_length,
                    max_turn_degrees=max_turn_degrees,
                ),
            }
        )
        paths.append(path)
    return paths


def _distinct_direction_proposals(
    directions: np.ndarray,
    scores: np.ndarray,
    *,
    limit: int,
    minimum_separation_degrees: float = 18.0,
) -> list[int]:
    """Return the strongest proposal from each local direction mode."""

    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    selected: list[int] = []
    cosine_limit = float(np.cos(np.radians(minimum_separation_degrees)))
    for raw_index in order:
        index = int(raw_index)
        if selected and any(
            float(np.dot(directions[index], directions[other])) > cosine_limit
            for other in selected
        ):
            continue
        selected.append(index)
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def _mean_radius_similarity(state: _TraceState) -> float:
    if state.radius_observations <= 0:
        return 0.5
    return float(
        state.radius_similarity_sum / max(1, state.radius_observations)
    )


def _trace_state_rank(
    state: _TraceState,
    *,
    step_length: float,
    max_turn_degrees: float,
) -> float:
    """Score one beam state with soft curvature and radius continuity terms."""

    supported_steps = state.supported_arc / max(float(step_length), 1e-12)
    turn_scale = max(float(max_turn_degrees), 1.0)
    turn_rms = float(np.sqrt(state.turn_squared_sum / max(1, state.steps)))
    curvature_penalty = (turn_rms / turn_scale) ** 2
    cumulative_turn_penalty = (
        state.cumulative_turn_degrees
        / max(turn_scale * max(1, state.steps), 1e-12)
    )
    absolute_support_reward = 0.20 * float(np.log1p(state.support_sum))
    return float(
        state.cumulative_score
        + 0.25 * supported_steps
        + absolute_support_reward
        + 0.12 * _mean_radius_similarity(state)
        # The recovered beam used 0.35 on RMS curvature alone.  The combined
        # 0.25 + 0.06 penalties are intentionally gentler so a genuine curved
        # or mildly zigzagging root can survive until endpoint consensus.
        - 0.25 * curvature_penalty
        - 0.06 * cumulative_turn_penalty
    )


def _prune_trace_beam(
    states: list[_TraceState],
    *,
    beam_width: int,
    step_length: float,
    max_turn_degrees: float,
) -> list[_TraceState]:
    """Keep the strongest endpoint/direction-distinct trace states."""

    ranked = sorted(
        states,
        key=lambda state: (
            _trace_state_rank(
                state,
                step_length=step_length,
                max_turn_degrees=max_turn_degrees,
            ),
            state.growth_arc,
            _mean_radius_similarity(state),
            -float(np.sqrt(state.turn_squared_sum / max(1, state.steps))),
        ),
        reverse=True,
    )
    retained: list[_TraceState] = []
    direction_cosine = float(np.cos(np.radians(15.0)))
    for state in ranked:
        duplicate = False
        for other in retained:
            if float(np.dot(state.direction, other.direction)) < direction_cosine:
                continue
            endpoint_close = (
                float(np.linalg.norm(state.current - other.current))
                <= 0.60 * float(step_length)
            )
            smaller_coverage = min(len(state.covered), len(other.covered))
            coverage_overlap = (
                len(state.covered & other.covered) / smaller_coverage
                if smaller_coverage
                else 0.0
            )
            # Fork states necessarily share their entire trunk coverage.  Do
            # not collapse them merely for that historical overlap after their
            # endpoints have separated; both proximity and overlap are needed
            # to identify a duplicate surface trace.
            if endpoint_close and coverage_overlap >= 0.70:
                duplicate = True
                break
        if duplicate:
            continue
        retained.append(state)
        if len(retained) >= max(1, int(beam_width)):
            break
    return retained


def extend_lateral_tip(
    points: np.ndarray,
    path: RootPath,
    blocked_mask: np.ndarray,
    d_bar: float,
    *,
    max_steps: int = 80,
    min_support: int = 4,
    point_tree: cKDTree | None = None,
    cooperate: Callable[[], None] | None = None,
) -> RootPath:
    """Continue a selected path when dense unclaimed support exists ahead.

    A residual-support probe prevents this pass from merely walking around the
    surface cap of a completed root.  Once continuation is justified, a
    conservative forward cone follows only currently unclaimed support and can
    bridge the assignment halo without changing path identity or hierarchy.
    """

    points = np.asarray(points, dtype=float)
    blocked = np.asarray(blocked_mask, dtype=bool)
    if len(path.points) < 3 or points.ndim != 2 or points.shape[1] != 3:
        return path
    if blocked.shape != (len(points),):
        raise ValueError("blocked_mask must have one value per point")
    available = ~blocked
    if not np.any(available):
        return path

    tree = point_tree if point_tree is not None else cKDTree(points)
    target_step = max(6.0 * float(d_bar), 0.004)
    assignment_radius = max(4.0 * float(d_bar), 0.006)
    search_radius = max(14.0 * float(d_bar), 0.009)
    support_radius = max(4.0 * float(d_bar), 0.003)
    direction = _tip_direction(path.points, window=max(20.0 * float(d_bar), 0.012))
    current = np.asarray(path.points[-1], dtype=float).copy()
    original_points = np.asarray(path.points, dtype=float).copy()
    original_covered = set(path.covered_indices)
    original_node_indices = path.node_indices
    original_tree = cKDTree(original_points)

    initial = _forward_supported_indices(
        points,
        tree,
        available,
        current,
        direction,
        query_radius=search_radius,
        target_step=target_step,
        support_radius=support_radius,
        min_support=min_support,
    )
    if len(initial):
        residual_distances, _ = original_tree.query(points[initial], k=1, workers=worker_threads())
        initial = initial[residual_distances > 0.90 * assignment_radius]
    path.score_components["tip_continuation_initial_support"] = float(len(initial))
    if len(initial) < int(min_support):
        path.score_components["tip_continuation_accepted"] = 0.0
        path.score_components["tip_extension_steps"] = 0.0
        path.score_components["tip_extension_length"] = 0.0
        return path

    continuation_start = LateralStart(
        start_id=-1,
        point=current,
        primary_point=np.asarray(path.points[-2], dtype=float),
        primary_index=0,
        member_indices=initial,
        direction=direction,
    )
    candidate = _grow_one_candidate(
        points=points,
        point_tree=tree,
        allowed_indices=set(np.flatnonzero(available).tolist()),
        start=continuation_start,
        initial_direction=direction,
        primary_tangent=direction,
        step_length=target_step,
        open_angle=75.0,
        max_steps=max(1, int(max_steps)),
        search_radius=search_radius,
        max_turn_degrees=45.0,
        minimum_local_support=min_support,
        cooperate=cooperate,
    )
    appended = np.asarray(candidate.points[2:], dtype=float)
    candidate_extension_length = 0.0
    if len(appended):
        extension_polyline = np.vstack([original_points[-1], appended])
        candidate_extension_length = float(np.linalg.norm(np.diff(extension_polyline, axis=0), axis=1).sum())
    extension_support = set(candidate.covered_indices)
    new_support_count = len(extension_support)
    accepted = bool(
        len(appended) >= 3
        and candidate_extension_length >= max(8.0 * float(d_bar), 0.008)
        and new_support_count >= max(20, 4 * len(appended))
    )
    if accepted:
        path.points = np.vstack([original_points, appended])
        path.covered_indices = original_covered | extension_support
        path.node_indices = None
        extension_steps = len(appended)
        extension_length = candidate_extension_length
    else:
        path.points = original_points
        path.covered_indices = original_covered
        path.node_indices = original_node_indices
        extension_steps = 0
        extension_length = 0.0
    path.score_components["tip_continuation_candidate_steps"] = float(len(appended))
    path.score_components["tip_continuation_candidate_length"] = candidate_extension_length
    path.score_components["tip_continuation_new_support"] = float(new_support_count)
    path.score_components["tip_continuation_accepted"] = float(accepted)
    path.score_components["tip_extension_steps"] = float(extension_steps)
    path.score_components["tip_extension_length"] = extension_length
    hit_limit = bool(accepted and len(appended) >= max(1, int(max_steps)))
    path.score_components["tip_extension_hit_limit"] = float(hit_limit)
    if hit_limit and "tip_extension_limit" not in path.qc_flags:
        path.qc_flags.append("tip_extension_limit")
    return path


def _tip_direction(path: np.ndarray, *, window: float) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    reverse_distance = np.concatenate([[0.0], np.cumsum(segment_lengths[::-1])])
    back = int(np.searchsorted(reverse_distance, float(window), side="left"))
    back = min(max(1, back), len(path) - 1)
    direction = path[-1] - path[-1 - back]
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    return direction


def _forward_supported_indices(
    points: np.ndarray,
    tree: cKDTree,
    available: np.ndarray,
    current: np.ndarray,
    direction: np.ndarray,
    *,
    query_radius: float,
    target_step: float,
    support_radius: float,
    min_support: int,
    excluded_indices: set[int] | None = None,
) -> np.ndarray:
    local = np.asarray(
        tree.query_ball_point(current, r=float(query_radius), workers=worker_threads()),
        dtype=int,
    )
    if not len(local):
        return np.empty(0, dtype=int)
    local = local[available[local]]
    if excluded_indices:
        local = np.asarray([index for index in local if int(index) not in excluded_indices], dtype=int)
    if not len(local):
        return np.empty(0, dtype=int)
    vectors = points[local] - current
    distances = np.linalg.norm(vectors, axis=1)
    valid_distance = distances >= 0.35 * float(target_step)
    if not np.any(valid_distance):
        return np.empty(0, dtype=int)
    local = local[valid_distance]
    vectors = vectors[valid_distance]
    distances = distances[valid_distance]
    unit = vectors / np.maximum(distances[:, None], 1e-12)
    forward = (unit @ direction) >= np.cos(np.radians(45.0))
    if not np.any(forward):
        return np.empty(0, dtype=int)
    local = local[forward]
    support = np.asarray(
        tree.query_ball_point(
            points[local],
            r=float(support_radius),
            workers=worker_threads(),
            return_length=True,
        ),
        dtype=int,
    )
    return local[support >= max(1, int(min_support))]


def reduce_similar_paths(candidates: list[RootPath], n_clusters: int | None = None) -> list[RootPath]:
    """Collapse tracing variants while retaining supported branch modes.

    ``grow_lateral_candidates`` deliberately explores several step lengths and
    opening angles from every detected junction.  Most paths are duplicate
    parameter variants, but a dense collar cluster can contain two biological
    roots with the same parent and start identity.  Endpoint and direction
    consensus therefore retain every mode supported by at least two variants;
    singleton modes are treated as parameter outliers.  The later overlap-aware
    selector still removes duplicates produced by neighbouring seed clusters.

    ``n_clusters`` is retained as an optional upper bound for API compatibility.
    """

    if len(candidates) <= 1:
        return candidates
    groups: dict[tuple[str, int | str], list[RootPath]] = {}
    for index, candidate in enumerate(candidates):
        start_key: int | str = candidate.start_index if candidate.start_index is not None else f"path-{index}"
        groups.setdefault((str(candidate.parent_id), start_key), []).append(candidate)
    reduced = [
        representative
        for members in groups.values()
        for representative in _endpoint_consensus_variants(members)
    ]
    reduced.sort(key=lambda path: (path.score, path.length, len(path.covered_indices)), reverse=True)
    if n_clusters is not None:
        reduced = reduced[: max(1, min(int(n_clusters), len(reduced)))]
    return reduced


def _endpoint_consensus_variants(members: list[RootPath]) -> list[RootPath]:
    """Return one medoid for each repeatable endpoint/direction mode.

    Two agreeing parameter variants are the minimum evidence for a second
    biological branch.  If no mode has that support, the historical single
    consensus result is retained so sparse starts are not discarded outright.
    """

    if len(members) <= 1:
        return list(members)
    mode_clusters = _endpoint_mode_clusters(members)
    family_keys = [
        _trace_parameter_family_key(path, index)
        for index, path in enumerate(members)
    ]

    def family_support(cluster: list[int]) -> int:
        return len({family_keys[index] for index in cluster})

    supported = [
        cluster
        for cluster in mode_clusters
        if family_support(cluster) >= 2
    ]
    if not supported:
        supported = [list(range(len(members)))]
        rejected_count = 0
    else:
        rejected_count = len(members) - sum(len(cluster) for cluster in supported)

    representatives: list[RootPath] = []
    representative_members: list[list[RootPath]] = []
    for cluster in supported:
        cluster_members = [members[index] for index in cluster]
        representative = _endpoint_consensus_variant(cluster_members)
        representative.score_components["variant_endpoint_mode_support"] = float(
            family_support(cluster)
        )
        representative.score_components["variant_endpoint_hypothesis_support"] = float(
            len(cluster)
        )
        representative.score_components["variant_endpoint_outliers_rejected"] = float(rejected_count)
        representatives.append(representative)
        representative_members.append(cluster_members)
    representatives, representative_members = _defer_distal_fork_modes(
        representatives,
        representative_members,
    )
    representatives, representative_members = _drop_truncated_prefix_modes(
        representatives,
        representative_members,
    )
    representatives.sort(
        key=lambda path: (path.score, path.length, len(path.covered_indices), str(path.root_id)),
        reverse=True,
    )
    for mode_index, representative in enumerate(representatives):
        representative.score_components["variant_endpoint_mode_count"] = float(len(representatives))
        representative.score_components["variant_endpoint_mode_index"] = float(mode_index)
    return representatives


def _trace_parameter_family_key(
    path: RootPath,
    fallback_index: int,
) -> tuple[str, int]:
    value = path.score_components.get("trace_parameter_family")
    if value is not None and np.isfinite(value):
        return ("family", int(round(float(value))))
    return ("path", int(fallback_index))


def _drop_truncated_prefix_modes(
    representatives: list[RootPath],
    member_groups: list[list[RootPath]],
) -> tuple[list[RootPath], list[list[RootPath]]]:
    """Remove short terminal modes that never depart from a longer mode."""

    keep = np.ones(len(representatives), dtype=bool)
    for short_index, short in enumerate(representatives):
        if (
            short.score_components.get("distal_fork_deferred_child", 0.0)
            > 0.0
        ):
            continue
        for long_index, long in enumerate(representatives):
            if short_index == long_index:
                continue
            if float(short.length) > 0.30 * float(long.length):
                continue
            spacings = np.linalg.norm(np.diff(short.points, axis=0), axis=1)
            spacings = spacings[spacings > 1e-12]
            if not len(spacings):
                continue
            spacing = max(float(np.median(spacings)), float(short.length) / 80.0)
            tolerance = max(2.5 * spacing, 0.04 * float(short.length))
            near_arc = _prefix_arc_near_path(
                short.points,
                long.points,
                spacing=spacing,
                tolerance=tolerance,
            )
            endpoint_distance = float(
                cKDTree(long.points).query(short.points[-1], k=1)[0]
            )
            if (
                near_arc >= 0.80 * float(short.length)
                and endpoint_distance <= tolerance
            ):
                keep[short_index] = False
                break
    return (
        [
            representative
            for index, representative in enumerate(representatives)
            if bool(keep[index])
        ],
        [
            group
            for index, group in enumerate(member_groups)
            if bool(keep[index])
        ],
    )


def _defer_distal_fork_modes(
    representatives: list[RootPath],
    member_groups: list[list[RootPath]],
) -> tuple[list[RootPath], list[list[RootPath]]]:
    """Keep one current-order path when hypotheses share a sustained prefix.

    A distal fork is a child insertion, not a second root attached at the
    original seed.  Its alternate support is deliberately left unclaimed so it
    can be seeded against the retained continuation on the next order.  Modes
    that diverge near insertion remain siblings.
    """

    representatives = list(representatives)
    groups = [list(group) for group in member_groups]
    while True:
        fork_pair: tuple[int, int] | None = None
        for left in range(len(representatives)):
            if (
                representatives[left].score_components.get(
                    "distal_fork_deferred_child",
                    0.0,
                )
                > 0.0
            ):
                continue
            for right in range(left + 1, len(representatives)):
                if (
                    representatives[right].score_components.get(
                        "distal_fork_deferred_child",
                        0.0,
                    )
                    > 0.0
                ):
                    continue
                if _modes_share_distal_fork_prefix(
                    representatives[left],
                    representatives[right],
                ):
                    fork_pair = (left, right)
                    break
            if fork_pair is not None:
                break
        if fork_pair is None:
            break

        left, right = fork_pair
        preferred = max(
            (representatives[left], representatives[right]),
            key=lambda path: (
                float(
                    path.score_components.get(
                        "variant_endpoint_mode_support",
                        1.0,
                    )
                ),
                float(path.score_components.get("trace_supported_arc", 0.0)),
                float(path.score),
                float(path.length),
                str(path.root_id),
            ),
        )
        deferred = (
            representatives[right]
            if preferred is representatives[left]
            else representatives[left]
        )
        prefix_arcs = _distal_fork_prefix_arcs(preferred, deferred)
        if prefix_arcs is None:
            break
        deferred_prefix_arc = float(prefix_arcs[1])
        preferred_prefix_arc = float(prefix_arcs[0])
        parent_junction = _point_at_polyline_arc(
            preferred.points,
            preferred_prefix_arc,
        )
        projected_deferred_arc = _nearest_polyline_arc(
            deferred.points,
            parent_junction,
        )
        deferred_suffix = _polyline_suffix_from_arc(
            deferred.points,
            max(deferred_prefix_arc, projected_deferred_arc),
        )
        if len(deferred_suffix) < 2:
            break
        if np.linalg.norm(deferred_suffix[0] - parent_junction) > 1e-12:
            deferred_suffix = np.vstack([parent_junction, deferred_suffix])
        else:
            deferred_suffix[0] = parent_junction

        preferred_support = (
            preferred.novel_support_indices
            if preferred.novel_support_indices is not None
            else preferred.covered_indices
        )
        deferred.covered_indices = set(deferred.covered_indices) - set(
            preferred.covered_indices
        )
        if deferred.novel_support_indices is not None:
            deferred.novel_support_indices = set(
                deferred.novel_support_indices
            ) - set(preferred_support)
        deferred.points = deferred_suffix
        deferred.raw_start_point = deferred_suffix[0].copy()
        deferred.parent_id = preferred.root_id
        deferred.parent_points = preferred.points
        deferred.order = int(preferred.order) + 1
        deferred.score_components["distal_fork_deferred_child"] = 1.0
        deferred.score_components["distal_fork_shared_prefix_arc"] = float(
            deferred_prefix_arc
        )
        deferred.score_components["distal_fork_parent_prefix_arc"] = float(
            preferred_prefix_arc
        )
        deferred.score = float(
            len(
                deferred.novel_support_indices
                if deferred.novel_support_indices is not None
                else deferred.covered_indices
            )
            + 20.0 * deferred.length
            + deferred.score_components.get("trace_rank_score", 0.0)
        )
        preferred.score_components["distal_fork_modes_deferred"] = (
            preferred.score_components.get("distal_fork_modes_deferred", 0.0)
            + deferred.score_components.get("distal_fork_modes_deferred", 0.0)
            + 1.0
        )
        preferred.score_components["distal_fork_deferred_variant_support"] = (
            preferred.score_components.get(
                "distal_fork_deferred_variant_support",
                0.0,
            )
            + float(
                deferred.score_components.get(
                    "variant_endpoint_mode_support",
                    1.0,
                )
            )
        )
        # Keep the trimmed alternate explicitly.  The pipeline resolves its
        # temporary parent candidate after overlap selection and carries it as
        # the next-order child instead of hoping seed detection rediscovers it.
        representatives[left] = preferred
        representatives[right] = deferred
    return representatives, groups


def _modes_share_distal_fork_prefix(
    left: RootPath,
    right: RootPath,
) -> bool:
    prefix_arcs = _distal_fork_prefix_arcs(left, right)
    if prefix_arcs is None:
        return False
    left_prefix, right_prefix, minimum_prefix = prefix_arcs
    return bool(min(left_prefix, right_prefix) >= minimum_prefix)


def _distal_fork_prefix_arcs(
    left: RootPath,
    right: RootPath,
) -> tuple[float, float, float] | None:
    left_points = np.asarray(left.points, dtype=float)
    right_points = np.asarray(right.points, dtype=float)
    if len(left_points) < 3 or len(right_points) < 3:
        return None
    shorter_length = min(float(left.length), float(right.length))
    if shorter_length <= 1e-12:
        return None
    spacings = [
        float(length)
        for path in (left_points, right_points)
        for length in np.linalg.norm(np.diff(path, axis=0), axis=1)
        if length > 1e-12
    ]
    if not spacings:
        return None
    spacing = max(
        min(float(np.median(spacings)), shorter_length / 40.0),
        shorter_length / 120.0,
        1e-9,
    )
    tolerance = max(2.5 * spacing, 0.03 * shorter_length)
    minimum_prefix = max(6.0 * spacing, 0.20 * shorter_length)
    left_prefix = _prefix_arc_near_path(
        left_points,
        right_points,
        spacing=spacing,
        tolerance=tolerance,
    )
    right_prefix = _prefix_arc_near_path(
        right_points,
        left_points,
        spacing=spacing,
        tolerance=tolerance,
    )
    return float(left_prefix), float(right_prefix), float(minimum_prefix)


def _prefix_arc_near_path(
    path: np.ndarray,
    reference: np.ndarray,
    *,
    spacing: float,
    tolerance: float,
) -> float:
    sampled = resample_polyline(path, spacing=float(spacing))
    if len(sampled) < 2:
        return 0.0
    distances, _ = cKDTree(reference).query(
        sampled,
        k=1,
        workers=worker_threads(),
    )
    segment_lengths = np.linalg.norm(np.diff(sampled, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    outside = np.asarray(distances) > float(tolerance)
    # Surface traces can hop between opposite sides of the same tube for a few
    # samples.  Require both a four-sample outside run and predominantly
    # outside support over the remaining tail.  A temporary surface-side
    # excursion may rejoin; a biological fork stays separated.
    for first_outside in range(max(0, len(outside) - 3)):
        if not bool(np.all(outside[first_outside : first_outside + 4])):
            continue
        if float(np.mean(outside[first_outside:])) < 0.80:
            continue
        return float(arc[max(0, first_outside - 1)])
    return float(arc[-1])


def _polyline_suffix_from_arc(
    path: np.ndarray,
    starting_arc: float,
) -> np.ndarray:
    """Return a polyline beginning exactly at a requested path arc."""

    polyline = np.asarray(path, dtype=float)
    if len(polyline) <= 1:
        return polyline.copy()
    segment_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    target = float(np.clip(starting_arc, 0.0, cumulative[-1]))
    if target <= 1e-12:
        return polyline.copy()
    if target >= cumulative[-1] - 1e-12:
        return polyline[-1:].copy()
    segment_index = int(np.searchsorted(cumulative, target, side="right") - 1)
    segment_index = min(max(segment_index, 0), len(segment_lengths) - 1)
    segment_length = float(segment_lengths[segment_index])
    if segment_length <= 1e-12:
        return polyline[segment_index + 1 :].copy()
    fraction = (target - cumulative[segment_index]) / segment_length
    first = (
        polyline[segment_index]
        + float(np.clip(fraction, 0.0, 1.0))
        * (polyline[segment_index + 1] - polyline[segment_index])
    )
    remainder = polyline[segment_index + 1 :]
    if len(remainder) and np.linalg.norm(remainder[0] - first) <= 1e-12:
        return remainder.copy()
    return np.vstack([first, remainder])


def _point_at_polyline_arc(
    path: np.ndarray,
    target_arc: float,
) -> np.ndarray:
    polyline = np.asarray(path, dtype=float)
    if len(polyline) == 0:
        return np.zeros(3, dtype=float)
    if len(polyline) == 1:
        return polyline[0].copy()
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    target = float(np.clip(target_arc, 0.0, cumulative[-1]))
    segment = min(
        max(int(np.searchsorted(cumulative, target, side="right") - 1), 0),
        len(lengths) - 1,
    )
    if lengths[segment] <= 1e-12:
        return polyline[segment].copy()
    fraction = (target - cumulative[segment]) / lengths[segment]
    return (
        polyline[segment]
        + float(np.clip(fraction, 0.0, 1.0))
        * (polyline[segment + 1] - polyline[segment])
    )


def _nearest_polyline_arc(
    path: np.ndarray,
    point: np.ndarray,
) -> float:
    polyline = np.asarray(path, dtype=float)
    query = np.asarray(point, dtype=float)
    if len(polyline) < 2:
        return 0.0
    vectors = np.diff(polyline, axis=0)
    squared = np.sum(vectors * vectors, axis=1)
    parameters = np.zeros(len(vectors), dtype=float)
    valid = squared > 1e-20
    parameters[valid] = np.clip(
        np.sum((query - polyline[:-1][valid]) * vectors[valid], axis=1)
        / squared[valid],
        0.0,
        1.0,
    )
    projections = polyline[:-1] + parameters[:, None] * vectors
    segment = int(np.argmin(np.linalg.norm(projections - query, axis=1)))
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(vectors, axis=1))]
    )
    return float(
        cumulative[segment]
        + parameters[segment] * np.sqrt(max(squared[segment], 0.0))
    )


def _endpoint_mode_clusters(members: list[RootPath]) -> list[list[int]]:
    """Complete-link clusters of compatible endpoint and direction evidence."""

    endpoints = np.asarray([path.points[-1] for path in members], dtype=float)
    lengths = np.asarray([max(path.length, 0.0) for path in members], dtype=float)
    positive_lengths = lengths[lengths > 1e-12]
    typical_length = float(np.median(positive_lengths)) if len(positive_lengths) else 1.0
    segment_lengths = [
        float(length)
        for path in members
        for length in np.linalg.norm(np.diff(path.points, axis=0), axis=1)
        if length > 1e-12
    ]
    typical_spacing = float(np.median(segment_lengths)) if segment_lengths else 0.0
    endpoint_tolerance = max(6.0 * typical_spacing, 0.08 * typical_length, 1e-9)
    directional_reach = max(3.0 * endpoint_tolerance, 0.60 * typical_length)

    net_directions = np.asarray([_candidate_net_direction(path) for path in members])
    terminal_directions = np.asarray([_candidate_terminal_direction(path) for path in members])
    endpoint_gap = np.linalg.norm(endpoints[:, None, :] - endpoints[None, :, :], axis=2)
    net_cosine = np.clip(net_directions @ net_directions.T, -1.0, 1.0)
    terminal_cosine = np.clip(terminal_directions @ terminal_directions.T, -1.0, 1.0)
    net_angle = np.degrees(np.arccos(net_cosine))
    terminal_angle = np.degrees(np.arccos(terminal_cosine))
    compatible = (endpoint_gap <= endpoint_tolerance) | (
        (endpoint_gap <= directional_reach)
        & (net_angle <= 22.5)
        & (terminal_angle <= 40.0)
    )
    np.fill_diagonal(compatible, True)

    # Complete-link merging prevents a chimeric intermediate variant from
    # chaining two otherwise distinct endpoint modes into one cluster.
    clusters: list[list[int]] = [[index] for index in range(len(members))]
    while True:
        merge: tuple[float, int, int] | None = None
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                cross = np.ix_(clusters[left], clusters[right])
                if not bool(np.all(compatible[cross])):
                    continue
                distance = float(np.mean(endpoint_gap[cross]))
                candidate = (distance, left, right)
                if merge is None or candidate < merge:
                    merge = candidate
        if merge is None:
            break
        _, left, right = merge
        clusters[left] = sorted(clusters[left] + clusters[right])
        del clusters[right]
    return clusters


def _candidate_net_direction(path: RootPath) -> np.ndarray:
    origin = (
        np.asarray(path.raw_start_point, dtype=float)
        if path.raw_start_point is not None and np.asarray(path.raw_start_point).shape == (3,)
        else np.asarray(path.points[0], dtype=float)
    )
    return _unit_or_zero(np.asarray(path.points[-1], dtype=float) - origin)


def _candidate_terminal_direction(path: RootPath) -> np.ndarray:
    points = np.asarray(path.points, dtype=float)
    if len(points) < 2:
        return np.zeros(3, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(segment_lengths.sum())
    reverse_arc = np.concatenate([[0.0], np.cumsum(segment_lengths[::-1])])
    window = max(0.20 * total_length, 4.0 * float(np.median(segment_lengths)))
    back = int(np.searchsorted(reverse_arc, window, side="left"))
    back = min(max(1, back), len(points) - 1)
    return _unit_or_zero(points[-1] - points[-1 - back])


def _unit_or_zero(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros(3, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _endpoint_consensus_variant(members: list[RootPath]) -> RootPath:
    """Choose the parameter variant whose tip agrees with the other variants.

    A high opening-angle variant can turn into a child at a junction and then
    receive the largest density score because it covers *both* biological
    roots.  Treating that outlier as the parent trace consumes the child and
    causes the leftover fragments to be rediscovered as higher orders.  The
    nine tracing variants from one start instead provide a small consensus
    ensemble: the medoid of their endpoints follows the continuation selected
    by most parameter settings.  Density/length only break genuine medoid
    ties, preserving the prior preference when all variants reach one tip.
    """

    if len(members) <= 1:
        return members[0]
    endpoints = np.asarray([path.points[-1] for path in members], dtype=float)
    pairwise = np.linalg.norm(endpoints[:, None, :] - endpoints[None, :, :], axis=2)
    median_distance = np.median(pairwise, axis=1)
    best_index = min(
        range(len(members)),
        key=lambda index: (
            float(median_distance[index]),
            -float(members[index].score),
            -float(members[index].length),
            -len(
                members[index].novel_support_indices
                if members[index].novel_support_indices is not None
                else members[index].covered_indices
            ),
            str(members[index].root_id),
        ),
    )
    selected = members[best_index]
    selected.score_components["variant_endpoint_consensus_median"] = float(
        median_distance[best_index]
    )
    return selected


def select_non_overlapping_paths(
    candidates: list[RootPath],
    points: np.ndarray,
    d_bar: float,
    max_paths: int | None = None,
    overlap_penalty: float = 1.25,
    *,
    rename_selected: bool = True,
    initial_used: set[int] | None = None,
) -> list[RootPath]:
    if not candidates:
        return []
    tree = cKDTree(points)
    radius = max(2.5 * d_bar, 0.004)
    for candidate in candidates:
        if not candidate.covered_indices:
            covered = set()
            for node in candidate.points:
                covered.update(tree.query_ball_point(node, r=radius, workers=worker_threads()))
            candidate.covered_indices = covered
    selected: list[RootPath] = []
    used: set[int] = set(initial_used or ())
    pool = sorted(candidates, key=lambda p: (p.score, p.length), reverse=True)
    while pool:
        best_path = None
        best_value = 0.0
        for path in pool:
            if any(
                _is_truncated_path_duplicate(
                    path,
                    retained,
                    d_bar=d_bar,
                )
                for retained in selected
            ):
                continue
            covered = (
                path.novel_support_indices
                if path.novel_support_indices is not None
                else path.covered_indices
            )
            if path.novel_support_indices is not None and not covered:
                continue
            overlap = len(covered & used)
            novel = len(covered - used)
            value = novel - overlap_penalty * overlap + 10.0 * path.length
            if value > best_value:
                best_value = value
                best_path = path
        if best_path is None:
            break
        selected.append(best_path)
        best_support = (
            best_path.novel_support_indices
            if best_path.novel_support_indices is not None
            else best_path.covered_indices
        )
        used.update(best_support)
        pool = [path for path in pool if path is not best_path]
        if max_paths is not None and len(selected) >= max_paths:
            break
    if rename_selected:
        for idx, path in enumerate(selected, start=1):
            path.root_id = f"lateral_{idx:03d}"
    return selected


def _is_truncated_path_duplicate(
    candidate: RootPath,
    retained: RootPath,
    *,
    d_bar: float,
) -> bool:
    """Return whether a short trace stays on the basal prefix of a longer one."""

    if candidate.length > 0.35 * retained.length or len(candidate.points) < 2:
        return False
    candidate_direction = _candidate_net_direction(candidate)
    retained_direction = _candidate_net_direction(retained)
    if float(np.dot(candidate_direction, retained_direction)) < float(
        np.cos(np.radians(35.0))
    ):
        return False
    distances, _ = cKDTree(retained.points).query(
        candidate.points,
        k=1,
        workers=worker_threads(),
    )
    tolerance = max(6.0 * float(d_bar), 0.05 * float(retained.length))
    return bool(float(np.quantile(distances, 0.90)) <= tolerance)


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
    dab = tree_b.query(a, k=1, workers=worker_threads())[0].mean()
    dba = tree_a.query(b, k=1, workers=worker_threads())[0].mean()
    return float((dab + dba) / 2.0)


def _novel_support_mask(
    points: np.ndarray,
    *,
    occupied_mask: np.ndarray,
    parent_path: np.ndarray,
    parent_radius_profile: np.ndarray | None,
    d_bar: float,
) -> np.ndarray:
    """Return support that is both unoccupied and outside the parent tube.

    Parent/collar points remain available to bridge a junction during tracing,
    but they cannot make a candidate rank more strongly.  This separates
    geometric reachability from evidence that the path explains new surface.
    """

    points = np.asarray(points, dtype=float)
    occupied = np.asarray(occupied_mask, dtype=bool)
    if occupied.shape != (len(points),):
        raise ValueError("occupied_mask must contain one value per point")
    novel = ~occupied.copy()
    parent = np.asarray(parent_path, dtype=float)
    if len(parent) == 0 or len(points) == 0:
        return novel

    radii = (
        np.asarray(parent_radius_profile, dtype=float)
        if parent_radius_profile is not None
        else np.empty(0, dtype=float)
    )
    if radii.shape != (len(parent),) or not np.all(np.isfinite(radii)):
        radii = np.full(len(parent), max(2.5 * float(d_bar), 0.002), dtype=float)
    distances, nearest = cKDTree(parent).query(points, k=1, workers=worker_threads())
    # Include a sampling margin beyond the robust surface radius so missed
    # parent-shell points at a flared collar are not treated as novel evidence.
    envelope = np.maximum(
        1.35 * radii[np.asarray(nearest, dtype=int)] + 2.0 * float(d_bar),
        max(4.0 * float(d_bar), 0.003),
    )
    novel &= np.asarray(distances, dtype=float) > envelope
    return novel


def _local_support_count(
    tree: cKDTree,
    point: np.ndarray,
    *,
    radius: float,
    support_mask: np.ndarray | None,
) -> int:
    nearby = np.asarray(
        tree.query_ball_point(point, r=radius, workers=worker_threads()),
        dtype=int,
    )
    if support_mask is None:
        return int(len(nearby))
    mask = np.asarray(support_mask, dtype=bool)
    return int(np.count_nonzero(mask[nearby]))


def _local_support_counts(
    tree: cKDTree,
    query_points: np.ndarray,
    *,
    radius: float,
    support_mask: np.ndarray | None,
) -> np.ndarray:
    """Batch local-density queries and optionally count only valid support."""

    query = np.asarray(query_points, dtype=float)
    if not len(query):
        return np.empty(0, dtype=float)
    if support_mask is None:
        return np.asarray(
            tree.query_ball_point(
                query,
                r=float(radius),
                workers=worker_threads(),
                return_length=True,
            ),
            dtype=float,
        )
    neighborhoods = tree.query_ball_point(
        query,
        r=float(radius),
        workers=worker_threads(),
    )
    mask = np.asarray(support_mask, dtype=bool)
    return np.fromiter(
        (
            np.count_nonzero(mask[np.asarray(nearby, dtype=int)])
            for nearby in neighborhoods
        ),
        dtype=float,
        count=len(query),
    )


def _estimate_local_radius(
    tree: cKDTree,
    points: np.ndarray,
    query_point: np.ndarray,
    direction: np.ndarray,
    *,
    radius: float,
    support_mask: np.ndarray | None,
) -> float:
    return float(
        _local_radius_estimates(
            tree,
            points,
            np.asarray(query_point, dtype=float)[None, :],
            np.asarray(direction, dtype=float)[None, :],
            radius=radius,
            support_mask=support_mask,
        )[0]
    )


def _local_radius_estimates(
    tree: cKDTree,
    points: np.ndarray,
    query_points: np.ndarray,
    directions: np.ndarray,
    *,
    radius: float,
    support_mask: np.ndarray | None,
) -> np.ndarray:
    """Estimate a robust local tube scale perpendicular to each trace tangent.

    The value is used only through a ratio to the preceding estimate.  It does
    not claim a calibrated physical radius and cannot reject a proposal.
    """

    query = np.asarray(query_points, dtype=float)
    axes = np.asarray(directions, dtype=float)
    if query.ndim != 2 or query.shape[1] != 3:
        raise ValueError("query_points must have shape (n, 3)")
    if axes.shape != query.shape:
        raise ValueError("directions must match query_points")
    neighborhoods = tree.query_ball_point(
        query,
        r=float(radius),
        workers=worker_threads(),
    )
    mask = None if support_mask is None else np.asarray(support_mask, dtype=bool)
    estimates = np.full(len(query), np.nan, dtype=float)
    source = np.asarray(points, dtype=float)
    for index, nearby_raw in enumerate(neighborhoods):
        nearby = np.asarray(nearby_raw, dtype=int)
        if mask is not None and len(nearby):
            nearby = nearby[mask[nearby]]
        if len(nearby) < 6:
            continue
        axis = axes[index]
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            continue
        axis = axis / axis_norm
        samples = source[nearby]
        centered = samples - np.median(samples, axis=0)
        axial = centered @ axis
        perpendicular = centered - axial[:, None] * axis
        radial = np.linalg.norm(perpendicular, axis=1)
        estimate = float(np.quantile(radial, 0.70))
        if np.isfinite(estimate) and estimate > 1e-12:
            estimates[index] = estimate
    return estimates


def _radius_continuity_scores(
    estimates: np.ndarray,
    reference_radius: float,
) -> np.ndarray:
    """Return soft [0, 1] similarity scores with neutral missing evidence."""

    values = np.asarray(estimates, dtype=float)
    scores = np.full(values.shape, 0.5, dtype=float)
    reference = float(reference_radius)
    if not np.isfinite(reference) or reference <= 1e-12:
        return scores
    valid = np.isfinite(values) & (values > 1e-12)
    if np.any(valid):
        smaller = np.minimum(values[valid], reference)
        larger = np.maximum(values[valid], reference)
        scores[valid] = smaller / np.maximum(larger, 1e-12)
    return scores


def _path_support_count(
    tree: cKDTree,
    path: np.ndarray,
    *,
    radius: float,
    support_mask: np.ndarray | None,
) -> int:
    return len(
        _path_support_indices(
            tree,
            path,
            radius=radius,
            support_mask=support_mask,
        )
    )


def _path_support_indices(
    tree: cKDTree,
    path: np.ndarray,
    *,
    radius: float,
    support_mask: np.ndarray | None,
) -> set[int]:
    covered: set[int] = set()
    mask = None if support_mask is None else np.asarray(support_mask, dtype=bool)
    for node in path:
        nearby = tree.query_ball_point(node, r=radius, workers=worker_threads())
        if mask is None:
            covered.update(int(index) for index in nearby)
        else:
            covered.update(int(index) for index in nearby if mask[int(index)])
    return covered


def _path_density_score(
    points: np.ndarray,
    tree: cKDTree,
    path: np.ndarray,
    radius: float,
    support_mask: np.ndarray | None = None,
) -> float:
    del points  # The tree owns the same coordinates; retained for API compatibility.
    support = _path_support_count(
        tree,
        path,
        radius=radius,
        support_mask=support_mask,
    )
    return float(support + 20.0 * np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


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




