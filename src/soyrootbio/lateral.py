from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from .geometry import (
    nearest_path_tangent,
    path_length,
    point_to_polyline_distance,
    resample_polyline,
    tangent_vectors,
)
from .primary import cluster_hdbscan
from .types import RootPath
from .runtime import worker_threads


MAIN_TRACER_MAX_TURN_DEGREES = 100.0
MAIN_TRACER_TURN_ALIGNMENT_WEIGHT = 0.57
MAIN_TRACER_LOCAL_DENSITY_WEIGHT = 0.20
MAIN_TRACER_STEP_DISTANCE_WEIGHT = 0.15
MAIN_TRACER_RADIUS_CONTINUITY_WEIGHT = 0.08
MAIN_TRACER_OLD_DIRECTION_WEIGHT = 0.75
MAIN_TRACER_NEW_DIRECTION_WEIGHT = 0.25
FORK_HYPOTHESIS_MAX_PER_VARIANT = 2


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
    surface_contact: bool = False
    surface_gap: float | None = None
    surface_contact_count: int = 0
    tip_guard_exception: bool = False
    tip_departure_support: int = 0
    tip_departure_distance: float = 0.0
    tip_departure_extent: float = 0.0
    tip_departure_angle_degrees: float = 0.0


@dataclass(frozen=True)
class _SupportIndex:
    """Spatial index over a filtered subset of the source point cloud."""

    points: np.ndarray
    tree: cKDTree
    source_indices: np.ndarray


def estimate_parent_radius_profile(
    parent_path: np.ndarray,
    parent_support_points: np.ndarray,
    d_bar: float,
    *,
    parent_tree: cKDTree | None = None,
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

    tree = parent_tree if parent_tree is not None else cKDTree(parent_path)
    distances, nearest = tree.query(support, k=1, workers=worker_threads())
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
    *,
    parent_tree: cKDTree | None = None,
) -> tuple[bool, dict[str, float]]:
    """Return whether a candidate is an offset trace of its parent surface.

    Real laterals may share a basal insertion and may briefly curl toward the
    collar. Within the global at-or-below-primary-top origin invariant, this
    parent-tracking test does not reject on insertion height alone. A path is
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
    tree = parent_tree if parent_tree is not None else cKDTree(parent_path)
    distances, nearest = tree.query(sampled, k=1, workers=worker_threads())
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


def is_ancestor_inward_candidate(
    path: RootPath,
    ancestor_path: np.ndarray,
    ancestor_radius_profile: np.ndarray,
    d_bar: float,
    *,
    ancestor_tree: cKDTree | None = None,
) -> tuple[bool, dict[str, float]]:
    """Reject a child that terminates inside an ancestor tube.

    The rule is directional: an outside-to-inside basal stub is invalid, while
    an inside-to-outside path is retained as a plausible emerging lateral.
    Every margin is expressed in the normalized sampling spacing.
    """

    ancestor = np.asarray(ancestor_path, dtype=float)
    radii = np.asarray(ancestor_radius_profile, dtype=float)
    if (
        len(path.points) < 3
        or len(ancestor) < 2
        or radii.shape != (len(ancestor),)
    ):
        return False, {}
    spacing = float(d_bar)
    if not np.isfinite(spacing) or spacing <= 0.0:
        return False, {}

    sampled = resample_polyline(
        path.points,
        spacing=max(2.0 * spacing, path.length / 80.0),
    )
    if len(sampled) < 3:
        return False, {}
    tree = ancestor_tree if ancestor_tree is not None else cKDTree(ancestor)
    distances, nearest = tree.query(
        sampled,
        k=1,
        workers=worker_threads(),
    )
    envelope = radii[np.asarray(nearest, dtype=int)] + 2.0 * spacing
    clearance = np.asarray(distances, dtype=float) - envelope
    segments = np.linalg.norm(np.diff(sampled, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segments)])
    total_arc = float(arc[-1])
    if total_arc <= 1e-12:
        return False, {}

    terminal_start = max(
        0.65 * total_arc,
        total_arc - max(8.0 * spacing, 0.30 * total_arc),
    )
    terminal = arc >= terminal_start
    initial = arc <= min(0.25 * total_arc, max(8.0 * spacing, 0.15 * total_arc))
    if not np.any(initial):
        initial[0] = True
    terminal_inside_fraction = float(np.mean(clearance[terminal] <= 0.0))
    initial_clearance = float(np.median(clearance[initial]))
    terminal_clearance = float(np.median(clearance[terminal]))
    inward_gain = initial_clearance - terminal_clearance
    final_clearance = float(clearance[-1])
    maximum_clearance = float(np.max(clearance))
    rejected = bool(
        terminal_inside_fraction >= 0.75
        and terminal_clearance <= 0.0
        and final_clearance <= 0.0
        and maximum_clearance >= 2.0 * spacing
        and inward_gain >= 2.0 * spacing
    )
    return rejected, {
        "ancestor_terminal_inside_fraction": terminal_inside_fraction,
        "ancestor_initial_clearance": initial_clearance,
        "ancestor_terminal_clearance": terminal_clearance,
        "ancestor_final_clearance": final_clearance,
        "ancestor_maximum_clearance": maximum_clearance,
        "ancestor_inward_clearance_gain": float(inward_gain),
        "ancestor_inward_rejected": float(rejected),
    }


def find_lateral_starting_points(
    points: np.ndarray,
    primary_mask: np.ndarray,
    primary_path: np.ndarray,
    closest_fraction: float = 0.03,
    min_cluster_size: int = 8,
    max_parent_distance: float | np.ndarray | None = None,
    minimum_branch_angle_degrees: float = 0.0,
    exclude_parent_tip_fraction: float = 0.0,
    parent_surface_points: np.ndarray | None = None,
    surface_contact_distance: float | None = None,
    parent_tree: cKDTree | None = None,
    tip_departure_distance: float | None = None,
) -> list[LateralStart]:
    """Cluster non-primary points closest to the primary root as branch starts.

    The closest_fraction parameter is interpreted as a percentile distance
    threshold, matching the paper-inspired nearest-boundary seed step.
    """
    non_primary = np.flatnonzero(~primary_mask)
    if len(non_primary) == 0:
        return []
    primary_tree = parent_tree if parent_tree is not None else cKDTree(primary_path)
    distances, primary_indices = point_to_polyline_distance(
        points[non_primary],
        primary_path,
        path_tree=primary_tree,
    )
    primary_indices = np.asarray(primary_indices, dtype=int)
    surface_contact = np.zeros(len(non_primary), dtype=bool)
    surface_distances = np.full(len(non_primary), np.inf, dtype=float)
    surface_parent_indices = primary_indices.copy()
    surface = (
        np.asarray(parent_surface_points, dtype=float)
        if parent_surface_points is not None
        else np.empty((0, 3), dtype=float)
    )
    if len(surface):
        if surface.ndim != 2 or surface.shape[1] != 3:
            raise ValueError("parent_surface_points must contain XYZ points")
        contact_limit = float(surface_contact_distance or 0.0)
        if not np.isfinite(contact_limit) or contact_limit <= 0.0:
            raise ValueError(
                "surface_contact_distance must be positive when "
                "parent_surface_points are provided"
            )
        surface_distances, surface_matches = cKDTree(surface).query(
            points[non_primary],
            k=1,
            workers=worker_threads(),
        )
        _, surface_to_parent = primary_tree.query(
            surface,
            k=1,
            workers=worker_threads(),
        )
        surface_parent_indices = np.asarray(surface_to_parent, dtype=int)[
            np.asarray(surface_matches, dtype=int)
        ]
        surface_contact = np.asarray(surface_distances, dtype=float) <= contact_limit
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
        eligible &= (distances <= limits) | surface_contact
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
    primary_tangents = tangent_vectors(primary_path)
    non_primary_positions = {
        int(point_index): position
        for position, point_index in enumerate(non_primary)
    }
    for label in sorted(label for label in np.unique(labels) if label >= 0):
        members = seed_indices[labels == label]
        if len(members) == 0:
            continue
        cluster_points = points[members]
        distances_to_primary, primary_matches = primary_tree.query(cluster_points, k=1)
        member_positions = np.asarray(
            [non_primary_positions[int(index)] for index in members],
            dtype=int,
        )
        member_surface_contact = surface_contact[member_positions]
        contact_count = int(np.count_nonzero(member_surface_contact))
        if contact_count >= min(3, len(members)):
            contacting_members = np.flatnonzero(member_surface_contact)
            best_member = int(
                contacting_members[
                    np.argmin(
                        surface_distances[
                            member_positions[contacting_members]
                        ]
                    )
                ]
            )
            primary_idx = int(
                surface_parent_indices[member_positions[best_member]]
            )
            used_surface_contact = True
            selected_surface_gap = float(
                surface_distances[member_positions[best_member]]
            )
        else:
            best_member = int(np.argmin(distances_to_primary))
            primary_idx = int(primary_matches[best_member])
            used_surface_contact = False
            selected_surface_gap = None
            contact_count = 0
        start_point = cluster_points[best_member]
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
        tip_guard_nodes = int(
            np.ceil(float(exclude_parent_tip_fraction) * len(primary_path))
        )
        inside_tip_guard = bool(
            tip_guard_nodes > 0
            and primary_idx >= len(primary_path) - tip_guard_nodes
        )
        tip_exception = False
        tip_evidence = {
            "support": 0,
            "departure": 0.0,
            "extent": 0.0,
            "angle": branch_angle,
        }
        if inside_tip_guard and tip_departure_distance is not None:
            tip_exception, tip_evidence = _supported_tip_departure(
                cluster_points,
                primary_path[primary_idx],
                direction / direction_norm,
                branch_angle_degrees=branch_angle,
                minimum_support=max(4, int(min_cluster_size)),
                minimum_departure=float(tip_departure_distance),
            )
        if inside_tip_guard and not tip_exception:
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
                surface_contact=used_surface_contact,
                surface_gap=selected_surface_gap,
                surface_contact_count=contact_count,
                tip_guard_exception=tip_exception,
                tip_departure_support=int(tip_evidence["support"]),
                tip_departure_distance=float(tip_evidence["departure"]),
                tip_departure_extent=float(tip_evidence["extent"]),
                tip_departure_angle_degrees=float(tip_evidence["angle"]),
            )
        )
    valid_labels = [label for label in np.unique(labels) if label >= 0]
    if not starts and not valid_labels and len(seed_indices) >= max(2, int(min_cluster_size)):
        seed_points = points[seed_indices]
        distances_to_primary, primary_matches = primary_tree.query(seed_points, k=1)
        member_positions = np.asarray(
            [non_primary_positions[int(index)] for index in seed_indices],
            dtype=int,
        )
        member_surface_contact = surface_contact[member_positions]
        contact_count = int(np.count_nonzero(member_surface_contact))
        if contact_count >= min(3, len(seed_indices)):
            contacting_members = np.flatnonzero(member_surface_contact)
            best_member = int(
                contacting_members[
                    np.argmin(
                        surface_distances[
                            member_positions[contacting_members]
                        ]
                    )
                ]
            )
            primary_idx = int(
                surface_parent_indices[member_positions[best_member]]
            )
            used_surface_contact = True
            selected_surface_gap = float(
                surface_distances[member_positions[best_member]]
            )
        else:
            best_member = int(np.argmin(distances_to_primary))
            primary_idx = int(primary_matches[best_member])
            used_surface_contact = False
            selected_surface_gap = None
            contact_count = 0
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
        tip_guard_nodes = int(
            np.ceil(float(exclude_parent_tip_fraction) * len(primary_path))
        )
        inside_tip_guard = bool(
            tip_guard_nodes > 0
            and primary_idx >= len(primary_path) - tip_guard_nodes
        )
        tip_exception = False
        tip_evidence = {
            "support": 0,
            "departure": 0.0,
            "extent": 0.0,
            "angle": branch_angle,
        }
        if inside_tip_guard and tip_departure_distance is not None:
            tip_exception, tip_evidence = _supported_tip_departure(
                seed_points,
                primary_path[primary_idx],
                direction / direction_norm,
                branch_angle_degrees=branch_angle,
                minimum_support=max(4, int(min_cluster_size)),
                minimum_departure=float(tip_departure_distance),
            )
        if inside_tip_guard and not tip_exception:
            return starts
        starts.append(
            LateralStart(
                0,
                seed_points[best_member],
                primary_path[primary_idx],
                primary_idx,
                seed_indices,
                direction,
                surface_contact=used_surface_contact,
                surface_gap=selected_surface_gap,
                surface_contact_count=contact_count,
                tip_guard_exception=tip_exception,
                tip_departure_support=int(tip_evidence["support"]),
                tip_departure_distance=float(tip_evidence["departure"]),
                tip_departure_extent=float(tip_evidence["extent"]),
                tip_departure_angle_degrees=float(tip_evidence["angle"]),
            )
        )
    return starts


def _supported_tip_departure(
    cluster_points: np.ndarray,
    parent_point: np.ndarray,
    direction: np.ndarray,
    *,
    branch_angle_degrees: float,
    minimum_support: int,
    minimum_departure: float,
) -> tuple[bool, dict[str, float | int]]:
    """Require a coherent departing tube before bypassing the parent-tip guard.

    The final fraction of a parent is normally excluded because terminal-cap
    points easily masquerade as a child seed.  A connected seed cluster may
    bypass that guard only when it contains several points, departs at a clear
    angle, reaches a sampling-scaled distance from the parent, and has sustained
    extent along the departing direction.  This is deliberately a seed-only
    exception; the candidate must still pass ordinary tracing and ownership
    checks.
    """

    cluster = np.asarray(cluster_points, dtype=float)
    origin = np.asarray(parent_point, dtype=float)
    axis = np.asarray(direction, dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    vectors = cluster - origin
    radial = np.linalg.norm(vectors, axis=1)
    projection = vectors @ axis
    departure = float(np.quantile(radial, 0.90)) if len(radial) else 0.0
    positive_projection = projection[projection > 0.0]
    extent = (
        float(np.quantile(positive_projection, 0.90))
        if len(positive_projection)
        else 0.0
    )
    support = int(np.count_nonzero(projection >= 0.25 * minimum_departure))
    accepted = bool(
        len(cluster) >= int(minimum_support)
        and support >= max(3, int(np.ceil(0.50 * minimum_support)))
        and float(branch_angle_degrees) >= 30.0
        and departure >= float(minimum_departure)
        and extent >= 0.75 * float(minimum_departure)
    )
    return accepted, {
        "support": support,
        "departure": departure,
        "extent": extent,
        "angle": float(branch_angle_degrees),
    }


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
    ancestor_exclusion_mask: np.ndarray | None = None,
    point_tree: cKDTree | None = None,
    parent_tree: cKDTree | None = None,
) -> list[RootPath]:
    if not starts:
        return []
    point_tree = point_tree if point_tree is not None else cKDTree(points)
    primary_tangents = tangent_vectors(primary_path)
    allowed_mask = ~np.asarray(primary_mask, dtype=bool)
    novel_support_mask = _novel_support_mask(
        points,
        occupied_mask=primary_mask,
        parent_path=primary_path,
        parent_radius_profile=parent_radius_profile,
        d_bar=d_bar,
        parent_tree=parent_tree,
    )
    if ancestor_exclusion_mask is not None:
        ancestor_exclusion = np.asarray(
            ancestor_exclusion_mask,
            dtype=bool,
        )
        if ancestor_exclusion.shape != (len(points),):
            raise ValueError(
                "ancestor_exclusion_mask must contain one value per point"
            )
        novel_support_mask &= ~ancestor_exclusion
    # Count only biologically eligible support in SciPy's batched
    # ``return_length`` path, without materialising full-cloud neighbor lists.
    novel_support_index = _build_support_index(points, novel_support_mask)
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

        def trace_variants(
            initial_direction: np.ndarray,
            direction_label: str,
            multipliers: tuple[float, ...],
            angles: tuple[float, ...],
        ) -> None:
            for multiplier in multipliers:
                step_length = max(multiplier * d_bar, 0.004)
                for open_angle in angles:
                    paths = _grow_candidate_hypotheses(
                        points=points,
                        point_tree=point_tree,
                        allowed_mask=allowed_mask,
                        start=start,
                        initial_direction=initial_direction,
                        primary_tangent=primary_tangent,
                        step_length=step_length,
                        open_angle=open_angle,
                        max_steps=max_steps,
                        search_radius=search_radius_factor * step_length,
                        limit_primary_angle_to_insertion=True,
                        density_support_index=novel_support_index,
                        cooperate=cooperate,
                    )
                    hypothesis_group = (
                        f"{start.start_id}:{direction_label}:"
                        f"{multiplier:g}:{int(open_angle)}"
                    )
                    for hypothesis_index, path in enumerate(paths):
                        if len(path.points) < 3:
                            continue
                        path.root_id = (
                            f"lateral_{start.start_id}_{direction_label}"
                            f"_s{multiplier:g}_a{int(open_angle)}"
                            f"_h{hypothesis_index}"
                        )
                        path.fork_hypothesis_group = hypothesis_group
                        path.fork_hypothesis_index = hypothesis_index
                        path.novel_support_indices = _path_support_indices(
                            novel_support_index.tree,
                            path.points,
                            radius=max(2.0 * d_bar, 0.004),
                            support_mask=None,
                            source_indices=novel_support_index.source_indices,
                        )
                        path.score_components["novel_density_support"] = float(
                            len(path.novel_support_indices)
                        )
                        growth_length = float(
                            path.score_components.get(
                                "trace_growth_arc",
                                path.length,
                            )
                        )
                        longest_path_reward = (
                            0.35 * growth_length / max(float(d_bar), 1e-12)
                        )
                        path.score_components["longest_path_reward"] = float(
                            longest_path_reward
                        )
                        path.score = float(
                            len(path.novel_support_indices)
                            + 20.0 * growth_length
                            + longest_path_reward
                            + path.score_components.get("trace_rank_score", 0.0)
                            + path.score_components.get(
                                "fork_hypothesis_evidence_score",
                                0.0,
                            )
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


def _grow_candidate_hypotheses(
    *,
    points: np.ndarray,
    point_tree: cKDTree,
    allowed_mask: np.ndarray,
    start: LateralStart,
    initial_direction: np.ndarray,
    primary_tangent: np.ndarray,
    step_length: float,
    open_angle: float,
    max_steps: int,
    search_radius: float,
    limit_primary_angle_to_insertion: bool,
    density_support_index: _SupportIndex | None,
    cooperate: Callable[[], None] | None,
) -> list[RootPath]:
    """Trace the default route plus one sustained recent-fork alternative.

    The alternate is produced by replaying the shared prefix and forcing one
    angularly distinct, surface-supported proposal at the unresolved fork.
    It is retained only when several subsequent nodes form a departing tube;
    the proposal score at the fork cannot by itself create a hypothesis.
    """

    observations: list[dict[str, float | int]] = []
    base = _grow_one_candidate(
        points=points,
        point_tree=point_tree,
        allowed_mask=allowed_mask,
        start=start,
        initial_direction=initial_direction,
        primary_tangent=primary_tangent,
        step_length=step_length,
        open_angle=open_angle,
        max_steps=max_steps,
        search_radius=search_radius,
        limit_primary_angle_to_insertion=limit_primary_angle_to_insertion,
        density_support_index=density_support_index,
        cooperate=cooperate,
        fork_observations=observations,
    )
    base_steps = max(0, len(base.points) - 2)
    recent_window = max(6, min(14, int(np.ceil(0.25 * max(base_steps, 1)))))
    recent = [
        observation
        for observation in observations
        if base_steps - int(observation["step_index"]) <= recent_window
    ]
    if not recent:
        base.score_components["fork_hypothesis_count"] = 1.0
        return [base]

    # Prefer the most distal supported separation.  Support, radius continuity,
    # and the alternate score make the ordering deterministic when two events
    # occur at the same step.
    recent.sort(
        key=lambda observation: (
            int(observation["step_index"]),
            float(observation["alternate_local_support"]),
            float(observation["alternate_radius_similarity"]),
            float(observation["alternate_step_score"]),
            -int(observation["alternate_point_index"]),
        ),
        reverse=True,
    )
    retained: list[RootPath] = [base]
    for observation in recent:
        if len(retained) >= FORK_HYPOTHESIS_MAX_PER_VARIANT:
            break
        fork_step = int(observation["step_index"])
        alternate = _grow_one_candidate(
            points=points,
            point_tree=point_tree,
            allowed_mask=allowed_mask,
            start=start,
            initial_direction=initial_direction,
            primary_tangent=primary_tangent,
            step_length=step_length,
            open_angle=open_angle,
            max_steps=max_steps,
            search_radius=search_radius,
            limit_primary_angle_to_insertion=limit_primary_angle_to_insertion,
            density_support_index=density_support_index,
            cooperate=cooperate,
            forced_step_indices={
                fork_step: int(observation["alternate_point_index"])
            },
        )
        common_prefix = _common_prefix_node_count(
            base.points,
            alternate.points,
            tolerance=max(0.10 * float(step_length), 1e-9),
        )
        alternate_suffix = np.asarray(
            alternate.points[max(0, common_prefix - 1) :],
            dtype=float,
        )
        base_suffix = np.asarray(
            base.points[max(0, common_prefix - 1) :],
            dtype=float,
        )
        suffix_arc = path_length(alternate_suffix)
        base_suffix_arc = path_length(base_suffix)
        suffix_extent = (
            float(np.linalg.norm(alternate_suffix[-1] - alternate_suffix[0]))
            if len(alternate_suffix) >= 2
            else 0.0
        )
        base_suffix_extent = (
            float(np.linalg.norm(base_suffix[-1] - base_suffix[0]))
            if len(base_suffix) >= 2
            else 0.0
        )
        alternate_to_base, _ = cKDTree(base.points).query(
            alternate_suffix,
            k=1,
            workers=worker_threads(),
        )
        departure = (
            float(np.quantile(alternate_to_base, 0.75))
            if len(alternate_to_base)
            else 0.0
        )
        suffix_nodes = max(0, len(alternate.points) - common_prefix)
        sustained = bool(
            common_prefix >= 2
            and suffix_nodes >= 4
            and suffix_extent >= max(
                3.0 * float(step_length),
                0.75 * float(search_radius),
            )
            and suffix_extent / max(suffix_arc, 1e-12) >= 0.35
            and departure >= max(
                1.50 * float(search_radius),
                2.25
                * max(
                    float(
                        base.score_components.get(
                            "trace_local_radius",
                            0.0,
                        )
                    ),
                    float(
                        alternate.score_components.get(
                            "trace_local_radius",
                            0.0,
                        )
                    ),
                ),
            )
            and float(observation["arm_angle_degrees"]) >= 42.0
            and float(
                alternate.score_components.get("trace_mean_support", 0.0)
            )
            >= 1.0
        )
        if not sustained:
            continue
        selected_support = float(observation["selected_local_support"])
        alternate_support = float(observation["alternate_local_support"])
        support_scale = max(selected_support, alternate_support, 1.0)
        extent_scale = max(base_suffix_extent, suffix_extent, 1e-12)
        selected_curvature = float(
            base.score_components.get("trace_smoothness", 0.0)
        )
        alternate_curvature = float(
            alternate.score_components.get("trace_smoothness", 0.0)
        )
        selected_evidence_score = float(
            0.30 * selected_support / support_scale
            + 0.25 * float(observation["selected_radius_similarity"])
            + 0.20 * selected_curvature
            + 0.25 * base_suffix_extent / extent_scale
        )
        alternate_evidence_score = float(
            0.30 * alternate_support / support_scale
            + 0.25 * float(observation["alternate_radius_similarity"])
            + 0.20 * alternate_curvature
            + 0.25 * suffix_extent / extent_scale
        )
        for path, hypothesis_index in ((base, 0), (alternate, 1)):
            path.fork_hypothesis_index = hypothesis_index
            path.fork_common_prefix_nodes = common_prefix
            path.score_components.update(
                {
                    "fork_hypothesis_count": 2.0,
                    "fork_hypothesis_index": float(hypothesis_index),
                    "fork_common_prefix_nodes": float(common_prefix),
                    "fork_step_index": float(fork_step),
                    "fork_arm_angle_degrees": float(
                        observation["arm_angle_degrees"]
                    ),
                    "fork_selected_local_support": float(
                        observation["selected_local_support"]
                    ),
                    "fork_alternate_local_support": float(
                        observation["alternate_local_support"]
                    ),
                    "fork_selected_radius_similarity": float(
                        observation["selected_radius_similarity"]
                    ),
                    "fork_alternate_radius_similarity": float(
                        observation["alternate_radius_similarity"]
                    ),
                    "fork_selected_local_radius": float(
                        observation["selected_local_radius"]
                    ),
                    "fork_alternate_local_radius": float(
                        observation["alternate_local_radius"]
                    ),
                    "fork_alternate_departure": departure,
                    "fork_alternate_supported_extent": suffix_extent,
                    "fork_alternate_supported_arc": suffix_arc,
                    "fork_selected_supported_extent": base_suffix_extent,
                    "fork_selected_supported_arc": base_suffix_arc,
                    "fork_selected_curvature_window_smoothness": (
                        selected_curvature
                    ),
                    "fork_alternate_curvature_window_smoothness": (
                        alternate_curvature
                    ),
                    "fork_hypothesis_evidence_score": (
                        selected_evidence_score
                        if hypothesis_index == 0
                        else alternate_evidence_score
                    ),
                }
            )
        retained.append(alternate)
    base.score_components["fork_hypothesis_count"] = float(len(retained))
    return retained


def _common_prefix_node_count(
    left: np.ndarray,
    right: np.ndarray,
    *,
    tolerance: float,
) -> int:
    count = 0
    for left_point, right_point in zip(
        np.asarray(left, dtype=float),
        np.asarray(right, dtype=float),
    ):
        if float(np.linalg.norm(left_point - right_point)) > float(tolerance):
            break
        count += 1
    return count


def _grow_one_candidate(
    points: np.ndarray,
    point_tree: cKDTree,
    allowed_mask: np.ndarray,
    start: LateralStart,
    initial_direction: np.ndarray,
    primary_tangent: np.ndarray,
    step_length: float,
    open_angle: float,
    max_steps: int,
    search_radius: float,
    limit_primary_angle_to_insertion: bool = False,
    max_turn_degrees: float = MAIN_TRACER_MAX_TURN_DEGREES,
    minimum_local_support: int = 1,
    cooperate: Callable[[], None] | None = None,
    density_support_mask: np.ndarray | None = None,
    density_support_index: _SupportIndex | None = None,
    forced_step_indices: dict[int, int] | None = None,
    fork_observations: list[dict[str, float | int]] | None = None,
) -> RootPath:
    """Grow one trace and optionally record or force supported fork choices."""

    allowed = np.asarray(allowed_mask, dtype=bool)
    if allowed.shape != (len(points),):
        raise ValueError("allowed_mask must contain one value per point")
    initial = np.asarray(initial_direction, dtype=float).copy()
    initial /= max(float(np.linalg.norm(initial)), 1e-12)
    parent_tangent = np.asarray(primary_tangent, dtype=float).copy()
    parent_tangent /= max(float(np.linalg.norm(parent_tangent)), 1e-12)
    nodes = [
        np.asarray(start.primary_point, dtype=float).copy(),
        np.asarray(start.point, dtype=float).copy(),
    ]
    current = nodes[-1].copy()
    direction = initial
    covered_mask = np.zeros(len(points), dtype=bool)
    selected_mask = np.zeros(len(points), dtype=bool)
    covered_members: list[int] = []
    support_tree = point_tree
    support_points = np.asarray(points, dtype=float)
    support_mask = density_support_mask
    if density_support_index is not None:
        support_tree = density_support_index.tree
        support_points = density_support_index.points
        support_mask = None
    local_radius = _estimate_local_radius(
        support_tree,
        support_points,
        current,
        direction,
        radius=0.75 * float(search_radius),
        support_mask=support_mask,
    )
    growth_arc = 0.0
    support_sum = 0.0
    turn_squared_sum = 0.0
    cumulative_turn_degrees = 0.0
    cumulative_score = 0.0
    radius_similarity_sum = 0.0
    radius_observations = 0
    fallback_steps = 0
    covered_recovery_steps = 0
    previous_turn_degrees = 0.0
    travel_fractions: list[float] = []

    for step_index in range(max_steps):
        if cooperate is not None and step_index % 8 == 0:
            cooperate()
        nearby_indices = point_tree.query_ball_point(
            current,
            r=search_radius,
            workers=worker_threads(),
        )
        local = np.asarray(nearby_indices, dtype=int)
        if len(local):
            local = local[allowed[local]]
        if len(local) < max(1, int(minimum_local_support)):
            break

        vectors = points[local] - current
        distances = np.linalg.norm(vectors, axis=1)
        nonzero = distances > 1e-12
        if not np.any(nonzero):
            break
        local = local[nonzero]
        vectors = vectors[nonzero]
        distances = distances[nonzero]
        unit = vectors / distances[:, None]
        turn_cos = unit @ direction
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
            open_ok = open_angle_to_primary <= float(open_angle)
            direction_ok = (
                turn_ok & open_ok
                if np.any(turn_ok & open_ok)
                else turn_ok
            )
        else:
            direction_ok = turn_ok

        travel_fraction = _adaptive_minimum_travel_fraction(
            step_index=step_index,
            previous_turn_degrees=previous_turn_degrees,
            max_turn_degrees=max_turn_degrees,
        )
        uncovered = ~covered_mask[local]
        valid = (
            direction_ok
            & uncovered
            & (distances >= travel_fraction * float(step_length))
        )
        if not np.any(valid) and travel_fraction > 0.30:
            fallback_fraction = _adaptive_minimum_travel_fraction(
                step_index=step_index,
                previous_turn_degrees=previous_turn_degrees,
                max_turn_degrees=max_turn_degrees,
                fallback=True,
            )
            valid = (
                direction_ok
                & uncovered
                & (distances >= fallback_fraction * float(step_length))
            )
            if np.any(valid):
                travel_fraction = fallback_fraction
                fallback_steps += 1
        if not np.any(valid):
            # The accepted-node halo deliberately clears an entire sampled
            # cross-section so a trace cannot crawl across a fused surface.
            # If that also masks the only forward continuation at a tight
            # bend or sparse gap, recover a short step without permitting a
            # return to an earlier centerline node.
            recovery_fraction = _adaptive_minimum_travel_fraction(
                step_index=step_index,
                previous_turn_degrees=previous_turn_degrees,
                max_turn_degrees=max_turn_degrees,
                fallback=True,
            )
            not_selected = ~selected_mask[local]
            recovery = (
                direction_ok
                & not_selected
                & (distances >= recovery_fraction * float(step_length))
            )
            if np.any(recovery) and len(nodes) > 2:
                prior_distances, _ = cKDTree(
                    np.asarray(nodes[:-1], dtype=float)
                ).query(
                    points[local],
                    k=1,
                    workers=worker_threads(),
                )
                recovery &= np.asarray(prior_distances, dtype=float) >= (
                    0.29 * float(step_length)
                )
            if np.any(recovery):
                valid = recovery
                travel_fraction = recovery_fraction
                fallback_steps += 1
                covered_recovery_steps += 1
        if not np.any(valid):
            break

        local = local[valid]
        unit = unit[valid]
        distances = distances[valid]
        turn_cos = turn_cos[valid]
        density = _local_support_counts(
            support_tree,
            points[local],
            radius=0.75 * float(search_radius),
            support_mask=support_mask,
        )
        distance_score = -np.abs(distances - step_length) / max(
            float(step_length),
            1e-12,
        )
        base_score = (
            MAIN_TRACER_TURN_ALIGNMENT_WEIGHT * turn_cos
            + MAIN_TRACER_LOCAL_DENSITY_WEIGHT * _normalize(density)
            + MAIN_TRACER_STEP_DISTANCE_WEIGHT * distance_score
        )

        # Radius estimation is deliberately limited to the strongest few
        # proposals.  It rewards a coherent tube scale without making radius
        # a rejection threshold or restoring a multi-hypothesis beam.
        shortlist_size = min(6, len(local))
        shortlist = np.argsort(-base_score, kind="stable")[:shortlist_size]
        shortlisted_radii = _local_radius_estimates(
            support_tree,
            support_points,
            points[local[shortlist]],
            unit[shortlist],
            radius=0.75 * float(search_radius),
            support_mask=support_mask,
        )
        radius_similarity = np.full(len(local), 0.5, dtype=float)
        radius_similarity[shortlist] = _radius_continuity_scores(
            shortlisted_radii,
            local_radius,
        )
        score = (
            base_score
            + MAIN_TRACER_RADIUS_CONTINUITY_WEIGHT * radius_similarity
        )
        ranked_positions = np.argsort(-score, kind="stable")
        best_position = int(ranked_positions[0])
        forced_step = bool(
            forced_step_indices and step_index in forced_step_indices
        )
        if forced_step:
            forced_matches = np.flatnonzero(
                local == int(forced_step_indices[step_index])
            )
            if len(forced_matches):
                best_position = int(forced_matches[0])
        if fork_observations is not None and len(ranked_positions) > 1:
            selected_direction = unit[best_position]
            selected_density = float(density[best_position])
            selected_score = float(score[best_position])
            for alternate_position_raw in ranked_positions:
                alternate_position = int(alternate_position_raw)
                if alternate_position == best_position:
                    continue
                separation = float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                np.dot(
                                    selected_direction,
                                    unit[alternate_position],
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                if separation < 42.0 or separation > 145.0:
                    continue
                alternate_density = float(density[alternate_position])
                alternate_score = float(score[alternate_position])
                if alternate_density < max(
                    float(minimum_local_support),
                    0.35 * selected_density,
                ):
                    continue
                # The turn term alone can differ by 0.57 for a right-angle
                # fork. Keep the alternate for sustained-window validation;
                # the one-point tangent preference must not suppress it here.
                if alternate_score < selected_score - 0.75:
                    continue
                fork_observations.append(
                    {
                        "step_index": int(step_index),
                        "selected_point_index": int(local[best_position]),
                        "alternate_point_index": int(local[alternate_position]),
                        "arm_angle_degrees": separation,
                        "selected_step_score": selected_score,
                        "alternate_step_score": alternate_score,
                        "selected_local_support": selected_density,
                        "alternate_local_support": alternate_density,
                        "selected_radius_similarity": float(
                            radius_similarity[best_position]
                        ),
                        "alternate_radius_similarity": float(
                            radius_similarity[alternate_position]
                        ),
                        "selected_local_radius": float(
                            shortlisted_radii[int(selected_radius_match[0])]
                        )
                        if len(
                            selected_radius_match := np.flatnonzero(
                                shortlist == best_position
                            )
                        )
                        else 0.0,
                        "alternate_local_radius": float(
                            shortlisted_radii[int(alternate_radius_match[0])]
                        )
                        if len(
                            alternate_radius_match := np.flatnonzero(
                                shortlist == alternate_position
                            )
                        )
                        else 0.0,
                    }
                )
                break
        next_index = int(local[best_position])
        next_point = np.asarray(points[next_index], dtype=float)
        segment = next_point - current
        segment_length = float(np.linalg.norm(segment))
        new_direction = segment / max(segment_length, 1e-12)
        turn_angle = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(direction, new_direction),
                        -1.0,
                        1.0,
                    )
                )
            )
        )

        proposed_radius = np.nan
        shortlist_match = np.flatnonzero(shortlist == best_position)
        if len(shortlist_match):
            proposed_radius = float(shortlisted_radii[int(shortlist_match[0])])
        if np.isfinite(proposed_radius) and proposed_radius > 0.0:
            similarity_value = float(radius_similarity[best_position])
            local_radius = (
                0.65 * float(local_radius) + 0.35 * proposed_radius
                if np.isfinite(local_radius) and local_radius > 0.0
                else proposed_radius
            )
            radius_similarity_sum += similarity_value
            radius_observations += 1

        evolved_direction = (
            new_direction
            if forced_step
            else (
                MAIN_TRACER_OLD_DIRECTION_WEIGHT * direction
                + MAIN_TRACER_NEW_DIRECTION_WEIGHT * new_direction
            )
        )
        direction = evolved_direction / max(
            float(np.linalg.norm(evolved_direction)),
            1e-12,
        )
        current = next_point
        nodes.append(current.copy())
        selected_mask[next_index] = True
        local_covered = point_tree.query_ball_point(
            current,
            # Inspect a full local cross-section after accepting a step.  Only
            # points at or behind that section are consumed below; forward
            # points remain available to the ordinary adaptive travel rule.
            r=0.90 * float(search_radius),
            workers=worker_threads(),
        )
        local_covered_array = np.asarray(local_covered, dtype=int)
        if len(local_covered_array):
            axial_progress = (
                points[local_covered_array] - current
            ) @ direction
            local_covered_array = local_covered_array[
                axial_progress <= 0.29 * float(step_length)
            ]
        if len(local_covered_array):
            local_covered_array = local_covered_array[allowed[local_covered_array]]
        if len(local_covered_array):
            newly_covered = local_covered_array[~covered_mask[local_covered_array]]
            covered_mask[local_covered_array] = True
            covered_members.extend(newly_covered.tolist())
        growth_arc += segment_length
        support_sum += float(density[best_position])
        turn_squared_sum += turn_angle**2
        cumulative_turn_degrees += turn_angle
        cumulative_score += float(score[best_position])
        previous_turn_degrees = turn_angle
        travel_fractions.append(float(travel_fraction))

    steps = max(0, len(nodes) - 2)
    turn_rms = float(
        np.sqrt(turn_squared_sum / max(1, steps))
    )
    radius_similarity_mean = (
        float(radius_similarity_sum / radius_observations)
        if radius_observations
        else 0.5
    )
    supported_steps = growth_arc / max(float(step_length), 1e-12)
    turn_scale = max(float(max_turn_degrees), 1.0)
    curvature_penalty = (turn_rms / turn_scale) ** 2
    cumulative_turn_penalty = (
        cumulative_turn_degrees
        / max(turn_scale * max(1, steps), 1e-12)
    )
    trace_rank_score = float(
        cumulative_score
        + 0.35 * supported_steps
        + 0.20 * np.log1p(support_sum)
        + 0.12 * radius_similarity_mean
        - 0.25 * curvature_penalty
        - 0.06 * cumulative_turn_penalty
    )
    path = RootPath(
        root_id="candidate",
        points=np.asarray(nodes, dtype=float),
        raw_start_point=np.asarray(start.point, dtype=float).copy(),
        covered_indices=set(covered_members),
    )
    path.score_components.update(
        {
            "trace_growth_arc": float(growth_arc),
            "trace_supported_arc": float(growth_arc),
            "trace_mean_support": float(support_sum / max(1, steps)),
            "trace_turn_rms_degrees": turn_rms,
            "trace_cumulative_turn_degrees": float(
                cumulative_turn_degrees
            ),
            "trace_smoothness": float(
                np.exp(-(turn_rms / turn_scale) ** 2)
            ),
            "trace_local_radius": (
                float(local_radius)
                if np.isfinite(local_radius)
                else 0.0
            ),
            "trace_radius_similarity": radius_similarity_mean,
            "trace_radius_observations": float(radius_observations),
            "trace_cumulative_score": float(cumulative_score),
            "trace_rank_score": trace_rank_score,
            "adaptive_travel_fallback_steps": float(fallback_steps),
            "adaptive_travel_covered_recovery_steps": float(
                covered_recovery_steps
            ),
            "adaptive_travel_fraction_min": float(
                min(travel_fractions, default=0.0)
            ),
            "adaptive_travel_fraction_max": float(
                max(travel_fractions, default=0.0)
            ),
            "surface_aware_seed": float(start.surface_contact),
            "surface_seed_gap": (
                float(start.surface_gap)
                if start.surface_gap is not None
                else 0.0
            ),
            "surface_seed_contact_count": float(
                start.surface_contact_count
            ),
            "tip_guard_exception": float(start.tip_guard_exception),
            "tip_departure_support": float(start.tip_departure_support),
            "tip_departure_distance": float(start.tip_departure_distance),
            "tip_departure_extent": float(start.tip_departure_extent),
            "tip_departure_angle_degrees": float(
                start.tip_departure_angle_degrees
            ),
        }
    )
    return path


def _adaptive_minimum_travel_fraction(
    *,
    step_index: int,
    previous_turn_degrees: float,
    max_turn_degrees: float,
    fallback: bool = False,
) -> float:
    """Return a step-relative travel threshold with no physical units."""

    if fallback:
        return 0.30
    if int(step_index) < 2:
        return 0.30
    if float(previous_turn_degrees) >= 0.50 * float(max_turn_degrees):
        return 0.30
    return 0.375


TIP_EXTENSION_MAX_STEPS = 90
TIP_EXTENSION_BATCH_STEPS = 30


def extend_lateral_tip(
    points: np.ndarray,
    path: RootPath,
    blocked_mask: np.ndarray,
    d_bar: float,
    *,
    max_steps: int = TIP_EXTENSION_MAX_STEPS,
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
        allowed_mask=available,
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


def resume_lateral_tip_in_batches(
    points: np.ndarray,
    path: RootPath,
    blocked_mask: np.ndarray,
    d_bar: float,
    *,
    max_steps: int = TIP_EXTENSION_MAX_STEPS,
    batch_steps: int = TIP_EXTENSION_BATCH_STEPS,
    min_support: int = 4,
    point_tree: cKDTree | None = None,
    cooperate: Callable[[], None] | None = None,
) -> RootPath:
    """Resume supported growth in bounded, ownership-safe batches.

    Each batch re-probes unowned support from the newly accepted tip. A batch
    is rolled back if its support overlaps a blocked owner or its endpoint
    loops into a nonterminal part of the existing path. The total cap remains
    unchanged; batching only permits evidence to be reconsidered between caps.
    """

    cloud = np.asarray(points, dtype=float)
    blocked = np.asarray(blocked_mask, dtype=bool).copy()
    if blocked.shape != (len(cloud),):
        raise ValueError("blocked_mask must have one value per point")
    tree = point_tree if point_tree is not None else cKDTree(cloud)
    total_step_limit = max(1, int(max_steps))
    per_batch = max(3, int(batch_steps))
    total_steps = 0
    total_length = 0.0
    total_new_support = 0
    accepted_batches = 0
    loop_rejections = 0
    ownership_rejections = 0
    stopped_on_open_support = False

    while total_steps < total_step_limit:
        if cooperate is not None:
            cooperate()
        before_points = np.asarray(path.points, dtype=float).copy()
        before_covered = set(path.covered_indices)
        before_node_indices = path.node_indices
        remaining = total_step_limit - total_steps
        this_batch_limit = min(per_batch, remaining)
        extend_lateral_tip(
            cloud,
            path,
            blocked,
            d_bar,
            max_steps=this_batch_limit,
            min_support=min_support,
            point_tree=tree,
            cooperate=cooperate,
        )
        accepted = bool(
            path.score_components.get("tip_continuation_accepted", 0.0)
            > 0.0
        )
        if not accepted:
            break
        batch_steps_added = int(
            path.score_components.get("tip_extension_steps", 0.0)
        )
        batch_length = float(
            path.score_components.get("tip_extension_length", 0.0)
        )
        new_support = set(path.covered_indices) - before_covered
        ownership_collision = any(
            blocked[index]
            for index in new_support
            if 0 <= int(index) < len(blocked)
        )
        loop_reentry = False
        if len(before_points) >= 5 and len(path.points) > len(before_points):
            protected_prefix = before_points[:-3]
            if len(protected_prefix):
                loop_gap, _ = cKDTree(protected_prefix).query(
                    np.asarray(path.points[-1], dtype=float),
                    k=1,
                    workers=worker_threads(),
                )
                loop_reentry = bool(
                    float(loop_gap) <= max(4.0 * float(d_bar), 0.006)
                )
        if ownership_collision or loop_reentry:
            path.points = before_points
            path.covered_indices = before_covered
            path.node_indices = before_node_indices
            path.score_components["tip_continuation_accepted"] = 0.0
            if ownership_collision:
                ownership_rejections += 1
            if loop_reentry:
                loop_rejections += 1
            break

        accepted_batches += 1
        total_steps += batch_steps_added
        total_length += batch_length
        total_new_support += len(new_support)
        if new_support:
            blocked[np.asarray(sorted(new_support), dtype=int)] = True
        hit_batch_cap = batch_steps_added >= this_batch_limit
        if not hit_batch_cap:
            stopped_on_open_support = True
            break

    path.score_components.update(
        {
            "tip_continuation_accepted": float(accepted_batches > 0),
            "tip_extension_steps": float(total_steps),
            "tip_extension_length": float(total_length),
            "tip_continuation_new_support": float(total_new_support),
            "tip_extension_batches": float(accepted_batches),
            "tip_extension_batch_steps": float(per_batch),
            "tip_extension_loop_rejections": float(loop_rejections),
            "tip_extension_ownership_rejections": float(
                ownership_rejections
            ),
            "tip_extension_hit_limit": float(
                total_steps >= total_step_limit
                and not stopped_on_open_support
            ),
        }
    )
    if path.score_components["tip_extension_hit_limit"] > 0.0:
        if "tip_extension_limit" not in path.qc_flags:
            path.qc_flags.append("tip_extension_limit")
    elif "tip_extension_limit" in path.qc_flags:
        path.qc_flags.remove("tip_extension_limit")
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
    valid_distance = distances >= 0.375 * float(target_step)
    if not np.any(valid_distance):
        valid_distance = distances >= 0.30 * float(target_step)
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

    deferred = [
        path for path in members if int(path.fork_hypothesis_index) > 0
    ]
    members = [
        path for path in members if int(path.fork_hypothesis_index) == 0
    ]
    if not members:
        return []
    if len(members) <= 1:
        representatives = list(members)
        _attach_deferred_fork_hypotheses(representatives, deferred)
        return representatives
    mode_clusters = _endpoint_mode_clusters(members)
    supported = [cluster for cluster in mode_clusters if len(cluster) >= 2]
    if not supported:
        supported = [list(range(len(members)))]
        rejected_count = 0
    else:
        rejected_count = len(members) - sum(len(cluster) for cluster in supported)

    representatives: list[RootPath] = []
    for cluster in supported:
        cluster_members = [members[index] for index in cluster]
        representative = _endpoint_consensus_variant(cluster_members)
        representative.score_components["variant_endpoint_mode_support"] = float(
            len(cluster)
        )
        representative.score_components["variant_endpoint_outliers_rejected"] = float(rejected_count)
        representatives.append(representative)
    representatives.sort(
        key=lambda path: (path.score, path.length, len(path.covered_indices), str(path.root_id)),
        reverse=True,
    )
    for mode_index, representative in enumerate(representatives):
        representative.score_components["variant_endpoint_mode_count"] = float(len(representatives))
        representative.score_components["variant_endpoint_mode_index"] = float(mode_index)
    _attach_deferred_fork_hypotheses(representatives, deferred)
    return representatives


def _attach_deferred_fork_hypotheses(
    representatives: list[RootPath],
    deferred: list[RootPath],
) -> None:
    """Keep alternate geometry for later-order tracing and topology review.

    A forced fork arm is mutually exclusive with the ordinary continuation of
    the same parameter trace. It therefore cannot become an additional root at
    the current order. Repeatable alternate endpoint modes are retained on the
    selected continuation as evidence; unassigned surface remains available to
    the next-order pass, where the arm can become a child and enter the stable
    fork-resurvey queue.
    """

    if not representatives or not deferred:
        return
    clusters = _endpoint_mode_clusters(deferred)
    supported = [cluster for cluster in clusters if len(cluster) >= 2]
    retained: list[RootPath] = []
    for cluster in supported[:2]:
        retained.append(
            _endpoint_consensus_variant(
                [deferred[index] for index in cluster]
            )
        )
    if not retained:
        return
    records = [
        {
            "root_id": str(path.root_id),
            "common_prefix_nodes": int(path.fork_common_prefix_nodes),
            "points_normalized": np.asarray(path.points, dtype=float).copy(),
            "score_components": dict(path.score_components),
            "reason": "mutually_exclusive_continuation_deferred_to_topology",
        }
        for path in retained
    ]
    for representative in representatives:
        representative.deferred_fork_hypotheses = list(records)
        representative.score_components[
            "deferred_fork_hypothesis_modes"
        ] = float(len(records))


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
    point_tree: cKDTree | None = None,
) -> list[RootPath]:
    if not candidates:
        return []
    tree = point_tree if point_tree is not None else cKDTree(points)
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
            growth_length = float(
                path.score_components.get("trace_growth_arc", path.length)
            )
            longest_path_reward = (
                0.35 * growth_length / max(float(d_bar), 1e-12)
            )
            value = (
                novel
                - overlap_penalty * overlap
                + 10.0 * growth_length
                + longest_path_reward
            )
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


def backtrace_to_primary(
    paths: list[RootPath],
    primary_path: np.ndarray,
    primary_points: np.ndarray | None = None,
    *,
    target_tree: cKDTree | None = None,
) -> list[RootPath]:
    target = primary_points if primary_points is not None and len(primary_points) else primary_path
    tree = target_tree if target_tree is not None else cKDTree(target)
    refined: list[RootPath] = []
    for path in paths:
        if len(path.points) < 2:
            refined.append(path)
            continue
        _, idx = tree.query(path.points[1], k=1)
        junction = target[int(idx)]
        new_points = path.points.copy()
        new_points[0] = junction
        path.score_components["synthetic_attachment_length"] = float(
            np.linalg.norm(new_points[1] - junction)
        )
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
    parent_tree: cKDTree | None = None,
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
    tree = parent_tree if parent_tree is not None else cKDTree(parent)
    distances, nearest = tree.query(points, k=1, workers=worker_threads())
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


def _build_support_index(
    points: np.ndarray,
    support_mask: np.ndarray,
) -> _SupportIndex:
    """Build an index containing only points that may count as support."""

    source = np.asarray(points, dtype=float)
    mask = np.asarray(support_mask, dtype=bool)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if mask.shape != (len(source),):
        raise ValueError("support_mask must contain one value per point")
    source_indices = np.flatnonzero(mask)
    support_points = source[source_indices]
    return _SupportIndex(
        points=support_points,
        tree=cKDTree(support_points),
        source_indices=source_indices,
    )


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
    source_indices: np.ndarray | None = None,
) -> set[int]:
    """Collect path support with one batched query over all centerline nodes.

    ``source_indices`` maps a filtered tree back to indices in the source cloud.
    """

    nodes = np.asarray(path, dtype=float)
    if not len(nodes):
        return set()
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("path must have shape (n, 3)")
    neighborhoods = tree.query_ball_point(
        nodes,
        r=float(radius),
        workers=worker_threads(),
    )
    covered: set[int] = set()
    mask = None if support_mask is None else np.asarray(support_mask, dtype=bool)
    index_map = (
        None if source_indices is None else np.asarray(source_indices, dtype=int)
    )
    for nearby_raw in neighborhoods:
        nearby = np.asarray(nearby_raw, dtype=int)
        if mask is not None and len(nearby):
            nearby = nearby[mask[nearby]]
        if index_map is not None and len(nearby):
            nearby = index_map[nearby]
        covered.update(nearby.tolist())
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




