from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geometry import tangent_vectors, vector_angle_degrees
from .lateral import estimate_parent_radius_profile
from .types import Normalization, RootPath, TopologyReport


PRIMARY_ID = "primary"


@dataclass
class _ContactedSiblingCrop:
    host_id: str
    host_index: int
    original_points: np.ndarray
    original_node_indices: np.ndarray | None
    original_score_components: dict[str, float]
    original_qc_flags: list[str]


def uncross_internal_primary_sibling_contacts(
    surface_points: np.ndarray,
    root_labels: np.ndarray,
    lateral_paths: list[RootPath],
    *,
    d_bar: float,
) -> tuple[set[str], list[dict[str, object]]]:
    """Swap distal arms when owned surface radii resolve an internal O1 contact.

    Greedy centreline growth can jump between two physically touching laterals.
    The resulting paths then contain complementary basal and distal arms.  This
    pass keeps the primary-owned basal identity fixed and swaps only the distal
    suffixes when three independent safeguards agree:

    * both paths are surface-seeded, primary-attached order-1 roots;
    * the contact is compact, internal to both paths, and has four substantial
      arms; and
    * radii estimated from points already owned by each path consistently favour
      the crossed pairing by a decisive margin.

    ``root_labels`` uses one-based positions into ``lateral_paths``.  Non-positive
    labels are ignored.  Ambiguous eligible contacts are left unchanged and
    receive a QC flag so they can be reviewed instead of silently rewired.
    """

    points = np.asarray(surface_points, dtype=float)
    labels = np.asarray(root_labels)
    spacing = float(d_bar)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("surface_points must contain XYZ points")
    if labels.ndim != 1 or len(labels) != len(points):
        raise ValueError("root_labels must contain one label per surface point")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("d_bar must be a positive finite distance")

    eligible = [
        (position, path)
        for position, path in enumerate(lateral_paths)
        if (
            path.parent_id == PRIMARY_ID
            and int(path.order) == 1
            and len(path.points) >= 5
            and path.score_components.get("surface_aware_seed", 0.0) > 0.0
            and path.score_components.get(
                "primary_surface_attachment",
                0.0,
            )
            > 0.0
        )
    ]
    pair_candidates: list[
        tuple[
            float,
            str,
            str,
            int,
            RootPath,
            int,
            RootPath,
            dict[str, object],
        ]
    ] = []
    for left_offset, (left_position, left) in enumerate(eligible):
        for right_position, right in eligible[left_offset + 1 :]:
            insertion_distance = _primary_insertion_distance(left, right)
            if insertion_distance <= 8.0 * spacing:
                continue
            contact = _compact_internal_sibling_contact(
                left.points,
                right.points,
                d_bar=spacing,
            )
            if contact is None:
                continue
            contact["insertion_distance"] = float(insertion_distance)
            pair_candidates.append(
                (
                    float(contact["gap"]),
                    str(left.root_id),
                    str(right.root_id),
                    left_position,
                    left,
                    right_position,
                    right,
                    contact,
                )
            )

    changed: set[str] = set()
    diagnostics: list[dict[str, object]] = []
    for (
        _,
        _,
        _,
        left_position,
        left,
        right_position,
        right,
        contact,
    ) in sorted(pair_candidates, key=lambda item: item[:3]):
        if left.root_id in changed or right.root_id in changed:
            diagnostics.append(
                _internal_contact_diagnostic(
                    left,
                    right,
                    contact,
                    action="skipped",
                    reason="root_already_uncrossed",
                )
            )
            continue

        left_support = points[labels == left_position + 1]
        right_support = points[labels == right_position + 1]
        if len(left_support) < 12 or len(right_support) < 12:
            _mark_internal_contact_ambiguous(left, right)
            diagnostics.append(
                _internal_contact_diagnostic(
                    left,
                    right,
                    contact,
                    action="ambiguous",
                    reason="insufficient_owned_surface_support",
                    left_support_count=int(len(left_support)),
                    right_support_count=int(len(right_support)),
                )
            )
            continue

        left_profile = estimate_parent_radius_profile(
            left.points,
            left_support,
            d_bar=spacing,
        )
        right_profile = estimate_parent_radius_profile(
            right.points,
            right_support,
            d_bar=spacing,
        )
        left_index = int(contact["left_index"])
        right_index = int(contact["right_index"])
        similarities = _multiscale_contact_radius_similarities(
            left.points,
            left_profile,
            left_index,
            right.points,
            right_profile,
            right_index,
            d_bar=spacing,
        )
        current_scores = [item[0] for item in similarities]
        swapped_scores = [item[1] for item in similarities]
        margins = [swapped - current for current, swapped in similarities]
        swapped_wins = int(sum(margin > 0.0 for margin in margins))
        mean_margin = float(np.mean(margins)) if margins else float("-inf")
        tangent_turns = _crossed_contact_tangent_turns(
            left.points,
            left_index,
            right.points,
            right_index,
            d_bar=spacing,
        )
        tangent_veto = any(
            not np.isfinite(turn) or float(turn) > 135.0
            for turn in tangent_turns
        )
        evidence = {
            "left_support_count": int(len(left_support)),
            "right_support_count": int(len(right_support)),
            "radius_scales_d_bar": [6.0, 10.0, 20.0],
            "current_radius_similarities": [
                float(value) for value in current_scores
            ],
            "swapped_radius_similarities": [
                float(value) for value in swapped_scores
            ],
            "radius_margins": [float(value) for value in margins],
            "swapped_scale_wins": swapped_wins,
            "mean_radius_margin": mean_margin,
            "crossed_tangent_turn_degrees": [
                float(value) for value in tangent_turns
            ],
        }
        if swapped_wins < 2 or mean_margin < 0.08:
            _mark_internal_contact_ambiguous(left, right)
            diagnostics.append(
                _internal_contact_diagnostic(
                    left,
                    right,
                    contact,
                    action="ambiguous",
                    reason="radius_pairing_margin",
                    **evidence,
                )
            )
            continue
        if tangent_veto:
            _mark_internal_contact_ambiguous(left, right)
            diagnostics.append(
                _internal_contact_diagnostic(
                    left,
                    right,
                    contact,
                    action="ambiguous",
                    reason="crossed_tangent_veto",
                    **evidence,
                )
            )
            continue

        _swap_internal_contact_suffixes(
            left,
            left_index,
            right,
            right_index,
        )
        for path in (left, right):
            path.score_components.update(
                {
                    "internal_o1_contact_uncrossed": 1.0,
                    "internal_o1_contact_gap": float(contact["gap"]),
                    "internal_o1_contact_radius_margin": mean_margin,
                    "internal_o1_contact_radius_scale_wins": float(
                        swapped_wins
                    ),
                }
            )
            if "internal_o1_contact_uncrossed" not in path.qc_flags:
                path.qc_flags.append("internal_o1_contact_uncrossed")
        child_changes = _reattach_uncrossed_direct_children(
            lateral_paths,
            left,
            right,
        )
        changed.update((str(left.root_id), str(right.root_id)))
        diagnostics.append(
            _internal_contact_diagnostic(
                left,
                right,
                contact,
                action="uncrossed",
                reason="owned_surface_radius_pairing",
                child_reattachments=child_changes,
                **evidence,
            )
        )

    return changed, diagnostics


def _primary_insertion_distance(left: RootPath, right: RootPath) -> float:
    left_point = (
        np.asarray(left.insertion_point, dtype=float)
        if left.insertion_point is not None
        else np.asarray(left.points[0], dtype=float)
    )
    right_point = (
        np.asarray(right.insertion_point, dtype=float)
        if right.insertion_point is not None
        else np.asarray(right.points[0], dtype=float)
    )
    return float(np.linalg.norm(left_point - right_point))


def _path_arc(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    return np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1))]
    )


def _near_run_arc_span(near: np.ndarray, arc: np.ndarray, index: int) -> float:
    start = int(index)
    end = int(index)
    while start > 0 and bool(near[start - 1]):
        start -= 1
    while end + 1 < len(near) and bool(near[end + 1]):
        end += 1
    return float(arc[end] - arc[start])


def _compact_internal_sibling_contact(
    left_points: np.ndarray,
    right_points: np.ndarray,
    *,
    d_bar: float,
) -> dict[str, object] | None:
    left = np.asarray(left_points, dtype=float)
    right = np.asarray(right_points, dtype=float)
    left_arc = _path_arc(left)
    right_arc = _path_arc(right)
    left_total = float(left_arc[-1])
    right_total = float(right_arc[-1])
    left_arm_minimum = max(8.0 * d_bar, 0.12 * left_total)
    right_arm_minimum = max(8.0 * d_bar, 0.12 * right_total)
    left_internal = np.flatnonzero(
        (left_arc >= left_arm_minimum)
        & (left_total - left_arc >= left_arm_minimum)
    )
    right_internal = np.flatnonzero(
        (right_arc >= right_arm_minimum)
        & (right_total - right_arc >= right_arm_minimum)
    )
    if not len(left_internal) or not len(right_internal):
        return None
    distances, nearest = cKDTree(right[right_internal]).query(
        left[left_internal],
        k=1,
    )
    best_position = int(np.argmin(distances))
    gap = float(distances[best_position])
    if gap > 1.25 * d_bar:
        return None
    left_index = int(left_internal[best_position])
    right_index = int(right_internal[int(nearest[best_position])])

    contact_tolerance = 1.25 * d_bar
    left_distances, _ = cKDTree(right).query(left, k=1)
    right_distances, _ = cKDTree(left).query(right, k=1)
    left_near = np.asarray(left_distances, dtype=float) <= contact_tolerance
    right_near = np.asarray(right_distances, dtype=float) <= contact_tolerance
    compact_limit = max(8.0 * d_bar, 0.15 * min(left_total, right_total))
    if (
        _near_run_arc_span(left_near, left_arc, left_index) > compact_limit
        or _near_run_arc_span(right_near, right_arc, right_index)
        > compact_limit
    ):
        return None
    tip_gap = float(np.linalg.norm(left[-1] - right[-1]))
    if tip_gap <= 4.0 * d_bar:
        return None
    return {
        "gap": gap,
        "left_index": left_index,
        "right_index": right_index,
        "left_arc_fraction": float(left_arc[left_index] / left_total),
        "right_arc_fraction": float(right_arc[right_index] / right_total),
        "left_contact_span": _near_run_arc_span(
            left_near,
            left_arc,
            left_index,
        ),
        "right_contact_span": _near_run_arc_span(
            right_near,
            right_arc,
            right_index,
        ),
        "tip_gap": tip_gap,
    }


def _arm_radius(
    profile: np.ndarray,
    arc: np.ndarray,
    contact_index: int,
    *,
    before: bool,
    extent: float,
) -> float:
    contact_arc = float(arc[contact_index])
    if before:
        mask = (arc < contact_arc) & (arc >= contact_arc - extent)
        fallback = max(0, contact_index - 1)
    else:
        mask = (arc > contact_arc) & (arc <= contact_arc + extent)
        fallback = min(len(profile) - 1, contact_index + 1)
    values = np.asarray(profile, dtype=float)[mask]
    if not len(values):
        values = np.asarray([profile[fallback]], dtype=float)
    return float(np.median(values))


def _radius_similarity(left: float, right: float) -> float:
    if (
        not np.isfinite(left)
        or not np.isfinite(right)
        or left <= 0.0
        or right <= 0.0
    ):
        return 0.0
    return float(min(left, right) / max(left, right))


def _multiscale_contact_radius_similarities(
    left_points: np.ndarray,
    left_profile: np.ndarray,
    left_index: int,
    right_points: np.ndarray,
    right_profile: np.ndarray,
    right_index: int,
    *,
    d_bar: float,
) -> list[tuple[float, float]]:
    left_arc = _path_arc(left_points)
    right_arc = _path_arc(right_points)
    similarities: list[tuple[float, float]] = []
    for scale in (6.0, 10.0, 20.0):
        extent = scale * d_bar
        left_before = _arm_radius(
            left_profile,
            left_arc,
            left_index,
            before=True,
            extent=extent,
        )
        left_after = _arm_radius(
            left_profile,
            left_arc,
            left_index,
            before=False,
            extent=extent,
        )
        right_before = _arm_radius(
            right_profile,
            right_arc,
            right_index,
            before=True,
            extent=extent,
        )
        right_after = _arm_radius(
            right_profile,
            right_arc,
            right_index,
            before=False,
            extent=extent,
        )
        current = 0.5 * (
            _radius_similarity(left_before, left_after)
            + _radius_similarity(right_before, right_after)
        )
        swapped = 0.5 * (
            _radius_similarity(left_before, right_after)
            + _radius_similarity(right_before, left_after)
        )
        similarities.append((float(current), float(swapped)))
    return similarities


def _contact_arm_direction(
    points: np.ndarray,
    contact_index: int,
    *,
    incoming: bool,
    d_bar: float,
) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    arc = _path_arc(values)
    span = max(3.0 * d_bar, 0.04 * float(arc[-1]))
    if incoming:
        target_arc = max(0.0, float(arc[contact_index]) - span)
        anchor = int(np.searchsorted(arc, target_arc, side="left"))
        return values[contact_index] - values[anchor]
    target_arc = min(float(arc[-1]), float(arc[contact_index]) + span)
    anchor = int(np.searchsorted(arc, target_arc, side="left"))
    anchor = min(anchor, len(values) - 1)
    return values[anchor] - values[contact_index]


def _crossed_contact_tangent_turns(
    left_points: np.ndarray,
    left_index: int,
    right_points: np.ndarray,
    right_index: int,
    *,
    d_bar: float,
) -> tuple[float, float]:
    left_incoming = _contact_arm_direction(
        left_points,
        left_index,
        incoming=True,
        d_bar=d_bar,
    )
    left_outgoing = _contact_arm_direction(
        left_points,
        left_index,
        incoming=False,
        d_bar=d_bar,
    )
    right_incoming = _contact_arm_direction(
        right_points,
        right_index,
        incoming=True,
        d_bar=d_bar,
    )
    right_outgoing = _contact_arm_direction(
        right_points,
        right_index,
        incoming=False,
        d_bar=d_bar,
    )
    return (
        vector_angle_degrees(left_incoming, right_outgoing),
        vector_angle_degrees(right_incoming, left_outgoing),
    )


def _splice_suffix(
    prefix_points: np.ndarray,
    suffix_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    merged = np.vstack([prefix_points, suffix_points])
    keep = np.concatenate(
        [
            [True],
            np.linalg.norm(np.diff(merged, axis=0), axis=1) > 1e-12,
        ]
    )
    return np.asarray(merged[keep], dtype=float), keep


def _swap_internal_contact_suffixes(
    left: RootPath,
    left_index: int,
    right: RootPath,
    right_index: int,
) -> None:
    left_points = np.asarray(left.points, dtype=float).copy()
    right_points = np.asarray(right.points, dtype=float).copy()
    left.points, left_keep = _splice_suffix(
        left_points[: left_index + 1],
        right_points[right_index + 1 :],
    )
    right.points, right_keep = _splice_suffix(
        right_points[: right_index + 1],
        left_points[left_index + 1 :],
    )
    left_indices = left.node_indices
    right_indices = right.node_indices
    if (
        left_indices is None
        or right_indices is None
        or len(left_indices) != len(left_points)
        or len(right_indices) != len(right_points)
    ):
        left.node_indices = None
        right.node_indices = None
        return
    left.node_indices = np.concatenate(
        [
            np.asarray(left_indices[: left_index + 1]),
            np.asarray(right_indices[right_index + 1 :]),
        ]
    )[left_keep]
    right.node_indices = np.concatenate(
        [
            np.asarray(right_indices[: right_index + 1]),
            np.asarray(left_indices[left_index + 1 :]),
        ]
    )[right_keep]


def _reattach_uncrossed_direct_children(
    lateral_paths: list[RootPath],
    left: RootPath,
    right: RootPath,
) -> list[dict[str, object]]:
    parents = (left, right)
    trees = (cKDTree(left.points), cKDTree(right.points))
    changes: list[dict[str, object]] = []
    for child in lateral_paths:
        if child is left or child is right:
            continue
        if child.parent_id not in {left.root_id, right.root_id}:
            continue
        evidence = (
            np.asarray(child.raw_start_point, dtype=float)
            if child.raw_start_point is not None
            else np.asarray(child.points[0], dtype=float)
        )
        proposals = []
        for parent, tree in zip(parents, trees, strict=True):
            gap, insertion_index = tree.query(evidence, k=1)
            proposals.append((float(gap), str(parent.root_id), parent, int(insertion_index)))
        gap, _, parent, insertion_index = min(proposals, key=lambda item: item[:2])
        previous_parent = str(child.parent_id)
        child.parent_id = str(parent.root_id)
        child.parent_points = parent.points
        child.insertion_index = insertion_index
        child.insertion_point = np.asarray(parent.points[insertion_index], dtype=float).copy()
        child.points[0] = child.insertion_point
        if previous_parent == child.parent_id:
            continue
        child.score_components["internal_o1_contact_child_reattached"] = 1.0
        if "internal_o1_contact_child_reattached" not in child.qc_flags:
            child.qc_flags.append("internal_o1_contact_child_reattached")
        changes.append(
            {
                "root_id": str(child.root_id),
                "old_parent_id": previous_parent,
                "new_parent_id": str(child.parent_id),
                "gap": gap,
                "insertion_index": insertion_index,
            }
        )
    return changes


def _mark_internal_contact_ambiguous(
    left: RootPath,
    right: RootPath,
) -> None:
    for path in (left, right):
        path.score_components["internal_o1_contact_ambiguous"] = 1.0
        if "internal_o1_contact_ambiguous" not in path.qc_flags:
            path.qc_flags.append("internal_o1_contact_ambiguous")


def _internal_contact_diagnostic(
    left: RootPath,
    right: RootPath,
    contact: dict[str, object],
    *,
    action: str,
    reason: str,
    **evidence: object,
) -> dict[str, object]:
    return {
        "left_root_id": str(left.root_id),
        "right_root_id": str(right.root_id),
        "left_index": int(contact["left_index"]),
        "right_index": int(contact["right_index"]),
        "gap": float(contact["gap"]),
        "insertion_distance": float(contact["insertion_distance"]),
        "left_arc_fraction": float(contact["left_arc_fraction"]),
        "right_arc_fraction": float(contact["right_arc_fraction"]),
        "left_contact_span": float(contact["left_contact_span"]),
        "right_contact_span": float(contact["right_contact_span"]),
        "tip_gap": float(contact["tip_gap"]),
        "action": action,
        "reason": reason,
        **evidence,
    }


def _reject_nonfinite_json(token: str):
    raise ValueError(f"Hierarchy correction contains non-finite JSON constant: {token}")


def repair_root_hierarchy(
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    *,
    d_bar: float,
    primary_surface_points: np.ndarray | None = None,
) -> tuple[list[RootPath], TopologyReport]:
    """Orient, attach, validate, and deterministically label a root tree.

    Candidate tracing is allowed to be imperfect, but downstream traits are not
    allowed to consume a cyclic or dangling hierarchy.  Parents are selected
    from already established lower-order paths using attachment distance and
    tangent continuity, then orders are recomputed recursively from the primary
    root (order 0).
    """

    report = TopologyReport()
    primary_path = np.asarray(primary_path, dtype=float)
    tolerance = max(6.0 * float(d_bar), 0.008)
    available: dict[str, RootPath] = {
        PRIMARY_ID: RootPath(
            root_id=PRIMARY_ID,
            points=primary_path,
            order=0,
            parent_id="",
            confidence=1.0,
        )
    }
    primary_surface = (
        np.asarray(primary_surface_points, dtype=float)
        if primary_surface_points is not None
        else np.empty((0, 3), dtype=float)
    )
    if primary_surface.ndim != 2 or primary_surface.shape[1] != 3:
        raise ValueError("primary_surface_points must contain XYZ points")
    primary_surface_tree = (
        cKDTree(primary_surface) if len(primary_surface) else None
    )
    provisional = sorted(
        [path for path in lateral_paths if len(path.points) >= 2],
        key=lambda item: (max(1, int(item.order)), str(item.parent_id), str(item.root_id)),
    )

    repaired: list[RootPath] = []
    old_to_current: dict[str, str] = {PRIMARY_ID: PRIMARY_ID}
    reassigned_roots: set[str] = set()
    for path in provisional:
        old_id = str(path.root_id)
        if path.raw_start_point is None:
            path.raw_start_point = np.asarray(path.points[0], dtype=float).copy()
        candidate_parents = [
            parent
            for parent in available.values()
            if parent.root_id == PRIMARY_ID or int(parent.order) < max(1, int(path.order))
        ]
        if not candidate_parents:
            candidate_parents = [available[PRIMARY_ID]]
        preferred_parent = old_to_current.get(str(path.parent_id), str(path.parent_id))
        surface_attachment = _primary_surface_attachment(
            path,
            primary_path,
            primary_surface,
            primary_surface_tree,
            d_bar=d_bar,
        )
        if surface_attachment is not None:
            (
                endpoint,
                parent_index,
                gap,
                branch_separation,
                contact_count,
            ) = surface_attachment
            parent = available[PRIMARY_ID]
            path.score_components.update(
                {
                    "primary_surface_attachment": 1.0,
                    "primary_surface_attachment_gap": float(gap),
                    "primary_surface_attachment_endpoint": float(endpoint),
                    "primary_surface_contact_node_count": float(
                        contact_count
                    ),
                }
            )
        else:
            parent, endpoint, parent_index, gap, branch_separation = _best_attachment(
                path,
                candidate_parents,
                preferred_parent=preferred_parent,
                tolerance=tolerance,
            )
        attachment_evidence = (
            np.asarray(path.raw_start_point, dtype=float).copy()
            if endpoint == 0 and path.raw_start_point is not None
            else np.asarray(path.points[-1], dtype=float).copy()
        )
        if endpoint == 1:
            path.points = path.points[::-1].copy()
            if path.node_indices is not None:
                path.node_indices = path.node_indices[::-1].copy()
            report.roots_reoriented += 1
        path.raw_start_point = attachment_evidence
        if parent.root_id != preferred_parent:
            reassigned_roots.add(old_id)
        path.parent_id = parent.root_id
        path.parent_points = parent.points
        path.insertion_index = int(parent_index)
        path.insertion_point = parent.points[int(parent_index)].copy()
        path.points[0] = path.insertion_point
        attachment_score = float(np.exp(-gap / max(tolerance, 1e-12)))
        length_score = float(1.0 - np.exp(-path.length / max(10.0 * d_bar, 1e-12)))
        support_score = float(min(1.0, len(path.covered_indices) / 30.0)) if path.covered_indices else 0.5
        trace_score = 0.5 if not np.isfinite(path.score) or path.score <= 0 else 1.0 - np.exp(-path.score / 100.0)
        path.confidence = float(
            np.clip(
                0.42 * attachment_score
                + 0.23 * branch_separation
                + 0.18 * length_score
                + 0.10 * support_score
                + 0.07 * trace_score,
                0.0,
                1.0,
            )
        )
        path.score_components.update(
            {
                "attachment": attachment_score,
                # Kept for compatibility with the v1 QC table.  For a branch
                # junction this is deliberately a separation/plausibility
                # score, not a reward for being collinear with its parent.
                "junction_tangent_continuity": branch_separation,
                "junction_branch_separation": branch_separation,
                "length_support": length_score,
                "point_support": support_score,
            }
        )
        _append_qc(path, gap=gap, tolerance=tolerance, d_bar=d_bar)
        if path.confidence < 0.55:
            report.low_confidence_roots += 1
        repaired.append(path)
        available[old_id] = path
        old_to_current[old_id] = old_id

    graph = _build_graph(repaired)
    for cycle in list(nx.simple_cycles(graph)):
        if not cycle:
            continue
        weakest_id = min(cycle, key=lambda root_id: _path_by_id(repaired, root_id).confidence)
        weakest = _path_by_id(repaired, weakest_id)
        weakest.parent_id = PRIMARY_ID
        weakest.parent_points = primary_path
        _, insertion_index = cKDTree(primary_path).query(weakest.points[0], k=1)
        weakest.insertion_index = int(insertion_index)
        weakest.insertion_point = primary_path[int(insertion_index)].copy()
        weakest.points[0] = weakest.insertion_point
        if "cycle_repaired" not in weakest.qc_flags:
            weakest.qc_flags.append("cycle_repaired")
        report.cycles_removed += 1
    reassigned_roots.update(
        _promote_base_attached_children(
            primary_path,
            repaired,
            d_bar=d_bar,
            tolerance=tolerance,
        )
    )
    repaired, divergence_reassignments = (
        _reparent_same_insertion_divergences(
            repaired,
            d_bar=d_bar,
        )
    )
    reassigned_roots.update(divergence_reassignments)
    repaired, basal_duplicate_promotions = (
        _merge_same_insertion_primary_duplicates(
            repaired,
            d_bar=d_bar,
        )
    )
    reassigned_roots.update(basal_duplicate_promotions)
    cropped_contacts = _crop_contacted_primary_sibling_suffixes(
        repaired,
        d_bar=d_bar,
    )
    repaired, contacted_promotions = (
        _join_contacted_sibling_continuations(
            repaired,
            cropped_contacts,
            d_bar=d_bar,
        )
    )
    reassigned_roots.update(contacted_promotions)
    repaired, duplicate_promotions = _merge_parallel_parent_duplicates(
        repaired,
        d_bar=d_bar,
    )
    reassigned_roots.update(duplicate_promotions)
    report.parents_reassigned = len(reassigned_roots)
    _assign_recursive_orders(repaired)
    _assign_stable_ids(repaired)
    _refresh_parent_references(primary_path, repaired)
    errors = validate_root_tree(repaired)
    report.warnings.extend(errors)
    report.disconnected_roots = sum("missing parent" in error for error in errors)
    return repaired, report


def validate_root_tree(paths: Iterable[RootPath]) -> list[str]:
    """Return invariant violations; an empty result proves a rooted tree."""

    paths = list(paths)
    by_id = {path.root_id: path for path in paths}
    errors: list[str] = []
    if len(by_id) != len(paths):
        errors.append("root IDs are not unique")
    graph = nx.DiGraph()
    graph.add_node(PRIMARY_ID)
    for path in paths:
        if path.parent_id != PRIMARY_ID and path.parent_id not in by_id:
            errors.append(f"{path.root_id}: missing parent {path.parent_id}")
            continue
        graph.add_edge(path.parent_id, path.root_id)
        expected_order = 1 if path.parent_id == PRIMARY_ID else int(by_id[path.parent_id].order) + 1
        if int(path.order) != expected_order:
            errors.append(
                f"{path.root_id}: order {path.order} does not equal parent order + 1 ({expected_order})"
            )
        if path.insertion_point is None or path.insertion_index is None:
            errors.append(f"{path.root_id}: insertion location is missing")
    if not nx.is_directed_acyclic_graph(graph):
        errors.append("root hierarchy contains a cycle")
    unreachable = set(graph.nodes) - set(nx.descendants(graph, PRIMARY_ID)) - {PRIMARY_ID}
    for root_id in sorted(unreachable):
        errors.append(f"{root_id}: root is not connected to primary")
    return errors


def hierarchy_frame(
    paths: Iterable[RootPath],
    *,
    normalization: Normalization | None = None,
    primary_path: np.ndarray | None = None,
    primary_confidence: float = 1.0,
    primary_qc_flags: Iterable[str] = (),
) -> pd.DataFrame:
    columns = [
        "root_id",
        "parent_id",
        "root_order",
        "insertion_index",
        "insertion_x",
        "insertion_y",
        "insertion_z",
        "coordinate_unit",
        "confidence",
        "qc_flags",
    ]
    rows = []
    if primary_path is not None and len(primary_path):
        insertion = np.asarray(primary_path, dtype=float)[0]
        if normalization is not None:
            insertion = normalization.inverse_points(insertion[None, :])[0]
        rows.append(
            {
                "root_id": PRIMARY_ID,
                "parent_id": "",
                "root_order": 0,
                "insertion_index": 0,
                "insertion_x": float(insertion[0]),
                "insertion_y": float(insertion[1]),
                "insertion_z": float(insertion[2]),
                "coordinate_unit": "mesh_unit" if normalization is not None else "analysis_normalized",
                "confidence": float(primary_confidence),
                "qc_flags": ";".join(primary_qc_flags),
            }
        )
    for path in paths:
        insertion = np.asarray(path.insertion_point if path.insertion_point is not None else path.points[0])
        if normalization is not None:
            insertion = normalization.inverse_points(insertion[None, :])[0]
        rows.append(
            {
                "root_id": path.root_id,
                "parent_id": path.parent_id,
                "root_order": int(path.order),
                "insertion_index": path.insertion_index,
                "insertion_x": float(insertion[0]),
                "insertion_y": float(insertion[1]),
                "insertion_z": float(insertion[2]),
                "coordinate_unit": "mesh_unit" if normalization is not None else "analysis_normalized",
                "confidence": float(path.confidence),
                "qc_flags": ";".join(path.qc_flags),
            }
        )
    return pd.DataFrame.from_records(rows, columns=columns)


def write_editable_hierarchy(
    path: str | Path,
    primary_path: np.ndarray,
    lateral_paths: Iterable[RootPath],
    *,
    primary_confidence: float = 1.0,
    primary_qc_flags: Iterable[str] = (),
) -> Path:
    """Write the complete topology contract as editable JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    roots = [
        {
            "root_id": PRIMARY_ID,
            "parent_id": None,
            "root_order": 0,
            "valid": True,
            "editable": False,
            "confidence": float(primary_confidence),
            "qc_flags": list(primary_qc_flags),
            "polyline": np.asarray(primary_path, dtype=float).tolist(),
            "geometry_fingerprint": _polyline_fingerprint(primary_path),
        }
    ]
    for root in lateral_paths:
        roots.append(
            {
                "root_id": root.root_id,
                "parent_id": root.parent_id,
                "root_order": int(root.order),
                "valid": True,
                "editable": True,
                "confidence": float(root.confidence),
                "qc_flags": list(root.qc_flags),
                "insertion_index": root.insertion_index,
                "insertion_point": None if root.insertion_point is None else np.asarray(root.insertion_point).tolist(),
                "polyline": np.asarray(root.points, dtype=float).tolist(),
                "geometry_fingerprint": _polyline_fingerprint(root.points),
            }
        )
    payload = {
        "schema": "soyrootbio.root-hierarchy/v1",
        "coordinate_space": "source_coordinates",
        "coordinate_unit": "mesh_unit",
        "instructions": "Coordinates are in source_coordinates. The primary row is immutable; override it with GUI endpoints/soil/guides. For lateral rows, edit parent_id, valid, or polyline and pass this file back as a correction file.",
        "roots": roots,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path


def apply_hierarchy_corrections(
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    correction_file: str | Path,
    *,
    normalization: Normalization | None = None,
) -> list[RootPath]:
    """Apply validated parent/order/polyline edits from an exported hierarchy."""

    payload = json.loads(
        Path(correction_file).read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json,
    )
    if not isinstance(payload, dict) or payload.get("schema") != "soyrootbio.root-hierarchy/v1":
        raise ValueError("Unsupported hierarchy correction schema; expected soyrootbio.root-hierarchy/v1")
    rows = payload.get("roots", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Hierarchy correction roots must be a list of objects")
    row_ids = [str(row.get("root_id", "")) for row in rows]
    if any(not root_id for root_id in row_ids):
        raise ValueError("Every hierarchy correction row must have a root_id")
    duplicate_ids = sorted({root_id for root_id in row_ids if row_ids.count(root_id) > 1})
    if duplicate_ids:
        raise ValueError("Hierarchy correction contains duplicate root IDs: " + ", ".join(duplicate_ids))
    coordinate_space = str(payload.get("coordinate_space", "source_coordinates"))
    if coordinate_space not in {"source_coordinates", "analysis_normalized"}:
        raise ValueError(f"Unsupported hierarchy coordinate_space: {coordinate_space}")
    primary_rows = [row for row in rows if str(row.get("root_id")) == PRIMARY_ID]
    for primary_row in primary_rows:
        if primary_row.get("valid", True) is False:
            raise ValueError("The primary root cannot be removed in a hierarchy correction; use a primary override.")
        if "polyline" in primary_row:
            edited_primary = np.asarray(primary_row["polyline"], dtype=float)
            if coordinate_space == "source_coordinates":
                if normalization is None:
                    # A full exported correction may include an unchanged
                    # primary.  Without a transform it cannot be verified.
                    edited_primary = None
                else:
                    edited_primary = normalization.transform_points(edited_primary)
            if edited_primary is not None and (
                edited_primary.shape != np.asarray(primary_path).shape
                or not np.allclose(edited_primary, primary_path, rtol=1e-7, atol=1e-9)
            ):
                raise ValueError("Primary polyline edits are not supported here; use endpoints, soil line, or guide sections.")
    corrections = {str(row.get("root_id")): row for row in rows if row.get("root_id") != PRIMARY_ID}
    current_ids = {path.root_id for path in lateral_paths}
    unknown_ids = sorted(set(corrections) - current_ids)
    if unknown_ids:
        raise ValueError("Hierarchy correction contains unknown or stale root IDs: " + ", ".join(unknown_ids))
    kept: list[RootPath] = []
    for path in lateral_paths:
        correction = corrections.get(path.root_id)
        if correction is None:
            kept.append(path)
            continue
        expected_fingerprint = correction.get("geometry_fingerprint")
        if expected_fingerprint:
            current_polyline = np.asarray(path.points, dtype=float)
            if coordinate_space == "source_coordinates" and normalization is not None:
                current_polyline = normalization.inverse_points(current_polyline)
            if _polyline_fingerprint(current_polyline) != str(expected_fingerprint):
                raise ValueError(f"Hierarchy correction is stale for {path.root_id}: geometry fingerprint changed")
        if correction.get("valid", True) is False:
            continue
        geometry_changed = False
        if "parent_id" in correction:
            corrected_parent = str(correction["parent_id"])
            geometry_changed = corrected_parent != path.parent_id
            path.parent_id = corrected_parent
        if "polyline" in correction:
            edited = np.asarray(correction["polyline"], dtype=float)
            if edited.ndim != 2 or edited.shape[1] != 3 or len(edited) < 2 or not np.all(np.isfinite(edited)):
                raise ValueError(f"Invalid corrected polyline for {path.root_id}")
            if coordinate_space == "source_coordinates":
                if normalization is None:
                    raise ValueError(
                        "A Normalization is required to import source-coordinate hierarchy polylines."
                    )
                edited = normalization.transform_points(edited)
            current_points = np.asarray(path.points, dtype=float)
            geometry_changed = geometry_changed or (
                edited.shape != current_points.shape
                or not np.allclose(
                    edited,
                    current_points,
                    rtol=1e-7,
                    atol=1e-9,
                )
            )
            path.points = edited
        if geometry_changed:
            # Automatic attachment confidence is no longer valid after a
            # human changes the parent or geometry.  Keep this explicit and
            # conservative until the edited result is reviewed.
            path.confidence = 0.0
            for flag in ("manual_correction", "attachment_confidence_invalidated", "low_confidence"):
                if flag not in path.qc_flags:
                    path.qc_flags.append(flag)
            for component in (
                "attachment",
                "junction_tangent_continuity",
                "junction_branch_separation",
            ):
                path.score_components.pop(component, None)
            path.score_components["manual_correction"] = 1.0
        kept.append(path)
    by_id = {path.root_id: path for path in kept}
    for path in kept:
        if path.parent_id != PRIMARY_ID and path.parent_id not in by_id:
            raise ValueError(f"Correction gives {path.root_id} a missing parent: {path.parent_id}")
    graph = _build_graph(kept)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Corrected hierarchy contains a cycle.")
    _assign_recursive_orders(kept)
    # IDs are immutable provenance keys.  Renumbering here would allow a
    # deleted ID to identify different geometry and would break correction
    # audit trails.  root_order is the authoritative post-edit order.
    _refresh_parent_references(np.asarray(primary_path, dtype=float), kept)
    errors = validate_root_tree(kept)
    if errors:
        raise ValueError("Invalid corrected hierarchy: " + "; ".join(errors))
    return kept


def _polyline_fingerprint(points: np.ndarray) -> str:
    rounded = np.round(np.asarray(points, dtype=np.float64), decimals=8)
    return hashlib.sha256(rounded.tobytes(order="C")).hexdigest()[:20]


def _primary_surface_attachment(
    path: RootPath,
    primary_path: np.ndarray,
    primary_surface: np.ndarray,
    surface_tree: cKDTree | None,
    *,
    d_bar: float,
) -> tuple[int, int, float, float, int] | None:
    """Return a supported endpoint contact with the primary surface.

    Only endpoint contacts are eligible.  A single exceptionally close
    endpoint is sufficient; otherwise at least two centerline nodes must
    support the contact.  The path must subsequently depart from the primary
    surface so a short fragment confined inside the collar is not promoted.
    """

    if (
        surface_tree is None
        or len(primary_surface) < 3
        or len(path.points) < 2
    ):
        return None
    spacing = float(d_bar)
    if not np.isfinite(spacing) or spacing <= 0.0:
        return None
    contact_limit = 2.5 * spacing
    strong_contact_limit = 1.5 * spacing
    departure_limit = 4.0 * spacing
    endpoint_points = np.asarray(
        [
            (
                path.raw_start_point
                if path.raw_start_point is not None
                else path.points[0]
            ),
            path.points[-1],
        ],
        dtype=float,
    )
    endpoint_gaps, endpoint_surface_indices = surface_tree.query(
        endpoint_points,
        k=1,
    )
    path_surface_gaps, _ = surface_tree.query(path.points, k=1)
    contact_count = int(
        np.count_nonzero(
            np.asarray(path_surface_gaps, dtype=float) <= contact_limit
        )
    )
    has_departure = bool(
        float(np.quantile(path_surface_gaps, 0.75)) >= departure_limit
        or float(np.max(endpoint_gaps)) >= 2.0 * departure_limit
    )
    if not has_departure:
        return None

    eligible: list[int] = []
    for endpoint in (0, 1):
        gap = float(endpoint_gaps[endpoint])
        if gap > contact_limit:
            continue
        if gap <= strong_contact_limit or contact_count >= 2:
            eligible.append(endpoint)
    if not eligible:
        return None
    endpoint = min(
        eligible,
        key=lambda index: (float(endpoint_gaps[index]), index),
    )
    surface_point = primary_surface[
        int(endpoint_surface_indices[endpoint])
    ]
    _, parent_index = cKDTree(primary_path).query(surface_point, k=1)
    parent_tangents = tangent_vectors(primary_path)
    if endpoint == 0:
        child_vector = (
            path.points[min(2, len(path.points) - 1)] - path.points[0]
        )
    else:
        child_vector = (
            path.points[max(0, len(path.points) - 3)] - path.points[-1]
        )
    angle = vector_angle_degrees(
        child_vector,
        parent_tangents[int(parent_index)],
    )
    branch_angle = min(float(angle), 180.0 - float(angle))
    branch_separation = max(
        float(
            np.sqrt(
                max(0.0, np.sin(np.radians(branch_angle)))
            )
        ),
        0.05,
    )
    return (
        int(endpoint),
        int(parent_index),
        float(endpoint_gaps[endpoint]),
        branch_separation,
        contact_count,
    )


def _best_attachment(
    path: RootPath,
    parents: list[RootPath],
    *,
    preferred_parent: str,
    tolerance: float,
) -> tuple[RootPath, int, int, float, float]:
    best = None
    for parent in parents:
        parent_tree = cKDTree(parent.points)
        parent_tangents = tangent_vectors(parent.points)
        for endpoint in (0, 1):
            endpoint_index = 0 if endpoint == 0 else -1
            gap, parent_index = parent_tree.query(path.points[endpoint_index], k=1)
            if endpoint == 0:
                child_vector = path.points[min(2, len(path.points) - 1)] - path.points[0]
            else:
                child_vector = path.points[max(0, len(path.points) - 3)] - path.points[-1]
            angle = vector_angle_degrees(child_vector, parent_tangents[int(parent_index)])
            branch_angle = min(float(angle), 180.0 - float(angle))
            # A true lateral should diverge from its parent.  sqrt(sin(theta))
            # remains permissive for acute soybean laterals while strongly
            # rejecting a parallel continuation that happens to be nearby.
            branch_separation = max(float(np.sqrt(max(0.0, np.sin(np.radians(branch_angle))))), 0.05)
            preferred_bonus = 0.45 if parent.root_id == preferred_parent else 0.0
            cost = float(gap / max(tolerance, 1e-12) + 0.25 * (1.0 - branch_separation) - preferred_bonus)
            item = (cost, parent, endpoint, int(parent_index), float(gap), float(branch_separation))
            if best is None or item[0] < best[0]:
                best = item
    assert best is not None
    return best[1], best[2], best[3], best[4], best[5]


def _promote_base_attached_children(
    primary_path: np.ndarray,
    paths: list[RootPath],
    *,
    d_bar: float,
    tolerance: float,
) -> set[str]:
    """Promote ambiguous node-zero children to the oldest plausible ancestor.

    Later tracing passes necessarily propose the lateral traced in the previous
    pass as parent.  At a shared crown junction that proposal is ambiguous: a
    genuine sibling can be just as close to the lateral's first node as to the
    primary.  Promotion is deliberately conservative in two independent ways:
    the insertion must lie inside a short basal arc of the proposed parent, and
    the unsnapped seed must lie inside the proposed ancestor's attachment
    envelope.  A junction that is clearly distal therefore retains its higher
    order even when its child points back toward the primary.
    """

    primary = RootPath(
        root_id=PRIMARY_ID,
        points=np.asarray(primary_path, dtype=float),
        order=0,
        parent_id="",
        confidence=1.0,
    )
    by_id = {path.root_id: path for path in paths}
    by_id[PRIMARY_ID] = primary
    basal_guard = max(14.0 * float(d_bar), 1.5 * float(tolerance))
    ancestor_envelope = max(14.0 * float(d_bar), 1.5 * float(tolerance))
    promoted_ids: set[str] = set()

    for path in paths:
        promotion_count = 0
        first_parent_arc: float | None = None
        final_ancestor_gap: float | None = None
        visited: set[str] = set()
        while path.parent_id != PRIMARY_ID and path.parent_id not in visited:
            visited.add(path.parent_id)
            parent = by_id.get(path.parent_id)
            if parent is None or len(parent.points) < 2:
                break
            parent_segments = np.linalg.norm(np.diff(parent.points, axis=0), axis=1)
            parent_arc = np.concatenate([[0.0], np.cumsum(parent_segments)])
            _, parent_index = cKDTree(parent.points).query(path.points[0], k=1)
            attachment_arc = float(parent_arc[int(parent_index)])
            if first_parent_arc is None:
                first_parent_arc = attachment_arc
            if attachment_arc > basal_guard:
                break

            ancestor = by_id.get(parent.parent_id)
            if ancestor is None or len(ancestor.points) < 2:
                break
            evidence = (
                np.asarray(path.raw_start_point, dtype=float)
                if path.raw_start_point is not None
                else np.asarray(path.points[0], dtype=float)
            )
            ancestor_gap, ancestor_index = cKDTree(ancestor.points).query(evidence, k=1)
            if float(ancestor_gap) > ancestor_envelope:
                break

            path.parent_id = ancestor.root_id
            path.parent_points = ancestor.points
            path.insertion_index = int(ancestor_index)
            path.insertion_point = ancestor.points[int(ancestor_index)].copy()
            path.points[0] = path.insertion_point
            promotion_count += 1
            final_ancestor_gap = float(ancestor_gap)

        if promotion_count:
            promoted_ids.add(path.root_id)
            path.score_components["shared_origin_promoted"] = 1.0
            path.score_components["shared_origin_promotion_count"] = float(promotion_count)
            path.score_components["shared_origin_parent_basal_arc"] = float(first_parent_arc or 0.0)
            path.score_components["shared_origin_ancestor_gap"] = float(final_ancestor_gap or 0.0)
            if "shared_origin_promoted" not in path.qc_flags:
                path.qc_flags.append("shared_origin_promoted")
    return promoted_ids


def _reparent_same_insertion_divergences(
    paths: list[RootPath],
    *,
    d_bar: float,
) -> tuple[list[RootPath], set[str]]:
    """Convert a same-insertion O1 fork into a parent and child.

    Neighbouring surface seed clusters can trace the same basal tube and then
    choose different arms at a real junction.  The thicker trace is retained
    as the O1 parent only when the two paths share a sustained basal prefix;
    the thinner divergent suffix is cropped to the junction and reparented.
    """

    retained = list(paths)
    spacing = float(d_bar)
    reassigned: set[str] = set()
    siblings = [
        path
        for path in retained
        if path.parent_id == PRIMARY_ID and len(path.points) >= 6
    ]
    processed: set[str] = set()
    for left_index, left in enumerate(siblings):
        if left.root_id in processed:
            continue
        for right in siblings[left_index + 1 :]:
            if right.root_id in processed:
                continue
            insertion_gap = float(
                np.linalg.norm(left.points[0] - right.points[0])
            )
            if insertion_gap > 3.0 * spacing:
                continue
            left_radius = float(
                left.score_components.get("trace_local_radius", 0.0)
            )
            right_radius = float(
                right.score_components.get("trace_local_radius", 0.0)
            )
            radius_min = min(left_radius, right_radius)
            radius_max = max(left_radius, right_radius)
            if radius_min <= 0.0 or radius_max / radius_min < 1.25:
                continue
            host, branch = (
                (left, right)
                if left_radius >= right_radius
                else (right, left)
            )
            if float(host.confidence) < float(branch.confidence) + 0.05:
                continue
            host_radius_similarity = float(
                host.score_components.get(
                    "trace_radius_similarity",
                    0.0,
                )
            )
            branch_radius_similarity = float(
                branch.score_components.get(
                    "trace_radius_similarity",
                    0.0,
                )
            )
            if host_radius_similarity + 0.02 < branch_radius_similarity:
                continue

            tolerance = max(
                8.0 * spacing,
                0.04 * min(float(host.length), float(branch.length)),
            )
            distances, nearest = cKDTree(host.points).query(
                branch.points,
                k=1,
            )
            near = np.asarray(distances, dtype=float) <= tolerance
            if not bool(near[0]) or bool(near[-1]):
                continue
            branch_segments = np.linalg.norm(
                np.diff(branch.points, axis=0),
                axis=1,
            )
            branch_arc = np.concatenate(
                [[0.0], np.cumsum(branch_segments)]
            )
            total_length = float(branch_arc[-1])
            divergence_index: int | None = None
            attachment_index: int | None = None
            for index in range(2, len(branch.points) - 2):
                if bool(near[index]):
                    continue
                prefix = near[:index]
                suffix = near[index:]
                if (
                    float(branch_arc[index])
                    < max(8.0 * spacing, 0.15 * total_length)
                    or total_length - float(branch_arc[index])
                    < max(10.0 * spacing, 0.20 * total_length)
                    or float(np.mean(prefix)) < 0.80
                    or float(np.mean(~suffix)) < 0.75
                ):
                    continue
                preceding_contact = np.flatnonzero(near[:index])
                if not len(preceding_contact):
                    continue
                divergence_index = int(index)
                attachment_index = int(
                    nearest[int(preceding_contact[-1])]
                )
                break
            if divergence_index is None or attachment_index is None:
                continue

            attachment = np.asarray(
                host.points[attachment_index],
                dtype=float,
            )
            suffix = np.asarray(
                branch.points[divergence_index:],
                dtype=float,
            )
            connector_gap = float(
                np.linalg.norm(attachment - suffix[0])
            )
            if connector_gap > max(
                12.0 * spacing,
                0.06 * float(branch.length),
            ):
                continue
            branch.points = np.vstack([attachment, suffix])
            branch.node_indices = None
            branch.parent_id = host.root_id
            branch.parent_points = host.points
            branch.insertion_index = attachment_index
            branch.insertion_point = attachment.copy()
            branch.raw_start_point = attachment.copy()
            branch.score_components.update(
                {
                    "same_insertion_divergence_reparented": 1.0,
                    "same_insertion_shared_prefix_length": float(
                        branch_arc[divergence_index]
                    ),
                    "same_insertion_parent_radius": float(
                        host.score_components.get(
                            "trace_local_radius",
                            0.0,
                        )
                    ),
                    "same_insertion_branch_radius": float(
                        branch.score_components.get(
                            "trace_local_radius",
                            0.0,
                        )
                    ),
                    "same_insertion_connector_gap": connector_gap,
                }
            )
            if "same_insertion_divergence_reparented" not in branch.qc_flags:
                branch.qc_flags.append(
                    "same_insertion_divergence_reparented"
                )
            reassigned.add(branch.root_id)
            processed.update({host.root_id, branch.root_id})
            break
    return retained, reassigned


def _merge_same_insertion_primary_duplicates(
    paths: list[RootPath],
    *,
    d_bar: float,
) -> tuple[list[RootPath], set[str]]:
    """Absorb a short O1 basal duplicate at the same primary insertion.

    Surface-aware seeding can legitimately find several clusters around one
    broad collar opening.  A short trace is merged only when it starts at the
    same insertion, points in the same direction, and remains on the longer
    sibling.  Distinct nearby laterals are therefore retained.
    """

    retained = list(paths)
    reassigned: set[str] = set()
    spacing = float(d_bar)
    changed = True
    while changed:
        changed = False
        siblings = [
            path
            for path in retained
            if path.parent_id == PRIMARY_ID and len(path.points) >= 3
        ]
        for shorter in sorted(
            siblings,
            key=lambda path: (
                float(path.length),
                str(path.root_id),
            ),
        ):
            for longer in siblings:
                if (
                    longer is shorter
                    or float(shorter.length)
                    > 0.35 * float(longer.length)
                ):
                    continue
                insertion_gap = float(
                    np.linalg.norm(
                        shorter.points[0] - longer.points[0]
                    )
                )
                if insertion_gap > 3.0 * spacing:
                    continue
                short_direction = (
                    shorter.points[
                        min(3, len(shorter.points) - 1)
                    ]
                    - shorter.points[0]
                )
                long_direction = (
                    longer.points[
                        min(3, len(longer.points) - 1)
                    ]
                    - longer.points[0]
                )
                short_norm = float(np.linalg.norm(short_direction))
                long_norm = float(np.linalg.norm(long_direction))
                if short_norm <= 1e-12 or long_norm <= 1e-12:
                    continue
                alignment = float(
                    np.dot(
                        short_direction / short_norm,
                        long_direction / long_norm,
                    )
                )
                if alignment < float(np.cos(np.radians(35.0))):
                    continue
                distances, _ = cKDTree(longer.points).query(
                    shorter.points[1:],
                    k=1,
                )
                tolerance = max(
                    12.0 * spacing,
                    0.08 * float(shorter.length),
                )
                if float(np.quantile(distances, 0.90)) > tolerance:
                    continue
                longer.covered_indices = set(
                    longer.covered_indices
                ) | set(shorter.covered_indices)
                if (
                    longer.novel_support_indices is not None
                    or shorter.novel_support_indices is not None
                ):
                    longer.novel_support_indices = set(
                        longer.novel_support_indices or ()
                    ) | set(shorter.novel_support_indices or ())
                longer.score_components[
                    "same_insertion_o1_duplicates_merged"
                ] = (
                    longer.score_components.get(
                        "same_insertion_o1_duplicates_merged",
                        0.0,
                    )
                    + 1.0
                )
                for child in retained:
                    if child.parent_id != shorter.root_id:
                        continue
                    child.parent_id = longer.root_id
                    child.parent_points = longer.points
                    reassigned.add(child.root_id)
                retained.remove(shorter)
                reassigned.add(longer.root_id)
                changed = True
                break
            if changed:
                break
    return retained, reassigned


def _crop_contacted_primary_sibling_suffixes(
    paths: list[RootPath],
    *,
    d_bar: float,
) -> dict[str, _ContactedSiblingCrop]:
    """Crop a root that enters and then follows a primary sibling.

    A physical lateral-lateral contact can make a greedy trace leave its own
    tube and travel down the contacted sibling.  The reciprocal surface arm is
    then rediscovered as a child of that sibling.  Cropping is intentionally
    limited to a long, tangent-aligned *terminal* overlap: a brief crossing or
    a normal branch junction is left untouched.
    """

    spacing = float(d_bar)
    siblings = [
        path
        for path in paths
        if path.parent_id == PRIMARY_ID and len(path.points) >= 8
    ]
    cropped: dict[str, _ContactedSiblingCrop] = {}
    for candidate in siblings:
        if (
            candidate.score_components.get(
                "surface_aware_seed",
                0.0,
            )
            <= 0.0
            and candidate.score_components.get(
                "primary_surface_attachment",
                0.0,
            )
            <= 0.0
        ):
            continue
        best: tuple[float, RootPath, int, int] | None = None
        candidate_arc = np.concatenate(
            [
                [0.0],
                np.cumsum(
                    np.linalg.norm(
                        np.diff(candidate.points, axis=0),
                        axis=1,
                    )
                ),
            ]
        )
        total_length = float(candidate_arc[-1])
        if total_length <= max(20.0 * spacing, 1e-12):
            continue
        candidate_tangents = tangent_vectors(candidate.points)
        for host in siblings:
            if (
                host is candidate
                or len(host.points) < 8
                or float(candidate.length)
                >= 0.90 * float(host.length)
            ):
                continue
            tolerance = max(
                8.0 * spacing,
                0.035
                * min(float(candidate.length), float(host.length)),
            )
            distances, nearest = cKDTree(host.points).query(
                candidate.points,
                k=1,
            )
            near = np.asarray(distances, dtype=float) <= tolerance
            if (
                float(np.mean(near)) < 0.25
                or not bool(near[-1])
                or bool(near[0])
            ):
                continue
            host_tangents = tangent_vectors(host.points)
            alignment = np.abs(
                np.sum(
                    candidate_tangents
                    * host_tangents[np.asarray(nearest, dtype=int)],
                    axis=1,
                )
            )
            for contact_index in range(2, len(candidate.points) - 3):
                suffix = slice(contact_index, len(candidate.points))
                prefix = slice(0, contact_index)
                suffix_length = total_length - float(
                    candidate_arc[contact_index]
                )
                prefix_length = float(candidate_arc[contact_index])
                if (
                    suffix_length
                    < max(12.0 * spacing, 0.20 * total_length)
                    or prefix_length < max(10.0 * spacing, 0.20 * total_length)
                    or float(np.mean(near[suffix])) < 0.85
                    or float(np.quantile(distances[suffix], 0.90))
                    > tolerance
                    or float(np.mean(near[prefix])) > 0.30
                ):
                    continue
                aligned_suffix = alignment[suffix][near[suffix]]
                if (
                    len(aligned_suffix) < 3
                    or float(np.mean(aligned_suffix))
                    < float(np.cos(np.radians(50.0)))
                ):
                    continue
                quality = (
                    float(np.mean(near[suffix]))
                    + suffix_length / max(total_length, 1e-12)
                    - float(np.quantile(distances[suffix], 0.90))
                    / max(tolerance, 1e-12)
                )
                contacting_suffix = np.flatnonzero(
                    near[contact_index:]
                )
                if not len(contacting_suffix):
                    continue
                actual_contact_index = int(
                    contact_index + contacting_suffix[0]
                )
                host_index = int(nearest[actual_contact_index])
                proposal = (
                    quality,
                    host,
                    actual_contact_index,
                    host_index,
                )
                if best is None or proposal[0] > best[0]:
                    best = proposal
                break
        if best is None:
            continue
        _, host, contact_index, host_index = best
        _, terminal_nearest = cKDTree(host.points).query(
            candidate.points[contact_index:],
            k=1,
        )
        host_direction = float(
            np.sign(
                int(terminal_nearest[-1])
                - int(terminal_nearest[0])
            )
        )
        original_length = float(candidate.length)
        original_node_count = int(len(candidate.points))
        crop_record = _ContactedSiblingCrop(
            host_id=host.root_id,
            host_index=host_index,
            original_points=np.asarray(
                candidate.points,
                dtype=float,
            ).copy(),
            original_node_indices=(
                None
                if candidate.node_indices is None
                else np.asarray(candidate.node_indices).copy()
            ),
            original_score_components=dict(
                candidate.score_components
            ),
            original_qc_flags=list(candidate.qc_flags),
        )
        candidate.points = np.asarray(
            candidate.points[: contact_index + 1],
            dtype=float,
        ).copy()
        candidate.node_indices = None
        candidate.score_components.update(
            {
                "contacted_sibling_suffix_cropped": 1.0,
                "contacted_sibling_original_length": original_length,
                "contacted_sibling_original_node_count": float(
                    original_node_count
                ),
                "contacted_sibling_crop_index": float(contact_index),
                "contacted_sibling_retained_length": float(
                    candidate.length
                ),
                "contacted_sibling_host_index": float(host_index),
                "contacted_sibling_host_direction": host_direction,
            }
        )
        if "contacted_sibling_suffix_cropped" not in candidate.qc_flags:
            candidate.qc_flags.append(
                "contacted_sibling_suffix_cropped"
            )
        cropped[candidate.root_id] = crop_record
    return cropped


def _restore_contacted_sibling_crop(
    root: RootPath,
    crop: _ContactedSiblingCrop,
) -> None:
    root.points = np.asarray(crop.original_points, dtype=float).copy()
    root.node_indices = (
        None
        if crop.original_node_indices is None
        else np.asarray(crop.original_node_indices).copy()
    )
    root.score_components = dict(crop.original_score_components)
    root.qc_flags = list(crop.original_qc_flags)


def _join_contacted_sibling_continuations(
    paths: list[RootPath],
    cropped_contacts: dict[str, _ContactedSiblingCrop],
    *,
    d_bar: float,
) -> tuple[list[RootPath], set[str]]:
    """Join a cropped surface-attached root to the nearby child arm.

    This handles a narrow contact seam without guessing at every ordinary
    fork.  Eligibility requires the preceding terminal-overlap crop and a
    child attached only a short arc away on that same sibling.
    """

    if not cropped_contacts:
        return paths, set()
    retained = list(paths)
    by_id = {path.root_id: path for path in retained}
    reassigned: set[str] = set()
    remove_ids: set[str] = set()
    pending_crop_ids = set(cropped_contacts)
    protected_host_ids = {
        crop.host_id for crop in cropped_contacts.values()
    }
    committed_crop_ids: set[str] = set()
    ambiguous_counts: dict[str, float] = {}
    spacing = float(d_bar)

    for root_id in sorted(cropped_contacts):
        crop = cropped_contacts[root_id]
        host_id = crop.host_id
        host_index = crop.host_index
        root = by_id.get(root_id)
        host = by_id.get(host_id)
        if root is None or root_id in remove_ids:
            continue
        if (
            host is None
            or host_id in remove_ids
            or host_id in pending_crop_ids
        ):
            continue
        if (
            root.score_components.get("surface_aware_seed", 0.0) <= 0.0
            and root.score_components.get(
                "primary_surface_attachment",
                0.0,
            )
            <= 0.0
        ):
            continue
        crop_index = int(
            root.score_components.get(
                "contacted_sibling_crop_index",
                -1.0,
            )
        )
        host_segments = np.linalg.norm(
            np.diff(host.points, axis=0),
            axis=1,
        )
        host_arc = np.concatenate([[0.0], np.cumsum(host_segments)])
        if not len(host_arc):
            continue
        contact_index = int(
            np.clip(host_index, 0, len(host.points) - 1)
        )
        contact_direction = float(
            root.score_components.get(
                "contacted_sibling_host_direction",
                0.0,
            )
        )
        bridge_limit = max(
            28.0 * spacing,
            0.14 * float(host.length),
        )
        candidates: list[
            tuple[
                float,
                float,
                float,
                float,
                float,
                RootPath,
                int,
            ]
        ] = []
        host_tree = cKDTree(host.points)
        root_direction = (
            root.points[-1]
            - root.points[max(0, len(root.points) - 3)]
        )
        root_radius = float(
            root.score_components.get("trace_local_radius", 0.0)
        )
        for child in retained:
            if (
                child.root_id in remove_ids
                or child.root_id in pending_crop_ids
                or child.root_id in protected_host_ids
                or child.parent_id not in {host_id, root_id}
                or child is root
                or len(child.points) < 3
                or float(child.length) < 8.0 * spacing
            ):
                continue
            if (
                child.parent_id == root_id
                and (
                    child.insertion_index is None
                    or int(child.insertion_index) <= crop_index
                )
            ):
                continue
            insertion_index = (
                child.insertion_index
                if child.parent_id == host_id
                else None
            )
            if insertion_index is None:
                _, insertion_index = host_tree.query(
                    child.points[0],
                    k=1,
                )
            insertion_index = int(
                np.clip(insertion_index, 0, len(host.points) - 1)
            )
            if (
                contact_direction != 0.0
                and (insertion_index - contact_index)
                * contact_direction
                < 0.0
            ):
                continue
            arc_gap = abs(
                float(host_arc[insertion_index])
                - float(host_arc[contact_index])
            )
            if arc_gap > bridge_limit:
                continue
            child_start_gap = float(
                np.linalg.norm(
                    child.points[0] - host.points[insertion_index]
                )
            )
            if child_start_gap > max(
                8.0 * spacing,
                0.04 * float(child.length),
            ):
                continue
            if float(child.length) < max(
                16.0 * spacing,
                1.5 * arc_gap,
            ):
                continue
            child_distances, _ = host_tree.query(
                child.points[1:],
                k=1,
            )
            departure = float(
                np.quantile(child_distances, 0.75)
            )
            if departure < max(
                6.0 * spacing,
                0.05 * float(child.length),
            ):
                continue
            child_direction = (
                child.points[min(2, len(child.points) - 1)]
                - child.points[0]
            )
            direct_turn = vector_angle_degrees(
                root_direction,
                child_direction,
            )
            if not np.isfinite(direct_turn) or direct_turn > 110.0:
                continue
            bridge_turn = 0.0
            if insertion_index != contact_index:
                bridge_direction = (
                    host.points[insertion_index]
                    - host.points[contact_index]
                )
                incoming_bridge_turn = vector_angle_degrees(
                    root_direction,
                    bridge_direction,
                )
                outgoing_bridge_turn = vector_angle_degrees(
                    bridge_direction,
                    child_direction,
                )
                if (
                    not np.isfinite(incoming_bridge_turn)
                    or not np.isfinite(outgoing_bridge_turn)
                    or incoming_bridge_turn > 110.0
                    or outgoing_bridge_turn > 110.0
                ):
                    continue
                bridge_turn = 0.5 * (
                    incoming_bridge_turn + outgoing_bridge_turn
                )
            child_radius = float(
                child.score_components.get("trace_local_radius", 0.0)
            )
            has_radius_evidence = bool(
                np.isfinite(root_radius)
                and root_radius > 1e-12
                and np.isfinite(child_radius)
                and child_radius > 1e-12
            )
            radius_similarity = 0.5
            if has_radius_evidence:
                radius_similarity = min(root_radius, child_radius) / max(
                    root_radius,
                    child_radius,
                )
                if radius_similarity < 0.45:
                    continue
            direction_similarity = max(
                0.0,
                1.0 - (direct_turn + 0.35 * bridge_turn) / 180.0,
            )
            continuity_score = (
                0.70 * direction_similarity
                + 0.30 * radius_similarity
            )
            candidates.append(
                (
                    continuity_score,
                    direct_turn,
                    radius_similarity,
                    arc_gap,
                    child_start_gap,
                    child,
                    insertion_index,
                )
            )
        if not candidates:
            continue
        ranked_candidates = sorted(
            candidates,
            key=lambda item: (
                -item[0],
                item[1],
                -item[2],
                -float(item[5].length),
                item[3],
                item[4],
                str(item[5].root_id),
            ),
        )
        if (
            len(ranked_candidates) > 1
            and ranked_candidates[0][0] - ranked_candidates[1][0]
            < 0.08
            and float(ranked_candidates[0][5].length)
            < 1.5 * float(ranked_candidates[1][5].length)
        ):
            ambiguous_count = float(len(ranked_candidates))
            ambiguous_counts[root_id] = ambiguous_count
            continue
        (
            continuity_score,
            direct_turn,
            radius_similarity,
            _,
            _,
            continuation,
            insertion_index,
        ) = ranked_candidates[0]
        if insertion_index >= contact_index:
            bridge = host.points[
                contact_index : insertion_index + 1
            ]
        else:
            bridge = host.points[
                insertion_index : contact_index + 1
            ][::-1]
        pieces = [np.asarray(root.points, dtype=float)]
        if len(bridge):
            pieces.append(np.asarray(bridge, dtype=float))
        pieces.append(np.asarray(continuation.points, dtype=float))
        merged = np.vstack(pieces)
        keep = np.concatenate(
            [
                [True],
                np.linalg.norm(
                    np.diff(merged, axis=0),
                    axis=1,
                )
                > 1e-12,
            ]
        )
        root.points = merged[keep]
        root.node_indices = None
        root.covered_indices = set(root.covered_indices) | set(
            continuation.covered_indices
        )
        if (
            root.novel_support_indices is not None
            or continuation.novel_support_indices is not None
        ):
            root.novel_support_indices = set(
                root.novel_support_indices or ()
            ) | set(continuation.novel_support_indices or ())
        root.score_components.update(
            {
                "contacted_sibling_continuation_joined": 1.0,
                "contacted_sibling_bridge_length": abs(
                    float(host_arc[insertion_index])
                    - float(host_arc[contact_index])
                ),
                "contacted_sibling_joined_child_length": float(
                    continuation.length
                ),
                "contacted_sibling_continuity_score": float(
                    continuity_score
                ),
                "contacted_sibling_direct_turn_degrees": float(
                    direct_turn
                ),
                "contacted_sibling_radius_similarity": float(
                    radius_similarity
                ),
            }
        )
        if (
            "contacted_sibling_continuation_joined"
            not in root.qc_flags
        ):
            root.qc_flags.append(
                "contacted_sibling_continuation_joined"
            )
        for descendant in retained:
            if descendant.parent_id != continuation.root_id:
                continue
            descendant.parent_id = root.root_id
            descendant.parent_points = root.points
            reassigned.add(descendant.root_id)
        remove_ids.add(continuation.root_id)
        reassigned.add(root.root_id)
        committed_crop_ids.add(root_id)

    # Cropping is provisional until its matching continuation is committed.
    # Restore every other backup in stable ID order so chained contacts cannot
    # leave a pending root truncated merely because another transaction ran
    # first.  Crop roots and crop hosts are protected from being consumed as
    # continuations above, so restoring here cannot invalidate a committed
    # splice.
    for root_id in sorted(pending_crop_ids - committed_crop_ids):
        root = by_id.get(root_id)
        if root is None:
            continue
        _restore_contacted_sibling_crop(
            root,
            cropped_contacts[root_id],
        )
        ambiguous_count = ambiguous_counts.get(root_id)
        if ambiguous_count is None:
            continue
        root.score_components[
            "contacted_sibling_continuation_ambiguous"
        ] = ambiguous_count
        if "contacted_sibling_continuation_ambiguous" not in root.qc_flags:
            root.qc_flags.append(
                "contacted_sibling_continuation_ambiguous"
            )

    return (
        [
            path
            for path in retained
            if path.root_id not in remove_ids
        ],
        reassigned,
    )


def _merge_parallel_parent_duplicates(
    paths: list[RootPath],
    *,
    d_bar: float,
) -> tuple[list[RootPath], set[str]]:
    """Absorb a child centerline that remains on its reported parent.

    This is a post-attachment safety net for contact zones.  It requires both
    sustained proximity and tangent alignment, so an orthogonal or escaping
    child is retained.  Support is returned to the parent and descendants are
    promoted one level.
    """

    retained = list(paths)
    promoted: set[str] = set()
    changed = True
    while changed:
        changed = False
        by_id = {path.root_id: path for path in retained}
        for path in list(retained):
            if path.parent_id == PRIMARY_ID:
                continue
            parent = by_id.get(path.parent_id)
            if parent is None or not _is_parallel_parent_duplicate(
                path,
                parent,
                d_bar=d_bar,
            ):
                continue
            parent.covered_indices = set(parent.covered_indices) | set(
                path.covered_indices
            )
            if (
                parent.novel_support_indices is not None
                or path.novel_support_indices is not None
            ):
                parent.novel_support_indices = set(
                    parent.novel_support_indices or ()
                ) | set(path.novel_support_indices or ())
            for child in retained:
                if child.parent_id != path.root_id:
                    continue
                child.parent_id = parent.root_id
                child.parent_points = parent.points
                promoted.add(child.root_id)
            parent.score_components["parallel_child_duplicates_merged"] = (
                parent.score_components.get(
                    "parallel_child_duplicates_merged",
                    0.0,
                )
                + 1.0
            )
            retained.remove(path)
            changed = True
            break
    return retained, promoted


def _is_parallel_parent_duplicate(
    path: RootPath,
    parent: RootPath,
    *,
    d_bar: float,
) -> bool:
    if len(path.points) < 3 or len(parent.points) < 3:
        return False
    child = np.asarray(path.points[1:], dtype=float)
    if len(child) < 2:
        return False
    distances, nearest = cKDTree(parent.points).query(child, k=1)
    spacing = float(d_bar)
    if not np.isfinite(spacing) or spacing <= 0.0:
        return False
    tolerance = max(
        12.0 * spacing,
        0.08 * min(float(path.length), float(parent.length)),
    )
    near = np.asarray(distances, dtype=float) <= tolerance
    if (
        float(np.mean(near)) < 0.80
        or float(distances[-1]) > tolerance
    ):
        return False
    child_tangents = tangent_vectors(child)
    parent_tangents = tangent_vectors(parent.points)
    alignment = np.abs(
        np.sum(
            child_tangents * parent_tangents[np.asarray(nearest, dtype=int)],
            axis=1,
        )
    )
    if float(np.mean(alignment[near])) < np.cos(np.radians(30.0)):
        return False

    # Proximity and a shallow angle alone do not identify a duplicate: a real
    # child can run alongside its parent for its complete (short) length.  A
    # merge therefore needs either shared point support or near-coincident
    # centerline geometry at the resolution of the input mesh.
    child_support = set(path.covered_indices)
    parent_support = set(parent.covered_indices)
    support_overlap_count = len(child_support & parent_support)
    strong_support_overlap = bool(
        len(child_support) >= 3
        and support_overlap_count >= 3
        and support_overlap_count / len(child_support) >= 0.65
    )

    minimum_length = min(float(path.length), float(parent.length))
    exact_tolerance = max(3.0 * spacing, 0.015 * minimum_length)
    exact_terminal_tolerance = max(
        4.0 * spacing,
        0.020 * minimum_length,
    )
    near_coincident_geometry = bool(
        float(np.quantile(distances, 0.90)) <= exact_tolerance
        and float(distances[-1]) <= exact_terminal_tolerance
        and float(np.mean(alignment)) >= np.cos(np.radians(15.0))
    )
    return strong_support_overlap or near_coincident_geometry


def _append_qc(path: RootPath, *, gap: float, tolerance: float, d_bar: float) -> None:
    if gap > tolerance and "attachment_gap" not in path.qc_flags:
        path.qc_flags.append("attachment_gap")
    if path.length < max(8.0 * d_bar, 0.008) and "short_trace" not in path.qc_flags:
        path.qc_flags.append("short_trace")
    if path.confidence < 0.55 and "low_confidence" not in path.qc_flags:
        path.qc_flags.append("low_confidence")


def _build_graph(paths: Iterable[RootPath]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(PRIMARY_ID)
    for path in paths:
        graph.add_edge(path.parent_id, path.root_id)
    return graph


def _path_by_id(paths: Iterable[RootPath], root_id: str) -> RootPath:
    for path in paths:
        if path.root_id == root_id:
            return path
    raise KeyError(root_id)


def _assign_recursive_orders(paths: list[RootPath]) -> None:
    children: dict[str, list[RootPath]] = defaultdict(list)
    for path in paths:
        children[path.parent_id].append(path)
    queue: deque[tuple[str, int]] = deque([(PRIMARY_ID, 0)])
    visited = {PRIMARY_ID}
    while queue:
        parent_id, parent_order = queue.popleft()
        for child in children.get(parent_id, []):
            child.order = parent_order + 1
            if child.root_id not in visited:
                visited.add(child.root_id)
                queue.append((child.root_id, child.order))


def _assign_stable_ids(paths: list[RootPath]) -> None:
    children: dict[str, list[RootPath]] = defaultdict(list)
    for path in paths:
        children[path.parent_id].append(path)
    counters: dict[int, int] = defaultdict(int)
    mapping: dict[str, str] = {PRIMARY_ID: PRIMARY_ID}
    queue = deque([PRIMARY_ID])
    while queue:
        parent_id = queue.popleft()
        siblings = sorted(
            children.get(parent_id, []),
            key=lambda root: (
                root.insertion_index if root.insertion_index is not None else 10**12,
                -root.length,
                root.root_id,
            ),
        )
        for root in siblings:
            old_id = root.root_id
            counters[int(root.order)] += 1
            new_id = f"root-o{int(root.order)}-{counters[int(root.order)]:03d}"
            mapping[old_id] = new_id
            root.root_id = new_id
            queue.append(old_id)
    for root in paths:
        root.parent_id = mapping.get(root.parent_id, root.parent_id)


def _refresh_parent_references(primary_path: np.ndarray, paths: list[RootPath]) -> None:
    by_id = {path.root_id: path for path in paths}
    for path in paths:
        parent_points = primary_path if path.parent_id == PRIMARY_ID else by_id[path.parent_id].points
        path.parent_points = parent_points
        _, parent_index = cKDTree(parent_points).query(path.points[0], k=1)
        path.insertion_index = int(parent_index)
        path.insertion_point = parent_points[int(parent_index)].copy()
        path.points[0] = path.insertion_point
