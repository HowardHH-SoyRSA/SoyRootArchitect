"""Final mesh-local label cleanup before assigned-support centerline fitting."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .surface_patches import (
    _polyline_projection_distance_and_arc,
    _segment_radius_profile,
)
from .types import RootPath


@dataclass
class _Partition:
    labels: np.ndarray
    edges: np.ndarray
    graph: object
    members: list[np.ndarray]
    owners: np.ndarray
    by_vertex: np.ndarray
    neighbors: list[dict[int, int]]
    boundary_labels: list[Counter]
    contact_vertices: dict[tuple[int, int], np.ndarray]


def cleanup_final_surface(
    points: np.ndarray,
    labels: np.ndarray,
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    *,
    d_bar: float,
    triangles: np.ndarray | None = None,
    excluded_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Clean unsupported islands and bounded holes using local mesh support.

    Decisions use frozen paths and labels within each of two phases. Assigned
    components are considered first, then holes are evaluated against those
    corrected labels. Above-collar and protected collar vertices are immutable;
    uncertain vertices are barriers. No spatial edge is invented.
    """
    source = np.asarray(points, dtype=float)
    before = np.asarray(labels, dtype=int)
    spacing = float(d_bar)
    paths = [np.asarray(primary_path, dtype=float)] + [
        np.asarray(root.points, dtype=float) for root in lateral_paths
    ]
    if source.ndim != 2 or source.shape[1] != 3 or not np.all(np.isfinite(source)):
        raise ValueError("points must contain finite XYZ coordinates")
    if before.shape != (len(source),):
        raise ValueError("labels must contain one value per point")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("d_bar must be positive and finite")
    for path in paths:
        if path.ndim != 2 or path.shape[1] != 3 or not np.all(np.isfinite(path)):
            raise ValueError("root paths must contain finite XYZ coordinates")
    for label, root in enumerate(lateral_paths, 1):
        start = int(root.body_start_index)
        if start < 0 or (len(paths[label]) and start >= len(paths[label])):
            raise ValueError("body_start_index must reference the root path")
        paths[label] = paths[label][start:]
    if np.any(before >= len(paths)):
        raise ValueError("assigned labels must reference a root path")
    excluded = (
        np.zeros(len(source), dtype=bool)
        if excluded_mask is None
        else np.asarray(excluded_mask, dtype=bool)
    )
    if excluded.shape != before.shape:
        raise ValueError("excluded_mask must contain one value per point")
    faces = (
        np.empty((0, 3), dtype=np.int64)
        if triangles is None
        else np.asarray(triangles, dtype=np.int64)
    )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("triangles must have shape (m, 3)")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(source)):
        raise ValueError("triangles contain an out-of-range vertex index")

    id_to_label = {"primary": 0}
    id_to_label.update({root.root_id: label for label, root in enumerate(lateral_paths, 1)})
    parents = np.full(len(paths), -1, dtype=int)
    for label, root in enumerate(lateral_paths, 1):
        parents[label] = int(id_to_label.get(root.parent_id, -1))
    report = {
        "policy": "final-local-surface-cleanup-v1",
        "status": "stable",
        "connectivity": "triangle_edges",
        "maximum_mesh_edge_length": 4.0 * spacing,
        "component_quality_rule": (
            "identical for every label: real mesh connectivity, at least three "
            "locally supported vertices, >=80% radius-envelope support, and "
            "longitudinal coverage >= max(2*d_bar, local radius)"
        ),
        "recipient_rule": (
            "touching root only; local supported mesh anchor plus radius-envelope "
            "fit; boundary contact and surface fit decide; local anchor size is "
            "used only for exact ties; total root size is never used"
        ),
        "parent_rule": (
            "unsupported compact lateral island, >=75% parent boundary contact, "
            "supported local parent anchor, and full parent-envelope fit"
        ),
        "hole_rule": (
            "bounded below-collar unassigned component with a clear locally "
            "supported touching recipient; uncertain/protected points are barriers"
        ),
        "tie_rule": "retain assigned owner or leave hole unassigned",
        "internal_connectors": "excluded using body_start_index at every lateral order",
        "assigned_component_count": 0,
        "unsupported_island_count": 0,
        "reassigned_island_count": 0,
        "reassigned_island_vertex_count": 0,
        "unassigned_island_count": 0,
        "unassigned_island_vertex_count": 0,
        "hole_component_count": 0,
        "filled_hole_count": 0,
        "filled_hole_vertex_count": 0,
        "changed_vertex_count": 0,
        "protected_vertex_count": int(np.count_nonzero(excluded)),
        "uncertain_vertex_count": int(np.count_nonzero(before == -2)),
        "moves": [],
        "components": [],
    }
    result = before.copy()
    if not len(faces):
        report.update(status="skipped_no_mesh", connectivity="no_mesh")
        return result, report

    edges = np.unique(
        np.sort(
            np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
            axis=1,
        ),
        axis=0,
    )
    lengths = np.linalg.norm(source[edges[:, 0]] - source[edges[:, 1]], axis=1)
    edges = edges[(edges[:, 0] != edges[:, 1]) & (lengths <= 4.0 * spacing)]
    edges = edges[~excluded[edges[:, 0]] & ~excluded[edges[:, 1]]]

    # Phase one: classify every assigned component with the same rule and move
    # only unsupported, compact islands with a clear supported recipient.
    assigned_partition = _partition(source, result, edges, excluded, include_holes=False)
    assigned_evidence = _Evidence(source, paths, spacing, assigned_partition)
    assigned_moves: list[tuple[int, int, dict | None]] = []
    demoted = np.zeros(len(source), dtype=bool)
    for component in sorted(
        range(len(assigned_partition.members)),
        key=lambda c: int(assigned_partition.members[c][0]),
    ):
        owner = int(assigned_partition.owners[component])
        if owner < 0:
            continue
        report["assigned_component_count"] += 1
        quality = assigned_evidence.component_quality(component, owner)
        candidates = assigned_evidence.recipient_candidates(component, exclude_label=owner)
        row = _component_row(assigned_partition, component, quality, candidates)
        if not quality["coherent_exposed_body"]:
            report["unsupported_island_count"] += 1
            target = None
            reason = None
            parent = int(parents[owner]) if owner < len(parents) else -1
            parent_claim = next((item for item in candidates if item["label"] == parent), None)
            if parent_claim is not None and _admissible_island(parent_claim):
                target = parent
                reason = "unsupported_lateral_island_on_supported_parent_surface"
            elif owner == 0:
                # A primary island has no declared parent. It may still be a
                # mislabeled speck on one clear supported child surface. For
                # laterals, do not skip a failed parent gate and jump to a
                # sibling or grandparent merely because it is larger nearby.
                target = _clear_recipient(candidates, minimum_boundary_fraction=0.75)
                if target is not None:
                    reason = "unsupported_primary_island_on_supported_neighbor_surface"
            if target is None and quality["compact"]:
                target = -1
                reason = "unsupported_compact_island_left_unassigned"
            if target is not None:
                claim = (
                    next(item for item in candidates if item["label"] == target)
                    if target >= 0 else None
                )
                assigned_moves.append((component, target, claim))
                row.update(target_label=target, decision=reason)
            else:
                row.update(target_label=owner, decision="retained_no_clear_supported_recipient")
        else:
            row.update(target_label=owner, decision="retained_supported_exposed_body")
        report["components"].append(row)

    for component, target, claim in assigned_moves:
        ids = assigned_partition.members[component]
        source_label = int(assigned_partition.owners[component])
        result[ids] = target
        if target == -1:
            demoted[ids] = True
        report["moves"].append(
            {
                "phase": "assigned_island",
                "first_vertex": int(ids[0]),
                "vertex_count": int(len(ids)),
                "source_label": source_label,
                "target_label": int(target),
                "boundary_fraction": claim["boundary_fraction"] if claim is not None else 0.0,
                "surface_fit": claim["surface_fit"] if claim is not None else None,
            }
        )
    report["reassigned_island_count"] = sum(target >= 0 for _, target, _ in assigned_moves)
    report["reassigned_island_vertex_count"] = int(
        sum(len(assigned_partition.members[c]) for c, target, _ in assigned_moves if target >= 0)
    )
    report["unassigned_island_count"] = sum(target == -1 for _, target, _ in assigned_moves)
    report["unassigned_island_vertex_count"] = int(
        sum(len(assigned_partition.members[c]) for c, target, _ in assigned_moves if target == -1)
    )

    # Phase two: rebuild components after island transfers, then fill only
    # bounded holes whose touching-root evidence has one clear winner.
    hole_partition = _partition(source, result, edges, excluded, include_holes=True)
    hole_evidence = _Evidence(source, paths, spacing, hole_partition)
    hole_moves: list[tuple[int, int, dict]] = []
    for component in sorted(
        range(len(hole_partition.members)),
        key=lambda c: int(hole_partition.members[c][0]),
    ):
        if int(hole_partition.owners[component]) != -1:
            continue
        report["hole_component_count"] += 1
        candidates = hole_evidence.recipient_candidates(component, exclude_label=-1)
        row = _component_row(hole_partition, component, None, candidates)
        contains_demoted_island = bool(np.any(demoted[hole_partition.members[component]]))
        target = (
            None
            if contains_demoted_island
            else _clear_recipient(candidates, minimum_boundary_fraction=0.50)
        )
        if contains_demoted_island:
            row.update(target_label=-1, decision="retained_unassigned_unsupported_island")
        elif target is None:
            row.update(target_label=-1, decision="retained_unassigned_no_clear_supported_recipient")
        else:
            claim = next(item for item in candidates if item["label"] == target)
            hole_moves.append((component, target, claim))
            row.update(target_label=target, decision="filled_from_local_supported_neighbor")
        report["components"].append(row)
    for component, target, claim in hole_moves:
        ids = hole_partition.members[component]
        result[ids] = target
        report["moves"].append(
            {
                "phase": "unassigned_hole",
                "first_vertex": int(ids[0]),
                "vertex_count": int(len(ids)),
                "source_label": -1,
                "target_label": int(target),
                "boundary_fraction": claim["boundary_fraction"],
                "surface_fit": claim["surface_fit"],
            }
        )
    report["filled_hole_count"] = len(hole_moves)
    report["filled_hole_vertex_count"] = int(
        sum(len(hole_partition.members[c]) for c, _, _ in hole_moves)
    )
    report["changed_vertex_count"] = int(np.count_nonzero(result != before))
    if not np.array_equal(result[excluded | (before == -2)], before[excluded | (before == -2)]):
        raise AssertionError("protected and uncertain labels must remain unchanged")
    return result, report


def _admissible_island(claim: dict) -> bool:
    return bool(
        claim["boundary_fraction"] >= 0.75
        and claim["target_supported"]
        and claim["compact"]
        and claim["local_anchor"]
    )


def _clear_recipient(candidates: list[dict], *, minimum_boundary_fraction: float) -> int | None:
    eligible = [
        item
        for item in candidates
        if item["boundary_contact_edges"] > 0
        and item["target_supported"]
        and item["compact"]
        and item["local_anchor"]
    ]
    if not eligible:
        return None
    if len(eligible) == 1:
        return (
            int(eligible[0]["label"])
            if eligible[0]["boundary_fraction"] >= minimum_boundary_fraction
            else None
        )
    boundary_best = sorted(eligible, key=lambda item: (-item["boundary_fraction"], item["label"]))
    fit_best = sorted(eligible, key=lambda item: (item["surface_fit"], item["label"]))
    if boundary_best[0]["label"] != fit_best[0]["label"]:
        return None
    if boundary_best[0]["boundary_fraction"] < minimum_boundary_fraction:
        return None
    ranked = sorted(eligible, key=lambda item: (-item["claim_score"], item["label"]))
    margin = ranked[0]["claim_score"] - ranked[1]["claim_score"]
    if margin > 0.05:
        return int(ranked[0]["label"])
    if abs(margin) <= 1e-12:
        sizes = sorted(eligible, key=lambda item: (-item["local_anchor_vertex_count"], item["label"]))
        if (
            len(sizes) == 1
            or sizes[0]["local_anchor_vertex_count"] > sizes[1]["local_anchor_vertex_count"]
        ):
            return int(sizes[0]["label"])
    return None


def _component_row(
    partition: _Partition,
    component: int,
    quality: dict | None,
    candidates: list[dict],
) -> dict:
    ids = partition.members[component]
    row = {
        "first_vertex": int(ids[0]),
        "vertex_count": int(len(ids)),
        "source_label": int(partition.owners[component]),
        "boundary_labels": {
            str(label): int(count)
            for label, count in sorted(partition.boundary_labels[component].items())
        },
        "candidates": candidates,
    }
    if quality is not None:
        row["source_quality"] = quality
    return row


def _partition(
    points: np.ndarray,
    labels: np.ndarray,
    edges: np.ndarray,
    excluded: np.ndarray,
    *,
    include_holes: bool,
) -> _Partition:
    active = (~excluded) & ((labels >= 0) | (include_holes & (labels == -1)))
    same = edges[
        active[edges[:, 0]]
        & active[edges[:, 1]]
        & (labels[edges[:, 0]] == labels[edges[:, 1]])
    ]
    graph = coo_matrix(
        (np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(points), len(points)),
    ).tocsr()
    same_graph = coo_matrix(
        (np.ones(len(same)), (same[:, 0], same[:, 1])), shape=(len(points), len(points))
    ).tocsr()
    _, raw = connected_components(same_graph, directed=False)
    ids = np.flatnonzero(active)
    if not len(ids):
        return _Partition(labels, edges, graph, [], np.empty(0, int), np.full(len(points), -1, int), [], [], {})
    _, compact = np.unique(raw[ids], return_inverse=True)
    by_vertex = np.full(len(points), -1, dtype=int)
    by_vertex[ids] = compact
    grouped = ids[np.argsort(compact, kind="stable")]
    members = np.split(grouped, np.cumsum(np.bincount(compact))[:-1])
    owners = np.array([labels[ix[0]] for ix in members], dtype=int)
    neighbors: list[dict[int, int]] = [{} for _ in members]
    boundary_labels: list[Counter] = [Counter() for _ in members]
    contacts: dict[tuple[int, int], set[int]] = {}
    for a, b in edges:
        ca, cb = int(by_vertex[a]), int(by_vertex[b])
        if ca >= 0 and ca != cb:
            boundary_labels[ca][int(labels[b])] += 1
        if cb >= 0 and ca != cb:
            boundary_labels[cb][int(labels[a])] += 1
        if ca < 0 or cb < 0 or ca == cb:
            continue
        neighbors[ca][cb] = neighbors[ca].get(cb, 0) + 1
        neighbors[cb][ca] = neighbors[cb].get(ca, 0) + 1
        contacts.setdefault((ca, cb), set()).add(int(b))
        contacts.setdefault((cb, ca), set()).add(int(a))
    contact_vertices = {
        key: np.array(sorted(values), dtype=int) for key, values in contacts.items()
    }
    return _Partition(
        labels, edges, graph, members, owners, by_vertex, neighbors,
        boundary_labels, contact_vertices,
    )


class _Evidence:
    def __init__(self, points: np.ndarray, paths: list[np.ndarray], spacing: float, partition: _Partition):
        self.points = points
        self.paths = paths
        self.spacing = spacing
        self.partition = partition
        self._projections: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._support: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._clusters: dict[tuple[int, int], list[np.ndarray]] = {}
        self.profiles = self._profiles()

    def projection(self, component: int, label: int) -> tuple[np.ndarray, np.ndarray]:
        key = (component, label)
        if key not in self._projections:
            self._projections[key] = _polyline_projection_distance_and_arc(
                self.points[self.partition.members[component]], self.paths[label]
            )
        return self._projections[key]

    def _profiles(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        profiles = {}
        for label in range(len(self.paths)):
            support = []
            for component in np.flatnonzero(self.partition.owners == label):
                own, _ = self.projection(int(component), label)
                best = own.copy()
                competitors = {
                    int(other)
                    for other in self.partition.boundary_labels[int(component)]
                    if 0 <= int(other) < len(self.paths) and int(other) != label
                }
                for other in sorted(competitors):
                    best = np.minimum(best, self.projection(int(component), other)[0])
                ids = self.partition.members[int(component)]
                support.append(ids[np.isfinite(own) & (own <= best + self.spacing)])
            support_ids = np.concatenate(support) if support else np.empty(0, dtype=int)
            if len(self.paths[label]) >= 2 and len(support_ids) >= 3:
                profiles[label] = _segment_radius_profile(
                    self.points[support_ids], self.paths[label], self.spacing
                )
            else:
                profiles[label] = (np.array([0.0]), np.array([self.spacing]))
        return profiles

    def support(self, component: int, label: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (component, label)
        if key not in self._support:
            distance, arc = self.projection(component, label)
            radius = np.interp(arc, *self.profiles[label])
            supported = (
                distance <= 1.75 * radius + 1.5 * self.spacing
                if len(self.paths[label]) >= 2
                else np.zeros(len(distance), dtype=bool)
            )
            self._support[key] = (np.asarray(supported, dtype=bool), arc, radius)
        return self._support[key]

    def clusters(self, component: int, label: int) -> list[np.ndarray]:
        key = (component, label)
        if key in self._clusters:
            return self._clusters[key]
        ids = self.partition.members[component]
        supported = self.support(component, label)[0]
        good = ids[supported]
        if not len(good):
            self._clusters[key] = []
            return []
        subgraph = self.partition.graph[good][:, good]
        _, group = connected_components(subgraph, directed=False)
        clusters = [good[group == value] for value in np.unique(group)]
        self._clusters[key] = clusters
        return clusters

    def component_quality(self, component: int, label: int) -> dict:
        ids = self.partition.members[component]
        supported, arc, radius = self.support(component, label)
        fraction = float(np.mean(supported)) if len(ids) else 0.0
        best_count = 0
        best_span = 0.0
        best_alignment = 1.0
        threshold = max(2.0 * self.spacing, float(np.median(radius)) if len(radius) else self.spacing)
        axial_span = float(np.ptp(arc)) if len(arc) else 0.0
        xyz = self.points[ids]
        spatial_span = float(np.linalg.norm(np.ptp(xyz, axis=0))) if len(xyz) else 0.0
        median_radius = float(np.median(radius)) if len(radius) else self.spacing
        compact = bool(
            axial_span <= max(8.0 * self.spacing, 6.0 * median_radius)
            and spatial_span <= max(12.0 * self.spacing, 8.0 * median_radius)
        )
        index_by_vertex = {int(vertex): index for index, vertex in enumerate(ids)}
        for cluster in self.clusters(component, label):
            local = np.array([index_by_vertex[int(vertex)] for vertex in cluster], dtype=int)
            span = float(np.ptp(arc[local])) if len(local) else 0.0
            alignment = _direction_alignment(
                self.points[cluster], self.paths[label], arc[local], self.spacing
            )
            if (span, len(cluster)) > (best_span, best_count):
                best_count, best_span, best_alignment = len(cluster), span, alignment
        coherent = bool(
            len(self.paths[label]) >= 2
            and fraction >= 0.80
            and best_count >= 3
            and best_span >= threshold
        )
        return {
            "coherent_exposed_body": coherent,
            "compact": compact,
            "supported_fraction": fraction,
            "largest_supported_cluster_vertices": int(best_count),
            "longitudinal_coverage": best_span,
            "required_longitudinal_coverage": threshold,
            "axial_alignment": best_alignment,
            "median_local_radius": median_radius,
            "axial_span": axial_span,
            "spatial_span": spatial_span,
        }

    def recipient_candidates(self, component: int, *, exclude_label: int) -> list[dict]:
        boundary = self.partition.boundary_labels[component]
        total = int(sum(boundary.values()))
        labels = sorted(
            {
                int(self.partition.owners[neighbor])
                for neighbor in self.partition.neighbors[component]
                if self.partition.owners[neighbor] >= 0
                and self.partition.owners[neighbor] != exclude_label
            }
        )
        rows = []
        for label in labels:
            supported, arc, radius = self.support(component, label)
            distance = self.projection(component, label)[0]
            boundary_edges = int(boundary.get(label, 0))
            boundary_fraction = boundary_edges / max(total, 1)
            target_supported = bool(len(supported) and np.all(supported))
            median_radius = float(np.median(radius)) if len(radius) else self.spacing
            axial_span = float(np.ptp(arc)) if len(arc) else 0.0
            xyz = self.points[self.partition.members[component]]
            spatial_span = float(np.linalg.norm(np.ptp(xyz, axis=0))) if len(xyz) else 0.0
            compact = bool(
                axial_span <= max(8.0 * self.spacing, 6.0 * median_radius)
                and spatial_span <= max(12.0 * self.spacing, 8.0 * median_radius)
            )
            anchor, anchor_count = self._local_anchor(component, label, arc, radius)
            normalized = distance / (radius + self.spacing)
            residual = np.abs(distance - radius) / (radius + self.spacing)
            surface_fit = float(np.mean(normalized + 0.25 * residual))
            claim_score = float(boundary_fraction - 0.25 * surface_fit)
            rows.append(
                {
                    "label": label,
                    "boundary_contact_edges": boundary_edges,
                    "boundary_fraction": boundary_fraction,
                    "target_supported": target_supported,
                    "compact": compact,
                    "axial_span": axial_span,
                    "spatial_span": spatial_span,
                    "median_local_radius": median_radius,
                    "surface_fit": surface_fit,
                    "local_anchor": anchor,
                    "local_anchor_vertex_count": int(anchor_count),
                    "claim_score": claim_score,
                }
            )
        return rows

    def _local_anchor(
        self, component: int, label: int, target_arc: np.ndarray, target_radius: np.ndarray
    ) -> tuple[bool, int]:
        if len(self.paths[label]) < 2 or not len(target_arc):
            return False, 0
        center = float(np.median(target_arc))
        local_radius = float(np.median(target_radius)) if len(target_radius) else self.spacing
        window = max(8.0 * self.spacing, 4.0 * local_radius)
        required = max(2.0 * self.spacing, local_radius)
        best = 0
        for neighbor in self.partition.neighbors[component]:
            if int(self.partition.owners[neighbor]) != label:
                continue
            contact = self.partition.contact_vertices.get((component, neighbor), np.empty(0, int))
            if not len(contact):
                continue
            ids = self.partition.members[neighbor]
            supported, arc, _ = self.support(neighbor, label)
            index_by_vertex = {int(vertex): index for index, vertex in enumerate(ids)}
            supported_contacts = {
                int(vertex)
                for vertex in contact
                if int(vertex) in index_by_vertex and supported[index_by_vertex[int(vertex)]]
            }
            if not supported_contacts:
                continue
            for cluster in self.clusters(neighbor, label):
                if not any(int(vertex) in supported_contacts for vertex in cluster):
                    continue
                local_indices = np.array(
                    [index_by_vertex[int(vertex)] for vertex in cluster], dtype=int
                )
                within = np.abs(arc[local_indices] - center) <= window
                local = local_indices[within]
                if len(local) < 3:
                    continue
                span = float(np.ptp(arc[local]))
                if span >= required:
                    best = max(best, int(len(local)))
        return best >= 3, best


def _direction_alignment(
    points: np.ndarray, path: np.ndarray, projected_arc: np.ndarray, spacing: float
) -> float:
    if len(points) < 3 or len(path) < 2:
        return 1.0
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / len(points)
    values, vectors = np.linalg.eigh(covariance)
    if values[-1] <= 2.0 * max(values[-2], spacing ** 2):
        return 1.0
    path_arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    ends = np.array(
        [
            [np.interp(value, path_arc, path[:, axis]) for axis in range(3)]
            for value in [float(projected_arc.min()), float(projected_arc.max())]
        ]
    )
    direction = ends[1] - ends[0]
    norm = float(np.linalg.norm(direction))
    if norm <= 2.0 * spacing:
        segment = int(
            np.clip(
                np.searchsorted(path_arc, np.median(projected_arc), side="right") - 1,
                0,
                len(path) - 2,
            )
        )
        direction = path[segment + 1] - path[segment]
        norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return 1.0
    return float(abs(np.dot(vectors[:, -1], direction / norm)))
