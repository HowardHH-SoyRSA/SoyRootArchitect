"""Audit native parent contacts and conservatively reconcile discrete patches.

The fitted exposed body and the frozen cleanup evidence have different jobs:
the former identifies child support, while the latter can justify a parent
claim.  Neither proximity to a prior connector nor component size alone is a
reason to transfer a surface patch.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .surface_patches import _polyline_projection_distance_and_arc as project
from .types import RootPath


def _native_edges(points: np.ndarray, triangles: np.ndarray, spacing: float) -> np.ndarray:
    edges = np.unique(np.sort(np.vstack((triangles[:, [0, 1]],
                                         triangles[:, [1, 2]],
                                         triangles[:, [2, 0]])), axis=1), axis=0)
    length = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    return edges[(edges[:, 0] != edges[:, 1]) & (length <= 4.0 * spacing)]


def _components(labels: np.ndarray, edges: np.ndarray) -> np.ndarray:
    same = edges[(labels[edges[:, 0]] == labels[edges[:, 1]])
                 & (labels[edges[:, 0]] >= 0)]
    graph = coo_matrix((np.ones(2 * len(same), dtype=np.uint8),
                        (np.r_[same[:, 0], same[:, 1]],
                         np.r_[same[:, 1], same[:, 0]])),
                       shape=(len(labels), len(labels))).tocsr()
    _, component = connected_components(graph, directed=False)
    return component


def _path_radius(root: RootPath, spacing: float) -> float:
    assessment = root.centerline_assessment
    value = assessment.get("section_radius_median")
    if value is None or not np.isfinite(value) or value <= 0:
        value = root.mean_radius
    return float(value) if value is not None and np.isfinite(value) and value > 0 else 2.0 * spacing


def _distal_body_support(points: np.ndarray, path: np.ndarray,
                         radius: float, spacing: float) -> dict:
    if len(path) < 2 or len(points) < 3:
        return {"supported": False, "distal_point_count": 0,
                "distal_span": 0.0, "path_length": 0.0}
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    distance, arc = project(points, path)
    distal = ((arc >= max(0.35 * length, 6.0 * spacing, 3.0 * radius))
              & (distance <= 1.75 * radius + 1.5 * spacing))
    count = int(np.count_nonzero(distal))
    span = float(np.ptp(arc[distal])) if count else 0.0
    return {"supported": bool(count >= 3 and span >= max(2.0 * spacing, radius)),
            "distal_point_count": count, "distal_span": span,
            "path_length": length}


def _angular_coverage(points: np.ndarray, parent: np.ndarray,
                      contact: np.ndarray, insertion: np.ndarray,
                      parent_radius: float) -> float | None:
    """Approximate coverage at the insertion within one local parent radius."""
    if len(contact) < 5 or len(parent) < 2 or parent_radius <= 0:
        return None
    tree = cKDTree(parent)
    node = tree.query(points[contact])[1]
    origin = int(tree.query(insertion)[1])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(parent, axis=0), axis=1))]
    use = np.abs(arc[node] - arc[origin]) <= parent_radius
    if np.count_nonzero(use) < 5:
        return None
    vectors = points[contact[use]] - parent[node[use]]
    tangent = np.gradient(parent, axis=0)[node[use]]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-12)
    vectors -= (vectors * tangent).sum(axis=1)[:, None] * tangent
    reference = np.tile(np.array([1.0, 0.0, 0.0]), (len(tangent), 1))
    reference -= (reference * tangent).sum(axis=1)[:, None] * tangent
    weak = np.linalg.norm(reference, axis=1) < .2
    if np.any(weak):
        reference[weak] = np.array([0.0, 1.0, 0.0])
        reference[weak] -= (reference[weak] * tangent[weak]).sum(axis=1)[:, None] * tangent[weak]
    reference /= np.maximum(np.linalg.norm(reference, axis=1)[:, None], 1e-12)
    other = np.cross(tangent, reference)
    angles = np.sort(np.arctan2((vectors * other).sum(axis=1),
                                (vectors * reference).sum(axis=1)))
    return float(np.degrees(2.0 * np.pi - np.diff(np.r_[angles, angles[0] + 2.0 * np.pi]).max()))


def reconcile_parent_contacts(
    points: np.ndarray, labels: np.ndarray, primary: np.ndarray,
    roots: list[RootPath], *, triangles: np.ndarray | None, d_bar: float,
    cleanup_report: dict | None = None,
    excluded_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Transfer only a detached, parent-supported contact patch.

    Each decision uses one frozen label snapshot. A preserved distal body, a
    clear parent claim from the preceding surface cleanup, and one existing
    parent component are required for transfer. Other discrete contacts remain
    labeled and are explicitly reported as unresolved violations.
    """
    p = np.asarray(points, float)
    before = np.asarray(labels, int)
    faces = np.empty((0, 3), int) if triangles is None else np.asarray(triangles, int)
    excluded = (np.zeros(len(p), bool) if excluded_mask is None
                else np.asarray(excluded_mask, bool))
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("points must contain finite XYZ coordinates")
    if before.shape != (len(p),) or excluded.shape != (len(p),):
        raise ValueError("labels and excluded_mask must match points")
    if faces.ndim != 2 or faces.shape[1] != 3 or (len(faces) and
            (faces.min() < 0 or faces.max() >= len(p))):
        raise ValueError("triangles must have valid vertex indices")
    if not np.isfinite(d_bar) or d_bar <= 0:
        raise ValueError("d_bar must be positive")
    result = before.copy()
    report = {"policy": "native-discrete-parent-contact-v1",
              "status": "evaluated" if len(faces) else "unresolved_no_mesh",
              "contact_definition": "native triangle edge no longer than 4*d_bar",
              "changed_vertex_count": 0, "discrete_patch_count": 0,
              "unresolved_patch_count": 0, "connected_wrap_count": 0,
              "contacts": []}
    if not len(faces):
        return result, report

    edges = _native_edges(p, faces, d_bar)
    edges = edges[~excluded[edges[:, 0]] & ~excluded[edges[:, 1]]]
    component = _components(before, edges)
    paths = [np.asarray(primary, float)] + [np.asarray(root.points, float) for root in roots]
    by_id = {"primary": 0, **{str(root.root_id): i for i, root in enumerate(roots, 1)}}
    cleanup_rows = {(int(row["source_label"]), int(row["first_vertex"]),
                     int(row["vertex_count"])): row
                    for row in (cleanup_report or {}).get("components", [])
                    if int(row.get("source_label", -1)) >= 0}
    left, right = before[edges[:, 0]], before[edges[:, 1]]
    for label, root in enumerate(roots, 1):
        parent_label = by_id.get(str(root.parent_id))
        if parent_label is None or parent_label == label:
            continue
        contact = (((left == label) & (right == parent_label)) |
                   ((right == label) & (left == parent_label)))
        touching_edges = edges[contact]
        if not len(touching_edges):
            continue
        child_vertices = np.unique(touching_edges[before[touching_edges] == label])
        contact_components = np.unique(component[child_vertices])
        owned = np.flatnonzero(before == label)
        owner_components = np.unique(component[owned])
        body = paths[label][int(root.body_start_index):]
        radius = _path_radius(root, d_bar)
        support = {}
        for part in owner_components:
            members = owned[component[owned] == part]
            support[int(part)] = _distal_body_support(p[members], body, radius, d_bar)
        body_anchors = {int(part) for part in owner_components
                        if support[int(part)]["supported"]}
        anchored = [part for part in contact_components if int(part) in body_anchors]
        insertion = (np.asarray(root.raw_start_point, float)
                     if root.raw_start_point is not None else paths[label][0])
        parent = paths[parent_label]
        # Prefer an independently supported component at the recorded origin.
        def origin_distance(part: int) -> tuple[float, int]:
            nearby = child_vertices[component[child_vertices] == part]
            return (float(np.min(np.linalg.norm(p[nearby] - insertion, axis=1))), int(part))
        attachment = min(anchored, key=origin_distance) if anchored else None
        for part in contact_components:
            part = int(part)
            members = owned[component[owned] == part]
            touched = child_vertices[component[child_vertices] == part]
            parent_contact = touching_edges[component[
                np.where(before[touching_edges[:, 0]] == label,
                         touching_edges[:, 0], touching_edges[:, 1])] == part]
            parent_vertices = np.unique(parent_contact[before[parent_contact] == parent_label])
            parent_components = np.unique(component[parent_vertices])
            prior = cleanup_rows.get((label, int(members[0]), int(len(members))))
            claim = next((item for item in prior.get("candidates", [])
                          if int(item["label"]) == parent_label), None) if prior else None
            other_claims = (prior.get("candidates", []) if prior else [])
            clear_parent = bool(
                claim is not None
                and claim.get("boundary_fraction", 0) >= .75
                and claim.get("target_supported")
                and claim.get("compact")
                and claim.get("local_anchor")
                and all(other is claim or not (
                    other.get("target_supported") and other.get("compact")
                    and other.get("local_anchor")
                    and other.get("claim_score", -np.inf) >= claim.get("claim_score", 0) - .05)
                        for other in other_claims)
            )
            radius_value = claim.get("median_local_radius") if claim else None
            parent_radius = (float(radius_value) if radius_value is not None
                             and np.isfinite(radius_value) and radius_value > 0 else 0.0)
            coverage = _angular_coverage(p, parent, touched, insertion, parent_radius)
            # A mislabeled ring can separate two otherwise supported parent
            # components. Relabeling observed triangles is allowed only when
            # the ring is almost circumferential and the parent claim is
            # exceptionally strong; no geometric edge is introduced.
            joins_supported_parent = bool(
                parent_label == 0 and len(parent_components) == 2
                and coverage is not None and coverage >= 300.0
                and claim is not None and claim.get("boundary_fraction", 0) >= .9
                and all(np.count_nonzero(component[before == parent_label] == side) >= 3
                        for side in parent_components)
            )
            row = {"root_id": str(root.root_id), "parent_id": str(root.parent_id),
                   "root_order": int(root.order), "label": label,
                   "first_vertex": int(members[0]), "vertex_count": int(len(members)),
                   "contact_edge_count": int(len(parent_contact)),
                   "contact_child_vertex_count": int(len(touched)),
                   "connected_to_distal_body": bool(support[part]["supported"]),
                   "distal_support": support[part],
                   "candidate_parent_component_count": int(len(parent_components)),
                   "joins_supported_parent_components": joins_supported_parent,
                   "parent_claim": claim,
                   "angular_coverage_degrees": coverage,
                   "changed_vertex_count": 0}
            if part == attachment:
                row["status"] = "attachment_body_contact"
                if coverage is not None and coverage >= 270.0:
                    row["status"] = "unresolved_connected_wrap"
                    report["connected_wrap_count"] += 1
                    report["unresolved_patch_count"] += 1
            elif len(owner_components) > 1 or len(contact_components) > 1:
                report["discrete_patch_count"] += 1
                if (body_anchors and part not in body_anchors and clear_parent
                        and (len(parent_components) == 1 or joins_supported_parent)
                        and not np.any(excluded[members])):
                    result[members] = parent_label
                    row["status"] = "reassigned_to_supported_parent"
                    row["changed_vertex_count"] = int(len(members))
                else:
                    row["status"] = "unresolved_discrete_parent_patch"
                    report["unresolved_patch_count"] += 1
            else:
                row["status"] = "unresolved_no_distal_attachment_support"
                report["unresolved_patch_count"] += 1
            report["contacts"].append(row)
    report["changed_vertex_count"] = int(np.count_nonzero(result != before))
    if report["unresolved_patch_count"]:
        report["status"] = "unresolved_contacts"
    elif report["changed_vertex_count"]:
        report["status"] = "corrected"
    else:
        report["status"] = "clear"
    return result, report


def mark_parent_contact_qc(roots: list[RootPath], report: dict) -> None:
    """Expose remaining discrete patches and connected wraps in root exports."""
    by_id: dict[str, set[str]] = {}
    for row in report.get("contacts", []):
        if row["status"].startswith("unresolved"):
            by_id.setdefault(row["root_id"], set()).add(row["status"])
    flags = {"parent_contact_unresolved", "parent_contact_discrete_unresolved",
             "parent_contact_wrap_unresolved"}
    for root in roots:
        root.qc_flags = [flag for flag in root.qc_flags if flag not in flags]
        statuses = by_id.get(str(root.root_id), set())
        if statuses:
            root.qc_flags.append("parent_contact_unresolved")
        if statuses & {"unresolved_discrete_parent_patch", "unresolved_iteration_limit",
                        "unresolved_competing_restriction"}:
            root.qc_flags.append("parent_contact_discrete_unresolved")
        if "unresolved_connected_wrap" in statuses:
            root.qc_flags.append("parent_contact_wrap_unresolved")
