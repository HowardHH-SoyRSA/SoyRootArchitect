"""Audit native parent contacts and conservatively reconcile discrete patches.

The independent pre-fit body and the frozen cleanup evidence have different jobs:
the former identifies child support, while the latter can justify a parent
claim.  Neither proximity to a prior connector nor component size alone is a
reason to transfer a surface patch.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .mesh_geometry import MeshGeometryContext, OwnershipGeometryGeneration
from .identity_repair import identity_reference, _sample
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


def _parent_wall_orientation(points, faces, labels, members, parent_label, child_path, parent_path):
    """Compare actual incident face normals with independent local axes.

    Projection span alone can mistake a patch on the parent's wall for a short
    child body. A parent wall has normals transverse to the parent shaft, and
    substantially more axial to an emerging child. Areas, not tessellation
    counts, weight this independent discriminator. Ambiguous directions remain
    unresolved; this is not evidence to cut any connected child tube.
    """
    result = {"supported": False, "native_face_count": 0,
              "parent_normal_alignment": None, "child_normal_alignment": None}
    if child_path is None or parent_path is None or len(child_path) < 2 or len(parent_path) < 2:
        return result
    patch = np.zeros(len(points), bool)
    patch[members] = True
    use = patch[faces].any(axis=1) & ((labels[faces] == parent_label) | patch[faces]).all(axis=1)
    triangle = faces[use]
    if len(triangle) < 3:
        return result
    normal = np.cross(points[triangle[:, 1]] - points[triangle[:, 0]],
                      points[triangle[:, 2]] - points[triangle[:, 0]])
    weight = np.linalg.norm(normal, axis=1)
    positive = weight > 0.
    normal, weight = normal[positive], weight[positive]
    if len(weight) < 3:
        return result
    normal /= weight[:, None]
    child_arc = project(points[members], child_path)[1]
    parent_arc = project(points[members], parent_path)[1]
    child_direction = _sample(child_path, np.array([np.median(child_arc)]))[1][0]
    parent_direction = _sample(parent_path, np.array([np.median(parent_arc)]))[1][0]
    if min(np.linalg.norm(child_direction), np.linalg.norm(parent_direction)) < .5:
        return result
    def median(values):
        order = np.argsort(values)
        return float(values[order[np.searchsorted(np.cumsum(weight[order]), .5 * weight.sum())]])
    child_alignment = median(np.abs(normal @ child_direction))
    parent_alignment = median(np.abs(normal @ parent_direction))
    result.update(native_face_count=len(weight), parent_normal_alignment=parent_alignment,
                  child_normal_alignment=child_alignment,
                  supported=bool(parent_alignment <= .5 and child_alignment >= .4
                                 and child_alignment - parent_alignment >= .25))
    return result


def _patch_transfer_preserves_attachments(before, edges, members, child_label, parent_label, roots):
    """Keep observed child/descendant attachments and prohibit new primary contacts."""
    after = before.copy()
    after[members] = parent_label
    if not np.any(after == child_label):
        return False, "unresolved_would_consume_root"
    changed = np.zeros(len(before), bool)
    changed[members] = True
    incident = edges[changed[edges].any(axis=1)]
    affected_labels = set(np.unique(before[incident]).tolist()) | {child_label, parent_label}
    lookup = {"primary": 0, **{r.root_id: i for i, r in enumerate(roots, 1)}}
    for label, root in enumerate(roots, 1):
        if label not in affected_labels:
            continue
        parent = lookup.get(root.parent_id)
        if parent is None:
            continue
        old = ((before[edges[:, 0]] == label) & (before[edges[:, 1]] == parent)) | \
              ((before[edges[:, 1]] == label) & (before[edges[:, 0]] == parent))
        new = ((after[edges[:, 0]] == label) & (after[edges[:, 1]] == parent)) | \
              ((after[edges[:, 1]] == label) & (after[edges[:, 0]] == parent))
        if old.any() and not new.any():
            return False, "unresolved_would_remove_supported_attachment"
        if root.order >= 2:
            old_primary = ((before[edges[:, 0]] == label) & (before[edges[:, 1]] == 0)) | \
                          ((before[edges[:, 1]] == label) & (before[edges[:, 0]] == 0))
            new_primary = ((after[edges[:, 0]] == label) & (after[edges[:, 1]] == 0)) | \
                          ((after[edges[:, 1]] == label) & (after[edges[:, 0]] == 0))
            if np.any(new_primary & ~old_primary):
                return False, "unresolved_would_create_higher_order_primary_contact"
    return True, "supported_attachments_preserved"


def _guard_parent_transfer_union(before, proposed, edges, roots, transfers):
    """Withhold interacting whole patches that jointly damage native attachments.

    All checks compare against the frozen ownership. Rejecting one transfer can
    remove a contact that another transfer would create, so repeat to a fixed
    point. All proposals implicated in one violation are withheld together;
    proposal order cannot select which supported surface is sacrificed.
    """
    result = proposed.copy()
    owner = np.full(len(before), -1, int)
    for index, row in enumerate(transfers):
        owner[np.asarray(row["vertex_indices"], int)] = index
    lookup = {"primary": 0, **{str(root.root_id): i for i, root in enumerate(roots, 1)}}
    rejected = {}
    while True:
        changed = result != before
        if not np.any(changed):
            break
        incident = edges[changed[edges].any(axis=1)]
        affected = set(np.unique(before[incident]).tolist()) | set(np.unique(result[incident]).tolist())
        violations = {}

        def implicate(vertices, reason, root_id):
            indices = np.asarray(vertices, int).ravel()
            indices = indices[changed[indices]]
            for proposal in np.unique(owner[indices]):
                if proposal >= 0:
                    record = violations.setdefault(int(proposal), {"reasons": set(), "root_ids": set()})
                    record["reasons"].add(reason)
                    record["root_ids"].add(str(root_id))

        for label, root in enumerate(roots, 1):
            if label not in affected:
                continue
            if np.any(before == label) and not np.any(result == label):
                implicate(np.flatnonzero(before == label), "would_consume_root", root.root_id)
            parent = lookup.get(str(root.parent_id))
            if parent is not None:
                old = ((before[edges[:, 0]] == label) & (before[edges[:, 1]] == parent)) | \
                      ((before[edges[:, 1]] == label) & (before[edges[:, 0]] == parent))
                new = ((result[edges[:, 0]] == label) & (result[edges[:, 1]] == parent)) | \
                      ((result[edges[:, 1]] == label) & (result[edges[:, 0]] == parent))
                if old.any() and not new.any():
                    implicate(edges[old], "would_remove_supported_attachment", root.root_id)
            if root.order >= 2:
                old_primary = ((before[incident[:, 0]] == label) & (before[incident[:, 1]] == 0)) | \
                              ((before[incident[:, 1]] == label) & (before[incident[:, 0]] == 0))
                new_primary = ((result[incident[:, 0]] == label) & (result[incident[:, 1]] == 0)) | \
                              ((result[incident[:, 1]] == label) & (result[incident[:, 0]] == 0))
                if np.any(new_primary & ~old_primary):
                    implicate(incident[new_primary & ~old_primary],
                              "would_create_higher_order_primary_contact", root.root_id)
        if not violations:
            break
        for index in sorted(violations):
            row = transfers[index]
            members = np.asarray(row["vertex_indices"], int)
            result[members] = before[members]
            rejected[index] = {"root_id": row["root_id"],
                               "first_vertex": row["first_vertex"],
                               "vertex_count": row["vertex_count"],
                               "reasons": sorted(violations[index]["reasons"]),
                               "affected_root_ids": sorted(violations[index]["root_ids"])}
            row["status"] = "unresolved_atomic_parent_transfer"
            row["changed_vertex_count"] = 0
            row["atomic_transfer_guard"] = rejected[index]
            if "independent_parent_wall_transfer" in row:
                row["independent_parent_wall_transfer"] = False
    return result, [rejected[index] for index in sorted(rejected)]


def reconcile_parent_contacts(
    points: np.ndarray, labels: np.ndarray, primary: np.ndarray,
    roots: list[RootPath], *, triangles: np.ndarray | None, d_bar: float,
    cleanup_report: dict | None = None,
    excluded_mask: np.ndarray | None = None,
    mesh_context: MeshGeometryContext | None = None,
    identity_references: dict | None = None,
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

    if mesh_context is not None:
        mesh_context.validate(p, faces)
        edges = mesh_context.bounded_edges(d_bar)
    else:
        edges = _native_edges(p, faces, d_bar)
    edges = edges[~excluded[edges[:, 0]] & ~excluded[edges[:, 1]]]
    ownership = (mesh_context.ownership(before) if mesh_context is not None
                 else OwnershipGeometryGeneration(before, p))
    component = ownership.components(edges)
    references = [identity_reference(root, identity_references) for root in roots]
    primary_reference = (identity_references or {}).get("primary", {})
    primary_body = (np.asarray(primary_reference["prior_path"], float)
                    if primary_reference.get("reference_valid") else np.asarray(primary, float))
    paths = [primary_body] + [item[0] for item in references]
    by_id = {"primary": 0, **{str(root.root_id): i for i, root in enumerate(roots, 1)}}
    cleanup_rows = {(int(row["source_label"]), int(row["first_vertex"]),
                     int(row["vertex_count"])): row
                    for row in (cleanup_report or {}).get("components", [])
                    if int(row.get("source_label", -1)) >= 0}
    measurement_context = None
    verified_selected = {}
    native_normals = None
    native_body_evidence = {}
    raw_edges = (mesh_context.edges if mesh_context is not None else
                 np.unique(np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0))
    def measurement_selection(label, root):
        nonlocal measurement_context
        if label not in verified_selected:
            from .measurement_evidence import make_evidence_context, validate_measurement_evidence
            if measurement_context is None:
                measurement_context = make_evidence_context(p, faces)
            evidence = root.centerline_assessment.get("measurement_evidence")
            valid, reason = validate_measurement_evidence(evidence, p, before, label, root.points,
                triangles=faces, body_start_index=root.body_start_index, context=measurement_context)
            ids = np.asarray(evidence.get("native_vertex_indices", []), int) if valid else np.empty(0, int)
            interval = np.asarray(evidence.get("supported_interval_mask", []), bool) if valid else np.empty(0, bool)
            steps = np.linalg.norm(np.diff(root.points, axis=0), axis=1)
            measured = float(steps[interval].sum()) if len(interval) == len(steps) else 0.
            verified_selected[label] = (ids, measured, reason)
        return verified_selected[label]
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
        owned = ownership.vertices(label)
        owner_components = np.unique(component[owned])
        body, reference_radius, reference_source = references[label - 1]
        radius = max(float(reference_radius or 0.), _path_radius(root, d_bar))
        support = {}
        for part in owner_components:
            members = owned[component[owned] == part]
            support[int(part)] = _distal_body_support(
                p[members], body if body is not None else np.empty((0, 3)), radius, d_bar)
            if body is not None:
                # An exposed proximal body is still biological evidence. The
                # old distal-only test could discard it after a different
                # component became the final fitted curve.
                distance, station = project(p[members], body)
                close = distance <= 1.75 * radius + 1.5 * d_bar
                supported_span = float(np.ptp(station[close])) if np.any(close) else 0.
                observed_body = np.count_nonzero(close) >= 3 and supported_span >= max(2 * d_bar, radius)
                support[int(part)]["independent_reference_body_support"] = bool(observed_body)
                support[int(part)]["supported"] |= bool(observed_body)
        body_anchors = {int(part) for part in owner_components
                        if support[int(part)]["supported"]}
        anchored = [part for part in contact_components if int(part) in body_anchors]
        insertion = (np.asarray(root.raw_start_point, float)
                     if root.raw_start_point is not None else
                     body[0] if body is not None else np.asarray(root.points, float)[0])
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
            coverage = (_angular_coverage(p, parent, touched, insertion, parent_radius)
                        if parent is not None else None)
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
                   "body_anchor_reference_source": reference_source,
                   "candidate_parent_component_count": int(len(parent_components)),
                   "joins_supported_parent_components": joins_supported_parent,
                   "parent_claim": claim,
                   "angular_coverage_degrees": coverage,
                   "changed_vertex_count": 0}
            # The nearest-origin component is not necessarily an attachment:
            # small parent-wall islands may lie closer to that reference than
            # the connected exposed child body. Require a current, independently
            # measurable body elsewhere and native surface orientation before
            # overriding the broad proximity/body-span guard below.
            wall = {"supported": False, "reason": "not_evaluated"}
            if (clear_parent and body is not None and parent is not None
                    and len(owner_components) > 1 and len(parent_components) == 1
                    and not support[part]["distal_point_count"] and not np.any(excluded[members])):
                ids, measured, reason = measurement_selection(label, root)
                selected_components = np.unique(component[ids]) if len(ids) else np.empty(0, int)
                retained_body = (len(ids) >= 8 and len(selected_components) == 1
                                 and int(selected_components[0]) != part
                                 and measured >= max(4 * d_bar, 2 * radius))
                if retained_body and not np.intersect1d(ids, members).size:
                    wall = _parent_wall_orientation(p, faces, before, members, parent_label, body, parent)
                    if wall["supported"]:
                        from .junction_tubes import vertex_normals
                        from .surface_patch_audit import native_body_anchor_evidence
                        if native_normals is None:
                            native_normals = vertex_normals(p, faces)
                        if label not in native_body_evidence:
                            native_body_evidence[label] = native_body_anchor_evidence(
                                p, owned, component, body, d_bar, normals=native_normals)
                        native_body = native_body_evidence[label]
                        supported_parts = native_body["supported_components"]
                        wall["independent_child_sections"] = native_body
                        wall["supported"] &= (int(selected_components[0]) in supported_parts
                                              and part not in supported_parts)
                    safe, safety_reason = _patch_transfer_preserves_attachments(
                        before, raw_edges, members, label, parent_label, roots)
                    wall.update(retained_measured_body_vertex_count=len(ids),
                                retained_measured_body_length=measured,
                                selected_body_preserved=True, attachment_guard=safety_reason)
                    wall["supported"] &= safe
                    wall["reason"] = "independent_parent_wall_with_retained_child_body" if wall["supported"] else "native_orientation_or_attachment_ambiguous"
                else:
                    wall = {"supported": False, "reason": "no_distinct_verified_body_to_preserve"}
            row["independent_parent_wall_evidence"] = wall
            if body is None or parent is None:
                row["status"] = (reference_source if body is None else "unresolved_parent_identity_reference")
                report["unresolved_patch_count"] += 1
            elif wall["supported"]:
                result[members] = parent_label
                row["status"] = "reassigned_to_supported_parent"
                row["changed_vertex_count"] = int(len(members))
                row["vertex_indices"] = members.tolist()
                row["independent_parent_wall_transfer"] = True
                report["discrete_patch_count"] += 1
            elif part == attachment:
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
                    selected_ids = measurement_selection(label, root)[0]
                    if np.intersect1d(selected_ids, members).size:
                        row["status"] = "unresolved_selected_measurement_body"
                        report["unresolved_patch_count"] += 1
                    else:
                        result[members] = parent_label
                        row["status"] = "reassigned_to_supported_parent"
                        row["changed_vertex_count"] = int(len(members))
                        row["vertex_indices"] = members.tolist()
                else:
                    row["status"] = "unresolved_discrete_parent_patch"
                    report["unresolved_patch_count"] += 1
            else:
                row["status"] = "unresolved_no_distal_attachment_support"
                report["unresolved_patch_count"] += 1
            report["contacts"].append(row)
    transfers = [row for row in report["contacts"] if row["status"] == "reassigned_to_supported_parent"]
    result, rejected = _guard_parent_transfer_union(before, result, raw_edges, roots, transfers)
    report["atomic_transfer_guard"] = {"rejected_patch_count": len(rejected), "rejections": rejected}
    report["unresolved_patch_count"] += len(rejected)
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
                        "unresolved_competing_restriction", "unresolved_atomic_parent_transfer"}:
            root.qc_flags.append("parent_contact_discrete_unresolved")
        if "unresolved_connected_wrap" in statuses:
            root.qc_flags.append("parent_contact_wrap_unresolved")
