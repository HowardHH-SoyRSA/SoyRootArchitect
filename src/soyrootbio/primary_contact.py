"""Conservatively separate higher-order lateral labels from the primary mesh."""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .mesh_geometry import MeshGeometryContext, OwnershipGeometryGeneration
from .surface_patches import _polyline_projection_distance_and_arc
from .identity_repair import identity_reference, _sample
from .types import RootPath


def audit_higher_order_primary_contacts(
    points: np.ndarray,
    labels: np.ndarray,
    roots: list[RootPath],
    *,
    triangles: np.ndarray | None,
    d_bar: float,
    mesh_context: MeshGeometryContext | None = None,
    nonroot_labels: tuple[int, ...] = (),
) -> dict:
    """Read the exact final labels against every native triangle edge.

    Long edges remain violations in this audit even though they cannot supply
    a safe surface-repair path. The function never proposes or edits labels.
    """

    source = np.asarray(points, dtype=float)
    owner = np.asarray(labels, dtype=int)
    faces = (
        np.empty((0, 3), dtype=np.int64)
        if triangles is None else np.asarray(triangles, dtype=np.int64)
    )
    if source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all():
        raise ValueError("points must contain finite XYZ coordinates")
    if owner.shape != (len(source),) or np.any(((owner < -2) & ~np.isin(owner, nonroot_labels)) | (owner > len(roots))):
        raise ValueError("labels must contain one valid owner per vertex")
    if faces.ndim != 2 or faces.shape[1] != 3 or (
        len(faces) and (faces.min() < 0 or faces.max() >= len(source))
    ):
        raise ValueError("triangles must have valid vertex indices")
    if not np.isfinite(d_bar) or d_bar <= 0:
        raise ValueError("d_bar must be positive and finite")
    if mesh_context is not None:
        mesh_context.validate(source, faces)
        geometry_sha256 = mesh_context.coordinate_sha256
    else:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(source.astype("<f8")).tobytes())
        digest.update(np.ascontiguousarray(faces.astype("<i8")).tobytes())
        geometry_sha256 = digest.hexdigest()
    label_sha256 = hashlib.sha256(
        np.ascontiguousarray(owner.astype("<i8")).tobytes()
    ).hexdigest()
    report = {
        "policy": "final-raw-native-primary-contact-audit-v1",
        "status": "unresolved_no_mesh" if not len(faces) else "clear",
        "geometry_sha256": geometry_sha256,
        "label_sha256": label_sha256,
        "contact_root_count": 0,
        "contact_edge_count": 0,
        "excluded_long_edge_contact_count": 0,
        "requires_correction": False,
        "contacts": [],
    }
    if not len(faces):
        return report
    edges = (
        mesh_context.edges if mesh_context is not None else
        np.unique(np.sort(np.vstack((
            faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
        )), axis=1), axis=0)
    )
    edges = edges[edges[:, 0] != edges[:, 1]]
    left, right = owner[edges[:, 0]], owner[edges[:, 1]]
    contact = (((left == 0) & (right > 0)) |
               ((right == 0) & (left > 0)))
    contact_edges = edges[contact]
    other_labels = np.maximum(left[contact], right[contact])
    orders = np.array([0, *(int(root.order) for root in roots)])
    higher = orders[other_labels] >= 2
    contact_edges = contact_edges[higher]
    other_labels = other_labels[higher]
    for label in np.unique(other_labels):
        touching = contact_edges[other_labels == label]
        lengths = np.linalg.norm(
            source[touching[:, 0]] - source[touching[:, 1]], axis=1
        )
        root = roots[int(label) - 1]
        row = {
            "root_id": str(root.root_id),
            "root_order": int(root.order),
            "parent_id": str(root.parent_id),
            "native_edge_vertex_pairs": touching.tolist(),
            "primary_vertex_indices": np.unique(
                touching[owner[touching] == 0]
            ).tolist(),
            "child_vertex_indices": np.unique(
                touching[owner[touching] == int(label)]
            ).tolist(),
            "contact_edge_count": int(len(touching)),
            "excluded_long_edge_contact_count": int(
                np.count_nonzero(lengths > 4.0 * d_bar)
            ),
            "status": "unresolved_native_primary_contact",
        }
        report["contacts"].append(row)
        report["contact_edge_count"] += row["contact_edge_count"]
        report["excluded_long_edge_contact_count"] += row[
            "excluded_long_edge_contact_count"
        ]
    report["contact_root_count"] = len(report["contacts"])
    if report["contact_root_count"]:
        report["status"] = "unresolved_contacts"
        report["requires_correction"] = True
    return report


def restrict_higher_order_primary_contacts(
    points: np.ndarray,
    labels: np.ndarray,
    roots: list[RootPath],
    *,
    triangles: np.ndarray | None,
    d_bar: float,
    excluded_mask: np.ndarray | None = None,
    mesh_context: MeshGeometryContext | None = None,
    identity_references: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Reassign a native primary/higher-order contact seam to uncertainty.

    Try the higher-order side first. If that would split or consume its owned
    body, reassign the touching primary-side vertices instead. All primary-side
    proposals use one frozen label snapshot. A failed proposal cannot veto an
    unrelated safe one, and every accepted union must preserve primary native
    connectivity and O1 contacts. No protected vertex moves.
    """
    p = np.asarray(points, float)
    before = np.asarray(labels, int)
    excluded = (np.zeros(len(p), bool) if excluded_mask is None
                else np.asarray(excluded_mask, bool))
    faces = (np.empty((0, 3), int) if triangles is None
             else np.asarray(triangles, int))
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("points must contain finite XYZ coordinates")
    if before.shape != (len(p),) or excluded.shape != (len(p),):
        raise ValueError("labels and excluded_mask must match points")
    if faces.ndim != 2 or faces.shape[1] != 3 or (len(faces) and
            (faces.min() < 0 or faces.max() >= len(p))):
        raise ValueError("triangles must have valid vertex indices")
    if not np.isfinite(d_bar) or d_bar <= 0:
        raise ValueError("d_bar must be positive and finite")
    if np.any((before < -2) | (before > len(roots))):
        raise ValueError("assigned labels must reference a root")

    result = before.copy()
    report = {
        "policy": "higher-order-primary-native-contact-v4",
        "status": "evaluated" if len(faces) else "unresolved_no_mesh",
        "contact_definition": "every nondegenerate native triangle edge",
        "repair_support_definition": "native triangle edge no longer than 4*d_bar",
        "repair": "reassign child-side or primary-side seam to uncertain; preserve owned components and O1 attachment",
        "changed_vertex_count": 0,
        "contact_root_count": 0,
        "unresolved_root_count": 0,
        "native_contact_edge_count_before": 0,
        "native_contact_edge_count_after": 0,
        "excluded_long_edge_contact_count_before": 0,
        "excluded_long_edge_contact_count_after": 0,
        "requires_correction": False,
        "contacts": [],
    }
    if not len(faces):
        return result, report

    if mesh_context is not None:
        mesh_context.validate(p, faces)
        raw_edges = mesh_context.edges
        raw_lengths = mesh_context.edge_lengths
    else:
        raw_edges = np.unique(np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]],
                                                  faces[:, [2, 0]])), axis=1), axis=0)
        raw_lengths = np.linalg.norm(p[raw_edges[:, 0]] - p[raw_edges[:, 1]], axis=1)
    nondegenerate = raw_edges[:, 0] != raw_edges[:, 1]
    raw_edges = raw_edges[nondegenerate]
    lengths = raw_lengths[nondegenerate]
    edges = raw_edges[lengths <= 4 * d_bar]
    edges.setflags(write=False)
    ownership = (mesh_context.ownership(before) if mesh_context is not None
                 else OwnershipGeometryGeneration(before, p))
    order_by_label = np.array([0, *(int(root.order) for root in roots)])
    label_by_id = {str(root.root_id): label for label, root in enumerate(roots, 1)}
    raw_left, raw_right = before[raw_edges[:, 0]], before[raw_edges[:, 1]]
    raw_contact = (((raw_left == 0) & (raw_right > 0)) |
                   ((raw_right == 0) & (raw_left > 0)))
    raw_contact_edges = raw_edges[raw_contact]
    raw_other_labels = np.maximum(raw_left[raw_contact], raw_right[raw_contact])
    raw_higher = order_by_label[raw_other_labels] > 1
    raw_contact_edges = raw_contact_edges[raw_higher]
    raw_other_labels = raw_other_labels[raw_higher]
    left, right = before[edges[:, 0]], before[edges[:, 1]]
    contact = (((left == 0) & (right > 0)) |
               ((right == 0) & (left > 0)))
    contact_edges = edges[contact]
    other_labels = np.maximum(left[contact], right[contact])
    higher = order_by_label[other_labels] > 1
    contact_edges = contact_edges[higher]
    other_labels = other_labels[higher]
    report["native_contact_edge_count_before"] = int(len(raw_contact_edges))
    report["excluded_long_edge_contact_count_before"] = int(
        len(raw_contact_edges) - len(contact_edges)
    )

    pending_primary: list[tuple[dict, np.ndarray]] = []
    for label in np.unique(raw_other_labels):
        root = roots[int(label) - 1]
        touching = contact_edges[other_labels == label]
        raw_touching = raw_contact_edges[raw_other_labels == label]
        child_vertices = np.unique(touching[before[touching] == label])
        editable = child_vertices[~excluded[child_vertices]]
        row = {
            "root_id": str(root.root_id),
            "root_order": int(root.order),
            "parent_id": str(root.parent_id),
            "contact_edge_count": int(len(raw_touching)),
            "repair_supported_contact_edge_count": int(len(touching)),
            "excluded_long_edge_contact_edge_count": int(len(raw_touching) - len(touching)),
            "contact_child_vertex_count": int(len(child_vertices)),
            "protected_contact_child_vertex_count": int(len(child_vertices) - len(editable)),
            "changed_child_vertex_count": 0,
            "changed_primary_vertex_count": 0,
            "changed_child_vertex_indices": [],
            "changed_primary_vertex_indices": [],
            "changed_vertex_count": 0,
        }
        report["contact_root_count"] += 1
        if not len(touching):
            row["status"] = "unresolved_excluded_long_edge_contact"
        elif not len(editable):
            row["status"] = "unresolved_protected_contact"
        else:
            owned = ownership.vertices(label)
            retained = np.setdiff1d(owned, editable, assume_unique=True)
            if not len(retained):
                row["status"] = "unresolved_would_consume_root"
            else:
                same_label_edges = ownership.edges(int(label), edges)
                body_anchor_status = _exposed_body_anchor_status(
                    p, root, owned, retained, same_label_edges, d_bar,
                    identity_references=identity_references,
                )
                if not _preserves_components(len(p), same_label_edges, editable, retained):
                    row["status"] = "unresolved_would_split_owned_surface"
                elif body_anchor_status != "preserved":
                    row["status"] = body_anchor_status
                elif not _preserves_parent_contact(
                    edges, before, int(label), label_by_id.get(str(root.parent_id)), editable
                ):
                    row["status"] = "unresolved_would_remove_parent_attachment"
                elif any(
                    not _preserves_parent_contact(
                        edges, before, int(label), descendant_label, editable
                    )
                    for descendant_label, descendant in enumerate(roots, 1)
                    if str(descendant.parent_id) == str(root.root_id)
                ):
                    row["status"] = "unresolved_would_remove_descendant_attachment"
                else:
                    result[editable] = -2
                    row["changed_child_vertex_count"] = int(len(editable))
                    row["changed_child_vertex_indices"] = editable.tolist()
                    row["status"] = ("unresolved_protected_contact" if len(editable) < len(child_vertices)
                                     else "reassigned_child_seam_uncertain")
        remaining = _contact_edges_for_label(touching, result, int(label))
        row["child_seam_status"] = row["status"]
        if len(remaining):
            primary_vertices = np.unique(remaining[before[remaining] == 0])
            if np.any(excluded[primary_vertices]):
                row["status"] = "unresolved_protected_primary_contact"
                row["primary_seam_status"] = "protected"
            else:
                pending_primary.append((row, primary_vertices))
        else:
            row["primary_seam_status"] = "not_needed"
        report["contacts"].append(row)

    if pending_primary:
        primary_edges = ownership.edges(0, edges)
        primary_owned = ownership.vertices(0)
        closure_guards = []
        closure_context = {}

        def rejection(proposed):
            retained = np.setdiff1d(primary_owned, proposed, assume_unique=True)
            if not len(retained) or not _preserves_components(len(p), primary_edges, proposed, retained):
                return "unresolved_would_split_primary_surface"
            if not _preserves_o1_contacts(edges, before, order_by_label, proposed):
                return "unresolved_would_remove_o1_contact"
            for guarded_vertices, closure in closure_guards:
                if np.isin(guarded_vertices, proposed).all() and not _closure_wall_preserved(
                        p, primary_owned, proposed, d_bar,
                        (identity_references or {}).get("primary"), closure_context, closure):
                    return "unresolved_joint_primary_wall_support"
            return None

        # Evaluate proposals against the same frozen surface, then validate the
        # accepted union. A bad seam elsewhere must not prevent a safe repair.
        individually_safe = []
        for row, primary_vertices in pending_primary:
            reason = rejection(primary_vertices)
            if reason == "unresolved_would_split_primary_surface":
                expanded, closure = _close_local_primary_seam(
                    p, before, raw_edges, primary_edges, primary_owned,
                    primary_vertices, excluded, d_bar,
                    (identity_references or {}).get("primary"), closure_context)
                row["primary_seam_closure"] = closure
                if len(expanded) > len(primary_vertices) and rejection(expanded) is None:
                    closure_guards.append((expanded, closure))
                    if rejection(expanded) is None:
                        primary_vertices = expanded
                        reason = None
            if reason is None:
                individually_safe.append((row, primary_vertices))
            else:
                row["primary_seam_status"] = reason
                row["status"] = reason
        proposed = (np.unique(np.concatenate([item[1] for item in individually_safe]))
                    if individually_safe else np.empty(0, dtype=int))
        accepted = individually_safe
        if len(proposed) and rejection(proposed) is not None:
            # Interacting individually safe seams can jointly cut a root. Use
            # a stable order and check each complete accumulated proposal.
            # The preference is more resolved native contacts, then less
            # released surface, never biological root length or root size.
            accepted, proposed = [], np.empty(0, dtype=int)
            for row, vertices in sorted(individually_safe, key=lambda item:
                    (-item[0]["repair_supported_contact_edge_count"], len(item[1]), item[0]["root_id"])):
                candidate = np.union1d(proposed, vertices)
                if rejection(candidate) is None:
                    accepted.append((row, vertices))
                    proposed = candidate
                else:
                    row["primary_seam_status"] = "unresolved_conflicting_primary_seam_proposals"
                    row["status"] = "unresolved_conflicting_primary_seam_proposals"
        if len(proposed):
            assert rejection(proposed) is None
            result[proposed] = -2
            for row, primary_vertices in accepted:
                row["changed_primary_vertex_count"] = int(len(primary_vertices))
                row["changed_primary_vertex_indices"] = primary_vertices.tolist()
                row["primary_seam_status"] = "reassigned_uncertain"
                row["ownership_resolution"] = "ambiguous_native_contact_separated_to_uncertainty"
                row["biological_identity_resolved"] = False
                row["status"] = ("reassigned_both_seams_uncertain"
                                 if row["changed_child_vertex_count"] else
                                 "reassigned_primary_seam_uncertain")

    for row in report["contacts"]:
        label = next(index for index, root in enumerate(roots, 1)
                     if str(root.root_id) == row["root_id"])
        touching = contact_edges[other_labels == label]
        raw_touching = raw_contact_edges[raw_other_labels == label]
        remaining_supported = len(_contact_edges_for_label(touching, result, label))
        remaining_raw = len(_contact_edges_for_label(raw_touching, result, label))
        row["remaining_contact_edge_count"] = int(remaining_raw)
        row["remaining_repair_supported_contact_edge_count"] = int(remaining_supported)
        row["remaining_excluded_long_edge_contact_edge_count"] = int(
            remaining_raw - remaining_supported
        )
        if remaining_raw and not remaining_supported and row["status"].startswith("reassigned"):
            row["status"] = "unresolved_excluded_long_edge_contact"
        row["changed_vertex_count"] = (row["changed_child_vertex_count"] +
                                       row["changed_primary_vertex_count"])
        if remaining_raw:
            report["unresolved_root_count"] += 1
        report["native_contact_edge_count_after"] += int(remaining_raw)
        report["excluded_long_edge_contact_count_after"] += int(
            remaining_raw - remaining_supported
        )
    report["changed_vertex_count"] = int(np.count_nonzero(result != before))
    report["status"] = ("unresolved_contacts" if report["unresolved_root_count"]
                        else "repaired_or_clear")
    report["requires_correction"] = bool(report["unresolved_root_count"])
    return result, report


def _close_local_primary_seam(points, labels, raw_edges, primary_edges, owned,
                              seam, excluded, spacing, reference, cache):
    """Include a stranded part of the same ambiguous native contact margin.

    This is uncertainty separation, not a claim of a new biological owner.
    A fragment must be enclosed by the seam/already uncertain surface, remain
    within two sampling spacings of it, have negligible axial extent, and leave
    independently supported circumferential primary wall at every local section.
    Native contacts with any other root, primary endpoints and supported long
    bodies are barriers. No geometry, child ownership or hierarchy is changed.
    """
    report = {"policy": "bounded-native-primary-seam-closure-v1",
              "status": "unresolved_no_independent_primary_reference",
              "closure_vertex_indices": [], "fragments": []}
    if not isinstance(reference, dict) or not reference.get("reference_valid"):
        return seam, report
    line = np.asarray(reference.get("prior_path", []), float)
    if line.ndim != 2 or line.shape[1:] != (3,) or len(line) < 2 or not np.isfinite(line).all():
        return seam, report
    if "primary_component" not in cache:
        cache["primary_component"] = _component_ids(len(owned), np.searchsorted(owned, primary_edges))
        cache["primary_projection"] = _polyline_projection_distance_and_arc(points[owned], line)
    original_component = cache["primary_component"]
    kept = primary_edges[~np.isin(primary_edges, seam).any(axis=1)]
    cut_component = _component_ids(len(owned), np.searchsorted(owned, kept))
    _, arc = cache["primary_projection"]
    total = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
    closure = []
    from scipy.spatial import cKDTree
    seam_tree = cKDTree(points[seam])
    for original in np.unique(original_component[np.searchsorted(owned, seam)]):
        domain = owned[(original_component == original) & ~np.isin(owned, seam)]
        parts = np.unique(cut_component[np.searchsorted(owned, domain)])
        if len(parts) <= 1:
            continue
        for part in parts:
            fragment = domain[cut_component[np.searchsorted(owned, domain)] == part]
            local_arc = arc[np.searchsorted(owned, fragment)]
            # Entire observed long bodies fail these local geometric bounds.
            if (np.any(excluded[fragment]) or np.ptp(local_arc) > 2 * spacing
                    or np.min(local_arc) <= 4 * spacing or total - np.max(local_arc) <= 4 * spacing
                    or np.max(seam_tree.query(points[fragment])[0]) > 2 * spacing):
                continue
            boundary_edges = raw_edges[np.isin(raw_edges, fragment).any(axis=1)]
            boundary = np.setdiff1d(np.unique(boundary_edges), fragment)
            if not len(boundary) or np.any((labels[boundary] != -2) & ~np.isin(boundary, seam)):
                continue
            # Require a connected retained primary component spanning both
            # axial sides of this contact and retaining a broad native wall.
            retained = domain[cut_component[np.searchsorted(owned, domain)] != part]
            supported_sections = []
            for station in np.unique([np.min(local_arc), np.median(local_arc), np.max(local_arc)]):
                section_ids = owned[(original_component == original) & (np.abs(arc-station) <= 2*spacing)]
                observed = section_ids[~np.isin(section_ids, np.r_[seam, fragment])]
                before_coverage, _ = _section_coverage(points, section_ids, line, station)
                after_coverage, direction = _section_coverage(points, observed, line, station)
                section_components = cut_component[np.searchsorted(owned, observed)]
                continuity = False
                for comp in np.unique(section_components):
                    component_ids = retained[cut_component[np.searchsorted(owned, retained)] == comp]
                    if not len(component_ids):
                        continue
                    component_arc = arc[np.searchsorted(owned, component_ids)]
                    if np.min(component_arc) <= station - 2*spacing and np.max(component_arc) >= station + 2*spacing:
                        continuity = True
                        break
                supported_sections.append({"station": float(station), "retained_vertex_count": int(len(observed)),
                    "angular_coverage_before_deg": before_coverage, "angular_coverage_after_deg": after_coverage,
                    "retained_axial_continuity": continuity})
                if (len(observed) < 6 or direction is None or not continuity
                        or before_coverage < 270 or after_coverage < 270
                        or after_coverage < .9 * before_coverage):
                    break
            else:
                closure.extend(fragment.tolist())
                report["fragments"].append({"vertex_indices": fragment.tolist(),
                    "original_primary_component": int(original),
                    "axial_extent": float(np.ptp(local_arc)), "native_boundary_labels": sorted(set(labels[boundary].tolist())),
                    "maximum_seam_distance": float(np.max(seam_tree.query(points[fragment])[0])),
                    "retained_wall_sections": supported_sections})
    if closure:
        report["status"] = "supported_local_contact_margin_closure"
        report["closure_vertex_indices"] = sorted(set(closure))
    else:
        report["status"] = "unresolved_no_supported_local_closure"
    return np.union1d(seam, closure).astype(int), report


def _closure_wall_preserved(points, owned, removed, spacing, reference, cache, report):
    """Recheck every closure section against the complete final proposal."""
    line = np.asarray(reference["prior_path"], float)
    _, arc = cache["primary_projection"]
    component = cache["primary_component"]
    keep = ~np.isin(owned, removed)
    for fragment in report["fragments"]:
        for section in fragment["retained_wall_sections"]:
            station = section["station"]
            ids = owned[keep & (component == fragment["original_primary_component"])
                        & (np.abs(arc-station) <= 2*spacing)]
            coverage, direction = _section_coverage(points, ids, line, station)
            if (len(ids) < 6 or direction is None or coverage < 270
                    or coverage < .9 * section["angular_coverage_before_deg"]):
                return False
    return True


def _section_coverage(points, indices, line, station):
    if len(indices) < 3:
        return 0., None
    center, tangent, _ = _sample(line, np.asarray([station]))
    direction = tangent[0]
    if not np.isfinite(direction).all() or np.linalg.norm(direction) < .5:
        return 0., None
    first = np.cross(direction, np.eye(3)[np.argmin(np.abs(direction))])
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    vectors = points[indices] - center[0]
    angles = np.sort(np.arctan2(vectors @ second, vectors @ first))
    coverage = float(np.degrees(2*np.pi - np.max(np.diff(np.r_[angles, angles[0]+2*np.pi]))))
    return coverage, direction


def mark_higher_order_primary_contact_qc(roots: list[RootPath], report: dict) -> None:
    """Carry segmentation errors and repair state into root QC exports."""
    by_id = {str(row["root_id"]): row for row in report.get("contacts", [])}
    flags = {"primary_contact_segmentation_error", "primary_contact_repaired",
             "primary_contact_unresolved"}
    for root in roots:
        root.qc_flags = [flag for flag in root.qc_flags if flag not in flags]
        row = by_id.get(str(root.root_id))
        if row is None:
            continue
        root.qc_flags.append("primary_contact_segmentation_error")
        root.qc_flags.append("primary_contact_unresolved" if row["remaining_contact_edge_count"]
                             else "primary_contact_repaired")


def _component_ids(count: int, edges: np.ndarray) -> np.ndarray:
    graph = coo_matrix((np.ones(2 * len(edges)),
                        (np.r_[edges[:, 0], edges[:, 1]],
                         np.r_[edges[:, 1], edges[:, 0]])),
                       shape=(count, count)).tocsr()
    return connected_components(graph, directed=False)[1]


def _contact_edges_for_label(edges: np.ndarray, labels: np.ndarray,
                             label: int) -> np.ndarray:
    left, right = labels[edges[:, 0]], labels[edges[:, 1]]
    return edges[((left == 0) & (right == label)) |
                 ((right == 0) & (left == label))]


def _preserves_components(count: int, edges: np.ndarray,
                          removed: np.ndarray, retained: np.ndarray) -> bool:
    # Include isolated retained vertices, but allocate only this root's domain.
    domain = np.unique(np.r_[edges.ravel(), removed, retained])
    local_edges = np.searchsorted(domain, edges)
    local_retained = np.searchsorted(domain, retained)
    before = _component_ids(len(domain), local_edges)
    kept = edges[(~np.isin(edges[:, 0], removed)) &
                 (~np.isin(edges[:, 1], removed))]
    after = _component_ids(len(domain), np.searchsorted(domain, kept))
    return not any(
        np.unique(after[local_retained[before[local_retained] == part]]).size > 1
        for part in np.unique(before[local_retained])
    )


def _exposed_body_anchor_status(
    points: np.ndarray, root: RootPath, owned: np.ndarray,
    retained: np.ndarray, same_label_edges: np.ndarray, spacing: float,
    *, identity_references: dict | None = None,
) -> str:
    """Keep a distal body anchor in each anchored native component.

    A seam cut can leave a connected proximal remnant while consuming the
    exposed child body. Anchor evidence is derived from frozen owned vertices
    near the distal half of the existing exposed centerline; a component with
    such evidence must retain some of it after the proposed cut.
    """
    body, radius, provenance = identity_reference(root, identity_references)
    if body is None:
        return provenance
    if len(body) < 2 or not len(owned):
        return "unresolved_no_exposed_body_anchor"
    length = float(np.linalg.norm(np.diff(body, axis=0), axis=1).sum())
    if length <= 1e-12:
        return "unresolved_no_exposed_body_anchor"
    distance, arc = _polyline_projection_distance_and_arc(points[owned], body)
    radius_limit = (
        1.5 * float(radius) if radius is not None and np.isfinite(radius)
        and radius > 0 else 0.0
    )
    anchor = owned[(arc >= 0.5 * length) &
                   (distance <= max(2.0 * spacing, radius_limit))]
    if not len(anchor):
        return "unresolved_no_exposed_body_anchor"
    component = _component_ids(len(owned), np.searchsorted(owned, same_label_edges))
    retained_anchors = np.intersect1d(anchor, retained, assume_unique=True)
    return "preserved" if all(
        np.any(component[np.searchsorted(owned, retained_anchors)] == part)
        for part in np.unique(component[np.searchsorted(owned, anchor)])
    ) else "unresolved_would_consume_exposed_body"


def _preserves_o1_contacts(edges: np.ndarray, labels: np.ndarray,
                           orders: np.ndarray, removed_primary: np.ndarray) -> bool:
    left, right = labels[edges[:, 0]], labels[edges[:, 1]]
    contact = (((left == 0) & (right > 0)) |
               ((right == 0) & (left > 0)))
    o1_edges = edges[contact]
    o1_labels = np.maximum(left[contact], right[contact])
    o1 = orders[o1_labels] == 1
    o1_edges, o1_labels = o1_edges[o1], o1_labels[o1]
    removed = np.isin(o1_edges[:, 0], removed_primary) | \
              np.isin(o1_edges[:, 1], removed_primary)
    for label in np.unique(o1_labels):
        affected = o1_labels == label
        total = int(np.count_nonzero(affected))
        lost = int(np.count_nonzero(removed & affected))
        if lost == total or lost > .20 * total:
            return False
    return True


def _preserves_parent_contact(edges: np.ndarray, labels: np.ndarray,
                              child: int, parent: int | None,
                              removed_child: np.ndarray) -> bool:
    if parent is None or parent == 0:
        return True
    left, right = labels[edges[:, 0]], labels[edges[:, 1]]
    contact = (((left == child) & (right == parent)) |
               ((right == child) & (left == parent)))
    parent_edges = edges[contact]
    if not len(parent_edges):
        return True
    lost = int(np.count_nonzero(
        np.isin(parent_edges[:, 0], removed_child) |
        np.isin(parent_edges[:, 1], removed_child)))
    return lost < len(parent_edges) and lost <= .5 * len(parent_edges)
