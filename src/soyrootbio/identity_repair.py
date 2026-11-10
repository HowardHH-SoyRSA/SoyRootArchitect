"""Conservative repair of observed interruptions in a root's surface ownership.

The frozen trace is a query prior, never proof that a seam belongs to a root.
Only independently measured native tube sections may support a transfer. This
module changes labels, not vertices, faces, centerlines, origins or parentage.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .mesh_geometry import MeshGeometryContext
from .surface_patches import _polyline_projection_distance_and_arc as project
from .types import RootPath


def _digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def freeze_identity_references(primary_path, roots: list[RootPath]) -> dict:
    """Snapshot pre-fit identity evidence without blessing a legacy final fit.

    Coordinates must share the caller's geometry coordinate system. Existing
    references are copied unchanged; callers retain this dictionary across all
    later fit/ownership passes. A legacy assessed root needs explicit evidence.
    """
    references = {}
    for root_id, path, start, radius, assessment in [
        ("primary", primary_path, 0, None, {}),
        *((r.root_id, r.points, r.body_start_index, r.mean_radius,
           r.centerline_assessment) for r in roots),
    ]:
        existing = assessment.get("prefit_identity_evidence")
        if existing is not None:
            references[str(root_id)] = deepcopy(existing)
        elif not assessment:
            line = np.asarray(path, float)
            references[str(root_id)] = {
                "policy": "pre-fit-exposed-body-v1", "prior_path": line.tolist(),
                "body_start_index": int(start), "mean_radius": radius,
                "coordinate_unit": "analysis_normalized",
                "path_fingerprint": _digest(line.astype("<f8")),
                "origin": line[0].tolist() if len(line) else None,
                "reference_valid": True,
            }
    return references


def identity_reference(root: RootPath, identity_references=None):
    """Return an independent exposed-body prior, radius and provenance status."""
    evidence = ((identity_references or {}).get(str(root.root_id))
                if identity_references is not None else
                root.centerline_assessment.get("prefit_identity_evidence"))
    if evidence is None:
        if root.centerline_assessment or identity_references is not None:
            return None, None, "unresolved_missing_prefit_reference"
        # An unassessed tracing root has not undergone final support clipping.
        evidence = freeze_identity_references(np.empty((0, 3)), [root])[root.root_id]
        provenance = "unassessed_pre_fit_trace"
    else:
        provenance = "persistent_pre_fit_reference"
    if not isinstance(evidence, dict) or not evidence.get("reference_valid", False):
        return None, None, "unresolved_invalid_prefit_reference"
    try:
        path = np.asarray(evidence["prior_path"], dtype=float)
        start = int(evidence.get("body_start_index", 0))
        radius = evidence.get("mean_radius")
        radius = float(radius) if radius is not None else None
    except (KeyError, TypeError, ValueError):
        return None, None, "unresolved_invalid_prefit_reference"
    if (path.ndim != 2 or path.shape[1] != 3 or not np.isfinite(path).all()
            or start < 0 or start >= len(path) or len(path[start:]) < 2
            or radius is not None and (not np.isfinite(radius) or radius <= 0)):
        return None, None, "unresolved_invalid_prefit_reference"
    return path[start:].copy(), radius, provenance


def _components(size, edges):
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(size, size)).tocsr()
    return connected_components(graph, directed=False)[1]


def _sample(path, stations):
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    index = np.clip(np.searchsorted(arc, stations, side="right") - 1, 0, len(path) - 2)
    vectors = path[index + 1] - path[index]
    lengths = np.linalg.norm(vectors, axis=1)
    direction = vectors / np.maximum(lengths[:, None], 1e-15)
    centers = path[index] + direction * (stations - arc[index])[:, None]
    return centers, direction, arc[-1]


def _native_contacts(labels, edges, roots):
    pairs = np.sort(labels[edges], axis=1)
    lookup = {"primary": 0, **{r.root_id: i for i, r in enumerate(roots, 1)}}
    contact = {}
    for label, root in enumerate(roots, 1):
        parent = lookup.get(root.parent_id)
        if parent is not None:
            pair = sorted((label, parent))
            contact[label] = edges[np.all(pairs == pair, axis=1)]
    return contact


def _transaction_guard(before, after, edges, roots, before_components):
    """Reject joint changes that consume a root, split a donor, or lose support."""
    changed = before != after
    rejected = {}
    if not np.any(changed):
        return rejected
    current = _components(len(before), edges[after[edges[:, 0]] == after[edges[:, 1]]])
    for donor in np.unique(before[changed & (before >= 0)]):
        moved = changed & (before == donor)
        receivers = np.unique(after[moved])
        old = np.flatnonzero(before == donor)
        if not np.any(after[old] == donor):
            rejected.update({int(v): "unresolved_would_consume_donor_root" for v in receivers})
            continue
        for part in np.unique(before_components[old]):
            members = old[before_components[old] == part]
            retained = members[after[members] == donor]
            # A fully transferred detached fragment already passed independent
            # tube and donor-core tests. No surviving donor component may split.
            if len(retained) and len(np.unique(current[retained])) > 1:
                rejected.update({int(v): "unresolved_would_split_donor_surface"
                                 for v in np.unique(after[members[changed[members]]])})
    old_contacts = _native_contacts(before, edges, roots)
    new_contacts = _native_contacts(after, edges, roots)
    for child, old in old_contacts.items():
        if len(old) and not len(new_contacts[child]):
            involved = np.unique(after[np.unique(old)[changed[np.unique(old)]]])
            rejected.update({int(v): "unresolved_would_remove_parent_or_descendant_attachment"
                             for v in involved if v > 0})
    for label, root in enumerate(roots, 1):
        if root.order < 2:
            continue
        old = ((before[edges[:, 0]] == 0) & (before[edges[:, 1]] == label)) | \
              ((before[edges[:, 1]] == 0) & (before[edges[:, 0]] == label))
        new = ((after[edges[:, 0]] == 0) & (after[edges[:, 1]] == label)) | \
              ((after[edges[:, 1]] == 0) & (after[edges[:, 0]] == label))
        if np.any(new & ~old):
            rejected[label] = "unresolved_would_contact_primary"
    return rejected


def reconcile_identity_ownership(
    points: np.ndarray, labels: np.ndarray, primary_path: np.ndarray,
    roots: list[RootPath], *, triangles: np.ndarray | None, d_bar: float,
    primary_top_reference: np.ndarray, gravity=(0., 0., -1.),
    excluded_mask: np.ndarray | None = None,
    mesh_context: MeshGeometryContext | None = None,
    identity_references: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Repair only uniquely supported parent/uncertain seams on native tubes.

    All hypotheses use frozen labels. Unassigned, excluded and nodule vertices
    cannot be claimed. Complete proposals compete and commit atomically; no
    partial strip survives a conflict or preservation rollback. Missing native
    surface and existing higher-order/primary contacts remain explicit reviews.
    """
    p, before = np.asarray(points, float), np.asarray(labels, dtype=np.int64)
    faces = np.empty((0, 3), np.int64) if triangles is None else np.asarray(triangles, np.int64)
    if (p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all()
            or before.shape != (len(p),) or np.any(before > len(roots))):
        raise ValueError("finite points and one valid owner label per vertex are required")
    if not np.isfinite(d_bar) or d_bar <= 0:
        raise ValueError("d_bar must be positive and finite")
    top, gravity = np.asarray(primary_top_reference, float), np.asarray(gravity, float)
    if (top.shape != (3,) or gravity.shape != (3,) or not np.isfinite(top).all()
            or not np.isfinite(gravity).all() or np.linalg.norm(gravity) <= 0):
        raise ValueError("an immutable primary top and finite gravity are required")
    excluded = np.zeros(len(p), bool) if excluded_mask is None else np.array(excluded_mask, bool, copy=True)
    if excluded.shape != before.shape:
        raise ValueError("excluded_mask must match labels")
    excluded |= (p - top) @ (-gravity / np.linalg.norm(gravity)) > 1e-9 * d_bar
    report = {
        "policy": "frozen-native-identity-seam-repair-v1", "status": "clear",
        "geometry_sha256": None, "evidence_labels_sha256": _digest(before.astype("<i8")),
        "changed_vertex_count": 0, "changed_vertex_indices": [], "decisions": [],
        "unresolved_root_count": 0, "root_geometry_and_hierarchy_changed": False,
        "evidence": "independent transverse rings, normals, direction, frozen competitors and native connectivity",
    }
    if not len(faces):
        report["status"] = "unresolved_no_native_mesh"
        return before.copy(), report
    context = mesh_context or MeshGeometryContext.build(p, faces)
    context.validate(p, faces)
    report["geometry_sha256"] = context.coordinate_sha256
    raw = context.edges[(context.edges[:, 0] != context.edges[:, 1]) & (context.edge_lengths > 0)]
    safe = context.bounded_edges(d_bar)
    safe = safe[~excluded[safe].any(axis=1)]
    raw_components = _components(len(p), raw)
    ownership = context.ownership(before)
    components = ownership.components(safe, active=(before >= 0) & ~excluded)
    original_components = ownership.components(raw)
    references = (freeze_identity_references(primary_path, roots)
                  if identity_references is None else deepcopy(identity_references))
    lookup = {"primary": 0, **{r.root_id: i for i, r in enumerate(roots, 1)}}
    paths = {0: np.asarray(primary_path, float)}
    for label, root in enumerate(roots, 1):
        paths[label] = identity_reference(root, references)[0]
    primary_ref = references.get("primary", {})
    if primary_ref.get("reference_valid"):
        paths[0] = np.asarray(primary_ref["prior_path"], float)
    # Imports are lazy to keep the reference accessor independent of fitting.
    from .attachment_constraint import _manifold_vertex_links
    from .junction_tubes import _measure_ring, _parent_core_mask, vertex_normals
    from .surface_patches import _segment_radius_profile
    normals = vertex_normals(p, faces)
    unsafe = np.zeros(len(p), bool)
    unsafe[np.unique(context.edges[context.edge_incidence != 2])] = True
    proposals, rows_by_label = {}, {}
    for label in sorted(range(1, len(roots) + 1), key=lambda v: roots[v - 1].root_id):
        root = roots[label - 1]
        owned = ownership.vertices(label)
        owned = owned[~excluded[owned]]
        parts, sizes = np.unique(components[owned], return_counts=True)
        high_contact = raw[((before[raw[:, 0]] == label) & (before[raw[:, 1]] == 0)) |
                           ((before[raw[:, 1]] == label) & (before[raw[:, 0]] == 0))]
        if len(parts) <= 1 and not (root.order >= 2 and len(high_contact)):
            continue
        row = {"root_id": root.root_id, "parent_id": root.parent_id,
               "component_sizes": sizes.tolist(), "status": "unresolved_disconnected_identity",
               "seams": [], "proposed_vertex_count": 0, "changed_vertex_count": 0}
        report["decisions"].append(row)
        rows_by_label[label] = row
        if root.order >= 2 and len(high_contact):
            row.update(status="unresolved_existing_primary_contact", native_contact_edge_count=len(high_contact))
            continue
        body, radius_hint, provenance = identity_reference(root, references)
        row["reference_source"] = provenance
        if body is None:
            row["status"] = provenance
            continue
        origin = np.asarray(references[root.root_id].get("origin", body[0]), float)
        if origin.shape != (3,) or not np.isfinite(origin).all() or \
                (origin - top) @ (-gravity / np.linalg.norm(gravity)) > 1e-9 * d_bar:
            row["status"] = "unresolved_origin_above_primary_top"
            continue
        parent = lookup.get(root.parent_id)
        if parent is None or paths.get(parent) is None:
            row["status"] = "unresolved_parent_reference"
            continue
        distance, arcs = project(p[owned], body)
        radius = max(float(np.median(distance)), d_bar)
        if radius_hint is not None:
            radius = max(min(radius, 2 * radius_hint), d_bar)
        intervals = []
        for part, size in zip(parts, sizes):
            indices = np.flatnonzero((components[owned] == part) & (distance <= 2 * radius + d_bar))
            if len(indices) >= 8:
                intervals.append((float(arcs[indices].min()), float(arcs[indices].max()), int(part), owned[indices]))
        intervals.sort(key=lambda v: (v[0], v[2]))
        proposals[label] = []
        for first, second in zip(intervals[:-1], intervals[1:]):
            a0, end, part_a, vertices_a = first
            start, b1, part_b, vertices_b = second
            seam = {"component_first_vertex": [int(vertices_a.min()), int(vertices_b.min())],
                    "status": "unresolved_independent_tube_evidence"}
            row["seams"].append(seam)
            if not set(raw_components[vertices_a]).intersection(raw_components[vertices_b]):
                seam["status"] = "unresolved_native_mesh_gap"
                continue
            if start < end - 2 * d_bar:
                seam["status"] = "unresolved_overlapping_component_identity"
                continue
            if start - end > max(8 * d_bar, 4 * radius):
                seam["status"] = "unresolved_seam_extent"
                continue
            flank = max(8 * d_bar, 3 * radius)
            low, high = max(a0, end - flank), min(b1, start + flank)
            stations = np.arange(low, high + .5 * d_bar, 2 * d_bar)
            if len(stations) < 7:
                continue
            centers, directions, _ = _sample(body, stations)
            nearby = context.point_tree.query_ball_point(centers, 3 * radius + 4 * d_bar)
            union = np.unique(np.concatenate([np.asarray(v, int) for v in nearby]))
            union = union[~excluded[union] & np.isin(before[union], [label, parent, -2])]
            if len(union) < 12:
                continue
            tree = cKDTree(p[union])
            rings, flank_a, flank_b = [], [], []
            for station, center, direction in zip(stations, centers, directions):
                ring = _measure_ring(p, normals, union, tree, center, direction, radius, d_bar)
                rings.append(ring)
                if ring is None:
                    continue
                ids = ring[3]
                if np.mean((before[ids] == label) & (components[ids] == part_a)) >= .8:
                    flank_a.append(float(station))
                if np.mean((before[ids] == label) & (components[ids] == part_b)) >= .8:
                    flank_b.append(float(station))
            seam["independent_flank_section_counts"] = [len(flank_a), len(flank_b)]
            if (len(flank_a) < 3 or len(flank_b) < 3 or np.ptp(flank_a) < 4 * d_bar - 1e-12
                    or np.ptp(flank_b) < 4 * d_bar - 1e-12):
                continue
            required = np.flatnonzero((stations >= min(flank_a)) & (stations <= max(flank_b)))
            if not len(required) or any(rings[v] is None for v in required):
                seam["status"] = "unresolved_unmeasured_seam_section"
                continue
            measured = [rings[v] for v in required]
            radii = np.array([v[2] for v in measured])
            directions = np.array([v[1] for v in measured])
            if radii.max() > 1.5 * radii.min() or np.any(np.sum(directions[1:] * directions[:-1], axis=1) < .8):
                seam["status"] = "unresolved_tube_direction_or_radius"
                continue
            claim = np.unique(np.concatenate([v[3] for v in measured]))
            claim = claim[np.isin(before[claim], [parent, -2]) & ~excluded[claim]]
            if not len(claim):
                seam["status"] = "unresolved_no_transferable_native_seam"
                continue
            if np.any(unsafe[claim]) or not _manifold_vertex_links(faces, context.vertex_face_incidence, claim):
                seam["status"] = "unresolved_open_or_nonmanifold_seam"
                continue
            boundary = raw[np.isin(raw, claim).any(axis=1)]
            competing_native = np.setdiff1d(np.unique(before[boundary]), [label, parent, -2, -1])
            if len(competing_native):
                seam.update(status="unresolved_competing_native_owner", competing_labels=competing_native.tolist())
                continue
            # Competing paths do not win by distance. A compatible observed
            # tube centered on another independently retained path is enough
            # to make this proposal ambiguous, including an uncertain seam.
            competitors = []
            for other, other_path in paths.items():
                if other == label or other_path is None or len(other_path) < 2:
                    continue
                donor = ownership.vertices(other)
                donor = donor[~excluded[donor] & ~np.isin(donor, claim)]
                if len(donor) < 12:
                    continue
                ds, da = project(p[claim], other_path)
                ps, pr = _segment_radius_profile(p[donor], other_path, d_bar)
                wall = np.interp(da, ps, pr)
                if np.all(ds > 1.5 * wall + 4 * d_bar):
                    continue
                # A proposed child patch inside the donor's measured envelope
                # is not proof of an independently exposed child surface.
                inside = ds <= .9 * wall + .25 * d_bar
                donor_arc = project(p[donor], other_path)[1]
                inside, _ = _parent_core_mask(p, normals, donor, claim, other_path,
                                               d_bar, donor_arc, inside)
                if np.any(inside):
                    competitors.append(int(other))
            if competitors:
                seam.update(status="unresolved_competing_body_envelope", competing_labels=competitors)
                continue
            temporary = before.copy()
            temporary[claim] = label
            domain = safe[(temporary[safe] == label).all(axis=1)]
            after_parts = _components(len(p), domain)
            if not set(after_parts[vertices_a]).intersection(after_parts[vertices_b]):
                seam["status"] = "unresolved_no_supported_native_connection"
                continue
            # Every transferred vertex must be in the same supported component
            # as both original flanks; do not leave a newly isolated strip.
            joined = np.intersect1d(after_parts[vertices_a], after_parts[vertices_b])
            if not np.all(np.isin(after_parts[claim], joined)):
                seam["status"] = "unresolved_detached_proposal"
                continue
            seam.update(status="supported_native_tube_seam", proposed_vertex_indices=claim.tolist(),
                        measured_section_count=len(measured))
            proposals[label].append(claim)
        if proposals[label]:
            proposals[label] = np.unique(np.concatenate(proposals[label]))
            row["proposed_vertex_count"] = len(proposals[label])
        else:
            proposals.pop(label)
    # All competitors see exactly the same evidence. Ties reject the whole
    # root transaction, avoiding order-dependent partial bridges.
    winners = np.full(len(p), -1, int)
    conflicting = set()
    for label, claim in proposals.items():
        previous = winners[claim]
        conflicting.update(int(v) for v in previous[previous >= 0])
        if np.any(previous >= 0):
            conflicting.add(label)
        winners[claim] = label
    after = before.copy()
    for label, claim in proposals.items():
        if label in conflicting:
            rows_by_label[label]["status"] = "unresolved_competing_proposals"
        else:
            after[claim] = label
    for _ in range(len(proposals) + 1):
        rejected = _transaction_guard(before, after, raw, roots, original_components)
        active = {label: reason for label, reason in rejected.items()
                  if np.any((after == label) & (after != before))}
        if not active:
            break
        for label, reason in active.items():
            moved = (after == label) & (after != before)
            after[moved] = before[moved]
            rows_by_label[label]["status"] = reason
    for label, row in rows_by_label.items():
        changed = np.flatnonzero((after == label) & (before != after))
        row["changed_vertex_count"] = len(changed)
        if len(changed):
            row["status"] = "repaired_supported_native_seam"
        # An accepted seam can leave a different, unresolved component.
        own = np.flatnonzero(after == label)
        remaining = _components(len(p), safe[(after[safe] == label).all(axis=1)])
        row["remaining_component_count"] = len(np.unique(remaining[own]))
        row["identity_resolved"] = row["remaining_component_count"] == 1 and not (
            roots[label - 1].order >= 2 and row.get("native_contact_edge_count", 0))
    changed = np.flatnonzero(before != after)
    report.update(changed_vertex_count=len(changed), changed_vertex_indices=changed.tolist(),
                  result_labels_sha256=_digest(after.astype("<i8")),
                  unresolved_root_count=sum(not r.get("identity_resolved", False) for r in report["decisions"]))
    if report["unresolved_root_count"]:
        report["status"] = "unresolved_identity_regions"
    elif len(changed):
        report["status"] = "repaired_supported_native_seams"
    return after, report
