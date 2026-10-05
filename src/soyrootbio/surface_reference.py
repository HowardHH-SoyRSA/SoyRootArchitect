"""Opt-in, specimen-bound ownership constraints; never imports edited paths.

The manifest names an NPZ containing ordered source geometry, requested owner
indices, and stable surface anchors. KEEP means that the automatic owner must
survive. Positive indices address manifest owners, not runtime root labels.
All decisions use one frozen automatic label snapshot. Reference topology and
polylines are deliberately absent from this format.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .centerline import _axis, _legal_attachment_indices, _native_junction_footprint
from .mesh_geometry import MeshGeometryContext
from .types import RootPath

KEEP = np.iinfo(np.int32).min
SCHEMA = "soyrootbio.surface-reference/v1"


def array_sha256(array, dtype) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


@dataclass(frozen=True)
class SurfaceReference:
    manifest: dict
    requested: np.ndarray
    anchors: np.ndarray
    boundary: np.ndarray
    manifest_sha256: str
    data_sha256: str


def tube_claim_limits(reference, labels, roots, excluded_mask):
    """Bound new automatic sleeve claims using frozen reviewed surface anchors.

    This only limits future proposals. Existing ownership is never removed,
    and uncertain anchor matches are left to the final reference validation.
    Vertices explicitly assigned by the later reference do not consume the
    extra-surface allowance; KEEP vertices will retain their automatic owner.
    """
    before = np.asarray(labels, dtype=int)
    excluded = np.asarray(excluded_mask, dtype=bool)
    if before.shape != reference.requested.shape or excluded.shape != before.shape:
        raise ValueError("Reference tube limits must match native vertices")
    owners = reference.manifest["owners"]
    scores = np.zeros((len(owners), len(roots)), dtype=int)
    for index in range(len(owners)):
        values = before[(reference.anchors == index + 1) & ~excluded]
        scores[index] = np.bincount(values[values > 0], minlength=len(roots) + 1)[1:]
    assignment = dict(zip(*linear_sum_assignment(-scores))) if len(roots) else {}
    retained = (reference.requested == KEEP) | excluded
    limits = {}
    for index, owner in enumerate(owners):
        column = assignment.get(index)
        anchor_count = np.count_nonzero((reference.anchors == index + 1) & ~excluded)
        if column is None or not anchor_count or scores[index, column] < .5 * anchor_count:
            continue
        label = int(column + 1)
        expected = reference.requested == index + 1
        outside = retained & ~expected
        limits[label] = {
            "reference_root_id": owner["reference_root_id"],
            "outside_mask": outside,
            "expected_vertices": int(expected.sum()),
            "existing_differences": int(np.count_nonzero((before == label) & outside)
                + np.count_nonzero(expected & excluded & (before != label))),
            "maximum_difference_fraction": float(reference.manifest.get("maximum_difference_fraction", .02)),
        }
    return limits


def load_surface_reference(path, source_points, triangles) -> SurfaceReference:
    """Fail closed on different geometry, ordering, duplicates, or data content."""
    path = Path(path)
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("utf-8-sig"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported surface reference schema")
    owners = manifest.get("owners", [])
    names = [row["reference_root_id"] for row in owners]
    if not names or len(names) != len(set(names)):
        raise ValueError("Surface reference owner identities must be unique")
    tolerance = float(manifest.get("maximum_difference_fraction", .02))
    if not 0 <= tolerance <= .05:
        raise ValueError("Surface reference differences must be bounded to at most 5 percent")
    if triangles is None or not len(triangles):
        raise ValueError("Surface reference requires native triangles; no connectivity is invented")
    data_path = path.parent / manifest["data_file"]
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if digest != manifest["data_sha256"]:
        raise ValueError("Surface reference data hash changed")
    with np.load(data_path, allow_pickle=False) as data:
        if (not np.array_equal(data["points"], source_points)
                or not np.array_equal(data["triangles"], triangles)):
            raise ValueError("Surface reference ordered source geometry does not match")
        requested = data["requested"].copy()
        anchors = data["anchors"].copy()
        boundary = data["boundary"].copy()
    n = len(source_points)
    if (requested.shape != (n,) or anchors.shape != (n,) or boundary.shape != (n,)
            or requested.dtype.kind not in "iu" or anchors.dtype.kind not in "iu"
            or boundary.dtype != np.bool_):
        raise ValueError("Surface reference arrays require one typed value per source vertex")
    valid = (requested == KEEP) | ((requested >= -2) & (requested <= len(owners)))
    if not np.all(valid) or np.any((anchors < 0) | (anchors > len(owners))):
        raise ValueError("Surface reference contains unknown owner indices")
    if np.any((anchors > 0) & (anchors != requested)):
        raise ValueError("Surface anchors must be inside their requested owner")
    for index in range(1, len(owners) + 1):
        if not np.any(requested == index) or not np.any(anchors == index):
            raise ValueError("Every surface reference owner requires support and an anchor")
    for value in (requested, anchors, boundary):
        value.setflags(write=False)
    return SurfaceReference(manifest, requested, anchors, boundary,
                            hashlib.sha256(raw).hexdigest(), digest)


def _seed_axis(points, labels, label, hint, geometry, spacing):
    ownership = geometry.ownership(labels)
    vertices = ownership.vertices(label)
    if not len(vertices):
        return np.asarray(hint)[None, :], {"component_count": 0, "body_vertices": 0}
    graph = ownership.local_graph(label, geometry.support_edges(spacing))
    count, components = connected_components(graph, directed=False)
    keep = components == np.bincount(components).argmax()
    support = points[vertices[keep]]
    if len(support) < 8:
        line = np.median(support, axis=0)[None, :]
    else:
        line = _axis(support, graph[keep][:, keep], np.asarray(hint), spacing)
    return line, {"component_count": int(count), "body_vertices": int(keep.sum()),
                  "excluded_fragment_vertices": int((~keep).sum())}


def _repair_affected_hierarchy(points, labels, primary, roots, affected, removed,
                               before_roots, geometry, spacing, top, gravity):
    """Re-examine changed basal native contacts; retain ambiguity with QC.

    A reference has no parent or order fields. Existing automatic parents are
    retained unless corrected basal connectivity supports a different parent.
    Empty former parents are replaced using native support, or their nearest
    surviving ancestor with an explicit unresolved flag. No surface is bridged.
    """
    by_id = {root.root_id: root for root in roots}
    old_by_id = {root.root_id: root for root in before_roots}
    numeric = {root.root_id: i for i, root in enumerate(roots, 1)}
    reverse = {i: rid for rid, i in numeric.items()}
    reverse[0] = "primary"
    ownership = geometry.ownership(labels)
    edges = geometry.support_edges(spacing)
    primary_surface = points[labels == 0]
    primary_tree = cKDTree(primary_surface) if len(primary_surface) else None
    rows = []
    # Include descendants whose parent body or identity changed; their surface
    # remains frozen unless explicitly in the requested mask.
    review = set(affected)
    review.update(r.root_id for r in roots if r.parent_id in affected | removed)
    axes = {}
    for root in roots:
        if root.root_id not in review:
            continue
        hint = root.raw_start_point if root.raw_start_point is not None else root.points[0]
        line, detail = _seed_axis(points, labels, numeric[root.root_id], hint, geometry, spacing)
        # A formerly misclassified child may actually reach the primary at
        # the opposite end of its corrected shaft. Require native contact at
        # that end; Euclidean proximity alone cannot change the parent.
        if len(line) > 1 and primary_tree is not None:
            boundary = ownership.boundary_edges(0, numeric[root.root_id], edges)
            contacts = points[np.unique(boundary)]
            indices = ownership.vertices(numeric[root.root_id])
            support = points[indices]
            radius = float(np.median(cKDTree(support).query(line)[0])) if len(support) else spacing
            contact_distance = (cKDTree(contacts).query(line[[0, -1]])[0]
                                if len(contacts) else np.full(2, np.inf))
            if len(boundary) >= 3 and np.min(contact_distance) <= max(6 * spacing, 3 * radius):
                if contact_distance[1] < contact_distance[0]:
                    line = line[::-1].copy()
            elif root.parent_id == "primary":
                distances = primary_tree.query(line[[0, -1]])[0]
                if distances[1] < distances[0]:
                    line = line[::-1].copy()
        axes[root.root_id] = line
        if root.root_id in affected:
            root.points = line.copy()
            root.body_start_index = 0
            root.score_components.pop("native_recovered_extension", None)
            root.score_components["surface_reference_body"] = 1.0
            root.node_indices = None
            root.start_index = None
            root.raw_start_point = line[0].copy()
        if detail["component_count"] > 1:
            root.qc_flags.append("surface_reference_disconnected_support")
        rows.append({"root_id": root.root_id, "previous_parent": root.parent_id, **detail})
    decisions = {row["root_id"]: row for row in rows}
    proposed = {r.root_id: r.parent_id for r in roots}
    hints = {}
    for root in roots:
        rid = root.root_id
        if rid not in review:
            continue
        axis = axes[rid]
        indices = ownership.vertices(numeric[rid])
        support = points[indices]
        radius = (float(np.median(cKDTree(support).query(axis)[0])) if len(support) else spacing)
        limit = max(6 * spacing, 3 * radius)
        # The start window is anchored in the independently reconstructed body.
        near = np.linalg.norm(points - axis[0], axis=1) <= limit
        crossing = edges[((labels[edges[:, 0]] == numeric[rid]) & (labels[edges[:, 1]] != numeric[rid]))
                         | ((labels[edges[:, 1]] == numeric[rid]) & (labels[edges[:, 0]] != numeric[rid]))]
        if len(crossing):
            crossing = crossing[np.any(near[crossing], axis=1)]
        candidates = {}
        footprints = {}
        for label in np.unique(labels[crossing]) if len(crossing) else []:
            if label < 0 or label == numeric[rid]:
                continue
            boundary = crossing[np.any(labels[crossing] == label, axis=1)]
            if len(boundary) < 3:
                continue
            parent_id = reverse[int(label)]
            # A current descendant cannot become a parent in the same frozen
            # decision. Cycle checking after all proposals adds a second gate.
            ancestor = parent_id
            seen = set()
            while ancestor != "primary" and ancestor in by_id and ancestor not in seen:
                if ancestor == rid:
                    break
                seen.add(ancestor)
                ancestor = by_id[ancestor].parent_id
            if ancestor == rid:
                continue
            candidates[parent_id] = int(len(boundary))
            footprints[parent_id] = points[np.unique(boundary)].mean(axis=0)
        old_parent = root.parent_id
        chosen = old_parent
        status = "retained_automatic_parent"
        ranked = sorted(candidates, key=lambda value: (-candidates[value], value))
        if "primary" in candidates and (old_parent == "primary" or len(candidates) == 1):
            chosen, status = "primary", "basal_native_primary_contact"
        elif old_parent == "primary" and decisions[rid]["component_count"] > 1:
            # A distal fragment cannot establish a new biological origin for
            # a disconnected main shaft. Keep the existing relationship and
            # report the missing continuous support instead of reversing it
            # with a child found at the cut.
            status = "unresolved_disconnected_basal_support"
        elif ranked and (len(ranked) == 1 or candidates[ranked[0]] >= 2 * candidates[ranked[1]]):
            chosen, status = ranked[0], "dominant_basal_native_contact"
        elif old_parent in candidates:
            status = "unresolved_competing_basal_contacts"
        elif old_parent in removed:
            # Keep a valid tree but do not claim the fallback is biologically
            # resolved. Descendant bodies are never deleted with their parent.
            while chosen in removed:
                chosen = old_by_id[chosen].parent_id
            status = "unresolved_removed_parent_ancestor_fallback"
        elif not candidates:
            parent_label = 0 if old_parent == "primary" else numeric.get(old_parent)
            footprint = (np.empty((0, 3)) if parent_label is None else
                         _native_junction_footprint(points, labels, edges, parent_label,
                             numeric[rid], axis[0], radius, spacing, top, gravity))
            if len(footprint):
                footprints[old_parent] = footprint.mean(axis=0)
                status = "retained_parent_bounded_native_cut"
            else:
                status = "unresolved_no_basal_native_contact"
        else:
            status = "unresolved_competing_basal_contacts"
        proposed[rid] = chosen
        hints[rid] = footprints.get(chosen, axis[0])
        decisions[rid].update(parent_candidates=candidates, proposed_parent=chosen, status=status)
    # Simultaneous proposals can form cycles even if each old tree was valid.
    for rid in sorted(proposed):
        seen = set()
        cursor = rid
        while cursor != "primary":
            if cursor in seen:
                for member in seen:
                    if member in decisions:
                        prior = old_by_id.get(member)
                        fallback = prior.parent_id if prior is not None else "primary"
                        while fallback in removed:
                            fallback = old_by_id[fallback].parent_id
                        proposed[member] = fallback
                        decisions[member]["status"] = "unresolved_competing_parent_cycle"
                break
            seen.add(cursor)
            cursor = proposed[cursor]
    pending = list(roots)
    resolved = {"primary": (0, primary)}
    while pending:
        ready = [r for r in pending if proposed[r.root_id] in resolved]
        if not ready:
            raise ValueError("Reference contact proposals cannot form an acyclic hierarchy")
        for root in ready:
            parent_id = proposed[root.root_id]
            parent_order, parent_line = resolved[parent_id]
            parent_changed = root.parent_id != parent_id
            root.parent_id = parent_id
            root.order = parent_order + 1
            if root.root_id in review or parent_changed:
                legal = _legal_attachment_indices(parent_line, np.asarray(top)[None, :], gravity)
                if not len(legal):
                    raise ValueError("Reference parent has no legal below-top insertion")
                hint = hints.get(root.root_id, root.points[0])
                idx = int(legal[np.argmin(np.linalg.norm(parent_line[legal] - hint, axis=1))])
                root.insertion_index = idx
                root.insertion_point = parent_line[idx].copy()
                root.parent_points = parent_line
                row = decisions[root.root_id]
                row["parent_id"] = parent_id
                row["order"] = root.order
                if row["status"].startswith("unresolved"):
                    root.qc_flags.append("surface_reference_hierarchy_unresolved")
            resolved[root.root_id] = (root.order, root.points)
        ready_ids = {root.root_id for root in ready}
        pending = [root for root in pending if root.root_id not in ready_ids]
    return rows


def apply_surface_reference(reference, points, labels, primary, roots, *,
                            d_bar, mesh_context, excluded_mask, primary_top_reference,
                            gravity=(0., 0., -1.)):
    """Commit only the approved mask, preserving all other semantic owners."""
    before = np.asarray(labels, dtype=int)
    points = np.asarray(points, float)
    mesh_context.validate(points, mesh_context.triangles)
    excluded = np.asarray(excluded_mask, bool)
    if before.shape != reference.requested.shape or excluded.shape != before.shape:
        raise ValueError("Surface reference and exclusions must match the final vertices")
    if np.any(before > len(roots)):
        raise ValueError("Automatic labels have no root identity")
    working = deepcopy(roots)
    owners = reference.manifest["owners"]
    k = len(owners)
    scores = np.zeros((k, len(roots)), dtype=int)
    for i in range(k):
        values = before[(reference.anchors == i + 1) & ~excluded]
        counts = np.bincount(values[values > 0], minlength=len(roots) + 1)
        scores[i] = counts[1:]
    assignment = dict(zip(*linear_sum_assignment(-scores))) if len(roots) else {}
    mapping = {}
    match_rows = []
    for i, owner in enumerate(owners):
        j = assignment.get(i)
        count = int(np.count_nonzero((reference.anchors == i + 1) & ~excluded))
        shared = int(scores[i, j]) if j is not None else 0
        if count == 0:
            raise ValueError("A reference anchor is entirely excluded by the immutable collar")
        # At least half the stable anchor must support the same automatic
        # identity. Otherwise create a new surface-supported identity instead
        # of stealing the name of an unrelated root.
        if j is not None and shared >= .5 * count:
            label = j + 1
            mode = "frozen_anchor_overlap"
        else:
            rid = "root-ref-" + hashlib.sha256((reference.data_sha256 + str(i)).encode()).hexdigest()[:12]
            if any(r.root_id == rid for r in working):
                raise ValueError("Reference-generated root identity already exists")
            mask = (reference.requested == i + 1) & ~excluded
            root = RootPath(rid, points[np.flatnonzero(mask)[:1]].copy())
            root.qc_flags.append("surface_reference_new_identity")
            working.append(root)
            label = len(working)
            mode = "new_identity_from_reviewed_surface"
        mapping[i + 1] = label
        match_rows.append({"reference_root_id": owner["reference_root_id"],
                           "root_id": working[label - 1].root_id, "mode": mode,
                           "anchor_vertices": count, "overlap_vertices": shared})
    requested = reference.requested
    scope = requested != KEEP
    proposal = before.copy()
    negative = scope & (requested <= 0) & ~excluded
    proposal[negative] = requested[negative]
    for index, label in mapping.items():
        proposal[(requested == index) & ~excluded] = label
    # Excluded vertices (including nodules) preserve their original nonroot
    # sentinel. The pipeline separately enforces above-collar == -1.
    proposal[excluded] = before[excluded]
    boundary_changes = []
    edges = mesh_context.support_edges(d_bar)
    target_labels = set(mapping.values())
    for label in sorted(set(before[scope & (before > 0)]) - target_labels):
        original = int(np.count_nonzero(before == label))
        remain = np.flatnonzero(proposal == label)
        # A small one-ring contact remnant is not an independent body. Only
        # explicitly reviewable boundary vertices may differ from the edit.
        if not (0 < len(remain) <= 8 and len(remain) <= .05 * original
                and np.all(reference.boundary[remain]) and not np.any(excluded[remain])):
            continue
        touch = edges[np.isin(edges, remain).any(axis=1)]
        neighboring = proposal[touch.ravel()]
        neighbors, counts = np.unique(neighboring[np.isin(neighboring, list(target_labels))], return_counts=True)
        if not len(neighbors):
            continue
        ranked = np.argsort(-counts, kind="stable")
        if len(ranked) > 1 and counts[ranked[0]] < 2 * counts[ranked[1]]:
            continue
        recipient = int(neighbors[ranked[0]])
        adjacent = np.unique(touch[np.any(proposal[touch] == recipient, axis=1)])
        if not np.all(np.isin(remain, adjacent)):
            continue
        proposal[remain] = recipient
        boundary_changes.append({"old_root_id": roots[label - 1].root_id,
                                 "root_id": working[recipient - 1].root_id,
                                 "vertices": remain.tolist(), "reason": "small_native_boundary_remnant"})
    changed = before != proposal
    permitted = scope | reference.boundary
    if np.any(changed & ~permitted):
        raise AssertionError("Reference correction escaped its reviewed scope")
    difference_rows = []
    tolerance = float(reference.manifest.get("maximum_difference_fraction", .02))
    for index, label in mapping.items():
        expected = requested == index
        missed = expected & (proposal != label)
        added = (proposal == label) & ~expected
        difference = int(missed.sum() + added.sum())
        fraction = difference / int(expected.sum())
        if fraction > tolerance:
            raise ValueError(f"Reference differences exceed the bounded tolerance for {owners[index-1]['reference_root_id']}: {fraction:.3%}")
        difference_rows.append({"reference_root_id": owners[index-1]["reference_root_id"],
                                "requested_vertices": int(expected.sum()),
                                "excluded_vertices": int((expected & excluded).sum()),
                                "difference_vertices": difference, "difference_fraction": fraction})
    affected = {working[label - 1].root_id for label in np.unique(np.r_[before[changed], proposal[changed]]) if label > 0}
    affected.update(working[label - 1].root_id for label in mapping.values())
    removed = set()
    while True:
        newly = {r.root_id for i, r in enumerate(working, 1)
                 if (r.root_id in affected or r.parent_id in removed)
                 and not np.any(proposal == i) and r.root_id not in removed}
        if not newly:
            break
        removed.update(newly)
    remap = np.zeros(len(working) + 1, int)
    retained = []
    for old_label, root in enumerate(working, 1):
        if root.root_id not in removed:
            retained.append(root)
            remap[old_label] = len(retained)
    result = proposal.copy()
    result[proposal > 0] = remap[proposal[proposal > 0]]
    hierarchy = _repair_affected_hierarchy(points, result, primary, retained, affected, removed,
                                           working, mesh_context, d_bar, primary_top_reference,
                                           np.asarray(gravity, float))
    report = {"schema": SCHEMA, "status": "applied", "manifest_sha256": reference.manifest_sha256,
              "data_sha256": reference.data_sha256,
              "reference_policy": "strong_scoped_surface_constraints; no edited hierarchy or centerlines",
              "changed_vertex_count": int(changed.sum()), "outside_scope_changed_vertices": 0,
              "scope_vertices": int(permitted.sum()), "matches": match_rows,
              "maximum_difference_fraction": tolerance, "reference_differences": difference_rows,
              "boundary_adjustments": boundary_changes, "retired_empty_root_ids": sorted(removed),
              "affected_root_ids": sorted(affected - removed), "hierarchy_decisions": hierarchy,
              "before_labels_sha256": array_sha256(before, "<i8"),
              "final_labels_sha256": array_sha256(result, "<i8"),
              "before_root_ids": [r.root_id for r in roots],
              "final_root_ids": [r.root_id for r in retained]}
    audit = {"before_labels": before.copy(), "final_labels": result.copy(),
             "requested": requested.copy(), "scope": permitted, "excluded": excluded.copy(),
             "changed_vertices": np.flatnonzero(changed)}
    return result, retained, report, audit
