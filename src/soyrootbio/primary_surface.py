"""Recognize lateral labels that track the primary wall without an exposed tube."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from .centerline import _primary_section, _plane_basis
from .geometry import resample_polyline, tangent_vectors
from .mesh_geometry import MeshGeometryContext
from .surface_patches import _segment_radius_profile


def _unsafe_native_vertices(faces, count):
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    unique, uses = np.unique(edges, axis=0, return_counts=True)
    unsafe = np.zeros(count, bool)
    unsafe[np.unique(unique[uses != 2])] = True
    return unsafe


def primary_wall_evidence(points, labels, primary, *, d_bar, triangles, excluded_mask=None,
                          mesh_context=None):
    """Fit transverse rings to frozen native geometry, irrespective of labels.

    The supplied primary axis is a local prior, not an expanded cylinder.
    Rejected, one-sided, open, and nonmanifold sections supply no ownership
    evidence. No surface or centerline geometry is generated.
    """
    p = np.asarray(points, float)
    labels = np.asarray(labels)
    excluded = np.zeros(len(p), bool) if excluded_mask is None else np.asarray(excluded_mask, bool)
    context = mesh_context or MeshGeometryContext.build(p, triangles)
    context.validate(p, triangles)
    report = {"policy": "native-primary-wall-transverse-evidence-v1", "accepted_section_count": 0}
    wall = np.zeros(len(p), bool)
    radius_at_point = np.full(len(p), d_bar)
    if triangles is None or not len(triangles) or np.count_nonzero((labels == 0) & ~excluded) < 20:
        return wall, radius_at_point, report
    line = resample_polyline(primary, 4 * d_bar)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    ps, pr = _segment_radius_profile(p[(labels == 0) & ~excluded], primary, d_bar)
    radius = np.interp(arc, ps, pr)
    width = np.maximum(3 * d_bar, 2 * radius)
    tangents = np.column_stack([np.interp(np.minimum(arc[-1], arc + width), arc, line[:, i])
                               - np.interp(np.maximum(0, arc - width), arc, line[:, i]) for i in range(3)])
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1)[:, None], 1e-12)
    assigned = cKDTree(line).query(p)[1]
    # An unresolved seam is not evidence for a closed tubular wall.
    faces = np.asarray(triangles)
    bad = _unsafe_native_vertices(faces, len(p))
    centers = line.copy()
    measured = radius.copy()
    accepted = np.zeros(len(line), bool)
    for i in range(len(line)):
        section = _primary_section(p, context.vertex_area_weights, context.point_tree,
                                   line, tangents, arc, assigned, radius, i, d_bar, 4 * d_bar)
        if not section["accepted"] or section.get("thin_support") or not section.get("prior_inside"):
            continue
        near = context.point_tree.query_ball_point(line[i], max(3 * radius[i], 6 * d_bar))
        near = np.asarray(near, int)
        slab = near[(np.abs((p[near] - line[i]) @ tangents[i]) <= section["half_width"])
                    & ~excluded[near]]
        if len(slab) < 16 or np.any(bad[slab]):
            continue
        transverse = (p[slab] - section["candidate"]) @ section["basis"].T
        radial = np.linalg.norm(transverse, axis=1)
        r = float(np.median(radial))
        # A compressed primary can be elliptical or flat-sided. A circular
        # radius test would exclude its broad wall and manufacture laterals.
        # Use the measured transverse envelope; never expand a radius sphere.
        envelope = ConvexHull(section["envelope"])
        members = np.flatnonzero((assigned == i) & ~excluded & ~bad)
        uv = (p[members] - line[i]) @ section["basis"].T
        signed = np.max(uv @ envelope.equations[:, :2].T + envelope.equations[:, 2], axis=1)
        wall[members] = np.abs(signed) <= 1.5 * d_bar
        centers[i] = section["candidate"]
        measured[i] = r
        accepted[i] = True
    delta = p - centers[assigned]
    axial = np.einsum("ij,ij->i", delta, tangents[assigned])
    radius_at_point = measured[assigned]
    wall &= np.abs(axial) <= 4 * d_bar
    report["accepted_section_count"] = int(accepted.sum())
    report["wall_vertex_count"] = int(wall.sum())
    return wall, radius_at_point, report


def reconcile_primary_surface_tracks(points, labels, primary, roots, *, d_bar, triangles,
                                     excluded_mask=None, mesh_context=None):
    """Return primary-wall ownership proposals without consuming exposed bodies."""
    points = np.asarray(points, float)
    before = np.asarray(labels, int)
    result = before.copy()
    context = mesh_context or MeshGeometryContext.build(points, triangles)
    wall, radius, report = primary_wall_evidence(
        points, before, primary, d_bar=d_bar, triangles=triangles,
        excluded_mask=excluded_mask, mesh_context=context,
    )
    report.update(roots=[], changed_vertex_count=0, retired_root_ids=[], label_mapping={},
                  confirmed_primary_vertex_indices=[])
    if triangles is None or not len(triangles):
        return result, roots, report
    excluded = np.zeros(len(points), bool) if excluded_mask is None else np.asarray(excluded_mask, bool)
    edges = context.bounded_edges(d_bar)
    # Normals certify continuation of the same wall across a label seam.
    faces = np.asarray(triangles)
    face_normal = np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                           points[faces[:, 2]] - points[faces[:, 0]])
    normals = np.zeros_like(points)
    for column in range(3):
        np.add.at(normals, faces[:, column], face_normal)
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    smooth = np.einsum("ij,ij->i", normals[edges[:, 0]], normals[edges[:, 1]]) >= np.cos(np.deg2rad(30))
    unsafe = _unsafe_native_vertices(faces, len(points))
    edges = edges[smooth & ~(excluded | unsafe)[edges].any(axis=1)]
    eligible = (before == 0) | ((before < 0) & wall)
    protected = np.zeros(len(points), bool)
    candidates = {}
    for label, root in enumerate(roots, 1):
        owned = np.flatnonzero(before == label)
        # A distal continuation must not dilute evidence for a separate wall
        # patch. Each native donor component is checked below; the connected
        # exposed tube remains protected independently of total root size.
        if len(owned) < 12 or np.count_nonzero(wall[owned]) < 8:
            continue
        body = _tubular_body_witnesses(points[owned], root.points, d_bar, normals[owned])
        protected[owned[body]] = True
        eligible[owned[~body]] = True
        candidates[label] = owned
        report["roots"].append({"root_id": str(root.root_id), "label": label,
                               "owned": len(owned), "wall_fraction": float(wall[owned].mean()),
                               "protected_tube_vertices": int(body.sum())})
    # Frozen, simultaneous native propagation into smooth wall patches. The
    # geodesic budget is local primary radius, never an unbounded label flood.
    safe = edges[eligible[edges].all(axis=1)]
    length = np.linalg.norm(points[safe[:, 0]] - points[safe[:, 1]], axis=1)
    cost = length / np.maximum(.5 * (radius[safe[:, 0]] + radius[safe[:, 1]]), d_bar)
    graph = coo_matrix((np.r_[cost, cost], (np.r_[safe[:, 0], safe[:, 1]],
                       np.r_[safe[:, 1], safe[:, 0]])), shape=(len(points), len(points))).tocsr()
    seeds = np.flatnonzero((before == 0) & ~excluded)
    distance = dijkstra(graph, directed=False, indices=seeds, min_only=True, limit=3.) if len(seeds) else np.full(len(points), np.inf)
    proposal = eligible & np.isfinite(distance) & ~excluded & ~unsafe & (before != 0)
    # Require primary section evidence for every donor component. Smoothness
    # alone must not transfer a nearby surface of a different branch.
    ownership = context.ownership(before)
    component = ownership.components(context.bounded_edges(d_bar))
    for row in report["roots"]:
        label = row["label"]
        owned = candidates[label]
        for part in np.unique(component[owned]):
            members = owned[component[owned] == part]
            if np.count_nonzero(wall[members]) < min(8, len(members)):
                proposal[members] = False
        # Preserve connected exposed tubes. Tiny unsupported edge fragments
        # can remain QC, but must not roll back an entire flat parent patch.
        retained = owned[~proposal[owned]]
        if np.any(protected[retained]):
            from .primary_contact import _preserves_components
            if not _preserves_components(len(points), ownership.edges(label, context.bounded_edges(d_bar)),
                                         owned[proposal[owned]], retained[protected[retained]]):
                proposal[owned] = False
                row["status"] = "unresolved_would_split_exposed_body"
    # A donor rollback must not strand a different proposed recipient. Check
    # the actual simultaneous result against the original primary anchors.
    primary_after = (before == 0) | proposal
    recipient_edges = edges[primary_after[edges].all(axis=1)]
    recipient_graph = coo_matrix((np.ones(2 * len(recipient_edges)),
        (np.r_[recipient_edges[:, 0], recipient_edges[:, 1]],
         np.r_[recipient_edges[:, 1], recipient_edges[:, 0]])),
        shape=(len(points), len(points))).tocsr()
    _, recipient_component = connected_components(recipient_graph, directed=False)
    proposal &= np.isin(recipient_component, np.unique(recipient_component[seeds]))
    for row in report["roots"]:
        owned = candidates[row["label"]]
        row["changed_vertex_count"] = int(proposal[owned].sum())
        row.setdefault("status", "primary_wall_reassigned" if row["changed_vertex_count"] else "retained")
        row["changed_vertex_indices"] = owned[proposal[owned]].tolist()
    result[proposal] = 0
    # Retire only empty leaf hypotheses, after surface evidence has explained
    # every owned vertex. Never remove a supported body or its descendants.
    retained_roots = list(roots)
    for root in reversed(roots):
        label = roots.index(root) + 1
        if (label in candidates and not np.any(result == label)
                and not any(r.parent_id == root.root_id for r in retained_roots)):
            retained_roots = [r for r in retained_roots if r is not root]
            report["retired_root_ids"].append(str(root.root_id))
    mapping = {0: 0, -1: -1, -2: -2}
    by_id = {r.root_id: i for i, r in enumerate(retained_roots, 1)}
    for label, root in enumerate(roots, 1):
        mapping[label] = by_id.get(root.root_id, 0)
    remapped = result.copy()
    for old, new in mapping.items():
        if old != new:
            remapped[result == old] = new
    report["changed_vertex_count"] = int(proposal.sum())
    report["confirmed_primary_vertex_indices"] = np.flatnonzero(proposal).tolist()
    report["label_mapping"] = mapping
    return remapped, retained_roots, report


def _tubular_body_witnesses(owned, prior, spacing, normals):
    """Protect sustained two-dimensional transverse support, including stubs."""
    protected = np.zeros(len(owned), bool)
    if len(owned) < 12 or len(prior) < 2:
        return protected
    tree = cKDTree(owned)
    line = resample_polyline(prior, 4 * spacing)
    tangents = tangent_vectors(line)
    good = np.zeros(len(line), bool)
    radii = np.full(len(line), 2 * spacing)
    for i, (station, tangent) in enumerate(zip(line, tangents)):
        radii[i] = max(float(np.median(tree.query(station, k=min(12, len(owned)))[0])), 2 * spacing)
        indices = np.asarray(tree.query_ball_point(station, 3 * radii[i] + 2 * spacing), int)
        local = owned[indices] - station
        indices = indices[np.abs(local @ tangent) <= 2 * spacing]
        slab = owned[indices] - station
        if len(slab) < 12:
            continue
        uv = slab @ _plane_basis(tangent).T
        eig = np.linalg.eigvalsh(np.cov(uv.T))
        angles = np.sort(np.arctan2(uv[:, 1], uv[:, 0]))
        coverage = 1 - np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max() / (2 * np.pi)
        normal_uv = normals[indices] @ _plane_basis(tangent).T
        normal_angles = np.sort(np.arctan2(normal_uv[:, 1], normal_uv[:, 0]))
        normal_coverage = 1 - np.diff(np.r_[normal_angles, normal_angles[0] + 2 * np.pi]).max() / (2 * np.pi)
        radial_distance = np.linalg.norm(uv, axis=1)
        radial_spread = np.quantile(radial_distance, .9) - np.quantile(radial_distance, .1)
        good[i] = (eig[0] >= .2 * eig[1] and coverage >= .6 and normal_coverage >= .6
                   and radial_spread <= .6 * np.median(radial_distance) + spacing
                   and np.median(np.abs(normals[indices] @ tangent)) <= .5)
    # A single accidental surrounding cross-section is not an exposed body.
    for run in np.split(np.flatnonzero(good), np.flatnonzero(np.diff(np.flatnonzero(good)) > 1) + 1):
        if len(run) < 3:
            continue
        for i in run:
            local = np.asarray(tree.query_ball_point(line[i], 2 * radii[i] + 2 * spacing), int)
            axial = np.abs((owned[local] - line[i]) @ tangents[i])
            protected[local[axial <= 4 * spacing]] = True
    return protected
