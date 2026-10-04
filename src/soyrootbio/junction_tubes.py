"""Conservative parent-to-child ownership from native exposed tube evidence.

Section and normal tolerances are dimensionless, scaled by local radii and
mesh spacing. They are conservative working bounds, not biological cutoffs.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .centerline import _axis, _fit_body, _plane_basis
from .junction_transections import _frame
from .mesh_geometry import MeshGeometryContext
from .surface_patches import _segment_radius_profile, _polyline_projection_distance_and_arc as project
from .types import RootPath


def reconcile_parent_owned_tubes(
    points: np.ndarray, labels: np.ndarray, primary_path: np.ndarray,
    roots: list[RootPath], *, d_bar: float, triangles: np.ndarray | None,
    excluded_mask: np.ndarray | None = None,
    mesh_context: MeshGeometryContext | None = None,
) -> tuple[np.ndarray, dict]:
    """Reclaim a measured child tube from its direct parent at any order.

    Every candidate uses the same frozen ownership. Recipient support must
    reach the existing child body on native edges. No path, topology, negative
    assignment or geometry is changed. Ambiguous claims keep their owner.
    """
    points, before = np.asarray(points, float), np.asarray(labels, int)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must contain finite XYZ coordinates")
    if before.shape != (len(points),) or np.any(before >= len(roots) + 1):
        raise ValueError("labels must reference roots or negative assignment states")
    if not np.isfinite(d_bar) or d_bar <= 0:
        raise ValueError("d_bar must be positive and finite")
    report = dict(policy="native-child-tube-continuation-v1", status="evaluated",
                  evidence="frozen labels, native edges and measured transverse surface normals",
                  transferred_vertex_count=0, ambiguous_vertex_count=0, junctions=[],
                  changed_vertex_indices=[])
    if triangles is None or not len(triangles):
        report['status'] = 'insufficient_native_mesh'
        return before.copy(), report
    context = mesh_context or MeshGeometryContext.build(points, triangles)
    context.validate(points, triangles)
    excluded = np.zeros(len(points), bool) if excluded_mask is None else np.array(excluded_mask, bool, copy=True)
    if excluded.shape != before.shape:
        raise ValueError("excluded_mask must match labels")
    unsafe = np.zeros(len(points), bool)
    unsafe[np.unique(context.edges[context.edge_incidence != 2])] = True
    excluded |= unsafe | (before < 0)
    report['unsafe_native_vertex_count'] = int(unsafe.sum())
    edges = context.support_edges(d_bar)
    safe_edges = edges[~excluded[edges].any(axis=1)]
    ownership = context.ownership(before)
    components = ownership.components(safe_edges, active=~excluded)
    normals = vertex_normals(points, np.asarray(triangles, int))
    lookup = {'primary': 0, **{r.root_id: i for i, r in enumerate(roots, 1)}}
    paths = [np.asarray(primary_path, float)] + [np.asarray(r.points, float) for r in roots]
    proposals = {}
    parent_profiles = {}
    parent_arcs = {}
    best, second = np.full(len(points), np.inf), np.full(len(points), np.inf)
    winner = np.full(len(points), -1, int)
    for label in sorted(range(1, len(roots) + 1), key=lambda k: roots[k-1].root_id):
        root = roots[label-1]
        parent = lookup.get(root.parent_id)
        if parent is None:
            continue
        if parent not in parent_profiles:
            donor = (before == parent) & ~excluded
            parent_profiles[parent] = _segment_radius_profile(points[donor], paths[parent], d_bar)
            parent_arcs[parent] = project(points[donor], paths[parent])[1]
        ids, scores, row = child_tube_witnesses(points, before, parent, label,
            paths[parent], paths[label][root.body_start_index:], spacing=d_bar,
            context=context, normals=normals, excluded=excluded,
            parent_profile=parent_profiles[parent], parent_surface_arc=parent_arcs[parent])
        row.update(root_id=root.root_id, parent_id=root.parent_id, root_order=root.order)
        choose = (before[ids] == parent) & ~excluded[ids]
        ids, scores = ids[choose], scores[choose]
        row['proposed_vertex_count'] = int(len(ids))
        report['junctions'].append(row)
        proposals[label] = row
        better = scores < best[ids]
        second[ids] = np.where(better, best[ids], np.minimum(second[ids], scores))
        best[ids] = np.minimum(best[ids], scores)
        winner[ids[better]] = label
    ambiguous = np.isfinite(second) & ((second - np.where(np.isfinite(best), best, 0)) < .15)
    report['ambiguous_vertex_count'] = int(ambiguous.sum())
    winner[ambiguous] = -1
    result = before.copy()
    result[winner > 0] = winner[winner > 0]

    # Arbitration may remove the connecting strip. Recheck against the
    # original largest child body, never against a newly grown recipient.
    def retain_connected_claims():
        for label, row in proposals.items():
            moved = (before != result) & (result == label)
            if not np.any(moved):
                continue
            child = np.flatnonzero((before == label) & ~excluded)
            parts, counts = np.unique(components[child], return_counts=True)
            anchors = child[components[child] == parts[np.argmax(counts)]]
            domain = ((result == label) & ~excluded)
            e = safe_edges[domain[safe_edges].all(axis=1)]
            graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                               shape=(len(points), len(points))).tocsr()
            groups = connected_components(graph, directed=False)[1]
            unreachable = moved & ~np.isin(groups, np.unique(groups[anchors[domain[anchors]]]))
            result[unreachable] = before[unreachable]

    retain_connected_claims()
    # A measured ring may contain an incomplete ownership strip. Never
    # leave a detached parent island, remove an original parent component,
    # or fill an unmeasured surface to make a proposed cut appear valid.
    original = context.ownership(before).components(context.edges)
    for _ in range(len(roots) + 1):
        # Restoring a rejected primary-to-O1 cut can expose primary beside
        # an O2 proposal. Recheck after every rollback, not only once against
        # the provisional simultaneous winners.
        for label, row in proposals.items():
            moved = (before != result) & (result == label)
            if row['root_order'] >= 2 and np.any(moved):
                boundary = context.edges[(result[context.edges[:, 0]] == label) ^
                                         (result[context.edges[:, 1]] == label)]
                if np.any(result[boundary] == 0):
                    result[moved] = before[moved]
                    row['status'] = 'unresolved_would_contact_primary'
        retain_connected_claims()
        changed = before != result
        if not np.any(changed):
            break
        current = context.ownership(result).components(context.edges)
        rejected = set()
        for parent in np.unique(before[changed]):
            old = np.flatnonzero(before == parent)
            for part in np.unique(original[old]):
                members = old[original[old] == part]
                retained = members[result[members] == parent]
                if not len(retained):
                    rejected.update(int(v) for v in result[members[changed[members]]])
                    continue
                parts, counts = np.unique(current[retained], return_counts=True)
                for fragment in parts[parts != parts[np.argmax(counts)]]:
                    island = np.zeros(len(points), bool)
                    island[retained[current[retained] == fragment]] = True
                    boundary = context.edges[island[context.edges[:, 0]] ^ island[context.edges[:, 1]]]
                    adjacent = np.unique(boundary)
                    adjacent = adjacent[changed[adjacent] & (before[adjacent] == parent)]
                    rejected.update(int(v) for v in result[adjacent])
        if not rejected:
            break
        for label in rejected:
            moved = changed & (result == label)
            result[moved] = before[moved]
            proposals[label]['status'] = 'unresolved_would_split_parent'
        retain_connected_claims()
    for label, row in proposals.items():
        row['transferred_vertex_count'] = int(np.sum((before != result) & (result == label)))
        if row['transferred_vertex_count']:
            row['status'] = 'reassigned_measured_child_tube'
        elif row['proposed_vertex_count'] and not row['status'].startswith('unresolved_'):
            row['status'] = 'unresolved_competition_or_connectivity'
        elif row['status'] == 'supported':
            row['status'] = 'no_parent_owned_tube_detected'
    report['transferred_vertex_count'] = int(np.sum(before != result))
    report['changed_vertex_indices'] = np.flatnonzero(before != result).tolist()
    report['unresolved_junction_count'] = sum(row['proposed_vertex_count'] > 0 and
        row['status'].startswith('unresolved_') for row in report['junctions'])
    return result, report


def vertex_normals(points, triangles):
    face_normal = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                           points[triangles[:, 2]] - points[triangles[:, 0]])
    normal = np.zeros_like(points)
    for column in range(3):
        np.add.at(normal, triangles[:, column], face_normal)
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)
    return normal


def _parent_core_mask(points, normals, donor, candidates, parent, spacing,
                      donor_arc, fallback):
    """Protect the measured parent envelope, including flattened sections.

    The tube proposal cannot certify its own parent envelope: remove its
    vertices from the wall witnesses. A broad angular spread of parent-wall
    normals is required. Missing sections retain the conservative radial
    guard; they are never completed with invented points.
    """
    _, arc = project(points[candidates], parent)
    parent_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(parent, axis=0), axis=1))]
    bins = np.round(arc / (2 * spacing)).astype(int)
    result = fallback.copy()
    permitted = ~np.isin(donor, candidates)
    measured_sections = 0
    for index in np.unique(bins):
        station = float(index * 2 * spacing)
        center, direction = _frame(parent, parent_arc, np.array([station]),
                                   max(4 * spacing, .02 * parent_arc[-1]))
        ids = donor[permitted & (np.abs(donor_arc - station) <= 2 * spacing)]
        ids = ids[np.abs(normals[ids] @ direction[0]) < .75]
        if len(ids) < 12:
            continue
        basis = _plane_basis(direction[0])
        uv = (points[ids] - center[0]) @ basis.T
        transverse_normal = normals[ids] @ basis.T
        angles = np.sort(np.arctan2(transverse_normal[:, 1], transverse_normal[:, 0]))
        coverage = 1 - np.diff(np.r_[angles, angles[0] + 2*np.pi]).max() / (2*np.pi)
        if coverage < .6:
            continue
        try:
            hull = ConvexHull(uv)
        except QhullError:
            continue
        query = (points[candidates[bins == index]] - center[0]) @ basis.T
        signed = (query @ hull.equations[:, :2].T + hull.equations[:, 2]).max(axis=1)
        result[bins == index] = signed <= .5 * spacing
        measured_sections += 1
    return result, measured_sections


def _measure_ring(points, normals, union, tree, center, direction, radius, spacing):
    ids = union[np.asarray(tree.query_ball_point(center, 2.2*radius+3*spacing), int)]
    delta = points[ids] - center
    surface_ids = ids[np.abs(delta @ direction) <= 2*spacing]
    ids = ids[(np.abs(delta @ direction) <= 2*spacing) & (np.abs(normals[ids] @ direction) <= .75)]
    if len(ids) < 12:
        return None
    basis = _plane_basis(direction)
    uv = (points[ids] - center) @ basis.T
    n = normals[ids] @ basis.T
    n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-12)
    transverse = np.column_stack((-n[:, 1], n[:, 0]))
    target = np.einsum('ij,ij->i', transverse, uv)
    keep = np.ones(len(ids), bool)
    for _ in range(3):
        shift, *_ = np.linalg.lstsq(transverse[keep], target[keep], rcond=None)
        radial = np.linalg.norm(uv - shift, axis=1)
        measured = float(np.median(radial[keep]))
        keep = np.abs(radial - measured) <= max(.35 * measured, spacing)
        if np.count_nonzero(keep) < 12:
            return None
    def coverage(v):
        angles = np.sort(np.arctan2(v[:, 1], v[:, 0]))
        return 1 - np.diff(np.r_[angles, angles[0] + 2*np.pi]).max() / (2*np.pi)
    if (coverage(uv[keep] - shift) < .68 or coverage(n[keep]) < .68
            or np.linalg.norm(shift) > max(.8 * radius, 2 * spacing)
            or abs(measured - radius) > max(.25 * radius, spacing)):
        return None
    actual = center + shift @ basis
    # Ring normals estimate the local axis independently of the trace. A
    # unique small eigenvalue is required; a flat patch/sphere is ambiguous.
    eig, vec = np.linalg.eigh(normals[ids[keep]].T @ normals[ids[keep]])
    axis = vec[:, 0]
    if np.dot(axis, direction) < 0:
        axis = -axis
    if (eig[1] < .18 * eig[2] or eig[0] > .6 * eig[1]
            or np.dot(axis, direction) < np.cos(np.deg2rad(35))):
        return None
    axis = (axis + direction) / np.linalg.norm(axis + direction)
    delta = points[surface_ids] - actual
    axial = delta @ direction
    radial = np.linalg.norm(delta - axial[:, None] * direction, axis=1)
    # Normals certify the ring collectively. An individual noisy normal
    # must not leave a one-vertex parent island inside its measured surface.
    surface = np.abs(radial-measured) <= max(.45*measured, 1.5*spacing)
    return (actual, axis, measured, surface_ids[surface],
            np.abs(radial[surface] - measured) / (measured + spacing))


def child_tube_witnesses(points, labels, parent_label, child_label, parent, child, *,
                          spacing, context, normals, excluded, parent_profile=None,
                          parent_surface_arc=None):
    """Measure sustained closed transverse rings without trusting the axis center.

    The automatic path supplies an orientation hint only. Native normals fit
    the section center and independently require circumferential coverage.
    A sphere, flat parent wall or open seam cannot supply a sustained tube.
    """
    owned = np.flatnonzero((labels == child_label) & ~excluded)
    donor = np.flatnonzero((labels == parent_label) & ~excluded)
    record = dict(root_label=child_label, parent_label=parent_label, status='insufficient_support',
                  accepted_sections=0, supported_vertices=0)
    if len(owned) < 12 or len(donor) < 12 or len(child) < 2 or len(parent) < 2:
        return np.empty(0, int), np.empty(0), record
    component = context.ownership(labels).components(context.support_edges(spacing), active=~excluded)
    parts, counts = np.unique(component[owned], return_counts=True)
    owned = owned[component[owned] == parts[np.argmax(counts)]]
    if len(owned) < 12:
        record['status'] = 'insufficient_connected_child_support'
        return np.empty(0, int), np.empty(0), record
    all_owned = context.ownership(labels).vertices(child_label)
    local_ids = np.searchsorted(all_owned, owned)
    graph = context.ownership(labels).local_graph(child_label, context.support_edges(spacing))
    initial = _axis(points[owned], graph[local_ids][:, local_ids], child[0], spacing)
    measured_body, _ = _fit_body(points[owned], initial, spacing)
    if len(measured_body) < 2:
        return np.empty(0, int), np.empty(0), record
    child = measured_body
    record['query_axis'] = 'temporary native exposed-body fit; no attachment or exported path changed'
    contact = context.ownership(labels).boundary_edges(parent_label, child_label, context.support_edges(spacing))
    contact = contact[~excluded[contact].any(axis=1)]
    contact = contact[np.isin(contact, owned).any(axis=1)]
    if not len(contact):
        record['status'] = 'no_native_parent_child_contact'
        return np.empty(0, int), np.empty(0), record
    parent_side = np.unique(contact[labels[contact] == parent_label])
    child_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(child, axis=0), axis=1))]
    cs, cr = _segment_radius_profile(points[owned], child, spacing)
    ps, pr = (_segment_radius_profile(points[donor], parent, spacing)
              if parent_profile is None else parent_profile)
    _, contact_parent_arc = project(points[parent_side], parent)
    radius_parent = max(float(np.median(np.interp(contact_parent_arc, ps, pr))), 2*spacing)
    # A fitted exposed body often starts beyond a parent-owned sleeve. Its
    # internal attachment may bend toward the wrong parent station. Extend
    # the exposed-body direction solely as a query prior; every reclaimed
    # surface still needs observed closed-ring and native-connectivity proof.
    _, start_tangent = _frame(child, child_arc, np.array([0.]), max(6*spacing, 3*cr[0]))
    extension = max(4*radius_parent, 8*cr[0], 16*spacing)
    child = np.vstack((child[0]-extension*start_tangent[0], child))
    child_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(child, axis=0), axis=1))]
    cs, cr = _segment_radius_profile(points[owned], child, spacing)
    _, contact_child_arc = project(points[parent_side], child)
    end = min(child_arc[-1], float(np.min(contact_child_arc)) + max(8*radius_parent, 20*spacing))
    stations = np.linspace(0., end, max(2, int(np.ceil(end/(2*spacing)))+1))
    radii = np.interp(stations, cs, cr)
    centers, tangent = _frame(child, child_arc, stations, max(3*spacing, float(np.median(radii))))
    union = np.union1d(owned, donor)
    tree = cKDTree(points[union])
    sections = []
    good = np.zeros(len(stations), bool)
    measured = np.zeros(len(stations))
    for index, (station, direction, radius) in enumerate(zip(centers, tangent, radii)):
        local = union[np.asarray(tree.query_ball_point(station, max(3*radius, 8*spacing)), int)]
        delta = points[local] - station
        local = local[(np.abs(delta @ direction) <= 1.5*spacing)
                      & (np.abs(normals[local] @ direction) <= .6)]
        if len(local) < 12:
            sections.append(None)
            continue
        basis = _plane_basis(direction)
        uv = (points[local] - station) @ basis.T
        normal_uv = normals[local] @ basis.T
        normal_uv /= np.maximum(np.linalg.norm(normal_uv, axis=1)[:, None], 1e-12)
        transverse = np.column_stack((-normal_uv[:, 1], normal_uv[:, 0]))
        target = np.einsum('ij,ij->i', transverse, uv)
        center, *_ = np.linalg.lstsq(transverse, target, rcond=None)
        radial = np.linalg.norm(uv-center, axis=1)
        radius = float(np.median(radial))
        # Trim residual outliers once, then demand broad geometry AND normal
        # coverage. Recentring a partial wall cannot manufacture normal coverage.
        keep = np.abs(radial-radius) <= max(.4*radius, 1.5*spacing)
        if np.count_nonzero(keep) < 12:
            sections.append(None)
            continue
        center, *_ = np.linalg.lstsq(transverse[keep], target[keep], rcond=None)
        radial = np.linalg.norm(uv[keep]-center, axis=1)
        radius = float(np.median(radial))
        def coverage(vectors):
            angles = np.sort(np.arctan2(vectors[:, 1], vectors[:, 0]))
            return 1-np.max(np.diff(np.r_[angles, angles[0]+2*np.pi]))/(2*np.pi)
        spread = float(np.quantile(radial,.9)-np.quantile(radial,.1))
        okay = (coverage(uv[keep]-center) >= .72 and coverage(normal_uv[keep]) >= .72
                and spread <= .55*radius + .5*spacing
                and radius >= .75*spacing and radius <= 2*max(radii[index], spacing)
                and np.linalg.norm(center) <= 2*max(radii[index], spacing))
        if not okay:
            sections.append(None)
            continue
        actual_center = station + center @ basis
        _, pa = project(actual_center[None], parent)
        parc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(parent, axis=0), axis=1))]
        _, pt = _frame(parent, parc, pa, max(radius_parent, 3*spacing))
        # Parallel continuation is a fork/hierarchy question, not proof that
        # a separate small tube is emerging from this parent.
        if abs(np.dot(pt[0], direction)) > np.cos(np.deg2rad(20)):
            sections.append(None)
            continue
        good[index], measured[index] = True, radius
        sections.append((actual_center, direction, radius))
    supported = np.zeros(len(points), bool)
    residual = np.full(len(points), np.inf)
    # A lone circular cap is not a body witness. Require a sustained run
    # either in the exposed body or during the measured backwards extension.
    sustained = np.zeros(len(stations), bool)
    valid = np.flatnonzero(good)
    # An unmeasurable slab is not a missing mesh: allow one intervening
    # station, but never fill it. Native connectivity is checked separately.
    for run in np.split(valid, np.flatnonzero(np.diff(valid) > 2) + 1):
        if len(run) < 3:
            continue
        median_radius = float(np.median(measured[run]))
        if stations[run[-1]] - stations[run[0]] < max(4 * spacing, 2 * median_radius):
            continue
        if np.quantile(measured[run], .9) > 1.5 * max(np.quantile(measured[run], .1), spacing):
            continue
        sustained[run] = True
    for index in np.flatnonzero(good):
        center, direction, radius = sections[index]
        ring = _measure_ring(points, normals, union, tree, center, direction, radius, spacing)
        if ring is None or np.mean(labels[ring[3]] == child_label) < .75:
            continue
        center, direction, radius, _, _ = ring
        initial_radius = radius
        backward = []
        for _ in range(int(np.ceil(max(6*radius_parent, 16*spacing) / (2*spacing)))):
            measured_ring = _measure_ring(points, normals, union, tree,
                center - 2*spacing*direction, direction, radius, spacing)
            if measured_ring is None:
                break
            center, direction, radius, ids, score = measured_ring
            if radius > 2.25*initial_radius or radius < .5*initial_radius:
                break
            backward.append((ids, score))
        # Tapered or short exposed bodies may not supply a full distal run.
        # Three independently measured backwards sections are sufficient;
        # no unsupported section or absent surface is filled between them.
        if sustained[index] or len(backward) >= 3:
            for ids, score in backward:
                supported[ids] = True
                residual[ids] = np.minimum(residual[ids], score)
            record['backward_sections'] = len(backward)
            break
    valid = np.flatnonzero(good)
    record['good_section_count'] = int(len(valid))
    for run in np.split(valid, np.flatnonzero(np.diff(valid)>2)+1):
        if len(run) < 3:
            continue
        median_radius = float(np.median(measured[run]))
        if stations[run[-1]]-stations[run[0]] < max(4*spacing, 2*median_radius):
            continue
        if np.quantile(measured[run],.9) > 1.5*max(np.quantile(measured[run],.1), spacing):
            continue
        record['accepted_sections'] += len(run)
        for index in run:
            center, direction, radius = sections[index]
            local = union[np.asarray(tree.query_ball_point(center, 2*radius+3*spacing), int)]
            delta = points[local]-center
            axial = delta @ direction
            radial = np.linalg.norm(delta-axial[:,None]*direction, axis=1)
            keep = ((np.abs(axial) <= 2.5*spacing) & (np.abs(radial-radius) <= max(.4*radius, 1.5*spacing))
                    & (np.abs(normals[local] @ direction) <= .7))
            ids = local[keep]
            supported[ids] = True
            residual[ids] = np.minimum(residual[ids], np.abs(radial[keep]-radius)/(radius+spacing))
    # A closed surface inside an overlapping parent is not exposed child
    # ownership. Keep the parent core even when a query finds such a tube.
    parent_candidates = np.flatnonzero(supported & (labels == parent_label))
    distance, arc = project(points[parent_candidates], parent)
    wall = np.interp(arc, ps, pr)
    internal = distance <= .9 * wall + .25 * spacing
    if len(parent_candidates):
        donor_arc = (project(points[donor], parent)[1]
                     if parent_surface_arc is None else parent_surface_arc)
        internal, measured_parent_sections = _parent_core_mask(points, normals, donor,
            parent_candidates, parent, spacing, donor_arc, internal)
        record['measured_parent_sections'] = measured_parent_sections
    record['protected_parent_core_vertices'] = int(internal.sum())
    supported[parent_candidates[internal]] = False
    # The tube witness must reach an existing child body on actual mesh edges.
    edges = context.support_edges(spacing)
    edges = edges[supported[edges].all(axis=1)]
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(points),len(points))).tocsr()
    component = connected_components(graph, directed=False)[1]
    anchors = owned[supported[owned]]
    supported &= np.isin(component, np.unique(component[anchors])) if len(anchors) else False
    ids = np.flatnonzero(supported)
    record.update(status='supported' if len(ids) else 'unresolved_no_sustained_connected_tube',
                  supported_vertices=len(ids), parent_radius=radius_parent)
    return ids, residual[ids], record
