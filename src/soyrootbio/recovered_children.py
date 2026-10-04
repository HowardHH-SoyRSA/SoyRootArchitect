"""Native-component child discovery on confirmed root continuations."""
from collections import defaultdict
from copy import deepcopy

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from .lateral import (estimate_parent_radius_profile, find_lateral_starting_points,
    grow_lateral_candidates, is_parent_tracking_candidate, reduce_similar_paths,
    resume_lateral_tip_in_batches, select_non_overlapping_paths)
from .primary_surface import _unsafe_native_vertices
from .native_sections import recovered_wall_sections
from .geometry import is_above_primary_top
from .topology import validate_root_tree, _terminal_continuation_evidence


def confirmed_extension_parents(roots, late_parent_ids):
    """Include committed topology repairs, not merely proposed fork arms.

    Topology repair runs after per-order lateral tracing. Its swapped and
    joined continuations therefore need the same surface/descendant survey
    as arms recovered later from unassigned vertices. Use the surviving
    root's provenance so stable-ID changes do not lose the repaired parent.
    """
    markers = ('fork_long_arm_reconciled', 'terminal_continuation_joined',
               'contacted_sibling_continuation_joined', 'displaced_tip_continuation_joined')
    return {root.root_id: ([key for key in markers if
                           root.score_components.get(key, 0) > 0 or key in root.qc_flags]
                          + (['unassigned_fork_recovered'] if root.root_id in late_parent_ids else []))
            for root in roots
            if root.root_id in late_parent_ids or any(
                root.score_components.get(key, 0) > 0 or key in root.qc_flags for key in markers)}


def _free_tip_continuation(points, labels, parent, candidate, paths, *, label,
                           mesh_context, excluded, spacing):
    """Assess a free terminal tube using measured native transverse centres.

    The tracing endpoint can sit on one side of an incompletely labelled
    tube. Closed contours supply direction evidence without changing any
    assigned ownership. Ordinary radius, competing-arm and mesh checks still
    apply; only the tracing growth-cap prerequisite is inapplicable here.
    """
    if candidate.insertion_index != len(parent.points) - 1:
        return {'status': 'rejected', 'reason': 'attachment_not_at_parent_tip'}
    support = np.asarray(sorted(candidate.covered_indices), int)
    temporary = labels.copy()
    temporary[support] = label
    barrier = excluded | ~np.isin(temporary, [-1, label])
    _, _, pc, pm = recovered_wall_sections(points, temporary, parent.points, label,
                                           mesh_context, spacing, barrier)
    _, _, cc, cm = recovered_wall_sections(points, temporary, candidate.points, label,
                                           mesh_context, spacing, barrier)
    if pm.sum() < 8 or cm.sum() < 8:
        return {'status': 'rejected', 'reason': 'terminal_closed_contours_missing'}
    # Do not let dropping unmeasured contour stations bridge a missing neck.
    if (len(pm) - 1 - np.flatnonzero(pm)[-1] > 4 or np.flatnonzero(cm)[0] > 4):
        return {'status': 'rejected', 'reason': 'terminal_closed_contours_not_at_junction'}
    measured_parent, measured_child = deepcopy(parent), deepcopy(candidate)
    measured_parent.points = pc[pm]
    measured_parent.covered_indices = set(np.flatnonzero(labels == label).tolist())
    measured_parent.novel_support_indices = set(measured_parent.covered_indices)
    measured_child.points = np.vstack([measured_parent.points[-1], cc[cm]])
    measured_child.insertion_index = len(measured_parent.points) - 1
    measured_child.insertion_point = measured_parent.points[-1].copy()
    measured_child.score_components['novel_density_support'] = float(len(support))
    # Existing paths carry analysis indices; the ownership snapshot is the
    # full mesh. Native foreign-label barriers already exclude their support.
    others = [deepcopy(root) for root in paths if root is not parent and root is not candidate]
    for root in others:
        root.covered_indices = set()
        root.novel_support_indices = set()
    evidence = _terminal_continuation_evidence(measured_parent, measured_child,
        [measured_parent, measured_child, *others], spacing=spacing, support_points=points,
        mesh_points=points, mesh_triangles=mesh_context.triangles,
        mesh_excluded_mask=barrier, mesh_tree=mesh_context.point_tree,
        mesh_context=mesh_context, require_growth_cap=False)
    if evidence['status'] == 'accepted':
        # A terminal curve must not return to the established basal body.
        basal = parent.points[:-max(4, int(np.ceil(.02 * len(parent.points))))]
        if len(basal) and cKDTree(basal).query(candidate.points[-1])[0] <= max(
                4 * spacing, 2 * evidence['parent_radius']):
            evidence.update(status='rejected', reason='continuation_loops_into_parent')
    evidence.update(parent_measured_sections=int(pm.sum()), child_measured_sections=int(cm.sum()))
    return evidence


def fill_recovered_surfaces(points, labels, roots, parent_ids, *, mesh_context, d_bar, excluded_mask):
    """Fill only measured free wall connected locally to its frozen owner.

    Offshoots can share a free component with a hole in the parent wall. A
    whole-component fill or branch trace cannot distinguish the two. Closed
    transverse evidence and a bounded smooth native path separate the wall;
    competing claims and every already assigned/uncertain vertex are retained.
    """
    before = np.asarray(labels, int)
    result = before.copy()
    proposals = np.full(len(points), -1, int)
    conflicts = np.zeros(len(points), bool)
    report = {'changed_vertex_count': 0, 'roots': []}
    faces = mesh_context.triangles
    if faces is None or not len(faces) or not parent_ids:
        return result, report
    unsafe = _unsafe_native_vertices(faces, len(points)) | excluded_mask
    normals = np.zeros_like(points)
    face_normal = np.cross(points[faces[:,1]]-points[faces[:,0]], points[faces[:,2]]-points[faces[:,0]])
    for column in range(3):
        np.add.at(normals, faces[:,column], face_normal)
    normals /= np.maximum(np.linalg.norm(normals,axis=1)[:,None], 1e-12)
    edges = mesh_context.bounded_edges(d_bar)
    smooth = np.einsum('ij,ij->i',normals[edges[:,0]],normals[edges[:,1]]) >= np.cos(np.deg2rad(30))
    edges = edges[smooth & ~unsafe[edges].any(axis=1)]
    for label, root in enumerate(roots, 1):
        if root.root_id not in parent_ids:
            continue
        wall, radius, centers, measured = recovered_wall_sections(points, before, root.points,
            label, mesh_context, d_bar, unsafe | ((before != -1) & (before != label)))
        active = ((before == label) | ((before == -1) & wall)) & ~unsafe
        local = edges[active[edges].all(axis=1)]
        cost = np.linalg.norm(points[local[:,0]]-points[local[:,1]],axis=1) / np.maximum(
            .5*(radius[local[:,0]]+radius[local[:,1]]), d_bar)
        graph = coo_matrix((np.r_[cost,cost], (np.r_[local[:,0],local[:,1]],
            np.r_[local[:,1],local[:,0]])), shape=(len(points),len(points))).tocsr()
        seeds = np.flatnonzero((before == label) & ~unsafe)
        distance = (dijkstra(graph, directed=False, indices=seeds, min_only=True, limit=3.)
                    if len(seeds) else np.full(len(points), np.inf))
        claim = (before == -1) & active & np.isfinite(distance)
        conflicts |= claim & (proposals >= 0)
        proposals[claim] = label
        report['roots'].append({'root_id': root.root_id, 'accepted_section_count': int(measured.sum()),
                               'proposed_vertex_count': int(claim.sum())})
    take = (proposals >= 0) & ~conflicts
    result[take] = proposals[take]
    report['changed_vertex_count'] = int(take.sum())
    report['competing_vertex_count'] = int(conflicts.sum())
    return result, report


def recover_children_on_extensions(points, labels, primary, roots, *, parent_ids, d_bar,
    mesh_context, excluded_mask, analysis_to_mesh, primary_top_reference, gravity,
    max_root_order=3, max_paths=None, cooperate=None):
    """Use frozen native ownership; never search inside another assigned root.

    A free component needs exactly one eligible touching parent. Parameter
    variants compete for its support before any new root is committed. New
    children can expose further children up to the requested order.
    """
    parent_reasons = confirmed_extension_parents(roots, parent_ids)
    report = {'policy': 'native-recovered-descendants-v2', 'accepted_count': 0, 'decisions': [],
              'survey_parent_reasons': parent_reasons, 'terminal_continuations': []}
    if (not parent_reasons or mesh_context is None or mesh_context.triangles is None
            or not len(mesh_context.triangles)):
        return roots, np.asarray(labels).copy(), report
    p = np.asarray(points)
    mesh_context.validate(p, mesh_context.triangles)
    excluded = np.asarray(excluded_mask, bool)
    working = np.asarray(labels, int).copy()
    result = list(roots)
    focus = set(parent_reasons)
    edges = mesh_context.bounded_edges(d_bar)
    unsafe = _unsafe_native_vertices(mesh_context.triangles, len(p)) | excluded
    used_ids = {str(r.root_id) for r in roots}
    native_support = {}
    working, surface_report = fill_recovered_surfaces(p, working, result, focus,
        mesh_context=mesh_context, d_bar=d_bar, excluded_mask=excluded)
    changed_parents = set(parent_ids) | {row['root_id'] for row in surface_report['roots']
                                        if row['proposed_vertex_count'] > 0}
    report['surface_passes'] = [surface_report]
    # A terminal continuation consumes a survey pass but not a root order.
    for generation in range(2 * max_root_order):
        if max_paths is not None and len(result) >= max_paths:
            break
        eligible = {i: r for i, r in enumerate(result, 1)
                    if r.root_id in focus and r.order < max_root_order}
        if not eligible:
            break
        free = (working == -1) & ~excluded
        same = edges[free[edges].all(axis=1)]
        graph = coo_matrix((np.ones(len(same)), (same[:, 0], same[:, 1])),
                           shape=(len(p), len(p))).tocsr()
        _, component = connected_components(graph, directed=False)
        members_by_part = {}
        free_indices = np.flatnonzero(free)
        order = np.argsort(component[free_indices], kind='stable')
        ordered = free_indices[order]
        cuts = np.flatnonzero(np.diff(component[ordered])) + 1
        for members in np.split(ordered, cuts):
            if len(members) >= 30:
                members_by_part[int(component[members[0]])] = members
        boundary = edges[free[edges[:, 0]] ^ free[edges[:, 1]]]
        contacts = defaultdict(list)
        owners = defaultdict(set)
        for a, b in boundary:
            vertex, other = (a, b) if free[a] else (b, a)
            part = int(component[vertex]); owner = int(working[other])
            if owner >= 0:
                owners[part].add(owner)
            if owner in eligible:
                contacts[part].append(int(vertex))
        proposals = []
        for part in sorted(contacts, key=lambda value: members_by_part[value][0] if value in members_by_part else len(p)):
            if cooperate:
                cooperate()
            members = members_by_part.get(part)
            if members is None or len(owners[part]) != 1:
                continue
            label = next(iter(owners[part])); parent = eligible[label]
            contact = np.unique(contacts[part])
            if np.any(unsafe[contact]):
                continue
            parent_tree = cKDTree(parent.points)
            support = p[working == label]
            radius = estimate_parent_radius_profile(parent.points, support, d_bar, parent_tree=parent_tree)
            blocked = np.ones(len(p), bool); blocked[members] = False
            starts = find_lateral_starting_points(p, blocked, parent.points,
                min_cluster_size=4, max_parent_distance=np.maximum(3*radius, 12*d_bar),
                minimum_branch_angle_degrees=18., exclude_parent_tip_fraction=0.,
                parent_surface_points=support, surface_contact_distance=2.5*d_bar,
                parent_tree=parent_tree)
            starts = [start for start in starts if not is_above_primary_top(
                start.primary_point, np.asarray(primary_top_reference)[None], gravity=np.asarray(gravity))]
            candidates = reduce_similar_paths(grow_lateral_candidates(p, starts, parent.points,
                blocked, d_bar, max_steps=50, point_tree=mesh_context.point_tree,
                parent_tree=parent_tree, parent_radius_profile=radius, cooperate=cooperate))
            accepted = []
            for candidate in candidates:
                resume_lateral_tip_in_batches(p, candidate, blocked, d_bar,
                    point_tree=mesh_context.point_tree, cooperate=cooperate)
                covered = np.asarray(sorted(candidate.covered_indices), int)
                if len(covered) < 30 or np.any(blocked[covered] | unsafe[covered]):
                    continue
                rejected, metrics = is_parent_tracking_candidate(candidate, parent.points, radius, d_bar,
                                                                 parent_tree=parent_tree)
                candidate.score_components.update(metrics)
                if rejected:
                    continue
                accepted.append(candidate)
            selected = select_non_overlapping_paths(accepted, p, d_bar, rename_selected=False,
                                                    point_tree=mesh_context.point_tree)
            for candidate in selected:
                # Attach within this native contact footprint, not at a nearby
                # turn or a neighbouring root with a similar spatial position.
                near = contact[np.argmin(np.linalg.norm(p[contact]-candidate.points[0], axis=1))]
                index = int(parent_tree.query(p[near])[1])
                candidate.points = np.vstack((parent.points[index], candidate.points))
                candidate.parent_id = parent.root_id
                candidate.parent_points = parent.points
                candidate.order = parent.order + 1
                candidate.insertion_index = index
                candidate.insertion_point = parent.points[index].copy()
                candidate.score_components['native_recovered_child'] = 1.
                candidate.qc_flags.append('native_recovered_child')
                proposals.append((part, parent, candidate))
            report['decisions'].append({'parent_id': str(parent.root_id), 'component_first_vertex': int(members[0]),
                'component_size': len(members), 'start_count': len(starts), 'candidate_count': len(candidates),
                'selected_count': len(selected), 'generation': generation})
        focus = set()
        generation_claims = np.full(len(p), -1, int)
        contested = np.zeros(len(p), bool)
        terminal = {}
        for part, parent, candidate in proposals:
            if candidate.insertion_index != len(parent.points) - 1:
                continue
            parent_label = next(i for i, root in enumerate(result, 1) if root is parent)
            evidence = _free_tip_continuation(p, working, parent, candidate,
                result + [row[2] for row in proposals], label=parent_label,
                mesh_context=mesh_context, excluded=unsafe, spacing=d_bar)
            report['terminal_continuations'].append({'parent_id': parent.root_id,
                'component_first_vertex': int(members_by_part[part][0]), **evidence})
            if evidence['status'] == 'accepted':
                terminal[id(candidate)] = parent_label
        joined_parents = set()
        for part, parent, candidate in proposals:
            if id(candidate) in terminal:
                label = terminal[id(candidate)]
                parent.points = np.vstack([parent.points, candidate.points[1:]])
                parent.node_indices = None
                for descendant in result:
                    if descendant.parent_id == parent.root_id:
                        descendant.parent_points = parent.points
                # Translate only this new full-mesh support into the parent's
                # existing analysis-index space.
                indices = np.asarray(sorted(candidate.covered_indices), int)
                full = np.zeros(len(p), bool); full[indices] = True
                if parent.root_id in native_support:
                    native_support[parent.root_id].update(indices.tolist())
                    added = set(indices.tolist())
                else:
                    added = set(np.flatnonzero(full[analysis_to_mesh]).tolist()) if analysis_to_mesh is not None else set(indices.tolist())
                parent.covered_indices |= added
                parent.novel_support_indices = set(parent.novel_support_indices or ()) | added
                parent.score_components['native_terminal_continuation_joined'] = 1.
                if 'native_terminal_continuation_joined' not in parent.qc_flags:
                    parent.qc_flags.append('native_terminal_continuation_joined')
                claim = indices[working[indices] == -1]
                contested[claim] |= generation_claims[claim] >= 0
                generation_claims[claim] = label
                changed_parents.add(parent.root_id)
                joined_parents.add(parent.root_id)
                focus.add(parent.root_id)
                continue
            if max_paths is not None and len(result) >= max_paths:
                break
            number = 1
            while f'root-o{candidate.order}-{number:03d}' in used_ids:
                number += 1
            candidate.root_id = f'root-o{candidate.order}-{number:03d}'
            used_ids.add(candidate.root_id)
            native_support[candidate.root_id] = set(candidate.covered_indices)
            trial = result + [candidate]
            if validate_root_tree(trial, primary_path=primary,
                    primary_top_reference=primary_top_reference, gravity=np.asarray(gravity)):
                continue
            result.append(candidate)
            changed_parents.add(parent.root_id)
            focus.add(candidate.root_id)
            # A temporary ownership snapshot is used only to discover the next
            # order. Existing assigned/uncertain/collar labels remain barriers.
            indices = np.asarray(sorted(candidate.covered_indices), int)
            claim = indices[working[indices] == -1]
            contested[claim] |= generation_claims[claim] >= 0
            generation_claims[claim] = len(result)
            report['accepted_count'] += 1
        take = (generation_claims >= 0) & ~contested
        working[take] = generation_claims[take]
        working[contested] = -2
        if joined_parents:
            working, surface_report = fill_recovered_surfaces(p, working, result, joined_parents,
                mesh_context=mesh_context, d_bar=d_bar, excluded_mask=excluded)
            report['surface_passes'].append(surface_report)
        if not focus:
            break
    for root in result:
        if root.root_id in changed_parents:
            root.score_components['native_recovered_extension'] = 1.
    if analysis_to_mesh is not None:
        for root in result:
            if root.root_id not in native_support:
                continue
            full = np.zeros(len(p), bool)
            full[list(native_support[root.root_id])] = True
            root.covered_indices = set(np.flatnonzero(full[analysis_to_mesh]).tolist())
            root.novel_support_indices = set(root.covered_indices)
    report['accepted_root_ids'] = sorted(native_support.keys() & {r.root_id for r in result})
    return result, working, report
