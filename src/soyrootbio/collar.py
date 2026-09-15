"""Joint, frozen-evidence ownership at the root crown.

Geometry is in one coordinate system (normally normalized mesh units). This
stage preserves the supplied hierarchy: surface contact alone cannot establish
a new biological parent. Unsupported attachments are explicit QC decisions.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from .surface_patches import _polyline_projection_distance_and_arc as project
from .surface_patches import _segment_radius_profile
from .types import RootPath


def _frame(path, arcs):
    stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    centers = np.column_stack([np.interp(arcs, stations, path[:, i]) for i in range(3)])
    i = np.clip(np.searchsorted(stations, arcs, side="right") - 1, 0, len(path) - 2)
    tangent = path[i + 1] - path[i]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-15)
    return centers, tangent


def _profile(points, path, density, fallback):
    """Measure physical arc bins; interpolate local spacing separately."""
    spacing = float(np.median(density)) if len(density) else fallback
    stations, radii = _segment_radius_profile(points, path, spacing)
    _, arc = project(points, path)
    h = np.full(len(stations), spacing)
    bins = np.clip(np.searchsorted((stations[:-1] + stations[1:]) / 2, arc), 0, len(stations)-1)
    for i in np.unique(bins):
        h[i] = np.median(density[bins == i])
    return stations, np.maximum(radii, h), h


def analyze_joint_collar(points, labels, primary_path, lateral_paths: list[RootPath], *,
                         d_bar, triangles=None, excluded_mask=None):
    """Return labels and an auditable simultaneous multi-root collar decision.

    Every score/anchor uses the input labels. All roots compete in one matrix;
    connectivity is checked again after arbitration, without sequential moves.
    Unsupported/tied regions retain their input labels and are listed for QC.
    All excluded points become unassigned and are never propagation seeds.
    """
    p, before = np.asarray(points, float), np.asarray(labels, int)
    primary = np.asarray(primary_path, float)
    paths = [primary] + [np.asarray(r.points, float) for r in lateral_paths]
    ids = ["primary"] + [str(r.root_id) for r in lateral_paths]
    if p.ndim != 2 or p.shape[1] != 3 or not np.all(np.isfinite(p)):
        raise ValueError("points must contain finite XYZ coordinates")
    if before.shape != (len(p),) or np.any(before >= len(paths)) or np.any(before < -2):
        raise ValueError("labels must reference roots or negative assignment states")
    if not np.isfinite(d_bar) or d_bar <= 0:
        raise ValueError("d_bar must be positive and finite")
    if len(set(ids)) != len(ids):
        raise ValueError("root IDs must be unique and distinct from primary")
    for path in paths:
        if path.ndim != 2 or path.shape[1] != 3 or not np.all(np.isfinite(path)):
            raise ValueError("root paths must contain finite XYZ coordinates")
    excluded = np.zeros(len(p), bool) if excluded_mask is None else np.asarray(excluded_mask, bool).copy()
    if excluded.shape != before.shape:
        raise ValueError("excluded_mask must contain one value per point")
    result = before.copy()
    result[excluded] = -1
    report = dict(policy="joint-root-collar-v1", status="evaluated",
        coordinate_system="input coordinates; radii, arcs and spacing share these units",
        bounds_rule="integral ds/max(local primary radius, 2*local spacing) <= 8; radial envelope 3*r+4*h",
        participation_rule="sampled emergence tube intersects adaptive collar envelope; all orders",
        tie_rule="retain frozen owner within 0.15 cost; canonical root ID column order",
        score_weights=dict(segment_distance=1., radius_residual=.25, emergence_direction=.4,
                           surface_continuity=.2, geodesic=.15),
        topology_policy="preserve hierarchy; report unsupported attachments; never infer parent from surface contact",
        support_preservation_rule="retain nearest exposed-section witnesses when a joint removal increases support distance by more than 0.5*local spacing",
        donor_connectivity_rule="retain removal components touching multiple surviving donor components; include full-mesh routes outside the collar",
        participating_roots=[], local_profiles={}, neighborhood_vertex_indices=[],
        changed_assignments=[], unresolved_regions=[], topology_decisions=[],
        changed_vertex_count=0, excluded_to_unassigned_count=int(np.sum(excluded & (before != -1))))

    def finish():
        changed = np.flatnonzero(result != before)
        report["changed_vertex_count"] = int(len(changed))
        for old, new in sorted(set(zip(before[changed].tolist(), result[changed].tolist()))):
            vertices = np.flatnonzero((before == old) & (result == new))
            report["changed_assignments"].append(dict(before_label=old, after_label=new,
                before_root=ids[old] if old >= 0 else str(old), after_root=ids[new] if new >= 0 else str(new),
                vertex_indices=vertices.tolist(), count=int(len(vertices))))
        report["uncertain_vertex_count"] = int(np.sum(result == -2))
        report["unassigned_vertex_count"] = int(np.sum(result == -1))
        return result, report

    if len(primary) < 2 or len(p) < 3 or np.sum((before == 0) & ~excluded) < 3:
        report["status"] = "insufficient_primary_support"
        return finish()
    # Local sampling density is measured on the actual full-resolution surface,
    # not inherited from a potentially reduced analysis cloud.
    distances = cKDTree(p).query(p, k=min(4, len(p)))[0][:, 1:]
    density = np.median(distances, axis=1)
    density = np.where(density > 0, density, d_bar)
    profiles = {}
    for label, path in enumerate(paths):
        support = (before == label) & ~excluded
        if len(path) >= 2 and np.sum(support) >= 3:
            profiles[label] = _profile(p[support], path, density[support], d_bar)
    ps, pr, ph = profiles[0]
    intrinsic = np.r_[0., np.cumsum(np.diff(ps) / np.maximum((pr[:-1]+pr[1:])/2, ph[:-1]+ph[1:]))]
    end = float(np.interp(8., intrinsic, ps))
    pd, pa = project(p, primary)
    radius, spacing = np.interp(pa, ps, pr), np.interp(pa, ps, ph)
    neighborhood = (pa <= end) & (pd <= 3*radius + 4*spacing)
    if excluded_mask is None:
        # With no selected-base mask, use the first primary cross-section
        # locally. Do not extend an oblique plane across distant lateral tips.
        tipward = _frame(primary, np.array([0.]))[1][0]
        excluded |= neighborhood & (((p-primary[0]) @ tipward) < -ph[0])
        result[excluded] = -1
        report["excluded_to_unassigned_count"] = int(np.sum(excluded & (before != -1)))
    report["above_collar_rule"] = "supplied selected-base mask" if excluded_mask is not None else "local first primary cross-section, one local-spacing tolerance"
    report["bounds"] = dict(primary_arc_min=0., primary_arc_max=end,
        start=primary[0].tolist(), end=_frame(primary, np.array([end]))[0][0].tolist(),
        xyz_min=p[neighborhood].min(axis=0).tolist() if neighborhood.any() else [],
        xyz_max=p[neighborhood].max(axis=0).tolist() if neighborhood.any() else [])
    id_to_label = dict(zip(ids, range(len(ids))))
    parent_labels = {}
    body_arcs = {}
    participants = [0]
    for label in sorted(range(1, len(paths)), key=lambda k: ids[k]):
        root, path = lateral_paths[label-1], paths[label]
        if not len(path):
            continue
        start = int(root.body_start_index)
        if start < 0 or start >= len(path):
            raise ValueError("body_start_index must reference the root path")
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        body_arcs[label] = float(arc[start])
        parent = id_to_label.get(root.parent_id)
        parent_labels[label] = parent
        _, insertion = project(path[:1], primary)
        rp = float(np.interp(insertion[0], ps, pr))
        hp = float(np.interp(insertion[0], ps, ph))
        rc = float(profiles[label][1][0]) if label in profiles else hp
        prefix_end = min(float(arc[-1]), float(arc[start] + max(3*rp, 4*rc, 8*hp)))
        # Sampling along arc, not path nodes, catches an intersecting segment
        # even when both stored endpoints are outside the neighborhood.
        sample_arc = np.linspace(0, prefix_end, max(2, int(np.ceil(prefix_end / hp))+1))
        sample = _frame(path, sample_arc)[0] if len(path)>1 else path
        sd, sa = project(sample, primary)
        local_r, local_h = np.interp(sa, ps, pr), np.interp(sa, ps, ph)
        intersects = (sa <= end + rc + local_h) & (sd <= 3*local_r + 4*local_h + rc)
        if not np.any(intersects):
            continue
        participants.append(label)
        report["participating_roots"].append(dict(root_id=ids[label], label=label,
            parent_id=root.parent_id, order=int(root.order), emergence_arc_max=prefix_end,
            body_start_arc=body_arcs[label], insertion_primary_arc=float(insertion[0])))
    report["participating_roots"].insert(0, dict(root_id="primary", label=0, parent_id=None, order=0))
    for label in participants:
        if label in profiles:
            s, r, h = profiles[label]
            report["local_profiles"][ids[label]] = dict(arc=s.tolist(), radius=r.tolist(), spacing=h.tolist())
    report["neighborhood_vertex_indices"] = np.flatnonzero(neighborhood).tolist()
    active = neighborhood & ~excluded & np.isin(before, [-2, -1, *participants])
    ix = np.flatnonzero(active)
    if not len(ix):
        report["status"] = "empty_neighborhood"
        return finish()
    # Actual mesh edges only. A point cloud cannot prove surface continuity.
    faces = np.empty((0, 3), int) if triangles is None else np.asarray(triangles, int)
    if faces.ndim != 2 or faces.shape[1] != 3 or (len(faces) and (faces.min() < 0 or faces.max() >= len(p))):
        raise ValueError("triangles must reference mesh vertices")
    edges = np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    edge_length = np.linalg.norm(p[edges[:, 0]]-p[edges[:, 1]], axis=1)
    edges = edges[edge_length <= 4*np.maximum(density[edges[:, 0]], density[edges[:, 1]])]
    local = np.full(len(p), -1, int)
    local[ix] = np.arange(len(ix))
    e = edges[(local[edges[:, 0]] >= 0) & (local[edges[:, 1]] >= 0)]
    u, v = local[e[:, 0]], local[e[:, 1]]
    length = np.linalg.norm(p[e[:, 0]]-p[e[:, 1]], axis=1)
    adjacency = coo_matrix((np.ones(2*len(e)), (np.r_[u,v], np.r_[v,u])), shape=(len(ix),len(ix))).tocsr()
    costs = np.full((len(ix),len(participants)), np.inf)
    eligible = np.zeros_like(costs, bool)
    local_radii = np.zeros_like(costs)
    own = before[ix]
    for col, label in enumerate(participants):
        if label not in profiles:
            continue
        path = paths[label]
        dist, arc = project(p[ix], path)
        s, r, h = profiles[label]
        rad, hloc = np.interp(arc,s,r), np.interp(arc,s,h)
        local_radii[:,col] = rad
        allow = dist <= 1.5*rad + hloc
        direction_penalty = np.zeros(len(ix))
        hierarchy_ok = True
        if label:
            root = lateral_paths[label-1]
            parent = parent_labels[label]
            hierarchy_ok = parent is not None and parent in profiles and root.order == (0 if parent == 0 else lateral_paths[parent-1].order)+1
            seen, cursor = {label}, parent
            while cursor not in (None, 0):
                if cursor in seen:
                    hierarchy_ok = False
                    break
                seen.add(cursor)
                cursor = id_to_label.get(lateral_paths[cursor-1].parent_id)
            if not hierarchy_ok:
                continue
            center, tangent = _frame(path, arc)
            par_dist, par_arc = project(p[ix], paths[parent])
            par_center, _ = _frame(paths[parent], par_arc)
            sr, rr, _ = profiles[parent]
            parent_rad = np.interp(par_arc, sr, rr)
            axis_dist, _ = project(center, paths[parent])
            radial = p[ix]-par_center
            radial /= np.maximum(np.linalg.norm(radial,axis=1)[:,None], 1e-15)
            outward = np.einsum("ij,ij->i", tangent, radial)
            exposed = (axis_dist >= parent_rad-.5*rad) & ((outward >= .2) | (axis_dist >= parent_rad+rad))
            # Internal connectors never claim a trunk. The primary tube gate
            # applies even to higher-order roots crossing through the crown.
            allow &= (arc >= body_arcs[label] - 1e-9*hloc) & exposed
            allow &= pd[ix] >= radius[ix] - .3*rad
            direction_penalty = .4*np.maximum(.2-outward,0)
        eligible[:,col] = allow
        degree = np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(),1)
        continuity = np.asarray(adjacency @ (own == label).astype(float)).ravel()/degree
        costs[allow,col] = (dist/(rad+hloc) + .25*np.abs(dist-rad)/(rad+hloc)
                           + direction_penalty + .2*(1-continuity))[allow]
    # Anchors must be geometrically supported original surfaces. Jointly
    # confident anchors are fixed, so another root cannot steal a bridge.
    best_raw = np.min(costs, axis=1)
    anchors = []
    for col, label in enumerate(participants):
        seed = (own == label) & eligible[:,col] & (costs[:,col] <= best_raw+.10)
        anchors.append(seed)
        allowed = eligible[:,col]
        keep = allowed[u] & allowed[v]
        scale = (local_radii[u,col]+local_radii[v,col])/2 + (density[ix[u]]+density[ix[v]])
        weights = length / np.maximum(scale,1e-15)
        graph = coo_matrix((np.r_[weights[keep],weights[keep]],
            (np.r_[u[keep],v[keep]],np.r_[v[keep],u[keep]])),shape=adjacency.shape).tocsr()
        distance = dijkstra(graph, directed=False, indices=np.flatnonzero(seed), min_only=True, limit=6.) if seed.any() else np.full(len(ix),np.inf)
        reachable = np.isfinite(distance)
        costs[~reachable,col] = np.inf
        costs[reachable,col] += .15*distance[reachable]
        if label:
            root = lateral_paths[label-1]
            report["topology_decisions"].append(dict(root_id=ids[label], parent_id=root.parent_id,
                order=int(root.order), action="preserve", exposed_anchor_count=int(seed.sum()),
                status="supported_emergence" if seed.sum() >= 3 else "unresolved_emergence",
                internal_connector_surface_policy="parent_owned; explicit body_start and geometric exposure gates"))
    winner_col = np.argmin(costs, axis=1)
    best = costs[np.arange(len(ix)),winner_col]
    ordered_costs = np.sort(costs, axis=1)
    second = ordered_costs[:,1] if len(participants)>1 else np.full(len(ix),np.inf)
    finite = np.isfinite(best)
    margin = np.full(len(ix),np.inf)
    both = finite & np.isfinite(second)
    margin[both] = second[both]-best[both]
    ambiguous = finite & (margin < .15)
    proposed = own.copy()
    accept = finite & ~ambiguous
    proposed[accept] = np.array(participants)[winner_col[accept]]
    # Surface scores cannot erase the last observed support of an exposed
    # centerline section. Check all proposed removals together against the
    # exclusion-only input; revert those witnesses and flag the conflict.
    support_loss = np.zeros(len(ix), bool)
    proposed_full = result.copy()
    proposed_full[ix] = proposed
    for label in participants:
        if label not in profiles or not np.any((own == label) & (proposed != label)):
            continue
        path = paths[label]
        s, r, h = profiles[label]
        sample_arc = np.linspace(0,s[-1],max(2,int(np.ceil(s[-1]/(2*np.median(h))))+1))
        stations, _ = _frame(path,sample_arc)
        exposed = np.ones(len(stations),bool)
        if label:
            parent = parent_labels[label]
            if parent not in profiles:
                continue
            axis_distance, parent_arc = project(stations,paths[parent])
            ps_parent,pr_parent,_ = profiles[parent]
            exposed = ((sample_arc >= body_arcs[label]) &
                       (axis_distance >= np.interp(parent_arc,ps_parent,pr_parent)))
        original_ids = np.flatnonzero((before==label) & ~excluded)
        final_ids = np.flatnonzero(proposed_full==label)
        old_distance, witness = cKDTree(p[original_ids]).query(stations)
        new_distance = cKDTree(p[final_ids]).query(stations)[0] if len(final_ids) else np.full(len(stations),np.inf)
        loss = exposed & (new_distance > old_distance + .5*np.interp(sample_arc,s,h))
        lost = local[original_ids[witness[loss]]]
        lost = lost[lost >= 0]
        support_loss[lost] = proposed[lost] != own[lost]
    proposed[support_loss] = own[support_loss]
    # Donor connectivity matters as much as recipient connectivity. A winning
    # child claim must not slice a previously connected trunk (or sibling).
    # Test removal components together on the full mesh, including routes
    # outside the crown, and retain cuts that separate surviving donor support.
    donor_cut = np.zeros(len(ix),bool)
    proposed_full[ix] = proposed
    for label in participants:
        removed = (before==label) & ~excluded & (proposed_full!=label)
        if not removed.any():
            continue
        surviving = (before==label) & ~excluded & (proposed_full==label)
        remaining = proposed_full==label
        keep = remaining[edges[:,0]] & remaining[edges[:,1]]
        graph = coo_matrix((np.ones(2*int(keep.sum())),
            (np.r_[edges[keep,0],edges[keep,1]],np.r_[edges[keep,1],edges[keep,0]])),
            shape=(len(p),len(p))).tocsr()
        _, donor_components = connected_components(graph,directed=False)
        keep = removed[edges[:,0]] & removed[edges[:,1]]
        graph = coo_matrix((np.ones(2*int(keep.sum())),
            (np.r_[edges[keep,0],edges[keep,1]],np.r_[edges[keep,1],edges[keep,0]])),
            shape=(len(p),len(p))).tocsr()
        _, removal_components = connected_components(graph,directed=False)
        boundary = np.vstack([edges,edges[:,::-1]])
        boundary = boundary[removed[boundary[:,0]] & surviving[boundary[:,1]]]
        for component in np.unique(removal_components[boundary[:,0]]):
            adjacent = boundary[removal_components[boundary[:,0]]==component,1]
            if len(np.unique(donor_components[adjacent])) > 1:
                restore = local[np.flatnonzero(removed & (removal_components==component))]
                donor_cut[restore[restore>=0]] = True
    proposed[donor_cut] = own[donor_cut]
    # Protect original supported exposed surfaces in unresolved overlaps.
    # No primary preference is used to break lateral/lateral ties.
    rejected = np.zeros(len(ix),bool)
    for col, label in enumerate(participants):
        allowed = proposed == label
        keep = allowed[u] & allowed[v]
        graph = coo_matrix((np.ones(2*int(keep.sum())),
            (np.r_[u[keep],v[keep]],np.r_[v[keep],u[keep]])),shape=adjacency.shape).tocsr()
        _, components = connected_components(graph,directed=False)
        reachable = np.isin(components,np.unique(components[anchors[col] & allowed]))
        rejected |= allowed & (own != label) & ~reachable
    proposed[rejected] = own[rejected]
    result[ix] = proposed
    unresolved = ambiguous | ~finite | rejected | support_loss | donor_cut
    for reason, mask in [("joint_score_tie",ambiguous),("no_connected_surface_anchor",~finite),
                         ("bridge_lost_after_arbitration",rejected),
                         ("exposed_centerline_support_would_be_lost",support_loss),
                         ("donor_surface_bridge_would_be_cut",donor_cut)]:
        if not mask.any():
            continue
        keep = mask[u] & mask[v]
        graph = coo_matrix((np.ones(2*int(keep.sum())),
            (np.r_[u[keep],v[keep]],np.r_[v[keep],u[keep]])),shape=adjacency.shape).tocsr()
        _, component = connected_components(graph,directed=False)
        for key in np.unique(component[mask]):
            vertices = ix[mask & (component==key)]
            report["unresolved_regions"].append(dict(reason=reason, vertex_indices=vertices.tolist(),
                count=int(len(vertices)), xyz_min=p[vertices].min(axis=0).tolist(), xyz_max=p[vertices].max(axis=0).tolist()))
    report["unresolved_vertex_count"] = int(unresolved.sum())
    report["unresolved_assigned_vertex_count"] = int(np.sum(unresolved & (proposed >= 0)))
    report["unresolved_negative_vertex_count"] = int(np.sum(unresolved & (proposed < 0)))
    report["connectivity"] = "triangle_edges" if len(edges) else "no_mesh_support"
    return finish()
