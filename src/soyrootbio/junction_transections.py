"""Trim attached primary protrusions using branch-facing transverse radii."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .centerline import _support_edges
from .surface_patches import _polyline_projection_distance_and_arc as project
from .surface_patches import _segment_radius_profile
from .types import RootPath


def _frame(path, arc, stations, tangent_span):
    centers = np.column_stack([np.interp(stations, arc, path[:, k]) for k in range(3)])
    left, right = np.maximum(0, stations-tangent_span), np.minimum(arc[-1], stations+tangent_span)
    tangent = np.column_stack([np.interp(right, arc, path[:, k])-np.interp(left, arc, path[:, k]) for k in range(3)])
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-12)
    return centers, tangent


def trim_primary_junctions(points: np.ndarray, labels: np.ndarray,
                           primary_path: np.ndarray, roots: list[RootPath], *,
                           d_bar: float, triangles: np.ndarray | None,
                           excluded_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Compare two parent flanks in the contacted child's facing sector.

    Proposals use frozen labels and geometry. A primary component can be split
    only outside its interpolated parent envelope, with an observed mesh path
    to the contacted O1. The insertion and centerlines themselves are untouched.
    Missing/invalid two-sided parent support is reported, never invented.
    """
    p, before, parent = np.asarray(points, float), np.asarray(labels, int), np.asarray(primary_path, float)
    spacing = float(d_bar)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("points must contain finite XYZ coordinates")
    if before.shape != (len(p),) or np.any(before >= len(roots)+1) or np.any(before < -2):
        raise ValueError("labels must reference roots or assignment states")
    if parent.ndim != 2 or parent.shape[1] != 3 or not np.isfinite(parent).all():
        raise ValueError("primary_path must contain finite XYZ coordinates")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("d_bar must be positive and finite")
    excluded = np.zeros(len(p), bool) if excluded_mask is None else np.asarray(excluded_mask, bool)
    if excluded.shape != before.shape:
        raise ValueError("excluded_mask must match labels")
    result = before.copy()
    report = dict(policy="branch-facing-transection-trimming-v1",
        coordinate_system="input coordinates",
        sector_half_angle_degrees=45, trimming_sector_half_angle_degrees=60,
        section_radius_percentile=80, section_half_width="max(1.5*d_bar, 0.2*parent_radius)",
        flank_distances_parent_radii=[2, 3, 4, 5, 6],
        abrupt_increase="excess > max(0.25*expected_radius, 2*d_bar, 3*flank_MAD)",
        maximum_flank_radius_ratio=1.6, parent_wall_margin="0.25*d_bar",
        mesh_policy="observed triangle edges; no point-cloud bridging",
        connector_policy="surface-axis reference only; never edit insertion or geometry",
        tie_margin=0.15, transferred_vertex_count=0, ambiguous_vertices=0, junctions=[])
    if len(parent) < 2 or triangles is None or not len(triangles):
        report['status'] = 'insufficient_parent_or_mesh'
        return result, report
    edges = _support_edges(p, triangles, spacing)
    edges = edges[~excluded[edges].any(axis=1)]
    pi = np.flatnonzero((before == 0) & ~excluded)
    if len(pi) < 8:
        report['status'] = 'insufficient_primary_support'
        return result, report
    parent_points = p[pi]
    pd, pa = project(parent_points, parent)
    radial_stations, radii = _segment_radius_profile(parent_points, parent, spacing)
    parcarc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(parent, axis=0), axis=1))]
    tree = cKDTree(parent_points)
    lookup = np.full(len(p), -1, int); lookup[pi] = np.arange(len(pi))
    # Contact evidence belongs to surface labels, not guessed insertion nodes.
    crossing = edges[(before[edges[:, 0]] == 0) ^ (before[edges[:, 1]] == 0)]
    at_start = before[crossing[:, 0]] == 0
    primary_end = np.where(at_start, crossing[:, 0], crossing[:, 1])
    child_end = np.where(at_start, crossing[:, 1], crossing[:, 0])
    contacts = {label: np.unique(primary_end[before[child_end] == label]) for label in np.unique(before[child_end]) if label > 0}
    claims = []
    for label, root in enumerate(roots, 1):
        if root.order != 1 or root.parent_id != 'primary':
            continue
        row = dict(root_id=root.root_id, label=label, transferred_vertex_count=0)
        report['junctions'].append(row)
        path = np.asarray(root.points, float)
        if path.ndim != 2 or path.shape[1] != 3 or not np.isfinite(path).all():
            raise ValueError("root paths must contain finite XYZ coordinates")
        if len(path) < 2 or label not in contacts:
            row['status'] = 'no_measurable_child_or_mesh_contact'; continue
        if not 0 <= root.body_start_index < len(path):
            raise ValueError("body_start_index must reference child path")
        body = path[root.body_start_index:]
        if len(body) < 2:
            row['status'] = 'no_measurable_child_or_mesh_contact'; continue
        ci = lookup[contacts[label]]
        cd, ca = project(parent_points[ci], body)
        # The earliest observed body contact identifies emergence; distal
        # contacts with the same root do not define additional emergence sites.
        near = ca <= np.min(ca) + 4*spacing
        contact_arc = float(np.median(pa[ci[near]]))
        rp = float(np.interp(contact_arc, radial_stations, radii))
        rp = max(rp, 2*spacing)
        # A partial primary line (e.g. SN14's excluded proximal component)
        # must not interpret remote support distance as a biological radius.
        if np.median(pd[ci[near]]) > 4*rp or rp > 4*float(np.median(radii)):
            row['status'] = 'unreliable_parent_reference'; continue
        contact = np.mean(parent_points[ci[near]], axis=0)
        center, tangent = _frame(parent, parcarc, np.array([contact_arc]), max(rp, 3*spacing))
        center, tangent = center[0], tangent[0]
        direction = contact-center
        direction -= np.dot(direction, tangent)*tangent
        if np.linalg.norm(direction) <= spacing:
            row['status'] = 'unresolved_branch_direction'; continue
        direction /= np.linalg.norm(direction)
        h = max(rp, 4*spacing)
        slab = max(1.5*spacing, 0.2*rp)
        offsets = np.array([-6,-5,-4,-3,-2,-1,-.5,0,.5,1,2,3,4,5,6])*h
        positions = contact_arc+offsets
        centers, tangents = _frame(parent, parcarc, positions, max(rp, 3*spacing))
        sector, opposite, section_counts = [], [], []
        for s, c, t in zip(positions, centers, tangents):
            if s < 2*slab or s > parcarc[-1]-2*slab:
                sector.append(np.nan); opposite.append(np.nan); section_counts.append(0); continue
            local = np.array(tree.query_ball_point(c, 6*h), int)
            delta = parent_points[local]-c
            axial = delta@t
            keep = (np.abs(axial) <= slab) & (np.abs(pa[local]-s) <= max(2*slab, .5*h))
            delta = delta[keep]-axial[keep,None]*t
            radial = np.linalg.norm(delta, axis=1)
            facing = direction-np.dot(direction,t)*t
            facing /= max(np.linalg.norm(facing),1e-12)
            cosine = delta@facing / np.maximum(radial, 1e-12)
            a = radial[cosine >= np.cos(np.pi/4)]
            b = radial[cosine <= -np.cos(np.pi/4)]
            sector.append(float(np.percentile(a,80)) if len(a)>=3 else np.nan)
            opposite.append(float(np.median(b)) if len(b)>=3 else np.nan)
            section_counts.append(int(len(a)))
        sector, opposite = np.array(sector), np.array(opposite)
        row.update(contact_primary_arc=contact_arc, local_parent_radius=rp,
                   region_center=center.tolist(), region_radius=6*h,
                   branch_direction=direction.tolist(), section_arcs=positions.tolist(),
                   section_facing_radius=[None if not np.isfinite(v) else v for v in sector],
                   section_opposite_radius=[None if not np.isfinite(v) else v for v in opposite],
                   section_facing_counts=section_counts)
        upper = (offsets <= -2*h) & np.isfinite(sector)
        lower = (offsets >= 2*h) & np.isfinite(sector)
        if upper.sum() < 2 or lower.sum() < 2:
            row['status'] = 'missing_two_sided_transections'; continue
        # Median of each side rejects a contaminated flank section without
        # allowing the central protrusion to inflate its own expected wall.
        ra, rb = float(np.median(sector[upper])), float(np.median(sector[lower]))
        if max(ra,rb)/max(min(ra,rb),spacing) > 1.6:
            row['status'] = 'inconsistent_flanks'; continue
        sa, sb = float(np.median(positions[upper])), float(np.median(positions[lower]))
        expected = float(np.interp(contact_arc,[sa,sb],[ra,rb]))
        mad = float(np.median(np.r_[np.abs(sector[upper]-ra),np.abs(sector[lower]-rb)]))
        central = (np.abs(offsets) <= h) & np.isfinite(sector)
        if not central.any():
            row['status'] = 'missing_central_transection'; continue
        peak = float(np.max(sector[central]))
        excess = peak-expected
        threshold = max(.25*expected, 2*spacing, 3*mad)
        row.update(upper_radius=ra,lower_radius=rb,expected_radius=expected,
                   peak_facing_radius=peak,excess=excess,threshold=threshold,flank_MAD=mad)
        if excess <= threshold:
            row['status'] = 'no_abrupt_directional_increase'; continue
        opp_flank = opposite[(upper|lower)&np.isfinite(opposite)]
        opp_center = opposite[central&np.isfinite(opposite)]
        if len(opp_flank) and len(opp_center) and np.median(opp_center)-np.median(opp_flank) > threshold:
            row['status'] = 'bilateral_widening'; continue
        local = np.array(tree.query_ball_point(center,6*h),int)
        local = local[np.abs(pa[local]-contact_arc)<=2*h]
        c,t = _frame(parent,parcarc,pa[local],max(rp,3*spacing))
        radial_vector = parent_points[local]-c
        radial_vector -= np.einsum('ij,ij->i',radial_vector,t)[:,None]*t
        radial = np.linalg.norm(radial_vector,axis=1)
        facing = direction-np.einsum('ij,j->i',t,direction)[:,None]*t
        facing /= np.maximum(np.linalg.norm(facing,axis=1)[:,None],1e-12)
        cosine = np.einsum('ij,ij->i',radial_vector,facing)/np.maximum(radial,1e-12)
        wall = np.interp(pa[local],[sa,sb],[ra,rb])
        child_support = p[(before==label)&~excluded]
        sd,ss = project(child_support,body)
        basal = sd[ss<=max(6*h,12*spacing)]
        rc = max(spacing,float(np.median(basal)) if len(basal)>=3 else spacing)
        # Existing centerline plus a local emergence-axis reference. This only
        # scores exterior surface and does not add a geometric connector.
        body_distance,body_arc = project(parent_points[local],body)
        connector_distance,_ = project(parent_points[local],np.vstack([center,body[0]]))
        fit_distance = np.minimum(body_distance,connector_distance)
        eligible = ((radial > wall+.25*spacing) & (cosine>=.5)
                    & (fit_distance<=1.75*rc+2*spacing)
                    & (body_arc<=max(6*h,12*spacing)))
        candidates = pi[local[eligible]]
        # Strict surface contact: a candidate reaches the contacted O1 using
        # only other exterior candidates and its original child-owned surface.
        allowed = (before==label)&~excluded
        allowed[candidates] = True
        e = edges[allowed[edges].all(axis=1)]
        g = coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(p),len(p))).tocsr()
        _,cc = connected_components(g,directed=False)
        connected = np.isin(cc[candidates],np.unique(cc[(before==label)&~excluded]))
        candidates = candidates[connected]
        score = (radial[eligible]/(wall[eligible]+spacing)-fit_distance[eligible]/(rc+spacing))[connected]
        row.update(status='detected',candidate_vertices=len(candidates),child_radius=rc)
        claims.append((label,candidates,score,row))
    best = np.full(len(p),-np.inf);second=best.copy();winner=np.zeros(len(p),int)
    for label,vertices,score,_ in claims:
        better = score>best[vertices]
        second[vertices]=np.where(better,best[vertices],np.maximum(second[vertices],score))
        best[vertices]=np.maximum(best[vertices],score)
        winner[vertices[better]]=label
    contested=np.isfinite(second)&((best-np.where(np.isfinite(second),second,0))<.15)
    winner[contested]=0; report['ambiguous_vertices']=int(contested.sum())
    for label,vertices,_,row in claims:
        vertices=vertices[winner[vertices]==label]
        allowed=(before==label)&~excluded;allowed[vertices]=True
        e=edges[allowed[edges].all(axis=1)]
        g=coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(p),len(p))).tocsr()
        _,cc=connected_components(g,directed=False)
        vertices=vertices[np.isin(cc[vertices],np.unique(cc[(before==label)&~excluded]))]
        result[vertices]=label
        row['transferred_vertex_count']=int(len(vertices))
    # Connectivity preservation is checked against the actual source mesh.
    # The shorter-edge graph above is only a conservative transfer path; its
    # optional edge filter must not manufacture a new parent disconnection.
    faces = np.asarray(triangles, int)
    mesh_edges = np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    _preserve_parent_components(p, before, result, mesh_edges, parent, spacing, report)
    for row in report['junctions']:
        row['transferred_vertex_count'] = int(np.sum((before == 0) & (result == row['label'])))
    report['transferred_vertex_count']=int(np.sum(before!=result))
    report['status']='evaluated'
    return result,report


def _preserve_parent_components(p, before, result, edges, parent, spacing, report):
    """Finish exterior islands at a cut, or reject cuts that split the parent.

    This is restricted to fragments newly detached from an original primary
    component. Original disconnected components are never silently absorbed.
    """
    def components(labels):
        e = edges[(labels[edges[:, 0]] == 0) & (labels[edges[:, 1]] == 0)]
        g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(p), len(p))).tocsr()
        return connected_components(g, directed=False)[1]
    original = components(before)
    rows = {r['label']: r for r in report['junctions'] if r['transferred_vertex_count']}
    report['exterior_cut_island_vertices'] = 0
    report['connectivity_rejected_root_ids'] = []
    rejected = set()
    for _ in range(len(rows)+1):
        current = components(result)
        retain = np.flatnonzero((before == 0) & (result == 0))
        islands = []
        for component in np.unique(original[retain]):
            ids = retain[original[retain] == component]
            parts, sizes = np.unique(current[ids], return_counts=True)
            if len(parts) > 1:
                for part in parts[parts != parts[np.argmax(sizes)]]:
                    islands.append(ids[current[ids] == part])
        if not islands:
            return
        restore = set()
        changed = False
        for ids in islands:
            member = np.zeros(len(p), bool); member[ids] = True
            boundary = edges[member[edges[:, 0]] ^ member[edges[:, 1]]]
            neighbor = np.where(member[boundary[:, 0]], boundary[:, 1], boundary[:, 0])
            labels = np.unique(result[neighbor])
            target = int(labels[0]) if len(labels) == 1 else -1
            accepted = False
            if target in rows and target not in rejected:
                row = rows[target]
                distance, arc = project(p[ids], parent)
                stations = np.array(row['section_arcs'])
                radii = np.array(row['section_facing_radius'], float)
                h = row['region_radius']/6
                upper = (stations <= row['contact_primary_arc']-2*h) & np.isfinite(radii)
                lower = (stations >= row['contact_primary_arc']+2*h) & np.isfinite(radii)
                wall = np.interp(arc, [np.median(stations[upper]), np.median(stations[lower])],
                                 [row['upper_radius'], row['lower_radius']])
                accepted = bool(np.all(distance > wall+.25*spacing)
                    and np.max(np.linalg.norm(p[ids]-row['region_center'], axis=1)) <= row['region_radius']
                    and np.linalg.norm(np.ptp(p[ids], axis=0)) <= 2*h)
            if accepted:
                result[ids] = target
                report['exterior_cut_island_vertices'] += int(len(ids))
                changed = True
            else:
                restore.update(int(v) for v in np.unique(result[neighbor[(before[neighbor] == 0) & (result[neighbor] > 0)]]))
        for label in sorted(restore):
            result[(before == 0) & (result == label)] = 0
            rejected.add(label)
            rows[label]['status'] = 'retained_to_preserve_primary_connectivity'
            report['connectivity_rejected_root_ids'].append(rows[label]['root_id'])
            changed = True
        if not changed:
            raise RuntimeError('Unable to preserve original primary component connectivity')
    raise RuntimeError('Primary component connectivity guard did not converge')
