"""Assignment-driven reconstruction, kept separate from the production tracer."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree, ConvexHull

from soyrootbio.geometry import resample_polyline, tangent_vectors, path_length
from soyrootbio.pipeline import _selected_base_exclusion_mask
from soyrootbio.primary import _plane_basis, _robust_cross_section_center
from w5168_assignment_audit import SOURCE, OUT, load_session, edges_of, save_json


def graph_for(points, edges):
    d = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    return coo_matrix((np.r_[d,d], (np.r_[edges[:,0],edges[:,1]], np.r_[edges[:,1],edges[:,0]])), shape=(len(points),len(points))).tocsr()


def resolve_unlabelled(session, edges, meta):
    p, lab = session.mesh.positions, session.mesh.root_labels.copy()
    a = meta['point_assignment']
    above = _selected_base_exclusion_mask(
        p, np.array(a['base_point_source_coordinates']), np.array(a['base_tipward_direction']),
        gravity=np.array(a['gravity_direction']),
        collar_neighborhood_radius=a['base_collar_neighborhood_radius_normalized']*meta['normalization_scale'],
        tolerance=a['above_base_tolerance_normalized']*meta['normalization_scale'])
    allowed = ~above
    edge_mask = allowed[edges[:,0]] & allowed[edges[:,1]]
    graph = graph_for(p, edges[edge_mask])
    distance, _, source = dijkstra(graph, indices=np.flatnonzero((lab>=0)&allowed), min_only=True, return_predecessors=True)
    candidates = (lab < 0) & allowed
    ix = np.flatnonzero(candidates & np.isfinite(distance))
    lab[ix] = lab[source[ix]]
    rows = [{'vertex':int(i),'previous_label':int(session.mesh.root_labels[i]),'label':int(lab[i]),
             'surface_distance':float(distance[i]),'seed_vertex':int(source[i])} for i in ix]
    report = {'above_mask_count':int(above.sum()), 'preserved_assigned_above':int(((session.mesh.root_labels>=0)&above).sum()),
              'resolved_uncertain':int(np.sum(session.mesh.root_labels[ix]==-2)),
              'resolved_unassigned':int(np.sum(session.mesh.root_labels[ix]==-1)),
              'unreachable_below':int(np.sum(candidates & ~np.isfinite(distance))),
              'remaining_uncertain':int(np.sum(lab==-2)), 'remaining_unassigned':int(np.sum(lab==-1)),
              'max_propagation_distance':float(np.max(distance[ix])) if len(ix) else 0,
              'assignments':rows}
    return lab, above, report


def surface_axis(points, edges, base_hint, spacing):
    """Initialize from mesh geodesics, independent of the supplied drawn line.

    Tiny disconnected islands do not define the longitudinal axis. Substantial
    disconnected sections are joined by their shortest gap, explicitly reported.
    Their ownership is never modified.
    """
    g = graph_for(points, edges)
    _, cc = connected_components(g, directed=False)
    counts = np.bincount(cc)
    largest = int(np.argmax(counts))
    retained_cc = np.flatnonzero(counts >= max(8, .04*counts[largest]))
    retained_cc = np.unique(np.r_[retained_cc, largest])
    use = np.flatnonzero(np.isin(cc, retained_cc))
    remap = np.full(len(points), -1)
    remap[use] = np.arange(len(use))
    sub = points[use]
    e = edges[np.all(remap[edges]>=0, axis=1)]
    e = remap[e]
    included = cc[use] == largest
    bridges = []
    while not np.all(included):
        src, dst = np.flatnonzero(included), np.flatnonzero(~included)
        distances, nearest = cKDTree(sub[src]).query(sub[dst])
        best = int(np.argmin(distances))
        if distances[best] > 3*spacing:
            # A detached island across a large gap is not evidence for a
            # continuous centreline or for a long free-space connector.
            keep=np.flatnonzero(included)
            mapping=np.full(len(sub),-1); mapping[keep]=np.arange(len(keep))
            e=mapping[e[np.all(mapping[e]>=0,axis=1)]]
            use=use[keep]; sub=sub[keep]
            break
        left, right = int(src[nearest[best]]), int(dst[best])
        bridges.append(float(distances[best]))
        e = np.vstack([e, [left,right]])
        included |= cc[use] == cc[use[right]]
    graph = graph_for(sub, e)
    seed = int(np.argmin(np.linalg.norm(sub-base_hint,axis=1)))
    d0 = dijkstra(graph,indices=seed)
    end = int(np.argmax(d0))
    de = dijkstra(graph,indices=end)
    start = int(np.argmax(de))
    if np.linalg.norm(sub[end]-base_hint) < np.linalg.norm(sub[start]-base_hint):
        start,end = end,start
        de = dijkstra(graph,indices=end)
    ds = dijkstra(graph,indices=start)
    length = float(ds[end])
    axial = np.clip((ds-de+length)*.5, 0, length)
    initialization='bidirectional_surface_geodesic'
    # A global longitudinal coordinate is more stable on short flared bases:
    # surface distance there can walk around the circumference. Use it only
    # for an elongated root whose geodesic extent is consistent with one axis.
    origin=sub.mean(0)
    _,singular,basis=np.linalg.svd(sub-origin,full_matrices=False)
    principal=(sub-origin)@basis[0]
    if singular[0]>3*max(singular[1],1e-12) and length<1.5*np.ptp(principal):
        if principal[start]>principal[end]: principal=-principal
        axial=principal-principal.min()
        length=float(axial.max())
        initialization='elongated_surface_principal_coordinate'
    step = max(3*spacing, length/650)
    grid = np.arange(0,length+step*.5,step)
    centers = []
    coordinates = []
    for x in grid:
        mask = np.abs(axial-x)<=max(step*.7,2*spacing)
        if mask.sum()<3:
            continue
        centers.append(np.mean(sub[mask],axis=0))
        coordinates.append(x)
    if len(centers)<2:
        centers = [sub[start],sub[end]]
    line = np.asarray(centers)
    # Endpoint centres are taken from assigned surface caps, never drawn tips.
    if len(line)>3:
        line[1:-1] = gaussian_filter1d(line, .7, axis=0, mode='nearest')[1:-1]
    return resample_polyline(line, max(1.5*spacing,1e-6)), {
        'assigned_points':len(points), 'axis_support_points':len(sub),
        'island_points_excluded_from_axis':len(points)-len(sub),
        'component_bridge_distances':bridges, 'axis_coordinate_extent':length,
        'initialization':initialization,
    }, use


def fit_sections(points, initial, spacing, iterations=3):
    """Fit transverse centres only to this root's assigned support.

    Projected arc locality prevents nearby distant turns entering a section.
    Sections with inadequate angular coverage retain their initialization.
    Smoothing is limited to one short pass, avoiding broad corner rounding.
    """
    line = resample_polyline(initial,max(2*spacing,path_length(initial)/700))
    tree = cKDTree(points)
    for _ in range(iterations):
        tangents = tangent_vectors(line)
        assigned_station = cKDTree(line).query(points)[1]
        arc = np.r_[0,np.cumsum(np.linalg.norm(np.diff(line,axis=0),axis=1))]
        updated = line.copy()
        for i in range(1,len(line)-1):
            t = tangents[i]
            station = line[i]
            nearest = tree.query(station,k=min(16,len(points)))[0]
            radius = max(6*spacing,2.5*float(np.median(nearest)))
            local = np.asarray(tree.query_ball_point(station,radius),dtype=int)
            if len(local)<8:
                continue
            offsets = points[local]-station
            axial = offsets@t
            valid = (np.abs(axial)<=2*spacing) & (np.abs(arc[assigned_station[local]]-arc[i])<=max(radius,4*spacing))
            if valid.sum()<8:
                continue
            basis = _plane_basis(t)
            radial = offsets[valid]@basis.T
            center = _robust_cross_section_center(radial,fit_circle=False)
            q = radial-center
            angles = np.sort(np.arctan2(q[:,1],q[:,0]))
            angular_gap = np.max(np.diff(np.r_[angles,angles[0]+2*np.pi]))
            if angular_gap>np.pi or np.linalg.norm(center)>radius*.65:
                continue
            updated[i] = station + .75*(center@basis)
        if len(updated)>3:
            # At most 1/8 weighting of each immediate neighbour.
            updated[1:-1] = .75*updated[1:-1]+.125*(updated[:-2]+updated[2:])
        line = updated
    return resample_polyline(line, max(1.5*spacing,1e-6))


def section_metrics(points, path, spacing):
    line = resample_polyline(path,max(3*spacing,path_length(path)/500))
    tangents = tangent_vectors(line)
    tree = cKDTree(points)
    offsets_out=[]
    outside=0
    tested=0
    for i in range(2,len(line)-2):
        nearest = tree.query(line[i], k=min(16,len(points)))[0]
        radius = max(6*spacing,2.5*float(np.median(nearest)))
        local = np.asarray(tree.query_ball_point(line[i],radius),dtype=int)
        offsets = points[local]-line[i]
        slab = offsets[np.abs(offsets@tangents[i])<=2*spacing]
        if len(slab)<8:
            continue
        radial = slab@_plane_basis(tangents[i]).T
        if np.linalg.matrix_rank(radial-radial.mean(0))<2:
            continue
        try:
            hull = ConvexHull(radial)
        except Exception:
            continue
        outside += int(np.max(hull.equations[:,-1])>spacing*.1)
        tested += 1
        center = _robust_cross_section_center(radial)
        r = max(spacing,float(np.median(np.linalg.norm(radial-center,axis=1))))
        offsets_out.append(float(np.linalg.norm(center)/r))
    return {'tested_sections':tested,'outside_transverse_hull':outside,
            'outside_fraction':outside/max(tested,1),
            'median_offset_over_radius':float(np.median(offsets_out)) if offsets_out else None,
            'p90_offset_over_radius':float(np.quantile(offsets_out,.9)) if offsets_out else None,
            'length':path_length(path)}


def render(session, old, new, labels, report):
    plt.rcParams.update({'font.size':10})
    p = session.mesh.positions
    fig, axes = plt.subplots(1,3,figsize=(15,9),layout='constrained')
    rng=np.random.default_rng(42)
    ix=rng.choice(len(p),min(len(p),70000),replace=False)
    root_colors = {r.numeric_label: plt.cm.turbo((i+.5)/len(session.roots)) for i,r in enumerate(session.roots.values())}
    colors=np.array([root_colors.get(int(v),(.65,.65,.65,1)) for v in labels[ix]])
    for ax,dims in zip(axes,[(0,2),(1,2),(0,1)]):
        ax.scatter(p[ix,dims[0]],p[ix,dims[1]],s=.3,c=colors,rasterized=True,alpha=.4)
        for line in new.values():
            ax.plot(line[:,dims[0]],line[:,dims[1]],lw=.5,c='black')
        ax.set_aspect('equal'); ax.set_xlabel('XYZ'[dims[0]]); ax.set_ylabel('XYZ'[dims[1]])
    fig.suptitle('Reconstructed centerlines on edited point assignments | mesh units')
    fig.savefig(OUT/'repaired_overview.png',dpi=160); plt.close(fig)
    interesting=['primary','root-o1-003','root-o1-014','root-o1-004',*report['new_roots']]
    fig,axes=plt.subplots(5,2,figsize=(16,23),layout='constrained')
    for ax,rid in zip(axes.flat,interesting):
        root=session.roots[rid]
        cloud=p[labels==root.numeric_label]
        if len(cloud)<2: continue
        _,_,basis=np.linalg.svd(cloud-cloud.mean(0),full_matrices=False)
        xy=(cloud-cloud.mean(0))@basis[:2].T
        before=(old[rid]-cloud.mean(0))@basis[:2].T
        after=(new[rid]-cloud.mean(0))@basis[:2].T
        ax.scatter(xy[:,0],xy[:,1],s=1,c='#b5bac5',alpha=.5,rasterized=True)
        ax.plot(before[:,0],before[:,1],c='#df5938',lw=1,label='Before centreline repair')
        ax.plot(after[:,0],after[:,1],c='#1469b2',lw=1,label='From assigned surface')
        ax.set_title(rid); ax.set_aspect('equal'); ax.legend(fontsize=8)
    fig.savefig(OUT/'centerline_comparison.png',dpi=140); plt.close(fig)


def main():
    s=load_session()
    meta=json.loads((SOURCE/'metadata.json').read_text())
    edges=edges_of(s.mesh.triangles)
    labels,above,assignment=resolve_unlabelled(s,edges,meta)
    print('Assignment:',json.dumps({k:v for k,v in assignment.items() if k!='assignments'}),flush=True)
    old={rid:r.points.copy() for rid,r in s.roots.items()}
    curves={}
    metrics=[]
    spacing=meta['d_bar_normalized']*meta['normalization_scale']
    for rid,root in sorted(s.roots.items(),key=lambda x:(x[1].order,x[0])):
        indices=np.flatnonzero(labels==root.numeric_label)
        cloud=s.mesh.positions[indices]
        if len(cloud)==0:
            metrics.append({'root_id':rid,'assigned_points':0,'status':'remove_empty_skeleton_no_assigned_surface'})
            print(rid,'empty root: no assigned point support',flush=True)
            continue
        remap=np.full(len(labels),-1,dtype=int); remap[indices]=np.arange(len(indices))
        local_edges=remap[edges[(labels[edges[:,0]]==root.numeric_label)&(labels[edges[:,1]]==root.numeric_label)]]
        if rid=='primary':
            hint=np.asarray(meta['point_assignment']['base_point_source_coordinates'])
        else:
            # Only orientation comes from the parent; no drawn path nodes are used.
            parent=curves[root.parent_id]
            d,j=cKDTree(parent).query(cloud)
            hint=cloud[int(np.argmin(d))]
        axis,detail,use=surface_axis(cloud,local_edges,hint,spacing)
        axis=fit_sections(cloud[use],axis,spacing)
        if rid!='primary':
            # The connector lies within the parent and is separate from the fitted child surface axis.
            parent=curves[root.parent_id]
            j=int(cKDTree(parent).query(axis[0])[1])
            connector=parent[j]
            detail['parent_connector_length']=float(np.linalg.norm(connector-axis[0]))
            axis=resample_polyline(np.vstack([connector,axis]),max(1.5*spacing,1e-6))
        curves[rid]=axis
        detail.update(root_id=rid,before=section_metrics(cloud,old[rid],spacing),after=section_metrics(cloud,axis,spacing),
                      status='insufficient_surface_for_reliable_centerline' if len(cloud)<30 else 'reconstructed')
        metrics.append(detail)
        print(rid, 'points',len(cloud),'length',round(detail['before']['length'],2),'->',round(detail['after']['length'],2),flush=True)
    audit=json.loads((OUT/'audit.json').read_text())
    report={'method':'assigned-surface-geodesic-sections-v1','spacing_mesh_units':spacing,
            'assignment':assignment,'roots':metrics,'new_roots':audit['new_roots']}
    save_json(OUT/'repair_metrics.json',report)
    np.savez_compressed(OUT/'candidate_curves.npz',**curves)
    np.save(OUT/'candidate_labels.npy',labels)
    render(s,old,curves,labels,report)


if __name__=='__main__':
    main()
