"""Release independently observed native branches from provisional primary.

This runs before lateral discovery. It does not create roots or centerlines:
released source vertices become available to the normal tracing competition.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .centerline import _axis, _fit_body
from .mesh_geometry import MeshGeometryContext
from .primary_contact import _preserves_components
from .primary_surface import primary_wall_evidence, _tubular_body_witnesses
from .junction_tubes import reconcile_parent_owned_tubes, vertex_normals
from .surface_patches import _polyline_projection_distance_and_arc as project
from .types import RootPath


def release_mapped_primary_branches(points, primary_mask, primary_path, *,
                                    analysis_to_mesh, d_bar, triangles,
                                    excluded_mask=None, mesh_context=None):
    """Use observed analysis owners only; unmapped native vertices are barriers."""
    count=len(points)
    before=np.asarray(primary_mask,bool)
    if analysis_to_mesh is None and len(before)!=count:
        return before.copy(),dict(status='unresolved_missing_native_mapping',
                                 released_vertex_count=0,unresolved_candidate_count=0)
    mapping=(np.arange(count) if analysis_to_mesh is None
             else np.asarray(analysis_to_mesh,int))
    if (mapping.shape!=before.shape or np.any(mapping<0) or np.any(mapping>=count)
            or len(np.unique(mapping))!=len(mapping)):
        raise ValueError('analysis mapping must identify distinct native vertices')
    observed=np.zeros(count,bool);observed[mapping]=True
    native=np.zeros(count,bool);native[mapping]=before
    excluded=~observed
    if excluded_mask is not None:excluded|=np.asarray(excluded_mask,bool)
    result,report=release_observed_primary_branches(
        points,native,primary_path,d_bar=d_bar,triangles=triangles,
        excluded_mask=excluded,mesh_context=mesh_context)
    report['observed_native_vertex_count']=int(observed.sum())
    report['unmapped_native_vertex_count']=int((~observed).sum())
    report['mapping_policy']='unmapped source vertices excluded; no ownership interpolation'
    return result[mapping],report


def release_observed_primary_branches(points, primary_mask, primary_path, *,
                                      d_bar, triangles, excluded_mask=None,
                                      mesh_context=None):
    """Require positive branch evidence before reopening provisional ownership.

    Unmeasured surfaces, missing native edges, primary-wall conflicts and cuts
    that split retained primary support remain unchanged with explicit QC.
    """
    source=np.asarray(points,float); before=np.asarray(primary_mask,bool)
    if before.shape!=(len(source),):raise ValueError('primary mask must match source geometry')
    excluded=(np.zeros(len(source),bool) if excluded_mask is None
              else np.array(excluded_mask,bool,copy=True))
    if excluded.shape!=before.shape:raise ValueError('exclusion must match source geometry')
    report=dict(policy='native-provisional-primary-branch-release-v1',status='unresolved_no_mesh',
                released_vertex_indices=[],released_vertex_count=0,decisions=[])
    if triangles is None or not len(triangles):return before.copy(),report
    context=mesh_context or MeshGeometryContext.build(source,triangles)
    context.validate(source,triangles)
    unsafe=np.zeros(len(source),bool)
    unsafe[np.unique(context.edges[context.edge_incidence!=2])]=True
    excluded|=unsafe
    labels=np.where(before,0,-1)
    wall,radius,wall_report=primary_wall_evidence(source,labels,primary_path,
        d_bar=d_bar,triangles=triangles,excluded_mask=excluded,mesh_context=context,
        longitudinal_confirmation=True)
    distance,_=project(source,primary_path)
    # Positive exterior evidence is deliberately stricter than simply failing
    # the wall detector. Broad/flat measured walls remain protected.
    exterior=before&~excluded&~wall&(distance>1.5*radius+2*d_bar)
    edges=context.support_edges(d_bar)
    e=edges[exterior[edges].all(axis=1)]
    graph=coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(source),len(source))).tocsr()
    components=connected_components(graph,directed=False)[1]
    normals=vertex_normals(source,np.asarray(triangles))
    roots=[];seed_sets=[]
    ownership=context.ownership(labels)
    full_graph=ownership.local_graph(0,edges)
    primary_ids=ownership.vertices(0)
    primary_tree=cKDTree(primary_path)
    candidate_ids=np.flatnonzero(exterior)
    candidate_ids=candidate_ids[np.argsort(components[candidate_ids],kind='stable')]
    groups=np.split(candidate_ids,np.flatnonzero(np.diff(components[candidate_ids]))+1)
    for seed in groups:
        if len(seed)<12:continue
        row=dict(first_vertex=int(seed[0]),candidate_vertex_count=len(seed))
        report['decisions'].append(row)
        local=np.searchsorted(primary_ids,seed)
        prior=_axis(source[seed],full_graph[local][:,local],source[seed[np.argmin(distance[seed])]],d_bar)
        body,_=_fit_body(source[seed],prior,d_bar)
        if len(body)<2:
            row['status']='unresolved_no_body_axis';continue
        tangent=body[-1]-body[0]
        j=int(primary_tree.query(body.mean(axis=0))[1])
        parent_tangent=primary_path[min(j+2,len(primary_path)-1)]-primary_path[max(j-2,0)]
        cosine=abs(tangent@parent_tangent)/max(np.linalg.norm(tangent)*np.linalg.norm(parent_tangent),1e-20)
        if cosine>np.cos(np.deg2rad(20)):
            row['status']='unresolved_parallel_primary_surface';continue
        protected=_tubular_body_witnesses(source[seed],body,d_bar,normals[seed])
        row['witnessed_seed_vertices']=int(protected.sum())
        # Every released seed has its own positive body witness. A partial
        # witness is not permission to release the unobserved remainder.
        if not np.all(protected):
            row['status']='unresolved_incomplete_branch_witness';continue
        label=len(roots)+1
        root=RootPath(f'provisional-native-{seed[0]}',body,parent_id='primary',order=1)
        roots.append(root);seed_sets.append(seed)
        labels[seed]=label
        row.update(root_id=root.root_id,status='witnessed_candidate',numeric_label=label)
    if roots:
        proposed,tubes=reconcile_parent_owned_tubes(source,labels,primary_path,roots,
            d_bar=d_bar,triangles=triangles,excluded_mask=excluded,mesh_context=context)
        # This candidate pool still has no biological root identities. Do not
        # consume an independently measured primary wall to improve discovery.
        moved=(proposed>0)&before&~excluded&~wall
        original_edges=edges[before[edges].all(axis=1)]
        for label,seed in enumerate(seed_sets,1):
            row=next(r for r in report['decisions'] if r.get('numeric_label')==label)
            ids=np.flatnonzero(moved&(proposed==label))
            if len(ids) and _preserves_components(len(source),original_edges,ids,
                                                  np.flatnonzero(before&~moved)):
                row['status']='released_observed_branch'
            else:
                moved[ids]=False;row['status']='unresolved_primary_connectivity'
        # Joint cuts can interact; validate the simultaneous retained primary.
        if not _preserves_components(len(source),original_edges,np.flatnonzero(moved),
                                      np.flatnonzero(before&~moved)):
            moved[:]=False
            for row in report['decisions']:
                if row['status']=='released_observed_branch':row['status']='unresolved_joint_connectivity'
        report['tube_evidence']=tubes
    else:moved=np.zeros(len(source),bool)
    result=before.copy();result[moved]=False
    report['released_vertex_indices']=np.flatnonzero(moved).tolist()
    report['released_vertex_count']=int(moved.sum())
    report['measured_primary_wall_vertices']=int(wall.sum())
    report['primary_wall_evidence']=wall_report
    report['unresolved_candidate_count']=sum(r['status'].startswith('unresolved_') for r in report['decisions'])
    report['status']='released_observed_branches' if moved.any() else 'retained_no_certified_branch'
    return result,report
