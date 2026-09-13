"""Frozen-skeleton replay of the original assignment stages and primary mask."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from soyrootbio.types import Normalization,RootPath
from soyrootbio.primary import tangent_plane_primary_segmentation,refine_primary_centerline
from soyrootbio.pipeline import (PipelineConfig,_resolve_primary_path,_selected_base_exclusion_mask,
    _assign_lateral_points,_analysis_root_labels,_resolve_parent_owned_junctions,
    _absorb_small_primary_surface_patches)
from w5168_assignment_audit import SOURCE,OUT,load_session,save_json


def main():
    s=load_session(); meta=json.loads((SOURCE/'metadata.json').read_text())
    c=meta['config'].copy(); c['input_path']=Path(c['input_path']); c['output_dir']=OUT/'assignment_replay'
    config=PipelineConfig(**c)
    norm=Normalization(np.array(meta['normalization_minimum']),meta['normalization_scale'])
    p=norm.transform_points(s.mesh.positions); spacing=meta['d_bar_normalized']
    coarse,_=_resolve_primary_path(s.mesh.positions,p,norm,spacing,config)
    a=meta['point_assignment']
    excluded=_selected_base_exclusion_mask(p,norm.transform_points(np.array(a['base_point_source_coordinates'])[None])[0],
        np.array(a['base_tipward_direction']),gravity=np.array(a['gravity_direction']),
        collar_neighborhood_radius=a['base_collar_neighborhood_radius_normalized'],tolerance=a['above_base_tolerance_normalized'])
    first=tangent_plane_primary_segmentation(p,coarse.points,spacing); first[excluded]=False
    centered=refine_primary_centerline(p,first,coarse.points,spacing,fit_circular_cross_sections=True)
    locked=tangent_plane_primary_segmentation(p,centered,spacing,complete_cross_section=True); locked[excluded]=False
    final_primary=refine_primary_centerline(p,locked,centered,spacing,fit_circular_cross_sections=True)
    primary=norm.transform_points(s._baseline_roots['primary'].points)
    roots=sorted([r for r in s._baseline_roots.values() if r.root_id!='primary'],key=lambda x:x.numeric_label)
    paths=[RootPath(root_id=r.root_id,parent_id=r.parent_id,order=r.order,points=norm.transform_points(r.points),
                    insertion_point=None if r.insertion_point is None else norm.transform_points(r.insertion_point[None])[0],
                    insertion_index=r.insertion_index) for r in roots]
    ll,comp=_assign_lateral_points(p,paths,locked,spacing,excluded_mask=excluded,return_competing_labels=True)
    raw=_analysis_root_labels(locked,ll)
    junction,jr=_resolve_parent_owned_junctions(p,raw,primary,paths,d_bar=spacing,assignment_radius=max(4*spacing,.006),ambiguity_margin=max(.75*spacing,.001),competing_labels=comp)
    second_junction,jr2=_resolve_parent_owned_junctions(p,junction,primary,paths,d_bar=spacing,assignment_radius=max(5*spacing,.008),ambiguity_margin=max(.75*spacing,.001),competing_labels=comp)
    cleaned,cr=_absorb_small_primary_surface_patches(p,second_junction,primary,paths,d_bar=spacing,triangles=s.mesh.triangles,excluded_mask=excluded,primary_support_points=p[locked])
    ref=s.mesh.root_labels; baseline=s._baseline_labels
    primary_to_lateral=(baseline==0)&(ref>0)
    stages={'initial_primary':int(first.sum()),'locked_primary':int(locked.sum()),
            'locked_primary_now_lateral':int((locked&primary_to_lateral).sum()),
            'initial_primary_now_lateral':int((first&primary_to_lateral).sum()),
            'primary_to_lateral_total':int(primary_to_lateral.sum()),
            'after_junction_primary_now_lateral':int(((junction==0)&primary_to_lateral).sum()),
            'after_second_junction_primary_now_lateral':int(((second_junction==0)&primary_to_lateral).sum()),
            'after_cleanup_primary_now_lateral':int(((cleaned==0)&primary_to_lateral).sum())}
    # Probe the difference between two nearest nodes and two distinct roots.
    distances=np.column_stack([cKDTree(path.points).query(p)[0] for path in paths])
    distinct=np.partition(distances,1,axis=1)[:,:2]
    margin=max(.75*spacing,.001); radius=max(4*spacing,.006)
    ambiguous=(~locked)&(~excluded)&(distinct[:,0]<=radius)&(distinct[:,1]-distinct[:,0]<=margin)
    missed=ambiguous&(ll>0)
    gap_rows=[]
    for root in s._baseline_roots.values():
        if root.order!=1:continue
        ix=np.flatnonzero(baseline==root.numeric_label)
        if not len(ix):continue
        insertion=root.points[0]
        gap=float(np.min(np.linalg.norm(s.mesh.positions[ix]-insertion,axis=1)))
        changed=np.flatnonzero((baseline==0)&(ref==root.numeric_label))
        gap_rows.append({'root_id':root.root_id,'insertion_to_own_surface_min':gap,'primary_points_relabelled_to_child':len(changed),
                         'relabelled_points_within_2_mesh_units':int(np.sum(np.linalg.norm(s.mesh.positions[changed]-insertion,axis=1)<2))})
    report={'primary_path_max_error_source_units':float(np.max(np.abs(final_primary-primary))*norm.scale) if final_primary.shape==primary.shape else None,
            'primary_shape_equal':final_primary.shape==primary.shape,
            'replayed_final_label_mismatches':int(np.sum(cleaned!=baseline)),
            'frozen_skeleton_replay':True,'stages':stages,
            'analysis_junction_resolved':jr['resolved_vertex_count'],'full_junction_resolved':jr2['resolved_vertex_count'],
            'patch_cleanup_resolved':cr['absorbed_vertex_count'],
            'distinct_root_ambiguous_count':int(ambiguous.sum()),'missed_by_two_nearest_nodes':int(missed.sum()),
            'missed_and_changed_by_manual_edits':int((missed&(baseline!=ref)).sum()),
            'junction_examples':sorted(gap_rows,key=lambda x:-x['primary_points_relabelled_to_child'])}
    save_json(OUT/'causal_assignment_replay.json',report)
    np.savez_compressed(OUT/'assignment_stage_labels.npz',initial_primary=first,locked_primary=locked,raw=raw,junction=junction,second_junction=second_junction,cleaned=cleaned,missed_ambiguity=missed)
    print(json.dumps({k:v for k,v in report.items() if k!='junction_examples'},indent=2),flush=True)

if __name__=='__main__':main()
