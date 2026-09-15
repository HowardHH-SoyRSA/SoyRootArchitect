"""Apply branch-facing transection trimming to complete immutable bundles."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from soyrootbio.editor.ply import read_labeled_ply
from soyrootbio.export import export_results
from soyrootbio.junction_transections import trim_primary_junctions
from soyrootbio.pipeline import _selected_base_exclusion_mask, _surface_connectivity_edges
from soyrootbio.traits import compute_traits
from soyrootbio.types import Normalization, RootPath
from validate_primary_o1_ownership import BatchSession, component_report, digest, save

SAMPLES=['SN14_6-2_20260405_5','Kaixinlv_3-2_20260525_7','W82_MS4-2_20260617_4',
         'W82_9cm_water_1-2_20260522_5','w5168-3_m4-2_20260415_6','BaxiNo2_4-2_20260525_5']


def run(source, out):
    out.mkdir(parents=True,exist_ok=False)
    files=[p for p in source.rglob('*') if p.is_file()]
    hashes={str(p):digest(p) for p in files}
    meta=json.loads((source/'metadata.json').read_text())
    session=BatchSession(source,session_dir=out/'session',load_existing_log=False)
    oldroots={rid:r.clone() for rid,r in session.roots.items()}
    norm=Normalization(np.array(meta['normalization_minimum']),meta['normalization_scale'])
    p=norm.transform_points(session.mesh.positions)
    parent=norm.transform_points(oldroots['primary'].points)
    laterals=sorted((r for rid,r in oldroots.items() if rid!='primary'),key=lambda r:r.numeric_label)
    numeric=np.array([0]+[r.numeric_label for r in laterals])
    assert np.array_equal(numeric,np.arange(len(numeric))), 'exporter needs contiguous source labels'
    paths=[RootPath(root_id=r.root_id,parent_id=r.parent_id,order=r.order,
        points=norm.transform_points(r.points),parent_points=norm.transform_points(oldroots[r.parent_id].points),
        insertion_point=None if r.insertion_point is None else norm.transform_points(r.insertion_point[None])[0],
        insertion_index=r.insertion_index,body_start_index=r.body_start_index,
        confidence=r.confidence,qc_flags=list(r.qc_flags),centerline_assessment=deepcopy(r.centerline_assessment)) for r in laterals]
    before=session.mesh.root_labels.copy()
    a=meta['point_assignment']
    excluded=_selected_base_exclusion_mask(p,norm.transform_points(np.array(a['base_point_source_coordinates'])[None])[0],
        np.array(a['base_tipward_direction']),gravity=np.array(a['gravity_direction']),
        collar_neighborhood_radius=a['base_collar_neighborhood_radius_normalized'],tolerance=a['above_base_tolerance_normalized'])
    started=time.perf_counter()
    after,stage=trim_primary_junctions(p,before,parent,paths,d_bar=meta['d_bar_normalized'],
        triangles=session.mesh.triangles,excluded_mask=excluded)
    save(out/'transection_decisions.json',stage)
    np.savez_compressed(out/'assignment_changes.npz',before=before,after=after,changed_indices=np.flatnonzero(before!=after))
    changed=before!=after
    assert np.all(before[changed]==0) and np.all(after[changed]>0)
    assert np.array_equal(after[excluded|(before!=0)],before[excluded|(before!=0)])
    affected_labels=set(before[changed])|set(after[changed])
    affected={rid for rid,r in oldroots.items() if r.numeric_label in affected_labels}
    for label in np.unique(after[changed]):
        r=laterals[int(label)-1]
        session.apply_operation('assign_points',{'root_id':r.root_id,
            'indices':np.flatnonzero(changed&(after==label)).tolist(),
            'reason':'Abrupt branch-facing transection radius increase; trim beyond two-sided parent envelope to contacted O1 via mesh edges.'})
    replay=BatchSession(source,session_dir=out/'session')
    np.testing.assert_array_equal(replay.mesh.root_labels,after)
    for rid,root in oldroots.items():
        for s in [session,replay]:
            now=s.roots[rid]
            np.testing.assert_array_equal(now.points,root.points)
            assert (now.parent_id,now.order,now.insertion_index,now.body_start_index)==(root.parent_id,root.order,root.insertion_index,root.body_start_index)
    def traits(labels):
        return compute_traits(parent,paths,p,labels==0,labels,norm,
            full_points=session.mesh.positions,triangles=session.mesh.triangles,full_root_labels=labels,
            mesh_metadata=meta.get('source_geometry',{}),primary_confidence=oldroots['primary'].confidence,
            primary_qc_flags=oldroots['primary'].qc_flags,primary_centerline_assessment=oldroots['primary'].centerline_assessment,
            tip_vector_window=session.tip_vector_window_mesh_units,gravity=session.gravity)
    tb,ta=traits(before),traits(after)
    b,a=tb.set_index('root_id').sort_index(),ta.set_index('root_id').sort_index()
    unchanged=sorted(set(oldroots)-affected)
    pd.testing.assert_frame_equal(b.loc[unchanged],a.loc[unchanged],check_exact=True)
    trait_changes=[]
    for rid in b.index:
        for col in b.columns:
            old,new=b.loc[rid,col],a.loc[rid,col]
            if (pd.isna(old) and pd.isna(new)) or old==new:continue
            assert col in {'point_count','mean_radius','mean_diameter','median_diameter','minimum_diameter','maximum_diameter','surface_area','volume'},(rid,col)
            trait_changes.append(dict(root_id=rid,trait=col,before=old,after=new))
    tb.to_csv(out/'traits_before.csv',index=False);ta.to_csv(out/'traits_after.csv',index=False)
    save(out/'trait_changes.json',trait_changes)
    e,_=_surface_connectivity_edges(p,session.mesh.triangles,meta['d_bar_normalized'])
    cb,ca=component_report(before,e,oldroots),component_report(after,e,oldroots)
    regressions=[aa['root_id'] for bb,aa in zip(cb,ca) if aa['components']>bb['components']]
    metadata=deepcopy(meta)
    metadata.update(branch_facing_transection_trimming=stage,ownership_source_bundle=str(source),
                    correction_geometry_policy='existing paths, insertions and hierarchy frozen; support traits recomputed')
    metadata['point_assignment'].update(primary_assigned_vertex_count=int(np.sum(after==0)),
        lateral_assigned_vertex_count=int(np.sum(after>0)),assigned_vertex_count=int(np.sum(after>=0)),
        uncertain_vertex_count=int(np.sum(after==-2)),unassigned_vertex_count=int(np.sum(after==-1)))
    metadata['final_centerline_fitting']['assignment_changed_since_fit']=bool(changed.any())
    metadata['final_centerline_fitting']['changed_surface_root_ids']=sorted(affected)
    metadata['selected_lateral_count']=len(paths)
    metadata['selected_order_counts']=dict(Counter(r.order for r in paths))
    bundle=out/'bundle'
    export_results(bundle,session.mesh.positions,parent,paths,after==0,np.where(after>0,after,0),ta,norm,metadata,
                   full_points=session.mesh.positions,triangles=session.mesh.triangles,full_root_labels=after)
    loaded=BatchSession(bundle,session_dir=out/'reload',load_existing_log=False)
    np.testing.assert_array_equal(loaded.mesh.root_labels,after)
    np.testing.assert_array_equal(loaded.mesh.positions,session.mesh.positions)
    np.testing.assert_array_equal(loaded.mesh.triangles,session.mesh.triangles)
    cloud=read_labeled_ply(bundle/'primary_points.ply')
    np.testing.assert_array_equal(cloud.positions,session.mesh.positions[after==0])
    assert set(loaded.roots)==set(oldroots)
    for rid,root in oldroots.items():
        np.testing.assert_allclose(loaded.roots[rid].points,root.points,rtol=0,atol=1e-12)
        assert (loaded.roots[rid].parent_id,loaded.roots[rid].order)==(root.parent_id,root.order)
    assert hashes=={str(p):digest(p) for p in files}
    states=lambda x:dict(primary=int(np.sum(x==0)),lateral=int(np.sum(x>0)),unassigned=int(np.sum(x==-1)),uncertain=int(np.sum(x==-2)))
    report=dict(sample=source.name,source=str(source),bundle=str(bundle),source_hashes=hashes,source_files_unchanged=True,
        transferred_vertices=int(changed.sum()),affected_o1=len(affected-{'primary'}),
        before=states(before),after=states(after),root_count=len(oldroots),order_counts=dict(Counter(r.order for r in oldroots.values())),
        junction_status_counts=dict(Counter(row['status'] for row in stage['junctions'])),
        negative_excluded_other_lateral_labels_unchanged=True,all_unaffected_root_traits_identical=True,
        hierarchy_and_paths_unchanged=True,full_mesh_unchanged=True,operation_replay_and_export_reload_match=True,
        surface_component_count_regressions=regressions,components_before=cb,components_after=ca,
        elapsed_seconds=time.perf_counter()-started)
    save(out/'validation.json',report)
    preview(p,before,after,stage,out)
    print(json.dumps({k:v for k,v in report.items() if k not in {'source_hashes','components_before','components_after'}},indent=2),flush=True)
    return report


def preview(p,before,after,stage,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=sorted([r for r in stage['junctions'] if r['transferred_vertex_count']>0],key=lambda r:-r['transferred_vertex_count'])[:4]
    if not rows:return
    fig,axes=plt.subplots(len(rows),3,figsize=(15,3.8*len(rows)),squeeze=False,layout='constrained')
    for n,row in enumerate(rows):
        center=np.array(row['region_center']);ids=np.flatnonzero(np.linalg.norm(p-center,axis=1)<row['region_radius']*.7)
        cloud=p[ids]-center;_,_,basis=np.linalg.svd(cloud[before[ids]==0],full_matrices=False);q=cloud@basis.T
        for col,labels in enumerate([before,after]):
            colors=np.where(labels[ids]==0,'#126bac',np.where(labels[ids]==row['label'],'#c21b97','#bbbbbb'))
            axes[n,col].scatter(q[:,0],q[:,1],s=3,c=colors);axes[n,col].set_aspect('equal')
            axes[n,col].set_title(row['root_id']+(' before' if col==0 else f" after: {row['transferred_vertex_count']} moved"))
        x=np.array(row['section_arcs'])-row['contact_primary_arc']
        axes[n,2].plot(x,np.array(row['section_facing_radius'],float),'-o',label='Branch-facing q80',ms=3)
        axes[n,2].plot(x,np.array(row['section_opposite_radius'],float),label='Opposite median')
        axes[n,2].axhline(row['expected_radius'],c='black',ls='--',label='Parent-wall estimate')
        axes[n,2].set_xlabel('Primary arc relative to contact');axes[n,2].legend(fontsize=8)
    fig.savefig(out/'before_after.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('E:/Seafile/Test files for BioInsAlgo/SoyRootBio_outputs'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',nargs='*',default=SAMPLES)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for sample in args.samples:
        print('Examining '+sample,flush=True)
        reports.append(run(args.base/sample,args.output/sample))
        save(args.output/'summary.json',reports)
