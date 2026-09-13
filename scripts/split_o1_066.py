from pathlib import Path
import json
import shutil
import hashlib
import pandas as pd
import open3d as o3d
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components,dijkstra
from scipy.spatial import cKDTree
from w5168_assignment_audit import BatchSession,edges_of,save_json
from w5168_assignment_repair import graph_for,surface_axis,fit_sections
from soyrootbio.geometry import resample_polyline,path_length
from soyrootbio.editor.session import EditorSession
from soyrootbio.editor.ply import read_labeled_ply

ROOT=Path('E:/SoyRSA Build/outputs/w51683_m4-2_20260415_corrected')
BUNDLE=ROOT/'corrected_bundle'
WORK=ROOT/'o1_066_split'

def inspect():
    WORK.mkdir(exist_ok=True)
    s=BatchSession(BUNDLE,session_dir=WORK/'inspection',load_existing_log=False)
    r=s.roots['root-o1-066']; p=s.mesh.positions; lab=s.mesh.root_labels
    ix=np.flatnonzero(lab==r.numeric_label); cloud=p[ix]
    edges=edges_of(s.mesh.triangles); lookup=np.full(len(p),-1);lookup[ix]=np.arange(len(ix))
    e=lookup[edges[np.all(lab[edges]==r.numeric_label,axis=1)]]
    boundary=edges[((lab[edges[:,0]]==r.numeric_label)&(lab[edges[:,1]]==0))|((lab[edges[:,1]]==r.numeric_label)&(lab[edges[:,0]]==0))]
    seeds=np.unique(lookup[boundary][lookup[boundary]>=0]);g=graph_for(cloud,e)
    dist=dijkstra(g,indices=seeds,min_only=True)
    for threshold in [.15,.25,.35,.45,.6,.8,1.,1.3,1.6,2.]:
        use=dist>threshold; sub=g[use][:,use];n,cc=connected_components(sub,directed=False)
        print(threshold,sorted(np.bincount(cc).tolist(),reverse=True)[:8])
    center=cloud.mean(0);_,_,basis=np.linalg.svd(cloud-center,full_matrices=False)
    q=(cloud-center)@basis.T
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,(a,b) in zip(axes,[(0,1),(0,2),(1,2)]):
        ax.scatter(q[:,a],q[:,b],c=dist,s=12,cmap='turbo');ax.scatter(q[seeds,a],q[seeds,b],c='black',marker='x')
        ax.set_aspect('equal')
    fig.savefig(WORK/'surface_inspection.png',dpi=150);plt.close(fig)
    np.savez(WORK/'surface.npz',indices=ix,points=cloud,edges=e,seeds=seeds,distance=dist)
    return s

def split():
    initial=inspect()
    data=np.load(WORK/'surface.npz');cloud=data['points'];ix=data['indices'];e=data['edges'];dist=data['distance']
    graph=graph_for(cloud,e)
    core=dist>.25
    n,cc=connected_components(graph[core][:,core],directed=False)
    assert n==2
    order=np.argsort(np.bincount(cc))[::-1]
    core_ix=np.flatnonzero(core)
    costs=np.column_stack([dijkstra(graph,indices=core_ix[cc==c],min_only=True) for c in order])
    groups=np.argmin(costs,axis=1)
    parent=initial.roots['primary'].points
    meta=json.loads((BUNDLE/'metadata.json').read_text());spacing=meta['d_bar_normalized']*meta['normalization_scale']
    curves={}; details={}
    for group,rid in enumerate(['root-o1-066','root-o1-077']):
        subset=np.flatnonzero(groups==group);pts=cloud[subset]
        remap=np.full(len(cloud),-1);remap[subset]=np.arange(len(subset))
        edges=remap[e[np.all(remap[e]>=0,axis=1)]]
        assert connected_components(graph_for(pts,edges),directed=False)[0]==1
        hint=pts[np.argmin(cKDTree(parent).query(pts)[0])]
        line,detail,use=surface_axis(pts,edges,hint,spacing)
        line=fit_sections(pts[use],line,spacing)
        attachment=int(cKDTree(parent).query(line[0])[1])
        line=resample_polyline(np.vstack([parent[attachment],line]),spacing*1.5)
        curves[rid]=line;details[rid]={'points':len(subset),'length_mesh_units':path_length(line),'parent_id':'primary','order':1}
    assert 'root-o1-077' not in initial.roots
    backup=WORK/'before_split_bundle'
    if not backup.exists():shutil.copytree(BUNDLE,backup)
    else:
        assert (backup/'segmented_root_structure.ply').read_bytes()==(BUNDLE/'segmented_root_structure.ply').read_bytes()
        assert (backup/'root_hierarchy.json').read_bytes()==(BUNDLE/'root_hierarchy.json').read_bytes()
    s=BatchSession(backup,session_dir=WORK/'repair_session')
    if 'root-o1-077' not in s.roots:
        assert not any(r.parent_id=='root-o1-066' for r in s.roots.values())
        s.apply_operation('split_root',{'root_id':'root-o1-066','position':s.roots['root-o1-066'].points[-3].tolist(),'new_root_id':'root-o1-077'})
    if s.roots['root-o1-077'].parent_id!='primary':
        s.apply_operation('reparent_root',{'root_id':'root-o1-077','new_parent_id':'primary'})
    for group,rid in enumerate(curves):
        s.apply_operation('assign_points',{'root_id':rid,'indices':ix[groups==group].tolist(),'reason':'Two distinct surface branches; split by mesh connectivity outside the primary insertion neighborhood.'})
        s.apply_operation('redraw_root',{'root_id':rid,'points':curves[rid].tolist(),'reason':'Fit centerline to separated branch assignment.'})
    s._validate_state();EditorSession._recompute_traits(s)
    staged=s.export_materialised(WORK/'staged_bundle')
    old=initial.mesh.root_labels;new=s.mesh.root_labels
    untouched=np.ones(len(old),dtype=bool);untouched[ix]=False
    assert np.array_equal(old[untouched],new[untouched])
    assert set(np.flatnonzero(np.isin(new,[s.roots[r].numeric_label for r in curves])))==set(ix.tolist())
    for rid,root in initial.roots.items():
        if rid!='root-o1-066':assert np.array_equal(root.points,s.roots[rid].points),rid
    replay=BatchSession(backup,session_dir=WORK/'repair_session')
    assert np.array_equal(replay.mesh.root_labels,new)
    for rid in s.roots:assert np.array_equal(s.roots[rid].points,replay.roots[rid].points)
    mesh=o3d.t.geometry.TriangleMesh(o3d.core.Tensor(s.mesh.positions,dtype=o3d.core.Dtype.Float32),o3d.core.Tensor(s.mesh.triangles,dtype=o3d.core.Dtype.Int64))
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(mesh)
    for rid in curves:
        path=s.roots[rid].points
        sd=scene.compute_signed_distance(o3d.core.Tensor(resample_polyline(path,.04).astype('float32')),nsamples=3).numpy()
        details[rid].update(outside_mesh_over_0_05=int((sd>.05).sum()),sample_count=len(sd),maximum_outside_mesh=float(max(0,sd.max())))
    center=cloud.mean(0);_,_,basis=np.linalg.svd(cloud-center,full_matrices=False)
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    colors=['#d044aa','#008a85']
    for ax,(a,b) in zip(axes,[(0,1),(0,2),(1,2)]):
        for group,rid in enumerate(curves):
            q=(cloud[groups==group]-center)@basis.T
            line=(s.roots[rid].points-center)@basis.T
            ax.scatter(q[:,a],q[:,b],s=8,c=colors[group],alpha=.55)
            ax.plot(line[:,a],line[:,b],c=colors[group],lw=2,label=rid)
            ax.scatter(line[0,a],line[0,b],c=colors[group],marker='x',s=40)
        ax.set_aspect('equal');ax.legend(fontsize=8)
    fig.suptitle('Two order-1 branches, each attached to primary')
    fig.savefig(WORK/'split_preview.png',dpi=160);plt.close(fig)
    # Install only after validating labels, topology and replay; retain full backup.
    for f in staged.iterdir():
        if f.is_file():shutil.copy2(f,BUNDLE/f.name)
    shutil.copytree(staged/'blobs',BUNDLE/'blobs',dirs_exist_ok=True)
    for src,dst in {'edited_root_hierarchy.json':'root_hierarchy.json','edited_root_traits.csv':'root_traits.csv','edited_root_system.rsml':'root_system.rsml','edited_segmented_root_structure.ply':'segmented_root_structure.ply'}.items():
        shutil.copy2(BUNDLE/src,BUNDLE/dst)
    shutil.copy2(BUNDLE/'edited_root_label_map.csv',BUNDLE/'csv/root_label_map.csv')
    rows=[]
    for root in s.roots.values():
        if root.order==0:continue
        for i,xyz in enumerate(root.points):rows.append({'root_id':root.root_id,'parent_id':root.parent_id,'root_order':root.order,'point_index':i,'x':xyz[0],'y':xyz[1],'z':xyz[2]})
    pd.DataFrame(rows).to_csv(BUNDLE/'lateral_skeletons.csv',index=False)
    meta['selected_lateral_count']=len(s.roots)-1;meta['selected_order_counts']={'1':len(s.roots)-1}
    meta['latest_local_correction']={'type':'two_order1_surface_split','roots':details,'report':'../o1_066_split/result.json','replay_baseline':str(backup)}
    save_json(BUNDLE/'metadata.json',meta)
    loaded=BatchSession(BUNDLE,session_dir=WORK/'reload_check',load_existing_log=False)
    assert np.array_equal(loaded.mesh.root_labels,new)
    assert len(loaded.roots)==76
    for rid in curves:
        r=loaded.roots[rid];assert r.order==1 and r.parent_id=='primary'
        assert np.array_equal(r.points,s.roots[rid].points)
    result={'roots':details,'all_560_points_preserved':len(ix)==560,'other_point_labels_unchanged':True,
            'other_centerlines_unchanged':True,'log_replay_matches':True,'bundle_reload_passed':True,'root_records':len(s.roots),'backup':str(backup)}
    save_json(WORK/'result.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':split()
