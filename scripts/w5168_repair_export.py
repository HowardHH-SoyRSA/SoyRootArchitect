"""Export a replayable repair session and an independently openable bundle."""
from __future__ import annotations
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soyrootbio.editor.session import EditorSession
from soyrootbio.editor.ply import read_labeled_ply
from w5168_assignment_audit import SOURCE,OUT,BatchSession,load_session,save_json


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def comparison(s,curves,labels):
    p=s.mesh.positions
    original=s._baseline_labels; edited=s.mesh.root_labels
    order0={r.numeric_label:r.order for r in s._baseline_roots.values()}
    order1={r.numeric_label:r.order for r in s.roots.values()}
    colors={-2:'#fa7a0d',-1:'#ababab',0:'#0072b2',1:'#ff00ff',2:'#009e73'}
    fig,axes=plt.subplots(2,3,figsize=(15,14),layout='constrained',gridspec_kw={'height_ratios':[3,1]})
    rng=np.random.default_rng(42); sampled=np.sort(rng.choice(len(p),90000,replace=False))
    for col,(title,lab,orders) in enumerate([('Original automatic',original,order0),('Manual point assignments',edited,order1),('Corrected copy',labels,order1)]):
        for row in (0,1):
            ix=sampled if row==0 else np.flatnonzero((p[:,2]>65)&(p[:,2]<82)&(p[:,0]>-6)&(p[:,0]<16))
            ax=axes[row,col]
            c=[colors.get(orders.get(int(lab[i]),int(lab[i])),'#009e73') for i in ix]
            ax.scatter(p[ix,0],p[ix,2],s=.6 if row==0 else 1.5,c=c,rasterized=True)
            if col==2:
                for line in curves.values():
                    shown=line if row==0 else line[(line[:,2]>61)&(line[:,2]<82)]
                    if len(shown)>1:ax.plot(shown[:,0],shown[:,2],c='black',lw=.45)
            ax.set_aspect('equal'); ax.set_title(title+(' | collar' if row else '')); ax.set_xlabel('X'); ax.set_ylabel('Z')
            if row:ax.set_xlim(-6,16);ax.set_ylim(65,82)
    fig.suptitle('Blue: primary | magenta: order 1 | green: order 2 | gray: unassigned\nCoordinates and scale are preserved; units are mesh units')
    fig.savefig(OUT/'segmentation_comparison.png',dpi=170);plt.close(fig)


def main():
    ref=load_session()
    source_files=[SOURCE/'segmented_root_structure.ply',SOURCE/'root_hierarchy.json',SOURCE/'metadata.json',SOURCE/'.soyrootbio-editor/operations.jsonl',SOURCE/'.soyrootbio-editor/materialised/edited_segmented_root_structure.ply']
    hashes={str(p):digest(p) for p in source_files}
    assert digest(SOURCE/'.soyrootbio-editor/operations.jsonl')==digest(OUT/'session/operations.jsonl'), 'Source edit log changed after audit snapshot'
    metrics=json.loads((OUT/'repair_metrics.json').read_text())
    curves=dict(np.load(OUT/'candidate_curves.npz')); labels=np.load(OUT/'candidate_labels.npy')
    target=OUT/'corrected_session'
    if target.exists():
        raise RuntimeError('Corrected session already exists; refuse to append a second repair.')
    shutil.copytree(OUT/'session',target)
    s=BatchSession(SOURCE,session_dir=target)
    for label in np.unique(labels[labels!=s.mesh.root_labels]):
        ix=np.flatnonzero((labels==label)&(labels!=s.mesh.root_labels))
        root=next(r for r in s.roots.values() if r.numeric_label==label)
        s.apply_operation('assign_points',{'root_id':root.root_id,'indices':ix.tolist(),
                          'reason':'Below-collar surface-geodesic propagation from fixed edited ownership.'})
    empty=[r['root_id'] for r in metrics['roots'] if r.get('assigned_points')==0]
    for rid in empty:
        assert not any(r.parent_id==rid for r in s.roots.values())
        s.apply_operation('delete_root',{'root_id':rid,'reason':'Zero assigned vertices in manual reference; remove unsupported skeleton-only record.'})
    for rid in sorted(curves,key=lambda rid:(s.roots[rid].order,rid)):
        s.apply_operation('redraw_root',{'root_id':rid,'points':curves[rid].tolist(),
                          'reason':'Reconstructed from edited assigned surface; drawn path was not used as fit target.',
                          'reconstruction_method':metrics['method']})
    s._validate_state()
    EditorSession._recompute_traits(s)
    export=s.export_materialised(OUT/'corrected_bundle')
    copies={'edited_root_hierarchy.json':'root_hierarchy.json','edited_root_traits.csv':'root_traits.csv',
            'edited_segmented_root_structure.ply':'segmented_root_structure.ply','edited_root_system.rsml':'root_system.rsml'}
    for src,dst in copies.items():shutil.copy2(export/src,export/dst)
    (export/'csv').mkdir(exist_ok=True)
    shutil.copy2(export/'edited_root_label_map.csv',export/'csv/root_label_map.csv')
    shutil.copy2(SOURCE/'primary_guidance.json',export/'primary_guidance.json')
    primary=s.roots['primary']
    pd.DataFrame(primary.points,columns=['x','y','z']).to_csv(export/'primary_skeleton.csv',index=False)
    rows=[]
    for root in s.roots.values():
        if root.root_id=='primary':continue
        for i,xyz in enumerate(root.points):
            rows.append({'root_id':root.root_id,'parent_id':root.parent_id,'root_order':root.order,'point_index':i,'x':xyz[0],'y':xyz[1],'z':xyz[2]})
    pd.DataFrame(rows).to_csv(export/'lateral_skeletons.csv',index=False)
    source_meta=json.loads((SOURCE/'metadata.json').read_text())
    # Preserve historical automatic policy as provenance, never as new results.
    metadata={k:source_meta[k] for k in ('coordinate_unit','coordinate_space','output_length_unit','output_area_unit','output_volume_unit','gravity_vector','normalization_minimum','normalization_scale','d_bar_normalized','source_geometry') if k in source_meta}
    metadata.update(schema='soyrootbio.assignment-repair/v1',source=str(SOURCE),repair_reference_log_sequence=233,
                    point_count=len(labels),full_resolution_point_count=len(labels),
                    config={'tip_vector_window_mesh_units':s.tip_vector_window_mesh_units,'gravity':s.gravity.tolist()},
                    point_assignment={'assigned_vertex_count':int((labels>=0).sum()),'uncertain_vertex_count':int((labels==-2).sum()),'unassigned_vertex_count':int((labels==-1).sum()),
                                      'repair_policy':'Original manual assigned labels locked; eligible negative labels propagated on below-collar mesh edges.',
                                      'above_collar_rule':source_meta['point_assignment']['rule'],
                                      'resolution_report':'../repair_metrics.json'},
                    selected_lateral_count=len(s.roots)-1,selected_order_counts=dict(Counter(r.order for r in s.roots.values() if r.order>0)),
                    reconstruction=metrics['method'],automatic_metadata_provenance=str(SOURCE/'metadata.json'))
    save_json(export/'metadata.json',metadata)
    # Verify both available loading modes, with no mutation of the source bundle.
    replay=BatchSession(SOURCE,session_dir=target)
    assert np.array_equal(replay.mesh.root_labels,labels)
    for rid in s.roots:assert np.array_equal(s.roots[rid].points,replay.roots[rid].points)
    loaded=BatchSession(export,session_dir=OUT/'corrected_bundle_load_check',load_existing_log=False)
    ply=read_labeled_ply(export/'segmented_root_structure.ply')
    assert np.array_equal(ply.positions,ref.mesh.positions) and np.array_equal(ply.triangles,ref.mesh.triangles)
    assert np.array_equal(ply.root_labels,labels)
    assert np.array_equal(labels[ref.mesh.root_labels>=0],ref.mesh.root_labels[ref.mesh.root_labels>=0])
    assert len(loaded.roots)==len(s.roots)
    for rid in s.roots:assert np.array_equal(s.roots[rid].points,loaded.roots[rid].points)
    assert hashes=={str(p):digest(p) for p in source_files}
    summary={'source_file_hashes_unchanged':hashes,'original_manual_log_sha256':digest(OUT/'session/operations.jsonl'),
             'replay_matches_corrected_export':True,'standard_bundle_reload_passed':True,'all_existing_assigned_labels_preserved':True,
             'positions_and_triangles_unchanged':True,'corrected_root_count':len(s.roots),'removed_empty_root_ids':empty,
             'final_log_sequence':s._sequence,'final_states':metadata['point_assignment'],
             'unsupported_tiny_labels':[r['root_id'] for r in metrics['roots'] if 0<r.get('assigned_points',0)<30],
             'maximum_attachment_error':float(max(np.linalg.norm(r.points[0]-s.roots[r.parent_id].points[r.insertion_index]) for r in s.roots.values() if r.parent_id))}
    save_json(OUT/'verification.json',summary)
    np.savez_compressed(OUT/'exported_curves.npz',**{rid:r.points for rid,r in s.roots.items()})
    comparison(ref,{rid:r.points for rid,r in s.roots.items()},labels)
    print(json.dumps({k:v for k,v in summary.items() if k!='source_file_hashes_unchanged'},indent=2))

if __name__=='__main__':main()
