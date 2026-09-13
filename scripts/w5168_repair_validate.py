from __future__ import annotations
import json
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soyrootbio.geometry import resample_polyline, path_length
from w5168_assignment_audit import load_session, OUT, save_json


def main():
    s=load_session()
    curves=dict(np.load(OUT/('exported_curves.npz' if (OUT/'exported_curves.npz').exists() else 'candidate_curves.npz')))
    labels=np.load(OUT/'candidate_labels.npy')
    p=s.mesh.positions
    mesh=o3d.t.geometry.TriangleMesh(o3d.core.Tensor(p,dtype=o3d.core.Dtype.Float32),o3d.core.Tensor(s.mesh.triangles,dtype=o3d.core.Dtype.Int64))
    scene=o3d.t.geometry.RaycastingScene(); scene.add_triangles(mesh)
    aggregate={}
    for stage,paths in [('original_automatic',{rid:r.points for rid,r in s._baseline_roots.items()}),
                        ('manual_reference',{rid:r.points for rid,r in s.roots.items()}),('corrected_export',curves)]:
        total=0; outside=0; outside_length=0; length=0
        for path in paths.values():
            arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
            sample=np.linspace(0,arc[-1],max(2,int(np.ceil(arc[-1]/.11))+1))
            dense=np.column_stack([np.interp(sample,arc,path[:,i]) for i in range(3)])
            sd=scene.compute_signed_distance(o3d.core.Tensor(dense.astype('float32')),nsamples=3).numpy()
            total+=len(sd);outside+=int((sd>.05).sum());length+=arc[-1]
            outside_length+=float(np.sum((sd[:-1]>.05)|(sd[1:]>.05)))*(arc[-1]/(len(sd)-1))
        aggregate[stage]={'roots':len(paths),'stations_at_spacing_at_most_0_11':total,
                          'outside_over_0_05':outside,'outside_fraction':outside/total,
                          'centerline_length_mesh_units':float(length),'exposed_segment_length_upper_estimate':outside_length}
    save_json(OUT/'validation_summary.json',aggregate)
    print('Stage summary',json.dumps(aggregate),flush=True)
    report=[]
    for rid,line in curves.items():
        item={'root_id':rid}
        for key,path in [('before',s.roots[rid].points),('after',line)]:
            dense=resample_polyline(path,.11)
            sd=scene.compute_signed_distance(o3d.core.Tensor(dense.astype('float32')),nsamples=3).numpy()
            item[key]={'stations':len(sd),'outside_over_0_05':int((sd>.05).sum()),'outside_over_0_15':int((sd>.15).sum()),
                       'max_outside':float(max(0,sd.max())),
                       'outside_arc_fractions':(np.flatnonzero(sd>.05)/max(1,len(sd)-1)).tolist()}
        report.append(item)
    save_json(OUT/'mesh_containment_metrics.json',report)
    for key in ('before','after'):
        print(key,{metric:sum(r[key][metric] for r in report) for metric in ('stations','outside_over_0_05','outside_over_0_15')})
    print('Worst',json.dumps(sorted(report,key=lambda r:-r['after']['outside_over_0_05'])[:7],indent=2))
    targets=['root-o1-024','root-o1-066','primary','root-manual-84ec467fed07']
    fig,axes=plt.subplots(4,3,figsize=(18,17),layout='constrained')
    for row,rid in enumerate(targets):
        root=s.roots[rid]
        cloud=p[labels==root.numeric_label]
        center=cloud.mean(0)
        _,_,basis=np.linalg.svd(cloud-center,full_matrices=False)
        # Orient all three views to the root's principal coordinate system.
        for col,(a,b) in enumerate([(0,1),(0,2),(1,2)]):
            ax=axes[row,col]
            xyz=(cloud-center)@basis.T
            ax.scatter(xyz[:,a],xyz[:,b],s=2,c='#babfc9')
            for path,color,title in [(root.points,'#dc563b','Before'),(curves[rid],'#1469b2','Repaired')]:
                q=(path-center)@basis.T
                ax.plot(q[:,a],q[:,b],lw=1,c=color,label=title)
                ax.scatter(q[0,a],q[0,b],marker='x',c=color,s=35)
            ax.set_aspect('equal'); ax.set_title(rid+f' view {col+1}'); ax.legend()
    fig.savefig(OUT/'detail_validation.png',dpi=140); plt.close(fig)

if __name__=='__main__': main()
