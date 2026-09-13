import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from w5168_assignment_audit import load_session,OUT,edges_of,save_json
from w5168_assignment_repair import graph_for

s=load_session(); p=s.mesh.positions; labels=np.load(OUT/'candidate_labels.npy')
edges=edges_of(s.mesh.triangles); _,cc=connected_components(graph_for(p,edges),directed=False)
primary=np.flatnonzero(labels==0); tree=cKDTree(p[primary]); primary_cc=np.unique(cc[primary])
report=[]
for rid in ['root-o1-027','root-o1-066','root-o1-065','root-o1-004','root-o1-024']:
    r=s.roots[rid]; ix=np.flatnonzero(labels==r.numeric_label); d,j=tree.query(p[ix]); e=edges[(labels[edges[:,0]]==r.numeric_label)&(labels[edges[:,1]]==0)|(labels[edges[:,1]]==r.numeric_label)&(labels[edges[:,0]]==0)]
    report.append({'root_id':rid,'same_full_mesh_component_as_primary':int(np.isin(cc[ix],primary_cc).sum()),'point_count':len(ix),
                   'min_parent_surface_distance':float(d.min()),'parent_boundary_edges':len(e),
                   'full_component_counts':list(zip(*[x.tolist() for x in np.unique(cc[ix],return_counts=True)]))})
repaired=np.load(OUT/'candidate_labels.npy')
pending=np.flatnonzero((repaired<0)&(p[:,2]<78.130592))
known=np.flatnonzero(repaired>=0); dist,j=cKDTree(p[known]).query(p[pending])
pending_report=[{'vertex':int(i),'label':int(repaired[known[k]]),'distance':float(d),'position':p[i].tolist()} for i,d,k in zip(pending,dist,j)]
save_json(OUT/'remaining_support.json',{'roots':report,'below_global_height_unassigned':pending_report})
print(report)
print('Nearby unassigned',sorted(pending_report,key=lambda x:x['distance'])[:30])
