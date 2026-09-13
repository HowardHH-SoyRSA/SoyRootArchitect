from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
from w5168_assignment_repair import surface_axis,fit_sections,resolve_unlabelled


def test_bent_tube_reconstruction_uses_surface_without_old_skeleton():
    t=np.linspace(0,np.pi*.75,130)
    angle=np.linspace(0,2*np.pi,30,endpoint=False)
    centers=np.column_stack([5*np.cos(t),5*np.sin(t),np.zeros(len(t))])
    radial=np.column_stack([np.cos(t),np.sin(t),np.zeros(len(t))])
    points=(centers[:,None,:]+.3*np.cos(angle)[None,:,None]*radial[:,None,:]+.3*np.sin(angle)[None,:,None]*np.array([0,0,1])).reshape(-1,3)
    edges=[]
    for i in range(len(t)):
        for j in range(len(angle)):
            a=i*len(angle)+j; b=i*len(angle)+(j+1)%len(angle)
            edges.append([a,b])
            if i+1<len(t):edges.append([a,a+len(angle)])
    line,info,use=surface_axis(points,np.array(edges),centers[0],.06)
    line=fit_sections(points[use],line,.06)
    d=cKDTree(centers).query(line[4:-4])[0]
    assert np.quantile(d,.95)<.15
    assert np.linalg.norm(line[0]-centers[0])<.35
    assert np.linalg.norm(line[-1]-centers[-1])<.35


def test_assignment_respects_collar_and_disconnected_vertices():
    positions=np.array([[0,0,-1],[0,0,-2],[0,0,-3],[0,0,1],[10,0,-2],[0,0,2]],dtype=float)
    labels=np.array([2,-2,-1,-2,-1,4])
    session=SimpleNamespace(mesh=SimpleNamespace(positions=positions,root_labels=labels))
    meta={'normalization_scale':1,'point_assignment':{
        'base_point_source_coordinates':[0,0,0],'base_tipward_direction':[0,0,-1],
        'gravity_direction':[0,0,-1],'base_collar_neighborhood_radius_normalized':3,
        'above_base_tolerance_normalized':1e-9}}
    result,above,report=resolve_unlabelled(session,np.array([[0,1],[1,2],[0,3],[3,5]]),meta)
    assert result.tolist()==[2,2,2,-2,-1,4]
    assert report['resolved_uncertain']==1 and report['resolved_unassigned']==1
    assert report['unreachable_below']==1
    assert np.array_equal(labels,[2,-2,-1,-2,-1,4])


def test_remote_same_label_island_does_not_pull_axis_across_gap():
    x=np.linspace(0,5,80)
    a=np.linspace(0,2*np.pi,12,endpoint=False)
    p=np.stack(np.broadcast_arrays(x[:,None],.2*np.cos(a)[None,:],.2*np.sin(a)[None,:]),axis=-1).reshape(-1,3)
    island=p[:80]+np.array([0,10,0])
    cloud=np.vstack([p,island])
    # Local geometric edges reproduce both disconnected surface components.
    edges=np.array(list(cKDTree(cloud).query_pairs(.14)),dtype=int)
    line,info,use=surface_axis(cloud,edges,p[0],.06)
    assert info['island_points_excluded_from_axis']>=80
    assert np.max(line[:,1])<1
