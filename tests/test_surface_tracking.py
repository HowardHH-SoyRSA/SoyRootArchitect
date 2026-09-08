import numpy as np

from soyrootbio.surface_tracking import LocalMeshConnectivity, TubeSectionTracker, section_modes, resolve_endpoint_fragments
from soyrootbio.types import RootPath


def tube(center_y, radius=.1, length=2., rings=41, sides=24):
    t=np.arange(sides)*2*np.pi/sides
    points=np.array([[x,center_y+radius*np.cos(a),radius*np.sin(a)] for x in np.linspace(0,length,rings) for a in t])
    faces=[]
    for i in range(rings-1):
        for j in range(sides):
            a=i*sides+j;b=i*sides+(j+1)%sides;c=a+sides;d=b+sides
            faces.extend([[a,b,c],[b,d,c]])
    return points,np.array(faces)


def two_tubes():
    p,f=tube(0);q,g=tube(.25)
    return np.vstack([p,q]),np.vstack([f,g+len(p)]),len(p)


def test_local_connectivity_blocks_close_tube_even_with_distant_crown_link():
    p,f,n=two_tubes()
    f=np.vstack([f,[0,1,n],[1,n,n+1]])
    mesh=LocalMeshConnectivity(p,f)
    anchor=p[20*24]
    local=np.flatnonzero(np.linalg.norm(p-anchor,axis=1)<.4)
    kept=mesh.retain(anchor,local,.4)
    assert len(kept)>20
    assert np.all(kept<n)
    assert np.any(local>=n)


def test_reduced_cloud_uses_full_mesh_connectivity():
    p,f,n=two_tubes();indices=np.arange(0,len(p),3)
    mesh=LocalMeshConnectivity(p[indices],f,full_points=p)
    anchor=p[20*24]
    candidates=np.flatnonzero(np.linalg.norm(p[indices]-anchor,axis=1)<.4)
    assert np.all(indices[mesh.retain(anchor,candidates,.4)]<n)


def test_single_ring_is_not_split_and_fused_double_ring_has_two_modes():
    angles=np.linspace(0,2*np.pi,120,endpoint=False)
    ring=np.column_stack([np.cos(angles),np.sin(angles)])
    assert len(section_modes(ring)[0])==1
    a=ring+[-.85,0];b=ring+[.85,0]
    fused=np.vstack([a[np.linalg.norm(a-[.85,0],axis=1)>.999],b[np.linalg.norm(b-[-.85,0],axis=1)>.999]])
    centers,_=section_modes(fused)
    assert len(centers)==2
    np.testing.assert_allclose(np.sort(centers[:,0]),[-.85,.85],atol=.1)


def test_two_section_identities_persist_through_fusion_and_separation():
    angles=np.linspace(0,2*np.pi,120,endpoint=False)
    ring=np.column_stack([np.cos(angles),np.sin(angles)])
    tracker=TubeSectionTracker();basis=np.array([[0,1,0],[0,0,1]])
    tracked=[]
    for x,gap in enumerate([2.5,2.2,1.9,1.7,1.9,2.2,2.5]):
        a=ring; b=ring+[gap,0]
        boundary=np.vstack([a[np.linalg.norm(a-[gap,0],axis=1)>.999],b[np.linalg.norm(b,axis=1)>.999]])
        tracked.append(tracker.update(boundary,np.array([x,0.,0.]),basis,np.median(boundary,axis=0)))
    assert tracker.multimode_sections==7
    np.testing.assert_allclose(tracked,np.zeros((7,2)),atol=.12)


def test_endpoint_fragment_join_keeps_origins_and_departing_branch():
    p,f,n=two_tubes();surface=LocalMeshConnectivity(p,f)
    x=np.linspace(.1,.8,12)
    root=RootPath('a',np.column_stack([x,np.full(len(x),.1),np.zeros(len(x))]))
    other=RootPath('b',np.column_stack([x,np.full(len(x),.35),np.zeros(len(x))]))
    x2=np.linspace(.85,1.5,12)
    child=RootPath('fragment',np.vstack([[.8,.35,0],np.column_stack([x2,np.full(len(x2),.1),np.zeros(len(x2))])]),parent_id='b',order=2)
    branch=RootPath('branch',np.array([[.7,.1,0],[.75,.1,0],[.8,.15,0],[.85,.2,0],[.9,.25,0]]),parent_id='a',order=2)
    kept=resolve_endpoint_fragments([root,other],[child,branch],surface,.02)
    assert kept==[branch]
    assert root.root_id=='a' and other.root_id=='b'
    np.testing.assert_allclose(root.points[-1],[1.5,.1,0])
    np.testing.assert_allclose(other.points[-1],[.8,.35,0])


def test_growing_trace_stays_on_short_tube_when_neighbor_continues():
    from scipy.spatial import cKDTree
    from soyrootbio.lateral import LateralStart, _grow_one_candidate

    p,f=tube(0,length=1.,rings=21)
    q,g=tube(.25,length=2.,rings=41)
    points=np.vstack([p,q]);faces=np.vstack([f,g+len(p)])
    start=LateralStart(start_id=0,point=p[24],primary_point=p[0],primary_index=0,
                       member_indices=np.arange(24),direction=np.array([1.,0,0]))
    kwargs=dict(points=points,point_tree=cKDTree(points),allowed_mask=np.ones(len(points),dtype=bool),
                start=start,initial_direction=np.array([1.,0,0]),primary_tangent=np.array([1.,0,0]),
                step_length=.08,open_angle=75,max_steps=100,search_radius=.3,limit_primary_angle_to_insertion=True)
    baseline=_grow_one_candidate(**kwargs)
    traced=_grow_one_candidate(**kwargs,surface=LocalMeshConnectivity(points,faces))
    assert baseline.points[:,0].max()>1.2
    assert traced.points[:,0].max()<=1.+1e-12
    assert traced.points[:,1].max()<=.1+1e-12


def test_centering_uses_incoming_mesh_patch_with_both_tubes_in_mask():
    from soyrootbio.primary import refine_primary_centerline

    p,f,n=two_tubes()
    x=np.linspace(.2,1.8,25)
    coarse=np.column_stack([x,np.full(len(x),.1),np.zeros(len(x))])
    centered=refine_primary_centerline(p,np.ones(len(p),dtype=bool),coarse,d_bar=.02,
        min_slice_points=6,surface=LocalMeshConnectivity(p,f),section_tracker=TubeSectionTracker())
    assert np.median(np.abs(centered[3:-3,1]))<.025


def test_nested_cavity_sheet_can_reach_its_outer_tube_without_reaching_neighbor():
    inner,fi=tube(0,radius=.05,length=1.,rings=21)
    outer,fo=tube(0,radius=.15,length=2.)
    neighbor,fn=tube(.4,radius=.15,length=2.)
    points=np.vstack([inner,outer,neighbor])
    faces=np.vstack([fi,fo+len(inner),fn+len(inner)+len(outer)])
    surface=LocalMeshConnectivity(points,faces)
    anchor=inner[18*24]
    indices=np.flatnonzero(np.linalg.norm(points-anchor,axis=1)<.4)
    retained=surface.retain(anchor,indices,.4,direction=np.array([1.,0,0]))
    assert np.any((retained>=len(inner)) & (points[retained,0]>1.))
    assert np.all(retained<len(inner)+len(outer))


def test_ambiguous_endpoint_continuations_are_retained_for_review():
    p,f,n=two_tubes();surface=LocalMeshConnectivity(p,f)
    x=np.linspace(.1,.8,12)
    root=RootPath('a',np.column_stack([x,np.full(len(x),.1),np.zeros(len(x))]))
    x2=np.linspace(.85,1.5,12)
    geometry=np.vstack([[.8,.35,0],np.column_stack([x2,np.full(len(x2),.1),np.zeros(len(x2))])])
    fragments=[RootPath(rid,geometry.copy(),parent_id='b',order=2) for rid in ('one','two')]
    before=root.points.copy()
    assert resolve_endpoint_fragments([root],fragments,surface,.02)==fragments
    np.testing.assert_array_equal(root.points,before)
    assert 'endpoint_continuation_ambiguous' in root.qc_flags


def test_tapering_cavity_cap_uses_multiple_sections_to_find_enclosing_tube():
    inner,fi=tube(0,radius=.07,length=1.,rings=101)
    # A rounded cavity termination has a disk-like section rather than a ring.
    taper=np.sqrt(np.clip((1.-inner[:,0])/.12, .005, 1.))
    inner[:,1:]*=taper[:,None]
    outer,fo=tube(0,radius=.20,length=1.5,rings=101)
    neighbor,fn=tube(.45,radius=.20,length=1.5,rings=101)
    points=np.vstack([inner,outer,neighbor])
    faces=np.vstack([fi,fo+len(inner),fn+len(inner)+len(outer)])
    surface=LocalMeshConnectivity(points,faces)
    anchor=inner[-24]
    indices=np.flatnonzero(np.linalg.norm(points-anchor,axis=1)<.4)
    retained=surface.retain(anchor,indices,.4,direction=np.array([1.,0,0]))
    assert np.any((retained>=len(inner)) & (points[retained,0]>1.1))
    assert np.all(retained<len(inner)+len(outer))


def test_coincident_vertices_preserve_separate_mesh_sheet_indices():
    p,f=tube(0,length=.5,rings=11)
    vertices=np.vstack([p,p]);faces=np.vstack([f,f+len(p)])
    surface=LocalMeshConnectivity(vertices,faces,full_points=vertices.copy())
    anchor=len(p)+5*24
    candidates=np.flatnonzero(np.linalg.norm(vertices-vertices[anchor],axis=1)<.2)
    kept=surface.retain(vertices[anchor],candidates,.2,anchor_index=anchor)
    assert len(kept)>10
    assert np.all(kept>=len(p))
