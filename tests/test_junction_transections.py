import numpy as np

from soyrootbio.junction_transections import trim_primary_junctions
from soyrootbio.junction_transections import _preserve_parent_components
from test_primary_o1_ownership import junction


def run(case, **kwargs):
    p,labels,parent,roots,faces,_=case
    return trim_primary_junctions(p,labels,parent,roots,d_bar=.004,triangles=faces,**kwargs)


def test_facing_spike_trims_attached_exterior_to_contacted_child():
    case=junction()
    p,before,_,roots,_,n=case
    path=roots[0].points.copy()
    after,report=run(case)
    target=(np.arange(len(p))>=n)&(p[:,0]>=.055)&(p[:,0]<.10)
    assert report['transferred_vertex_count']>0,report
    assert np.all(after[target]==1),report
    np.testing.assert_array_equal(after[:n],before[:n])
    internal=(np.arange(len(p))>=n)&(p[:,0]<=.035)
    np.testing.assert_array_equal(after[internal],before[internal])
    np.testing.assert_array_equal(after[before!=0],before[before!=0])
    np.testing.assert_array_equal(roots[0].points,path)


def test_surface_onset_can_differ_from_internal_insertion():
    case=junction()
    case[3][0].points[0,0]=.10  # exported body starts at its owned surface
    result,report=run(case)
    assert report['transferred_vertex_count']>0,report
    assert np.all(result[(case[0][:,0]>.06)&(case[0][:,0]<.095)]==1)


def test_missing_mesh_contact_is_not_replaced_by_spatial_proximity():
    case=list(junction())
    p,_,_,_,faces,_=case
    cut=(p[faces,0].min(axis=1)<.10)&(p[faces,0].max(axis=1)>=.10)
    case[4]=faces[~cut]
    after,report=run(case)
    np.testing.assert_array_equal(after,case[1])
    assert report['transferred_vertex_count']==0


def test_excluded_and_uncertain_vertices_do_not_seed_or_bridge():
    case=junction()
    p,labels,_,_,_,n=case
    excluded=(np.arange(len(p))>=n)&(p[:,0]>=.08)&(p[:,0]<.10)
    labels[excluded]=-1
    after,report=run(case,excluded_mask=excluded)
    np.testing.assert_array_equal(after,labels)
    assert report['transferred_vertex_count']==0


def test_no_abrupt_increase_retains_parent_wall():
    case=junction()
    p,labels,_,_,_,n=case
    # Only the parent wall remains primary-owned at emergence.
    labels[(np.arange(len(p))>=n)&(p[:,0]>.04)]=1
    after,report=run(case)
    np.testing.assert_array_equal(after,labels)
    assert report['transferred_vertex_count']==0


def test_needs_parent_sections_on_both_sides():
    case=junction()
    p,labels,_,_,_,n=case
    labels[(np.arange(len(p))<n)&(p[:,2]<-.03)]=-1
    after,report=run(case)
    np.testing.assert_array_equal(after,labels)
    assert report['junctions'][0]['status']=='missing_two_sided_transections'


def test_rejects_a_cut_that_disconnects_parent_wall_support():
    p=np.array([[.1,0,z] for z in range(5)]+[[1.,0,2.]])
    before=np.array([0,0,0,0,0,1]);after=before.copy();after[2]=1
    edges=np.array([[0,1],[1,2],[2,3],[3,4],[2,5]])
    report={'junctions':[dict(label=1,root_id='child',transferred_vertex_count=1,
        section_arcs=[-2.,-1.,5.,6.],section_facing_radius=[.2,.2,.2,.2],
        contact_primary_arc=2.,region_radius=6.,upper_radius=.2,lower_radius=.2,
        region_center=[0.,0.,2.])]}
    _preserve_parent_components(p,before,after,edges,np.array([[0.,0.,0.],[0.,0.,4.]]),.01,report)
    np.testing.assert_array_equal(after,before)
    assert report['connectivity_rejected_root_ids']==['child']


def test_coordinate_rotation_and_scale_do_not_change_ownership():
    case=list(junction());expected,_=run(case)
    angle=.37
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    case[0]=case[0]@rotation.T*3
    case[2]=case[2]@rotation.T*3
    case[3][0].points=case[3][0].points@rotation.T*3
    result,_=trim_primary_junctions(case[0],case[1],case[2],case[3],d_bar=.012,triangles=case[4])
    np.testing.assert_array_equal(result,expected)
