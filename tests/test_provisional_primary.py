import numpy as np
from soyrootbio.provisional_primary import (
    release_observed_primary_branches, release_mapped_primary_branches,
)
from test_fork_patch_reconciliation import _wall_fixture
from test_primary_o1_ownership import junction


def test_measured_flat_primary_is_not_reopened_for_discovery():
    p,_,primary,_,faces=_wall_fixture()
    p[:,0]*=1.8;p[:,1]*=.5
    before=np.ones(len(p),bool)
    after,report=release_observed_primary_branches(p,before,primary,d_bar=.15,triangles=faces)
    np.testing.assert_array_equal(after,before)
    assert report['released_vertex_count']==0


def test_absent_native_mesh_does_not_release_provisional_surface():
    p,labels,primary,_,_,_=junction()
    before=labels==0
    after,report=release_observed_primary_branches(p,before,primary,d_bar=.004,triangles=None)
    np.testing.assert_array_equal(after,before)
    assert report['status']=='unresolved_no_mesh'


def test_native_branch_can_be_reopened_without_moving_source_geometry():
    p,_,primary,_,faces,n=junction()
    # Close the distal tip. An open terminal ring must remain unresolved:
    # removing its shaft would strand unsafe primary-owned vertices.
    ring=np.arange(len(p)-24,len(p));cap=len(p)
    faces=np.vstack([faces,[[ring[i],cap,ring[(i+1)%24]] for i in range(24)]])
    p=np.vstack([p,[.35,0,0]])
    before=np.ones(len(p),bool)
    xyz=p.copy();f=faces.copy();axis=primary.copy()
    after,report=release_observed_primary_branches(p,before,primary,d_bar=.004,triangles=faces)
    assert report['released_vertex_count']>0,report
    assert np.all(after[:n])
    np.testing.assert_array_equal(p,xyz);np.testing.assert_array_equal(faces,f)
    np.testing.assert_array_equal(primary,axis)


def test_open_branch_is_retained_with_connectivity_qc():
    p,_,primary,_,faces,_=junction()
    before=np.ones(len(p),bool)
    after,report=release_observed_primary_branches(p,before,primary,d_bar=.004,triangles=faces)
    np.testing.assert_array_equal(after,before)
    assert report['unresolved_candidate_count']>0
    assert report['decisions'][0]['status']=='unresolved_primary_connectivity'


def test_collar_exclusion_cannot_be_released_as_a_branch():
    p,_,primary,_,faces,n=junction()
    before=np.ones(len(p),bool);excluded=np.arange(len(p))>=n
    after,report=release_observed_primary_branches(p,before,primary,d_bar=.004,triangles=faces,excluded_mask=excluded)
    np.testing.assert_array_equal(after[excluded],before[excluded])


def test_unmapped_native_vertices_are_barriers_and_never_interpolated(monkeypatch):
    from soyrootbio import provisional_primary
    points=np.arange(30,dtype=float).reshape(-1,3)
    mapping=np.array([8,2,5]);mask=np.array([True,False,True])
    def observe(source,native,path,**kwargs):
        np.testing.assert_array_equal(np.flatnonzero(native),[5,8])
        np.testing.assert_array_equal(np.flatnonzero(~kwargs['excluded_mask']),[2,5,8])
        native=native.copy();native[5]=False
        return native,dict(released_vertex_count=1)
    monkeypatch.setattr(provisional_primary,'release_observed_primary_branches',observe)
    after,report=release_mapped_primary_branches(points,mask,points[:2],
        analysis_to_mesh=mapping,d_bar=.1,triangles=None)
    np.testing.assert_array_equal(after,[True,False,False])
    assert report['unmapped_native_vertex_count']==7
    np.testing.assert_array_equal(mask,[True,False,True])
