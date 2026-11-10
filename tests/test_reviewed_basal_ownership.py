import numpy as np
import pytest
from soyrootbio.basal_ownership import BasalRecoveryParameters, recover_reviewed_basal_ownership


def strip():
    p=np.array([[x,y,0.] for x in range(6) for y in [0.,.1]])
    f=[]
    for x in range(5):
        a=2*x;f.extend([[a,a+1,a+2],[a+1,a+2,a+3]])
    return p,np.array(f)


def run(p,f,labels,prior,*,body=None,independent=None,excluded=None,**kwargs):
    n=len(p)
    return recover_reviewed_basal_ownership(p,labels,f,reviewed_preorder_labels=prior,
        reviewed_parent_paths={1:np.array([[0.,.05,0.],[5.,.05,0.]])},
        parent_body_masks={1:labels==1 if body is None else body},parent_radii={1:.2},
        label_orders={0:0,1:1,2:2,3:2},independent_body_masks={} if independent is None else independent,
        excluded_mask=np.zeros(n,bool) if excluded is None else excluded,
        immutable_primary_top=[0.,0.,1.],gravity=[0.,0.,-1.],spacing=.1,
        parameters=BasalRecoveryParameters(6.),**kwargs)


def test_recovers_reviewed_neck_while_preserving_independent_distal_child():
    p,f=strip();labels=np.array([1,1,-1,-1,-1,-1,2,2,2,2,2,2]);prior=np.ones(12,int)
    independent=np.zeros(12,bool);independent[8:]=True
    after,r=run(p,f,labels,prior,independent={2:independent})
    np.testing.assert_array_equal(after[:8],1)
    np.testing.assert_array_equal(after[8:],2)
    assert r['changed_vertex_count']==6
    assert sum(x['committed_vertex_count'] for x in r['decisions'])==6


def test_excluded_parent_neighbour_remains_visible_but_cannot_change():
    p,f=strip();labels=np.array([1,1,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1]);prior=np.ones(12,int)
    excluded=np.zeros(12,bool);excluded[:2]=True
    after,r=run(p,f,labels,prior,excluded=excluded)
    np.testing.assert_array_equal(after[:6],1)
    np.testing.assert_array_equal(after[6:],labels[6:])


def test_whole_donor_supersession_requires_explicit_review():
    p,f=strip();labels=np.r_[1,1,np.full(10,2)];prior=np.ones(12,int)
    after,r=run(p,f,labels,prior)
    np.testing.assert_array_equal(after,labels)
    allowed,r=run(p,f,labels,prior,supersedable_labels=[2])
    np.testing.assert_array_equal(allowed,1)
    assert r['superseded_surface_labels']==[2]


def test_detached_parent_fragment_is_not_a_receiving_body_anchor():
    p,f=strip();labels=np.array([1,1,-2,-2,-2,-2,1,1,-1,-1,-1,-1]);prior=np.ones(12,int)
    body=np.zeros(12,bool);body[:2]=True
    after,r=run(p,f,labels,prior,body=body)
    np.testing.assert_array_equal(after,labels)
    assert sum(x['vertices'] for x in r['recipient_rejections'])==4


def test_whole_donor_rollback_cannot_leave_a_disconnected_recipient_bridge():
    p,f=strip();labels=np.array([1,1,2,2,2,2,-1,-1,-1,-1,-1,-1]);prior=np.ones(12,int)
    after,r=run(p,f,labels,prior)
    np.testing.assert_array_equal(after,labels)
    assert sum(x['vertices'] for x in r['recipient_rejections'])==6


def test_above_top_and_uncertain_vertices_are_never_claimed():
    p,f=strip();labels=np.array([1,1,-1,-1,-2,-2,-1,-1,-1,-1,-1,-1]);prior=np.ones(12,int)
    p[2:4,2]=2.
    after,_=run(p,f,labels,prior)
    np.testing.assert_array_equal(after,labels)


def test_no_native_mesh_route_does_not_create_a_bridge():
    p,f=strip();labels=np.array([1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]);prior=np.ones(12,int)
    f=f[2:]
    after,_=run(p,f,labels,prior)
    np.testing.assert_array_equal(after,labels)


def test_unreviewed_owner_cannot_be_overwritten_by_envelope_alone():
    p,f=strip();labels=np.array([1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]);prior=np.full(12,-99)
    after,_=run(p,f,labels,prior)
    np.testing.assert_array_equal(after,labels)


def test_invalid_bounds_are_rejected():
    p,f=strip();labels=np.ones(12,int)
    with pytest.raises(ValueError,match='body witness'):
        run(p,f,labels,labels,body=np.zeros(12,bool))


def test_surviving_independent_donor_component_cannot_be_split():
    p,f=strip();labels=np.tile([1,2],6);prior=np.full(12,-99,int);prior[7]=1
    independent=labels==2;independent[7]=False
    after,r=run(p,f,labels,prior,independent={2:independent})
    np.testing.assert_array_equal(after,labels)
    assert any(x['reason']=='retained_to_avoid_splitting_surviving_donor' for x in r['donor_rejections'])


def test_recipient_rollback_rechecks_donor_components_until_stable():
    # Donor chain: retained 2 -- proposed-to-parent1 3 -- proposed-to-parent4 4.
    # Vertex4 touches only a detached parent4 patch, so its rollback must also
    # roll back vertex3 or the donor becomes two disconnected fragments.
    p=np.column_stack([np.arange(12),np.arange(12)%2,np.zeros(12)])
    faces=np.array([[0,1,3],[2,3,9],[3,4,10],[4,5,6],[7,8,11]])
    before=np.array([1,1,2,2,2,4,4,4,4,-2,-2,-2]);prior=np.full(12,-99)
    prior[3]=1;prior[4]=4
    body1=np.isin(np.arange(12),[0,1]);body4=np.isin(np.arange(12),[7,8])
    independent=np.zeros(12,bool);independent[2]=True
    after,report=recover_reviewed_basal_ownership(p,before,faces,reviewed_preorder_labels=prior,
        reviewed_parent_paths={l:np.array([[0.,0.,0.],[12.,0.,0.]]) for l in [1,4]},
        parent_body_masks={1:body1,4:body4},parent_radii={1:2.,4:2.},label_orders={1:1,2:2,4:1},
        independent_body_masks={2:independent},excluded_mask=np.zeros(12,bool),
        immutable_primary_top=[0.,0.,1.],gravity=[0.,0.,-1.],spacing=.1,
        parameters=BasalRecoveryParameters(15.))
    np.testing.assert_array_equal(after,before)
    assert any(x['reason']=='retained_donor_after_recipient_rollback' for x in report['donor_rejections'])
