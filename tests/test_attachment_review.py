from copy import deepcopy
import json

import numpy as np
import pytest

from soyrootbio.attachment_review import connected_body_evidence, review_basal_attachments
from soyrootbio.mesh_geometry import MeshGeometryContext
from soyrootbio.types import RootPath


def detached_parent_patch():
    theta=np.arange(24)*2*np.pi/24
    x=np.linspace(0.,.35,71)
    tube=np.array([[s,.01*np.cos(t),.01*np.sin(t)] for s in x for t in theta])
    points=np.vstack([tube,tube+[0.,.15,0.]])
    faces=[]
    for offset in (0,len(tube)):
        for i in range(len(x)-1):
            for j in range(24):
                a,b=offset+24*i+j,offset+24*i+(j+1)%24
                faces.extend([[a,b,a+24],[b,b+24,a+24]])
    labels=np.r_[np.where(tube[:,0]<.10,0,2),np.ones(len(tube),int)]
    domain=(np.arange(len(points))<len(tube))&(points[:,0]>=.06)&(points[:,0]<.10)
    labels[domain]=-2
    fragment=domain&(points[:,0]>=.075)&(points[:,0]<.08)&(points[:,1]>0)
    labels[fragment]=1
    primary=np.array([[.035,0.,.4],[.035,0.,0.],[.035,0.,-.4]])
    roots=[RootPath('parent',np.array([[0.,.15,0.],[.35,.15,0.]]),order=1),
           RootPath('child',np.array([[.09,0.,0.],[.35,0.,0.]]),parent_id='parent',order=2)]
    return points,labels,primary,roots,np.asarray(faces),domain,fragment


def run(case,**kwargs):
    points,labels,primary,roots,faces,*_=case
    return review_basal_attachments(points,labels,primary,roots,d_bar=.004,triangles=faces,
                                   primary_top_reference=primary[0],**kwargs)


def test_independent_primary_body_reparents_and_preserves_root_identity():
    case=detached_parent_patch()
    p,before,primary,roots,faces,domain,fragment=case
    frozen=deepcopy(case)
    after,updated,report=run(case)
    assert report['changed_parent_count']==1
    assert updated[1].root_id=='child' and updated[1].parent_id=='primary' and updated[1].order==1
    assert np.all(after[domain]==2)
    np.testing.assert_array_equal(after[~domain],before[~domain])
    np.testing.assert_array_equal(p,frozen[0]);np.testing.assert_array_equal(faces,frozen[4])
    assert roots[1].parent_id=='parent'
    for old,new in zip(roots,updated):np.testing.assert_array_equal(old.points,new.points)
    json.dumps(report,allow_nan=False)


def test_detached_patch_cannot_supply_connected_parent_evidence():
    p,labels,primary,roots,faces,_,fragment=detached_parent_patch()
    owner,_,rows=connected_body_evidence(p,labels,primary,roots,d_bar=.004,
                                        mesh_context=MeshGeometryContext.build(p,faces))
    assert np.all(owner[fragment]==-1)
    assert rows[1]['body_component_count']==1 and rows[1]['detached_vertex_count']==fragment.sum()


def test_root_list_permutation_does_not_change_proposals():
    case=list(detached_parent_patch());expected,_,_=run(case)
    case[3]=case[3][::-1]
    case[1]=np.where(case[1]==1,2,np.where(case[1]==2,1,case[1]))
    actual,updated,report=run(case)
    actual=np.where(actual==1,2,np.where(actual==2,1,actual))
    np.testing.assert_array_equal(actual,expected)
    assert next(r for r in updated if r.root_id=='child').parent_id=='primary'


@pytest.mark.parametrize('barrier',['excluded','unassigned','missing_faces','above_top'])
def test_uncertainty_is_not_a_bridge_without_native_measured_support(barrier):
    case=list(detached_parent_patch());p,labels,primary,roots,faces,domain,_=case
    cut=(p[:,0]>=.085)&(p[:,0]<.09)&domain
    kwargs={}
    if barrier=='excluded':kwargs['excluded_mask']=cut
    elif barrier=='unassigned':labels[cut]=-1
    elif barrier=='missing_faces':
        crossing=(p[faces,0].min(axis=1)<.085)&(p[faces,0].max(axis=1)>=.085)&np.any(domain[faces],axis=1)
        case[4]=faces[~crossing]
    else:primary[0,2]=-.1
    after,updated,report=run(case,**kwargs)
    np.testing.assert_array_equal(after,labels)
    assert updated[1].parent_id=='parent' and report['changed_parent_count']==0


def test_genuine_parent_body_contact_keeps_child_order():
    case=list(detached_parent_patch());p,labels,primary,roots,faces,domain,fragment=case
    # The first cylinder itself is now an O1 parent followed by its O2 branch.
    labels[(labels==0)|(labels==-2)]=1
    roots[0].points=np.array([[0.,0.,0.],[.09,0.,0.]])
    # Separate surface at y=.15 is unassigned, not a second parent body.
    labels[p[:,1]>.1]=-1
    after,updated,report=run(case)
    np.testing.assert_array_equal(after,labels)
    assert updated[1].parent_id=='parent' and updated[1].order==2


def test_no_mesh_keeps_uncertain_topology():
    case=list(detached_parent_patch());case[4]=None
    after,updated,report=run(case)
    np.testing.assert_array_equal(after,case[1])
    assert report['status']=='unresolved_no_native_mesh'


def test_nonmanifold_junction_stays_unresolved_even_with_a_closed_tube():
    case=list(detached_parent_patch());p,labels,_,_,faces,domain,_=case
    which=np.flatnonzero(np.all(domain[faces],axis=1))[0]
    case[4]=np.vstack([faces,faces[which]])
    after,updated,report=run(case)
    np.testing.assert_array_equal(after,labels)
    assert updated[1].parent_id=='parent'
    assert report['decisions'][0]['status']=='unresolved_open_or_nonmanifold_junction'


def test_competing_supported_bodies_do_not_choose_by_size_or_distance():
    case=list(detached_parent_patch());p,labels,primary,roots,faces,domain,_=case
    labels[(labels==0)&(p[:,1]>0)]=3
    roots.append(RootPath('competitor',np.array([[0.,0.,0.],[.055,0.,0.]]),order=1))
    after,updated,report=run(case)
    np.testing.assert_array_equal(after,labels)
    assert updated[1].parent_id=='parent'
    assert report['decisions'][0]['status']=='unresolved_competing_body_contacts'


def test_overlong_child_and_descendant_survive_independent_parent_correction():
    case=list(detached_parent_patch());p,labels,primary,roots,faces,*_=case
    roots[0].points[-1,0]=.16
    roots.append(RootPath('descendant',np.array([[.2,0.,0.],[.2,-.1,0.]]),
                          order=3,parent_id='child'))
    assert roots[1].length>roots[0].length
    _,updated,report=run(case)
    assert report['changed_parent_count']==1
    assert len(updated)==3 and updated[2].parent_id=='child' and updated[2].order==2
    np.testing.assert_array_equal(updated[2].points,roots[2].points)


def test_scoped_tube_repair_still_compares_other_roots(monkeypatch):
    from soyrootbio import junction_tubes

    points, labels, primary, roots, faces, domain, _ = detached_parent_patch()
    labels[domain] = 0
    roots[1].parent_id = 'primary'
    roots[1].order = 1
    claim = np.flatnonzero(domain)
    calls = []

    def tied_witnesses(points, labels, parent, child, *args, **kwargs):
        calls.append(child)
        return claim, np.ones(len(claim)), {'status': 'supported'}

    monkeypatch.setattr(junction_tubes, 'child_tube_witnesses', tied_witnesses)
    after, report = junction_tubes.reconcile_parent_owned_tubes(
        points, labels, primary, roots, d_bar=.004, triangles=faces,
        review_root_ids={'child'},
    )
    assert sorted(calls) == [1, 2]
    np.testing.assert_array_equal(after, labels)
    assert report['ambiguous_vertex_count'] == len(claim)
