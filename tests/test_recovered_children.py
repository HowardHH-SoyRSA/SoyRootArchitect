import numpy as np
import pytest

from soyrootbio.mesh_geometry import MeshGeometryContext
from soyrootbio.centerline import _native_junction_footprint
from soyrootbio.recovered_children import (recover_children_on_extensions, fill_recovered_surfaces,
                                         confirmed_extension_parents, _free_tip_continuation)
from soyrootbio.types import RootPath


def branch_fixture():
    # Closed native tube with a parent-owned basal ring and an unassigned
    # exposed branch, transverse to its parent's retained centerline.
    x = np.linspace(0, .8, 81)
    angles = np.arange(16) * 2*np.pi/16
    points = np.array([[a, -.5 + .03*np.cos(t), -.5 + .03*np.sin(t)]
                       for a in x for t in angles])
    faces = []
    for i in range(len(x)-1):
        for j in range(16):
            a, b = i*16+j, i*16+(j+1)%16
            faces.extend([[a,b,a+16],[b,b+16,a+16]])
    points = np.vstack([points, [0,-.5,-.5], [.8,-.5,-.5]])
    for j in range(16):
        faces.extend([[len(points)-2,(j+1)%16,j],
                      [len(points)-1,80*16+j,80*16+(j+1)%16]])
    labels = np.full(len(points), -1)
    labels[:12*16] = 1
    labels[-2] = 1
    primary = np.array([[0,0,0],[0,0,-.5],[0,0,-1.]])
    parent = RootPath('parent', np.column_stack([np.zeros(101),np.linspace(0,-1,101),
                        np.full(101,-.5)]), insertion_index=1, insertion_point=primary[1].copy())
    return points, labels, np.asarray(faces), primary, parent


def run(points, labels, faces, primary, parent, **kwargs):
    return recover_children_on_extensions(points, labels, primary, [parent],
        parent_ids=kwargs.pop('parent_ids', {'parent'}), d_bar=.01, mesh_context=MeshGeometryContext.build(points,faces),
        excluded_mask=kwargs.pop('excluded_mask',np.zeros(len(points),bool)),
        analysis_to_mesh=kwargs.pop('analysis_to_mesh',None),
        primary_top_reference=primary[0], gravity=np.array([0,0,-1.]), **kwargs)


def test_discovers_connected_unassigned_branch_and_preserves_snapshot():
    p,l,f,primary,parent=branch_fixture()
    original=l.copy(); prior=parent.points.copy()
    roots,updated,report=run(p,l,f,primary,parent,analysis_to_mesh=np.arange(0,len(p),2))
    assert report['accepted_count']==1
    assert len(roots)==2
    child=roots[1]
    assert child.parent_id=='parent' and child.order==2
    assert child.length>.5
    assert child.covered_indices
    assert max(child.covered_indices)<len(p[::2])
    np.testing.assert_array_equal(l,original)
    np.testing.assert_array_equal(updated[original!=-1],original[original!=-1])
    assert np.count_nonzero(updated==2)>500
    np.testing.assert_array_equal(parent.points,prior)
    np.testing.assert_array_equal(child.insertion_point,parent.points[child.insertion_index])


@pytest.mark.parametrize('marker', ['fork_long_arm_reconciled', 'terminal_continuation_joined',
                                  'contacted_sibling_continuation_joined', 'displaced_tip_continuation_joined'])
def test_surveys_committed_topology_extension_without_late_recovery(marker):
    p,l,f,primary,parent=branch_fixture()
    parent.qc_flags.append(marker)
    roots,updated,report=run(p,l,f,primary,parent,parent_ids=set())
    assert report['accepted_count']==1
    assert report['survey_parent_reasons']=={'parent':[marker]}
    assert roots[1].parent_id=='parent'
    assert parent.score_components['native_recovered_extension']==1.
    np.testing.assert_array_equal(updated[l!=-1],l[l!=-1])


def test_proposed_fork_and_long_child_alone_do_not_trigger_recovery():
    p,l,f,primary,parent=branch_fixture()
    parent.score_components.update(fork_hypothesis_count=2., child_parent_length_ratio=3.)
    parent.qc_flags.extend(['overlong_child', 'terminal_continuation_unresolved'])
    assert confirmed_extension_parents([parent],set())=={}
    roots,updated,report=run(p,l,f,primary,parent,parent_ids=set())
    assert len(roots)==1 and report['accepted_count']==0
    np.testing.assert_array_equal(updated,l)


def terminal_fixture():
    p,l,f,primary,parent=branch_fixture()
    l[:]=-1; l[p[:,0]<=.4]=1
    primary[:,1]=-.5
    parent.points=np.column_stack([np.linspace(0,.4,41),np.full(41,-.5),np.full(41,-.5)])
    parent.insertion_point=primary[1].copy()
    parent.covered_indices=set(np.flatnonzero(l==1).tolist())
    parent.score_components['fork_long_arm_reconciled']=1.
    child=RootPath('proposal',np.column_stack([np.linspace(.4,.8,41),np.full(41,-.5),np.full(41,-.5)]),
                   parent_id=parent.root_id,order=2,insertion_index=40,
                   insertion_point=parent.points[-1].copy(),
                   covered_indices=set(np.flatnonzero(l==-1).tolist()))
    return p,l,f,primary,parent,child


@pytest.mark.parametrize('barrier', [None, 'excluded', 'uncertain', 'foreign', 'competing'])
def test_uncapped_terminal_recovery_requires_closed_native_uncontested_tube(barrier):
    p,l,f,primary,parent,child=terminal_fixture()
    excluded=np.zeros(len(p),bool);paths=[parent,child]
    joint=(p[:,0]>=.37)&(p[:,0]<=.4)
    if barrier=='excluded': excluded[joint]=True
    elif barrier=='uncertain': l[joint]=-2
    elif barrier=='foreign': l[joint]=3
    elif barrier=='competing':
        paths.append(RootPath('competing',np.array([[.4,-.5,-.5],[.4,-.1,-.5]]),
            parent_id=parent.root_id,order=2,insertion_point=parent.points[-1].copy()))
    before=l.copy();prior=parent.points.copy()
    evidence=_free_tip_continuation(p,l,parent,child,paths,label=1,
        mesh_context=MeshGeometryContext.build(p,f),excluded=excluded,spacing=.01)
    assert (evidence['status']=='accepted') is (barrier is None)
    np.testing.assert_array_equal(l,before)
    np.testing.assert_array_equal(parent.points,prior)


def test_terminal_tube_extends_parent_instead_of_creating_tip_child():
    p,l,f,primary,parent,child=terminal_fixture()
    prior=parent.points.copy()
    roots,updated,report=run(p,l,f,primary,parent,parent_ids=set(),analysis_to_mesh=np.arange(0,len(p),2))
    assert len(roots)==1
    assert parent.length>.7
    np.testing.assert_array_equal(parent.points[:len(prior)],prior)
    assert any(row['status']=='accepted' for row in report['terminal_continuations'])
    assert np.count_nonzero(updated==1)>1100
    np.testing.assert_array_equal(updated[l!=-1],l[l!=-1])


@pytest.mark.parametrize('barrier', ['excluded','primary_contact','open_mesh','no_mesh','limit','order'])
def test_withholds_branches_without_safe_unique_native_parent(barrier):
    p,l,f,primary,parent=branch_fixture();kwargs={}
    if barrier=='excluded':
        kwargs['excluded_mask']=l==-1
    elif barrier=='primary_contact':
        l[-1]=0
    elif barrier=='open_mesh':
        f=f[:-32]
    elif barrier=='no_mesh':
        f=None
    elif barrier=='limit':
        kwargs['max_paths']=1
    elif barrier=='order':
        kwargs['max_root_order']=1
    roots,updated,report=run(p,l,f,primary,parent,**kwargs)
    assert len(roots)==1
    assert report['accepted_count']==0


@pytest.mark.parametrize('open_wall', [False, True])
def test_fills_only_closed_native_shaft_surface(open_wall):
    p,l,f,primary,parent=branch_fixture()
    l[:]=1
    ring=np.arange(len(p)-2)//16
    angle=np.arange(len(p)-2)%16
    hole=np.flatnonzero((ring>=25)&(ring<=55)&(angle<6))
    l[hole]=-1
    parent.points=np.column_stack([np.linspace(0,.8,81),np.full(81,-.48),np.full(81,-.5)])
    if open_wall:
        f=f[~np.any(f%16==12,axis=1)]
    result,report=fill_recovered_surfaces(p,l,[parent],{'parent'},
        mesh_context=MeshGeometryContext.build(p,f),d_bar=.01,excluded_mask=np.zeros(len(p),bool))
    assert np.count_nonzero(result[hole]==1)==(0 if open_wall else len(hole))
    np.testing.assert_array_equal(result[l>=0],l[l>=0])


@pytest.mark.parametrize('middle_owner,connected', [(-1,True),(-2,False),(3,False)])
def test_native_junction_evidence_respects_uncertain_and_foreign_barriers(middle_owner,connected):
    p,l,f,primary,parent=branch_fixture()
    l[:]=middle_owner
    l[p[:,0]<=.3]=1;l[p[:,0]>=.45]=2
    before=l.copy()
    context=MeshGeometryContext.build(p,f)
    footprint=_native_junction_footprint(p,l,context.bounded_edges(.01),1,2,
        np.array([.35,-.5,-.5]),.1,.01,np.array([[0,0,0]]),np.array([0,0,-1.]))
    assert bool(len(footprint))==connected
    np.testing.assert_array_equal(l,before)
