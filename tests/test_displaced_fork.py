from copy import deepcopy

import numpy as np
import pytest

from soyrootbio.displaced_fork import reroute_displaced_forks
from soyrootbio.mesh_geometry import MeshGeometryContext
from soyrootbio.types import RootPath


def fixture():
    angle = np.arange(24)*2*np.pi/24
    points = np.array([[x,3*np.cos(a),3*np.sin(a)] for x in range(181) for a in angle],float)
    faces = []
    for x in range(180):
        for a in range(24):
            b=(a+1)%24
            faces.extend(((24*x+a,24*x+b,24*(x+1)+a),(24*x+b,24*(x+1)+b,24*(x+1)+a)))
    incoming=np.column_stack((np.arange(81),np.zeros(81),np.zeros(81)))
    short=np.array([[80+.1*i,0,.4*i] for i in range(1,21)])
    parent=RootPath('parent',np.vstack((incoming,short)),order=1,parent_id='primary',
        covered_indices=set(range(100*24)),score_components={'trace_local_radius':3.})
    child=RootPath('child',np.array([[80,30,0]]+[[x,0,0] for x in range(84,175)],float),
        parent_id='wrong',order=2,insertion_index=80,insertion_point=np.array([80.,30.,0.]),
        covered_indices=set(range(80*24,181*24)),score_components={'trace_local_radius':3.})
    former=RootPath('wrong',incoming+np.array([0,30,0]),order=1,parent_id='primary')
    return parent,child,former,points,np.asarray(faces)


def run(paths,points,faces,**kwargs):
    return reroute_displaced_forks(paths,spacing=1.,support_points=points,mesh_points=points,
        mesh_triangles=faces,excluded=kwargs.get('excluded'),
        mesh_context=MeshGeometryContext.build(points,faces),attachment_status=kwargs.get('status',{}))


def test_supported_junction_replaces_only_unsupported_attachment():
    p,c,f,points,faces=fixture()
    child_body=c.points[1:].copy();parent_path=p.points.copy();support=c.covered_indices.copy()
    decisions=run([p,c,f],points,faces)
    assert len(decisions)==1
    assert c.parent_id==p.root_id
    np.testing.assert_array_equal(c.points[1:],child_body)
    np.testing.assert_array_equal(p.points,parent_path)
    assert c.covered_indices==support
    assert not decisions[0]['former_connector_supported']
    assert decisions[0]['native_mesh_connected']


@pytest.mark.parametrize('condition',['accepted','excluded','competing','anchor_descendant'])
def test_uncertain_alternative_preserves_original_attachment(condition):
    p,c,f,points,faces=fixture();paths=[p,c,f];kwargs={}
    if condition=='accepted':kwargs['status']={'child':'accepted'}
    if condition=='excluded':kwargs['excluded']=(points[:,0]>=78)&(points[:,0]<=90)
    if condition=='competing':
        other=deepcopy(p);other.root_id='competing';paths.append(other)
    if condition=='anchor_descendant':
        paths.append(RootPath('descendant',np.array([[80.,30.,0.],[80.,32.,0.]]),
                              order=3,parent_id='child',insertion_index=0))
    old=c.points.copy()
    assert run(paths,points,faces,**kwargs)==[]
    assert c.parent_id=='wrong'
    np.testing.assert_array_equal(c.points,old)
