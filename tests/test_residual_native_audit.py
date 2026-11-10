from copy import deepcopy

import numpy as np

from soyrootbio.identity_repair import freeze_identity_references
from soyrootbio.surface_patch_audit import audit_discrete_child_patches
from soyrootbio.types import RootPath


def native_tube(*, wall_patch=False, interrupted=False, long_native_edge=False):
    spacing = .04
    radius = .3  # More than three times the old radius-free 2*d_bar threshold.
    theta = np.arange(48) * 2 * np.pi / 48
    stations = np.arange(81) * .05
    tube = np.asarray([[x, radius*np.cos(t), radius*np.sin(t)] for x in stations for t in theta])
    points = tube.copy()
    faces = []
    for i in range(len(stations)-1):
        for j in range(len(theta)):
            a, b = 48*i+j, 48*i+(j+1)%48
            faces += [[a,b,a+48],[b,b+48,a+48]]
    labels = np.ones(len(points), int)
    labels[:48] = 0
    if interrupted:
        labels[(points[:,0] > 1.9) & (points[:,0] < 2.1)] = 0
    if wall_patch:
        # Existing parent surface contains one detached child-owned wall island.
        offset = len(points)
        wall = np.array([[.1,1.,0.], [.16,1.,0.], [.1,1.06,0.], [.16,1.06,0.]])
        points = np.vstack([points,wall])
        labels = np.r_[labels,[0,1,0,0]]
        faces.extend([[offset,offset+1,offset+2],[offset+1,offset+3,offset+2]])
        if long_native_edge:
            # This native triangle connects the island to the tube through an
            # anomalously long edge; it is not a valid repair corridor.
            faces.append([offset+1, 48*10, 48*10+1])
    path = np.array([[0.,0.,0.],[4.,0.,0.]])
    root = RootPath('child',path,mean_radius=None)
    reference = freeze_identity_references(path,[root])['child']
    root.centerline_assessment={'prefit_identity_evidence':reference,'fit_applied':True}
    # Prove body queries remain anchored to the independent long prior.
    root.points = np.array([[0.,0.,0.],[.05,0.,0.]])
    return points,np.asarray(faces),labels,root,spacing


def test_thick_native_single_body_without_radius_is_not_an_isolated_patch():
    points,faces,labels,root,spacing = native_tube()
    frozen=(points.copy(),faces.copy(),labels.copy(),deepcopy(root.centerline_assessment))
    report=audit_discrete_child_patches(points,labels,[root],triangles=faces,d_bar=spacing)
    assert report['status']=='clear'
    assert report['unresolved_patch_count']==0
    assert report['unresolved_body_anchor_root_count']==0
    evidence=report['body_anchor_evidence'][0]
    assert evidence['status']=='supported_native_transverse_body'
    assert evidence['sections'][0]['median_radius'] > 3*spacing
    assert evidence['radius_source']=='native_sections_against_preserved_prefit_path'
    for before,after in zip(frozen[:3],(points,faces,labels)):
        np.testing.assert_array_equal(before,after)
    assert root.centerline_assessment==frozen[3]


def test_true_detached_parent_wall_patch_remains_flagged_while_body_is_protected():
    points,faces,labels,root,spacing=native_tube(wall_patch=True)
    report=audit_discrete_child_patches(points,labels,[root],triangles=faces,d_bar=spacing)
    assert report['unresolved_patch_count']==1
    patch=report['patches'][0]
    assert patch['vertex_indices']==[len(points)-3]
    assert patch['disconnection_basis']=='distinct_raw_native_owned_components'
    assert patch['native_owned_component_count']==2
    assert patch['body_anchor_component_identified']


def test_two_supported_native_bodies_stay_explicitly_unresolved():
    points,faces,labels,root,spacing=native_tube(interrupted=True)
    report=audit_discrete_child_patches(points,labels,[root],triangles=faces,d_bar=spacing)
    assert report['status']=='unresolved_patches'
    assert any(p['component_has_independent_body_support'] for p in report['patches'])
    assert all(p['status']=='unresolved_disconnected_exposed_body' for p in report['patches'])


def test_single_flat_unverified_component_is_identity_review_not_discrete_patch():
    points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
    labels=np.array([0,1,1,1]);faces=np.array([[0,1,2],[1,3,2]])
    root=RootPath('child',np.array([[0.,0.,4.],[1.,0.,4.]]),mean_radius=None)
    report=audit_discrete_child_patches(points,labels,[root],triangles=faces,d_bar=.5)
    assert report['status']=='unresolved_body_identity'
    assert report['unresolved_patch_count']==0
    assert report['unresolved_body_anchor_root_count']==1
    assert report['body_anchor_ambiguities'][0]['status']=='unresolved_single_component_body_identity'


def test_long_raw_native_connection_is_not_misreported_as_disconnection():
    points,faces,labels,root,spacing=native_tube(wall_patch=True,long_native_edge=True)
    report=audit_discrete_child_patches(points,labels,[root],triangles=faces,d_bar=spacing)
    assert report['unresolved_patch_count']==0
    assert report['status']=='unresolved_body_identity'
    assert report['body_anchor_ambiguities'][0]['status']=='unresolved_long_native_edge_body_connection'
    assert report['body_anchor_ambiguities'][0]['bounded_owned_component_count']==2
    assert report['body_anchor_ambiguities'][0]['native_owned_component_count']==1


def test_bundle_validator_counts_identity_roots_separately_from_fragments():
    import importlib.util
    from pathlib import Path

    script=Path(__file__).resolve().parents[1]/'scripts/validate_identity_measurements.py'
    spec=importlib.util.spec_from_file_location('identity_measurement_validator',script)
    validator=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    points,faces,labels,root,spacing=native_tube(wall_patch=True,long_native_edge=True)
    offset=len(points)
    points=np.vstack([points,points[-4:]+np.array([.3,0.,0.])])
    labels=np.r_[labels,[0,1,0,0]]
    faces=np.vstack([faces,[[offset,offset+1,offset+2],
                           [offset+1,offset+3,offset+2],
                           [offset+1,48*20,48*20+1]]])
    audit=audit_discrete_child_patches(points,labels,[root],triangles=faces,d_bar=spacing)
    assert len(audit['body_anchor_ambiguities'])==2
    assert audit['unresolved_body_anchor_root_count']==1
    metadata={'coordinate_unit':'mm','d_bar_mm':spacing,'final_surface_patch_audit':audit,
              'final_compliance_audit':{'discrete_child_patch_status':audit['status'],
                'unresolved_native_body_identity_count':1,'native_body_identity_status':'unresolved'}}
    roots=[{'root_id':root.root_id,'polyline':root.points.tolist(),'root_order':root.order,
            'parent_id':root.parent_id,'body_start_index':0,
            'centerline_assessment':root.centerline_assessment}]
    checked=validator._audit_patches(roots,{0:{'root_id':'primary'},1:{'root_id':root.root_id}},
                                    points,faces,labels,np.zeros(len(points),bool),metadata)
    assert checked['unresolved_body_anchor_root_count']==1
