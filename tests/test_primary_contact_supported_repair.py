import numpy as np

from soyrootbio.primary_contact import restrict_higher_order_primary_contacts
from soyrootbio.types import RootPath


def _native_edges_fixture(primary_edges, contacts):
    # Each edge has its own uncertain triangle corner, so native owner
    # connectivity is exactly the stated graph, including isolated vertices.
    points = [[float(i), 0., 0.] for i in range(7)]
    labels = [0, 0, 0, 0, 1, 2, 3]
    faces = []
    for a, b in [*primary_edges, (1, 4), *contacts]:
        corner = len(points)
        points.append([(points[a][0]+points[b][0])/2, .1, 0.])
        labels.append(-2)
        faces.append([a, b, corner])
    roots = [RootPath('parent', np.array([[1., 0, 0], [4., 0, 0]]), order=1, parent_id='primary'),
             RootPath('child-a', np.array([[0., 0, 0], [5., 0, 0]]), order=2, parent_id='parent'),
             RootPath('child-b', np.array([[2., 0, 0], [6., 0, 0]]), order=2, parent_id='parent')]
    return np.asarray(points), np.asarray(labels), np.asarray(faces), roots


def test_unsafe_primary_seam_does_not_veto_independent_safe_seam():
    # 0 is a removable end; 2 is an articulation. Both children have one
    # native vertex, so the child-side candidate correctly cannot consume it.
    p, labels, faces, roots = _native_edges_fixture([(0,1),(1,2),(2,3)], [(0,5),(2,6)])
    after, report = restrict_higher_order_primary_contacts(p, labels, roots, triangles=faces, d_bar=2.)
    assert after[0] == -2
    np.testing.assert_array_equal(after[1:], labels[1:])
    rows = {r['root_id']: r for r in report['contacts']}
    assert rows['child-a']['remaining_contact_edge_count'] == 0
    assert rows['child-b']['remaining_contact_edge_count'] == 1
    assert rows['child-b']['primary_seam_status'] == 'unresolved_would_split_primary_surface'
    assert rows['child-a']['biological_identity_resolved'] is False
    assert rows['child-a']['ownership_resolution'] == 'ambiguous_native_contact_separated_to_uncertainty'


def test_individually_safe_primary_seams_cannot_jointly_split_primary():
    # Opposite vertices of a cycle are individually removable; their union
    # disconnects the retained primary and therefore cannot be applied.
    p, labels, faces, roots = _native_edges_fixture([(0,1),(1,2),(2,3),(3,0)], [(0,5),(2,6)])
    after, report = restrict_higher_order_primary_contacts(p, labels, roots, triangles=faces, d_bar=2.)
    assert sum(after[[0,2]] == -2) == 1
    assert report['unresolved_root_count'] == 1
    np.testing.assert_array_equal(after[labels > 0], labels[labels > 0])
    assert any(r['status'] == 'unresolved_conflicting_primary_seam_proposals' for r in report['contacts'])
    reordered = [roots[0], roots[2], roots[1]]
    remap = labels.copy()
    remap[labels == 2], remap[labels == 3] = 3, 2
    changed, _ = restrict_higher_order_primary_contacts(p, remap, reordered, triangles=faces, d_bar=2.)
    np.testing.assert_array_equal(changed[labels == 0], after[labels == 0])


def test_assessed_child_without_independent_prior_cannot_use_shortened_fit_as_anchor():
    from soyrootbio.primary_contact import _exposed_body_anchor_status
    root = RootPath('child', np.array([[0.,0,0],[1.,0,0]]),
                    centerline_assessment={'status':'fitted'})
    owned = np.arange(3)
    status = _exposed_body_anchor_status(np.array([[0.,0,0],[.5,0,0],[1.,0,0]]), root,
        owned, owned, np.array([[0,1],[1,2]]), .1)
    assert status == 'unresolved_missing_prefit_reference'


def _tube_with_seam_enclosed_primary_fragment():
    angles = np.arange(64)*2*np.pi/64
    z = np.arange(101)*.1
    p = np.array([[np.cos(t), np.sin(t), s] for s in z for t in angles])
    faces = []
    for k in range(100):
        for j in range(64):
            a, b = k*64+j, k*64+(j+1)%64
            c, d = (k+1)*64+j, (k+1)*64+(j+1)%64
            faces.extend([[a,b,c],[b,d,c]])
    faces = np.asarray(faces)
    edges = np.unique(np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    labels = np.zeros(len(p),int)
    for k in range(48,53):
        for j in range(30,35):
            if k in {48,52} or j in {30,34}:
                labels[k*64+j] = 2
    labels[50*64+34] = 0  # native primary bridge through the ambiguous margin
    contact = edges[(labels[edges[:,0]]==0)&(labels[edges[:,1]]==2) |
                    (labels[edges[:,0]]==2)&(labels[edges[:,1]]==0)]
    seam = np.unique(contact[labels[contact]==0])
    primary_edges = edges[np.all(labels[edges]==0,axis=1)]
    reference = {'reference_valid':True, 'prior_path':[[0.,0.,0.],[0.,0.,10.]]}
    return p,labels,edges,primary_edges,seam,reference


def test_local_seam_closure_keeps_observed_circumferential_primary_body():
    from soyrootbio.primary_contact import _close_local_primary_seam, _preserves_components
    p,labels,edges,pe,seam,reference = _tube_with_seam_enclosed_primary_fragment()
    owned = np.flatnonzero(labels==0)
    assert not _preserves_components(len(p),pe,seam,np.setdiff1d(owned,seam))
    expanded, report = _close_local_primary_seam(p,labels,edges,pe,owned,seam,
        np.zeros(len(p),bool),.1,reference,{})
    assert report['status'] == 'supported_local_contact_margin_closure'
    assert report['closure_vertex_indices'] == [50*64+32]
    assert _preserves_components(len(p),pe,expanded,np.setdiff1d(owned,expanded))
    assert all(s['angular_coverage_after_deg'] >= 270 for f in report['fragments'] for s in f['retained_wall_sections'])
    assert set(expanded) <= set(owned)


def test_local_seam_closure_cannot_consume_another_root_attachment_or_endpoint():
    from soyrootbio.primary_contact import _close_local_primary_seam
    p,labels,edges,pe,seam,reference = _tube_with_seam_enclosed_primary_fragment()
    # One native neighbor is an independently retained O1 root. Its attachment
    # is a barrier even though the primary fragment is only a single vertex.
    anchor = 50*64+31
    labels[anchor] = 1
    seam = seam[seam != anchor]
    pe = edges[np.all(labels[edges]==0,axis=1)]
    expanded, report = _close_local_primary_seam(p,labels,edges,pe,np.flatnonzero(labels==0),seam,
        np.zeros(len(p),bool),.1,reference,{})
    np.testing.assert_array_equal(expanded,seam)
    assert not report['closure_vertex_indices']
    p,labels,edges,pe,seam,reference = _tube_with_seam_enclosed_primary_fragment()
    reference['prior_path'] = [[0.,0.,4.99],[0.,0.,10.]]
    expanded, report = _close_local_primary_seam(p,labels,edges,pe,np.flatnonzero(labels==0),seam,
        np.zeros(len(p),bool),.1,reference,{})
    np.testing.assert_array_equal(expanded,seam)
    assert not report['closure_vertex_indices']


def test_local_seam_closure_is_unit_and_translation_invariant():
    from soyrootbio.primary_contact import _close_local_primary_seam
    p,labels,edges,pe,seam,reference = _tube_with_seam_enclosed_primary_fragment()
    owned = np.flatnonzero(labels==0)
    original, _ = _close_local_primary_seam(p,labels,edges,pe,owned,seam,
        np.zeros(len(p),bool),.1,reference,{})
    scale, offset = .005, np.array([.3,-.7,1.2])
    converted = {'reference_valid':True, 'prior_path':(np.array(reference['prior_path'])*scale+offset).tolist()}
    actual, report = _close_local_primary_seam(p*scale+offset,labels,edges,pe,owned,seam,
        np.zeros(len(p),bool),.1*scale,converted,{})
    np.testing.assert_array_equal(actual,original)
    assert report['closure_vertex_indices'] == [50*64+32]
