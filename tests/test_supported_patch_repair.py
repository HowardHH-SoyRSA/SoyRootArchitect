from copy import deepcopy

import numpy as np

from soyrootbio.identity_repair import freeze_identity_references
from soyrootbio.measurement_evidence import build_measurement_evidence
from soyrootbio.parent_contact import reconcile_parent_contacts
from soyrootbio.types import RootPath


def _parent_wall_and_exposed_child(*, body_attachment=True, descendant=False):
    spacing = .08
    count = 64
    angles = np.arange(count) * 2 * np.pi / count
    z = np.arange(126) * spacing
    parent = np.array([[np.cos(t), np.sin(t), h] for h in z for t in angles])
    faces = []
    for i in range(len(z) - 1):
        for j in range(count):
            a, b = i * count + j, i * count + (j + 1) % count
            faces.extend([[a, b, a + count], [b, b + count, a + count]])
    offset = len(parent)
    child_count = 24
    x = 1.12 + np.arange(37) * spacing
    child = np.array([[v, .1 * np.cos(t), 5.3 + .1 * np.sin(t)]
                      for v in x for t in np.arange(child_count) * 2 * np.pi / child_count])
    for i in range(len(x) - 1):
        for j in range(child_count):
            a, b = offset + i * child_count + j, offset + i * child_count + (j + 1) % child_count
            faces.extend([[a, b, a + child_count], [b, b + child_count, a + child_count]])
    points = np.vstack((parent, child))
    labels = np.r_[np.zeros(len(parent), int), np.ones(len(child), int)]
    patch = np.array([i * count + j for i in (60, 61, 62, 63, 64) for j in (0, 1, 63)])
    labels[patch] = 1
    if body_attachment:
        faces.append([67 * count, offset + 5, offset + 6])
    primary = np.array([[0., 0., 0.], [0., 0., 10.]])
    prior = np.array([[.7, 0., 4.7], [1.3, 0., 5.3], [4., 0., 5.3]])
    root = RootPath('child', prior.copy(), mean_radius=.1)
    reference = freeze_identity_references(primary, [root])['child']
    root.points = np.column_stack((x, np.zeros(len(x)), np.full(len(x), 5.3)))
    roots = [root]
    if descendant:
        start = len(points)
        points = np.vstack((points, points[patch[4]] + [[.05, 0, 0], [.05, .02, 0]]))
        labels = np.r_[labels, [2, 2]]
        faces.append([int(patch[4]), start, start + 1])
        roots.append(RootPath('descendant', points[start:].copy(), parent_id='child', order=2))
    faces = np.asarray(faces)
    root.centerline_assessment = {
        'section_radius_median': .1,
        'prefit_identity_evidence': reference,
        'measurement_evidence': build_measurement_evidence(
            points, labels, 1, root.points, triangles=faces,
            selected_vertex_indices=np.arange(offset, offset + len(child)), spacing=spacing,
            identity_resolved=False),
    }
    cleanup = {'components': [{'source_label': 1, 'first_vertex': int(patch.min()), 'vertex_count': len(patch),
        'candidates': [{'label': 0, 'boundary_fraction': 1., 'target_supported': True, 'compact': True,
                        'local_anchor': True, 'median_local_radius': 1., 'claim_score': .9}]}]}
    return points, labels, primary, roots, faces, patch, cleanup, spacing


def _run(case, **kwargs):
    p, labels, primary, roots, faces, _, cleanup, spacing = case
    return reconcile_parent_contacts(p, labels, primary, roots, triangles=faces, d_bar=spacing,
                                     cleanup_report=cleanup, **kwargs)


def test_proximal_parent_wall_patch_is_reassigned_with_selected_body_preserved():
    case = _parent_wall_and_exposed_child()
    before = deepcopy(case)
    p, labels, _, roots, _, patch, _, _ = case
    result, report = _run(case)
    np.testing.assert_array_equal(result[patch], 0)
    np.testing.assert_array_equal(result[np.setdiff1d(np.arange(len(p)), patch)],
                                  labels[np.setdiff1d(np.arange(len(p)), patch)])
    evidence = roots[0].centerline_assessment['measurement_evidence']
    assert np.all(result[evidence['native_vertex_indices']] == 1)
    accepted = [row for row in report['contacts'] if row.get('independent_parent_wall_transfer')]
    assert len(accepted) == 1
    assert accepted[0]['connected_to_distal_body']  # Old proximity/span test misclassified this patch.
    assert accepted[0]['independent_parent_wall_evidence']['supported']
    assert accepted[0]['independent_parent_wall_evidence']['independent_child_sections']['supported_components']
    np.testing.assert_array_equal(p, before[0])
    np.testing.assert_array_equal(case[4], before[4])
    np.testing.assert_array_equal(roots[0].points, before[3][0].points)
    assert roots[0].parent_id == before[3][0].parent_id


def test_transfer_cannot_remove_only_observed_parent_attachment():
    case = _parent_wall_and_exposed_child(body_attachment=False)
    result, report = _run(case)
    np.testing.assert_array_equal(result, case[1])
    reasons = [r['independent_parent_wall_evidence'].get('attachment_guard') for r in report['contacts']]
    assert 'unresolved_would_remove_supported_attachment' in reasons


def test_transfer_cannot_create_higher_order_primary_contact_or_orphan_descendant():
    case = _parent_wall_and_exposed_child(descendant=True)
    result, report = _run(case)
    np.testing.assert_array_equal(result, case[1])
    reasons = [r['independent_parent_wall_evidence'].get('attachment_guard') for r in report['contacts']]
    assert any(reason in {'unresolved_would_remove_supported_attachment',
                          'unresolved_would_create_higher_order_primary_contact'} for reason in reasons)


def test_stale_selected_support_cannot_authorize_wall_transfer():
    case = _parent_wall_and_exposed_child()
    case[3][0].points[0] += [.01, 0., 0.]
    result, _ = _run(case)
    np.testing.assert_array_equal(result, case[1])


def test_measured_patch_component_is_never_sacrificed_for_parent_claim():
    case = _parent_wall_and_exposed_child()
    p, labels, _, roots, faces, patch, _, spacing = case
    ids = np.union1d(roots[0].centerline_assessment['measurement_evidence']['native_vertex_indices'], patch)
    roots[0].centerline_assessment['measurement_evidence'] = build_measurement_evidence(
        p, labels, 1, roots[0].points, triangles=faces, selected_vertex_indices=ids,
        spacing=spacing, identity_resolved=False)
    result, _ = _run(case)
    np.testing.assert_array_equal(result, labels)


def test_older_discrete_patch_branch_also_preserves_current_selected_support():
    case = _parent_wall_and_exposed_child()
    p, labels, primary, roots, faces, patch, _, spacing = case
    root = roots[0]
    independent = RootPath('child', root.points.copy(), mean_radius=.1)
    root.centerline_assessment['prefit_identity_evidence'] = freeze_identity_references(primary, [independent])['child']
    ids = np.union1d(root.centerline_assessment['measurement_evidence']['native_vertex_indices'], patch)
    root.centerline_assessment['measurement_evidence'] = build_measurement_evidence(
        p, labels, 1, root.points, triangles=faces, selected_vertex_indices=ids,
        spacing=spacing, identity_resolved=False)
    result, report = _run(case)
    np.testing.assert_array_equal(result, labels)
    assert any(row['status'] == 'unresolved_selected_measurement_body' for row in report['contacts'])


def test_actual_child_shaft_normals_do_not_supply_parent_wall_evidence():
    from soyrootbio.parent_contact import _parent_wall_orientation
    theta = np.arange(24) * 2 * np.pi / 24
    p = np.array([[x, .1*np.cos(t), .1*np.sin(t)] for x in np.arange(6)*.08 for t in theta])
    faces = []
    for i in range(5):
        for j in range(24):
            a, b = 24*i+j, 24*i+(j+1)%24
            faces.extend([[a, b, a+24], [b, b+24, a+24]])
    evidence = _parent_wall_orientation(p, np.asarray(faces), np.ones(len(p), int),
        np.arange(len(p)), 0, np.array([[0., 0, 0], [1., 0, 0]]), np.array([[0., 0, -1.], [0., 0, 1.]]))
    assert not evidence['supported']
    assert evidence['child_normal_alignment'] < .01


def test_excluded_collar_wall_vertices_are_immutable():
    case = _parent_wall_and_exposed_child()
    mask = np.zeros(len(case[0]), bool)
    mask[case[5]] = True
    result, _ = _run(case, excluded_mask=mask)
    np.testing.assert_array_equal(result, case[1])


def test_older_discrete_patch_branch_cannot_remove_sole_native_attachment():
    case = _parent_wall_and_exposed_child(body_attachment=False)
    root = case[3][0]
    independent = RootPath('child', root.points.copy(), mean_radius=.1)
    root.centerline_assessment['prefit_identity_evidence'] = freeze_identity_references(case[2], [independent])['child']
    result, report = _run(case)
    np.testing.assert_array_equal(result, case[1])
    rejected = report['atomic_transfer_guard']['rejections']
    assert len(rejected) == 1
    assert rejected[0]['reasons'] == ['would_remove_supported_attachment']
    assert report['changed_vertex_count'] == 0
    assert report['unresolved_patch_count'] == 1


def test_individually_safe_wall_transfers_cannot_jointly_remove_last_attachment():
    from soyrootbio.parent_contact import _guard_parent_transfer_union, _patch_transfer_preserves_attachments
    # Child 1 has two detached wall contacts and a separate exposed body (0).
    # Removing either wall alone preserves its other observed attachment.
    # Child 2 has a separate, safe wall transfer and a surviving body contact.
    labels = np.array([1, 1, 1, 0, 0, 2, 2, 0, 0])
    edges = np.array([[1, 3], [2, 4], [5, 7], [6, 8]])
    roots = [RootPath('a', np.zeros((2, 3))), RootPath('b', np.ones((2, 3)))]
    transfers = [{'root_id': root_id, 'first_vertex': member, 'vertex_count': 1,
                  'vertex_indices': [member], 'changed_vertex_count': 1,
                  'status': 'reassigned_to_supported_parent'}
                 for root_id, member in [('a', 1), ('a', 2), ('b', 5)]]
    for row in transfers:
        label = int(labels[row['first_vertex']])
        assert _patch_transfer_preserves_attachments(labels, edges, row['vertex_indices'], label, 0, roots)[0]
    proposed = labels.copy()
    proposed[[1, 2, 5]] = 0
    result, rejected = _guard_parent_transfer_union(labels, proposed, edges, roots, transfers)
    expected = labels.copy()
    expected[5] = 0
    np.testing.assert_array_equal(result, expected)
    assert len(rejected) == 2
    assert all(row['reasons'] == ['would_remove_supported_attachment'] for row in rejected)
    # The outcome depends on frozen geometry, not proposal traversal order.
    reverse_result, _ = _guard_parent_transfer_union(labels, proposed, edges, roots, deepcopy(transfers[::-1]))
    np.testing.assert_array_equal(reverse_result, expected)


def test_atomic_guard_rechecks_contacts_created_by_a_rejected_transfer():
    from soyrootbio.parent_contact import _guard_parent_transfer_union
    # Transferring parent 1's wall to primary would expose a forbidden primary
    # edge to daughter 2, but also creates a new attachment for child 3. Once
    # that transfer is withheld, child 3's other wall must stay too.
    labels = np.array([1, 1, 0, 2, 3, 3, 0])
    edges = np.array([[0, 2], [1, 2], [0, 3], [0, 4], [5, 6]])
    roots = [RootPath('parent', np.zeros((2, 3))),
             RootPath('daughter', np.zeros((2, 3)), parent_id='parent', order=2),
             RootPath('other', np.zeros((2, 3)))]
    transfers = [{'root_id': root_id, 'first_vertex': member, 'vertex_count': 1,
                  'vertex_indices': [member], 'changed_vertex_count': 1,
                  'status': 'reassigned_to_supported_parent'}
                 for root_id, member in [('parent', 0), ('other', 5)]]
    proposed = labels.copy()
    proposed[[0, 5]] = 0
    result, rejected = _guard_parent_transfer_union(labels, proposed, edges, roots, transfers)
    np.testing.assert_array_equal(result, labels)
    assert len(rejected) == 2
    assert 'would_create_higher_order_primary_contact' in rejected[0]['reasons']
    assert rejected[1]['reasons'] == ['would_remove_supported_attachment']
