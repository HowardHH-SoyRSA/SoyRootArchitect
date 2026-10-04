from copy import deepcopy

import numpy as np

from soyrootbio.junction_tubes import reconcile_parent_owned_tubes
from soyrootbio.types import RootPath
from test_primary_o1_ownership import junction


def family(order=2):
    p, labels, parent, roots, faces, n = junction()
    # Move the independent primary far away; the same observed junction is
    # now an O1 parent with an O2 child. No coordinate or new edge is added.
    primary = parent + [2., 0., 0.]
    roots[0].root_id = "child"
    roots[0].parent_id, roots[0].order = "parent", order
    roots.insert(0, RootPath("parent", parent, order=order-1))
    return p, labels + 1, primary, roots, faces, n


def run(case, **kw):
    p, labels, primary, roots, faces, _ = case
    return reconcile_parent_owned_tubes(p, labels, primary, roots, d_bar=.004, triangles=faces, **kw)


def test_parent_protrusion_is_corrected_at_higher_orders_without_editing_paths():
    for order in (2, 3):
        case = family(order)
        original = deepcopy(case[3])
        after, report = run(case)
        p, before, _, roots, _, n = case
        target = (np.arange(len(p)) >= n) & (p[:, 0] >= .055) & (p[:, 0] < .10)
        assert np.all(after[target] == 2)
        np.testing.assert_array_equal(after[:n], before[:n])
        np.testing.assert_array_equal(after[before == 2], before[before == 2])
        assert report["transferred_vertex_count"] > 0
        for a, b in zip(roots, original):
            assert (a.root_id, a.parent_id, a.order) == (b.root_id, b.parent_id, b.order)
            np.testing.assert_array_equal(a.points, b.points)


def test_family_iteration_and_numeric_labels_do_not_change_winners():
    case = list(family())
    expected, _ = run(case)
    case[3] = list(reversed(case[3]))
    case[1] = np.where(case[1] == 1, 2, 1)
    actual, _ = run(case)
    np.testing.assert_array_equal(np.where(actual == 1, 2, 1), expected)


def test_excluded_collar_and_negative_assignment_are_barriers_at_every_order():
    case = family()
    p, labels, _, _, _, n = case
    excluded = (np.arange(len(p)) >= n) & (p[:, 0] >= .08) & (p[:, 0] < .10)
    labels[excluded] = -1
    after, report = run(case, excluded_mask=excluded)
    np.testing.assert_array_equal(after, labels)
    assert report["transferred_vertex_count"] == 0


def test_no_mesh_cannot_infer_parent_child_surface_connectivity():
    case = list(family())
    case[4] = None
    after, report = run(case)
    np.testing.assert_array_equal(after, case[1])
    assert report["status"] == "insufficient_native_mesh"


def test_internal_contact_patch_does_not_hide_exposed_branch_direction():
    case = list(junction())
    p, labels, _, _, _, n = case
    patch = (np.arange(len(p)) < n) & (p[:, 0] < -.035) & (np.abs(p[:, 2]) < .006)
    labels[patch] = 1
    after, report = run(case)
    target = (np.arange(len(p)) >= n) & (p[:, 0] >= .055) & (p[:, 0] < .10)
    assert np.all(after[target] == 1)
    np.testing.assert_array_equal(after[:n], labels[:n])


def test_higher_order_transfer_cannot_create_native_primary_contact():
    case = family()
    p, before, _, _, _, n = case
    marker = np.flatnonzero((np.arange(len(p)) >= n) & (p[:, 0] > .055) & (p[:, 0] < .065))[0]
    before[marker] = 0
    after, report = run(case)
    child = next(r for r in report['junctions'] if r['root_id'] == 'child')
    assert child['status'] == 'unresolved_would_contact_primary'
    assert child['transferred_vertex_count'] == 0
    np.testing.assert_array_equal(after, before)


def test_closed_child_surface_inside_parent_envelope_is_not_reclaimed():
    case = junction()
    p, before, _, _, _, n = case
    after, report = run(case)
    internal = (np.arange(len(p)) >= n) & (p[:, 0] <= .035)
    np.testing.assert_array_equal(after[internal], before[internal])
    np.testing.assert_array_equal(after[before == 1], before[before == 1])
    assert report['transferred_vertex_count'] > 0


def test_biased_automatic_child_path_does_not_define_surface_ownership():
    case = junction()
    expected, _ = run(case)
    case[3][0].points = np.array([[0., 0., -.08], [.35, .02, .04]])
    actual, _ = run(case)
    np.testing.assert_array_equal(actual, expected)


def test_flat_parent_envelope_does_not_hide_an_exposed_child_tube():
    case = junction()
    points, before, _, _, _, n = case
    points[:n, 0] *= .4
    points[:n, 1] *= 1.6
    points[n:, 1:] *= .6
    after, report = run(case)
    target = (np.arange(len(points)) >= n) & (points[:, 0] >= .035) & (points[:, 0] < .10)
    internal = (np.arange(len(points)) >= n) & (points[:, 0] <= .01)
    assert np.all(after[target] == 1)
    np.testing.assert_array_equal(after[internal], before[internal])
    np.testing.assert_array_equal(after[:n], before[:n])
    assert report['junctions'][0]['measured_parent_sections'] > 0


def test_native_scan_gap_cannot_be_bridged_by_a_tube_query():
    case = list(junction())
    points, before, _, _, faces, _ = case
    crossing = ((points[faces, 0].min(axis=1) < .10) &
                (points[faces, 0].max(axis=1) >= .10))
    case[4] = faces[~crossing]
    after, report = run(case)
    np.testing.assert_array_equal(after, before)
    assert report['transferred_vertex_count'] == 0


def test_absent_child_body_is_reported_without_inventing_a_root():
    case = junction()
    case[1][:] = 0
    after, report = run(case)
    np.testing.assert_array_equal(after, case[1])
    assert report['junctions'][0]['status'] == 'insufficient_support'
    assert report['transferred_vertex_count'] == 0


def test_parent_cut_rollback_cannot_expose_primary_to_an_accepted_grandchild(monkeypatch):
    # Isolate the ownership-guard transaction from geometric detection. A
    # native-edge branch at vertex 1 splits primary if removed. Its rollback
    # exposes primary beside the simultaneous grandchild claim at vertex 3.
    from types import SimpleNamespace
    import soyrootbio.junction_tubes as module
    from soyrootbio.mesh_geometry import OwnershipGeometryGeneration

    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0],
                       [1, 1, 0], [1, 2, 0], [2, 1, 0]], float)
    edges = np.array([[0, 1], [1, 2], [1, 3], [3, 4], [1, 5], [3, 5]])
    before = np.array([0, 0, 0, 1, 2, 1])
    context = SimpleNamespace(edges=edges, edge_incidence=np.full(len(edges), 2),
        validate=lambda *a: None, support_edges=lambda spacing: edges,
        ownership=lambda labels: OwnershipGeometryGeneration(labels, points))
    roots = [RootPath('parent', points[[3, 5]], parent_id='primary', order=1),
             RootPath('child', points[[3, 4]], parent_id='parent', order=2)]

    def witness(points, labels, parent, child, *args, **kwargs):
        return (np.array([1 if child == 1 else 3]), np.array([0.]),
                dict(status='supported', supported_vertices=1))

    monkeypatch.setattr(module, 'child_tube_witnesses', witness)
    after, report = module.reconcile_parent_owned_tubes(points, before, points[[0, 2]],
        roots, d_bar=.1, triangles=np.array([[0, 1, 3]]), mesh_context=context)
    np.testing.assert_array_equal(after, before)
    status = {row['root_id']: row['status'] for row in report['junctions']}
    assert status['parent'] == 'unresolved_would_split_parent'
    assert status['child'] == 'unresolved_would_contact_primary'

