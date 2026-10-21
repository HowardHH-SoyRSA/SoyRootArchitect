"""Regression: inconclusive child evidence cannot authorize primary expansion."""
import numpy as np

from soyrootbio import primary_surface
from test_fork_patch_reconciliation import _wall_fixture
from test_primary_o1_ownership import junction


def test_wall_evidence_does_not_propagate_into_unmeasured_child(monkeypatch):
    points, labels, primary, roots, faces = _wall_fixture()
    child = np.flatnonzero(labels == 1)
    wall = np.zeros(len(points), bool)
    wall[child[points[child, 2] < 4.]] = True
    monkeypatch.setattr(primary_surface, 'primary_wall_evidence',
                        lambda *a, **k: (wall, np.ones(len(points)), {}))
    monkeypatch.setattr(primary_surface, '_tubular_body_witnesses',
                        lambda owned, *a: np.zeros(len(owned), bool))
    geometry = points.copy(), faces.copy(), roots[0].points.copy()
    after, retained, report = primary_surface.reconcile_primary_surface_tracks(
        points, labels, primary, roots, d_bar=.15, triangles=faces)
    assert np.any(after[child[wall[child]]] == 0)
    assert np.all(after[child[~wall[child]]] == 1)
    assert retained == roots
    assert np.all(wall[report['confirmed_primary_vertex_indices']])
    np.testing.assert_array_equal(points, geometry[0])
    np.testing.assert_array_equal(faces, geometry[1])
    np.testing.assert_array_equal(roots[0].points, geometry[2])


def test_uncertified_short_body_is_not_split_by_wall_cut(monkeypatch):
    points, labels, primary, roots, faces = _wall_fixture()
    wall = (labels == 1) & (points[:, 2] > 5.) & (points[:, 2] < 7.)
    monkeypatch.setattr(primary_surface, 'primary_wall_evidence',
                        lambda *a, **k: (wall, np.ones(len(points)), {}))
    monkeypatch.setattr(primary_surface, '_tubular_body_witnesses',
                        lambda owned, *a: np.zeros(len(owned), bool))
    after, retained, report = primary_surface.reconcile_primary_surface_tracks(
        points, labels, primary, roots, d_bar=.15, triangles=faces)
    np.testing.assert_array_equal(after, labels)
    assert retained == roots
    assert report['roots'][0]['status'] == 'unresolved_would_split_exposed_body'
    assert report['unresolved_root_count'] == 1


def test_longitudinal_flanks_reject_emerging_branch_from_transverse_hull():
    p, labels, axis, _, faces, parent_count = junction()
    # Deliberately mark the entire branch as primary: its label cannot certify
    # the convex-hull protrusion as biological primary wall.
    wall, _, report = primary_surface.primary_wall_evidence(
        p, labels*0, axis, d_bar=.004, triangles=faces, longitudinal_confirmation=True)
    exposed = (np.arange(len(p)) >= parent_count) & (p[:, 0] > .055)
    assert not wall[exposed].any()
    assert wall[:parent_count].sum() > parent_count//2
    assert report['local_wall_vertex_count'] > report['wall_vertex_count']
    # The evidence must follow source geometry, independent of coordinate
    # orientation or a change in physical scale.
    rotation = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    rotated, _, _ = primary_surface.primary_wall_evidence(
        3*p@rotation+2, labels*0, 3*axis@rotation+2,
        d_bar=.012, triangles=faces, longitudinal_confirmation=True)
    np.testing.assert_array_equal(rotated, wall)
