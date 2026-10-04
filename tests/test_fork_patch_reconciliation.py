import numpy as np

from soyrootbio.primary_surface import reconcile_primary_surface_tracks
from soyrootbio.topology import _comparable_fork_evidence, _fork_flank_directions
from soyrootbio.types import RootPath


def test_unrelated_fork_score_is_not_evidence_against_an_unscored_arm():
    parent = RootPath("p", np.zeros((2, 3)), score_components={"fork_hypothesis_independent_evidence_score": .95})
    child = RootPath("c", np.zeros((2, 3)))
    assert _comparable_fork_evidence(parent, child) == (0., 0.)
    parent.fork_hypothesis_group = "previous-junction"
    child.fork_hypothesis_group = "current-junction"
    assert _comparable_fork_evidence(parent, child) == (0., 0.)
    child.fork_hypothesis_group = parent.fork_hypothesis_group
    assert _comparable_fork_evidence(parent, child) == (.95, 0.)
    parent.score_components['fork_step_index'] = 4
    child.score_components['fork_step_index'] = 9
    assert _comparable_fork_evidence(parent, child) == (0., 0.)


def test_junction_connector_does_not_override_sustained_arm_direction():
    parent = np.vstack((np.column_stack((np.arange(41) * .25, np.zeros(41), np.zeros(41))),
                        np.column_stack((np.full(8, 10.), np.arange(1, 9) * .25, np.zeros(8)))))
    child = np.array([[10., 0., 0.], [10., .5, 0.], [10.5, .5, 0.], [11., 0., 0.],
                      [12., 0., 0.], [14., 0., 0.], [16., 0., 0.]])
    short, long, separation = _fork_flank_directions(parent, child, 40, .1)
    assert short > 80 and long < 20
    assert 70 < separation < 110


def _wall_fixture(stations=61):
    n = 48
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.array([[np.cos(a), np.sin(a), .2 * z]
                       for z in range(stations) for a in angles])
    faces = []
    for z in range(stations - 1):
        for a in range(n):
            b = (a + 1) % n
            faces.extend(((z*n+a, z*n+b, (z+1)*n+a),
                          (z*n+b, (z+1)*n+b, (z+1)*n+a)))
    # Caps make the source wall closed and manifold.
    points = np.vstack((points, [0, 0, 0], [0, 0, .2*(stations-1)]))
    for a in range(n):
        b = (a+1) % n
        faces.extend(((stations*n, b, a), (stations*n+1, (stations-1)*n+a, (stations-1)*n+b)))
    labels = np.zeros(len(points), int)
    for z in range(15, 46):
        labels[z*n:z*n+5] = 1
    primary = np.column_stack((np.zeros(stations), np.zeros(stations), np.arange(stations)*.2))
    root = RootPath("wall-track", np.array([[.98, .18, z] for z in np.linspace(3, 9, 31)]),
                    order=2, parent_id="parent", mean_radius=.2)
    return points, labels, primary, [root], np.asarray(faces)


def test_flat_primary_wall_is_reassigned_by_native_support():
    points, labels, primary, roots, faces = _wall_fixture()
    result, retained, report = reconcile_primary_surface_tracks(
        points, labels, primary, roots, d_bar=.15, triangles=faces,
    )
    assert np.count_nonzero(result[labels == 1] == 0) > .9 * np.count_nonzero(labels == 1)
    assert np.array_equal(result[labels == 0], labels[labels == 0])
    assert report["changed_vertex_count"] > 100


def test_point_cloud_does_not_prove_primary_wall_ownership():
    points, labels, primary, roots, _ = _wall_fixture()
    result, retained, report = reconcile_primary_surface_tracks(
        points, labels, primary, roots, d_bar=.15, triangles=None,
    )
    np.testing.assert_array_equal(result, labels)
    assert retained == roots and report["changed_vertex_count"] == 0


def test_excluded_wall_vertices_remain_unassigned():
    points, labels, primary, roots, faces = _wall_fixture()
    excluded = points[:, 2] >= 8
    labels[excluded] = -1
    result, _, _ = reconcile_primary_surface_tracks(
        points, labels, primary, roots, d_bar=.15, triangles=faces, excluded_mask=excluded,
    )
    assert np.all(result[excluded] == -1)


def test_disconnected_coincident_wall_cannot_be_reclaimed_by_proximity():
    points, labels, primary, roots, faces = _wall_fixture()
    # Duplicate a patch without sharing native vertex indices with primary.
    patch = np.flatnonzero(labels == 1)
    patch_faces = faces[np.isin(faces, patch).all(axis=1)]
    copied = np.unique(patch_faces)
    count = len(points)
    extra_faces = np.searchsorted(copied, patch_faces) + count
    points = np.vstack((points, points[copied]))
    labels[:] = 0
    labels = np.r_[labels, np.ones(len(copied), int)]
    faces = np.vstack((faces, extra_faces))
    result, _, report = reconcile_primary_surface_tracks(points, labels, primary, roots,
                                                         d_bar=.15, triangles=faces)
    np.testing.assert_array_equal(result[count:], labels[count:])


def test_primary_patch_transfer_preserves_the_separate_exposed_child_body():
    points, labels, primary, roots, faces = _wall_fixture()
    body, _, _, _, body_faces = _wall_fixture()
    # An independently supported tube, with the same child label as the
    # erroneous wall patch. It must survive even when most patch vertices go.
    tube = np.column_stack((4 + .4 * body[:, 2], .25 * body[:, 0], 6 + .25 * body[:, 1]))
    count = len(points)
    patch = labels == 1
    points = np.vstack((points, tube))
    labels = np.r_[labels, np.ones(len(tube), int)]
    faces = np.vstack((faces, body_faces + count))
    roots[0].points = np.array([[x, 0., 6.] for x in np.linspace(4, 8.8, 31)])
    result, retained, report = reconcile_primary_surface_tracks(points, labels, primary, roots,
                                                               d_bar=.15, triangles=faces)
    assert np.count_nonzero(result[:count][patch] == 0) > .9 * patch.sum()
    assert np.all(result[count:] == 1)
    assert retained == roots


def test_long_exposed_body_does_not_dilute_separate_primary_wall_evidence():
    points, labels, primary, roots, faces = _wall_fixture()
    body, _, _, _, body_faces = _wall_fixture(stations=241)
    tube = np.column_stack((4 + .4 * body[:, 2], .25 * body[:, 0], 6 + .25 * body[:, 1]))
    count = len(points)
    patch = labels == 1
    points = np.vstack((points, tube))
    labels = np.r_[labels, np.ones(len(tube), int)]
    faces = np.vstack((faces, body_faces + count))
    roots[0].points = np.array([[x, 0., 6.] for x in np.linspace(4, 23.2, 101)])
    result, retained, report = reconcile_primary_surface_tracks(points, labels, primary, roots,
                                                               d_bar=.15, triangles=faces)
    assert np.count_nonzero(result[:count][patch] == 0) > .9 * patch.sum()
    assert np.all(result[count:] == 1)
    assert retained == roots
