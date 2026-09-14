import numpy as np
import pytest
from scipy.spatial import cKDTree

from soyrootbio.competition import RootSegmentIndex
from soyrootbio.pipeline import (
    _assign_full_root_labels, _assign_lateral_points,
    _resolve_parent_owned_junctions, _absorb_small_primary_surface_patches,
)
from soyrootbio.types import RootPath


def root(name, points, **kwargs):
    return RootPath(root_id=name, points=np.asarray(points, dtype=float), **kwargs)


def test_two_nearest_nodes_from_same_root_hide_actual_competitor():
    a = root("dense", [[-.001, 0, 0], [.001, 0, 0], [.1, 0, 0]])
    b = root("sparse", [[-.1, .002, 0], [.1, .002, 0]])
    point = np.array([[0., .001, 0.]])
    _, old = cKDTree(np.vstack([a.points, b.points])).query(point, k=2)
    assert np.all(old < len(a.points))
    labels, pairs = _assign_lateral_points(point, [a, b], np.array([False]), .01, return_competing_labels=True)
    assert labels.tolist() == [-1]
    assert pairs == {0: (1, 2)}


@pytest.mark.parametrize("other", [
    [[0, -1, 0], [0, 1, 0]],  # crossing
    [[-1, .02, 0], [1, .02, 0]],  # adjacent
    [[-1, .019, 0], [1, .021, 0]],  # nearly parallel
])
def test_distinct_roots_with_sparse_crossing_adjacent_and_parallel_segments(other):
    index = RootSegmentIndex([np.array([[-1., 0, 0], [1., 0, 0]]), np.array(other)])
    result = index.query(np.array([[0., .01, .002]]), max_distance=.03)
    assert set(result.labels[0]) == {0, 1}
    assert result.ambiguous(.011)[0]


def test_same_root_segments_never_compete():
    index = RootSegmentIndex([np.array([[-1., 0, 0], [0., 0, 0], [1., 0, 0]])])
    result = index.query(np.array([[0., .01, 0.]]), max_distance=.1)
    assert result.labels.tolist() == [[0, -1]]
    assert not result.ambiguous(.1)[0]


def test_local_interpolated_radius_changes_winner():
    a = np.array([[0., 0, 0], [1., 0, 0]])
    b = a + [0, .12, 0]
    p = np.array([[.5, .1, 0.]])
    raw = RootSegmentIndex([a, b]).query(p, max_distance=.2)
    surface = RootSegmentIndex([a, b], radii=[[.08, .12], [.001, .001]]).query(p, max_distance=.2)
    assert raw.labels[0, 0] == 1
    assert surface.labels[0, 0] == 0
    assert surface.radii[0, 0] == pytest.approx(.1)
    assert surface.distances[0, 0] == pytest.approx(0.)


def test_radius_is_taken_at_closest_projection_not_another_segment_surface():
    # A farther fat segment must not replace the root's closest projection.
    a = np.array([[0., 0, 0], [1., 0, 0], [1., 1, 0], [0., 1, 0]])
    result = RootSegmentIndex([a], radii=[[0., 0., .9, .9]]).query(
        np.array([[.25, .1, 0.]]), max_distance=.2)
    assert result.centerline_distances[0, 0] == pytest.approx(.1)
    assert result.radii[0, 0] == 0


def test_endpoints_singletons_repeated_nodes_empty_paths_and_missing_neighbors():
    paths = [np.empty((0, 3)), np.array([[0., 0, 0], [0., 0, 0], [1., 0, 0]]), np.array([[2., 0, 0]])]
    result = RootSegmentIndex(paths).query(np.array([[-.1, 0, 0], [1.5, 0, 0], [99, 0, 0]]), max_distance=.6)
    assert result.labels.tolist() == [[1, -1], [1, 2], [-1, -1]]
    np.testing.assert_allclose(result.centerline_distances[:2, 0], [.1, .5])
    assert np.all(np.isinf(result.distances[2]))
    assert RootSegmentIndex([]).query(np.empty((0, 3)), max_distance=1).labels.shape == (0, 2)


def test_ties_use_root_identity_and_are_independent_of_index_order_and_chunking():
    a = np.array([[-1., 0, 0], [1., 0, 0]])
    b = a + [0., 2., 0.]
    points = np.tile([0., 1. + 1e-14, 0.], (7, 1))
    first = RootSegmentIndex([a, b], labels=[8, 3]).query(points, max_distance=2, chunk_size=1)
    second = RootSegmentIndex([b, a], labels=[3, 8]).query(points, max_distance=2, chunk_size=4)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.labels[0].tolist() == [3, 8]


def test_third_root_is_found_beyond_arbitrarily_many_same_root_nodes():
    a = np.column_stack([np.linspace(-.005, .005, 500), np.zeros((500, 2))])
    b = np.array([[-10., .003, 0], [10., .003, 0]])
    c = b + [0., 10., 0.]
    result = RootSegmentIndex([a, b, c]).query(np.array([[0., .001, 0.]]), max_distance=.01)
    assert result.labels.tolist() == [[0, 1]]


def test_index_agrees_with_exhaustive_per_root_segment_oracle():
    rng = np.random.default_rng(91)
    paths = [rng.normal(size=(n, 3)) for n in [1, 2, 12, 50]]
    profiles = [rng.uniform(0, .2, len(p)) for p in paths]
    points = rng.normal(size=(130, 3))
    result = RootSegmentIndex(paths, radii=profiles).query(points, max_distance=.4, chunk_size=13)
    expected = np.full_like(result.labels, -1)
    for i, point in enumerate(points):
        candidates = []
        for label, (path, radii) in enumerate(zip(paths, profiles)):
            starts, ends = (path[:-1], path[1:]) if len(path) > 1 else (path, path)
            delta = ends - starts
            t = np.clip(np.sum((point - starts) * delta, axis=1) / np.maximum(np.sum(delta**2, axis=1), 1e-30), 0, 1)
            d = np.linalg.norm(point - (starts + t[:, None] * delta), axis=1)
            k = int(np.argmin(d))
            radius = radii[k] + t[k] * (radii[min(k + 1, len(radii) - 1)] - radii[k])
            residual = abs(d[k] - radius)
            if residual <= .4:
                candidates.append((residual, label))
        for slot, (_, label) in enumerate(sorted(candidates)[:2]):
            expected[i, slot] = label
    np.testing.assert_array_equal(result.labels, expected)


def test_parent_child_evidence_survives_sampled_primary_mask():
    parent = np.array([[0., 0, -.2], [0., 0, .2]])
    child = root("child", [[0, 0, 0], [.1, 0, 0]], insertion_point=np.zeros(3), insertion_index=0)
    points = np.array([[0., .005, 0], [.09, .005, 0]])
    labels, pairs = _assign_lateral_points(points, [child], np.array([True, False]), .01,
        primary_path=parent, return_root_labels=True, return_competing_labels=True)
    assert labels[0] == -2 and pairs[0] == (0, 1)
    resolved, report = _resolve_parent_owned_junctions(points, labels, parent, [child], d_bar=.01,
        assignment_radius=.04, ambiguity_margin=.01, competing_labels=pairs)
    assert resolved.tolist() == [0, 1]
    assert report["resolved_vertex_count"] == 1


def test_full_resolution_preserves_exclusion_and_resolves_surface_competition():
    primary = np.array([[-1., 0, 0], [1., 0, 0]])
    child = root("other", [[-1., .02, 0], [1., .02, 0]])
    labels, pairs = _assign_full_root_labels(np.array([[0., .01, 0], [.2, .01, 0]]), primary,
        [child], d_bar=.01, excluded_mask=np.array([False, True]), return_competing_labels=True)
    assert labels.tolist() == [-2, -1] and pairs == {0: (0, 1)}


def test_patch_cleanup_preserves_unresolved_distinct_root_evidence():
    primary = np.array([[0., 0, -.2], [0., 0, .2]])
    points = np.array([[.1, 0, 0], [.1, .01, -.01], [.1, -.01, -.01], [.1, 0, .01]])
    labels = np.array([-2, 0, 0, 0])
    triangles = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1]])
    resolved, report = _absorb_small_primary_surface_patches(points, labels, primary, [], d_bar=.01,
        triangles=triangles, competing_labels={0: (1, 2)})
    np.testing.assert_array_equal(resolved, labels)
    assert report["competition_protected_vertex_count"] == 1


def test_invalid_duplicate_root_identities_are_rejected():
    a = root("same", [[0., 0, 0], [1., 0, 0]])
    with pytest.raises(ValueError, match="unique"):
        _assign_lateral_points(np.array([[0., 0, 0]]), [a, a], np.array([False]), .01)


def test_collinear_resampling_does_not_change_root_distances_or_radii():
    sparse = np.array([[-1., 0, 0], [1., 0, 0]])
    x = np.linspace(-1., 1., 251)
    dense = np.column_stack([x, np.zeros((len(x), 2))])
    other = sparse + [0., .05, 0.]
    query = np.array([[.157, .02, 0.], [-.697, .04, 0.], [1.01, .01, 0.]])
    a = RootSegmentIndex([sparse, other], radii=[[.01, .03], .01]).query(query, max_distance=.1)
    b = RootSegmentIndex([dense, other], radii=[np.linspace(.01, .03, 251), .01]).query(query, max_distance=.1)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_allclose(a.distances, b.distances, atol=1e-14)


def test_parallel_parent_child_competition_outside_basal_prefix_stays_uncertain():
    parent = np.array([[0., 0, 0], [1., 0, 0]])
    child = root("child", [[0., 0, 0], [0., .01, 0], [1., .01, 0]], insertion_point=np.zeros(3))
    query = np.array([[.07, .005, 0.]])
    labels, pairs = _assign_full_root_labels(query, parent, [child], d_bar=.01, return_competing_labels=True)
    resolved, _ = _resolve_parent_owned_junctions(query, labels, parent, [child], d_bar=.01,
        assignment_radius=.05, ambiguity_margin=.01, competing_labels=pairs)
    assert resolved.tolist() == [-2]


def test_junction_rechecks_surface_metric_with_supplied_radius_profiles():
    parent = np.array([[0., 0, -.2], [0., 0, .2]])
    child = root("child", [[0., 0, 0], [.1, 0, 0]], insertion_point=np.zeros(3))
    query = np.array([[.02, .005, 0.]])
    radii = {0: np.array([.02, .02]), 1: np.array([.005, .005])}
    labels, pairs = _assign_full_root_labels(query, parent, [child], d_bar=.01,
        root_radii=radii, return_competing_labels=True)
    assert labels[0] == -2
    resolved, _ = _resolve_parent_owned_junctions(query, labels, parent, [child], d_bar=.01,
        assignment_radius=.05, ambiguity_margin=.01, competing_labels=pairs, root_radii=radii)
    assert resolved[0] == 0
