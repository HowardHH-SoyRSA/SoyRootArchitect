import numpy as np

import soyrootbio.pipeline as pipeline_module
from soyrootbio.pipeline import (
    _ExposedSegmentIndexCache, _assign_full_root_labels,
    _assign_lateral_points, _nearest_exposed_segments,
)
from soyrootbio.surface_patches import _polyline_projection_distance_and_arc
from soyrootbio.types import RootPath


def test_full_assignment_compares_distinct_roots_even_when_two_nearest_nodes_match():
    points = np.array([[.06, 0., .01], [.10, 0., .01]])
    primary = np.array([[0., 0., 0.], [0., 0., 1.]])
    child = RootPath("child", np.array([[.10, 0., 0.], [.10, 0., .02]]))
    labels, competing = _assign_full_root_labels(
        points, primary, [child], d_bar=.1, return_competing_labels=True)
    assert labels.tolist() == [-2, 1]
    assert competing == {0: (1, 0)}


def test_analysis_assignment_compares_distinct_laterals():
    points = np.array([[.06, 0., .01], [.10, 0., .01]])
    first = RootPath("first", np.array([[.10, 0., 0.], [.10, 0., .02]]))
    other = RootPath("other", np.array([[0., 0., 0.], [0., 0., 1.]]))
    labels, competing = _assign_lateral_points(
        points, [first, other], np.zeros(len(points), bool), .1,
        return_competing_labels=True)
    assert labels.tolist() == [-1, 1]
    assert competing == {0: (1, 2)}


def test_competitor_outside_margin_does_not_make_uncertainty():
    points = np.array([[.10, 0., .01]])
    primary = np.array([[0., 0., 0.], [0., 0., 1.]])
    child = RootPath("child", np.array([[.10, 0., 0.], [.10, 0., .02]]))
    labels, competing = _assign_full_root_labels(
        points, primary, [child], d_bar=.1, return_competing_labels=True)
    assert labels.tolist() == [1]
    assert competing == {}


def test_internal_child_connector_does_not_claim_exposed_surface():
    primary = np.array([[0., 0., 0.], [0., 0., .5], [0., 0., 1.]])
    child = RootPath("child", np.array([[0., 0., .5], [.1, 0., .5],
                                         [.1, 0., .7]]), body_start_index=1)
    sample = np.array([[0., 0., .5], [.1, 0., .7]])
    full = _assign_full_root_labels(sample, primary, [child], d_bar=.001)
    assert full.tolist() == [0, 1]
    lateral = _assign_lateral_points(
        sample, [RootPath("parent", primary), child], np.zeros(2, bool), .001)
    assert lateral.tolist() == [1, 2]


def test_sparse_centerline_segments_compete_at_their_interiors():
    primary = np.array([[0., 0., 0.], [0., 0., 1.]])
    child = RootPath("child", np.array([[.014, 0., 0.], [.014, 0., 1.]]))
    samples = np.array([[.014, 0., .5], [.007, 0., .5]])
    labels, competing = _assign_full_root_labels(
        samples, primary, [child], d_bar=.001,
        return_competing_labels=True,
    )
    assert labels.tolist() == [1, -2]
    assert competing == {1: (0, 1)}


def test_spatial_segment_search_matches_all_segment_projection_in_assignment_range():
    random = np.random.default_rng(17)
    paths = [(label, random.uniform(-.5, .5, size=(4, 3)))
             for label in range(3)]
    points = random.uniform(-.5, .5, size=(80, 3))
    radius, margin = .22, .04
    nearest, owner, second_distance, second = _nearest_exposed_segments(
        points, paths, d_bar=.01, radius=radius, margin=margin)
    exact = np.column_stack([
        _polyline_projection_distance_and_arc(points, path)[0]
        for _, path in paths
    ])
    ordered = np.argsort(exact, axis=1, kind="stable")
    for i in range(len(points)):
        first, other = ordered[i, :2]
        if exact[i, first] > radius:
            continue
        assert owner[i] == first
        assert np.isclose(nearest[i], exact[i, first], atol=1e-12)
        if exact[i, other] <= exact[i, first] + margin:
            assert second[i] == other
            assert np.isclose(second_distance[i], exact[i, other], atol=1e-12)


def test_segment_index_cache_reuses_geometry_across_query_policies(monkeypatch):
    paths = [
        (1, np.array([[0., 0., 0.], [0., 0., 1.]])),
        (2, np.array([[.2, 0., 0.], [.2, 0., 1.]])),
    ]
    points = np.array([[.01, 0., .5], [.19, 0., .5], [.1, 0., .5]])
    policies = [(.03, .005), (.3, .05)]
    expected = [
        _nearest_exposed_segments(points, paths, d_bar=.01, radius=radius, margin=margin)
        for radius, margin in policies
    ]
    real_build = pipeline_module._build_exposed_segment_index
    builds = []

    def counted_build(*args, **kwargs):
        builds.append(1)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "_build_exposed_segment_index", counted_build)
    cache = _ExposedSegmentIndexCache(max_entries=5)
    for (radius, margin), reference in zip(policies, expected, strict=True):
        observed = _nearest_exposed_segments(
            points, paths, d_bar=.01, radius=radius, margin=margin,
            segment_index_cache=cache,
        )
        assert all(np.array_equal(actual, prior) for actual, prior in zip(observed, reference))
    assert len(builds) == 1


def test_segment_index_cache_invalidates_changed_path_body_and_labels(monkeypatch):
    primary = np.array([[0., 0., 0.], [0., 0., 1.]])
    child = RootPath("child", np.array([[.1, 0., 0.], [.1, 0., 1.]]))
    points = np.array([[.1, 0., .5], [0., 0., .5]])
    real_build = pipeline_module._build_exposed_segment_index
    builds = []

    def counted_build(*args, **kwargs):
        builds.append(1)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "_build_exposed_segment_index", counted_build)
    cache = _ExposedSegmentIndexCache(max_entries=5)

    def assign_full(spacing=.01):
        return _assign_full_root_labels(
            points, primary, [child], d_bar=spacing,
            segment_index_cache=cache,
        )

    assert assign_full().tolist() == [1, 0]
    assert assign_full().tolist() == [1, 0]
    assert len(builds) == 1
    excluded = np.array([True, False])
    assert _assign_full_root_labels(
        points, primary, [child], d_bar=.01,
        excluded_mask=excluded, segment_index_cache=cache,
    ).tolist() == [-1, 0]
    assert len(builds) == 1  # The exclusion mask is applied after the query.
    child.body_start_index = 1
    assign_full()
    assert len(builds) == 2
    child.points[1, 0] = .15
    assign_full()
    assert len(builds) == 3
    assign_full(.02)
    assert len(builds) == 4
    _assign_lateral_points(
        points, [child], np.zeros(len(points), bool), .02,
        segment_index_cache=cache,
    )
    assert len(builds) == 5  # No primary segment in the analysis policy.
    child.points[1, 0] = .1
    child.body_start_index = 0
    assert assign_full().tolist() == [1, 0]
    assert len(builds) == 5  # The original exact index is still in the LRU.


def test_segment_index_cache_distinguishes_label_sequence(monkeypatch):
    first = np.array([[0., 0., 0.], [0., 0., 1.]])
    other = np.array([[.2, 0., 0.], [.2, 0., 1.]])
    sample = np.array([[0., 0., .5]])
    real_build = pipeline_module._build_exposed_segment_index
    builds = []

    def counted_build(*args, **kwargs):
        builds.append(1)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "_build_exposed_segment_index", counted_build)
    cache = _ExposedSegmentIndexCache()
    left = _nearest_exposed_segments(
        sample, [(1, first), (2, other)], d_bar=.01,
        radius=.3, margin=.01, segment_index_cache=cache,
    )
    right = _nearest_exposed_segments(
        sample, [(2, first), (1, other)], d_bar=.01,
        radius=.3, margin=.01, segment_index_cache=cache,
    )
    assert left[1].tolist() == [1]
    assert right[1].tolist() == [2]
    assert len(builds) == 2
