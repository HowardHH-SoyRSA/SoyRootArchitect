"""Parity and invalidation gates for geometry/ownership/selection reuse."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from soyrootbio.mesh_geometry import MeshGeometryContext
from soyrootbio.pipeline import _ExposedSegmentIndexCache, _nearest_exposed_segments
from soyrootbio.lateral import _is_truncated_path_duplicate, select_non_overlapping_paths
from soyrootbio.types import RootPath


def test_geometry_context_owns_readonly_snapshots_and_checks_mutated_inputs():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    faces = np.array([[0, 1, 2]])
    context = MeshGeometryContext.build(points, faces)
    for array in (context.points, context.triangles, context.edges,
                  context.edge_lengths, context.face_areas, context.face_centroids,
                  context.point_tree.data, context.centroid_tree.data,
                  context.vertex_face_incidence.data, context.bounded_edges(1.),
                  context.support_edges(1.)):
        with pytest.raises(ValueError):
            array.flat[0] = 5
    fingerprint = context.coordinate_sha256
    points[0] = 2
    with pytest.raises(ValueError, match="same ordered"):
        context.validate(points, faces)
    np.testing.assert_array_equal(context.points[0], [0., 0., 0.])
    assert context.coordinate_sha256 == fingerprint
    faces[0, 1] = 0
    with pytest.raises(ValueError, match="same ordered"):
        context.validate(context.points, faces)


def test_components_are_scoped_to_ownership_and_active_edge_policies():
    points = np.column_stack([np.arange(6.), np.zeros((6, 2))])
    context = MeshGeometryContext.build(points, None)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    labels = np.array([1, 1, 1, 1, 2, 2])
    first = context.ownership(labels)
    component = first.components(edges)
    assert component[0] == component[3]
    assert context.ownership(labels.copy()) is first
    active = np.ones(6, bool)
    active[1] = False
    assert first.components(edges, active)[0] != first.components(edges, active)[2]
    assert first.components(edges[:1])[0] != first.components(edges[:1])[2]
    labels[1] = 2
    second = context.ownership(labels)
    assert second.number == first.number + 1
    assert second.components(edges)[0] != second.components(edges)[2]
    assert first.components(edges) is component
    np.testing.assert_array_equal(first.vertices(1), [0, 1, 2, 3])
    np.testing.assert_array_equal(second.vertices(1), [0, 2, 3])
    labels[1] = 1
    third = context.ownership(labels)
    assert third.number == second.number + 1
    assert third is not first
    assert third.components(edges) is not component


def test_compact_weighted_graphs_preserve_native_edges_and_isolated_support():
    random = np.random.default_rng(50)
    points = random.normal(size=(60, 3))
    context = MeshGeometryContext.build(points, None)
    edges = np.array([[i, i + 1] for i in range(0, 58, 2)])
    edges.setflags(write=False)
    labels = random.integers(-2, 5, size=60)
    generation = context.ownership(labels)
    for label in range(5):
        vertices = np.flatnonzero(labels == label)
        same = edges[(labels[edges[:, 0]] == label) & (labels[edges[:, 1]] == label)]
        local = np.searchsorted(vertices, same)
        distance = np.maximum(np.linalg.norm(points[same[:, 0]] - points[same[:, 1]], axis=1), 1e-15)
        reference = coo_matrix((np.r_[distance, distance],
                               (np.r_[local[:, 0], local[:, 1]], np.r_[local[:, 1], local[:, 0]])),
                              shape=(len(vertices), len(vertices))).tocsr()
        actual = generation.local_graph(label, edges)
        np.testing.assert_array_equal(actual.toarray(), reference.toarray())
        np.testing.assert_array_equal(connected_components(actual)[1], connected_components(reference)[1])


def test_readonly_edge_view_does_not_hide_changes_through_writable_base():
    points = np.column_stack([np.arange(4.), np.zeros((4, 2))])
    context = MeshGeometryContext.build(points, None)
    generation = context.ownership(np.ones(4, dtype=int))
    base = np.array([[0, 1], [1, 2]])
    edges = base.view()
    edges.setflags(write=False)
    np.testing.assert_array_equal(generation.edges(1, edges), base)
    assert generation.components(edges)[0] == generation.components(edges)[2]
    base[0] = [2, 3]
    np.testing.assert_array_equal(generation.edges(1, edges), base)
    assert generation.components(edges)[0] != generation.components(edges)[2]


def test_exact_query_cache_keeps_second_root_and_numeric_label_ties():
    line = np.array([[0., 0., 0.], [0., 0., .1], [0., 0., 1.]])
    paths = [(9, line), (3, line.copy()), (1, line + [.1, 0, 0])]
    points = np.array([[0., 0., .4], [.05, 0., .4], [2., 0., .4]])
    cache = _ExposedSegmentIndexCache(max_entries=1, max_query_bytes=1000)

    def query(radius=.2):
        return _nearest_exposed_segments(points, paths, d_bar=.1, radius=radius,
                                         margin=.001, segment_index_cache=cache)

    first = query()
    assert first[1].tolist() == [3, 1, -1]
    assert first[3].tolist() == [9, 3, -1]
    again = query()
    for actual, expected in zip(again, first):
        np.testing.assert_array_equal(actual, expected)
    again[1][:] = 100
    assert query()[1].tolist() == [3, 1, -1]
    for mutate in (lambda: points.__setitem__((0, 0), .1),
                   lambda: line.__setitem__((1, 0), .2)):
        mutate()
        actual = query()
        expected = _nearest_exposed_segments(points, paths, d_bar=.1, radius=.2, margin=.001)
        for cached, plain in zip(actual, expected):
            np.testing.assert_array_equal(cached, plain)
    wide = query(radius=3.)
    wide_reference = _nearest_exposed_segments(points, paths, d_bar=.1, radius=3., margin=.001)
    for actual, expected in zip(wide, wide_reference):
        np.testing.assert_array_equal(actual, expected)
    assert wide[1][-1] >= 0 and wide[3][-1] >= 0
    assert cache._query_bytes <= cache.max_query_bytes


def test_unchanged_body_subdivision_survives_a_changed_sibling(monkeypatch):
    import soyrootbio.pipeline as module
    paths = [(1, np.array([[0., 0., 0.], [0., 0., 1.]])),
             (2, np.array([[.1, 0., 0.], [.1, 0., 1.]]))]
    real = module._subdivide_exposed_path
    calls = []

    def counted(line, spacing):
        calls.append(line.copy())
        return real(line, spacing)

    monkeypatch.setattr(module, "_subdivide_exposed_path", counted)
    cache = module._ExposedSegmentIndexCache()
    cache.get_or_build(paths, .01)
    assert len(calls) == 2
    paths[1][1][1, 0] = .2
    cache.get_or_build(paths, .01)
    assert len(calls) == 3
    cache.get_or_build([(7, paths[0][1]), (8, paths[1][1])], .01)
    assert len(calls) == 3  # Pieces are geometry; numeric tie labels are in the tree.
    cache.get_or_build(paths, .01, body_start_indices=[1, 0])
    assert len(calls) == 4
    assert cache._subdivision_bytes <= cache.max_subdivision_bytes


def _selection_oracle(candidates, spacing, penalty, initial, limit):
    # Original ordered greedy policy, deliberately recomputing dynamic evidence.
    selected, used = [], set(initial)
    pool = sorted(candidates, key=lambda path: (path.score, path.length), reverse=True)
    while pool:
        best, best_value = None, 0.
        for path in pool:
            if any(_is_truncated_path_duplicate(path, prior, d_bar=spacing) for prior in selected):
                continue
            covered = path.covered_indices if path.novel_support_indices is None else path.novel_support_indices
            if path.novel_support_indices is not None and not covered:
                continue
            growth = float(path.score_components.get("trace_growth_arc", path.length))
            value = len(covered - used) - penalty * len(covered & used) + 10. * growth + .35 * growth / max(spacing, 1e-12)
            if value > best_value:
                best, best_value = path, value
        if best is None:
            break
        selected.append(best)
        used.update(best.covered_indices if best.novel_support_indices is None else best.novel_support_indices)
        pool = [path for path in pool if path is not best]
        if limit is not None and len(selected) >= limit:
            break
    return [path.root_id for path in selected]


@pytest.mark.parametrize("seed", range(8))
def test_incremental_selection_matches_ordered_oracle(seed):
    random = np.random.default_rng(seed)
    candidates = []
    for i in range(35):
        angle = (i % 5) * np.pi / 6
        length = float(random.choice([.02, .08, .3, .3]))
        line = np.linspace(0, length, 7)[:, None] * np.array([[np.cos(angle), np.sin(angle), 0.]])
        covered = set(random.choice(160, size=35, replace=False).tolist())
        candidates.append(RootPath(str(i), line, score=float(i % 3), covered_indices=covered,
                                   novel_support_indices=(set() if i % 9 == 0 else covered if i % 4 else None)))
    candidates.append(candidates[5])  # Duplicate object identities retain the old removal policy.
    initial = set(range(25))
    for limit in (None, 1, 8):
        expected = _selection_oracle(deepcopy(candidates), .01, 1.25, initial, limit)
        actual = select_non_overlapping_paths(deepcopy(candidates), np.zeros((160, 3)), .01,
                                              overlap_penalty=1.25, initial_used=initial,
                                              max_paths=limit, rename_selected=False)
        assert [path.root_id for path in actual] == expected


def test_duplicate_evidence_is_evaluated_once_per_directed_pair(monkeypatch):
    import soyrootbio.lateral as module
    candidates = [RootPath(str(i), np.array([[i, 0., 0.], [i, 0., 1.]]),
                           covered_indices={i}) for i in range(12)]
    real = module._is_truncated_path_duplicate
    visited = set()

    def counted(path, retained, **kwargs):
        key = (id(path), id(retained))
        assert key not in visited
        visited.add(key)
        return real(path, retained, **kwargs)

    monkeypatch.setattr(module, "_is_truncated_path_duplicate", counted)
    selected = module.select_non_overlapping_paths(candidates, np.zeros((12, 3)), .01,
                                                  rename_selected=False)
    assert [path.root_id for path in selected] == [str(i) for i in range(12)]
