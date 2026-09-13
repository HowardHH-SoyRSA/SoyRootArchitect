import numpy as np
import pytest

from soyrootbio.pipeline import _resolve_primary_o1_ownership
from soyrootbio.types import RootPath


def junction():
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    z = np.linspace(-0.4, 0.4, 161)
    parent = np.array([[0.04 * np.cos(t), 0.04 * np.sin(t), s] for s in z for t in theta])
    x = np.linspace(0, 0.35, 71)
    body = np.array([[s, 0.01 * np.cos(t), 0.01 * np.sin(t)] for s in x for t in theta])
    faces = []
    for offset, count in [(0, len(z)), (len(parent), len(x))]:
        for i in range(count - 1):
            for j in range(24):
                a, b = offset + i * 24 + j, offset + i * 24 + (j + 1) % 24
                faces.extend([[a, b, a + 24], [b, b + 24, a + 24]])
    points = np.vstack([parent, body])
    labels = np.r_[np.zeros(len(parent), dtype=int), np.where(body[:, 0] < 0.10, 0, 1)]
    primary = np.array([[0., 0., -0.4], [0., 0., 0.4]])
    child = RootPath(root_id="child", parent_id="primary", order=1,
                     points=np.array([[0., 0., 0.], [0.35, 0., 0.]]), insertion_index=0)
    return points, labels, primary, [child], np.asarray(faces), len(parent)


def run(case, **kwargs):
    p, labels, primary, children, triangles, _ = case
    return _resolve_primary_o1_ownership(p, labels, primary, children,
                                         d_bar=0.004, triangles=triangles, **kwargs)


def test_trims_exposed_primary_protrusion_preserving_internal_connector_and_far_regions():
    case = junction()
    p, labels, primary, children, _, count = case
    original_path = children[0].points.copy()
    result, report = run(case)
    target = (np.arange(len(p)) >= count) & (p[:, 0] >= 0.055) & (p[:, 0] < 0.10)
    assert np.all(result[target] == 1)
    assert report["transferred_vertex_count"] >= target.sum()
    np.testing.assert_array_equal(result[:count], labels[:count])
    internal = (np.arange(len(p)) >= count) & (p[:, 0] <= 0.035)
    np.testing.assert_array_equal(result[internal], labels[internal])
    np.testing.assert_array_equal(result[labels != 0], labels[labels != 0])
    np.testing.assert_array_equal(children[0].points, original_path)


def test_excluded_uncertain_stays_unassigned_and_cannot_bridge():
    case = junction()
    p, labels, _, _, _, count = case
    excluded = (np.arange(len(p)) >= count) & (p[:, 0] >= 0.08) & (p[:, 0] < 0.10)
    labels[excluded] = -2
    result, report = run(case, excluded_mask=excluded)
    assert np.all(result[excluded] == -1)
    assert report["transferred_vertex_count"] == 0
    assert report["excluded_uncertain_to_unassigned_count"] == excluded.sum()


def test_mesh_disconnection_blocks_spatially_near_child_support():
    case = list(junction())
    p, _, _, _, faces, _ = case
    crossed = (p[faces, 0].min(axis=1) < 0.10) & (p[faces, 0].max(axis=1) >= 0.10)
    case[4] = faces[~crossed]
    result, report = run(case)
    np.testing.assert_array_equal(result, case[1])
    assert report["transferred_vertex_count"] == 0


def test_exact_segments_invariant_to_collinear_skeleton_resampling():
    case = list(junction())
    before, _ = run(case)
    case[2] = np.column_stack([np.zeros(91), np.zeros(91), np.linspace(-0.4, 0.4, 91)])
    case[3][0].points = np.column_stack([np.linspace(0, 0.35, 81), np.zeros(81), np.zeros(81)])
    after, _ = run(case)
    np.testing.assert_array_equal(after, before)


def test_no_child_surface_or_wrong_hierarchy_cannot_claim_primary():
    case = junction()
    case[1][case[1] == 1] = 0
    result, report = run(case)
    np.testing.assert_array_equal(result, case[1])
    assert report["junctions"][0]["status"] == "insufficient_child_support"
    case = junction()
    case[3][0].order = 2
    result, _ = run(case)
    np.testing.assert_array_equal(result, case[1])


def test_invalid_sampling_density_rejected():
    p, labels, primary, children, faces, _ = junction()
    with pytest.raises(ValueError, match="d_bar"):
        _resolve_primary_o1_ownership(p, labels, primary, children, d_bar=0, triangles=faces)


def test_equal_sibling_claims_preserve_primary_and_do_not_depend_on_order():
    case = junction()
    p, labels, primary, children, faces, _ = case
    original = labels.copy()
    # Two equally supported hypotheses must not win by list order.
    labels[(labels == 1) & (p[:, 1] < 0)] = 2
    children.append(RootPath(root_id="sibling", parent_id="primary", order=1,
                            points=children[0].points.copy(), insertion_index=0))
    result, report = run(case)
    assert report["multi_claim_vertex_count"] > 0
    assert report["ambiguous_claim_vertex_count"] > 0
    np.testing.assert_array_equal(result[original == 0], original[original == 0])
    swapped = labels.copy()
    swapped[labels == 1], swapped[labels == 2] = 2, 1
    second, _ = _resolve_primary_o1_ownership(p, swapped, primary, children[::-1],
                                              d_bar=0.004, triangles=faces)
    np.testing.assert_array_equal(second[original == 0], result[original == 0])


def test_point_cloud_fallback_and_uniform_scale():
    case = junction()
    p, labels, primary, children, _, _ = case
    result, report = _resolve_primary_o1_ownership(p, labels, primary, children, d_bar=0.004)
    assert report["connectivity"] == "radius_knn"
    assert report["transferred_vertex_count"] > 0
    children[0].points *= 10
    scaled, _ = _resolve_primary_o1_ownership(p * 10, labels, primary * 10, children, d_bar=0.04)
    np.testing.assert_array_equal(scaled, result)
