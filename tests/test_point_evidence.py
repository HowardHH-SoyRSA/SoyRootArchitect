import numpy as np
import pytest

from soyrootbio.point_evidence import assess_point_only_evidence
from soyrootbio.types import RootPath


def sheets(gap=0.2):
    x, y = np.meshgrid(np.linspace(0, 1, 15), np.linspace(0, 1, 15))
    sheet = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    points = np.vstack([sheet, sheet + [0, 0, gap]])
    labels = np.repeat([0, 1], len(sheet))
    primary = np.array([[0, 0, 0], [1, 0, 0]], float)
    root = RootPath("o2", primary + [0, 0, gap], order=2, parent_id="unobserved-o1")
    return points, labels, primary, [root]


def test_separated_sheets_have_no_observed_contact_but_remain_unresolved():
    points, labels, primary, roots = sheets()
    report = assess_point_only_evidence(points, labels, primary, roots)
    assert report["contacts"] == []
    assert report["status"] == "unresolved_no_mesh"
    assert report["native_patch_compliance"] == "unresolved_no_mesh"
    assert report["reliable_frame_point_count"] == len(points)
    assert report["tangent_compatible_edge_count"] > 0


def test_near_parallel_surfaces_show_proximity_without_tangent_contact():
    points, labels, primary, roots = sheets(gap=0.06)
    before = points.copy(), labels.copy(), roots[0].points.copy()
    report = assess_point_only_evidence(points, labels, primary, roots)
    assert report["higher_order_primary_possible_contact_count"] == 1
    assert report["contacts"][0]["proximity_edge_count"] > 0
    assert report["contacts"][0]["tangent_compatible_edge_count"] == 0
    assert report["changed_vertex_count"] == 0
    assert report["mesh_generated"] is False
    np.testing.assert_array_equal(points, before[0])
    np.testing.assert_array_equal(labels, before[1])
    np.testing.assert_array_equal(roots[0].points, before[2])


def test_gap_is_not_bridged_by_unbounded_neighbors():
    points, labels, primary, roots = sheets()
    labels[:] = 0
    points[len(points) // 2:] += [4, 0, 0]
    report = assess_point_only_evidence(points, labels, primary, roots)
    assert report["roots"][0]["multi_point_component_count"] == 2
    assert report["discrete_patch_candidate_count"] == 1


def test_excluded_and_negative_labels_are_barriers():
    points, labels, primary, roots = sheets(gap=0.06)
    excluded = labels == 1
    report = assess_point_only_evidence(points, labels, primary, roots, excluded_mask=excluded)
    assert report["contacts"] == []
    labels[excluded] = -2
    other = assess_point_only_evidence(points, labels, primary, roots)
    assert other["contacts"] == []
    assert other["excluded_or_negative_point_count"] == int(excluded.sum())


def test_sampled_unassigned_strip_withholds_shortcut_edges():
    x, y = np.meshgrid(np.arange(9), np.arange(9))
    points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)]).astype(float)
    labels = np.where(points[:, 0] == 4, -1, 0)
    report = assess_point_only_evidence(points, labels, points[:2], [])
    assert report["barrier_rejected_edge_count"] > 0
    assert report["roots"][0]["multi_point_component_count"] == 2


def test_occupied_volume_and_sparse_evidence_never_certify_mesh_rules():
    rng = np.random.default_rng(17)
    points = rng.uniform(-1, 1, size=(300, 3))
    report = assess_point_only_evidence(points, np.zeros(len(points), int), points[:2], [], input_mode="occupied_volume")
    assert report["native_contact_compliance"] == "unresolved_no_mesh"
    assert report["frame_definition"].endswith("principal axis")
    sparse = assess_point_only_evidence(points[:2], np.zeros(2, int), points[:2], [])
    assert sparse["reliable_frame_point_count"] == 0
    assert sparse["neighborhood_edge_count"] == 0
    assert sparse["changed_vertex_count"] == 0


def test_duplicates_are_reported_and_invalid_labels_rejected():
    points, labels, primary, roots = sheets()
    points = np.vstack([points, points[0]])
    labels = np.r_[labels, 0]
    report = assess_point_only_evidence(points, labels, primary, roots)
    assert report["duplicate_neighbor_pair_count"] > 0
    with pytest.raises(ValueError, match="existing root"):
        assess_point_only_evidence(points, labels + 3, primary, roots)
