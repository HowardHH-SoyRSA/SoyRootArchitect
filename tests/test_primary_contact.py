import numpy as np

from soyrootbio.primary_contact import (
    mark_higher_order_primary_contact_qc,
    restrict_higher_order_primary_contacts,
)
from soyrootbio.types import RootPath


def _root(root_id, order, parent_id):
    return RootPath(root_id, np.array([[0., 0., 0.], [1., 0., 0.]]),
                    order=order, parent_id=parent_id)


def test_only_higher_order_native_primary_contacts_are_repaired():
    points = np.array([
        [0., 0., 0.], [0., 1., 0.], [-1., 0., 0.],
        [1., 0., 0.], [2., 0., 0.], [2., 1., 0.], [1., 1., 0.],
        [1., 2., 0.], [2., 2., 0.], [2., 3., 0.], [1., 3., 0.],
        [0.1, 0.1, 0.],
    ])
    labels = np.array([0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 2])
    faces = np.array([[0, 1, 2], [0, 3, 6], [3, 4, 6], [4, 5, 6],
                      [1, 7, 10], [7, 8, 10], [8, 9, 10]])
    roots = [_root("o1", 1, "primary"), _root("o2", 2, "o1"),
             _root("o3", 3, "o2")]
    after, report = restrict_higher_order_primary_contacts(
        points, labels, roots, triangles=faces, d_bar=1.)
    assert report["contact_root_count"] == 2
    assert report["unresolved_root_count"] == 0
    assert report["requires_correction"] is False
    assert {row["root_id"] for row in report["contacts"]} == {"o2", "o3"}
    assert np.all(after[[0, 1, 2, 4, 5, 8, 9, 11]] == labels[[0, 1, 2, 4, 5, 8, 9, 11]])
    assert np.all(after[[3, 6, 7, 10]] == -2)
    assert report["changed_vertex_count"] == 4
    assert sorted(v for row in report["contacts"]
                  for v in row["changed_child_vertex_indices"]) == [3, 6, 7, 10]
    mark_higher_order_primary_contact_qc(roots, report)
    assert not any("primary_contact" in flag for flag in roots[0].qc_flags)
    assert "primary_contact_repaired" in roots[1].qc_flags
    assert "primary_contact_segmentation_error" in roots[2].qc_flags
    reordered_roots = [roots[2], roots[0], roots[1]]
    relabel = np.array([-2, -1, 0, 2, 3, 1])
    reordered_labels = relabel[labels + 2]
    reordered_after, reordered_report = restrict_higher_order_primary_contacts(
        points, reordered_labels, reordered_roots, triangles=faces, d_bar=1.)
    undo = np.array([-2, -1, 0, 3, 1, 2])
    np.testing.assert_array_equal(undo[reordered_after + 2], after)
    assert {row["root_id"]: row["status"] for row in reordered_report["contacts"]} == \
           {row["root_id"]: row["status"] for row in report["contacts"]}
    # A second pass cannot create a new primary contact.
    again, second = restrict_higher_order_primary_contacts(
        points, after, roots, triangles=faces, d_bar=1.)
    np.testing.assert_array_equal(again, after)
    assert second["contact_root_count"] == 0


def test_contact_moves_primary_side_when_child_cut_would_split_body():
    points = np.array([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.],
                       [0., 3., 0.], [1., 0., 0.], [2., 0., 0.],
                       [2., 1., 0.], [1., 1., 0.], [3., 0., 0.],
                       [1., 3., 0.], [3., 1., 0.]])
    labels = np.array([0, 0, 0, 0, 2, 2, 2, -1, -1, 1, -1])
    faces = np.array([[0, 1, 2], [1, 2, 3], [1, 3, 9],
                      [0, 4, 7], [4, 5, 8], [4, 6, 10]])
    roots = [_root("o1", 1, "primary"), _root("o2", 2, "o1")]
    after, report = restrict_higher_order_primary_contacts(
        points, labels, roots, triangles=faces, d_bar=1.)
    assert after[0] == -2
    np.testing.assert_array_equal(after[1:], labels[1:])
    row = report["contacts"][0]
    assert row["status"] == "reassigned_primary_seam_uncertain"
    assert row["child_seam_status"] == "unresolved_would_split_owned_surface"
    assert row["remaining_contact_edge_count"] == 0
    assert row["changed_primary_vertex_indices"] == [0]
    assert report["requires_correction"] is False
    # The unrelated O1 attachment through the primary remains intact.
    assert np.all(after[[1, 3, 9]] == labels[[1, 3, 9]])


def test_contact_cut_rolls_back_if_both_sides_would_split_or_disappear():
    points = np.array([[0., 0., 0.], [0., 1., 0.], [1., 0., 0.],
                       [2., 0., 0.], [2., 1., 0.], [3., 0., 0.], [3., 1., 0.]])
    labels = np.array([0, 0, 1, 1, 1, -1, -1])
    faces = np.array([[0, 1, 2], [2, 3, 5], [2, 4, 6]])
    root = _root("o2", 2, "o1")
    after, report = restrict_higher_order_primary_contacts(
        points, labels, [root], triangles=faces, d_bar=1.)
    np.testing.assert_array_equal(after, labels)
    assert report["contacts"][0]["child_seam_status"] == "unresolved_would_split_owned_surface"
    assert report["contacts"][0]["status"] == "unresolved_would_split_primary_surface"
    assert report["requires_correction"] is True
    assert report["contacts"][0]["remaining_contact_edge_count"] > 0
    mark_higher_order_primary_contact_qc([root], report)
    assert "primary_contact_unresolved" in root.qc_flags


def test_primary_side_reassignment_preserves_the_only_o1_contact():
    points = np.array([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.],
                       [1., 0., 0.], [2., 0., 0.], [2., 1., 0.],
                       [1., 1., 0.], [3., 0., 0.], [1., 3., 0.]])
    labels = np.array([0, 0, 0, 2, 2, 2, -1, -1, 1])
    faces = np.array([[0, 1, 2], [0, 8, 6],
                      [0, 3, 6], [3, 4, 7], [3, 5, 7]])
    roots = [_root("o1", 1, "primary"), _root("o2", 2, "o1")]
    after, report = restrict_higher_order_primary_contacts(
        points, labels, roots, triangles=faces, d_bar=1.)
    np.testing.assert_array_equal(after, labels)
    assert report["contacts"][0]["status"] == "unresolved_would_remove_o1_contact"
    assert report["contacts"][0]["remaining_contact_edge_count"] > 0


def test_child_side_reassignment_preserves_declared_parent_attachment():
    points = np.array([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.],
                       [1., 0., 0.], [2., 0., 0.], [2., 1., 0.],
                       [1., 1., 0.], [1., 2., 0.], [3., 1., 0.],
                       [1., 3., 0.]])
    labels = np.array([0, 0, 0, 2, 2, 2, -1, 1, -1, -1])
    faces = np.array([[0, 1, 2], [0, 3, 6], [3, 4, 5],
                      [3, 7, 8], [1, 7, 9]])
    roots = [_root("o1", 1, "primary"), _root("o2", 2, "o1")]
    after, report = restrict_higher_order_primary_contacts(
        points, labels, roots, triangles=faces, d_bar=1.)
    assert after[0] == -2
    assert after[3] == 2
    row = report["contacts"][0]
    assert row["child_seam_status"] == "unresolved_would_remove_parent_attachment"
    assert row["status"] == "reassigned_primary_seam_uncertain"


def test_protected_and_meshless_contacts_are_left_for_qc():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    labels = np.array([0, 1, 1])
    root = _root("o2", 2, "o1")
    protected = np.array([False, True, True])
    after, report = restrict_higher_order_primary_contacts(
        points, labels, [root], triangles=np.array([[0, 1, 2]]),
        d_bar=1., excluded_mask=protected)
    np.testing.assert_array_equal(after, labels)
    assert report["contacts"][0]["child_seam_status"] == "unresolved_protected_contact"
    assert report["contacts"][0]["status"] == "unresolved_would_split_primary_surface"
    no_mesh, report = restrict_higher_order_primary_contacts(
        points, labels, [root], triangles=None, d_bar=1.)
    np.testing.assert_array_equal(no_mesh, labels)
    assert report["status"] == "unresolved_no_mesh"
