import json

import numpy as np

from soyrootbio.parent_contact import mark_parent_contact_qc, reconcile_parent_contacts
from soyrootbio.pipeline import _merge_final_contact_proposals
from soyrootbio.types import RootPath


def _claim(label, first_vertex, count, *, supported=True):
    return {"components": [{
        "source_label": label, "first_vertex": first_vertex,
        "vertex_count": count,
        "candidates": [{"label": 0, "boundary_fraction": 1.0,
                        "target_supported": supported, "compact": True,
                        "local_anchor": True, "median_local_radius": .5,
                        "claim_score": .8}],
    }]}


def _strips_with_parent_patch():
    points = []
    faces = []
    labels = []
    for label, x in [(0, 0.0), (1, 3.0)]:
        offset = len(points)
        for z in np.arange(0.0, 10.01, .25):
            points.extend([[x + .2, y, z] for y in (-.12, 0.0, .12)])
            labels.extend([label] * 3)
        for i in range(40):
            for j in range(2):
                a = offset + 3 * i + j
                faces.extend([[a, a + 1, a + 3], [a + 1, a + 4, a + 3]])
    patch = np.array([3 * i + 1 for i in (20, 21, 22)])
    labels = np.asarray(labels, int)
    labels[patch] = 1
    root = RootPath("child", np.array([[3., 0., 0.], [3., 0., 10.]]),
                    parent_id="primary", order=1)
    return np.asarray(points), labels, np.array([[0., 0., 0.], [0., 0., 10.]]), [root], np.asarray(faces), patch


def test_discrete_parent_patch_moves_only_with_clear_parent_evidence():
    points, labels, primary, roots, faces, patch = _strips_with_parent_patch()
    kwargs = dict(triangles=faces, d_bar=.15,
                  cleanup_report=_claim(1, int(patch[0]), len(patch)))
    result, report = reconcile_parent_contacts(points, labels, primary, roots, **kwargs)
    np.testing.assert_array_equal(result[patch], np.zeros(len(patch), int))
    np.testing.assert_array_equal(result[labels == 1][len(patch):],
                                  labels[labels == 1][len(patch):])
    assert report["changed_vertex_count"] == len(patch)
    assert report["discrete_patch_count"] == 1
    assert report["unresolved_patch_count"] == 0
    json.dumps(report, allow_nan=False)
    repeated, repeated_report = reconcile_parent_contacts(
        points, result, primary, roots, **kwargs)
    np.testing.assert_array_equal(repeated, result)
    assert repeated_report["changed_vertex_count"] == 0


def test_uncertain_parent_claim_keeps_patch_and_marks_unresolved():
    points, labels, primary, roots, faces, patch = _strips_with_parent_patch()
    result, report = reconcile_parent_contacts(
        points, labels, primary, roots, triangles=faces, d_bar=.15,
        cleanup_report=_claim(1, int(patch[0]), len(patch), supported=False))
    np.testing.assert_array_equal(result, labels)
    assert report["unresolved_patch_count"] == 1
    mark_parent_contact_qc(roots, report)
    assert "parent_contact_unresolved" in roots[0].qc_flags
    assert "parent_contact_discrete_unresolved" in roots[0].qc_flags


def test_all_unresolved_contact_decisions_surface_in_root_qc():
    roots = [RootPath("child", np.array([[0., 0., 0.], [0., 0., 1.]]))]
    mark_parent_contact_qc(roots, {"contacts": [
        {"root_id": "child", "status": "unresolved_no_distal_attachment_support"},
    ]})
    assert "parent_contact_unresolved" in roots[0].qc_flags
    mark_parent_contact_qc(roots, {"contacts": [
        {"root_id": "child", "status": "unresolved_iteration_limit"},
    ]})
    assert "parent_contact_discrete_unresolved" in roots[0].qc_flags
    mark_parent_contact_qc(roots, {"contacts": []})
    assert not any(flag.startswith("parent_contact_") for flag in roots[0].qc_flags)


def test_simultaneous_contact_decisions_withhold_a_conflicted_root_atomically():
    roots = [RootPath("a", np.array([[0., 0., 0.]])),
             RootPath("b", np.array([[1., 0., 0.]]))]
    frozen = np.array([1, 1, 2, 2, 0])
    parent = np.array([0, 1, 0, 2, 0])
    attachment = np.array([-2, 1, 2, -2, 0])
    parent_report = {
        "changed_vertex_count": 2, "unresolved_patch_count": 0,
        "contacts": [
            {"root_id": "a", "parent_id": "primary", "label": 1,
             "status": "reassigned_to_supported_parent",
             "changed_vertex_count": 1},
            {"root_id": "b", "parent_id": "primary", "label": 2,
             "status": "reassigned_to_supported_parent",
             "changed_vertex_count": 1},
        ],
    }
    restriction = {"changed_vertex_count": 2,
                   "changed_by_root": {"a": [0], "b": [3]}}
    merged = _merge_final_contact_proposals(
        frozen, parent, parent_report, attachment, restriction, roots)
    assert merged.tolist() == [1, 1, 0, -2, 0]
    assert parent_report["competing_restriction_root_ids"] == ["a"]
    assert parent_report["contacts"][0]["status"] == "unresolved_competing_restriction"
    assert restriction["changed_by_root"] == {"b": [3]}
    mark_parent_contact_qc(roots, parent_report)
    assert "parent_contact_discrete_unresolved" in roots[0].qc_flags


def test_no_mesh_status_survives_empty_proposal_merge():
    labels = np.array([0, 1])
    parent_report = {"status": "unresolved_no_mesh", "changed_vertex_count": 0,
                     "unresolved_patch_count": 0, "contacts": []}
    restriction = {"changed_vertex_count": 0, "changed_by_root": {}}
    merged = _merge_final_contact_proposals(
        labels, labels, parent_report, labels, restriction,
        [RootPath("child", np.array([[0., 0., 0.]]))])
    np.testing.assert_array_equal(merged, labels)
    assert parent_report["status"] == "unresolved_no_mesh"


def test_parent_transfer_waits_when_recipient_seam_is_restricted():
    roots = [RootPath("parent", np.array([[0., 0., 0.]])),
             RootPath("child", np.array([[1., 0., 0.]]), parent_id="parent", order=2)]
    frozen = np.array([1, 1, 2, 2])
    parent = np.array([1, 1, 1, 2])
    attachment = np.array([-2, 1, 2, 2])
    parent_report = {"status": "corrected", "changed_vertex_count": 1,
                     "unresolved_patch_count": 0,
                     "contacts": [{"root_id": "child", "parent_id": "parent", "label": 2,
                                   "status": "reassigned_to_supported_parent",
                                   "changed_vertex_count": 1}]}
    restriction = {"changed_vertex_count": 1,
                   "changed_by_root": {"parent": [0]}}
    merged = _merge_final_contact_proposals(
        frozen, parent, parent_report, attachment, restriction, roots)
    assert merged.tolist() == [-2, 1, 2, 2]
    assert parent_report["recipient_restriction_root_ids"] == ["child"]
    assert parent_report["contacts"][0]["status"] == "unresolved_competing_restriction"


def test_circumferential_parent_patch_can_join_observed_parent_components():
    n = 12
    theta = np.arange(n) * 2 * np.pi / n
    rings = [np.column_stack((.5*np.cos(theta), .5*np.sin(theta),
                              np.full(n, z))) for z in (-.25, 0., .25)]
    points = np.vstack(rings).tolist()
    labels = [0]*n + [1]*n + [0]*n
    faces = []
    for layer in range(2):
        for i in range(n):
            a = layer*n+i
            b = layer*n+(i+1)%n
            c = (layer+1)*n+i
            d = (layer+1)*n+(i+1)%n
            faces.extend([[a, b, c], [b, d, c]])
    offset = len(points)
    for z in np.arange(0., 10.01, .25):
        points.extend([[3.2, y, z] for y in (-.12, 0., .12)])
        labels.extend([1]*3)
    for i in range(40):
        for j in range(2):
            a = offset+3*i+j
            faces.extend([[a, a+1, a+3], [a+1, a+4, a+3]])
    points = np.asarray(points)
    labels = np.asarray(labels)
    root = RootPath("child", np.array([[0., 0., 0.], [3., 0., 0.],
                                        [3., 0., 10.]]),
                    parent_id="primary", order=1, body_start_index=1,
                    raw_start_point=np.zeros(3))
    result, report = reconcile_parent_contacts(
        points, labels, np.array([[0., 0., -2.], [0., 0., 2.]]),
        [root], triangles=np.asarray(faces), d_bar=.3,
        cleanup_report=_claim(1, n, n))
    np.testing.assert_array_equal(result[n:2*n], np.zeros(n, int))
    assert report["contacts"][0]["joins_supported_parent_components"]
    assert report["contacts"][0]["angular_coverage_degrees"] >= 300
    np.testing.assert_array_equal(result[offset:], labels[offset:])


def test_missing_mesh_and_excluded_region_do_not_create_contact_bridges():
    points, labels, primary, roots, faces, patch = _strips_with_parent_patch()
    result, report = reconcile_parent_contacts(
        points, labels, primary, roots, triangles=None, d_bar=.15)
    np.testing.assert_array_equal(result, labels)
    assert report["status"] == "unresolved_no_mesh"
    excluded = np.zeros(len(points), bool)
    excluded[patch] = True
    result, _ = reconcile_parent_contacts(
        points, labels, primary, roots, triangles=faces, d_bar=.15,
        cleanup_report=_claim(1, int(patch[0]), len(patch)),
        excluded_mask=excluded)
    np.testing.assert_array_equal(result[patch], labels[patch])
