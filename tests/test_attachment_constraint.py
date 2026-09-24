import numpy as np
from scipy.spatial import cKDTree
from types import SimpleNamespace

from soyrootbio.attachment_constraint import (
    AttachmentBounds,
    assess_attachment_footprints,
    restrict_rejected_contacts,
)
from soyrootbio.types import RootPath
from soyrootbio.topology import repair_root_hierarchy
from soyrootbio.pipeline import _defer_rejected_attachment_starts


def _disk_and_exposed_tube():
    """Observed disk and a connected tube, with no inferred mesh bridges."""
    n = 16
    angles = np.arange(n) * 2 * np.pi / n
    rings = [(.025, 0.), (.05, 0.)] + [(.05, float(z)) for z in np.arange(.02, .501, .02)]
    points = [[0., 0., 0.]]
    for radius, z in rings:
        points.extend([[radius * np.cos(a), radius * np.sin(a), z] for a in angles])
    points = np.asarray(points, float)
    faces = []
    inner = 1
    for j in range(n):
        faces.append([0, inner + j, inner + (j + 1) % n])
    for ring in range(len(rings) - 1):
        a = 1 + ring * n
        b = a + n
        for j in range(n):
            q = (j + 1) % n
            faces.extend([[a+j, b+j, b+q], [a+j, b+q, a+q]])
    labels = np.ones(len(points), int)
    labels[:1+n] = 0
    primary = np.array([[-.2, 0., 0.], [.2, 0., 0.]])
    child = RootPath("child", np.array([[0., 0., z] for z in np.arange(0., .501, .02)]),
                     parent_id="primary", raw_start_point=np.zeros(3))
    return points, labels, primary, [child], np.asarray(faces, int)


def test_observed_disk_passes_bounded_triangle_geometry():
    points, labels, primary, roots, faces = _disk_and_exposed_tube()
    row = assess_attachment_footprints(points, labels, primary, roots,
                                       triangles=faces, d_bar=.01)["junctions"][0]
    assert row["status"] == "accepted"
    assert row["patch_component_count"] == 1
    assert row["boundary_component_count"] == 1
    assert row["euler_characteristic"] == 1
    assert row["area"] < row["area_limit"]
    assert .04 < row["child_radius"] < .06


def test_longitudinal_limit_rejects_and_restricts_only_proximal_child_interface():
    points, labels, primary, roots, faces = _disk_and_exposed_tube()
    report = assess_attachment_footprints(
        points, labels, primary, roots, triangles=faces, d_bar=.01,
        bounds=AttachmentBounds(longitudinal_radii=.5))
    row = report["junctions"][0]
    assert row["status"] == "rejected_oversized_or_elongated"
    assert row["longitudinal_extent"] > row["longitudinal_limit"]
    after, restriction = restrict_rejected_contacts(
        labels, report, roots, points, triangles=faces, d_bar=.01)
    assert restriction["changed_vertex_count"] > 0
    assert restriction["connectivity_rollbacks"] == []
    assert np.all(after[(points[:, 2] >= .2)] == labels[(points[:, 2] >= .2)])
    assert np.all(after[labels == 0] == 0)


def test_compactness_is_an_independent_footprint_gate():
    points, labels, primary, roots, faces = _disk_and_exposed_tube()
    row = assess_attachment_footprints(
        points, labels, primary, roots, triangles=faces, d_bar=.01,
        bounds=AttachmentBounds(maximum_inverse_compactness=1.001))["junctions"][0]
    assert row["status"] == "rejected_oversized_or_elongated"
    assert row["area"] < row["area_limit"]
    assert row["inverse_compactness"] > row["inverse_compactness_limit"]


def test_open_junction_is_unresolved_and_never_forced_closed():
    points, labels, primary, roots, faces = _disk_and_exposed_tube()
    faces = faces[1:]
    row = assess_attachment_footprints(points, labels, primary, roots,
                                       triangles=faces, d_bar=.01)["junctions"][0]
    assert row["status"] == "unresolved_open_or_nonmanifold_junction"


def test_collar_inflation_cannot_increase_distal_child_radius():
    points, labels, primary, roots, faces = _disk_and_exposed_tube()
    baseline = assess_attachment_footprints(points, labels, primary, roots,
                                            triangles=faces, d_bar=.01)["junctions"][0]
    inflated = points.copy()
    # Only the basal child ring is flared; the stable exposed sections remain.
    inflated[17:33, :2] *= 2.5
    changed = assess_attachment_footprints(inflated, labels, primary, roots,
                                           triangles=faces, d_bar=.01)["junctions"][0]
    assert changed["child_radius"] == baseline["child_radius"]
    assert changed["area_limit"] == baseline["area_limit"]


def test_vertex_pinched_nonmanifold_junction_is_unresolved():
    points, labels, primary, roots, faces = _disk_and_exposed_tube()
    first = len(points)
    points = np.vstack([points, [[.01, 0., .01], [0., .01, .01], [-.01, 0., .01]]])
    labels = np.r_[labels, [0, 0, 0]]
    tetra = np.array([[0, first, first+1], [0, first+1, first+2],
                      [0, first+2, first], [first, first+2, first+1]])
    faces = np.vstack([faces, tetra])
    row = assess_attachment_footprints(points, labels, primary, roots,
                                       triangles=faces, d_bar=.01)["junctions"][0]
    assert row["status"] == "unresolved_nonmanifold_vertex_link"


def test_nearby_disconnected_parent_surface_is_not_an_accepted_subpatch():
    points, labels, primary, roots, faces = _disk_and_exposed_tube()
    first = len(points)
    points = np.vstack([points, [[.04, 0., .025], [.05, 0., .025],
                                 [.04, .01, .025], [.04, 0., .035]]])
    labels = np.r_[labels, [0, 0, 0, 0]]
    faces = np.vstack([faces, [[first, first+1, first+2],
                               [first, first+1, first+3],
                               [first, first+2, first+3],
                               [first+1, first+2, first+3]]])
    row = assess_attachment_footprints(points, labels, primary, roots,
                                       triangles=faces, d_bar=.01)["junctions"][0]
    assert row["status"] == "accepted"
    assert row["candidate_component_count"] == 2
    assert row["discarded_noncontact_face_count"] == 4
    assert row["patch_component_count"] == 1


def test_no_mesh_preserves_all_labels_and_reports_qc():
    points, labels, primary, roots, _ = _disk_and_exposed_tube()
    report = assess_attachment_footprints(points, labels, primary, roots,
                                          triangles=None, d_bar=.01)
    after, restriction = restrict_rejected_contacts(labels, report, roots, points)
    assert report["junctions"][0]["status"] == "unresolved_no_mesh"
    assert restriction["changed_vertex_count"] == 0
    np.testing.assert_array_equal(after, labels)


def test_sibling_assessment_uses_frozen_evidence_independent_of_root_order():
    p1, l1, _, roots1, f1 = _disk_and_exposed_tube()
    p2 = p1 + np.array([1., 0., 0.])
    l2 = np.where(l1 == 1, 2, 0)
    points = np.vstack([p1, p2])
    faces = np.vstack([f1, f1 + len(p1)])
    labels = np.r_[l1, l2]
    first = roots1[0]
    first.root_id = "first"
    second = RootPath("second", first.points + np.array([1., 0., 0.]),
                      parent_id="primary", raw_start_point=np.array([1., 0., 0.]))
    primary = np.array([[-.2, 0., 0.], [1.2, 0., 0.]])
    a = assess_attachment_footprints(points, labels, primary, [first, second],
                                     triangles=faces, d_bar=.01)
    permuted = np.where(labels == 1, 2, np.where(labels == 2, 1, labels))
    b = assess_attachment_footprints(points, permuted, primary, [second, first],
                                     triangles=faces, d_bar=.01)
    statuses_a = {row["root_id"]: row["status"] for row in a["junctions"]}
    statuses_b = {row["root_id"]: row["status"] for row in b["junctions"]}
    assert statuses_a == statuses_b == {"first": "accepted", "second": "accepted"}


def test_attachment_geometry_is_scale_invariant():
    points, labels, primary, roots, faces = _disk_and_exposed_tube()
    for factor in (1e-3, 1., 1e3):
        root = RootPath("child", roots[0].points * factor,
                        parent_id="primary", raw_start_point=np.zeros(3))
        row = assess_attachment_footprints(
            points * factor, labels, primary * factor, [root],
            triangles=faces, d_bar=.01 * factor)["junctions"][0]
        assert row["status"] == "accepted"
        assert row["area"] / row["area_limit"] < 1


def test_rejected_footprint_reassesses_neighbor_from_same_labels():
    points, labels, primary, roots, faces = _disk_and_exposed_tube()
    neighbor = RootPath("neighbor", roots[0].points + np.array([.03, 0., 0.]),
                        parent_id="primary", raw_start_point=np.array([.03, 0., 0.]))
    report = assess_attachment_footprints(
        points, labels, primary, [roots[0], neighbor], triangles=faces, d_bar=.01,
        bounds=AttachmentBounds(longitudinal_radii=.5))
    first, second = report["junctions"]
    assert first["status"] == "rejected_oversized_or_elongated"
    assert first["neighboring_insertions_for_reassessment"] == ["neighbor"]
    assert first["neighbor_reassessment"][0]["attachment_status"] == second["status"]
    assert second["status"] == "unresolved_insufficient_exposed_surface"


def test_rejected_attachment_constrains_parent_inference_before_order_recompute():
    primary = np.array([[0., 0., 1.], [0., 0., .8], [0., 0., .5], [0., 0., 0.]])
    parent = RootPath("first", np.array([[0., 0., .8], [.2, 0., .75], [.4, 0., .7]]),
                      order=1, parent_id="primary")
    child = RootPath("second", np.array([[.2, 0., .75], [.2, .15, .72], [.2, .3, .7]]),
                     order=2, parent_id="first")
    evidence = {"per_order": [{"assessment": {"junctions": [
        {"root_id": "first", "status": "accepted"},
        {"root_id": "second", "status": "rejected_oversized_or_elongated",
         "neighboring_insertions_for_reassessment": []},
    ]}}]}
    repaired, report = repair_root_hierarchy(
        primary, [parent, child], d_bar=.005, attachment_evidence=evidence)
    assert any(row["root_id_before_stable_ids"] == "second" and
               row["decision"] == "retain_traced_parent_pending_direct_attachment_evidence"
               for row in report.attachment_constraint_decisions)
    assert any("attachment_parent_unresolved" in root.qc_flags for root in repaired)


def test_unresolved_attachment_cannot_be_reparented_from_external_contact():
    primary = np.array([[0., 0., 1.], [0., 0., .8], [0., 0., .5], [0., 0., 0.]])
    parent = RootPath("first", np.array([[0., 0., .8], [.2, 0., .75], [.4, 0., .7]]),
                      order=1, parent_id="primary")
    child = RootPath("second", np.array([[.2, 0., .75], [.2, .15, .72], [.2, .3, .7]]),
                     order=2, parent_id="first")
    evidence = {"per_order": [{"assessment": {"junctions": [
        {"root_id": "first", "status": "accepted"},
        {"root_id": "second", "status": "unresolved_open_or_nonmanifold_junction"},
    ]}}]}
    repaired, report = repair_root_hierarchy(
        primary, [parent, child], d_bar=.005, attachment_evidence=evidence)
    assert any(row["root_id_before_stable_ids"] == "second" and
               row["decision"] == "retain_traced_parent_pending_direct_attachment_evidence"
               for row in report.attachment_constraint_decisions)
    assert any("attachment_parent_unresolved" in root.qc_flags for root in repaired)


def test_rejected_junction_defers_only_local_later_order_starts():
    near = SimpleNamespace(point=np.array([.03, 0., 0.]),
                           primary_point=np.array([0., 0., 0.]))
    distal = SimpleNamespace(point=np.array([0., 0., .4]),
                             primary_point=np.array([0., 0., .35]))
    eligible, deferred = _defer_rejected_attachment_starts(
        [near, distal], [(cKDTree(np.array([[0., 0., 0.]])), .1)])
    assert deferred == 1
    assert eligible == [distal]


def test_restriction_rolls_back_if_it_splits_stable_child_body():
    points = np.array([[0., 0., .5], [-.1, 0., 2.],
                       [.1, 0., 2.2], [1., 0., 1.]])
    faces = np.array([[0, 1, 3], [0, 2, 3]])
    labels = np.array([1, 1, 1, 0])
    root = RootPath("child", np.array([[0., 0., 0.], [0., 0., 3.]]),
                    parent_id="primary")
    report = {"junctions": [{"root_id": "child",
                            "status": "rejected_oversized_or_elongated",
                            "patch_vertex_indices": [0],
                            "radius_evidence": {"stable_arc_start": 2.1}}]}
    after, restriction = restrict_rejected_contacts(
        labels, report, [root], points, triangles=faces, d_bar=1.)
    np.testing.assert_array_equal(after, labels)
    assert restriction["connectivity_rollbacks"] == ["child"]
    assert restriction["changed_vertex_count"] == 0
