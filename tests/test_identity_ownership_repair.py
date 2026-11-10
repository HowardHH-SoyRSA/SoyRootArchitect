from copy import deepcopy
import json

import numpy as np
import pytest

from soyrootbio.identity_repair import (
    freeze_identity_references, reconcile_identity_ownership, _transaction_guard,
    _components,
)
from soyrootbio.mesh_geometry import MeshGeometryContext
from soyrootbio.surface_patch_audit import audit_discrete_child_patches
from soyrootbio.types import RootPath


def interrupted_tube(seam_owner=-2):
    theta = np.arange(24) * 2 * np.pi / 24
    stations = np.arange(81) * .005
    tube = np.array([[x, .01 * np.cos(t), .01 * np.sin(t)]
                     for x in stations for t in theta])
    points = np.vstack([tube, tube + [0., .15, 0.]])
    faces = []
    for offset in (0, len(tube)):
        for i in range(len(stations) - 1):
            for j in range(24):
                a, b = offset + 24 * i + j, offset + 24 * i + (j + 1) % 24
                faces.extend([[a, b, a + 24], [b, b + 24, a + 24]])
    labels = np.r_[np.ones(len(tube), int), np.zeros(len(tube), int)]
    labels[(np.arange(len(points)) < len(tube)) & (points[:, 0] < .015)] = 0
    seam = (np.arange(len(points)) < len(tube)) & (points[:, 0] >= .19) & (points[:, 0] <= .21)
    labels[seam] = seam_owner
    primary = np.array([[0., .15, 0.], [.4, .15, 0.]])
    root = RootPath("child", np.array([[0., 0., 0.], [.4, 0., 0.]]), mean_radius=.01)
    return points, labels, primary, [root], np.asarray(faces), seam


def run(case, **kwargs):
    points, labels, primary, roots, faces, _ = case
    return reconcile_identity_ownership(
        points, labels, primary, roots, triangles=faces, d_bar=.004,
        primary_top_reference=np.array([0., .15, .4]), **kwargs)


@pytest.mark.parametrize("seam_owner", [-2, 0])
def test_independent_observed_tube_repairs_seam_without_changing_geometry(seam_owner):
    case = interrupted_tube(seam_owner)
    frozen = deepcopy(case)
    after, report = run(case)
    points, before, primary, roots, faces, seam = case
    assert report["changed_vertex_count"] == int(seam.sum())
    assert np.all(after[seam] == 1)
    np.testing.assert_array_equal(after[~seam], before[~seam])
    for original, actual in zip(frozen[:3], case[:3]):
        np.testing.assert_array_equal(original, actual)
    np.testing.assert_array_equal(faces, frozen[4])
    np.testing.assert_array_equal(roots[0].points, frozen[3][0].points)
    assert roots[0].parent_id == frozen[3][0].parent_id
    assert report["decisions"][0]["identity_resolved"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("barrier", ["unassigned", "nodule", "excluded", "missing_mesh", "above_top"])
def test_native_or_protected_gap_is_not_filled(barrier):
    case = list(interrupted_tube())
    p, labels, _, _, faces, seam = case
    options = {}
    if barrier == "unassigned":
        labels[seam] = -1
    elif barrier == "nodule":
        labels[seam] = -3
    elif barrier == "excluded":
        options["excluded_mask"] = seam
    elif barrier == "above_top":
        # Same selected top, but part of the tube is now above that reference.
        p[:, 2] += .405
        case[3][0].points[:, 2] += .405
    else:
        crossing = (p[faces, 0].min(axis=1) < .2025) & (p[faces, 0].max(axis=1) > .2025)
        case[4] = faces[~crossing]
    after, report = run(case, **options)
    np.testing.assert_array_equal(after, labels)
    assert report["changed_vertex_count"] == 0
    if barrier == "missing_mesh":
        assert any(s["status"] == "unresolved_native_mesh_gap"
                   for row in report["decisions"] for s in row["seams"])


def test_short_native_path_and_flat_patch_cannot_certify_a_tube():
    case = list(interrupted_tube())
    # The topology is unchanged, but every circumferential ring is flattened.
    case[0][:, 2] = 0.
    after, report = run(case)
    np.testing.assert_array_equal(after, case[1])
    assert report["unresolved_root_count"] == 1


def test_existing_higher_order_primary_contact_is_explicit_not_reparented():
    case = list(interrupted_tube(0))
    case[3][0].order = 2
    case[3][0].parent_id = "unknown-parent"
    after, report = run(case)
    np.testing.assert_array_equal(after, case[1])
    assert case[3][0].order == 2
    assert report["decisions"][0]["status"] == "unresolved_existing_primary_contact"
    assert report["decisions"][0]["native_contact_edge_count"] > 0


def test_legacy_shortened_assessment_is_not_used_as_identity_evidence():
    case = list(interrupted_tube())
    case[3][0].points = np.array([[0., 0., 0.], [.03, 0., 0.]])
    case[3][0].centerline_assessment = {"fit_applied": True, "traits_geometry_source": "accepted_fit"}
    after, report = run(case)
    np.testing.assert_array_equal(after, case[1])
    assert report["decisions"][0]["status"] == "unresolved_missing_prefit_reference"


def test_frozen_identity_survives_later_centerline_shortening():
    case = list(interrupted_tube())
    references = freeze_identity_references(case[2], case[3])
    saved = deepcopy(references)
    case[3][0].points = np.array([[0., 0., 0.], [.03, 0., 0.]])
    case[3][0].centerline_assessment = {"fit_applied": True}
    after, report = run(case, identity_references=references)
    assert report["changed_vertex_count"] == int(case[5].sum())
    assert references == saved
    assert np.all(after[case[5]] == 1)


def test_final_audit_uses_long_prefit_body_not_short_final_stub():
    case = list(interrupted_tube(0))
    p, labels, primary, roots, faces, _ = case
    references = freeze_identity_references(primary, roots)
    # Make the left fragment tiny and the remaining original exposed body long.
    labels[(np.arange(len(p)) < len(p) // 2) & (p[:, 0] >= .02) & (p[:, 0] <= .035)] = 0
    labels[(np.arange(len(p)) < len(p) // 2) & (p[:, 0] >= .19) & (p[:, 0] <= .21)] = 1
    roots[0].points = np.array([[0., 0., 0.], [.015, 0., 0.]])
    roots[0].centerline_assessment = {"fit_applied": True, "prefit_identity_evidence": references["child"]}
    report = audit_discrete_child_patches(p, labels, roots, triangles=faces, d_bar=.004)
    assert report["unresolved_patch_count"] == 1
    assert max(p[report["patches"][0]["vertex_indices"], 0]) < .02
    assert report["patches"][0]["body_anchor_reference_source"] == "persistent_pre_fit_reference"


def test_two_simultaneous_cuts_must_not_split_donor_or_remove_descendant_contact():
    # One native donor path with two alternative routes; each removal alone is
    # safe, but the simultaneous transaction would split its surviving body.
    edges = np.array([[0, 1], [1, 2], [0, 3], [3, 2], [2, 4], [4, 5]])
    before = np.array([0, 0, 0, 0, 0, 3])
    after = np.array([0, 1, 0, 2, 0, 3])
    roots = [RootPath("a", np.array([[0., 0., 0.], [1., 0., 0.]])),
             RootPath("b", np.array([[0., 0., 0.], [1., 0., 0.]])),
             RootPath("descendant", np.array([[0., 0., 0.], [1., 0., 0.]]))]
    components = _components(len(before), edges[(before[edges[:, 0]] == before[edges[:, 1]])])
    reasons = _transaction_guard(before, after, edges, roots, components)
    assert reasons[1] == reasons[2] == "unresolved_would_split_donor_surface"
    after = before.copy()
    after[4] = 1
    reasons = _transaction_guard(before, after, edges, roots, components)
    assert reasons[1] == "unresolved_would_remove_parent_or_descendant_attachment"


def test_competing_measured_parent_envelope_prevents_transfer():
    case = list(interrupted_tube(0))
    # The supposed parent is the same biological tube: transferring the band
    # would merely make the labels prettier without independent child identity.
    case[2][:, 1] = 0.
    case[1][case[0][:, 1] > .1] = -1
    after, report = run(case)
    np.testing.assert_array_equal(after, case[1])
    assert report["changed_vertex_count"] == 0


@pytest.mark.parametrize("factor", [.1, 10.])
def test_length_units_do_not_change_repair_decisions(factor):
    case = interrupted_tube()
    p, labels, primary, roots, faces, seam = case
    roots[0].points *= factor
    roots[0].mean_radius *= factor
    after, report = reconcile_identity_ownership(
        p * factor, labels, primary * factor, roots, triangles=faces,
        d_bar=.004 * factor, primary_top_reference=np.array([0., .15, .4]) * factor)
    assert report["changed_vertex_count"] == int(seam.sum())
    assert np.all(after[seam] == 1)


def test_context_and_no_context_agree_and_second_pass_is_idempotent():
    case = interrupted_tube()
    expected, _ = run(case)
    actual, report = run(case, mesh_context=MeshGeometryContext.build(case[0], case[4]))
    np.testing.assert_array_equal(actual, expected)
    second = list(case)
    second[1] = actual
    final, repeat = run(second)
    np.testing.assert_array_equal(final, actual)
    assert repeat["changed_vertex_count"] == 0


def test_parent_patch_cleanup_cannot_reclassify_long_body_from_short_final_fit():
    from soyrootbio.parent_contact import reconcile_parent_contacts

    p, labels, primary, roots, faces, _ = interrupted_tube(0)
    references = freeze_identity_references(primary, roots)
    large = np.flatnonzero((labels == 1) & (p[:, 0] > .21))
    roots[0].points = np.array([[.015, 0., 0.], [.185, 0., 0.]])
    roots[0].centerline_assessment = {
        "fit_applied": True, "prefit_identity_evidence": references["child"]}
    cleanup = {"components": [{"source_label": 1, "first_vertex": int(large[0]),
        "vertex_count": len(large), "candidates": [{"label": 0, "boundary_fraction": 1.,
            "target_supported": True, "compact": True, "local_anchor": True,
            "median_local_radius": .01, "claim_score": 1.}]}]}
    after, report = reconcile_parent_contacts(p, labels, primary, roots,
        triangles=faces, d_bar=.004, cleanup_report=cleanup)
    np.testing.assert_array_equal(after, labels)
    row = next(r for r in report["contacts"] if r["first_vertex"] == int(large[0]))
    assert row["distal_support"]["independent_reference_body_support"]
    assert row["changed_vertex_count"] == 0
    # A legacy assessed root also cannot use the shortened curve to justify it.
    roots[0].centerline_assessment.pop("prefit_identity_evidence")
    after, report = reconcile_parent_contacts(p, labels, primary, roots,
        triangles=faces, d_bar=.004, cleanup_report=cleanup)
    np.testing.assert_array_equal(after, labels)
    assert all(r["status"] == "unresolved_missing_prefit_reference" for r in report["contacts"])


def test_rejected_attachment_cannot_release_exposed_body_projected_onto_short_fit():
    from soyrootbio.attachment_constraint import restrict_rejected_contacts

    p, labels, primary, roots, faces, _ = interrupted_tube(0)
    references = freeze_identity_references(primary, roots)
    # Releasing this terminal portion would keep one connected proximal
    # remnant, so a pure component-count guard cannot detect the body loss.
    exposed_tail = np.flatnonzero((labels == 1) & (p[:, 0] >= .23))
    roots[0].points = np.array([[.015, 0., 0.], [.185, 0., 0.]])
    roots[0].centerline_assessment = {
        "fit_applied": True, "prefit_identity_evidence": references["child"]}
    assessment = {"junctions": [{"status": "rejected_oversized_or_elongated",
        "patch_vertex_indices": exposed_tail.tolist(),
        "radius_evidence": {"stable_arc_start": .2}}]}
    after, report = restrict_rejected_contacts(labels, assessment, roots, p,
        triangles=faces, d_bar=.004)
    np.testing.assert_array_equal(after, labels)
    assert report["rollback_reasons"]["child"] == "would_remove_independently_supported_exposed_body"
    roots[0].centerline_assessment.pop("prefit_identity_evidence")
    after, report = restrict_rejected_contacts(labels, assessment, roots, p,
        triangles=faces, d_bar=.004)
    np.testing.assert_array_equal(after, labels)
    assert report["rollback_reasons"]["child"] == "unresolved_missing_prefit_reference"
