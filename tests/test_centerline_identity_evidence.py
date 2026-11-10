import copy
import json

import numpy as np
import pytest

from soyrootbio.centerline import (
    _origin_supported_component, _verified_terminal_support, refit_final_centerlines,
)
from soyrootbio.geometry import path_length
from soyrootbio.measurement_evidence import (
    assess_origin_coverage, assess_terminal_direction, build_measurement_evidence,
    infer_component_indices, make_evidence_context,
    rebase_assessment, resolve_measurement_support, validate_measurement_evidence,
    upgrade_measurement_evidence,
)
from soyrootbio.types import RootPath


def tube(start, stop, *, y=0., radius=.04, rings=51, sides=12):
    x = np.linspace(start, stop, rings)
    angle = 2 * np.pi * np.arange(sides) / sides
    points = np.array([[a, y + radius * np.cos(b), radius * np.sin(b)] for a in x for b in angle])
    faces = []
    for i in range(rings - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            faces.extend([[a, b, a + sides], [b, b + sides, a + sides]])
    points = np.vstack([points, [start, y, 0], [stop, y, 0]])
    for j in range(sides):
        faces.extend([[rings * sides, (j + 1) % sides, j],
                      [rings * sides + 1, (rings - 1) * sides + j,
                       (rings - 1) * sides + (j + 1) % sides]])
    line = np.column_stack([x, np.full(rings, y), np.zeros(rings)])
    return points, np.asarray(faces), line


def test_whole_prior_body_beats_a_small_basal_patch_without_largest_count_rule():
    stub, _, _ = tube(0., .12, rings=41)
    body, _, line = tube(.30, 3., rings=71)
    # The competing basal patch is deliberately oversampled; vertex population
    # alone would choose it, whereas full independent prior correspondence wins.
    stub = np.repeat(stub, 5, axis=0)
    points = np.vstack([stub, body])
    labels = np.r_[np.zeros(len(stub), int), np.ones(len(body), int)]
    prior = np.column_stack([np.linspace(0., 3., 151), np.zeros((151, 2))])
    root = RootPath("lateral", prior, mean_radius=.04)
    chosen, reason, evidence = _origin_supported_component(points, labels, prior, root, .01, return_evidence=True)
    assert chosen == 1
    assert reason == "whole_prior_supported_component"
    assert evidence[1]["matched_prior_length"] > 10 * evidence[0]["matched_prior_length"]


def test_ambiguous_parallel_bodies_do_not_certify_an_origin():
    a, _, line = tube(0., 2., y=-.05)
    b, _, _ = tube(0., 2., y=.05)
    line[:, 1] = 0.
    root = RootPath("ambiguous", line)
    _, reason = _origin_supported_component(np.vstack([a, b]),
        np.r_[np.zeros(len(a), int), np.ones(len(b), int)], line, root, .02)
    assert reason == "origin_component_ambiguous"


def test_true_source_gap_preserves_long_body_descendants_and_original_reference():
    primary_pts, primary_faces, primary = tube(-.6, 0., y=0., radius=.05, rings=31)
    stub, stub_faces, _ = tube(.04, .14, rings=11)
    body, body_faces, prior_body = tube(.32, 2.5, rings=101)
    child_pts, child_faces, child_line = tube(1.8, 2.25, y=.16, radius=.025, rings=31)
    points = np.vstack([primary_pts, stub, body, child_pts])
    faces = np.vstack([primary_faces, stub_faces + len(primary_pts),
                       body_faces + len(primary_pts) + len(stub),
                       child_faces + len(primary_pts) + len(stub) + len(body)])
    labels = np.r_[np.zeros(len(primary_pts), int), np.ones(len(stub) + len(body), int),
                   np.full(len(child_pts), 2, int)]
    prior = np.column_stack([np.linspace(0., 2.5, 126), np.zeros((126, 2))])
    parent = RootPath("body", prior, mean_radius=.04, insertion_point=primary[-1].copy())
    child = RootPath("child", child_line, parent_id="body", order=2,
                     insertion_point=np.array([1.8, 0., 0.]))
    before_points, before_labels = points.copy(), labels.copy()
    fitted, report = refit_final_centerlines(points, labels, primary, [parent, child],
        d_bar=.02, triangles=faces, primary_top_reference=np.array([0., 0., 1.]))
    assert path_length(parent.points[parent.body_start_index:]) > 2.
    assert parent.centerline_assessment["identity_qc_passed"] is False
    assert parent.centerline_assessment["measurement_evidence"]["component_measurement_only"]
    ids = parent.centerline_assessment["measurement_evidence"]["native_vertex_indices"]
    assert len(ids) == len(body)
    assert np.min(points[ids, 0]) >= .32
    assert child.parent_id == "body"
    np.testing.assert_array_equal(child.insertion_point, parent.points[child.insertion_index])
    assert child.insertion_point[2] <= 1.
    np.testing.assert_array_equal(points, before_points)
    np.testing.assert_array_equal(labels, before_labels)
    reference = copy.deepcopy(parent.centerline_assessment["prefit_identity_evidence"])
    assert reference["prior_path"] == prior.tolist()
    refit_final_centerlines(points, labels, fitted, [parent, child], d_bar=.02,
        triangles=faces, primary_top_reference=np.array([0., 0., 1.]))
    assert parent.centerline_assessment["prefit_identity_evidence"] == reference
    json.dumps(report, allow_nan=False)


def test_native_interval_evidence_rejects_an_empty_middle_even_with_endpoints():
    left, lf, _ = tube(0., .2, rings=11)
    right, rf, _ = tube(1.8, 2., rings=11)
    points = np.vstack([left, right])
    faces = np.vstack([lf, rf + len(left)])
    line = np.array([[0., 0., 0.], [.2, 0., 0.], [1.8, 0., 0.], [2., 0., 0.]])
    evidence = build_measurement_evidence(points, np.ones(len(points), int), 1, line,
        triangles=faces, spacing=.02, identity_resolved=False)
    assert evidence["supported_interval_mask"] == [True, False, True]
    assert not evidence["biological_tip_verified"]


def test_coarse_real_native_edges_support_intervals_without_fabricating_samples():
    points, faces, line = tube(0., 2., rings=9, sides=8, radius=.12)
    evidence = build_measurement_evidence(points, np.ones(len(points), int), 1, line,
        triangles=faces, spacing=.06)
    assert all(evidence["supported_interval_mask"])


@pytest.mark.parametrize("change", ["geometry", "ownership", "path", "topology", "body_start"])
def test_stale_support_cannot_be_refreshed_or_measured(change):
    points, faces, line = tube(0., 1.)
    labels = np.ones(len(points), int)
    evidence = build_measurement_evidence(points, labels, 1, line, triangles=faces, spacing=.02)
    altered_points, altered_faces, altered_line, altered_labels = points.copy(), faces.copy(), line.copy(), labels.copy()
    body_start = 0
    if change == "geometry": altered_points[0, 0] += .01
    if change == "ownership": altered_labels[0] = -1
    if change == "path": altered_line[0, 1] += .01
    if change == "topology": altered_faces = altered_faces[:-1]
    if change == "body_start": body_start = 1
    indices, reason = resolve_measurement_support(evidence, altered_points, altered_labels, 1,
        altered_line, triangles=altered_faces, body_start_index=body_start)
    assert indices is None and "stale" in reason


def test_coordinate_rebase_preserves_native_ids_and_rejects_edited_geometry():
    points, faces, line = tube(0., 1.)
    labels = np.ones(len(points), int)
    evidence = build_measurement_evidence(points, labels, 1, line, triangles=faces, spacing=.02)
    reference = {"prior_path": line.tolist(), "origin": line[0].tolist(), "mean_radius": .04,
                 "reference_valid": True}
    assessment = {"measurement_evidence": evidence, "prefit_identity_evidence": reference}
    scale, offset = 17., np.array([20., -5., 3.])
    converted = rebase_assessment(assessment, old_points=points, old_labels=labels, old_label=1,
        old_path=line, new_points=points * scale + offset, new_labels=labels * 7, new_label=7,
        new_path=line * scale + offset, scale=scale, offset=offset, triangles=faces)
    assert validate_measurement_evidence(converted["measurement_evidence"], points * scale + offset,
        labels * 7, 7, line * scale + offset, triangles=faces)[0]
    assert converted["measurement_evidence"]["native_vertex_indices"] == evidence["native_vertex_indices"]
    np.testing.assert_allclose(converted["prefit_identity_evidence"]["prior_path"], line * scale + offset)
    bad = points.copy(); bad[0, 0] += .1
    rejected = rebase_assessment(assessment, old_points=bad, old_labels=labels, old_label=1,
        old_path=line, new_points=bad * scale + offset, new_labels=labels, new_label=1,
        new_path=line * scale + offset, scale=scale, offset=offset, triangles=faces)
    assert rejected["measurement_evidence"]["status"] == "invalidated"
    assert not rejected["prefit_identity_evidence"]["reference_valid"]


def test_legacy_component_number_is_not_trusted_after_topology_changes():
    points, faces, _ = tube(0., 1.)
    ids, reason = infer_component_indices(points, np.ones(len(points), int), 1,
        faces, {"component_count": 2, "selected_component": 0}, context=make_evidence_context(points, faces))
    assert ids is None and reason == "legacy_component_topology_changed"


def test_valid_current_evidence_can_rebase_an_unavailable_legacy_identity_reference():
    points, faces, line = tube(0., 1.)
    labels = np.ones(len(points), int)
    assessment = {"measurement_evidence": build_measurement_evidence(points, labels, 1, line,
        triangles=faces, spacing=.02, identity_resolved=False),
        "prefit_identity_evidence": {"prior_path": [], "origin": [], "reference_valid": False}}
    rebased = rebase_assessment(assessment, old_points=points, old_labels=labels, old_label=1,
        old_path=line, new_points=points * 2 + 3, new_labels=labels, new_label=1,
        new_path=line * 2 + 3, scale=2., offset=np.full(3, 3.), triangles=faces)
    assert rebased["prefit_identity_evidence"]["prior_path"] == []
    assert not rebased["prefit_identity_evidence"]["reference_valid"]
    assert rebased["measurement_evidence"]["status"] == "current"


def test_native_continuation_into_another_owner_is_not_a_biological_tip():
    points, faces, line = tube(0., 2., rings=101)
    labels = np.where(points[:, 0] <= 1., 0, 1)
    _, report = refit_final_centerlines(points, labels, line[:51], [], d_bar=.02,
        triangles=faces, primary_top_reference=np.array([0., 0., 1.]))
    assert not report["roots"][0]["biological_tip_verified"]
    assert not report["roots"][0]["measurement_evidence"]["biological_tip_verified"]
    _, whole = refit_final_centerlines(points, np.zeros(len(points), int), line, [],
        d_bar=.02, triangles=faces, primary_top_reference=np.array([0., 0., 1.]))
    assert whole["roots"][0]["biological_tip_verified"]


def test_origin_coverage_distinguishes_axial_crop_from_radial_centering():
    prior = np.column_stack([np.linspace(0., 3., 151), np.zeros((151, 2))])
    cropped = prior.copy(); cropped[0, 0] = .1
    lost = assess_origin_coverage(cropped, prior[0], reference_path=prior, spacing=.02)
    assert lost["origin_coverage_complete"] is False
    assert lost["axial_loss"] == pytest.approx(.1)
    radial = prior.copy(); radial[0, 1] = .2
    centered = assess_origin_coverage(radial, prior[0], reference_path=prior, spacing=.02)
    assert centered["origin_coverage_complete"] is True
    assert centered["axial_loss"] == pytest.approx(0.)
    assert centered["radial_offset"] == pytest.approx(.2)
    assert not assess_origin_coverage(prior, prior[0], spacing=.02)["origin_coverage_complete"]


def test_primary_fit_cannot_certify_origin_lost_before_observed_tube():
    points, faces, _ = tube(.1, 3., radius=.2, rings=146)
    prior = np.column_stack([np.linspace(0., 3., 151), np.zeros((151, 2))])
    fitted, report = refit_final_centerlines(points, np.zeros(len(points), int), prior, [],
        d_bar=.02, triangles=faces, primary_top_reference=np.zeros(3))
    assessment = report["roots"][0]
    assert fitted[0, 0] == pytest.approx(.1)
    assert assessment["fit_qc_passed"] and assessment["biological_tip_verified"]
    assert assessment["origin_coverage_complete"] is False
    assert assessment["measurement_evidence"]["origin_coverage_complete"] is False
    np.testing.assert_array_equal(report["primary_top_point_normalized"], np.zeros(3))


def test_interval_policy_upgrade_uses_validated_support_and_refuses_stale_geometry():
    angle = np.arange(20) * 2 * np.pi / 20
    points = np.vstack([np.zeros((1, 3)), np.column_stack([.1*np.cos(angle), .1*np.sin(angle), np.zeros(20)])])
    faces = np.array([[0, i + 1, (i + 1) % 20 + 1] for i in range(20)])
    path = np.array([[0., 0., -.03], [0., 0., .03]])
    labels = np.zeros(len(points), int)
    legacy = build_measurement_evidence(points, labels, 0, path, triangles=faces, spacing=.03)
    legacy.pop("interval_support_version")
    legacy["supported_interval_mask"] = [True]
    upgraded, reason = upgrade_measurement_evidence(legacy, points, labels, 0, path, triangles=faces)
    assert reason == "evidence_interval_policy_upgraded"
    assert upgraded["interval_support_version"] == 2
    assert upgraded["supported_interval_mask"] == [False]
    assert legacy["supported_interval_mask"] == [True]
    changed = points.copy(); changed[0, 2] += .01
    rejected, reason = upgrade_measurement_evidence(legacy, changed, labels, 0, path, triangles=faces)
    assert rejected is None and reason == "stale_measurement_geometry_fingerprint"


@pytest.mark.parametrize("scale", [.01, 1., 200.])
def test_terminal_direction_rejects_collapsed_or_subsampling_segments_at_every_scale(scale):
    for terminal_length in (0., 1e-10):
        line = np.array([[0., 0., 0.], [1., 0., 0.], [1. + terminal_length, 0., 0.]]) * scale
        detail = assess_terminal_direction(line, spacing=.02 * scale)
        assert not detail["terminal_direction_supported"]
        assert detail["terminal_direction_reason"] == "degenerate_terminal_segment"
    real = assess_terminal_direction(np.array([[0., 0., 0.], [.01, 0., 0.]]) * scale,
                                     spacing=.02 * scale)
    assert real["terminal_direction_supported"]


def test_terminal_direction_rejects_nonfinite_coordinates():
    detail = assess_terminal_direction([[0., 0., 0.], [np.nan, 0., 0.]], spacing=.02)
    assert not detail["terminal_direction_supported"]
    assert detail["terminal_direction_reason"] == "nonfinite_terminal_segment"
    json.dumps(detail, allow_nan=False)


def test_duplicate_final_station_cannot_certify_tip_despite_positive_body_length():
    points, faces, line = tube(0., 1.)
    labels = np.zeros(len(points), int)
    context = make_evidence_context(points, faces)
    ids = np.arange(len(points))
    reference = {"prior_path": line.tolist(), "reference_valid": True}
    assert _verified_terminal_support(points, faces, ids, line, reference, .02, context)
    duplicated = np.vstack([line, line[-1]])
    assert not _verified_terminal_support(points, faces, ids, duplicated, reference, .02, context)
    evidence = build_measurement_evidence(points, labels, 0, duplicated, triangles=faces,
        spacing=.02, tip_supported=True, context=context)
    assert not evidence["biological_tip_verified"]
    assert evidence["terminal_direction_support_version"] == 1
    assert evidence["terminal_direction_reason"] == "degenerate_terminal_segment"
    assert sum(evidence["supported_interval_mask"]) > 0
    # A connector alone cannot supply a direction for a single-location body.
    connector_only = build_measurement_evidence(points, labels, 0, line[-2:], triangles=faces,
        spacing=.02, tip_supported=True, body_start_index=1, context=context)
    assert not connector_only["biological_tip_verified"]


@pytest.mark.parametrize("interval_version", [1, 2])
def test_terminal_upgrade_downgrades_collapsed_tip_without_changing_frozen_support(interval_version):
    points, faces, line = tube(0., 1.)
    labels = np.zeros(len(points), int)
    collapsed = np.repeat(line[-1:], 3, axis=0)
    legacy = build_measurement_evidence(points, labels, 0, collapsed, triangles=faces,
        spacing=.02, tip_supported=True)
    legacy.pop("terminal_direction_support_version")
    legacy["interval_support_version"] = interval_version
    legacy["biological_tip_verified"] = True
    before = copy.deepcopy(legacy)
    upgraded, reason = upgrade_measurement_evidence(legacy, points, labels, 0, collapsed, triangles=faces)
    assert upgraded["terminal_direction_support_version"] == 1
    assert upgraded["interval_support_version"] == 2
    assert not upgraded["biological_tip_verified"]
    assert upgraded["terminal_direction_reason"] == "degenerate_terminal_segment"
    for field in ("native_vertex_indices", "path_fingerprint", "ownership_fingerprint",
                  "support_fingerprint", "identity_resolved", "attachment_supported"):
        assert upgraded[field] == before[field]
    assert legacy == before
    assert "upgraded" in reason
    changed = collapsed.copy(); changed[0, 1] += .01
    rejected, reason = upgrade_measurement_evidence(legacy, points, labels, 0, changed, triangles=faces)
    assert rejected is None and reason == "stale_measurement_path"
