"""Lateral fitting must improve geometry without destroying supported bodies."""
import json

import numpy as np
import pytest

from soyrootbio import centerline
from soyrootbio.geometry import path_length


def _straight_tube():
    axis = np.column_stack([np.linspace(0, 2, 151), np.zeros((151, 2))])
    angle = np.linspace(0, 2*np.pi, 32, endpoint=False)
    offsets = np.column_stack([np.zeros(32), .08*np.cos(angle), .08*np.sin(angle)])
    return axis, (axis[:, None, :]+offsets).reshape(-1, 3)


def test_valid_prior_wins_over_a_backtracking_mesh_axis():
    axis, points = _straight_tube()
    pilot = axis.copy()
    pilot[:, 2] += .025*np.sin(np.arange(len(axis))*np.pi*.72)
    line, detail = centerline._fit_lateral_body(points, pilot, axis, .012)
    assert detail["fit_qc_passed"]
    assert detail["selected_candidate"] == "prior"
    assert detail["traits_geometry_source"] == "validated_prior"
    np.testing.assert_allclose(line, axis, atol=1e-12)
    rejected_axis = next(x for x in detail["lateral_candidate_assessments"] if x["name"] == "starting_axis")
    assert "sampling_scale_oscillation" in rejected_axis["rejections"]
    json.dumps(detail, allow_nan=False)


def test_regularized_candidate_removes_jitter_when_both_seeds_are_noisy():
    axis, points = _straight_tube()
    jitter = axis.copy()
    jitter[:, 2] += .025*np.sin(np.arange(len(axis))*np.pi*.72)
    line, detail = centerline._fit_lateral_body(points, jitter, jitter.copy(), .012)
    assert detail["fit_qc_passed"]
    assert detail["selected_candidate"].startswith("regularized_")
    assert path_length(jitter) > 2*path_length(axis)
    assert path_length(line) == pytest.approx(path_length(axis), rel=.02)
    assert np.quantile(np.abs(line[:, 2]), .95) < .005
    assert line[0, 0] < .025 and line[-1, 0] > 1.975


def test_local_envelope_preserves_a_flattened_root_full_extent():
    axis, points = _straight_tube()
    points[:, 1] *= 3
    line, detail = centerline._fit_lateral_body(points, axis, axis.copy(), .012)
    assert detail["fit_qc_passed"]
    assert path_length(line) > 1.95
    assert np.max(np.linalg.norm(line[:, 1:], axis=1)) < .01


def test_centering_uses_measured_half_width_instead_of_nearest_wall_distance():
    axis, points = _straight_tube()
    prior = axis + [0., 0., .025]
    assessment = centerline._lateral_candidate_metrics(
        points, prior, .012, np.ones(len(points)), centerline.cKDTree(points))
    assert assessment["valid"]
    assert .29 < assessment["centering_error_radius_fraction"] < .34


def test_centering_improvement_policy_controls_replacement(monkeypatch):
    axis, points = _straight_tube()
    prior = axis + [0., 0., .015]
    _, detail = centerline._fit_lateral_body(points, axis, prior, .012)
    assert detail["selected_candidate"] != "prior"
    monkeypatch.setitem(centerline.POLICY, "lateral_centering_improvement_margin", .3)
    line, detail = centerline._fit_lateral_body(points, axis, prior, .012)
    assert detail["selected_candidate"] == "prior"
    np.testing.assert_allclose(line, prior, atol=1e-12)


def test_legacy_section_width_policy_is_effective(monkeypatch):
    axis, points = _straight_tube()
    tree = centerline.cKDTree(points)
    tangents = np.tile([1., 0., 0.], (len(axis), 1))
    arc = axis[:, 0]
    assigned = centerline.cKDTree(axis).query(points)[1]
    widths = []
    for width in (1., 3.):
        monkeypatch.setitem(centerline.POLICY, "section_half_width_spacing", width)
        _, half_width, supported = centerline._section(points, tree, axis, tangents, arc, assigned, 75, .01)
        assert supported
        widths.append(half_width)
    assert widths == pytest.approx([.01, .03])


def test_one_endpoint_crop_preserves_the_supported_interior():
    axis, points = _straight_tube()
    overextended = axis.copy()
    overextended[:, 0] = np.linspace(-.03, 2.03, len(axis))
    line, detail = centerline._fit_lateral_body(points, overextended, overextended.copy(), .012)
    assert detail["fit_qc_passed"]
    assert 1.9 < path_length(line) <= 2.01
    assert line[0, 0] >= -.006 and line[-1, 0] <= 2.006
    selected = next(x for x in detail["lateral_candidate_assessments"] if x["name"] == detail["selected_candidate"])
    assert selected["endpoint_trim_applied"]
    assert "extrapolated_endpoint" in selected["pretrim_rejections"]


def test_endpoint_crop_keeps_the_original_window_when_search_radius_tapers():
    # Approaching a tapered shaft decreases nearest-surface search radius.
    # Recomputing the crop budget at each station would stop before the first
    # supported ring, despite that ring lying inside the original local window.
    x = np.linspace(.15, 2., 201)
    radius = .025+.04*(x-.15)/1.85
    angle = np.linspace(0, 2*np.pi, 32, endpoint=False)
    points = np.stack([np.broadcast_to(x[:, None], (len(x), len(angle))),
                       radius[:, None]*np.cos(angle), radius[:, None]*np.sin(angle)], axis=-1).reshape(-1, 3)
    prior = np.column_stack([np.linspace(0, 2., 201), np.zeros((201, 2))])
    line, detail = centerline._assess_lateral_candidate(
        points, prior, .005, np.ones(len(points)), centerline.cKDTree(points))
    assert detail["valid"]
    assert detail["endpoint_trim_applied"]
    assert .145 <= line[0, 0] <= .19
    assert path_length(line) > 1.8
    assert detail["endpoint_trim_removed_length_normalized"] < detail["endpoint_trim_start_limit_normalized"]


def test_family_attachment_can_choose_an_independently_validated_prior():
    from test_final_centerlines import tube, straight
    from soyrootbio.types import RootPath

    primary = straight(0, 3) + [0., 0., .1]
    parent_axis = np.column_stack([np.ones(101), np.linspace(.18, 2, 101), np.zeros(101)])
    parent_prior = parent_axis + [0., 0., -.02]
    branch_index = 50
    hint = parent_prior[branch_index]
    child_axis = np.column_stack([np.linspace(1.1, 1.6, 41),
                                  np.full(41, hint[1]), np.full(41, hint[2])])
    clouds, faces, offset = [], [], 0
    for axis, radius in ((primary,.12),(parent_axis,.08),(child_axis,.025)):
        cloud, face = tube(axis, radius=radius)
        clouds.append(cloud); faces.append(face+offset); offset += len(cloud)
    parent = RootPath("parent", parent_prior, insertion_point=primary[20].copy())
    child = RootPath("child", child_axis, parent_id="parent", order=2,
                     insertion_point=hint.copy(), insertion_index=branch_index)
    labels = np.concatenate([np.full(len(cloud), i) for i, cloud in enumerate(clouds)])
    centerline.refit_final_centerlines(np.vstack(clouds), labels, primary, [parent,child],
        d_bar=.004, triangles=np.vstack(faces), primary_top_reference=primary[0])
    assessment = parent.centerline_assessment
    assert assessment["fit_applied"]
    assert assessment["fit_qc_passed"]
    assert assessment["selected_candidate"] == "prior"
    assert assessment["attachment_candidate_fallback_from"] != "prior"
    assert assessment["traits_geometry_source"] == "validated_prior"
    assert assessment["status"] != "attachment_preserving_prior_retained"
