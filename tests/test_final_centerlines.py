import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from soyrootbio.centerline import _primary_fit_qc
from soyrootbio.centerline import _supported_connector
from soyrootbio.centerline import refit_final_centerlines
from soyrootbio.export import _lateral_skeleton_frame, write_rsml
from soyrootbio.geometry import path_length, tangent_vectors
from soyrootbio.primary import _plane_basis
from soyrootbio.traits import compute_traits
from soyrootbio.types import Normalization, RootPath


def tube(axis, radius=0.08, sides=24):
    angles = np.arange(sides) * 2 * np.pi / sides
    points = []
    for center, tangent in zip(axis, tangent_vectors(axis)):
        # Continuous frame: an arbitrary per-ring basis can flip and create
        # long, twisted mesh edges that falsely disconnect the synthetic tube.
        normal = np.array([0., 0., 1.])
        normal -= np.dot(normal, tangent) * tangent
        normal /= np.linalg.norm(normal)
        basis = np.vstack([normal, np.cross(tangent, normal)])
        points.extend(center + radius * (np.cos(angles)[:, None] * basis[0] + np.sin(angles)[:, None] * basis[1]))
    faces = []
    for i in range(len(axis) - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            faces.extend([[a, b, a + sides], [b, b + sides, a + sides]])
    return np.asarray(points), np.asarray(faces)


def irregular_tube(axis, angles, radius=0.08):
    angles = np.sort(np.mod(np.asarray(angles, dtype=float), 2 * np.pi))
    points = []
    for center, tangent in zip(axis, tangent_vectors(axis)):
        normal = np.array([0.0, 0.0, 1.0])
        normal -= np.dot(normal, tangent) * tangent
        normal /= np.linalg.norm(normal)
        basis = np.vstack([normal, np.cross(tangent, normal)])
        points.extend(
            center
            + radius
            * (
                np.cos(angles)[:, None] * basis[0]
                + np.sin(angles)[:, None] * basis[1]
            )
        )
    sides = len(angles)
    faces = []
    for i in range(len(axis) - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            faces.extend([[a, b, a + sides], [b, b + sides, a + sides]])
    return np.asarray(points), np.asarray(faces)


def straight(start=0, end=2, count=61):
    return np.column_stack([np.linspace(start, end, count), np.zeros((count, 2))])


def test_final_support_replaces_offset_body_and_overextended_tip():
    axis = straight()
    points, triangles = tube(axis)
    old = straight(0, 3) + [0, .07, 0]
    labels = np.zeros(len(points), dtype=int)
    before = labels.copy()
    fitted, report = refit_final_centerlines(points, labels, old, [], d_bar=.04, triangles=triangles)
    np.testing.assert_array_equal(labels, before)
    assert np.max(np.linalg.norm(fitted[:, 1:], axis=1)) < .015
    assert 1.94 < fitted[-1, 0] <= 2
    assert 1.85 < path_length(fitted) < 2.05
    assert report["roots"][0]["tip_displacement"] > .9
    json.dumps(report, allow_nan=False)


def test_curved_tube_follows_bend_and_adapts_sections():
    t = np.linspace(0, np.pi * .8, 100)
    axis = np.column_stack([.3 * np.cos(t), .3 * np.sin(t), np.zeros(len(t))])
    points, triangles = tube(axis, .035)
    fitted, report = refit_final_centerlines(points, np.zeros(len(points), int), axis + [0, 0, .025], [], d_bar=.025, triangles=triangles)
    distance = cKDTree(axis).query(fitted)[0]
    assert np.quantile(distance, .95) < .015
    assert path_length(fitted) > .65
    detail = report["roots"][0]
    assert detail["section_half_width_min"] < detail["section_half_width_max"]


def test_primary_fit_area_angular_balance_resists_dense_wall_sampling():
    x = np.linspace(0.0, 2.0, 81)
    axis = np.column_stack([x, 0.08 * np.sin(0.8 * x), np.zeros(len(x))])
    dense_side = np.linspace(-0.45, 0.45, 36, endpoint=False)
    sparse_remainder = np.linspace(0.45, 2 * np.pi - 0.45, 14, endpoint=False)
    points, triangles = irregular_tube(
        axis,
        np.r_[dense_side, sparse_remainder],
        radius=0.07,
    )
    prior = axis + [0.0, 0.025, 0.01]

    fitted, report = refit_final_centerlines(
        points,
        np.zeros(len(points), dtype=int),
        prior,
        [],
        d_bar=0.025,
        triangles=triangles,
    )

    detail = report["roots"][0]
    assert detail["fit_qc_passed"] is True
    assert detail["fit_applied"] is True
    assert detail["traits_geometry_source"] == "accepted_fit"
    assert np.quantile(cKDTree(axis).query(fitted)[0], 0.95) < 0.02
    assert path_length(fitted) / path_length(axis) < 1.08
    assert detail["turn_p95_degrees_radius_scale"] < 30.0


def test_primary_fit_rejects_one_sided_wall_and_retains_prior_for_traits():
    axis = straight(0.0, 2.0, 81)
    points, triangles = irregular_tube(
        axis,
        np.linspace(-0.45 * np.pi, 0.45 * np.pi, 24),
        radius=0.08,
    )

    fitted, report = refit_final_centerlines(
        points,
        np.zeros(len(points), dtype=int),
        axis,
        [],
        d_bar=0.03,
        triangles=triangles,
    )

    detail = report["roots"][0]
    np.testing.assert_allclose(fitted, axis)
    assert detail["fit_qc_passed"] is False
    assert detail["fit_applied"] is False
    assert detail["status"] == "qc_rejected_prior_retained"
    assert detail["traits_geometry_source"] == "retained_prior"
    assert detail["section_rejection_counts"]["one_sided_wall"] > 0
    assert "centerline_primary_low_angular_coverage" in report["primary_qc_flags"]


def test_primary_qc_detects_oscillation_backtracking_and_length_inflation():
    prior = straight(0.0, 2.0, 81)
    x = np.linspace(0.0, 2.0, 161)
    candidate = np.column_stack(
        [x, 0.12 * np.where(np.arange(len(x)) % 2, 1.0, -1.0), np.zeros(len(x))]
    )
    candidate[60:80, 0] = np.linspace(candidate[59, 0], candidate[59, 0] - 0.60, 20)
    support, _ = tube(prior, radius=0.08)
    sections = [
        {"accepted": True, "coverage": 0.9, "radius": 0.08}
        for _ in range(len(candidate))
    ]

    metrics, flags = _primary_fit_qc(candidate, prior, sections, support, 0.02)

    assert metrics["fit_qc_passed"] is False
    assert "centerline_primary_oscillation" in flags
    assert "centerline_primary_backtracking" in flags
    assert "centerline_primary_length_inflation" in flags


@pytest.mark.parametrize("mesh", [True, False])
def test_disconnected_fragment_is_not_bridged_or_used_for_tip(mesh):
    a, fa = tube(straight(0, 1, 40))
    b, fb = tube(straight(1.5, 2, 20))
    points = np.vstack([a, b])
    faces = np.vstack([fa, fb + len(a)]) if mesh else None
    fitted, report = refit_final_centerlines(points, np.zeros(len(points), int), straight(), [], d_bar=.03, triangles=faces)
    assert fitted[:, 0].max() <= 1
    assert report["roots"][0]["excluded_fragment_points"] == len(b)
    assert "centerline_disconnected_support" in report["primary_qc_flags"]


@pytest.mark.parametrize("count", [0, 3, 7])
def test_unmeasurable_support_has_no_fabricated_segments(count):
    points = straight(count=max(count, 1))[:count]
    fitted, report = refit_final_centerlines(points, np.zeros(count, int), straight(), [], d_bar=.1)
    assert len(fitted) == 1
    assert path_length(fitted) == 0
    assert report["roots"][0]["status"] in {"no_support", "insufficient_support"}


def test_child_connector_is_separate_and_traits_use_exposed_body():
    primary = straight(0, 3)
    parent_points, parent_faces = tube(primary, radius=.2)
    child_axis = np.column_stack([np.ones(50), np.linspace(.18, 1, 50), np.zeros(50)])
    child_points, child_faces = tube(child_axis, radius=.045)
    points = np.vstack([parent_points, child_points])
    labels = np.r_[np.zeros(len(parent_points), int), np.ones(len(child_points), int)]
    faces = np.vstack([parent_faces, child_faces + len(parent_points)])
    child = RootPath("child", np.vstack([[1, 0, 0], child_axis]), insertion_point=np.array([1, 0, 0]))
    fitted, _ = refit_final_centerlines(points, labels, primary, [child], d_bar=.04, triangles=faces)
    assert child.body_start_index == 1
    np.testing.assert_array_equal(child.points[0], fitted[child.insertion_index])
    np.testing.assert_array_equal(child.parent_points, fitted)
    traits = compute_traits(fitted, [child], points[:1], np.array([True]), np.array([0]), Normalization(np.zeros(3), 1),
                            full_points=points, full_root_labels=labels, triangles=faces)
    row = traits.set_index("root_id").loc["child"]
    assert row.point_count == len(child_points)
    assert row.parent_connector_length > .1
    assert row.exposed_body_length == pytest.approx(path_length(child.points[1:]))
    assert row.length == pytest.approx(row.parent_connector_length + row.exposed_body_length)
    assert row.base_vector_start_y == pytest.approx(child.points[1, 1])


def test_distant_child_does_not_get_a_free_space_connector():
    parent = straight(0, 3)
    a, fa = tube(parent)
    child_axis = straight(0, 1) + [1, 1, 0]
    b, fb = tube(child_axis)
    child = RootPath("child", np.vstack([[1, 0, 0], child_axis]), insertion_point=np.array([1, 0, 0]))
    refit_final_centerlines(np.vstack([a, b]), np.r_[np.zeros(len(a), int), np.ones(len(b), int)], parent,
                           [child], d_bar=.04, triangles=np.vstack([fa, fb + len(a)]))
    assert child.body_start_index == 0
    assert child.points[:, 1].min() > .9
    assert child.centerline_assessment["parent_connector_supported"] is False


def test_parent_owned_protrusion_supports_connector_but_missing_interval_does_not():
    axis = np.column_stack([np.ones(71), np.linspace(0, .7, 71), np.zeros(71)])
    points, _ = tube(axis, radius=.04)
    parent = points[points[:, 1] < .55]
    child = points[points[:, 1] >= .55]
    candidate = axis[[0, -1]]
    assert _supported_connector(candidate, parent, child, .015)
    parent_with_gap = parent[(parent[:, 1] < .2) | (parent[:, 1] > .4)]
    assert not _supported_connector(candidate, parent_with_gap, child, .015)


def test_fitting_handles_children_before_parents_without_relabeling():
    parent = straight(0, 3)
    a, fa = tube(parent)
    b_axis = straight(0, 1) + [1, .5, 0]
    b, fb = tube(b_axis)
    c_axis = straight(0, .5) + [1, 1, 0]
    c, fc = tube(c_axis)
    child = RootPath("child", c_axis, parent_id="lateral", order=2)
    lateral = RootPath("lateral", b_axis)
    roots = [child, lateral]
    labels = np.r_[np.zeros(len(a), int), np.full(len(b), 2), np.ones(len(c), int)]
    _, report = refit_final_centerlines(np.vstack([a, b, c]), labels, parent, roots, d_bar=.04,
                                      triangles=np.vstack([fa, fb + len(a), fc + len(a) + len(b)]))
    assert [r.root_id for r in roots] == ["child", "lateral"]
    assert [r["numeric_label"] for r in report["roots"]] == [0, 2, 1]
    np.testing.assert_array_equal(child.parent_points, lateral.points)


def test_final_fitting_rejects_lateral_origin_above_primary_top():
    primary = straight(0, 1)
    child = RootPath(
        "child",
        np.array([[0.5, 0.0, 0.1], [0.7, 0.0, 0.1]]),
        insertion_point=np.array([0.5, 0.0, 0.1]),
        insertion_index=30,
    )
    points = np.vstack([primary, child.points])
    labels = np.r_[np.zeros(len(primary), int), np.ones(len(child.points), int)]

    with pytest.raises(ValueError, match="above the primary-root top"):
        refit_final_centerlines(
            points,
            labels,
            primary,
            [child],
            d_bar=0.04,
        )


def test_immutable_selected_top_survives_shortened_primary_fit_with_o2_child():
    primary = np.column_stack(
        [
            np.linspace(0.0, 0.10, 81),
            np.zeros(81),
            np.linspace(1.0, 0.0, 81),
        ]
    )
    primary_support_axis = np.column_stack(
        [
            np.linspace(0.07, 0.10, 41),
            np.zeros(41),
            np.linspace(0.30, 0.0, 41),
        ]
    )
    primary_support, primary_faces = tube(primary_support_axis, radius=0.04)

    o1_axis = np.column_stack(
        [
            np.linspace(0.15, 0.75, 61),
            np.zeros(61),
            np.full(61, 0.75),
        ]
    )
    o1_support, o1_faces = tube(o1_axis, radius=0.03)
    o1 = RootPath(
        "lateral",
        np.vstack([[0.08, 0.0, 0.20], o1_axis]),
        parent_id="primary",
        order=1,
        insertion_point=np.array([0.08, 0.0, 0.20]),
        insertion_index=0,
    )

    o2_axis = np.column_stack(
        [
            np.full(41, 0.45),
            np.linspace(0.06, 0.40, 41),
            np.full(41, 0.75),
        ]
    )
    o2_support, o2_faces = tube(o2_axis, radius=0.02)
    o2 = RootPath(
        "child",
        np.vstack([[0.45, 0.0, 0.75], o2_axis]),
        parent_id="lateral",
        order=2,
        insertion_point=np.array([0.45, 0.0, 0.75]),
        insertion_index=0,
    )

    points = np.vstack([primary_support, o1_support, o2_support])
    labels = np.concatenate(
        [
            np.zeros(len(primary_support), dtype=int),
            np.ones(len(o1_support), dtype=int),
            np.full(len(o2_support), 2, dtype=int),
        ]
    )
    triangles = np.vstack(
        [
            primary_faces,
            o1_faces + len(primary_support),
            o2_faces + len(primary_support) + len(o1_support),
        ]
    )

    fitted_primary, report = refit_final_centerlines(
        points,
        labels,
        primary,
        [o1, o2],
        d_bar=0.03,
        triangles=triangles,
        primary_top_reference=np.array([0.0, 0.0, 1.0]),
    )

    assert np.max(fitted_primary[:, 2]) < 0.40
    assert o1.body_start_index == 0
    assert o1.centerline_assessment["parent_connector_supported"] is False
    assert o2.insertion_point[2] > np.max(fitted_primary[:, 2])
    assert o2.insertion_point[2] <= report["primary_top_point_normalized"][2]
    np.testing.assert_allclose(
        report["primary_top_point_normalized"],
        [0.0, 0.0, 1.0],
    )
    np.testing.assert_allclose(
        report["fitted_primary_top_point_normalized"],
        fitted_primary[np.argmax(fitted_primary[:, 2])],
    )


@pytest.mark.parametrize("join_prior_to_body", [True, False])
@pytest.mark.parametrize("top_z", [0.5, 1.0])
def test_parent_refit_preserves_below_top_o2_attachment(join_prior_to_body, top_z, tmp_path):
    primary = np.column_stack([
        np.linspace(0.0, 1.0, 51),
        np.zeros(51),
        np.linspace(0.5, 0.1, 51),
    ])
    primary_support, primary_faces = tube(primary, radius=0.035)
    body_start = 0.8 if join_prior_to_body else 1.3
    o1_axis = np.column_stack([
        np.linspace(body_start, body_start + 0.7, 71),
        np.full(71, 0.1),
        np.full(71, 0.7),
    ])
    o1_support, o1_faces = tube(o1_axis, radius=0.025)
    o1_prior = np.array([
        [0.25, 0.0, 0.4],
        [0.56, 0.03, 0.45],
        [0.65, 0.08, 0.55],
        [0.74, 0.1, 0.64],
        [0.8, 0.1, 0.7],
    ])
    if join_prior_to_body:
        o1_prior = np.vstack([o1_prior, o1_axis[1::10]])
    o1 = RootPath(
        "o1", o1_prior, order=1,
        insertion_point=o1_prior[0].copy(), insertion_index=25,
    )
    o2_axis = np.column_stack([
        np.full(51, 0.56),
        np.linspace(0.06, 0.46, 51),
        np.full(51, 0.45),
    ])
    o2_support, o2_faces = tube(o2_axis, radius=0.02)
    o2 = RootPath(
        "o2", np.vstack([o1_prior[1], o2_axis]),
        parent_id="o1", order=2,
        insertion_point=o1_prior[1].copy(), insertion_index=1,
    )
    points = np.vstack([primary_support, o1_support, o2_support])
    labels = np.r_[
        np.zeros(len(primary_support), dtype=int),
        np.ones(len(o1_support), dtype=int),
        np.full(len(o2_support), 2, dtype=int),
    ]
    triangles = np.vstack([
        primary_faces,
        o1_faces + len(primary_support),
        o2_faces + len(primary_support) + len(o1_support),
    ])
    original_labels = labels.copy()

    fitted_primary, report = refit_final_centerlines(
        points, labels, primary, [o1, o2], d_bar=0.04,
        triangles=triangles, primary_top_reference=np.array([0.0, 0.0, top_z]),
    )

    np.testing.assert_array_equal(labels, original_labels)
    np.testing.assert_allclose(o2.insertion_point, o1.points[o2.insertion_index])
    np.testing.assert_allclose(o2.parent_points, o1.points)
    assert o2.insertion_point[2] <= primary[0, 2]
    assert report["origins_above_primary_top"] == 0
    assert fitted_primary.shape[1] == 3
    if join_prior_to_body:
        assert o1.centerline_assessment["parent_connector_mode"] == "preserved_topology"
        assert o1.body_start_index > o2.insertion_index
        assert o1.centerline_assessment["preserved_topology_required_by"] == ["o2"]
        assert o1.centerline_assessment["preserved_topology_length"] > 0
        assert o1.centerline_assessment["exposed_body_length"] > 0
        traits_frame = compute_traits(
            fitted_primary, [o1, o2], points[:1], np.array([True]),
            np.array([0]), Normalization(np.zeros(3), 1),
            full_points=points, full_root_labels=labels, triangles=triangles,
        )
        summary = traits_frame.attrs["system_summary"]
        assert summary["root_system_length_complete"] is False
        assert summary["root_system_length_unavailable_root_count"] >= 1
        traits = traits_frame.set_index("root_id")
        assert np.isnan(traits.loc["o1", "length"])
        assert traits.loc["o1", "preserved_topology_length"] > 0
        assert traits.loc["o1", "exposed_body_length"] > 0
        assert traits.loc["o1", "topology_path_length"] > traits.loc["o1", "exposed_body_length"]
        skeleton = _lateral_skeleton_frame([o1, o2], Normalization(np.zeros(3), 1))
        assert "preserved_topology" in set(skeleton.loc[skeleton.root_id == "o1", "centerline_region"])
        rsml_path = write_rsml(
            tmp_path / "roots.rsml", fitted_primary, [o1, o2],
            traits.reset_index(), {"source": "attachment-test"},
        )
        assert "preserved_topology_length" in rsml_path.read_text(encoding="utf-8")
    else:
        assert o1.centerline_assessment["parent_connector_mode"] == "retained_prior"
        assert o1.centerline_assessment["fit_applied"] is False
        assert o1.centerline_assessment["exposed_body_supported"] is False
        np.testing.assert_allclose(o1.points, o1_prior)
        traits_frame = compute_traits(
            fitted_primary, [o1, o2], points[:1], np.array([True]),
            np.array([0]), Normalization(np.zeros(3), 1),
            full_points=points, full_root_labels=labels, triangles=triangles,
        )
        summary = traits_frame.attrs["system_summary"]
        assert summary["root_system_length_complete"] is False
        assert summary["root_system_volume_estimate_complete"] is False
        assert summary["root_system_volume_estimate_unavailable_root_count"] >= 1
        traits = traits_frame.set_index("root_id")
        assert np.isnan(traits.loc["o1", "length"])
        assert np.isnan(traits.loc["o1", "exposed_body_length"])
        assert np.isnan(traits.loc["o1", "volume"])
        assert traits.loc["o1", "topology_path_length"] > 0
        skeleton = _lateral_skeleton_frame([o1, o2], Normalization(np.zeros(3), 1))
        assert set(skeleton.loc[skeleton.root_id == "o1", "centerline_region"]) == {"retained_prior"}


def test_unresolvable_attachment_does_not_partially_mutate_roots():
    primary = straight(0, 1, 31) + [0.0, 0.0, 0.5]
    parent = RootPath(
        "o1", np.array([[0.4, 0.0, 0.7], [0.8, 0.0, 0.7]]),
        insertion_point=np.array([0.4, 0.0, 0.4]), insertion_index=12,
    )
    child = RootPath(
        "o2", np.array([[0.5, 0.0, 0.45], [0.5, 0.3, 0.45]]),
        parent_id="o1", order=2,
        insertion_point=np.array([0.5, 0.0, 0.45]), insertion_index=0,
    )
    original = [(root.points.copy(), root.insertion_point.copy(), root.insertion_index) for root in (parent, child)]
    with pytest.raises(ValueError, match="neither fitted nor repaired path"):
        refit_final_centerlines(
            primary.copy(), np.zeros(len(primary), dtype=int), primary,
            [parent, child], d_bar=0.04,
            primary_top_reference=np.array([0.0, 0.0, 0.5]),
        )
    for root, (points, insertion_point, insertion_index) in zip((parent, child), original):
        np.testing.assert_array_equal(root.points, points)
        np.testing.assert_array_equal(root.insertion_point, insertion_point)
        assert root.insertion_index == insertion_index
        assert root.centerline_assessment == {}
