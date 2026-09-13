import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from soyrootbio.centerline import refit_final_centerlines
from soyrootbio.centerline import _supported_connector
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
