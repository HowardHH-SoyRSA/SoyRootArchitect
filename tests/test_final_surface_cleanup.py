import json

import numpy as np
import pytest

from soyrootbio.final_surface_cleanup import cleanup_final_surface
from soyrootbio.types import RootPath


def strips(centers=(0.0, 3.0, 6.0), lengths=(10.0, 10.0, 10.0)):
    points, faces, labels, paths, stations = [], [], [], [], []
    for label, (x, length) in enumerate(zip(centers, lengths)):
        z = np.arange(0.0, length + 0.001, 0.25)
        offset = len(points)
        stations.append({float(value): offset + 3 * i + 1 for i, value in enumerate(z)})
        points.extend([[x + 0.2, y, value] for value in z for y in [-0.12, 0.0, 0.12]])
        labels.extend([label] * (len(z) * 3))
        for i in range(len(z) - 1):
            for j in range(2):
                a = offset + 3 * i + j
                faces.extend([[a, a + 1, a + 3], [a + 1, a + 4, a + 3]])
        paths.append(np.array([[x, 0.0, 0.0], [x, 0.0, length]]))
    roots = [
        RootPath(root_id="root-1", parent_id="primary", order=1, points=paths[1]),
        RootPath(root_id="root-2", parent_id="root-1", order=2, points=paths[2]),
    ]
    return (
        np.asarray(points, dtype=float),
        np.asarray(labels, dtype=int),
        paths[0],
        roots,
        np.asarray(faces, dtype=int),
        stations,
    )


def run(case, *, excluded_mask=None):
    points, labels, primary, roots, faces, _ = case
    result, report = cleanup_final_surface(
        points,
        labels,
        primary,
        roots,
        d_bar=0.15,
        triangles=faces,
        excluded_mask=excluded_mask,
    )
    json.dumps(report, allow_nan=False)
    return result, report


@pytest.mark.parametrize("order", [1, 2, 5])
def test_lateral_island_on_parent_uses_same_quality_rule_at_every_order(order):
    case = list(strips())
    case[3][0].order = order
    island = case[5][0][5.0]
    case[1][island] = 1
    after, report = run(case)
    assert after[island] == 0
    move = next(row for row in report["moves"] if row["first_vertex"] == island)
    assert move["source_label"] == 1 and move["target_label"] == 0
    assert move["boundary_fraction"] == pytest.approx(1.0)


def test_higher_order_speck_returns_to_its_actual_parent():
    case = list(strips())
    island = case[5][1][5.0]
    case[1][island] = 2
    after, report = run(case)
    assert after[island] == 1
    move = next(row for row in report["moves"] if row["first_vertex"] == island)
    assert move["target_label"] == 1


def test_primary_island_has_no_main_component_exemption():
    case = list(strips())
    island = case[5][1][5.0]
    case[1][island] = 0
    after, _ = run(case)
    assert after[island] == 1


def test_local_parent_anchor_survives_an_unsupported_remote_parent_vertex():
    case = list(strips())
    remote = case[5][0][1.0]
    case[0][remote, 0] += 0.50
    island = case[5][0][5.0]
    case[1][island] = 1
    after, report = run(case)
    assert after[island] == 0
    assert report["reassigned_island_vertex_count"] == 1


def test_hole_uses_touching_lateral_not_total_primary_size():
    case = list(strips(lengths=(20.0, 10.0, 10.0)))
    hole = np.array([case[5][1][4.75], case[5][1][5.0], case[5][1][5.25]])
    case[1][hole] = -1
    after, report = run(case)
    np.testing.assert_array_equal(after[hole], np.ones(len(hole), dtype=int))
    assert report["filled_hole_count"] == 1
    assert report["filled_hole_vertex_count"] == 3


def test_repeated_cleanup_is_stable_with_the_same_frozen_paths():
    case = list(strips(lengths=(20.0, 10.0, 10.0)))
    island = case[5][0][5.0]
    hole = np.array([case[5][1][4.75], case[5][1][5.0], case[5][1][5.25]])
    case[1][island] = 1
    case[1][hole] = -1
    first, _ = run(case)
    repeated = list(case)
    repeated[1] = first
    second, report = run(repeated)
    np.testing.assert_array_equal(second, first)
    assert report["changed_vertex_count"] == 0


def test_equal_boundary_and_surface_claims_leave_hole_unassigned():
    case = list(strips(centers=(0.0, 0.4, 6.0)))
    points, labels, _, _, faces, stations = case
    hole = len(points)
    case[0] = np.vstack([points, [[0.2, 0.0, 5.0]]])
    case[1] = np.r_[labels, -1]
    primary = stations[0][5.0]
    child = stations[1][5.0]
    case[4] = np.vstack(
        [faces, [[hole, primary - 1, primary], [hole, primary, primary + 1],
                 [hole, child - 1, child], [hole, child, child + 1]]]
    )
    after, report = run(case)
    assert after[hole] == -1
    row = next(row for row in report["components"] if row["first_vertex"] == hole)
    assert row["decision"] == "retained_unassigned_no_clear_supported_recipient"


def test_boundary_and_surface_winners_must_agree_before_filling_a_hole():
    case = list(strips(centers=(0.0, 0.4, 6.0)))
    points, labels, _, _, faces, stations = case
    hole = len(points)
    case[0] = np.vstack([points, [[0.25, 0.0, 5.0]]])
    case[1] = np.r_[labels, -1]
    primary = stations[0][5.0]
    child = stations[1][5.0]
    case[4] = np.vstack(
        [
            faces,
            [
                [hole, primary - 1, primary],
                [hole, primary, primary + 1],
                [hole, primary + 2, primary + 3],
                [hole, child - 1, child],
            ],
        ]
    )
    after, _ = run(case)
    assert after[hole] == -1


def test_weak_parent_contact_does_not_force_a_lateral_island_to_parent():
    case = list(strips(centers=(0.0, 0.4, 6.0)))
    points, labels, _, _, faces, stations = case
    island = len(points)
    case[0] = np.vstack([points, [[0.2, 0.0, 5.0]]])
    case[1] = np.r_[labels, 2]
    primary = stations[0][5.0]
    child = stations[1][5.0]
    case[4] = np.vstack(
        [
            faces,
            [
                [island, primary - 1, primary],
                [island, child - 1, child],
                [island, child, child + 1],
            ],
        ]
    )
    after, report = run(case)
    assert after[island] == -1
    row = next(row for row in report["components"] if row["first_vertex"] == island)
    assert row["decision"] == "unsupported_compact_island_left_unassigned"


def test_supported_detached_body_is_preserved():
    case = list(strips())
    points, labels, _, _, faces, _ = case
    z = points[faces, 2]
    child = np.all(labels[faces] == 1, axis=1)
    cut = child & (z.min(axis=1) < 5.0) & (z.max(axis=1) >= 5.0)
    case[4] = faces[~cut]
    after, report = run(case)
    np.testing.assert_array_equal(after, labels)
    child_rows = [row for row in report["components"] if row["source_label"] == 1]
    assert len(child_rows) == 2
    assert all(row["source_quality"]["coherent_exposed_body"] for row in child_rows)


def test_unsupported_compact_island_without_a_recipient_becomes_unassigned():
    case = list(strips())
    isolated = len(case[0])
    case[0] = np.vstack([case[0], [[20.0, 0.0, 5.0]]])
    case[1] = np.r_[case[1], 1]
    # A self-contained tiny triangle has no contact with another label.
    case[0] = np.vstack([case[0], [[20.05, 0.0, 5.0], [20.0, 0.05, 5.0]]])
    case[1] = np.r_[case[1], 1, 1]
    case[4] = np.vstack([case[4], [[isolated, isolated + 1, isolated + 2]]])
    after, report = run(case)
    np.testing.assert_array_equal(after[isolated:isolated + 3], -np.ones(3, dtype=int))
    assert report["unassigned_island_count"] == 1
    assert report["unassigned_island_vertex_count"] == 3
    assert report["filled_hole_vertex_count"] == 0


def test_protected_and_uncertain_vertices_never_change():
    case = list(strips())
    protected = case[5][0][4.0]
    uncertain = case[5][1][4.0]
    case[1][protected] = -1
    case[1][uncertain] = -2
    excluded = np.zeros(len(case[0]), dtype=bool)
    excluded[protected] = True
    after, report = run(case, excluded_mask=excluded)
    assert after[protected] == -1
    assert after[uncertain] == -2
    assert report["protected_vertex_count"] == 1


def test_missing_mesh_skips_without_spatial_bridging():
    case = list(strips())
    case[4] = None
    before = case[1].copy()
    after, report = run(case)
    np.testing.assert_array_equal(after, before)
    assert report["status"] == "skipped_no_mesh"


@pytest.mark.parametrize("spacing", [0.0, -1.0, float("nan")])
def test_invalid_spacing_is_rejected(spacing):
    case = strips()
    with pytest.raises(ValueError, match="d_bar"):
        cleanup_final_surface(
            case[0], case[1], case[2], case[3], d_bar=spacing, triangles=case[4]
        )
