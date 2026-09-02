from __future__ import annotations

import numpy as np

from soyrootbio.pipeline import (
    _absorb_small_primary_surface_patches,
    _assign_full_root_labels,
    _assign_lateral_points,
    _group_indices_by_label,
    _resolve_parent_owned_junctions,
)
from soyrootbio.types import RootPath


def _root(
    root_id: str,
    points: list[list[float]],
    *,
    parent_id: str,
    order: int,
    insertion_index: int,
) -> RootPath:
    polyline = np.asarray(points, dtype=float)
    return RootPath(
        root_id=root_id,
        points=polyline,
        parent_id=parent_id,
        order=order,
        insertion_point=polyline[0].copy(),
        insertion_index=insertion_index,
    )


def test_primary_owns_uncertain_vertices_at_a_child_insertion() -> None:
    primary = np.array(
        [
            [0.0, 0.0, -0.2],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.2],
        ]
    )
    child = _root(
        "root-o1-001",
        [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.10, 0.0, 0.0]],
        parent_id="primary",
        order=1,
        insertion_index=1,
    )
    points = np.array(
        [
            [0.0, 0.01, 0.0],
            [0.08, 0.01, 0.0],
            [0.0, 0.0, 0.25],
        ]
    )
    raw, competing_labels = _assign_full_root_labels(
        points,
        primary,
        [child],
        d_bar=0.01,
        return_competing_labels=True,
    )
    assert raw[0] == -2
    assert set(competing_labels[0]) == {0, 1}

    resolved, report = _resolve_parent_owned_junctions(
        points,
        raw,
        primary,
        [child],
        d_bar=0.01,
        assignment_radius=0.06,
        ambiguity_margin=0.01,
        competing_labels=competing_labels,
    )

    assert resolved[0] == 0
    assert resolved[1] != 0
    assert report["resolved_vertex_count"] == 1
    assert report["per_parent_vertex_count"] == {"primary": 1}


def test_sampled_assignment_returns_the_actual_competing_lateral_labels() -> None:
    left = _root(
        "root-o1-001",
        [[0.0, 0.0, 0.0], [0.05, 0.03, 0.0]],
        parent_id="primary",
        order=1,
        insertion_index=0,
    )
    right = _root(
        "root-o1-002",
        [[0.0, 0.0, 0.0], [0.05, -0.03, 0.0]],
        parent_id="primary",
        order=1,
        insertion_index=0,
    )

    labels, competing_labels = _assign_lateral_points(
        np.array([[0.0, 0.005, 0.0]]),
        [left, right],
        np.array([False]),
        d_bar=0.01,
        return_competing_labels=True,
    )

    assert labels[0] == -1
    assert set(competing_labels[0]) == {1, 2}


def test_order_two_junction_is_owned_by_its_direct_order_one_parent() -> None:
    primary = np.array([[0.0, 0.0, -0.2], [0.0, 0.0, 0.2]])
    order_one = _root(
        "root-o1-001",
        [[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [0.20, 0.0, 0.0]],
        parent_id="primary",
        order=1,
        insertion_index=0,
    )
    order_two = _root(
        "root-o2-001",
        [[0.10, 0.0, 0.0], [0.10, 0.05, 0.0], [0.10, 0.10, 0.0]],
        parent_id="root-o1-001",
        order=2,
        insertion_index=1,
    )
    points = np.array(
        [
            [0.10, 0.0, 0.01],
            [0.12, 0.0, 0.01],
            [0.16, 0.0, 0.01],
        ]
    )
    labels = np.array([-2, 1, 1])

    resolved, report = _resolve_parent_owned_junctions(
        points,
        labels,
        primary,
        [order_one, order_two],
        d_bar=0.01,
        assignment_radius=0.04,
        ambiguity_margin=0.01,
        competing_labels={0: (1, 2)},
    )

    assert resolved[0] == 1
    assert report["per_parent_vertex_count"] == {"root-o1-001": 1}


def test_sibling_junction_uses_common_parent_but_unassigned_is_untouched() -> None:
    primary = np.array(
        [[0.0, 0.0, -0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]
    )
    left = _root(
        "root-o1-001",
        [[0.0, 0.0, 0.0], [0.05, 0.03, 0.0], [0.10, 0.06, 0.0]],
        parent_id="primary",
        order=1,
        insertion_index=1,
    )
    right = _root(
        "root-o1-002",
        [[0.0, 0.0, 0.0], [0.05, -0.03, 0.0], [0.10, -0.06, 0.0]],
        parent_id="primary",
        order=1,
        insertion_index=1,
    )
    points = np.array([[0.0, 0.01, 0.0], [0.0, 0.02, 0.0]])
    labels = np.array([-2, -1])

    resolved, report = _resolve_parent_owned_junctions(
        points,
        labels,
        primary,
        [left, right],
        d_bar=0.01,
        assignment_radius=0.04,
        ambiguity_margin=0.01,
        competing_labels={0: (1, 2)},
    )

    np.testing.assert_array_equal(resolved, [0, -1])
    assert report["resolved_vertex_count"] == 1


def test_uncertain_contact_far_from_any_insertion_remains_uncertain() -> None:
    primary = np.array([[0.0, 0.0, -0.2], [0.0, 0.0, 0.2]])
    first = _root(
        "root-o1-001",
        [[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [0.20, 0.0, 0.0]],
        parent_id="primary",
        order=1,
        insertion_index=0,
    )
    second = _root(
        "root-o1-002",
        [[0.0, 0.0, 0.15], [0.10, 0.0, 0.15], [0.20, 0.0, 0.15]],
        parent_id="primary",
        order=1,
        insertion_index=1,
    )
    crossing_contact = np.array([[0.18, 0.0, 0.075]])

    resolved, report = _resolve_parent_owned_junctions(
        crossing_contact,
        np.array([-2]),
        primary,
        [first, second],
        d_bar=0.005,
        assignment_radius=0.08,
        ambiguity_margin=0.005,
        competing_labels={0: (1, 2)},
    )

    assert resolved[0] == -2
    assert report["resolved_vertex_count"] == 0


def test_nested_junction_prefers_the_observed_direct_parent_pair() -> None:
    primary = np.array(
        [[0.0, 0.0, -0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]
    )
    order_one = _root(
        "root-o1-001",
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.06, 0.0, 0.0]],
        parent_id="primary",
        order=1,
        insertion_index=1,
    )
    order_two = _root(
        "root-o2-001",
        [[0.01, 0.0, 0.0], [0.01, 0.05, 0.0], [0.01, 0.10, 0.0]],
        parent_id="root-o1-001",
        order=2,
        insertion_index=1,
    )

    resolved, report = _resolve_parent_owned_junctions(
        np.array([[0.01, 0.005, 0.0]]),
        np.array([-2]),
        primary,
        [order_one, order_two],
        d_bar=0.01,
        assignment_radius=0.04,
        ambiguity_margin=0.01,
        competing_labels={0: (0, 2)},
    )

    assert resolved[0] == 1
    assert report["per_parent_vertex_count"] == {"root-o1-001": 1}
    assert report["multi_claim_vertex_count"] == 1


def test_unrelated_competitors_inside_a_junction_envelope_stay_uncertain() -> None:
    primary = np.array(
        [
            [0.0, 0.0, -0.1],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.15],
            [0.0, 0.0, 0.25],
        ]
    )
    first = _root(
        "root-o1-001",
        [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.10, 0.0, 0.0]],
        parent_id="primary",
        order=1,
        insertion_index=1,
    )
    other_parent = _root(
        "root-o1-002",
        [[0.0, 0.0, 0.15], [-0.05, 0.0, 0.15], [-0.10, 0.0, 0.15]],
        parent_id="primary",
        order=1,
        insertion_index=2,
    )
    unrelated_child = _root(
        "root-o2-001",
        [[-0.05, 0.0, 0.15], [-0.05, 0.05, 0.15]],
        parent_id="root-o1-002",
        order=2,
        insertion_index=1,
    )

    resolved, report = _resolve_parent_owned_junctions(
        np.array([[0.0, 0.01, 0.0]]),
        np.array([-2]),
        primary,
        [first, other_parent, unrelated_child],
        d_bar=0.01,
        assignment_radius=0.04,
        ambiguity_margin=0.01,
        competing_labels={0: (1, 3)},
    )

    assert resolved[0] == -2
    assert report["resolved_vertex_count"] == 0


def test_parent_child_distance_must_remain_ambiguous() -> None:
    primary = np.array(
        [[0.0, 0.0, -0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]
    )
    child = _root(
        "root-o1-001",
        [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.10, 0.0, 0.0]],
        parent_id="primary",
        order=1,
        insertion_index=1,
    )

    resolved, report = _resolve_parent_owned_junctions(
        np.array([[0.035, 0.0, 0.0]]),
        np.array([-2]),
        primary,
        [child],
        d_bar=0.01,
        assignment_radius=0.04,
        ambiguity_margin=0.01,
        competing_labels={0: (0, 1)},
    )

    assert resolved[0] == -2
    assert report["resolved_vertex_count"] == 0


def test_parent_support_indices_are_grouped_by_one_label_partition() -> None:
    grouped = _group_indices_by_label(
        np.array([2, -2, 0, 1, 2, 0, -1, 1]),
        {0, 2},
    )

    np.testing.assert_array_equal(grouped[0], [2, 5])
    np.testing.assert_array_equal(grouped[2], [0, 4])
    assert set(grouped) == {0, 2}


def test_small_discrete_patches_on_primary_surface_are_absorbed() -> None:
    points: list[list[float]] = []
    triangles: list[list[int]] = []
    labels: list[int] = []

    def add_fan(z: float, center_label: int) -> int:
        center = len(points)
        points.append([0.10, 0.0, z])
        labels.append(center_label)
        ring = []
        for y, dz in ((0.012, -0.008), (-0.012, -0.008), (0.0, 0.012)):
            ring.append(len(points))
            points.append([0.10, y, z + dz])
            labels.append(0)
        triangles.extend(
            [
                [center, ring[0], ring[1]],
                [center, ring[1], ring[2]],
                [center, ring[2], ring[0]],
            ]
        )
        return center

    o1_island = add_fan(-0.30, 1)
    uncertain_island = add_fan(-0.10, -2)
    unassigned_island = add_fan(0.10, -1)
    excluded_unassigned = add_fan(0.30, -1)

    main_o1 = []
    for x, y in ((0.10, 0.0), (0.12, 0.01), (0.14, 0.0)):
        main_o1.append(len(points))
        points.append([x, y, 0.45])
        labels.append(1)
    triangles.append(main_o1)
    main_boundary = []
    for y in (-0.015, 0.015):
        main_boundary.append(len(points))
        points.append([0.10, y, 0.45])
        labels.append(0)
    triangles.extend(
        [
            [main_o1[0], main_boundary[0], main_boundary[1]],
            [main_o1[0], main_o1[1], main_boundary[1]],
        ]
    )

    child = _root(
        "root-o1-001",
        [[0.0, 0.0, 0.45], [0.10, 0.0, 0.45], [0.20, 0.0, 0.45]],
        parent_id="primary",
        order=1,
        insertion_index=2,
    )
    excluded = np.zeros(len(points), dtype=bool)
    excluded[excluded_unassigned] = True
    primary_z = np.linspace(-0.6, 0.6, 25)
    primary = np.column_stack(
        [np.zeros(len(primary_z)), np.zeros(len(primary_z)), primary_z]
    )
    resolved, report = _absorb_small_primary_surface_patches(
        np.asarray(points, dtype=float),
        np.asarray(labels, dtype=int),
        primary,
        [child],
        d_bar=0.01,
        triangles=np.asarray(triangles, dtype=int),
        excluded_mask=excluded,
    )

    assert resolved[o1_island] == 0
    assert resolved[uncertain_island] == 0
    assert resolved[unassigned_island] == 0
    assert resolved[excluded_unassigned] == -1
    assert np.all(resolved[np.asarray(main_o1)] == 1)
    assert report["absorbed_patch_count"] == 3
    assert report["absorbed_vertex_count"] == 3
    assert report["per_source_label"] == {
        "root-o1-001": {"patch_count": 1, "vertex_count": 1},
        "unassigned": {"patch_count": 1, "vertex_count": 1},
        "uncertain": {"patch_count": 1, "vertex_count": 1},
    }


def test_small_patch_with_non_primary_boundary_is_not_absorbed() -> None:
    primary = np.asarray([[0.0, 0.0, -0.2], [0.0, 0.0, 0.2]])
    child = _root(
        "root-o1-001",
        [[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]],
        parent_id="primary",
        order=1,
        insertion_index=0,
    )
    points = np.asarray(
        [
            [0.10, 0.00, 0.0],
            [0.10, 0.01, 0.0],
            [0.10, -0.01, 0.0],
            [0.10, 0.00, 0.01],
        ]
    )
    labels = np.asarray([-2, 0, 1, 1])
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]])

    resolved, report = _absorb_small_primary_surface_patches(
        points,
        labels,
        primary,
        [child],
        d_bar=0.01,
        triangles=triangles,
    )

    assert resolved[0] == -2
    assert report["absorbed_patch_count"] == 0
