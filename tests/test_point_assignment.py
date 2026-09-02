from __future__ import annotations

import numpy as np

from soyrootbio.pipeline import (
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
