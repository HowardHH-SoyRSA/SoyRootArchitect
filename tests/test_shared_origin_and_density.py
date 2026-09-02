from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from soyrootbio.lateral import (
    LateralStart,
    _adaptive_minimum_travel_fraction,
    _grow_one_candidate,
    _novel_support_mask,
    _path_density_score,
    _radius_continuity_scores,
)
from soyrootbio.topology import repair_root_hierarchy, validate_root_tree
from soyrootbio.types import RootPath


def _primary() -> np.ndarray:
    z = np.linspace(1.0, 0.0, 101)
    return np.column_stack([np.zeros_like(z), np.zeros_like(z), z])


def _root(
    root_id: str,
    points: list[list[float]],
    *,
    order: int,
    parent_id: str,
) -> RootPath:
    polyline = np.asarray(points, dtype=float)
    return RootPath(
        root_id=root_id,
        points=polyline,
        raw_start_point=polyline[0].copy(),
        order=order,
        parent_id=parent_id,
        covered_indices=set(range(40)),
        score=100.0,
    )


def _root_nearest_tip(paths: list[RootPath], tip: np.ndarray) -> RootPath:
    return min(paths, key=lambda path: float(np.linalg.norm(path.points[-1] - tip)))


def test_shared_primary_crown_laterals_are_repaired_as_order_one_siblings() -> None:
    primary = _primary()
    first = _root(
        "first-crown-root",
        [[0.0, 0.0, 0.80], [0.03, 0.0, 0.77], [0.14, 0.0, 0.67], [0.30, 0.0, 0.55]],
        order=1,
        parent_id="primary",
    )
    shared_origin = _root(
        "provisional-child",
        [[0.0, 0.0, 0.80], [0.0, 0.03, 0.77], [0.0, 0.15, 0.66], [0.0, 0.31, 0.54]],
        order=2,
        parent_id=first.root_id,
    )
    # The biological surface seed is close to, but not artificially snapped
    # onto, the primary centreline (0.9 * repair tolerance for d_bar=0.001).
    shared_origin.raw_start_point = np.array([0.0072, 0.0, 0.80])

    repaired, _ = repair_root_hierarchy(primary, [first, shared_origin], d_bar=0.001)

    assert validate_root_tree(repaired) == []
    assert len(repaired) == 2
    assert all(path.order == 1 for path in repaired)
    assert all(path.parent_id == "primary" for path in repaired)
    promoted = _root_nearest_tip(repaired, shared_origin.points[-1])
    assert "shared_origin_promoted" in promoted.qc_flags
    assert promoted.score_components["shared_origin_ancestor_gap"] == pytest.approx(0.0072)


def test_distal_child_outside_ancestor_envelope_remains_order_two() -> None:
    primary = _primary()
    first = _root(
        "first-order",
        [[0.0, 0.0, 0.80], [0.08, 0.0, 0.72], [0.20, 0.0, 0.62], [0.32, 0.0, 0.50]],
        order=1,
        parent_id="primary",
    )
    distal_child_tip = np.array([0.20, 0.27, 0.48])
    distal_child = _root(
        "distal-child",
        [[0.20, 0.0, 0.62], [0.20, 0.05, 0.59], [0.20, 0.16, 0.54], distal_child_tip.tolist()],
        order=2,
        parent_id=first.root_id,
    )

    repaired, _ = repair_root_hierarchy(primary, [first, distal_child], d_bar=0.001)

    assert validate_root_tree(repaired) == []
    child = _root_nearest_tip(repaired, distal_child_tip)
    parent = next(path for path in repaired if path.root_id == child.parent_id)
    assert child.order == 2
    assert parent.order == 1
    assert parent.parent_id == "primary"


def test_terminal_endpoint_is_used_when_reorienting_a_reversed_root() -> None:
    primary = _primary()
    reversed_root = _root(
        "reversed",
        [[0.30, 0.0, 0.90], [0.20, 0.0, 0.80], [0.10, 0.0, 0.60], [0.0, 0.0, 0.50]],
        order=1,
        parent_id="primary",
    )

    repaired, report = repair_root_hierarchy(primary, [reversed_root], d_bar=0.001)

    assert report.roots_reoriented == 1
    np.testing.assert_allclose(repaired[0].points[0], [0.0, 0.0, 0.50])


def test_shared_origin_promotion_recursively_recomputes_genuine_descendant_order() -> None:
    primary = _primary()
    first = _root(
        "first-crown-root",
        [[0.0, 0.0, 0.80], [0.03, 0.0, 0.77], [0.14, 0.0, 0.67], [0.30, 0.0, 0.55]],
        order=1,
        parent_id="primary",
    )
    promoted_tip = np.array([0.0, 0.31, 0.54])
    promoted = _root(
        "shared-crown-root",
        [[0.0, 0.0, 0.80], [0.0, 0.04, 0.76], [0.0, 0.18, 0.65], promoted_tip.tolist()],
        order=2,
        parent_id=first.root_id,
    )
    descendant_tip = np.array([-0.20, 0.18, 0.48])
    descendant = _root(
        "genuine-descendant",
        [[0.0, 0.18, 0.65], [-0.04, 0.18, 0.61], [-0.12, 0.18, 0.55], descendant_tip.tolist()],
        order=3,
        parent_id=promoted.root_id,
    )

    repaired, _ = repair_root_hierarchy(
        primary,
        [first, promoted, descendant],
        d_bar=0.001,
    )

    assert validate_root_tree(repaired) == []
    promoted_result = _root_nearest_tip(repaired, promoted_tip)
    descendant_result = _root_nearest_tip(repaired, descendant_tip)
    assert promoted_result.order == 1
    assert promoted_result.parent_id == "primary"
    assert descendant_result.order == 2
    assert descendant_result.parent_id == promoted_result.root_id


def test_path_density_score_ignores_excluded_dense_parent_points() -> None:
    path = np.column_stack(
        [
            np.linspace(0.0, 0.30, 16),
            np.zeros(16),
            np.full(16, 0.50),
        ]
    )
    offsets = np.array([[0.0, 0.006, 0.0], [0.0, -0.006, 0.0]])
    novel_support = np.vstack([path + offset for offset in offsets])
    rng = np.random.default_rng(20260720)
    excluded_parent_points = np.repeat(path, 20, axis=0)
    excluded_parent_points += rng.normal(scale=0.0015, size=excluded_parent_points.shape)

    baseline = _path_density_score(
        novel_support,
        cKDTree(novel_support),
        path,
        radius=0.012,
        support_mask=np.ones(len(novel_support), dtype=bool),
    )
    combined = np.vstack([novel_support, excluded_parent_points])
    support_mask = np.r_[
        np.ones(len(novel_support), dtype=bool),
        np.zeros(len(excluded_parent_points), dtype=bool),
    ]
    filtered = _path_density_score(
        combined,
        cKDTree(combined),
        path,
        radius=0.012,
        support_mask=support_mask,
    )
    unfiltered = _path_density_score(
        combined,
        cKDTree(combined),
        path,
        radius=0.012,
    )

    assert filtered == pytest.approx(baseline)
    assert unfiltered > filtered


def test_novel_support_excludes_unoccupied_points_inside_parent_tube() -> None:
    parent = np.column_stack([np.zeros(11), np.zeros(11), np.linspace(1.0, 0.0, 11)])
    points = np.array(
        [
            [0.004, 0.0, 0.50],
            [0.030, 0.0, 0.50],
        ]
    )

    support = _novel_support_mask(
        points,
        occupied_mask=np.zeros(len(points), dtype=bool),
        parent_path=parent,
        parent_radius_profile=np.full(len(parent), 0.005),
        d_bar=0.001,
    )

    assert support.tolist() == [False, True]


def test_growth_density_chooses_novel_branch_over_denser_excluded_branch() -> None:
    upper = np.array([0.8, 0.6, 0.0])
    lower = np.array([0.8, -0.6, 0.0])
    angles_upper = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    angles_lower = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    excluded_dense = upper + np.column_stack(
        [
            0.04 * np.cos(angles_upper),
            np.zeros_like(angles_upper),
            0.04 * np.sin(angles_upper),
        ]
    )
    novel_sparse = lower + np.column_stack(
        [
            0.04 * np.cos(angles_lower),
            np.zeros_like(angles_lower),
            0.04 * np.sin(angles_lower),
        ]
    )
    points = np.vstack([upper, lower, excluded_dense, novel_sparse])
    support_mask = np.r_[
        False,
        True,
        np.zeros(len(excluded_dense), dtype=bool),
        np.ones(len(novel_sparse), dtype=bool),
    ]
    start = LateralStart(
        start_id=0,
        point=np.zeros(3),
        primary_point=np.zeros(3),
        primary_index=0,
        member_indices=np.array([0, 1]),
        direction=np.array([1.0, 0.0, 0.0]),
    )

    selected = _grow_one_candidate(
        points=points,
        point_tree=cKDTree(points),
        allowed_indices={0, 1},
        start=start,
        initial_direction=np.array([1.0, 0.0, 0.0]),
        primary_tangent=np.array([0.0, 0.0, 1.0]),
        step_length=1.0,
        open_angle=90.0,
        max_steps=1,
        search_radius=1.1,
        density_support_mask=support_mask,
    )

    np.testing.assert_allclose(selected.points[-1], lower)


def test_growth_keeps_primary_angle_constraint_at_insertion() -> None:
    def direction(angle_degrees: float) -> np.ndarray:
        angle = np.radians(angle_degrees)
        return np.array([np.sin(angle), 0.0, np.cos(angle)])

    inside_cone = direction(20.0)
    dense_outside_cone = direction(80.0)
    support_offsets = np.column_stack(
        [
            np.zeros(16),
            np.linspace(-0.01, 0.01, 16),
            np.zeros(16),
        ]
    )
    outside_support = dense_outside_cone + support_offsets
    points = np.vstack([inside_cone, dense_outside_cone, outside_support])
    support_mask = np.ones(len(points), dtype=bool)
    start = LateralStart(
        start_id=0,
        point=np.zeros(3),
        primary_point=np.array([0.0, 0.0, -0.25]),
        primary_index=0,
        member_indices=np.array([0, 1]),
        direction=direction(50.0),
    )

    selected = _grow_one_candidate(
        points=points,
        point_tree=cKDTree(points),
        allowed_indices={0, 1},
        start=start,
        initial_direction=direction(50.0),
        primary_tangent=np.array([0.0, 0.0, 1.0]),
        step_length=1.0,
        open_angle=35.0,
        max_steps=1,
        search_radius=1.05,
        limit_primary_angle_to_insertion=True,
        density_support_mask=support_mask,
    )

    np.testing.assert_allclose(selected.points[-1], inside_cone)


def test_growth_follows_evolving_tangent_after_insertion() -> None:
    def direction(angle_degrees: float) -> np.ndarray:
        angle = np.radians(angle_degrees)
        return np.array([np.sin(angle), 0.0, np.cos(angle)])

    insertion_step = direction(30.0)
    curved_continuation = insertion_step + direction(60.0)
    short_primary_aligned_child = insertion_step + direction(10.0)
    support_offsets = np.column_stack(
        [
            np.zeros(20),
            np.linspace(-0.01, 0.01, 20),
            np.zeros(20),
        ]
    )
    continuation_support = curved_continuation + support_offsets
    points = np.vstack(
        [
            insertion_step,
            curved_continuation,
            short_primary_aligned_child,
            continuation_support,
        ]
    )
    support_mask = np.ones(len(points), dtype=bool)
    start = LateralStart(
        start_id=0,
        point=np.zeros(3),
        primary_point=np.array([0.0, 0.0, -0.25]),
        primary_index=0,
        member_indices=np.array([0, 1, 2]),
        direction=direction(10.0),
    )

    selected = _grow_one_candidate(
        points=points,
        point_tree=cKDTree(points),
        allowed_indices={0, 1, 2},
        start=start,
        initial_direction=direction(10.0),
        primary_tangent=np.array([0.0, 0.0, 1.0]),
        step_length=1.0,
        open_angle=35.0,
        max_steps=2,
        search_radius=1.05,
        limit_primary_angle_to_insertion=True,
        density_support_mask=support_mask,
    )

    np.testing.assert_allclose(selected.points[-1], curved_continuation)

    fixed_reference = _grow_one_candidate(
        points=points,
        point_tree=cKDTree(points),
        allowed_indices={0, 1, 2},
        start=start,
        initial_direction=direction(10.0),
        primary_tangent=np.array([0.0, 0.0, 1.0]),
        step_length=1.0,
        open_angle=35.0,
        max_steps=2,
        search_radius=1.05,
        limit_primary_angle_to_insertion=False,
        density_support_mask=support_mask,
    )

    np.testing.assert_allclose(fixed_reference.points[-1], short_primary_aligned_child)


def test_single_path_tracer_does_not_emit_fork_hypotheses() -> None:
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.6, 0.0],
            [2.0, -0.6, 0.0],
            [3.0, 1.2, 0.0],
            [3.0, -1.2, 0.0],
        ]
    )
    start = LateralStart(
        start_id=0,
        point=np.zeros(3),
        primary_point=np.zeros(3),
        primary_index=0,
        member_indices=np.arange(len(points)),
        direction=np.array([1.0, 0.0, 0.0]),
    )

    path = _grow_one_candidate(
        points=points,
        point_tree=cKDTree(points),
        allowed_indices=set(range(len(points))),
        start=start,
        initial_direction=np.array([1.0, 0.0, 0.0]),
        primary_tangent=np.array([1.0, 0.0, 0.0]),
        step_length=1.0,
        open_angle=35.0,
        max_steps=3,
        search_radius=1.3,
        limit_primary_angle_to_insertion=True,
    )

    assert isinstance(path, RootPath)
    assert len(path.points) == 5
    assert abs(float(path.points[-1, 1])) > 0.5
    assert "trace_hypothesis_rank" not in path.score_components


def test_adaptive_minimum_travel_policy_stays_in_requested_ranges() -> None:
    assert _adaptive_minimum_travel_fraction(
        step_index=0,
        previous_turn_degrees=0.0,
        max_turn_degrees=70.0,
    ) == pytest.approx(0.30)
    assert _adaptive_minimum_travel_fraction(
        step_index=4,
        previous_turn_degrees=5.0,
        max_turn_degrees=70.0,
    ) == pytest.approx(0.375)
    assert _adaptive_minimum_travel_fraction(
        step_index=4,
        previous_turn_degrees=40.0,
        max_turn_degrees=70.0,
    ) == pytest.approx(0.30)
    assert _adaptive_minimum_travel_fraction(
        step_index=4,
        previous_turn_degrees=5.0,
        max_turn_degrees=70.0,
        fallback=True,
    ) == pytest.approx(0.30)


def test_stable_trace_falls_back_to_shorter_travel() -> None:
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.34, 0.0, 0.0],
        ]
    )
    start = LateralStart(
        start_id=0,
        point=np.zeros(3),
        primary_point=np.array([-0.2, 0.0, 0.0]),
        primary_index=0,
        member_indices=np.arange(len(points)),
        direction=np.array([1.0, 0.0, 0.0]),
    )
    path = _grow_one_candidate(
        points=points,
        point_tree=cKDTree(points),
        allowed_indices=set(range(len(points))),
        start=start,
        initial_direction=np.array([1.0, 0.0, 0.0]),
        primary_tangent=np.array([1.0, 0.0, 0.0]),
        step_length=1.0,
        open_angle=90.0,
        max_steps=3,
        search_radius=1.05,
        limit_primary_angle_to_insertion=True,
    )

    np.testing.assert_allclose(path.points[-1], [2.34, 0.0, 0.0])
    assert path.score_components["adaptive_travel_fallback_steps"] == 1.0
    assert (
        path.score_components[
            "adaptive_travel_covered_recovery_steps"
        ]
        == 0.0
    )
    assert path.score_components["adaptive_travel_fraction_min"] == pytest.approx(
        0.30
    )


def test_radius_continuity_remains_a_soft_similarity_reward() -> None:
    np.testing.assert_allclose(
        _radius_continuity_scores(np.array([0.02, 0.04, np.nan]), 0.02),
        [1.0, 0.5, 0.5],
    )
