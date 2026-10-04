"""Regression checks for exact, results-preserving tracing optimizations.

The replay mode deliberately starts each forced hypothesis at the insertion.
Comparing every attempted trace catches state loss in rejected alternatives,
which would be invisible in a comparison of only retained hypotheses.
"""

from __future__ import annotations

import copy
from dataclasses import fields
from typing import Any

import numpy as np
import pytest
from scipy.spatial import cKDTree

from soyrootbio import lateral
from soyrootbio.types import RootPath


def _assert_exact(left: Any, right: Any) -> None:
    """Compare all biological/score state, including optional fields and NaNs."""

    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_exact(left_item, right_item)
    elif isinstance(left, (float, np.floating)) and np.isnan(left):
        assert isinstance(right, (float, np.floating)) and np.isnan(right)
    else:
        assert left == right


def _assert_path_exact(left: RootPath, right: RootPath) -> None:
    for field in fields(RootPath):
        _assert_exact(getattr(left, field.name), getattr(right, field.name))


def _fork_case(max_steps: int) -> dict[str, Any]:
    common_x = np.arange(0.01, 0.101, 0.01)
    short_x = np.arange(0.11, 0.151, 0.01)
    long_y = np.arange(0.01, 0.301, 0.01)
    common = np.column_stack([common_x, np.zeros((len(common_x), 2))])
    short_arm = np.column_stack([short_x, np.zeros((len(short_x), 2))])
    long_arm = np.column_stack(
        [np.full(len(long_y), 0.10), long_y, np.zeros(len(long_y))]
    )
    return _trace_arguments(np.vstack([common, short_arm, long_arm]), max_steps, 0.022)


def _rejected_fork_case(*, long_prefix: bool = False) -> dict[str, Any]:
    # The offset band provides repeatable angular proposals but never a
    # departing tube. Accepted-node halos also trigger covered-step recovery.
    end = 0.80 if long_prefix else 0.40
    side_start = 0.10 if long_prefix else 0.26
    spine_x = np.arange(0.01, end + 0.001, 0.01)
    side_x = np.arange(side_start, end + 0.001, 0.01)
    spine = np.column_stack([spine_x, np.zeros((len(spine_x), 2))])
    side = np.column_stack(
        [side_x, np.full(len(side_x), 0.020), np.zeros(len(side_x))]
    )
    return _trace_arguments(np.vstack([spine, side]), 100, 0.032)


def _trace_arguments(points: np.ndarray, max_steps: int, search_radius: float) -> dict[str, Any]:
    direction = np.array([1.0, 0.0, 0.0])
    start = lateral.LateralStart(
        start_id=0,
        point=np.array([0.01, 0.0, 0.0]),
        primary_point=np.zeros(3),
        primary_index=0,
        member_indices=np.arange(len(points)),
        direction=direction.copy(),
        surface_contact=True,
        surface_gap=0.002,
        surface_contact_count=3,
        tip_guard_exception=True,
        tip_departure_support=7,
        tip_departure_distance=0.025,
        tip_departure_extent=0.050,
        tip_departure_angle_degrees=70.0,
    )
    return dict(
        points=points,
        point_tree=cKDTree(points),
        allowed_mask=np.ones(len(points), dtype=bool),
        start=start,
        initial_direction=direction,
        primary_tangent=np.array([0.0, 1.0, 0.0]),
        step_length=0.01,
        open_angle=90.0,
        max_steps=max_steps,
        search_radius=search_radius,
        limit_primary_angle_to_insertion=True,
        density_support_index=None,
        cooperate=None,
    )


def _trace_all_attempts(
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, Any],
    *,
    replay_from_checkpoint: bool,
) -> tuple[list[RootPath], list[tuple[dict[int, int] | None, RootPath]], list[int]]:
    original = lateral._grow_one_candidate
    attempts: list[tuple[dict[int, int] | None, RootPath]] = []
    observation_sizes: list[int] = []

    def record_attempt(**kwargs: Any) -> RootPath:
        path = original(**kwargs)
        # Hypothesis acceptance later mutates score_components on the base.
        # Capture the raw growth output before those annotations are applied.
        attempts.append((copy.deepcopy(kwargs.get("forced_step_indices")), copy.deepcopy(path)))
        observations = kwargs.get("fork_observations")
        if observations is not None:
            observation_sizes.append(len(observations))
        return path

    with monkeypatch.context() as patch:
        patch.setattr(lateral, "_grow_one_candidate", record_attempt)
        retained = lateral._grow_candidate_hypotheses(
            **arguments,
            replay_from_checkpoint=replay_from_checkpoint,
        )
    return retained, attempts, observation_sizes


@pytest.mark.parametrize("max_steps, expected_hypotheses", [(6, 1), (10, 1), (14, 2), (45, 2)])
def test_checkpoint_matches_full_replay_at_supported_and_capped_forks(
    monkeypatch: pytest.MonkeyPatch,
    max_steps: int,
    expected_hypotheses: int,
) -> None:
    arguments = _fork_case(max_steps)
    replay, replay_attempts, _ = _trace_all_attempts(
        monkeypatch, arguments, replay_from_checkpoint=False
    )
    resumed, resumed_attempts, _ = _trace_all_attempts(
        monkeypatch, arguments, replay_from_checkpoint=True
    )
    assert len(replay) == len(resumed) == expected_hypotheses
    assert len(replay_attempts) == len(resumed_attempts)
    for (replay_forced, replay_path), (resumed_forced, resumed_path) in zip(
        replay_attempts, resumed_attempts
    ):
        assert replay_forced == resumed_forced
        _assert_path_exact(replay_path, resumed_path)
        assert len(resumed_path.points) <= max_steps + 2
    for replay_path, resumed_path in zip(replay, resumed):
        _assert_path_exact(replay_path, resumed_path)


@pytest.mark.parametrize("long_prefix", [False, True])
def test_rejected_forks_and_covered_recovery_match_full_replay(
    monkeypatch: pytest.MonkeyPatch, long_prefix: bool
) -> None:
    arguments = _rejected_fork_case(long_prefix=long_prefix)
    replay, replay_attempts, _ = _trace_all_attempts(
        monkeypatch, arguments, replay_from_checkpoint=False
    )
    resumed, resumed_attempts, observation_sizes = _trace_all_attempts(
        monkeypatch, arguments, replay_from_checkpoint=True
    )
    assert len(replay) == len(resumed) == 1
    assert len(replay_attempts) == len(resumed_attempts) > 5
    assert replay_attempts[0][1].score_components["adaptive_travel_covered_recovery_steps"] > 0.0
    assert any(
        path.score_components["adaptive_travel_covered_recovery_steps"] > 0.0
        for _, path in replay_attempts[1:]
    )
    for (replay_forced, replay_path), (resumed_forced, resumed_path) in zip(
        replay_attempts, resumed_attempts
    ):
        assert replay_forced == resumed_forced
        _assert_path_exact(replay_path, resumed_path)
    _assert_path_exact(replay[0], resumed[0])
    if long_prefix:
        # The original recorder saw 72 observations. Only the observable tail
        # may be retained, while all 13 eligible attempts must still be made.
        assert len(observation_sizes) == 1 and 0 < observation_sizes[0] <= 14
        assert [next(iter(forced)) for forced, _ in resumed_attempts[1:]] == list(
            range(77, 64, -1)
        )


def test_stable_best_proposal_keeps_first_tie_and_forcing_is_optional() -> None:
    # Both proposals have identical turn, density, distance, and unavailable
    # radius evidence. Use the tree's proposal order as the stable tie oracle.
    points = np.array([[0.020, 0.003, 0.0], [0.020, -0.003, 0.0]])
    arguments = _trace_arguments(points, 1, 0.025)
    arguments.pop("cooperate")
    ordinary = lateral._grow_one_candidate(**arguments)
    np.testing.assert_array_equal(ordinary.points[-1], points[0])
    forced = lateral._grow_one_candidate(**arguments, forced_step_indices={0: 1})
    np.testing.assert_array_equal(forced.points[-1], points[1])
    absent_force = lateral._grow_one_candidate(**arguments, forced_step_indices={0: 999})
    _assert_path_exact(ordinary, absent_force)


class _RecordingTree:
    """Real neighbor membership with observable query worker choices."""

    def __init__(self, points: np.ndarray) -> None:
        self.tree = cKDTree(points)
        self.calls: list[tuple[str, int, int]] = []

    def query_ball_point(self, query: np.ndarray, **kwargs: Any) -> Any:
        count = 1 if np.asarray(query).ndim == 1 else len(query)
        self.calls.append(("ball", count, kwargs.get("workers", 1)))
        return self.tree.query_ball_point(query, **kwargs)

    def query(self, query: np.ndarray, **kwargs: Any) -> Any:
        count = 1 if np.asarray(query).ndim == 1 else len(query)
        self.calls.append(("nearest", count, kwargs.get("workers", 1)))
        return self.tree.query(query, **kwargs)


def _anisotropic_support() -> np.ndarray:
    return np.asarray(
        [
            [x, y, z]
            for x in (-0.006, 0.0, 0.006)
            for y in (-0.002, 0.0, 0.002)
            for z in (-0.001, 0.0, 0.001)
        ],
        dtype=float,
    )


def _uncached_radius_reference(
    tree: cKDTree,
    points: np.ndarray,
    queries: np.ndarray,
    directions: np.ndarray,
    radius: float,
    mask: np.ndarray | None,
) -> np.ndarray:
    # Frozen arithmetic from the pre-optimization tracer. In particular,
    # neighbor membership may be shared, but perpendicular projection cannot.
    result = np.full(len(queries), np.nan)
    neighborhoods = tree.query_ball_point(queries, radius)
    for index, (nearby_raw, direction) in enumerate(zip(neighborhoods, directions)):
        nearby = np.asarray(nearby_raw, dtype=int)
        if mask is not None:
            nearby = nearby[mask[nearby]]
        if len(nearby) < 6:
            continue
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            continue
        axis = direction / norm
        samples = points[nearby]
        centered = samples - np.median(samples, axis=0)
        axial = centered @ axis
        perpendicular = centered - axial[:, None] * axis
        value = float(np.quantile(np.linalg.norm(perpendicular, axis=1), 0.70))
        if np.isfinite(value) and value > 1e-12:
            result[index] = value
    return result


@pytest.mark.parametrize("masked", [False, True])
def test_tiny_density_and_radius_queries_use_one_worker(
    monkeypatch: pytest.MonkeyPatch, masked: bool
) -> None:
    points = _anisotropic_support()
    tree = _RecordingTree(points)
    monkeypatch.setattr(lateral, "worker_threads", lambda: 3)
    queries = np.zeros((3, 3))
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    mask = points[:, 0] >= 0.0 if masked else None
    density = lateral._local_support_counts(
        tree, queries, radius=0.010, support_mask=mask
    )
    radii = lateral._local_radius_estimates(
        tree, points, queries, directions, radius=0.010, support_mask=mask
    )
    np.testing.assert_array_equal(density, np.full(3, 18 if masked else 27, dtype=float))
    np.testing.assert_array_equal(
        radii,
        _uncached_radius_reference(tree.tree, points, queries, directions, 0.010, mask),
    )
    assert radii[0] != radii[1]
    assert np.isnan(radii[2])
    assert tree.calls and all(workers == 1 for _, _, workers in tree.calls)


def test_bulk_support_query_keeps_configured_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    parent = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    tree = _RecordingTree(parent)
    monkeypatch.setattr(lateral, "worker_threads", lambda: 3)
    points = np.column_stack(
        [np.linspace(0.0, 1.0, 2048), np.full(2048, 0.050), np.zeros(2048)]
    )
    mask = lateral._novel_support_mask(
        points,
        occupied_mask=np.zeros(len(points), dtype=bool),
        parent_path=parent,
        parent_radius_profile=np.full(len(parent), 0.010),
        d_bar=0.001,
        parent_tree=tree,
    )
    assert np.all(mask)
    assert tree.calls == [("nearest", 2048, 3)]


def _scalar_fork_alternative(
    unit: np.ndarray,
    density: np.ndarray,
    score: np.ndarray,
    ranked: np.ndarray,
    best: int,
    minimum_support: int,
) -> tuple[int, float] | None:
    """Original scalar proposal order and inclusive angular thresholds."""

    for position_raw in ranked:
        position = int(position_raw)
        if position == best:
            continue
        angle = float(
            np.degrees(
                np.arccos(np.clip(np.dot(unit[best], unit[position]), -1.0, 1.0))
            )
        )
        if angle < 42.0 or angle > 145.0:
            continue
        if float(density[position]) < max(float(minimum_support), 0.35 * float(density[best])):
            continue
        if float(score[position]) < float(score[best]) - 0.75:
            continue
        return position, angle
    return None


@pytest.mark.parametrize("boundary", [42.0, 145.0])
@pytest.mark.parametrize("offset", [-1e-9, -1e-12, 0.0, 1e-12, 1e-9])
def test_batched_fork_angles_preserve_scalar_boundary_decisions(
    boundary: float, offset: float
) -> None:
    angle = np.radians(boundary + offset)
    unit = np.array([[1.0, 0.0, 0.0], [np.cos(angle), np.sin(angle), 0.0], [0.0, 1.0, 0.0]])
    density = np.array([10.0, 3.5, 8.0])
    score = np.array([1.0, 1.0, 0.9])
    ranked = np.argsort(-score, kind="stable")
    expected = _scalar_fork_alternative(unit, density, score, ranked, 0, 1)
    actual = lateral._first_fork_alternative(unit, density, score, ranked, 0, 1)
    _assert_exact(expected, actual)


def test_batched_fork_selection_preserves_rank_ties_support_and_score_limits() -> None:
    random = np.random.default_rng(20261002)
    for case in range(80):
        unit = random.normal(size=(32, 3))
        unit /= np.linalg.norm(unit, axis=1)[:, None]
        density = random.integers(0, 20, size=32).astype(float)
        score = random.choice([0.0, 0.25, 0.50, 1.0], size=32)
        ranked = np.argsort(-score, kind="stable")
        # A forced step can select a proposal other than ranked[0].
        best = int(ranked[0]) if case % 2 else int(random.integers(0, 32))
        minimum_support = 1 if case % 3 else 5
        expected = _scalar_fork_alternative(unit, density, score, ranked, best, minimum_support)
        actual = lateral._first_fork_alternative(unit, density, score, ranked, best, minimum_support)
        _assert_exact(expected, actual)
    # Score and relative density equality are inclusive. Adjacent floats below
    # each cutoff must remain ineligible, even if they have the earliest rank.
    unit = np.tile([0.0, 1.0, 0.0], (5, 1))
    unit[0] = [1.0, 0.0, 0.0]
    density = np.array([10.0, np.nextafter(3.5, 0.0), 3.5, 3.5, 3.5])
    score = np.array([1.0, 1.0, np.nextafter(0.25, 0.0), 0.25, 0.25])
    ranked = np.argsort(-score, kind="stable")
    assert lateral._first_fork_alternative(unit, density, score, ranked, 0, 1) == (3, 90.0)
    assert lateral._first_fork_alternative(unit, density, score, ranked, 0, 4) is None


def test_density_cache_isolates_frozen_support_masks_and_query_radii() -> None:
    points = _anisotropic_support()
    tree = _RecordingTree(points)
    left_mask = points[:, 0] <= 0.0
    right_mask = points[:, 0] > 0.0
    left = lateral._TraceEvidenceCache(points, tree, points, left_mask)
    right = lateral._TraceEvidenceCache(points, tree, points, right_mask)
    # Cache ownership generations must freeze their masks, even when the caller
    # subsequently edits the array it supplied to the constructor.
    left_mask[:] = False
    right_mask[:] = False
    indices = np.array([0, 13, 26, 13])
    for cache, frozen_mask in ((left, points[:, 0] <= 0.0), (right, points[:, 0] > 0.0)):
        for radius in (0.003, 0.010, 0.003):
            expected = np.asarray(
                [
                    np.count_nonzero(frozen_mask[np.asarray(tree.tree.query_ball_point(points[index], radius), dtype=int)])
                    for index in indices
                ],
                dtype=float,
            )
            np.testing.assert_array_equal(cache.density(indices, radius=radius), expected)
    assert left.density(np.array([13]), radius=0.010)[0] == 18
    assert right.density(np.array([13]), radius=0.010)[0] == 9
    count = len(tree.calls)
    left.density(indices, radius=0.010)
    right.density(indices, radius=0.010)
    assert len(tree.calls) == count


def test_cache_eviction_is_bounded_and_does_not_change_neighborhood_evidence() -> None:
    points = _anisotropic_support()
    tree = _RecordingTree(points)
    cache = lateral._TraceEvidenceCache(
        points,
        tree,
        points,
        max_density_entries=2,
        max_neighborhood_entries=2,
        max_neighborhood_bytes=2000,
    )
    queries = points[[0, 13, 26, 4, 18]]
    for query in queries:
        for radius in (0.003, 0.010):
            ((nearby, centered),) = cache.neighborhoods(query[None, :], radius=radius)
            expected_indices = np.asarray(
                tree.tree.query_ball_point(query[None, :], radius)[0], dtype=int
            )
            np.testing.assert_array_equal(nearby, expected_indices)
            expected_centered = (
                points[expected_indices] - np.median(points[expected_indices], axis=0)
                if len(expected_indices) >= 6
                else np.empty((0, 3))
            )
            np.testing.assert_array_equal(centered, expected_centered)
            assert cache.neighborhood_bytes <= 2000
            assert len(cache._neighborhoods) <= 2
        cache.density(np.array([int(np.flatnonzero(np.all(points == query, axis=1))[0])]), radius=0.010)
        assert len(cache._density) <= 2
    # Evicted entries must be recomputed exactly when revisited.
    first = cache.neighborhoods(queries[:1], radius=0.010)[0][0]
    np.testing.assert_array_equal(first, tree.tree.query_ball_point(queries[:1], 0.010)[0])
    # A neighborhood too large for the byte budget must still be returned in
    # full; the memory limit controls storage, never scientific evidence.
    tiny_cache = lateral._TraceEvidenceCache(
        points, tree, points, max_neighborhood_bytes=16
    )
    all_support = tiny_cache.neighborhoods(np.zeros((1, 3)), radius=0.010)[0][0]
    assert len(all_support) == len(points)
    assert tiny_cache.neighborhood_bytes == 0


@pytest.mark.parametrize("masked", [False, True])
def test_cached_radius_reprojects_shared_neighborhood_for_each_direction(masked: bool) -> None:
    points = _anisotropic_support()
    tree = _RecordingTree(points)
    mask = points[:, 0] >= 0.0 if masked else None
    cache = lateral._TraceEvidenceCache(points, tree, points, mask)
    queries = np.zeros((3, 3))
    for directions in (
        np.tile([1.0, 0.0, 0.0], (3, 1)),
        np.array([[0.0, 3.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]),
    ):
        actual = lateral._local_radius_estimates(
            tree,
            points,
            queries,
            directions,
            radius=0.010,
            support_mask=mask,
            evidence_cache=cache,
        )
        expected = _uncached_radius_reference(tree.tree, points, queries, directions, 0.010, mask)
        np.testing.assert_array_equal(actual, expected)
    # Three repeated coordinates share membership. Changing direction should
    # change the radius evidence without another spatial query.
    assert tree.calls == [("ball", 1, 1)]
    np.testing.assert_array_equal(actual[:2], actual[0])
    assert np.isnan(actual[-1])


@pytest.mark.parametrize("masked", [False, True])
def test_cached_and_uncached_growth_preserve_all_state_and_fork_evidence(masked: bool) -> None:
    arguments = _rejected_fork_case()
    points = arguments["points"]
    mask = points[:, 1] == 0.0 if masked else None
    cache = lateral._TraceEvidenceCache(
        points, arguments["point_tree"], points, mask
    )
    for forced in (None, {30: 48}, {29: 47}):
        old_observations: list[dict[str, float | int]] = []
        new_observations: list[dict[str, float | int]] = []
        uncached = lateral._grow_one_candidate(
            **arguments,
            density_support_mask=mask,
            forced_step_indices=forced,
            fork_observations=old_observations,
        )
        cached = lateral._grow_one_candidate(
            **arguments,
            density_support_mask=mask,
            forced_step_indices=forced,
            fork_observations=new_observations,
            evidence_cache=cache,
        )
        _assert_path_exact(uncached, cached)
        _assert_exact(old_observations, new_observations)
