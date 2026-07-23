from __future__ import annotations

import numpy as np

from soyrootbio.lateral import (
    estimate_parent_radius_profile,
    extend_lateral_tip,
    is_parent_tracking_candidate,
)
from soyrootbio.pipeline import _prune_parent_tracking_paths
from soyrootbio.types import RootPath


def _x_tube(stations: np.ndarray, radius: float = 0.0015, ring_points: int = 8) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, ring_points, endpoint=False)
    rings = [
        np.column_stack(
            [
                np.full(ring_points, station),
                radius * np.cos(angles),
                radius * np.sin(angles),
            ]
        )
        for station in stations
    ]
    return np.vstack(rings)


def _z_tube(centerline: np.ndarray, radius: float = 0.020, ring_points: int = 16) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, ring_points, endpoint=False)
    rings = [
        center
        + np.column_stack(
            [
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.zeros(ring_points),
            ]
        )
        for center in centerline
    ]
    return np.vstack(rings)


def test_tip_continuation_crosses_assignment_halo_and_reaches_supported_tip() -> None:
    main = _x_tube(np.arange(0.108, 0.401, 0.001))
    # A dense orthogonal branch is present, but a 45-degree continuation cone
    # must keep the selected path on the original forward axis.
    branch_y = np.arange(0.006, 0.081, 0.004)
    distractor = np.column_stack(
        [
            np.full(len(branch_y), 0.160),
            branch_y,
            np.zeros(len(branch_y)),
        ]
    )
    points = np.vstack([main, distractor])
    path = RootPath(
        root_id="root-a",
        points=np.array(
            [
                [0.000, 0.0, 0.0],
                [0.040, 0.0, 0.0],
                [0.080, 0.0, 0.0],
                [0.100, 0.0, 0.0],
            ]
        ),
        order=2,
        parent_id="root-parent",
        covered_indices={0},
    )

    result = extend_lateral_tip(
        points,
        path,
        np.zeros(len(points), dtype=bool),
        d_bar=0.001,
        max_steps=80,
    )

    assert result is path
    assert result.root_id == "root-a"
    assert result.order == 2
    assert result.parent_id == "root-parent"
    assert result.points[-1, 0] > 0.36
    assert abs(float(result.points[-1, 1])) < 0.01
    assert result.score_components["tip_continuation_accepted"] == 1.0
    assert result.score_components["tip_extension_length"] > 0.20
    assert len(result.covered_indices) > 100


def test_tip_continuation_rejects_sparse_forward_noise() -> None:
    points = np.array(
        [
            [0.108, 0.000, 0.000],
            [0.112, 0.001, 0.000],
            [0.116, -0.001, 0.000],
        ]
    )
    original = np.array(
        [
            [0.000, 0.0, 0.0],
            [0.040, 0.0, 0.0],
            [0.080, 0.0, 0.0],
            [0.100, 0.0, 0.0],
        ]
    )
    path = RootPath(root_id="root-a", points=original.copy(), covered_indices={1})

    extend_lateral_tip(
        points,
        path,
        np.zeros(len(points), dtype=bool),
        d_bar=0.001,
    )

    np.testing.assert_array_equal(path.points, original)
    assert path.covered_indices == {1}
    assert path.score_components["tip_extension_steps"] == 0.0


def test_parent_tracking_filter_rejects_surface_traces_and_keeps_departing_lateral() -> None:
    z = np.linspace(1.0, 0.0, 101)
    parent = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    support = _z_tube(parent)
    radii = estimate_parent_radius_profile(parent, support, d_bar=0.001)

    tipward_z = np.linspace(0.60, 0.20, 81)
    tipward_hugger = RootPath(
        root_id="tipward-hugger",
        points=np.column_stack(
            [
                np.r_[0.0, np.full(len(tipward_z) - 1, 0.020)],
                np.zeros(len(tipward_z)),
                tipward_z,
            ]
        ),
    )
    collarward_z = np.linspace(0.55, 0.92, 75)
    collarward_hugger = RootPath(
        root_id="collarward-hugger",
        points=np.column_stack(
            [
                np.linspace(0.0, 0.026, len(collarward_z)),
                np.zeros(len(collarward_z)),
                collarward_z,
            ]
        ),
    )
    lateral_t = np.linspace(0.0, 1.0, 81)
    departing = RootPath(
        root_id="departing",
        points=np.column_stack(
            [
                0.30 * lateral_t,
                np.zeros_like(lateral_t),
                0.55 - 0.10 * lateral_t,
            ]
        ),
    )

    tipward_rejected, tipward_metrics = is_parent_tracking_candidate(
        tipward_hugger, parent, radii, d_bar=0.001
    )
    collarward_rejected, collarward_metrics = is_parent_tracking_candidate(
        collarward_hugger, parent, radii, d_bar=0.001
    )
    departing_rejected, departing_metrics = is_parent_tracking_candidate(
        departing, parent, radii, d_bar=0.001
    )

    assert tipward_rejected
    assert tipward_metrics["parent_parallel_fraction"] > 0.90
    assert collarward_rejected
    assert collarward_metrics["parent_collarward_progress"] > 0.20
    assert not departing_rejected
    assert departing_metrics["parent_terminal_outside_fraction"] > 0.70


def test_pruning_parent_tracking_path_promotes_its_real_child() -> None:
    primary = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    false_parent = RootPath(
        root_id="false-parent",
        points=np.array([[0.0, 0.0, 0.8], [0.02, 0.0, 0.6]]),
        order=1,
        parent_id="primary",
        parent_points=primary,
        score_components={"parent_tracking_rejected": 1.0},
    )
    child = RootPath(
        root_id="real-child",
        points=np.array([[0.02, 0.0, 0.6], [0.20, 0.0, 0.5]]),
        order=2,
        parent_id="false-parent",
        parent_points=false_parent.points,
    )

    kept = _prune_parent_tracking_paths([false_parent, child])

    assert [path.root_id for path in kept] == ["real-child"]
    assert child.parent_id == "primary"
    assert child.order == 1
    np.testing.assert_array_equal(child.parent_points, primary)
    assert child.score_components["parent_tracking_ancestor_promotions"] == 1.0
    assert "parent_tracking_parent_removed" in child.qc_flags
