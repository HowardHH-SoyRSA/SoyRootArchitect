from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

import soyrootbio.pipeline as pipeline_module
from soyrootbio.geometry import mean_nearest_neighbor_distance, normalize_unit_box, path_length
from soyrootbio.lateral import LateralStart
from soyrootbio.pipeline import _trace_lateral_orders
from soyrootbio.topology import repair_root_hierarchy, validate_root_tree
from soyrootbio.types import RootPath


def _tube_surface(centerline: np.ndarray, radius: float, ring_points: int) -> np.ndarray:
    """Return deterministic circular rings around a supplied centreline."""

    centerline = np.asarray(centerline, dtype=float)
    tangents = np.gradient(centerline, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    angles = np.linspace(0.0, 2.0 * np.pi, ring_points, endpoint=False)
    rings = []
    for center, tangent in zip(centerline, tangents):
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(helper, tangent))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis_u = helper - np.dot(helper, tangent) * tangent
        axis_u /= np.linalg.norm(axis_u)
        axis_v = np.cross(tangent, axis_u)
        rings.append(
            center
            + radius
            * (
                np.cos(angles)[:, None] * axis_u[None, :]
                + np.sin(angles)[:, None] * axis_v[None, :]
            )
        )
    return np.vstack(rings)


def _trace_known_primary(
    primary_centerline: np.ndarray,
    surfaces: list[np.ndarray],
    expected_laterals: list[np.ndarray],
) -> tuple[list[RootPath], list[np.ndarray]]:
    primary_surface_count = len(surfaces[0])
    points, normalization = normalize_unit_box(np.vstack(surfaces))
    primary = normalization.transform_points(primary_centerline)
    expected = [normalization.transform_points(centerline) for centerline in expected_laterals]
    primary_mask = np.zeros(len(points), dtype=bool)
    primary_mask[:primary_surface_count] = True
    d_bar = mean_nearest_neighbor_distance(points)

    traced, _, _, _ = _trace_lateral_orders(
        points,
        primary,
        primary_mask,
        d_bar,
        max_root_order=3,
        max_paths=None,
    )
    repaired, _ = repair_root_hierarchy(primary, traced, d_bar=d_bar)
    assert validate_root_tree(repaired) == []
    return repaired, expected


def test_tracing_rejects_start_above_primary_top_before_candidate_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
        ]
    )
    points = np.vstack(
        [
            primary,
            np.array(
                [
                    [0.02, 0.0, 1.10],
                    [0.03, 0.0, 1.12],
                    [0.04, 0.0, 1.14],
                    [0.05, 0.0, 1.16],
                ]
            ),
        ]
    )
    primary_mask = np.zeros(len(points), dtype=bool)
    primary_mask[: len(primary)] = True
    invalid_start = LateralStart(
        start_id=0,
        point=points[3],
        primary_point=np.array([0.0, 0.0, 1.05]),
        primary_index=0,
        member_indices=np.arange(3, len(points)),
    )

    monkeypatch.setattr(
        pipeline_module,
        "find_lateral_starting_points",
        lambda *args, **kwargs: [invalid_start],
    )

    def fail_candidate_growth(*args, **kwargs):
        raise AssertionError("an above-top start reached candidate growth")

    monkeypatch.setattr(
        pipeline_module,
        "grow_lateral_candidates",
        fail_candidate_growth,
    )
    report: dict[str, object] = {}

    roots, starts, candidates, order_counts = _trace_lateral_orders(
        points,
        primary,
        primary_mask,
        d_bar=0.02,
        max_root_order=3,
        max_paths=None,
        origin_report=report,
    )

    assert roots == []
    assert starts == 1
    assert candidates == 0
    assert order_counts == {}
    assert report["policy"] == "lateral-origin-at-or-below-primary-top-v1"
    assert report["rejected_start_count"] == 1
    assert report["rejected_refined_path_count"] == 0
    assert report["per_order"] == [
        {
            "root_order": 1,
            "detected_start_count": 1,
            "eligible_start_count": 0,
            "rejected_start_count": 1,
            "rejected_refined_path_count": 0,
        }
    ]


def test_clean_surface_branching_stops_at_junction_and_preserves_two_orders() -> None:
    z = np.linspace(1.0, 0.0, 121)
    primary = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    t = np.linspace(0.0, 1.0, 91)
    first_order = np.column_stack([0.42 * t, np.zeros_like(t), 0.72 - 0.18 * t])
    u = np.linspace(0.0, 1.0, 61)
    junction = np.array([0.42 * 0.55, 0.0, 0.72 - 0.18 * 0.55])
    second_order = junction + np.column_stack([0.08 * u, 0.24 * u, -0.09 * u])
    surfaces = [
        _tube_surface(primary, 0.030, 24),
        _tube_surface(first_order, 0.014, 18),
        _tube_surface(second_order, 0.009, 14),
    ]

    repaired, expected = _trace_known_primary(
        primary,
        surfaces,
        [first_order, second_order],
    )

    assert len(repaired) == 2
    by_order = {root.order: root for root in repaired}
    assert set(by_order) == {1, 2}
    assert by_order[1].parent_id == "primary"
    assert by_order[2].parent_id == by_order[1].root_id
    assert (
        np.linalg.norm(by_order[2].points[0] - expected[1][0])
        < 0.040
    )
    for order, expected_centerline in enumerate(expected, start=1):
        root = by_order[order]
        distances = cKDTree(expected_centerline).query(root.points, k=1)[0]
        assert float(np.median(distances)) < 0.013
        assert float(np.max(distances)) < 0.040
        expected_length = path_length(expected_centerline)
        assert abs(root.length - expected_length) / expected_length < 0.20


def test_clean_surface_independent_laterals_remain_first_order() -> None:
    z = np.linspace(1.0, 0.0, 101)
    primary = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    laterals = []
    surfaces = [_tube_surface(primary, 0.030, 20)]
    for insertion_z, azimuth in zip(
        (0.80, 0.60, 0.40),
        (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0),
    ):
        t = np.linspace(0.0, 1.0, 71)
        centerline = np.column_stack(
            [
                0.32 * np.cos(azimuth) * t,
                0.32 * np.sin(azimuth) * t,
                insertion_z - 0.14 * t,
            ]
        )
        laterals.append(centerline)
        surfaces.append(_tube_surface(centerline, 0.012, 16))

    repaired, _ = _trace_known_primary(primary, surfaces, laterals)

    assert len(repaired) == 3
    assert all(root.order == 1 and root.parent_id == "primary" for root in repaired)
