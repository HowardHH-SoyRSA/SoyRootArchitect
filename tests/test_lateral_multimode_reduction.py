from __future__ import annotations

import numpy as np

from soyrootbio.lateral import reduce_similar_paths
from soyrootbio.types import RootPath


def _variant(
    root_id: str,
    tip: tuple[float, float, float],
    *,
    score: float,
    support_offset: int,
    start_index: int = 7,
) -> RootPath:
    tip_point = np.asarray(tip, dtype=float)
    fraction = np.linspace(0.0, 1.0, 21)[:, None]
    points = fraction * tip_point[None, :]
    support = set(range(support_offset, support_offset + 40))
    return RootPath(
        root_id=root_id,
        points=points,
        raw_start_point=np.zeros(3),
        start_index=start_index,
        parent_id="primary",
        covered_indices=support,
        novel_support_indices=support,
        score=score,
    )


def test_reduction_retains_two_repeatable_modes_from_one_start() -> None:
    candidates = [
        _variant("east-a", (1.00, 0.00, 0.0), score=20.0, support_offset=0),
        _variant("east-b", (1.01, 0.02, 0.0), score=21.0, support_offset=50),
        _variant("east-c", (0.98, -0.01, 0.0), score=19.0, support_offset=100),
        _variant("north-a", (0.00, 1.00, 0.0), score=17.0, support_offset=150),
        _variant("north-b", (0.02, 1.01, 0.0), score=18.0, support_offset=200),
        _variant("north-c", (-0.01, 0.98, 0.0), score=16.0, support_offset=250),
        # An isolated high-score turn must not displace either repeatable mode.
        _variant("isolated-outlier", (-0.8, -0.8, 0.0), score=10_000.0, support_offset=300),
    ]

    reduced = reduce_similar_paths(candidates)

    assert len(reduced) == 2
    tips = np.asarray([path.points[-1] for path in reduced])
    assert np.min(np.linalg.norm(tips - np.array([1.0, 0.0, 0.0]), axis=1)) < 0.04
    assert np.min(np.linalg.norm(tips - np.array([0.0, 1.0, 0.0]), axis=1)) < 0.04
    assert all(path.score_components["variant_endpoint_mode_count"] == 2.0 for path in reduced)
    assert all(path.score_components["variant_endpoint_mode_support"] == 3.0 for path in reduced)
    assert all(path.score_components["variant_endpoint_outliers_rejected"] == 1.0 for path in reduced)


def test_reduction_collapses_same_direction_variants_with_different_reach() -> None:
    candidates = [
        _variant("short", (0.70, 0.00, 0.0), score=10.0, support_offset=0),
        _variant("middle", (0.85, 0.01, 0.0), score=11.0, support_offset=50),
        _variant("long", (1.00, -0.01, 0.0), score=12.0, support_offset=100),
    ]

    reduced = reduce_similar_paths(candidates)

    assert len(reduced) == 1
    assert reduced[0].score_components["variant_endpoint_mode_count"] == 1.0
    assert reduced[0].score_components["variant_endpoint_mode_support"] == 3.0


def test_reduction_does_not_invent_two_modes_from_two_disagreeing_variants() -> None:
    east = _variant("east", (1.0, 0.0, 0.0), score=20.0, support_offset=0)
    north = _variant("north", (0.0, 1.0, 0.0), score=10.0, support_offset=50)

    reduced = reduce_similar_paths([east, north])

    assert reduced == [east]
    assert reduced[0].score_components["variant_endpoint_mode_count"] == 1.0
