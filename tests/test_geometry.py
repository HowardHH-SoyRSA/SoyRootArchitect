import numpy as np
import pytest
from scipy.spatial import cKDTree

import soyrootbio.geometry as geometry_module
from soyrootbio.geometry import (
    is_above_primary_top,
    mean_nearest_neighbor_distance,
    normalize_unit_box,
    path_length,
    primary_top_excess,
    resample_polyline,
)


def test_normalize_unit_box_and_inverse():
    points = np.array([[2.0, 4.0, 10.0], [4.0, 8.0, 14.0], [6.0, 6.0, 18.0]])
    normalized, transform = normalize_unit_box(points)
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(np.ptp(normalized, axis=0).max(), 1.0)
    np.testing.assert_allclose(transform.inverse_points(normalized), points)


def test_mean_nearest_neighbor_distance_line():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert np.isclose(mean_nearest_neighbor_distance(points), 1.0)


def test_mean_nearest_neighbor_distance_reuses_matching_point_tree(monkeypatch):
    points = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [2.0, 0.5, 0.0], [3.0, 1.0, 0.0],
    ])
    expected = mean_nearest_neighbor_distance(points)
    shared_tree = cKDTree(points)

    def reject_rebuild(*args, **kwargs):
        raise AssertionError("the unchanged point tree was rebuilt")

    monkeypatch.setattr(geometry_module, "cKDTree", reject_rebuild)
    assert mean_nearest_neighbor_distance(points, point_tree=shared_tree) == expected

    with pytest.raises(ValueError, match="same ordered points"):
        mean_nearest_neighbor_distance(points, point_tree=cKDTree(points[::-1]))


def test_resample_polyline_preserves_length_endpoints():
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    resampled = resample_polyline(points, spacing=0.25)
    np.testing.assert_allclose(resampled[0], points[0])
    np.testing.assert_allclose(resampled[-1], points[-1])
    assert np.isclose(path_length(resampled), 1.0)


def test_primary_top_rule_uses_configured_gravity_axis():
    primary = np.array(
        [
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 4.0],
            [0.0, 0.0, 8.0],
        ]
    )
    gravity = np.array([-1.0, 0.0, 0.0])

    assert is_above_primary_top(
        np.array([2.1, 0.0, -100.0]),
        primary,
        gravity=gravity,
    )
    excess, tolerance = primary_top_excess(
        np.array([2.0, 0.0, 100.0]),
        primary,
        gravity=gravity,
    )
    assert abs(excess) <= tolerance

