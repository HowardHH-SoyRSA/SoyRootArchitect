import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from soyrootbio.geometry import mean_nearest_neighbor_distance
from soyrootbio.primary import cluster_hdbscan, refine_primary_centerline, tangent_plane_primary_segmentation


def test_hdbscan_does_not_create_nested_worker_pools(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeHDBSCAN:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def fit_predict(self, data: np.ndarray) -> np.ndarray:
            return np.zeros(len(data), dtype=int)

    monkeypatch.setitem(sys.modules, "hdbscan", SimpleNamespace(HDBSCAN=FakeHDBSCAN))
    labels = cluster_hdbscan(np.arange(18, dtype=float).reshape(6, 3), min_cluster_size=2)

    assert labels.tolist() == [0, 0, 0, 0, 0, 0]
    assert captured["core_dist_n_jobs"] == 1


def _curved_tube(radius: float = 0.018) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, 120)
    centerline = np.column_stack([
        0.08 * np.sin(1.4 * np.pi * t),
        0.03 * np.cos(1.1 * np.pi * t),
        t,
    ])
    tangents = np.gradient(centerline, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    tube_points = []
    surface_path = []
    angles = np.linspace(0.0, 2.0 * np.pi, 28, endpoint=False)
    for center, tangent in zip(centerline, tangents):
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(helper, tangent)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        u = helper - np.dot(helper, tangent) * tangent
        u /= np.linalg.norm(u)
        v = np.cross(tangent, u)
        ring = center + radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
        tube_points.append(ring)
        surface_path.append(center + radius * u)
    return np.vstack(tube_points), centerline, np.asarray(surface_path)


def test_refine_primary_centerline_moves_surface_path_to_tube_axis():
    points, expected_centerline, surface_path = _curved_tube()
    d_bar = mean_nearest_neighbor_distance(points)

    refined = refine_primary_centerline(
        points,
        np.ones(len(points), dtype=bool),
        surface_path,
        d_bar=d_bar,
        fit_circular_cross_sections=True,
    )

    expected_tree = cKDTree(expected_centerline)
    coarse_error = np.median(expected_tree.query(surface_path, k=1)[0])
    refined_error = np.median(expected_tree.query(refined, k=1)[0])
    assert refined_error < 0.35 * coarse_error
    assert expected_tree.query(refined[0], k=1)[0] < 0.35 * coarse_error
    assert np.linalg.norm(refined[-1] - surface_path[-1]) < 1e-12


def test_lateral_mode_does_not_extrapolate_from_a_narrow_surface_patch():
    angles = np.linspace(-np.pi / 6.0, np.pi / 6.0, 18)
    z = np.linspace(0.0, 1.0, 60)
    points = np.vstack(
        [
            np.column_stack(
                [
                    0.02 * np.cos(angles),
                    0.02 * np.sin(angles),
                    np.full_like(angles, station),
                ]
            )
            for station in z
        ]
    )
    surface_path = np.column_stack([np.full_like(z, 0.02), np.zeros_like(z), z])
    d_bar = mean_nearest_neighbor_distance(points)

    refined = refine_primary_centerline(
        points,
        np.ones(len(points), dtype=bool),
        surface_path,
        d_bar=d_bar,
    )

    assert np.median(cKDTree(points).query(refined, k=1)[0]) < 0.015


def test_two_pass_primary_segmentation_completes_an_off_wall_collar():
    z = np.linspace(0.0, 1.0, 300)
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    radius = 0.008
    points = np.vstack(
        [
            np.column_stack(
                [
                    radius * np.cos(angles),
                    radius * np.sin(angles),
                    np.full_like(angles, station),
                ]
            )
            for station in z
        ]
    )
    surface_path = np.column_stack([np.full_like(z, radius), np.zeros_like(z), z])
    d_bar = mean_nearest_neighbor_distance(points)

    first_mask = tangent_plane_primary_segmentation(
        points,
        surface_path,
        d_bar=d_bar,
    )
    centered = refine_primary_centerline(
        points,
        first_mask,
        surface_path,
        d_bar=d_bar,
        fit_circular_cross_sections=True,
    )
    completed_mask = tangent_plane_primary_segmentation(
        points,
        centered,
        d_bar=d_bar,
        complete_cross_section=True,
    )

    first_collar_fraction = first_mask.reshape(len(z), len(angles))[:12].mean()
    completed_collar_fraction = completed_mask.reshape(len(z), len(angles))[:12].mean()
    assert completed_collar_fraction > 0.90
    assert completed_collar_fraction > first_collar_fraction + 0.20
