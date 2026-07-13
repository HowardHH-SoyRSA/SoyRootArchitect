import numpy as np
from scipy.spatial import cKDTree

from soyrootbio.geometry import mean_nearest_neighbor_distance
from soyrootbio.primary import refine_primary_centerline


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
    )

    expected_tree = cKDTree(expected_centerline)
    coarse_error = np.median(expected_tree.query(surface_path, k=1)[0])
    refined_error = np.median(expected_tree.query(refined, k=1)[0])
    assert refined_error < 0.35 * coarse_error
    assert np.linalg.norm(refined[0] - surface_path[0]) < 1e-12
    assert np.linalg.norm(refined[-1] - surface_path[-1]) < 1e-12
