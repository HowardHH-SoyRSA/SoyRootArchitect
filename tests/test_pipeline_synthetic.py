import numpy as np

from soyrootbio.geometry import mean_nearest_neighbor_distance, normalize_unit_box
from soyrootbio.lateral import find_lateral_starting_points, grow_lateral_candidates, select_non_overlapping_paths
from soyrootbio.primary import estimate_primary_path, tangent_plane_primary_segmentation


def synthetic_root_points():
    rng = np.random.default_rng(3)
    z = np.linspace(0, 1, 120)
    primary = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    primary += rng.normal(scale=0.006, size=primary.shape)
    lateral_z = np.linspace(0.36, 0.78, 70)
    lateral = np.column_stack([0.9 * (lateral_z - 0.36), np.zeros_like(lateral_z), lateral_z])
    lateral += rng.normal(scale=0.006, size=lateral.shape)
    return np.vstack([primary, lateral])


def test_primary_and_lateral_steps_on_synthetic_cloud():
    points, norm = normalize_unit_box(synthetic_root_points())
    d_bar = mean_nearest_neighbor_distance(points)
    primary = estimate_primary_path(points, points[np.argmin(points[:, 2])], points[np.argmax(points[:, 2])], d_bar=d_bar)
    assert primary.length > 0.75
    primary_mask = tangent_plane_primary_segmentation(points, primary.points, d_bar=d_bar)
    assert primary_mask.sum() > 40
    starts = find_lateral_starting_points(points, primary_mask, primary.points, closest_fraction=0.15, min_cluster_size=4)
    assert starts
    candidates = grow_lateral_candidates(points, starts, primary.points, primary_mask, d_bar=d_bar, max_steps=20)
    selected = select_non_overlapping_paths(candidates, points, d_bar=d_bar, max_paths=3)
    assert selected
    assert selected[0].length > 0.05

