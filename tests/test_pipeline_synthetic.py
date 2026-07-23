import numpy as np

from soyrootbio.geometry import mean_nearest_neighbor_distance, normalize_unit_box
from soyrootbio.lateral import find_lateral_starting_points, grow_lateral_candidates, select_non_overlapping_paths
from soyrootbio.pipeline import _lateral_start_distance_limits
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


def test_lateral_start_gate_accepts_a_local_parent_radius_profile():
    z = np.linspace(0.0, 1.0, 80)
    primary_path = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    primary_points = np.column_stack([np.full_like(z, 0.04), np.zeros_like(z), z])
    lateral = np.column_stack(
        [
            np.linspace(0.055, 0.25, 50),
            np.zeros(50),
            np.full(50, 0.5),
        ]
    )
    points = np.vstack([primary_points, lateral])
    primary_mask = np.zeros(len(points), dtype=bool)
    primary_mask[: len(primary_points)] = True
    scalar_starts = find_lateral_starting_points(
        points,
        primary_mask,
        primary_path,
        min_cluster_size=4,
        max_parent_distance=0.05,
    )
    local_limits = np.full(len(primary_path), 0.05)
    local_limits[35:45] = 0.08
    adaptive_starts = find_lateral_starting_points(
        points,
        primary_mask,
        primary_path,
        min_cluster_size=4,
        max_parent_distance=local_limits,
    )

    assert not scalar_starts
    assert adaptive_starts
    radial = adaptive_starts[0].point - adaptive_starts[0].primary_point
    assert np.dot(adaptive_starts[0].direction, radial) > 0.0


def test_lateral_start_gate_disables_diameter_bridge_for_a_strongly_flared_parent():
    d_bar = 0.001
    ordinary_radii = np.full(20, 14.0 * d_bar)
    flared_radii = ordinary_radii.copy()
    flared_radii[-2:] = 19.0 * d_bar

    ordinary_limits = _lateral_start_distance_limits(ordinary_radii, d_bar)
    flared_limits = _lateral_start_distance_limits(flared_radii, d_bar)

    # The ordinary parent receives the collar bridge (2r + 2*d_bar).
    assert np.allclose(ordinary_limits, 30.0 * d_bar)
    # Once the parent is strongly flared, the same ordinary-width nodes use
    # only the local-radius-plus-sampling envelope (r + 9*d_bar).
    assert np.allclose(flared_limits[:-2], 23.0 * d_bar)
    assert np.allclose(flared_limits[-2:], 28.0 * d_bar)

