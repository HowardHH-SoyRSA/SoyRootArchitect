import numpy as np
import pytest

from soyrootbio.primary import GRAVITY, rank_primary_candidates
from soyrootbio.geometry import tangent_vectors
from soyrootbio.pipeline import (
    _assign_full_root_labels,
    _assign_lateral_points,
    _points_above_base_mask,
    _selected_base_exclusion_mask,
)
from soyrootbio.topology import repair_root_hierarchy, validate_root_tree
from soyrootbio.traits import compute_traits
from soyrootbio.types import Normalization, RootPath


REQUESTED_PRIMARY_SCORE_COMPONENTS = {
    "basal_location",
    "local_radius_thickness_continuity",
    "downward_extent",
    "path_length",
    "graph_centrality",
}


@pytest.mark.parametrize(
    ("gravity", "points", "expected"),
    [
        (
            GRAVITY,
            np.array(
                [
                    [1.0, 2.0, 3.01],
                    [1.5, 2.5, 3.00],
                    [1.0, 2.0, 2.99],
                ]
            ),
            [True, False, False],
        ),
        (
            np.array([-1.0, 0.0, 0.0]),
            np.array(
                [
                    [1.01, 2.0, 3.0],
                    [1.00, 2.5, 3.5],
                    [0.99, 2.0, 3.0],
                ]
            ),
            [True, False, False],
        ),
    ],
)
def test_above_base_mask_follows_local_tipward_axis_and_keeps_base_plane(
    gravity: np.ndarray,
    points: np.ndarray,
    expected: list[bool],
):
    base = np.array([1.0, 2.0, 3.0])

    mask = _points_above_base_mask(points, base, gravity)

    np.testing.assert_array_equal(mask, expected)


def test_tilted_base_plane_keeps_both_sides_of_the_selected_cross_section():
    base = np.array([0.0, 0.0, 0.0])
    tipward = np.array([1.0, 0.0, -1.0])
    tipward /= np.linalg.norm(tipward)
    radial = np.array([1.0, 0.0, 1.0])
    radial /= np.linalg.norm(radial)
    points = np.vstack(
        [
            base + 0.03 * radial,
            base - 0.03 * radial,
            base - 0.03 * tipward,
            base + 0.03 * tipward,
        ]
    )

    mask = _points_above_base_mask(points, base, tipward, tolerance=1e-6)

    np.testing.assert_array_equal(mask, [False, False, True, False])


def test_base_exclusion_uses_gravity_beyond_the_local_collar_neighbourhood():
    base = np.zeros(3)
    tipward = np.array([1.0, 0.0, -1.0])
    points = np.array(
        [
            [0.02, 0.0, 0.02],   # Same tilted collar cross-section.
            [-2.0, 0.0, -1.0],   # Crosses the oblique plane, but is below base.
            [2.0, 0.0, 1.0],     # Far and vertically above the selected base.
        ]
    )

    mask = _selected_base_exclusion_mask(
        points,
        base,
        tipward,
        gravity=GRAVITY,
        collar_neighborhood_radius=0.1,
    )

    np.testing.assert_array_equal(mask, [False, False, True])


def test_above_base_points_are_excluded_inside_lateral_assignment_radius():
    base = np.zeros(3)
    points = np.array(
        [
            [0.001, 0.0, 0.002],  # Above base, but only 0.002 from the path.
            [0.001, 0.0, 0.000],  # Exactly on the base plane.
            [0.001, 0.0, -0.002],
        ]
    )
    lateral = RootPath(
        root_id="root-o1-001",
        points=np.array([[0.0, 0.0, 0.0], [0.004, 0.0, 0.0]]),
        order=1,
        parent_id="primary",
    )
    primary_mask = np.zeros(len(points), dtype=bool)
    above_base = _points_above_base_mask(points, base, GRAVITY)

    labels_without_exclusion = _assign_lateral_points(
        points,
        [lateral],
        primary_mask,
        d_bar=0.001,
    )
    labels_with_exclusion = _assign_lateral_points(
        points,
        [lateral],
        primary_mask,
        d_bar=0.001,
        excluded_mask=above_base,
    )
    full_labels_with_exclusion = _assign_full_root_labels(
        points,
        np.array([[1.0, 1.0, -1.0], [1.0, 1.0, -2.0]]),
        [lateral],
        d_bar=0.001,
        excluded_mask=above_base,
    )

    np.testing.assert_array_equal(labels_without_exclusion, [1, 1, 1])
    np.testing.assert_array_equal(labels_with_exclusion, [0, 1, 1])
    np.testing.assert_array_equal(full_labels_with_exclusion, [-1, 1, 1])


def _thick_taproot_cloud() -> np.ndarray:
    """Return a deterministic, tapered cylindrical surface ordered top to bottom."""

    heights = np.linspace(1.0, 0.0, 51)
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    rings = []
    for height in heights:
        radius = 0.018 + 0.012 * height
        rings.append(
            np.column_stack(
                [
                    radius * np.cos(angles),
                    radius * np.sin(angles),
                    np.full_like(angles, height),
                ]
            )
        )
    return np.vstack(rings)


def test_primary_candidates_expose_five_scores_and_follow_gravity():
    points = _thick_taproot_cloud()

    candidates = rank_primary_candidates(
        points,
        d_bar=0.006,
        graph_k=18,
        max_candidates=4,
        collar_seed_count=8,
    )

    assert candidates
    assert [candidate.rank for candidate in candidates] == list(range(1, len(candidates) + 1))
    assert [candidate.score for candidate in candidates] == sorted(
        (candidate.score for candidate in candidates), reverse=True
    )
    for candidate in candidates:
        assert set(candidate.components) == REQUESTED_PRIMARY_SCORE_COMPONENTS
        assert all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in candidate.components.values())
        assert np.isfinite(candidate.score)
        assert np.isfinite(candidate.confidence) and 0.0 <= candidate.confidence <= 1.0
        np.testing.assert_allclose(candidate.path[0], candidate.start, atol=1e-12)
        np.testing.assert_allclose(candidate.path[-1], candidate.end, atol=1e-12)
        # Paths must be stored from the upper collar toward the lower tip.
        assert candidate.start[2] > candidate.end[2]
        assert np.dot(candidate.path[-1] - candidate.path[0], GRAVITY) > 0.75


def _hierarchy_candidates() -> tuple[np.ndarray, list[RootPath]]:
    primary = np.column_stack(
        [
            np.zeros(11),
            np.zeros(11),
            np.linspace(1.0, 0.0, 11),
        ]
    )
    first_order = RootPath(
        root_id="candidate-first",
        # Deliberately tip-to-insertion: repair must orient this path.
        points=np.array(
            [
                [0.30, 0.0, 0.65],
                [0.20, 0.0, 0.70],
                [0.10, 0.0, 0.75],
                [0.00, 0.0, 0.80],
            ]
        ),
        order=1,
        parent_id="primary",
        covered_indices=set(range(40)),
        score=120.0,
    )
    second_order = RootPath(
        root_id="candidate-second",
        points=np.array(
            [
                [0.20, 0.0, 0.70],
                [0.20, 0.10, 0.65],
                [0.20, 0.20, 0.60],
                [0.20, 0.30, 0.55],
            ]
        ),
        order=2,
        parent_id="candidate-first",
        covered_indices=set(range(40, 80)),
        score=100.0,
    )
    short_first_order = RootPath(
        root_id="candidate-short",
        points=np.array(
            [
                [0.00, 0.0, 0.40],
                [0.02, 0.0, 0.40],
            ]
        ),
        order=1,
        parent_id="primary",
    )
    return primary, [second_order, short_first_order, first_order]


def test_hierarchy_repair_enforces_recursive_orders_and_validation_metadata():
    primary, candidates = _hierarchy_candidates()

    repaired, report = repair_root_hierarchy(primary, candidates, d_bar=0.01)

    assert len({root.root_id for root in repaired}) == len(repaired)
    assert all(root.root_id.startswith(f"root-o{root.order}-") for root in repaired)
    assert validate_root_tree(repaired) == []
    assert report.warnings == []
    assert report.roots_reoriented >= 1

    by_id = {root.root_id: root for root in repaired}
    assert {root.order for root in repaired} == {1, 2}
    assert all(root.parent_id == "primary" for root in repaired if root.order == 1)
    for root in repaired:
        expected_order = 1 if root.parent_id == "primary" else by_id[root.parent_id].order + 1
        assert root.order == expected_order
        assert root.parent_points is not None
        assert root.insertion_index is not None
        assert 0 <= root.insertion_index < len(root.parent_points)
        assert root.insertion_point is not None
        np.testing.assert_allclose(root.insertion_point, root.parent_points[root.insertion_index])
        np.testing.assert_allclose(root.points[0], root.insertion_point)
        assert np.isfinite(root.confidence) and 0.0 <= root.confidence <= 1.0
        assert isinstance(root.qc_flags, list)
        assert len(root.qc_flags) == len(set(root.qc_flags))
        assert all(isinstance(flag, str) and flag for flag in root.qc_flags)
        assert {
            "attachment",
            "junction_tangent_continuity",
            "length_support",
            "point_support",
        }.issubset(root.score_components)

    short_root = min(repaired, key=lambda root: root.length)
    assert "short_trace" in short_root.qc_flags
    assert ("low_confidence" in short_root.qc_flags) == (short_root.confidence < 0.55)
    assert report.low_confidence_roots == sum(root.confidence < 0.55 for root in repaired)


def test_hierarchy_repair_removes_overlong_child_and_its_descendants():
    primary = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
        ]
    )
    parent = RootPath(
        root_id="candidate-parent",
        points=np.array(
            [
                [0.0, 0.0, 1.5],
                [0.4, 0.0, 1.5],
            ]
        ),
        order=1,
        parent_id="primary",
    )
    overlong_child = RootPath(
        root_id="candidate-overlong-child",
        points=np.array(
            [
                [0.4, 0.0, 1.5],
                [0.4, 0.7, 1.5],
            ]
        ),
        order=2,
        parent_id=parent.root_id,
    )
    dependent_descendant = RootPath(
        root_id="candidate-dependent-descendant",
        points=np.array(
            [
                [0.4, 0.7, 1.5],
                [0.4, 0.7, 1.4],
            ]
        ),
        order=3,
        parent_id=overlong_child.root_id,
    )
    valid_sibling = RootPath(
        root_id="candidate-valid-sibling",
        points=np.array(
            [
                [0.0, 0.0, 0.5],
                [-0.3, 0.0, 0.5],
            ]
        ),
        order=1,
        parent_id="primary",
    )
    overlong_first_order = RootPath(
        root_id="candidate-overlong-first-order",
        points=np.array(
            [
                [0.0, 0.0, 1.0],
                [2.2, 0.0, 1.0],
            ]
        ),
        order=1,
        parent_id="primary",
    )

    repaired, report = repair_root_hierarchy(
        primary,
        [
            parent,
            overlong_child,
            dependent_descendant,
            valid_sibling,
            overlong_first_order,
        ],
        d_bar=0.005,
    )

    assert len(repaired) == 2
    assert all(root.order == 1 for root in repaired)
    assert validate_root_tree(repaired, primary_path=primary) == []
    assert report.overlong_children_removed == 2
    assert report.overlong_descendants_removed == 1
    assert len(report.overlong_child_details) == 2
    assert any(
        detail["parent_id"] == "primary"
        for detail in report.overlong_child_details
    )
    for detail in report.overlong_child_details:
        assert detail["child_length_normalized"] > detail["parent_length_normalized"]
        assert detail["child_parent_length_ratio"] > 1.0


def _surface_samples(path: np.ndarray, radius: float) -> np.ndarray:
    """Sample circular sections normal to the local centreline tangent."""

    rings = []
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    for point, tangent in zip(path, tangent_vectors(path), strict=True):
        tangent = tangent / max(np.linalg.norm(tangent), 1e-12)
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(tangent, reference))) > 0.9:
            reference = np.array([1.0, 0.0, 0.0])
        basis_u = np.cross(tangent, reference)
        basis_u /= max(np.linalg.norm(basis_u), 1e-12)
        basis_v = np.cross(tangent, basis_u)
        offsets = radius * (
            np.cos(angles)[:, None] * basis_u[None, :]
            + np.sin(angles)[:, None] * basis_v[None, :]
        )
        rings.append(point + offsets)
    return np.vstack(rings)


def test_traits_preserve_directional_angles_and_vector_coordinates():
    primary = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, -1.0],
        ]
    )
    lateral_points = np.array(
        [
            [0.0, 0.000, 0.000],
            [0.0, 0.250, -0.250],
            [0.0, 0.500, -0.500],
            [0.0, 0.625, -0.375],
            [0.0, 0.750, -0.250],
            [0.0, 0.875, -0.125],
            [0.0, 1.000, 0.000],
        ]
    )
    lateral = RootPath(
        root_id="root-o1-001",
        points=lateral_points,
        order=1,
        parent_id="primary",
        parent_points=primary,
        insertion_point=lateral_points[0].copy(),
        insertion_index=2,
        confidence=0.9,
    )
    primary_surface = _surface_samples(primary, radius=0.02)
    lateral_surface = _surface_samples(lateral_points, radius=0.01)
    points = np.vstack([primary_surface, lateral_surface])
    primary_mask = np.zeros(len(points), dtype=bool)
    primary_mask[: len(primary_surface)] = True
    lateral_labels = np.zeros(len(points), dtype=int)
    lateral_labels[len(primary_surface) :] = 1

    traits = compute_traits(
        primary,
        [lateral],
        points,
        primary_mask,
        lateral_labels,
        Normalization(minimum=np.zeros(3), scale=1.0),
        lateral_start_count=1,
    )

    required_columns = {
        "tortuosity",
        "mean_diameter",
        "median_diameter",
        "minimum_diameter",
        "maximum_diameter",
        "root_start_x",
        "root_start_y",
        "root_start_z",
        "root_tip_x",
        "root_tip_y",
        "root_tip_z",
        "gravity_dx",
        "gravity_dy",
        "gravity_dz",
    }
    for prefix in ("tip_vector", "tip_start_vector", "base_vector", "primary_vector"):
        required_columns.update(
            {
                f"{prefix}_start_x",
                f"{prefix}_start_y",
                f"{prefix}_start_z",
                f"{prefix}_end_x",
                f"{prefix}_end_y",
                f"{prefix}_end_z",
                f"{prefix}_dx",
                f"{prefix}_dy",
                f"{prefix}_dz",
            }
        )
    assert required_columns.issubset(traits.columns)

    row = traits.loc[traits["root_id"] == lateral.root_id].iloc[0]
    angle_columns = [
        "tip_gravity_angle_deg",
        "tip_start_gravity_angle_deg",
        "tip_primary_angle_deg",
    ]
    assert np.all(np.isfinite(row[angle_columns].to_numpy(dtype=float)))
    assert np.all((row[angle_columns].to_numpy(dtype=float) >= 0.0))
    assert np.all((row[angle_columns].to_numpy(dtype=float) <= 180.0))
    assert row["tip_gravity_angle_deg"] == pytest.approx(135.0)
    assert row["tip_start_gravity_angle_deg"] == pytest.approx(90.0)
    # The lateral starts downwards at 45° to the downward primary tangent but
    # bends upwards at its tip.  This distinguishes the requested start-based
    # primary angle from the former 135° tip-based definition.
    assert row["tip_primary_angle_deg"] == pytest.approx(45.0)
    assert row["tip_angle_primary_deg"] == pytest.approx(45.0)
    assert row["tip_angle_parent_deg"] == pytest.approx(135.0)
    assert row["tortuosity"] == pytest.approx(np.sqrt(2.0))
    assert row["mean_diameter"] == pytest.approx(0.02)
    assert row["length_unit"] == "mesh_unit"
    assert row["area_unit"] == "mesh_unit^2"
    assert row["volume_unit"] == "mesh_unit^3"
    assert row["coordinate_unit"] == "mesh_unit"
    assert not any("_mm" in column for column in traits.columns)

    np.testing.assert_allclose(
        row[["tip_vector_dx", "tip_vector_dy", "tip_vector_dz"]].to_numpy(dtype=float),
        [0.0, 0.25, 0.25],
    )
    np.testing.assert_allclose(
        row[
            ["tip_start_vector_dx", "tip_start_vector_dy", "tip_start_vector_dz"]
        ].to_numpy(dtype=float),
        [0.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(
        row[["base_vector_dx", "base_vector_dy", "base_vector_dz"]].to_numpy(dtype=float),
        [0.0, 0.25, -0.25],
    )
    np.testing.assert_allclose(
        row[["primary_vector_dx", "primary_vector_dy", "primary_vector_dz"]].to_numpy(dtype=float),
        [0.0, 0.0, -0.5],
    )
    np.testing.assert_allclose(
        row[["gravity_dx", "gravity_dy", "gravity_dz"]].to_numpy(dtype=float),
        GRAVITY,
    )


def test_traits_are_reported_in_denormalized_source_mesh_units():
    primary = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0]])
    points = _surface_samples(primary, radius=0.02)
    full_points = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0], [0.0, 3.0, 0.0]]
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    traits = compute_traits(
        primary,
        [],
        points,
        np.ones(len(points), dtype=bool),
        np.zeros(len(points), dtype=int),
        Normalization(minimum=np.array([5.0, -2.0, 7.0]), scale=10.0),
        full_points=full_points,
        triangles=triangles,
        full_root_labels=np.zeros(len(full_points), dtype=int),
        mesh_metadata={"surface_area_source_units2": 6.0},
    )

    row = traits.iloc[0]
    assert row["length"] == pytest.approx(10.0)
    assert row["chord_length"] == pytest.approx(10.0)
    assert row["surface_area"] == pytest.approx(6.0)
    np.testing.assert_allclose(
        row[["root_start_x", "root_start_y", "root_start_z"]].to_numpy(dtype=float),
        [5.0, -2.0, 7.0],
    )
    np.testing.assert_allclose(
        row[["root_tip_x", "root_tip_y", "root_tip_z"]].to_numpy(dtype=float),
        [5.0, -2.0, 17.0],
    )
    assert row["coordinate_unit"] == "mesh_unit"
    assert not any("mm" in column.lower() for column in traits.columns)
    assert traits.attrs["system_summary"]["root_system_surface_area"] == pytest.approx(6.0)


def test_mesh_unit_tip_vector_is_invariant_to_polyline_sampling_density():
    primary = np.array(
        [
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 8.0],
            [0.0, 0.0, 6.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
        ]
    )
    sparse = np.array(
        [
            [0.0, 0.0, 6.0],
            [2.0, 0.0, 5.0],
            [4.0, 1.0, 4.0],
            [5.0, 3.0, 3.0],
            [7.0, 6.0, 1.0],
        ]
    )
    dense = np.vstack(
        [
            *(np.linspace(start, end, 9, endpoint=False) for start, end in zip(sparse[:-1], sparse[1:], strict=True)),
            sparse[-1][None, :],
        ]
    )

    def lateral_traits(path: np.ndarray) -> np.ndarray:
        lateral = RootPath(
            root_id="root-o1-001",
            points=path,
            order=1,
            parent_id="primary",
            parent_points=primary,
            insertion_point=path[0].copy(),
            insertion_index=2,
            confidence=0.9,
        )
        primary_surface = _surface_samples(primary, radius=0.05)
        lateral_surface = _surface_samples(path, radius=0.03)
        points = np.vstack([primary_surface, lateral_surface])
        primary_mask = np.zeros(len(points), dtype=bool)
        primary_mask[: len(primary_surface)] = True
        lateral_labels = np.zeros(len(points), dtype=int)
        lateral_labels[len(primary_surface) :] = 1
        traits = compute_traits(
            primary,
            [lateral],
            points,
            primary_mask,
            lateral_labels,
            Normalization(minimum=np.zeros(3), scale=1.0),
            tip_vector_window=2.0,
        )
        return traits.loc[traits["root_id"] == lateral.root_id].iloc[0]

    sparse_row = lateral_traits(sparse)
    dense_row = lateral_traits(dense)
    vector_columns = ["tip_vector_dx", "tip_vector_dy", "tip_vector_dz"]
    sparse_vector = sparse_row[vector_columns].to_numpy(dtype=float)
    dense_vector = dense_row[vector_columns].to_numpy(dtype=float)
    expected = (sparse[-1] - sparse[-2]) / np.linalg.norm(sparse[-1] - sparse[-2]) * 2.0

    np.testing.assert_allclose(sparse_vector, expected, atol=1e-12)
    np.testing.assert_allclose(dense_vector, expected, atol=1e-12)
    np.testing.assert_allclose(sparse_vector, dense_vector, atol=1e-12)
    assert sparse_row["tip_vector_arc_window"] == pytest.approx(2.0)
    assert dense_row["tip_vector_arc_window"] == pytest.approx(2.0)
    assert sparse_row["tip_vector_window_unit"] == "mesh_unit"
    assert sparse_row["tip_gravity_angle_deg"] == pytest.approx(dense_row["tip_gravity_angle_deg"])
