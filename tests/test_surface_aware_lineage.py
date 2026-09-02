from __future__ import annotations

import numpy as np

import soyrootbio.lateral as lateral_module
from soyrootbio.lateral import (
    find_lateral_starting_points,
    is_ancestor_inward_candidate,
)
from soyrootbio.topology import (
    _ContactedSiblingCrop,
    _crop_contacted_primary_sibling_suffixes,
    _is_parallel_parent_duplicate,
    _join_contacted_sibling_continuations,
    _merge_parallel_parent_duplicates,
    _merge_same_insertion_primary_duplicates,
    _reparent_same_insertion_divergences,
    _swap_internal_contact_suffixes,
    repair_root_hierarchy,
    uncross_internal_primary_sibling_contacts,
    validate_root_tree,
)
from soyrootbio.types import RootPath


def _root(
    root_id: str,
    points: list[list[float]],
    *,
    order: int,
    parent_id: str,
) -> RootPath:
    polyline = np.asarray(points, dtype=float)
    return RootPath(
        root_id=root_id,
        points=polyline,
        raw_start_point=polyline[0].copy(),
        order=order,
        parent_id=parent_id,
        covered_indices=set(range(20)),
        novel_support_indices=set(range(20)),
        score=20.0,
    )


def _contact_test_path(
    root_id: str,
    prefix: np.ndarray,
    suffix: np.ndarray,
    *,
    node_offset: int,
) -> RootPath:
    points = np.vstack([prefix, suffix[1:]])
    path = _root(
        root_id,
        points.tolist(),
        order=1,
        parent_id="primary",
    )
    path.insertion_point = points[0].copy()
    path.insertion_index = node_offset
    path.node_indices = np.arange(
        node_offset,
        node_offset + len(points),
        dtype=int,
    )
    path.score_components.update(
        {
            "surface_aware_seed": 1.0,
            "primary_surface_attachment": 1.0,
        }
    )
    return path


def _owned_tube_surface(
    path: np.ndarray,
    radii: np.ndarray,
    *,
    samples_per_station: int = 16,
) -> np.ndarray:
    points = np.asarray(path, dtype=float)
    tangents = np.empty_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents[1:-1] = points[2:] - points[:-2]
    tangents /= np.maximum(
        np.linalg.norm(tangents, axis=1, keepdims=True),
        1e-12,
    )
    rings = []
    for point, tangent, radius in zip(points, tangents, radii, strict=True):
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(tangent, reference))) > 0.90:
            reference = np.array([0.0, 1.0, 0.0])
        first_normal = np.cross(tangent, reference)
        first_normal /= np.linalg.norm(first_normal)
        second_normal = np.cross(tangent, first_normal)
        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            samples_per_station,
            endpoint=False,
        )
        rings.append(
            point
            + float(radius)
            * (
                np.cos(angles)[:, None] * first_normal
                + np.sin(angles)[:, None] * second_normal
            )
        )
    return np.vstack(rings)


def _contact_surface_fixture(
    *,
    equal_radii: bool = False,
    extreme_crossed_turn: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[RootPath]]:
    station = np.linspace(-10.0, 0.0, 21)
    left_prefix = np.column_stack(
        [station, np.zeros_like(station), np.zeros_like(station)]
    )
    left_suffix = np.column_stack(
        [
            np.linspace(0.0, 10.0, 21),
            np.zeros(21),
            np.zeros(21),
        ]
    )
    right_prefix = np.column_stack(
        [
            np.zeros(21),
            station,
            np.zeros_like(station),
        ]
    )
    if extreme_crossed_turn:
        distance = np.linspace(0.0, 10.0, 21)
        angle = np.radians(150.0)
        right_suffix = np.column_stack(
            [
                distance * np.cos(angle),
                distance * np.sin(angle),
                np.zeros_like(distance),
            ]
        )
    else:
        right_suffix = np.column_stack(
            [
                np.zeros(21),
                np.linspace(0.0, 10.0, 21),
                np.zeros(21),
            ]
        )
    left = _contact_test_path(
        "left",
        left_prefix,
        left_suffix,
        node_offset=100,
    )
    right = _contact_test_path(
        "right",
        right_prefix,
        right_suffix,
        node_offset=1000,
    )
    if equal_radii:
        left_radii = np.full(len(left.points), 0.40)
        right_radii = np.full(len(right.points), 0.40)
    else:
        contact_index = len(left_prefix) - 1
        left_radii = np.where(
            np.arange(len(left.points)) <= contact_index,
            0.60,
            0.20,
        )
        right_radii = np.where(
            np.arange(len(right.points)) <= contact_index,
            0.20,
            0.60,
        )
    left_surface = _owned_tube_surface(left.points, left_radii)
    right_surface = _owned_tube_surface(right.points, right_radii)
    surface = np.vstack([left_surface, right_surface])
    labels = np.concatenate(
        [
            np.ones(len(left_surface), dtype=int),
            np.full(len(right_surface), 2, dtype=int),
        ]
    )
    return surface, labels, [left, right]


def test_surface_contact_seed_beats_centerline_nearest_singleton(
    monkeypatch,
) -> None:
    primary_path = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0],
        ]
    )
    primary_surface = np.array(
        [
            [1.0, 0.0, 0.40],
            [1.0, 0.0, 0.41],
            [1.0, 0.0, 0.42],
        ]
    )
    root_contact = np.array(
        [
            [1.05, 0.00, 0.40],
            [1.05, 0.01, 0.41],
            [1.05, -0.01, 0.42],
        ]
    )
    centerline_nearest_noise = np.array([[0.20, 0.0, 0.41]])
    points = np.vstack(
        [primary_surface, root_contact, centerline_nearest_noise]
    )
    occupied = np.zeros(len(points), dtype=bool)
    occupied[: len(primary_surface)] = True
    monkeypatch.setattr(
        lateral_module,
        "cluster_hdbscan",
        lambda values, min_cluster_size: np.zeros(len(values), dtype=int),
    )

    starts = find_lateral_starting_points(
        points,
        occupied,
        primary_path,
        min_cluster_size=4,
        max_parent_distance=0.40,
        parent_surface_points=primary_surface,
        surface_contact_distance=0.10,
    )

    assert len(starts) == 1
    assert starts[0].surface_contact
    assert starts[0].surface_contact_count == 3
    assert starts[0].surface_gap < 0.06
    assert starts[0].point[0] > 1.0


def test_endpoint_primary_surface_contact_overrides_contacted_lateral() -> None:
    primary = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
        ]
    )
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    primary_surface = np.vstack(
        [
            np.column_stack(
                [
                    np.cos(angles),
                    np.sin(angles),
                    np.full_like(angles, z),
                ]
            )
            for z in (0.45, 0.50, 0.55)
        ]
    )
    contacted = _root(
        "contacted",
        [[0.0, 0.0, 0.5], [1.5, 0.0, 0.5], [2.5, 0.0, 0.5]],
        order=1,
        parent_id="primary",
    )
    primary_touching_fragment = _root(
        "primary-touching-fragment",
        [[2.0, 0.0, 0.5], [1.5, 0.0, 0.5], [1.05, 0.0, 0.5]],
        order=2,
        parent_id="contacted",
    )
    distal_child = _root(
        "distal-child",
        [[2.5, 0.0, 0.5], [2.5, 0.4, 0.5], [2.5, 0.8, 0.5]],
        order=2,
        parent_id="contacted",
    )

    repaired, _ = repair_root_hierarchy(
        primary,
        [contacted, primary_touching_fragment, distal_child],
        d_bar=0.05,
        primary_surface_points=primary_surface,
    )

    assert validate_root_tree(repaired) == []
    assert primary_touching_fragment.parent_id == "primary"
    assert primary_touching_fragment.order == 1
    assert (
        primary_touching_fragment.score_components[
            "primary_surface_attachment"
        ]
        == 1.0
    )
    assert distal_child.parent_id == contacted.root_id
    assert distal_child.order == 2


def test_ancestor_inward_rule_is_directional() -> None:
    z = np.linspace(1.0, 0.0, 21)
    primary = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    radius = np.ones(len(primary), dtype=float)
    inward = _root(
        "inward",
        [
            [2.0, 0.0, 0.5],
            [1.5, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [0.8, 0.0, 0.5],
        ],
        order=2,
        parent_id="lateral",
    )
    outward = _root(
        "outward",
        [
            [0.8, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [1.5, 0.0, 0.5],
            [2.0, 0.0, 0.5],
        ],
        order=1,
        parent_id="primary",
    )
    distal = _root(
        "distal",
        [
            [2.0, 0.0, 0.5],
            [2.2, 0.0, 0.4],
            [2.4, 0.0, 0.3],
        ],
        order=2,
        parent_id="lateral",
    )

    rejected, metrics = is_ancestor_inward_candidate(
        inward,
        primary,
        radius,
        d_bar=0.10,
    )
    assert rejected
    assert metrics["ancestor_terminal_inside_fraction"] >= 0.75
    assert not is_ancestor_inward_candidate(
        outward,
        primary,
        radius,
        d_bar=0.10,
    )[0]
    assert not is_ancestor_inward_candidate(
        distal,
        primary,
        radius,
        d_bar=0.10,
    )[0]


def test_contacted_sibling_suffix_is_cropped_and_joined_to_child_arm() -> None:
    host_points = np.column_stack(
        [
            np.linspace(0.0, 15.0, 31),
            np.zeros(31),
            np.zeros(31),
        ]
    )
    contacted_points = np.array(
        [
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [5.0, 2.0, 0.0],
            [5.0, 1.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    child_points = np.array(
        [
            [6.0, 0.0, 0.0],
            [6.0, -1.0, 0.0],
            [6.0, -2.0, 0.0],
            [7.0, -3.0, 0.0],
        ]
    )
    host = _root(
        "host",
        host_points.tolist(),
        order=1,
        parent_id="primary",
    )
    contacted = _root(
        "contacted",
        contacted_points.tolist(),
        order=1,
        parent_id="primary",
    )
    continuation = _root(
        "continuation",
        child_points.tolist(),
        order=2,
        parent_id="contacted",
    )
    continuation.insertion_index = 8
    contacted.score_components.update(
        {
            "surface_aware_seed": 1.0,
            "trace_local_radius": 0.20,
        }
    )
    continuation.score_components["trace_local_radius"] = 0.22

    cropped = _crop_contacted_primary_sibling_suffixes(
        [host, contacted, continuation],
        d_bar=0.10,
    )

    assert set(cropped) == {"contacted"}
    assert cropped["contacted"].host_id == "host"
    assert cropped["contacted"].host_index == 10
    np.testing.assert_allclose(contacted.points[-1], [5.0, 0.0, 0.0])
    assert (
        contacted.score_components[
            "contacted_sibling_suffix_cropped"
        ]
        == 1.0
    )

    joined, reassigned = _join_contacted_sibling_continuations(
        [host, contacted, continuation],
        cropped,
        d_bar=0.10,
    )

    assert {path.root_id for path in joined} == {"host", "contacted"}
    assert reassigned == {"contacted"}
    np.testing.assert_allclose(contacted.points[-1], child_points[-1])
    assert (
        contacted.score_components[
            "contacted_sibling_continuation_joined"
        ]
        == 1.0
    )
    assert (
        contacted.score_components[
            "contacted_sibling_direct_turn_degrees"
        ]
        < 1e-9
    )
    assert (
        contacted.score_components[
            "contacted_sibling_radius_similarity"
        ]
        > 0.90
    )


def test_contact_join_rejects_exact_contact_direction_and_radius_distractors() -> None:
    host_points = np.column_stack(
        [
            np.linspace(0.0, 15.0, 31),
            np.zeros(31),
            np.zeros(31),
        ]
    )
    contacted_points = np.array(
        [
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [5.0, 2.0, 0.0],
            [5.0, 1.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    host = _root(
        "host",
        host_points.tolist(),
        order=1,
        parent_id="primary",
    )
    contacted = _root(
        "contacted",
        contacted_points.tolist(),
        order=1,
        parent_id="primary",
    )
    desired = _root(
        "desired",
        [
            [6.0, 0.0, 0.0],
            [6.0, -1.0, 0.0],
            [6.0, -2.0, 0.0],
            [7.0, -3.0, 0.0],
        ],
        order=2,
        parent_id="contacted",
    )
    desired.insertion_index = 8
    wrong_direction = _root(
        "wrong-direction",
        [
            [5.0, 0.0, 0.0],
            [5.0, 2.0, 0.0],
            [5.0, 4.0, 0.0],
            [5.0, 6.0, 0.0],
        ],
        order=2,
        parent_id="host",
    )
    wrong_direction.insertion_index = 10
    wrong_radius = _root(
        "wrong-radius",
        [
            [5.0, 0.0, 0.0],
            [5.0, -2.0, 0.0],
            [5.0, -4.0, 0.0],
            [5.0, -6.0, 0.0],
        ],
        order=2,
        parent_id="host",
    )
    wrong_radius.insertion_index = 10
    contacted.score_components.update(
        {
            "surface_aware_seed": 1.0,
            "trace_local_radius": 0.20,
        }
    )
    desired.score_components["trace_local_radius"] = 0.20
    wrong_direction.score_components["trace_local_radius"] = 0.20
    wrong_radius.score_components["trace_local_radius"] = 1.00

    paths = [
        host,
        contacted,
        desired,
        wrong_direction,
        wrong_radius,
    ]
    cropped = _crop_contacted_primary_sibling_suffixes(
        paths,
        d_bar=0.10,
    )
    joined, reassigned = _join_contacted_sibling_continuations(
        paths,
        cropped,
        d_bar=0.10,
    )

    assert {path.root_id for path in joined} == {
        "host",
        "contacted",
        "wrong-direction",
        "wrong-radius",
    }
    assert reassigned == {"contacted"}
    np.testing.assert_allclose(contacted.points[-1], [7.0, -3.0, 0.0])
    assert wrong_direction.parent_id == "host"
    assert wrong_radius.parent_id == "host"


def test_same_insertion_short_o1_duplicate_is_absorbed() -> None:
    longer = _root(
        "longer",
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        order=1,
        parent_id="primary",
    )
    shorter = _root(
        "shorter",
        [
            [0.1, 0.0, 0.0],
            [0.8, 0.1, 0.0],
            [1.5, 0.0, 0.0],
        ],
        order=1,
        parent_id="primary",
    )
    child = _root(
        "child",
        [[1.5, 0.0, 0.0], [1.5, 1.0, 0.0], [1.5, 2.0, 0.0]],
        order=2,
        parent_id="shorter",
    )

    retained, reassigned = _merge_same_insertion_primary_duplicates(
        [longer, shorter, child],
        d_bar=0.10,
    )

    assert {path.root_id for path in retained} == {"longer", "child"}
    assert child.parent_id == "longer"
    assert reassigned == {"longer", "child"}
    assert (
        longer.score_components[
            "same_insertion_o1_duplicates_merged"
        ]
        == 1.0
    )


def test_contact_crop_is_restored_without_unique_continuation() -> None:
    host = _root(
        "host",
        np.column_stack(
            [
                np.linspace(0.0, 15.0, 31),
                np.zeros(31),
                np.zeros(31),
            ]
        ).tolist(),
        order=1,
        parent_id="primary",
    )
    contacted = _root(
        "contacted",
        [
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [5.0, 2.0, 0.0],
            [5.0, 1.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        order=1,
        parent_id="primary",
    )
    contacted.score_components["surface_aware_seed"] = 1.0
    original = contacted.points.copy()

    cropped = _crop_contacted_primary_sibling_suffixes(
        [host, contacted],
        d_bar=0.10,
    )
    assert set(cropped) == {"contacted"}

    retained, reassigned = _join_contacted_sibling_continuations(
        [host, contacted],
        cropped,
        d_bar=0.10,
    )

    assert retained == [host, contacted]
    assert reassigned == set()
    np.testing.assert_allclose(contacted.points, original)
    assert "contacted_sibling_suffix_cropped" not in contacted.qc_flags


def test_chained_contact_crops_restore_without_consuming_pending_roles() -> None:
    first = _root(
        "first",
        [
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [5.0, 2.0, 0.0],
            [5.0, 1.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ],
        order=1,
        parent_id="third",
    )
    second = _root(
        "second",
        [
            [6.0, 0.0, 0.0],
            [6.0, 1.0, 0.0],
            [6.0, 2.0, 0.0],
            [7.0, 3.0, 0.0],
            [8.0, 4.0, 0.0],
        ],
        order=2,
        parent_id="first",
    )
    third = _root(
        "third",
        np.column_stack(
            [
                np.linspace(0.0, 15.0, 31),
                np.zeros(31),
                np.zeros(31),
            ]
        ).tolist(),
        order=1,
        parent_id="primary",
    )
    originals = {
        root.root_id: root.points.copy()
        for root in (first, second, third)
    }

    def provisional_crop(
        root: RootPath,
        *,
        host_id: str,
        host_index: int,
        crop_index: int,
    ) -> _ContactedSiblingCrop:
        record = _ContactedSiblingCrop(
            host_id=host_id,
            host_index=host_index,
            original_points=root.points.copy(),
            original_node_indices=None,
            original_score_components=dict(root.score_components),
            original_qc_flags=list(root.qc_flags),
        )
        root.points = root.points[: crop_index + 1].copy()
        root.score_components.update(
            {
                "surface_aware_seed": 1.0,
                "contacted_sibling_suffix_cropped": 1.0,
                "contacted_sibling_crop_index": float(crop_index),
                "contacted_sibling_host_direction": 1.0,
            }
        )
        root.qc_flags.append("contacted_sibling_suffix_cropped")
        return record

    crops = {
        "first": provisional_crop(
            first,
            host_id="third",
            host_index=10,
            crop_index=7,
        ),
        "second": provisional_crop(
            second,
            host_id="third",
            host_index=12,
            crop_index=3,
        ),
        "third": provisional_crop(
            third,
            host_id="second",
            host_index=0,
            crop_index=20,
        ),
    }
    second.insertion_index = 8

    retained, reassigned = _join_contacted_sibling_continuations(
        [first, second, third],
        crops,
        d_bar=0.10,
    )

    assert [root.root_id for root in retained] == [
        "first",
        "second",
        "third",
    ]
    assert reassigned == set()
    for root in retained:
        np.testing.assert_allclose(root.points, originals[root.root_id])
        assert "contacted_sibling_suffix_cropped" not in root.qc_flags


def test_same_insertion_thinner_divergence_becomes_child() -> None:
    host = _root(
        "host",
        np.column_stack(
            [
                np.linspace(0.0, 10.0, 21),
                np.zeros(21),
                np.zeros(21),
            ]
        ).tolist(),
        order=1,
        parent_id="primary",
    )
    branch_points = np.vstack(
        [
            np.column_stack(
                [
                    np.linspace(0.0, 5.0, 11),
                    np.zeros(11),
                    np.zeros(11),
                ]
            ),
            np.column_stack(
                [
                    np.full(10, 5.0),
                    np.linspace(0.5, 5.0, 10),
                    np.zeros(10),
                ]
            ),
        ]
    )
    branch = _root(
        "branch",
        branch_points.tolist(),
        order=1,
        parent_id="primary",
    )
    host.confidence = 0.90
    branch.confidence = 0.70
    host.score_components.update(
        {
            "trace_local_radius": 0.40,
            "trace_radius_similarity": 0.92,
        }
    )
    branch.score_components.update(
        {
            "trace_local_radius": 0.20,
            "trace_radius_similarity": 0.84,
        }
    )

    retained, reassigned = _reparent_same_insertion_divergences(
        [host, branch],
        d_bar=0.10,
    )

    assert retained == [host, branch]
    assert reassigned == {"branch"}
    assert branch.parent_id == "host"
    np.testing.assert_allclose(branch.points[0], [5.0, 0.0, 0.0])
    np.testing.assert_allclose(branch.points[-1], [5.0, 5.0, 0.0])
    assert (
        branch.score_components[
            "same_insertion_divergence_reparented"
        ]
        == 1.0
    )


def test_same_insertion_similar_radius_siblings_remain_o1() -> None:
    first = _root(
        "first",
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 2.0, 0.0]],
        order=1,
        parent_id="primary",
    )
    second = _root(
        "second",
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, -2.0, 0.0]],
        order=1,
        parent_id="primary",
    )
    first.confidence = second.confidence = 0.90
    first.score_components.update(
        {
            "trace_local_radius": 0.40,
            "trace_radius_similarity": 0.90,
        }
    )
    second.score_components.update(
        {
            "trace_local_radius": 0.38,
            "trace_radius_similarity": 0.90,
        }
    )

    retained, reassigned = _reparent_same_insertion_divergences(
        [first, second],
        d_bar=0.10,
    )

    assert retained == [first, second]
    assert reassigned == set()
    assert all(path.parent_id == "primary" for path in retained)


def test_brief_primary_sibling_crossing_is_not_cropped() -> None:
    host = _root(
        "host",
        np.column_stack(
            [
                np.linspace(0.0, 10.0, 21),
                np.zeros(21),
                np.zeros(21),
            ]
        ).tolist(),
        order=1,
        parent_id="primary",
    )
    crossing = _root(
        "crossing",
        [
            [5.0, -4.0, 0.0],
            [5.0, -2.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 2.0, 0.0],
            [5.0, 4.0, 0.0],
        ],
        order=1,
        parent_id="primary",
    )
    original = crossing.points.copy()

    cropped = _crop_contacted_primary_sibling_suffixes(
        [host, crossing],
        d_bar=0.10,
    )

    assert cropped == {}
    np.testing.assert_allclose(crossing.points, original)


def test_internal_primary_sibling_contact_uncrosses_with_owned_radius_evidence() -> None:
    surface, labels, paths = _contact_surface_fixture()
    left, right = paths
    original_left_tip = left.points[-1].copy()
    original_right_tip = right.points[-1].copy()
    child = _root(
        "child",
        [
            [7.0, 0.0, 0.0],
            [7.0, 0.0, 1.0],
            [7.0, 0.0, 2.0],
        ],
        order=2,
        parent_id="left",
    )
    child.raw_start_point = child.points[0].copy()
    paths.append(child)

    changed, diagnostics = uncross_internal_primary_sibling_contacts(
        surface,
        labels,
        paths,
        d_bar=0.10,
    )

    assert changed == {"left", "right"}
    np.testing.assert_allclose(left.points[-1], original_right_tip)
    np.testing.assert_allclose(right.points[-1], original_left_tip)
    assert len(left.node_indices) == len(left.points)
    assert len(right.node_indices) == len(right.points)
    assert left.node_indices[-1] >= 1000
    assert right.node_indices[-1] < 1000
    assert child.parent_id == "right"
    assert "internal_o1_contact_child_reattached" in child.qc_flags
    assert all(
        "internal_o1_contact_uncrossed" in path.qc_flags
        for path in (left, right)
    )
    assert diagnostics[0]["action"] == "uncrossed"
    assert diagnostics[0]["swapped_scale_wins"] >= 2
    assert diagnostics[0]["mean_radius_margin"] >= 0.08
    assert diagnostics[0]["radius_scales_d_bar"] == [6.0, 10.0, 20.0]


def test_internal_primary_sibling_contact_keeps_ambiguous_radius_pairing() -> None:
    surface, labels, paths = _contact_surface_fixture(equal_radii=True)
    originals = [path.points.copy() for path in paths]

    changed, diagnostics = uncross_internal_primary_sibling_contacts(
        surface,
        labels,
        paths,
        d_bar=0.10,
    )

    assert changed == set()
    for path, original in zip(paths, originals, strict=True):
        np.testing.assert_allclose(path.points, original)
        assert "internal_o1_contact_ambiguous" in path.qc_flags
    assert diagnostics[0]["action"] == "ambiguous"
    assert diagnostics[0]["reason"] == "radius_pairing_margin"


def test_internal_primary_sibling_contact_respects_extreme_tangent_veto() -> None:
    surface, labels, paths = _contact_surface_fixture(
        extreme_crossed_turn=True,
    )
    originals = [path.points.copy() for path in paths]

    changed, diagnostics = uncross_internal_primary_sibling_contacts(
        surface,
        labels,
        paths,
        d_bar=0.10,
    )

    assert changed == set()
    for path, original in zip(paths, originals, strict=True):
        np.testing.assert_allclose(path.points, original)
        assert "internal_o1_contact_ambiguous" in path.qc_flags
    assert diagnostics[0]["action"] == "ambiguous"
    assert diagnostics[0]["reason"] == "crossed_tangent_veto"
    assert max(diagnostics[0]["crossed_tangent_turn_degrees"]) > 135.0


def test_internal_contact_suffix_dedup_keeps_node_indices_aligned() -> None:
    left = _root(
        "left",
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        order=1,
        parent_id="primary",
    )
    right = _root(
        "right",
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        order=1,
        parent_id="primary",
    )
    left.node_indices = np.array([10, 11, 12])
    right.node_indices = np.array([20, 21, 22, 23])

    _swap_internal_contact_suffixes(left, 1, right, 1)

    assert len(left.node_indices) == len(left.points)
    assert len(right.node_indices) == len(right.points)
    np.testing.assert_array_equal(left.node_indices, [10, 11, 23])
    np.testing.assert_array_equal(right.node_indices, [20, 21, 12])


def test_parallel_parent_duplicate_with_shared_support_is_absorbed() -> None:
    parent = _root(
        "parent",
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        order=1,
        parent_id="primary",
    )
    duplicate = _root(
        "duplicate",
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.06, 0.0],
            [3.0, 0.05, 0.0],
            [4.0, 0.06, 0.0],
        ],
        order=2,
        parent_id="parent",
    )
    parent.covered_indices = set(range(100, 140))
    duplicate.covered_indices = set(range(110, 130))

    assert _is_parallel_parent_duplicate(
        duplicate,
        parent,
        d_bar=0.01,
    )
    retained, _ = _merge_parallel_parent_duplicates(
        [parent, duplicate],
        d_bar=0.01,
    )

    assert retained == [parent]
    assert parent.covered_indices == set(range(100, 140))
    assert parent.score_components["parallel_child_duplicates_merged"] == 1.0


def test_near_coincident_duplicate_without_shared_support_is_recognized() -> None:
    parent = _root(
        "parent",
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        order=1,
        parent_id="primary",
    )
    duplicate = _root(
        "duplicate",
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.01, 0.0],
            [3.0, 0.01, 0.0],
            [4.0, 0.01, 0.0],
        ],
        order=2,
        parent_id="parent",
    )
    parent.covered_indices = set(range(100, 120))
    duplicate.covered_indices = set(range(200, 220))

    assert _is_parallel_parent_duplicate(
        duplicate,
        parent,
        d_bar=0.01,
    )


def test_parallel_but_distinct_shallow_child_is_retained() -> None:
    parent = _root(
        "parent",
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        order=1,
        parent_id="primary",
    )
    shallow_child = _root(
        "shallow-child",
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.04, 0.0],
            [3.0, 0.07, 0.0],
            [4.0, 0.10, 0.0],
        ],
        order=2,
        parent_id="parent",
    )
    parent.covered_indices = set(range(100, 140))
    shallow_child.covered_indices = set(range(200, 220))

    assert not _is_parallel_parent_duplicate(
        shallow_child,
        parent,
        d_bar=0.01,
    )
    retained, reassigned = _merge_parallel_parent_duplicates(
        [parent, shallow_child],
        d_bar=0.01,
    )

    assert retained == [parent, shallow_child]
    assert reassigned == set()
    assert shallow_child.parent_id == "parent"
