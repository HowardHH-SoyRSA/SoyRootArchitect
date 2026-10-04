from __future__ import annotations

import hashlib

import numpy as np
import pytest
from scipy.spatial import cKDTree

from soyrootbio.mesh_geometry import MeshGeometryContext
from soyrootbio.topology import (
    _join_displaced_tip_continuations,
    _reconcile_overlong_forks,
    _terminal_mesh_connection,
    repair_root_hierarchy,
    validate_root_tree,
)
from soyrootbio.types import RootPath


def _tube_fixture(*, branch: bool = False, disconnect: bool = False):
    centers = np.array(
        [[0.5 * index, 0.0, 0.0] for index in range(41)]
        + [
            ([20.0, 0.5 * index, 0.0] if branch else [20.0 + 0.5 * index, 0.05 * index, 0.0])
            for index in range(1, 17)
        ],
        dtype=float,
    )
    rings = []
    for center in centers:
        angle = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        if branch and center[1] > 0.0:
            ring = np.column_stack((0.30 * np.cos(angle),
                                    np.zeros(16), 0.30 * np.sin(angle)))
        else:
            ring = np.column_stack((np.zeros(16), 0.30 * np.cos(angle),
                                    0.30 * np.sin(angle)))
        rings.extend(center + ring)
    mesh = np.asarray(rings, dtype=float)
    triangles = []
    for station in range(len(centers) - 1):
        if disconnect and station == 40:
            continue
        for index in range(16):
            following = (index + 1) % 16
            a = 16 * station + index
            b = 16 * station + following
            c = 16 * (station + 1) + index
            d = 16 * (station + 1) + following
            triangles.extend(((a, b, c), (b, d, c)))
    parent = RootPath(
        root_id="parent", points=centers[:41].copy(), order=1,
        parent_id="primary", covered_indices=set(range(41 * 16)),
        qc_flags=["tip_extension_limit"],
        score_components={"surface_aware_seed": 1.0},
    )
    child = RootPath(
        root_id="child", points=centers[40:].copy(), order=2,
        parent_id="parent", insertion_index=40,
        insertion_point=centers[40].copy(),
        covered_indices=set(range(40 * 16, len(mesh))),
        novel_support_indices=set(range(41 * 16, len(mesh))),
        score_components={"novel_density_support": 16.0 * 16.0},
    )
    return parent, child, mesh, np.asarray(triangles, dtype=int)


def _run(parent, child, mesh, triangles, *, extra=(), accepted_attachment=False, primary_length=40.0):
    paths = [parent, child, *extra]
    decisions = []
    stats = {}
    _reconcile_overlong_forks(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, primary_length]]), paths,
        d_bar=0.15, support_points=mesh, mesh_points=mesh,
        mesh_triangles=triangles, terminal_decision_log=decisions,
        queue_stats=stats,
        attachment_status_by_root={"child": "accepted"} if accepted_attachment else None,
    )
    return paths, decisions, stats


@pytest.mark.parametrize("disconnect", [False, True])
@pytest.mark.parametrize("exclude_joint", [False, True])
def test_terminal_native_connection_matches_cached_geometry(disconnect, exclude_joint):
    parent, child, mesh, triangles = _tube_fixture(disconnect=disconnect)
    context = MeshGeometryContext.build(mesh, triangles)
    excluded = (np.arange(len(mesh)) // 16 == 40) if exclude_joint else None
    common = dict(
        window=4.0, radius=0.3, spacing=0.15,
        mesh_points=mesh, mesh_triangles=triangles,
        mesh_excluded_mask=excluded,
    )
    fallback = _terminal_mesh_connection(
        parent.points, child.points, mesh_tree=cKDTree(mesh), **common,
    )
    reused = _terminal_mesh_connection(
        parent.points, child.points, mesh_tree=context.point_tree,
        mesh_context=context, **common,
    )
    assert reused == fallback
    assert reused[0] is (not disconnect and not exclude_joint)


def test_topology_context_rejects_changed_native_geometry():
    parent, child, mesh, triangles = _tube_fixture()
    context = MeshGeometryContext.build(mesh, triangles)
    changed = mesh.copy()
    changed[0, 0] += 0.01
    with pytest.raises(ValueError, match="same ordered native geometry"):
        repair_root_hierarchy(
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 40.0]]),
            [parent, child], d_bar=0.15,
            support_points=mesh, mesh_points=changed,
            mesh_triangles=triangles, mesh_context=context,
        )


def test_supported_tip_tube_joins_parent_and_reassesses_descendant():
    parent, child, mesh, triangles = _tube_fixture()
    parent_snapshot = parent.points.copy()
    descendant = RootPath(
        root_id="descendant", points=np.array([[22.5, 0.25, 0.0], [22.5, 1.0, 0.0]]),
        order=3, parent_id="child", insertion_index=5,
        insertion_point=child.points[5].copy(),
    )
    paths, decisions, stats = _run(parent, child, mesh, triangles, extra=(descendant,))
    assert child not in paths
    assert len(parent.points) == 57
    np.testing.assert_allclose(parent.points[-1], child.points[-1])
    assert descendant.parent_id == "parent"
    assert descendant.order == 2
    assert descendant.insertion_index == 45
    assert decisions[0]["action"] == "joined"
    assert decisions[0]["proposal_generation"] == decisions[0]["resurvey_iteration"]
    assert decisions[0]["parent_geometry_sha256"] == hashlib.sha256(
        parent_snapshot.astype("<f8").tobytes()
    ).hexdigest()
    assert decisions[0]["reevaluation_requested_root_ids"] == [
        "parent", "primary", "descendant",
    ]
    assert decisions[0]["native_mesh_connected"]
    assert stats["terminal_continuation_joins"] == 1


def test_right_angle_tip_child_is_preserved():
    parent, child, mesh, triangles = _tube_fixture(branch=True)
    paths, decisions, stats = _run(parent, child, mesh, triangles)
    assert child in paths
    assert len(parent.points) == 41
    assert decisions[0]["reason"] == "windowed_direction_discontinuity"
    assert stats["terminal_continuation_joins"] == 0


def test_supported_terminal_join_can_exceed_supervisor_length_with_warning():
    parent, child, mesh, triangles = _tube_fixture()
    paths, decisions, stats = _run(parent, child, mesh, triangles, primary_length=25.0)
    assert child not in paths
    assert parent.length > 25.0
    assert "child_longer_than_parent" in parent.qc_flags
    assert decisions[0]["action"] == "joined"
    assert 0.0 < decisions[0]["joined_length_penalty"] <= 0.05
    assert stats["terminal_continuation_joins"] == 1


def test_disconnected_tip_surface_is_preserved():
    parent, child, mesh, triangles = _tube_fixture(disconnect=True)
    paths, decisions, stats = _run(parent, child, mesh, triangles)
    assert child in paths
    assert decisions[0]["reason"] == "native_mesh_connection_missing"
    assert stats["terminal_continuation_joins"] == 0


def test_competing_supported_tip_arm_preserves_both_children():
    parent, child, mesh, triangles = _tube_fixture()
    competing = RootPath(
        root_id="competing", points=np.array(
            [[20.0, 0.0, 0.0]] + [[20.0, 0.5 * index, 0.0] for index in range(1, 17)]
        ),
        order=2, parent_id="parent", insertion_index=40,
        insertion_point=parent.points[-1].copy(),
    )
    paths, decisions, stats = _run(parent, child, mesh, triangles, extra=(competing,))
    assert child in paths and competing in paths
    assert decisions[0]["reason"] == "competing_supported_tip_arm"
    assert "terminal_continuation_unresolved" in child.qc_flags
    assert stats["terminal_continuation_joins"] == 0


def test_separately_supported_attachment_remains_a_child():
    parent, child, mesh, triangles = _tube_fixture()
    paths, decisions, stats = _run(
        parent, child, mesh, triangles, accepted_attachment=True,
    )
    assert child in paths
    assert decisions[0]["reason"] == "separate_attachment_surface_accepted"
    assert stats["terminal_continuation_joins"] == 0


def test_full_topology_repair_reports_join_with_stable_parent_id():
    parent, child, mesh, triangles = _tube_fixture()
    primary = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 40.0]])
    repaired, report = repair_root_hierarchy(
        primary, [parent, child], d_bar=0.15,
        support_points=mesh, mesh_points=mesh, mesh_triangles=triangles,
        primary_top_reference=primary[-1],
    )
    assert len(repaired) == 1
    assert repaired[0].order == 1
    assert len(repaired[0].points) == 57
    assert report.terminal_continuation_joins == 1
    assert report.terminal_continuation_decisions[0]["final_parent_id"] == repaired[0].root_id
    assert validate_root_tree(
        repaired, primary_path=primary, primary_top_reference=primary[-1]
    ) == []


@pytest.mark.parametrize("disconnect", [False, True])
def test_displaced_endpoint_join_requires_native_connection_and_preserves_descendant(disconnect):
    parent, child, mesh, triangles = _tube_fixture(disconnect=disconnect)
    neighbor = RootPath("neighbor", np.array([[20., y, 0.] for y in np.linspace(-8, -1, 15)]),
                        order=1, parent_id="primary")
    child.points = np.vstack((neighbor.points[-1], child.points))
    child.parent_id = "neighbor"
    child.insertion_index = len(neighbor.points) - 1
    child.insertion_point = neighbor.points[-1].copy()
    for root in (parent, child):
        root.score_components["trace_local_radius"] = .3
    descendant = RootPath("descendant", np.array([child.points[7], child.points[7] + [0, 2, 0]]),
                          parent_id="child", order=3, insertion_index=7,
                          insertion_point=child.points[7].copy())
    paths = [parent, neighbor, child, descendant]
    target = child.points[-1].copy()
    descendant_origin = descendant.insertion_point.copy()
    decisions = _join_displaced_tip_continuations(
        np.array([[0., 0., 0.], [0., 0., 40.]]), paths,
        spacing=.15, support_points=mesh, mesh_points=mesh, mesh_triangles=triangles,
        excluded=None, mesh_context=None, attachment_status={},
    )
    if disconnect:
        assert decisions == [] and child in paths
        assert descendant.parent_id == "child"
    else:
        assert len(decisions) == 1 and child not in paths
        np.testing.assert_allclose(parent.points[-1], target)
        assert descendant.parent_id == "parent" and descendant.order == 2
        np.testing.assert_allclose(descendant.insertion_point, descendant_origin)
