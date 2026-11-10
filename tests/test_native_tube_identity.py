import copy

import numpy as np
import pytest

from soyrootbio.mesh_geometry import MeshGeometryContext
from soyrootbio.tube_tracking import NativeTubeTracker, reconcile_shared_o1_tubes
from soyrootbio.types import RootPath


def cylinder(length, y, radius=.08, rings=121, sides=24):
    x = np.linspace(0, length, rings)
    angles = np.arange(sides) * 2 * np.pi / sides
    points = np.array([[a, y + radius * np.cos(b), radius * np.sin(b)]
                       for a in x for b in angles])
    faces = []
    for ring in range(rings - 1):
        for side in range(sides):
            a = ring * sides + side
            b = ring * sides + (side + 1) % sides
            faces.extend([[a, b, a + sides], [b, b + sides, a + sides]])
    points = np.vstack([points, [0, y, 0], [length, y, 0]])
    for side in range(sides):
        following = (side + 1) % sides
        faces.extend([[rings * sides, following, side],
                      [rings * sides + 1, (rings - 1) * sides + side,
                       (rings - 1) * sides + following]])
    return points, np.asarray(faces)


def crossing_case():
    b, fb = cylinder(2., 0.)
    a, fa = cylinder(3., .30)
    points = np.vstack([b, a])
    context = MeshGeometryContext.build(points, np.vstack([fb, fa + len(b)]))
    x = np.linspace(0, 3, 151)
    mixed_line = np.column_stack([x, .30 * np.clip((x - 1.2) / .6, 0, 1), x * 0])
    owner_line = np.column_stack([x, x * 0 + .30, x * 0])
    mixed_support = set(np.r_[np.arange(len(b)), len(b) + np.flatnonzero(a[:, 0] > 1.3)])
    owner_support = set(range(len(b), len(points)))
    roots = []
    for name, line, seed, support in [
        ("mixed", mixed_line, np.array([.2, .08, 0]), mixed_support),
        ("owner", owner_line, np.array([.2, .38, 0]), owner_support),
    ]:
        roots.append(RootPath(name, line, raw_start_point=seed,
                              covered_indices=support.copy(), novel_support_indices=support.copy(),
                              score_components={"surface_aware_seed": 1.,
                                                "surface_seed_contact_count": 10.,
                                                "surface_seed_gap": .005}))
    return context, roots


def repair(context, roots, excluded=None):
    return reconcile_shared_o1_tubes(
        roots, context.points, np.zeros(len(context.points), bool), context=context,
        analysis_to_mesh=np.arange(len(context.points)),
        excluded=np.zeros(len(context.points), bool) if excluded is None else excluded,
        spacing=.01,
    )


def test_native_reciprocal_identity_separates_tubes_without_reference_labels():
    context, roots = crossing_case()
    geometry = context.points.copy()
    faces = context.triangles.copy()
    starts = [root.points[0].copy() for root in roots]
    report = repair(context, roots)
    assert report["status"] == "corrected"
    mixed, owner = roots
    assert abs(mixed.points[-1, 0] - 2.) < .05
    assert abs(owner.points[-1, 0] - 3.) < .05
    assert np.max(np.abs(mixed.points[10:, 1])) < .02
    assert np.max(np.abs(owner.points[10:, 1] - .30)) < .02
    assert all(root.parent_id == "primary" and root.order == 1 for root in roots)
    for root, start in zip(roots, starts):
        np.testing.assert_array_equal(root.points[0], start)
    np.testing.assert_array_equal(context.points, geometry)
    np.testing.assert_array_equal(context.triangles, faces)


def test_native_pair_decisions_do_not_depend_on_candidate_order():
    context, roots = crossing_case()
    opposite = copy.deepcopy(roots[::-1])
    first = repair(context, roots)
    second = repair(context, opposite)
    assert first["corrected_candidate_ids"] == second["corrected_candidate_ids"]
    for left, right in zip(roots, opposite[::-1]):
        np.testing.assert_array_equal(left.points, right.points)
        assert left.covered_indices == right.covered_indices


def test_crossing_without_independent_primary_seed_is_not_rewired():
    context, roots = crossing_case()
    roots[1].score_components["surface_seed_contact_count"] = 0
    before = [root.points.copy() for root in roots]
    assert repair(context, roots)["status"] == "no_shared_suffix"
    for root, original in zip(roots, before):
        np.testing.assert_array_equal(root.points, original)


def test_point_only_input_abstains():
    context, roots = crossing_case()
    point_context = MeshGeometryContext.build(context.points, None)
    before = [root.points.copy() for root in roots]
    assert repair(point_context, roots)["status"] == "unresolved_no_mesh"
    for root, original in zip(roots, before):
        np.testing.assert_array_equal(root.points, original)


def test_native_tracker_stops_at_exclusion_instead_of_bridging():
    points, faces = cylinder(3., 0.)
    context = MeshGeometryContext.build(points, faces)
    excluded = (points[:, 0] > 1.2) & (points[:, 0] < 1.45)
    tracker = NativeTubeTracker(context, excluded, .01)
    x = np.linspace(0, 3, 151)
    line, wall, report = tracker.track(np.column_stack([x, x * 0, x * 0]))
    assert 1.1 < line[-1, 0] < 1.22
    assert not np.any(excluded[wall])
    assert report["status"] == "missing_or_merged_contour"


def test_open_longitudinal_seam_does_not_supply_partial_tube_evidence():
    points, faces = cylinder(3., 0.)
    # Remove a longitudinal strip, leaving every transverse section open.
    strip = np.any((faces % 24) == 0, axis=1)
    context = MeshGeometryContext.build(points, faces[~strip])
    tracker = NativeTubeTracker(context, np.zeros(len(points), bool), .01)
    x = np.linspace(0, 3, 151)
    _, wall, report = tracker.track(np.column_stack([x, x * 0, x * 0]))
    assert report["status"] == "no_seed"
    assert len(wall) == 0


def test_wrong_analysis_mapping_fails_before_mutation():
    context, roots = crossing_case()
    before = [root.points.copy() for root in roots]
    with pytest.raises(ValueError, match="exact analysis vertex correspondence"):
        reconcile_shared_o1_tubes(roots, context.points, np.zeros(len(context.points), bool),
                                 context=context, analysis_to_mesh=np.arange(len(context.points))[::-1],
                                 excluded=np.zeros(len(context.points), bool), spacing=.01)
    for root, original in zip(roots, before):
        np.testing.assert_array_equal(root.points, original)


def test_native_growth_rejects_nearby_disconnected_tube_but_keeps_native_route():
    from soyrootbio.native_growth import NativeStepGuard
    context, _ = crossing_case()
    guard = NativeStepGuard(context, context.points, np.arange(len(context.points)),
                            np.zeros(len(context.points), bool), np.zeros(len(context.points), bool), .01)
    p = context.points
    anchor = np.argmin(np.linalg.norm(p - [1., .08, 0], axis=1))
    same = np.argmin(np.linalg.norm(p - [1.2, .08, 0], axis=1))
    other = np.argmin(np.linalg.norm(p - [1.2, .38, 0], axis=1))
    np.testing.assert_array_equal(guard.reachable(p[anchor], [same, other], .45), [True, False])


def test_native_growth_cannot_route_through_blocked_surface():
    from soyrootbio.native_growth import NativeStepGuard
    points, faces = cylinder(3., 0.)
    context = MeshGeometryContext.build(points, faces)
    blocked = (points[:, 0] > 1.2) & (points[:, 0] < 1.4)
    guard = NativeStepGuard(context, points, None, blocked, np.zeros(len(points), bool), .01)
    anchor = np.argmin(np.linalg.norm(points - [1.1, .08, 0], axis=1))
    other = np.argmin(np.linalg.norm(points - [1.5, .08, 0], axis=1))
    assert not guard.reachable(points[anchor], [other], .6)[0]
    # The initial seed can leave a parent-owned branch neck along real faces.
    assert guard.reachable(points[anchor], [other], .6,
                           junction_origin=np.array([1., .08, 0.]))[0]
    # It cannot travel backward through the parent's core to a different arm.
    assert not guard.reachable(points[other], [anchor], .6,
                               junction_origin=np.array([1., .08, 0.]))[0]


def test_closed_short_tube_uses_radius_scaled_support_and_native_cap():
    points, faces = cylinder(.48, 0.)
    context = MeshGeometryContext.build(points, faces)
    tracker = NativeTubeTracker(context, np.zeros(len(points), bool), .01)
    x = np.linspace(0, .48, 80)
    result = tracker.track(np.column_stack([x, x * 0, x * 0]))
    assert 6 <= result[2]['stations'] < 20
    assert tracker.sustained(result) == (True, 'closed_native_cap')


def test_short_tube_cut_by_exclusion_is_not_a_native_cap():
    points, faces = cylinder(.7, 0.)
    context = MeshGeometryContext.build(points, faces)
    tracker = NativeTubeTracker(context, points[:, 0] > .42, .01)
    x = np.linspace(0, .7, 100)
    result = tracker.track(np.column_stack([x, x * 0, x * 0]))
    assert result[2]['stations'] < 20
    assert not tracker.sustained(result)[0]


def test_final_component_follows_early_body_instead_of_larger_stolen_tail():
    from soyrootbio.centerline import _origin_supported_component
    context, roots = crossing_case()
    # The first native body has fewer vertices than the stolen distal tube.
    b, _ = cylinder(2., 0.)
    components = np.r_[np.zeros(len(b), int), np.ones(len(context.points) - len(b), int)]
    chosen, reason = _origin_supported_component(context.points, components, roots[0].points, roots[0], .01)
    assert chosen == 0
    assert reason == 'early_body_supported_component'


def test_known_unresolved_mixed_identity_cannot_win_greedy_selection():
    from soyrootbio.lateral import select_non_overlapping_paths
    context, roots = crossing_case()
    roots[0].score_components['native_identity_conflict_unresolved'] = 1.
    selected = select_non_overlapping_paths(roots, context.points, .01, rename_selected=False)
    assert roots[0] not in selected
    assert roots[1] in selected


def test_incomplete_short_witness_retains_supported_prefix_without_shared_tail():
    context, roots = crossing_case()
    # A missing short-root band prevents proof of a complete terminal cap.
    excluded = ((context.points[:, 1] < .15) & (context.points[:, 0] > .60)
                & (context.points[:, 0] < .85))
    report = repair(context, roots, excluded)
    assert any(row['action'] == 'retain_supported_basal_prefix' for row in report['pairs'])
    assert roots[0].points[-1, 0] < .61
    assert 'native_tube_terminal_unresolved' in roots[0].qc_flags
    assert roots[1].points[-1, 0] > 2.9
    assert not (roots[0].covered_indices & roots[1].covered_indices)


def test_native_growth_keeps_exact_anchor_identity_at_duplicate_coordinates():
    from soyrootbio.native_growth import NativeStepGuard
    p, f = cylinder(2., 0.)
    points = np.vstack([p, p])
    context = MeshGeometryContext.build(points, np.vstack([f, f + len(p)]))
    guard = NativeStepGuard(context, points, None, np.zeros(len(points), bool),
                            np.zeros(len(points), bool), .01)
    a = np.argmin(np.linalg.norm(p - [1., .08, 0], axis=1))
    b = np.argmin(np.linalg.norm(p - [1.2, .08, 0], axis=1))
    np.testing.assert_array_equal(
        guard.reachable(points[a + len(p)], [b, b + len(p)], .4,
                        analysis_anchor=a + len(p)), [False, True])
