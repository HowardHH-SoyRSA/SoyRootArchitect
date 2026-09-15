import json

import numpy as np
import pytest

from soyrootbio.surface_patches import correct_surface_patches
from soyrootbio.types import RootPath


def strips(centers=(0., 3., 6.), ranges=None):
    """Three triangulated axial surface strips with independently known axes."""
    points, faces, labels, paths = [], [], [], []
    for label, x in enumerate(centers):
        z = np.arange(0., 10.01, .25) if ranges is None else ranges[label]
        offset = len(points)
        points.extend([[x + .2, y, s] for s in z for y in [-.12, 0., .12]])
        labels.extend([label] * (len(z) * 3))
        for i in range(len(z) - 1):
            for j in range(2):
                a = offset + 3 * i + j
                faces.extend([[a, a + 1, a + 3], [a + 1, a + 4, a + 3]])
        paths.append(np.array([[x, 0., 0.], [x, 0., 10.]]))
    roots = [RootPath(root_id=f"root-{i}", points=p, order=i, parent_id="primary")
             for i, p in enumerate(paths[1:], 1)]
    return np.array(points), np.array(labels), paths[0], roots, np.array(faces)


def run(case, **kwargs):
    p, labels, primary, roots, faces = case
    result, report = correct_surface_patches(p, labels, primary, roots,
                                             d_bar=.15, triangles=faces, **kwargs)
    assert report["converged"]
    if "energy" in report:
        assert np.all(np.diff(report["energy"]) < 0)
    json.dumps(report, allow_nan=False)
    return result, report


@pytest.mark.parametrize("source,target", [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)])
def test_isolated_islands_use_the_same_rules_for_all_root_orders(source, target):
    case = strips()
    truth = case[1].copy()
    island = np.flatnonzero(truth == target)[61]
    case[1][island] = source
    before_points = case[0].copy()
    before_paths = [r.points.copy() for r in case[3]]
    after, report = run(case)
    np.testing.assert_array_equal(after, truth)
    np.testing.assert_array_equal(case[0], before_points)
    for root, path in zip(case[3], before_paths):
        np.testing.assert_array_equal(root.points, path)
    assert report["reassigned_vertex_count"] == 1
    assert report["reassigned_patch_count"] == 1


def test_alternating_chimeric_patches_propagate_to_stability():
    case = strips()
    truth = case[1].copy()
    target_ids = np.flatnonzero(truth == 2)
    for band in range(1, 8):
        case[1][target_ids[band * 12:(band + 1) * 12]] = band % 2
    after, report = run(case)
    np.testing.assert_array_equal(after, truth)
    assert report["reassigned_patch_count"] == 7
    assert report["passes"] >= 2
    again, second = run((case[0], after, *case[2:]))
    np.testing.assert_array_equal(again, after)
    assert second["reassigned_vertex_count"] == 0


def test_adjacent_root_contact_does_not_absorb_supported_bodies():
    case = list(strips(centers=(0., .8, 6.)))
    # Add a real local contact while preserving two independently supported axes.
    # Move the second strip to its facing surface; its distance to its axis stays .2.
    case[0][case[1] == 1, 0] -= .4
    case[4] = np.vstack([case[4], [[60, 61, 183], [61, 183, 184]]])
    after, report = run(case)
    np.testing.assert_array_equal(after, case[1])
    assert report["reassigned_vertex_count"] == 0
    assert any(len(c["candidates"]) > 1 for c in report["components"])


@pytest.mark.parametrize("source_order", [0, 1, 3])
def test_largest_component_invading_collar_can_be_reassigned(source_order):
    case = list(strips(ranges=[np.arange(0, 10.01, .25), np.arange(0, 1.01, .25),
                               np.arange(0, 10.01, .25)]))
    p, labels, _, roots, _ = case
    # Lateral emerges at the collar and then runs outwards; primary surface
    # invasion is much larger than its real supported body.
    roots[0].points = np.array([[0., 0., 10.], [3., 0., 10.], [4., 0., 10.]])
    child = labels == 1
    p[child] = np.column_stack([3. + p[child, 2], p[child, 1], np.full(child.sum(), 10.2)])
    truth = labels.copy()
    invasion = (truth == 0) & (p[:, 2] >= 6.)
    labels[invasion] = 1
    assert invasion.sum() > child.sum()
    if source_order == 0:
        case[2], roots[0].points = roots[0].points, case[2]
        permutation = np.array([1, 0, 2])
        case[1] = permutation[labels]
        truth = permutation[truth]
    else:
        roots[0].order = source_order
    after, report = run(case)
    np.testing.assert_array_equal(after, truth)
    assert report["reassigned_vertex_count"] == invasion.sum()


def test_valid_detached_support_is_preserved_without_a_mesh_bridge():
    case = list(strips())
    p, labels, _, _, faces = case
    # Cut a transverse mesh seam without altering the correctly assigned body.
    z = p[faces, 2]
    case[4] = faces[~((z.min(axis=1) < 5.) & (z.max(axis=1) >= 5.))]
    after, report = run(case)
    np.testing.assert_array_equal(after, labels)
    assert report["initial_component_count"] == 6
    assert report["reassigned_vertex_count"] == 0


def test_supported_detached_fragment_survives_contact_with_another_root():
    case = list(strips(centers=(0., .7, 6.)))
    p, labels, _, _, faces = case
    p[labels == 1, 0] -= .4
    z = p[faces, 2]
    child_faces = np.all(labels[faces] == 1, axis=1)
    cut = child_faces & (((z.min(axis=1) < 5.) & (z.max(axis=1) >= 5.)) |
                         ((z.min(axis=1) < 5.75) & (z.max(axis=1) >= 5.75)))
    case[4] = np.vstack([faces[~cut], [[60, 61, 183], [61, 183, 184]]])
    after, report = run(case)
    np.testing.assert_array_equal(after, labels)
    fragment = next(c for c in report["components"] if c["first_vertex"] == 183)
    candidates = {c["label"]: c for c in fragment["candidates"]}
    assert candidates[0]["admissible"]
    assert candidates[1]["body_anchor"]
    assert candidates[1]["cost"] < candidates[0]["cost"]


@pytest.mark.parametrize("barrier", ["unassigned", "uncertain", "excluded", "missing_faces", "no_mesh"])
def test_unsupported_regions_cannot_connect_a_patch_to_body(barrier):
    case = list(strips())
    p, labels, _, _, faces = case
    target = labels == 2
    wrong = target & (p[:, 2] < 5.)
    labels[wrong] = 1
    seam = target & (p[:, 2] == 5.)
    excluded = np.zeros(len(p), dtype=bool)
    if barrier in {"unassigned", "uncertain"}:
        labels[seam] = -1 if barrier == "unassigned" else -2
    elif barrier == "excluded":
        excluded[seam] = True
    elif barrier == "missing_faces":
        z = p[faces, 2]
        case[4] = faces[~((z.min(axis=1) < 5.) & (z.max(axis=1) >= 5.))]
    else:
        case[4] = None
    before = labels.copy()
    after, _ = run(case, excluded_mask=excluded)
    np.testing.assert_array_equal(after, before)


def test_large_triangle_over_gap_does_not_supply_connectivity():
    case = list(strips())
    p, labels, _, _, faces = case
    ids = np.flatnonzero(labels == 2)
    labels[ids[:60]] = 1
    seam_faces = np.any(np.isin(faces, ids[60:66]), axis=1)
    case[4] = np.vstack([faces[~seam_faces], [[ids[57], ids[58], ids[75]]]])
    after, _ = run(case)
    np.testing.assert_array_equal(after, labels)


def test_label_permutation_and_order_changes_do_not_change_correction():
    case = strips()
    case[1][250:280] = 1
    first, _ = run(case)
    # Move old primary to higher order, old higher-order root to primary.
    permutation = np.array([2, 1, 0])
    paths = [case[2]] + [r.points for r in case[3]]
    roots = [RootPath(root_id=f"permuted-{i}", points=paths[old], order=7-i,
                      parent_id="unrelated") for i, old in enumerate([1, 0], 1)]
    second, _ = run((case[0], permutation[case[1]], paths[2], roots, case[4]))
    np.testing.assert_array_equal(second, permutation[first])


def test_segment_resampling_and_scale_leave_correction_unchanged():
    case = strips()
    case[1][250:280] = 1
    first, _ = run(case)
    primary = np.linspace(case[2][0], case[2][-1], 31)
    roots = [RootPath(root_id=r.root_id, points=np.linspace(r.points[0], r.points[-1], 19) * 10)
             for r in case[3]]
    second, _ = correct_surface_patches(case[0] * 10, case[1], primary * 10, roots,
                                        d_bar=1.5, triangles=case[4])
    np.testing.assert_array_equal(second, first)


def test_equal_neighbor_claims_retain_owner_deterministically():
    case = list(strips(centers=(6., 0., 0.)))
    p, labels, _, _, faces = case
    island = len(p)
    case[0] = np.vstack([p, [[.2, 0., 5.]]])
    case[1] = np.r_[labels, 0]
    case[4] = np.vstack([faces, [[island, 183, 184], [island, 306, 307]]])
    after, report = run(case)
    assert after[island] == 0
    row = next(c for c in report["components"] if c["first_vertex"] == island)
    candidates = {c["label"]: c for c in row["candidates"]}
    assert candidates[1]["cost"] == pytest.approx(candidates[2]["cost"])
    assert candidates[1]["cost"] < candidates[0]["cost"]
    repeated, _ = run(case)
    np.testing.assert_array_equal(repeated, after)


def test_all_neighbors_compete_even_when_better_geometry_has_less_boundary_contact():
    case = list(strips(centers=(6., .5, 0.)))
    p, labels, _, _, faces = case
    island = len(p)
    case[0] = np.vstack([p, [[.2, 0., 5.]]])
    case[1] = np.r_[labels, 0]
    case[4] = np.vstack([faces, [[island, 183, 184], [island, 184, 185], [island, 306, 307]]])
    after, report = run(case)
    assert after[island] == 2
    row = next(c for c in report["components"] if c["first_vertex"] == island)
    candidates = {c["label"]: c for c in row["candidates"]}
    assert candidates[1]["admissible"] and candidates[2]["admissible"]
    assert candidates[1]["boundary_contact_edges"] > candidates[2]["boundary_contact_edges"]
    assert candidates[1]["cost"] > candidates[2]["cost"]


def test_perpendicular_patch_has_directional_evidence_at_a_single_projection_station():
    case = list(strips(centers=(0., .4, 6.)))
    case[2] = np.array([[-1., 0., 5.], [1., 0., 5.]])
    case[4] = np.vstack([case[4], [[60, 61, 183], [61, 183, 184]]])
    _, report = run(case)
    body = next(c for c in report["components"] if c["source_label"] == 1)
    perpendicular = next(c for c in body["candidates"] if c["label"] == 0)
    assert perpendicular["direction_measured"]
    assert perpendicular["direction_penalty"] == pytest.approx(1.)
    assert not perpendicular["admissible"]


def test_an_unsupported_vertex_cannot_be_used_as_a_bridge_inside_a_patch():
    case = list(strips())
    p, labels, _, _, _ = case
    target = labels == 2
    patch = target & (p[:, 2] < 5.)
    labels[patch] = 1
    # A mesh-connected patch with a geometrically unsupported middle cannot
    # carry the target label across it, even though both ends fit its axis.
    p[patch & (p[:, 2] == 2.5), 0] += .5
    after, _ = run(case)
    np.testing.assert_array_equal(after, labels)


@pytest.mark.parametrize("order", [1, 2, 5])
def test_internal_connector_does_not_supply_a_surface_body(order):
    case = list(strips())
    p, labels, primary, roots, _ = case
    roots[0].order = order
    roots[0].points = np.vstack([primary, roots[0].points])
    roots[0].body_start_index = 2
    truth = labels.copy()
    labels[61] = 1
    after, _ = run(case)
    np.testing.assert_array_equal(after, truth)


@pytest.mark.parametrize("spacing", [0., -1., float("nan")])
def test_invalid_spacing_is_rejected(spacing):
    case = strips()
    with pytest.raises(ValueError, match="d_bar"):
        correct_surface_patches(case[0], case[1], case[2], case[3], d_bar=spacing,
                                triangles=case[4])
