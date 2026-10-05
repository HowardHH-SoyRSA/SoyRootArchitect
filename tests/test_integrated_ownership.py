"""Interactions between automatic tube ownership and scoped surface constraints."""

from copy import deepcopy
import hashlib
import json

import numpy as np

from soyrootbio.centerline import refit_final_centerlines
from soyrootbio.junction_tubes import reconcile_parent_owned_tubes
from soyrootbio.mesh_geometry import MeshGeometryContext
from soyrootbio.surface_reference import (KEEP, SCHEMA, SurfaceReference, apply_surface_reference,
                                         load_surface_reference, tube_claim_limits)
from soyrootbio.topology import validate_root_tree
from soyrootbio.traits import compute_traits
from soyrootbio.types import Normalization, RootPath
from test_primary_o1_ownership import junction


def test_tube_then_reference_refit_preserves_unrelated_correction_and_selected_top(tmp_path):
    points, labels, _, children, faces, parent_count = junction()
    # Two independently exposed branches on the same parent. The second is
    # outside the reviewed region and must retain its automatic correction.
    other_surface = points[parent_count:].copy()
    other_surface[:, 0] *= -1
    other_surface[:, 2] -= .2
    other_faces = faces[np.all(faces >= parent_count, axis=1)] - parent_count + len(points)
    other_labels = np.where(labels[parent_count:] == 1, 2, 0)
    first_surface_count = len(points)
    points = np.vstack((points, other_surface))
    labels = np.r_[labels, other_labels]
    faces = np.vstack((faces, other_faces))
    primary = np.column_stack((np.zeros(156), np.zeros(156), np.linspace(.375, -.4, 156)))
    selected_top = primary[0].copy()
    children[0].insertion_index = 75
    children[0].insertion_point = primary[75].copy()
    children.append(RootPath('unrelated-child', np.array([[0., 0., -.2], [-.35, 0., -.2]]),
                             parent_id='primary', order=1, insertion_index=115,
                             insertion_point=primary[115].copy()))
    excluded = points[:, 2] > selected_top[2]
    labels[excluded] = -1
    original_geometry = points.copy(), faces.copy()
    context = MeshGeometryContext.build(points, faces)

    corrected, tube_report = reconcile_parent_owned_tubes(
        points, labels, primary, children, d_bar=.004, triangles=faces,
        excluded_mask=excluded, mesh_context=context)
    first_body = (np.arange(len(points)) >= parent_count) & (np.arange(len(points)) < first_surface_count)
    second_body = np.arange(len(points)) >= first_surface_count
    first_transfer = first_body & (points[:, 0] >= .055) & (points[:, 0] < .1)
    second_transfer = second_body & (points[:, 0] <= -.055) & (points[:, 0] > -.1)
    assert tube_report['transferred_vertex_count'] > 0
    assert np.all(corrected[first_transfer] == 1)
    assert np.all(corrected[second_transfer] == 2)

    # The approved reference retains the first shaft but leaves a bounded
    # junction cut. That cut deliberately overrides an automatic tube claim.
    requested = np.full(len(points), KEEP, np.int32)
    requested[first_body & (points[:, 0] >= .075)] = 1
    released = first_body & (corrected == 1) & (points[:, 0] < .075)
    requested[released] = -1
    anchors = np.zeros(len(points), np.int32)
    anchors[first_body & (points[:, 0] >= .15)] = 1
    data = tmp_path / 'reviewed.npz'
    np.savez_compressed(data, points=points, triangles=faces, requested=requested,
                        anchors=anchors, boundary=np.zeros(len(points), bool))
    manifest = tmp_path / 'reviewed.json'
    manifest.write_text(json.dumps({
        'schema': SCHEMA, 'data_file': data.name,
        'data_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
        'owners': [{'reference_root_id': 'reviewed-child'}],
        'maximum_difference_fraction': .02,
    }))
    reference = load_surface_reference(manifest, points, faces)

    fitted_primary, fit_report = refit_final_centerlines(
        points, corrected, primary, children, d_bar=.004, triangles=faces,
        primary_top_reference=selected_top, mesh_context=context)
    pre_reference_children = deepcopy(children)
    priors = {'primary': {'points': fitted_primary.copy(), 'body_start_index': 0,
                          'vertices': np.flatnonzero(corrected == 0),
                          'assessment': fit_report['roots'][0]}}
    priors.update({r.root_id: {'points': r.points.copy(), 'body_start_index': r.body_start_index,
                              'vertices': np.flatnonzero(corrected == label),
                              'assessment': dict(r.centerline_assessment)}
                   for label, r in enumerate(children, 1)})
    final_labels, final_roots, reference_report, audit = apply_surface_reference(
        reference, points, corrected, fitted_primary, children, d_bar=.004,
        mesh_context=context, excluded_mask=excluded,
        primary_top_reference=selected_top)
    assert reference_report['outside_scope_changed_vertices'] == 0
    assert reference_report['reference_differences'][0]['difference_vertices'] == 0
    assert np.any(released & (corrected == 1))
    assert np.all(final_labels[released] == -1)
    np.testing.assert_array_equal(final_labels[second_body], corrected[second_body])
    assert np.all(final_labels[second_transfer] == 2)
    np.testing.assert_array_equal(final_labels[excluded], -1)
    np.testing.assert_array_equal(audit['before_labels'], corrected)

    affected = set(reference_report['affected_root_ids'])
    mapping = {'primary': 0, **{r.root_id: label for label, r in enumerate(final_roots, 1)}}
    preserved = {rid: prior for rid, prior in priors.items()
                 if rid in mapping and rid not in affected
                 and np.array_equal(prior['vertices'], np.flatnonzero(final_labels == mapping[rid]))}
    final_primary, final_report = refit_final_centerlines(
        points, final_labels, fitted_primary, final_roots, d_bar=.004,
        triangles=faces, primary_top_reference=selected_top,
        mesh_context=context, preserved_bodies=preserved)
    unrelated = next(r for r in final_roots if r.root_id == 'unrelated-child')
    previous_unrelated = next(r for r in pre_reference_children if r.root_id == 'unrelated-child')
    np.testing.assert_array_equal(unrelated.points, previous_unrelated.points)
    assert unrelated.centerline_assessment['body_fit_reused']
    np.testing.assert_array_equal(final_report['primary_top_point_normalized'], selected_top)
    assert not validate_root_tree(final_roots, primary_path=final_primary,
                                  primary_top_reference=selected_top)
    assert all(r.insertion_point[2] < selected_top[2] for r in final_roots)
    np.testing.assert_array_equal(final_labels[excluded], -1)
    np.testing.assert_array_equal(points, original_geometry[0])
    np.testing.assert_array_equal(faces, original_geometry[1])

    lateral_labels = np.where(final_labels > 0, final_labels, 0)
    traits = compute_traits(final_primary, final_roots, points, final_labels == 0,
                            lateral_labels, Normalization(np.zeros(3), 1.),
                            full_points=points, triangles=faces, full_root_labels=final_labels,
                            primary_centerline_assessment=final_report['roots'][0]).set_index('root_id')
    for rid, label in mapping.items():
        assert traits.loc[rid, 'point_count'] == np.count_nonzero(final_labels == label)


def test_reviewed_bound_declines_whole_new_tube_without_removing_existing_support():
    points, before, primary, children, faces, _ = junction()
    excluded = np.zeros(len(points), bool)
    automatic, _ = reconcile_parent_owned_tubes(
        points, before, primary, children, d_bar=.004, triangles=faces)
    additions = np.flatnonzero((automatic == 1) & (before == 0))
    assert len(additions) > 0
    requested = np.full(len(points), KEEP, np.int32)
    requested[before == 1] = 1
    anchors = np.where(before == 1, 1, 0).astype(np.int32)
    reference = SurfaceReference(
        {'owners': [{'reference_root_id': 'reviewed'}], 'maximum_difference_fraction': .02},
        requested, anchors, excluded.copy(), 'manifest', 'data')
    limits = tube_claim_limits(reference, before, children, excluded)
    limited, report = reconcile_parent_owned_tubes(
        points, before, primary, children, d_bar=.004, triangles=faces,
        reference_limits=limits)
    np.testing.assert_array_equal(limited, before)
    row = report['junctions'][0]
    assert row['status'] == 'retained_reference_bound'
    assert row['proposed_vertex_count'] > 0
    assert row['transferred_vertex_count'] == 0
    assert row['reference_claim_limit']['proposed_difference_fraction'] > .02

    # One extra reviewed-owner vertex is within tolerance. All automatic
    # claims, including that small permitted difference, remain unchanged.
    requested[automatic == 1] = 1
    requested[additions[0]] = KEEP
    accepted, accepted_report = reconcile_parent_owned_tubes(
        points, before, primary, children, d_bar=.004, triangles=faces,
        reference_limits=tube_claim_limits(reference, before, children, excluded))
    np.testing.assert_array_equal(accepted, automatic)
    assert accepted_report['transferred_vertex_count'] > 0


def test_reference_tube_limit_counts_only_extras_that_survive_explicit_scope():
    roots = [RootPath('child', np.array([[0., 0., 0.], [1., 0., 0.]]))]
    before = np.array([1, 1, 1, 0, 0, -1])
    excluded = np.array([False, False, False, False, False, True])
    reference = SurfaceReference(
        {'owners': [{'reference_root_id': 'reviewed'}], 'maximum_difference_fraction': .02},
        np.array([1, 1, -1, 1, KEEP, 1]), np.array([1, 1, 0, 0, 0, 0]),
        np.zeros(6, bool), 'manifest', 'data')
    limit = tube_claim_limits(reference, before, roots, excluded)[1]
    # The explicit release at vertex 2 disappears later. The excluded
    # requested vertex remains a difference; it cannot be reassigned.
    assert limit['existing_differences'] == 1
    assert limit['expected_vertices'] == 4
    np.testing.assert_array_equal(limit['outside_mask'], [False, False, False, False, True, False])
    reference.anchors[:] = 0
    assert tube_claim_limits(reference, before, roots, excluded) == {}
