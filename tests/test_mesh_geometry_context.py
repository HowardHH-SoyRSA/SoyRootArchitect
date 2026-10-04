import numpy as np
import pytest

from soyrootbio.attachment_constraint import assess_attachment_footprints
from soyrootbio.final_surface_cleanup import cleanup_final_surface
from soyrootbio.mesh_geometry import MeshGeometryContext
from soyrootbio.ownership_ledger import reconcile_released_primary_contact_vertices
from soyrootbio.primary_contact import (
    audit_higher_order_primary_contacts,
    restrict_higher_order_primary_contacts,
)
from soyrootbio.surface_patches import correct_surface_patches
from soyrootbio.types import RootPath


def _mesh():
    points = np.array([
        [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
        [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    roots = [
        RootPath("o1", points[[2, 5]], order=1, parent_id="primary", mean_radius=.3),
        RootPath("o2", points[[1, 6, 7]], order=2, parent_id="o1", mean_radius=.3),
    ]
    return points, faces, roots


def test_shared_native_geometry_preserves_stage_decisions():
    points, faces, roots = _mesh()
    original_points, original_faces = points.copy(), faces.copy()
    context = MeshGeometryContext.build(points, faces)
    assert context.bounded_edges(.5) is context.bounded_edges(.5)
    with pytest.raises(ValueError, match="same ordered"):
        context.validate(points[::-1], faces)
    labels = np.array([0, 2, 1, 1, 1, 1, 2, 2])
    primary = points[[0, 4]]

    stage_calls = [
        lambda **kw: correct_surface_patches(points, labels, primary, roots,
                                              d_bar=.5, triangles=faces, **kw),
        lambda **kw: cleanup_final_surface(points, labels, primary, roots,
                                           d_bar=.5, triangles=faces, **kw),
        lambda **kw: restrict_higher_order_primary_contacts(
            points, labels, roots, triangles=faces, d_bar=.5, **kw),
    ]
    for call in stage_calls:
        uncached_labels, uncached_report = call()
        cached_labels, cached_report = call(mesh_context=context)
        np.testing.assert_array_equal(cached_labels, uncached_labels)
        assert cached_report == uncached_report

    uncached_audit = audit_higher_order_primary_contacts(
        points, labels, roots, triangles=faces, d_bar=.5)
    cached_audit = audit_higher_order_primary_contacts(
        points, labels, roots, triangles=faces, d_bar=.5,
        mesh_context=context)
    assert cached_audit == uncached_audit
    uncached_attachment = assess_attachment_footprints(
        points, labels, primary, roots, triangles=faces, d_bar=.5)
    cached_attachment = assess_attachment_footprints(
        points, labels, primary, roots, triangles=faces, d_bar=.5,
        mesh_context=context)
    assert cached_attachment == uncached_attachment

    released = labels.copy()
    released[1] = -2
    plain, _, plain_report = reconcile_released_primary_contact_vertices(
        points, labels, released, primary, roots, triangles=faces, d_bar=.5)
    shared, _, shared_report = reconcile_released_primary_contact_vertices(
        points, labels, released, primary, roots, triangles=faces, d_bar=.5,
        mesh_context=context)
    np.testing.assert_array_equal(shared, plain)
    assert shared_report == plain_report
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(faces, original_faces)
