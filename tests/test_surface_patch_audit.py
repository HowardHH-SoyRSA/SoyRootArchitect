import numpy as np

from soyrootbio.mesh_geometry import MeshGeometryContext
from soyrootbio.surface_patch_audit import audit_discrete_child_patches
from soyrootbio.types import RootPath


def test_disconnected_child_patch_on_parent_is_reported_without_relabeling():
    points = np.array([
        [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
        [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    root = RootPath("child", points[[1, 5]], order=1, parent_id="primary")
    labels = np.array([0, 1, 0, 0, 0, 1, 0, 1])
    frozen = labels.copy()
    context = MeshGeometryContext.build(points, faces)
    report = audit_discrete_child_patches(
        points, labels, [root], triangles=faces, d_bar=.5,
        mesh_context=context,
    )
    np.testing.assert_array_equal(labels, frozen)
    assert report["status"] == "unresolved_patches"
    assert report["unresolved_patch_count"] == 1
    assert report["patches"][0]["vertex_indices"] == [7]
    assert report["patches"][0]["parent_id"] == "primary"
    assert audit_discrete_child_patches(
        points, labels, [root], triangles=None, d_bar=.5,
    )["status"] == "unresolved_no_mesh"


def test_only_owned_component_without_exposed_body_anchor_is_unresolved():
    points = np.array([
        [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
        [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    root = RootPath("child", points[[1, 5]], order=1, parent_id="primary")
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    report = audit_discrete_child_patches(
        points, labels, [root], triangles=faces, d_bar=.5)
    assert report["status"] == "unresolved_patches"
    assert report["patches"][0]["vertex_indices"] == [7]
    assert report["patches"][0]["status"] == "unresolved_exposed_body_anchor_ambiguous"
