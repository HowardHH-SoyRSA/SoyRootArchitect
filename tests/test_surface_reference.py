import hashlib
import json

import numpy as np
import pytest

from soyrootbio.mesh_geometry import MeshGeometryContext
from soyrootbio.surface_reference import (
    KEEP, SCHEMA, apply_surface_reference, load_surface_reference,
)
from soyrootbio.topology import validate_root_tree
from soyrootbio.types import RootPath


def fixture():
    points, faces = [], []
    for y in (0., 1., 2.):
        offset = len(points)
        for x in np.linspace(0., 1., 16):
            for angle in np.arange(8) * np.pi / 4:
                points.append([x, y + .04 * np.cos(angle), .04 * np.sin(angle)])
        for i in range(15):
            for j in range(8):
                a, b = offset + i * 8 + j, offset + i * 8 + (j + 1) % 8
                faces.extend([[a, b, a + 8], [b, b + 8, a + 8]])
    points, faces = np.array(points), np.array(faces)
    labels = np.repeat([1, 2, 3], 128)
    labels[64:128] = 4  # a falsely independent continuation, with a child
    primary = np.array([[0., 0., 2.], [0., 0., 0.], [0., 0., -2.]])
    roots = []
    for i, y in enumerate((0., 1., 2., 0.), 1):
        start = .5 if i == 4 else 0.
        line = np.array([[start, y, 0.], [1., y, 0.]])
        roots.append(RootPath(f"auto-{i}", line, parent_id="primary",
                              insertion_point=primary[1].copy(), insertion_index=1))
    roots[2].parent_id = "auto-4"
    roots[2].order = 2
    requested = np.full(len(labels), KEEP, np.int32)
    requested[:128] = 1
    requested[128:256] = 2
    anchors = np.zeros(len(labels), np.int32)
    anchors[:64] = 1
    anchors[128:256] = 2
    return points, faces, labels, primary, roots, requested, anchors


def write_reference(tmp_path, points, faces, requested, anchors, boundary=None, **fields):
    data_path = tmp_path / "ref.npz"
    np.savez_compressed(data_path, points=points, triangles=faces, requested=requested,
                        anchors=anchors, boundary=np.zeros(len(points), bool) if boundary is None else boundary)
    manifest = {"schema": SCHEMA, "data_file": data_path.name,
                "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "owners": [{"reference_root_id": "edited-renamed-a"},
                           {"reference_root_id": "edited-renamed-b"}],
                **fields}
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(manifest))
    return path


def run(tmp_path, *, excluded=None, boundary=None, mutate=None):
    points, faces, labels, primary, roots, requested, anchors = fixture()
    if mutate:
        mutate(points, faces, labels, roots, requested, anchors)
    path = write_reference(tmp_path, points, faces, requested, anchors, boundary)
    reference = load_surface_reference(path, points, faces)
    return apply_surface_reference(reference, points, labels, primary, roots, d_bar=.04,
                mesh_context=MeshGeometryContext.build(points, faces),
                excluded_mask=np.zeros(len(points), bool) if excluded is None else excluded,
                primary_top_reference=primary[0]), (points, faces, labels, primary, roots)


def test_reviewed_surface_wins_and_unrelated_child_survives_empty_parent(tmp_path):
    (labels, roots, report, audit), (_, _, before, primary, original) = run(tmp_path)
    assert report["changed_vertex_count"] == 64
    assert report["retired_empty_root_ids"] == ["auto-4"]
    assert np.all(labels[:128] == 1)
    assert np.all(labels[256:] == 3)
    assert [r.root_id for r in roots] == ["auto-1", "auto-2", "auto-3"]
    assert roots[2].parent_id == "primary"
    assert "surface_reference_hierarchy_unresolved" in roots[2].qc_flags
    assert not validate_root_tree(roots, primary_path=primary, primary_top_reference=primary[0])
    assert np.array_equal(original[0].points, [[0., 0., 0.], [1., 0., 0.]])
    assert original[2].parent_id == "auto-4"  # no input mutation
    np.testing.assert_array_equal(audit["before_labels"], before)


@pytest.mark.parametrize("change", ["positions", "order", "faces", "duplicate", "point_only", "hash"])
def test_fail_closed_on_geometry_and_provenance_mismatch(tmp_path, change):
    points, faces, _, _, _, requested, anchors = fixture()
    path = write_reference(tmp_path, points, faces, requested, anchors)
    points, faces = points.copy(), faces.copy()
    if change == "positions":
        points[0, 0] += 1e-9
    elif change == "order":
        points[[0, 1]] = points[[1, 0]]
    elif change == "duplicate":
        points[0] = points[1]
    elif change == "faces":
        faces[0] = faces[0, ::-1]
    elif change == "point_only":
        faces = None
    elif change == "hash":
        with (tmp_path / "ref.npz").open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        load_surface_reference(path, points, faces)


def test_above_collar_exclusion_overrides_constraint_with_explicit_difference(tmp_path):
    excluded = np.zeros(384, bool)
    excluded[127] = True
    def mutate(p, f, labels, roots, requested, anchors):
        labels[127] = -1
    (labels, roots, report, audit), _ = run(tmp_path, excluded=excluded, mutate=mutate)
    assert labels[127] == -1
    assert report["reference_differences"][0]["excluded_vertices"] == 1
    assert report["reference_differences"][0]["difference_vertices"] == 1


def test_excessive_reference_difference_aborts_transaction(tmp_path):
    excluded = np.zeros(384, bool)
    excluded[120:128] = True
    def mutate(p, f, labels, roots, requested, anchors):
        labels[120:128] = -1
    with pytest.raises(ValueError, match="exceed"):
        run(tmp_path, excluded=excluded, mutate=mutate)


def test_nonroot_object_labels_survive_an_overlapping_reference(tmp_path):
    excluded = np.zeros(384, bool)
    excluded[127] = True
    def mutate(p, f, labels, roots, requested, anchors):
        labels[127] = -3
    (labels, roots, report, audit), _ = run(tmp_path, excluded=excluded, mutate=mutate)
    assert labels[127] == audit['before_labels'][127] == audit['final_labels'][127] == -3


def test_edited_hierarchy_and_polyline_are_ignored(tmp_path):
    points, faces, labels, primary, roots, requested, anchors = fixture()
    path = write_reference(tmp_path, points, faces, requested, anchors,
                           hierarchy={"auto-1": {"parent_id": "not-a-root", "order": 999}},
                           polyline=[[float("nan"), 0, 0]])
    ref = load_surface_reference(path, points, faces)
    result, updated, report, _ = apply_surface_reference(ref, points, labels, primary, roots,
        d_bar=.04, mesh_context=MeshGeometryContext.build(points, faces),
        excluded_mask=np.zeros(len(labels), bool), primary_top_reference=primary[0])
    assert updated[0].parent_id == "primary"
    assert all(np.isfinite(root.points).all() for root in updated)


def test_two_reference_bodies_can_split_one_automatic_identity(tmp_path):
    def mutate(p, f, labels, roots, requested, anchors):
        labels[128:256] = 1  # automatic merge of independent native components
    (labels, roots, report, _), _ = run(tmp_path, mutate=mutate)
    assert len({int(labels[0]), int(labels[128])}) == 2
    assert any(row["mode"] == "new_identity_from_reviewed_surface" for row in report["matches"])
    assert all(len(root.points) > 0 for root in roots)
    assert report["outside_scope_changed_vertices"] == 0


def test_disconnected_reference_support_keeps_fragment_without_surface_bridge(tmp_path):
    def mutate(p, f, labels, roots, requested, anchors):
        requested[256:264] = 1  # physically separate source component
    (labels, roots, report, audit), (_, faces, _, _, _) = run(tmp_path, mutate=mutate)
    assert np.all(labels[256:264] == 1)
    assert "surface_reference_disconnected_support" in roots[0].qc_flags
    row = next(row for row in report["hierarchy_decisions"] if row["root_id"] == "auto-1")
    assert row["excluded_fragment_vertices"] == 8
    assert faces.shape == (720, 3)  # source mesh unchanged


def test_corrected_child_is_oriented_to_opposite_native_primary_origin(tmp_path):
    points, faces, labels, primary, roots, requested, anchors = fixture()
    # A real native ring at the opposite end contacts primary surface. Merely
    # close unrelated surfaces in the fixture have no such connecting faces.
    ring = points[120:128].copy()
    ring[:, 0] = 1.08
    start = len(points)
    points = np.vstack([points,ring])
    join = []
    for j in range(8):
        a,b=120+j,120+(j+1)%8
        join.extend([[a,b,start+j],[b,start+(j+1)%8,start+j]])
    faces=np.vstack([faces,join])
    labels=np.r_[labels,np.zeros(8,int)]
    requested=np.r_[requested,np.full(8,KEEP,np.int32)]
    anchors=np.r_[anchors,np.zeros(8,np.int32)]
    primary=np.array([[1.08,0.,2.],[1.08,0.,0.],[1.08,0.,-2.]])
    roots[0].parent_id='auto-2'
    roots[0].order=2
    path=write_reference(tmp_path,points,faces,requested,anchors)
    ref=load_surface_reference(path,points,faces)
    _,updated,report,_=apply_surface_reference(ref,points,labels,primary,roots,d_bar=.04,
        mesh_context=MeshGeometryContext.build(points,faces),excluded_mask=np.zeros(len(labels),bool),
        primary_top_reference=primary[0])
    root=next(r for r in updated if r.root_id=='auto-1')
    assert root.parent_id=='primary' and root.order==1
    assert root.points[0,0] > root.points[-1,0]
