from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soyrootbio.io import (
    _mesh_audit,
    _sanitize_mesh_arrays,
    load_root_geometry,
    write_labeled_ply,
)


def _unit_cube() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    triangles = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, triangles


def test_mesh_audit_volume_is_translation_invariant_per_component() -> None:
    cube, cube_faces = _unit_cube()
    offset = np.array([1.0e9, -2.0e9, 3.0e9])
    vertices = np.vstack([cube + offset, cube + offset + np.array([4.0, 0.0, 0.0])])
    triangles = np.vstack([cube_faces, cube_faces + len(cube)])

    audit = _mesh_audit(vertices, triangles)

    assert audit["volume_reliable"] is True
    assert audit["connected_component_count"] == 2
    assert audit["signed_volume_source_units3"] == pytest.approx(2.0, abs=1e-12)
    assert audit["absolute_volume_source_units3"] == pytest.approx(2.0, abs=1e-12)


def test_labeled_ply_preserves_large_offset_coordinate_deltas(tmp_path: Path) -> None:
    vertices = np.array(
        [
            [1.0e8, 1.0e8, 1.0e8],
            [1.0e8 + 1.0, 1.0e8, 1.0e8],
            [1.0e8, 1.0e8 + 1.0, 1.0e8],
        ]
    )
    path = tmp_path / "large-offset.ply"
    write_labeled_ply(path, vertices, triangles=np.array([[0, 1, 2]]))

    payload = path.read_bytes()
    marker = b"end_header\n"
    vertex_offset = payload.index(marker) + len(marker)
    header = payload[:vertex_offset].decode("ascii")
    assert "property double x\nproperty double y\nproperty double z\n" in header
    assert "root_id -2=uncertain -1=unassigned" in header
    assert "root_order 254=uncertain 255=unassigned" in header
    assert "assignment_state 0=unassigned 1=assigned 2=uncertain" in header
    vertex_dtype = np.dtype(
        [
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("root_id", "<i4"),
            ("root_order", "u1"),
            ("assignment_state", "u1"),
        ]
    )
    records = np.frombuffer(payload, dtype=vertex_dtype, count=len(vertices), offset=vertex_offset)
    stored = np.column_stack([records["x"], records["y"], records["z"]])
    np.testing.assert_array_equal(stored, vertices)


def test_mesh_sanitization_remaps_faces_after_nonfinite_vertex_removal() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    triangles = np.array([[0, 2, 3], [0, 1, 4], [0, 3, 4]])

    clean_vertices, clean_triangles, audit = _sanitize_mesh_arrays(vertices, triangles)

    assert np.all(np.isfinite(clean_vertices))
    np.testing.assert_array_equal(clean_vertices, vertices[[0, 2, 3, 4]])
    np.testing.assert_array_equal(clean_triangles, [[0, 1, 2], [0, 2, 3]])
    assert audit == {
        "nonfinite_vertex_count": 1,
        "faces_dropped_nonfinite_vertices": 1,
    }


def test_mesh_sanitization_rejects_out_of_range_face_indices() -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        _sanitize_mesh_arrays(np.zeros((4, 3)), np.array([[0, 1, 4]]))


def test_csv_loader_prefers_named_xyz_over_leading_numeric_columns(tmp_path: Path) -> None:
    path = tmp_path / "exported-skeleton.csv"
    rows = ["root_id,node_id,x,y,z"]
    expected = []
    for index in range(10):
        xyz = [0.125 + index, 10.25 + index, -5.5 - index]
        expected.append(xyz)
        rows.append(f"primary,{1000 + index},{xyz[0]},{xyz[1]},{xyz[2]}")
    path.write_text("\n".join(rows), encoding="utf-8")

    cloud = load_root_geometry(path)

    np.testing.assert_allclose(cloud.points, expected)
    np.testing.assert_allclose(cloud.full_points, expected)
