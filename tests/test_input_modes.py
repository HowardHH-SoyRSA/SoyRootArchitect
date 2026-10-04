from pathlib import Path

import numpy as np
import pytest

import soyrootbio.io as geometry_io
from soyrootbio.cli import build_parser


def equivalent_files(tmp_path: Path):
    # Shared ordered coordinates, including an unused vertex and duplicate.
    x, y = np.meshgrid(np.arange(10), np.arange(10))
    points = np.column_stack([x.ravel(), y.ravel(), np.zeros(100)]).astype(float)
    points = np.vstack([points, [50, 50, 50], points[0]])
    faces = []
    for row in range(9):
        for col in range(9):
            a = row * 10 + col
            faces.extend([[a, a + 1, a + 10], [a + 1, a + 11, a + 10]])
    faces = np.asarray(faces)
    csv = tmp_path / "points.csv"
    np.savetxt(csv, points, delimiter=",", header="x,y,z", comments="")
    vertex_ply, mesh_ply = tmp_path / "points.ply", tmp_path / "mesh.ply"
    geometry_io.write_labeled_ply(vertex_ply, points)
    geometry_io.write_labeled_ply(mesh_ply, points, triangles=faces)
    return points, faces, [csv, vertex_ply, mesh_ply]


def test_equivalent_format_caps_preserve_full_and_original_geometry(tmp_path):
    points, faces, paths = equivalent_files(tmp_path)
    clouds = [geometry_io.load_root_geometry(path, sample_points=23, random_seed=91) for path in paths]
    for cloud in clouds:
        assert len(cloud.points) == 23
        np.testing.assert_array_equal(cloud.full_points, points)
        np.testing.assert_array_equal(cloud.original_points, points)
        np.testing.assert_array_equal(cloud.points, points[cloud.analysis_indices])
        np.testing.assert_array_equal(cloud.analysis_indices, clouds[0].analysis_indices)
        assert cloud.source_metadata["reduction_reason"] == "explicit_analysis_cap"
        assert cloud.source_metadata["mesh_generated"] is False
    np.testing.assert_array_equal(clouds[-1].triangles, faces)
    np.testing.assert_array_equal(clouds[-1].original_triangles, faces)
    assert clouds[-1].source_metadata["native_mesh_support"] is True
    assert clouds[0].source_metadata["input_mode"] == "surface_points"
    assert clouds[-1].source_metadata["input_mode"] == "triangle_mesh"


def test_automatic_preflight_is_shared_across_formats(tmp_path, monkeypatch):
    points, _, paths = equivalent_files(tmp_path)
    calls = []
    def target(values, **kwargs):
        calls.append(values.copy())
        return 31, 1900.0, "projected_runtime_over_limit"
    monkeypatch.setattr(geometry_io, "_automatic_analysis_target", target)
    clouds = [geometry_io.load_root_geometry(path, sample_points=0) for path in paths]
    assert len(calls) == 3
    for cloud, values in zip(clouds, calls):
        np.testing.assert_array_equal(values, points)
        np.testing.assert_array_equal(cloud.points, clouds[0].points)
        assert len(cloud.points) == 31


@pytest.mark.parametrize("suffix", ["csv", "xyz", "txt", "pts"])
def test_text_caps_and_nonfinite_original_preservation(tmp_path, suffix):
    values = np.arange(300, dtype=float).reshape(-1, 3)
    values[17, 0] = np.nan
    path = tmp_path / f"input.{suffix}"
    np.savetxt(path, values, delimiter="," if suffix == "csv" else " ")
    cloud = geometry_io.load_root_geometry(path, sample_points=10, input_mode="occupied_volume")
    assert len(cloud.points) == 10
    assert len(cloud.full_points) == 99
    np.testing.assert_array_equal(cloud.original_points, values)
    assert cloud.source_metadata["input_mode"] == "occupied_volume"
    assert cloud.source_metadata["nonfinite_vertex_count"] == 1
    assert not cloud.source_metadata["native_mesh_support"]
    assert "not_calibrated" in cloud.source_metadata["measurement_interpretation"]


def test_input_contract_requires_native_faces_and_cannot_discard_them(tmp_path):
    _, _, paths = equivalent_files(tmp_path)
    with pytest.raises(ValueError, match="requires native"):
        geometry_io.load_root_geometry(paths[0], input_mode="triangle_mesh")
    for mode in ["surface_points", "occupied_volume"]:
        with pytest.raises(ValueError, match="contains native faces"):
            geometry_io.load_root_geometry(paths[2], input_mode=mode)
    with pytest.raises(ValueError, match="input_mode"):
        geometry_io.load_root_geometry(paths[0], input_mode="guess")


def test_cli_exposes_representation_and_common_cap():
    args = build_parser().parse_args(["run", "--input", "input.csv", "--output", "result", "--input-mode", "occupied_volume", "--sample-points", "100"])
    assert args.input_mode == "occupied_volume"
    assert args.sample_points == 100
