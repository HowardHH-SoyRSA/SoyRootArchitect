from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import soyrootbio.pipeline as pipeline
from soyrootbio.primary_guidance import (
    PRIMARY_GUIDANCE_FILENAME,
    PrimaryGuidance,
    read_primary_guidance,
    write_primary_guidance,
)
from soyrootbio.types import Normalization, RootPath


def _guidance() -> PrimaryGuidance:
    return PrimaryGuidance(
        start=np.array([2.967263, 58.345428, 92.801102]),
        end=np.array([-19.679707, 67.221672, -55.30003]),
        soil_z=92.9,
        guides=np.array([
            [2.482088804244995, 57.3425407409668, 86.56834411621094],
            [-20.057720184326172, 75.8220443725586, 2.9997711181640625],
        ]),
    )


def test_guidance_round_trip_keeps_source_precision_and_order(tmp_path: Path) -> None:
    selected = _guidance()
    path = write_primary_guidance(tmp_path / PRIMARY_GUIDANCE_FILENAME, selected, input_path="old/root.ply")
    loaded = read_primary_guidance(path, expected_input="moved/root.ply")
    np.testing.assert_array_equal(loaded.start, selected.start)
    np.testing.assert_array_equal(loaded.end, selected.end)
    np.testing.assert_array_equal(loaded.guides, selected.guides)
    assert loaded.soil_z == selected.soil_z
    assert loaded.use_endpoints is True

    # The combined file can also serve the existing two CLI file options.
    start, end = pipeline.read_endpoint_file(path)
    np.testing.assert_array_equal(start, selected.start)
    np.testing.assert_array_equal(end, selected.end)
    config = pipeline.PipelineConfig(Path("root.ply"), tmp_path / "out", guide_file=path)
    np.testing.assert_array_equal(pipeline._read_primary_guides(config), selected.guides)


def test_guidance_empty_guides_and_non_overwriting_export(tmp_path: Path) -> None:
    selected = PrimaryGuidance(np.zeros(3), np.ones(3), None, np.empty((0, 3)))
    path = write_primary_guidance(tmp_path / PRIMARY_GUIDANCE_FILENAME, selected, input_path="root.ply", overwrite=False)
    assert read_primary_guidance(path).guides.shape == (0, 3)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_primary_guidance(path, selected, input_path="root.ply", overwrite=False)
    assert path.read_bytes() == original


@pytest.mark.parametrize("key,value,match", [
    ("schema_version", 99, "schema version"),
    ("schema_version", True, "schema version"),
    ("coordinate_space", "normalized", "source mesh coordinates"),
    ("coordinate_unit", "mm", "source mesh coordinates"),
    ("start", [1, 2], "Collar"),
    ("end", [1, 2, float("nan")], "Tip"),
    ("guides", [[1, 2]], "Guides"),
    ("guides", [[True, False, True]], "Guides"),
    ("soil_z", float("inf"), "finite number"),
    ("use_endpoints", "true", "true or false"),
])
def test_guidance_rejects_invalid_saved_data(tmp_path: Path, key, value, match) -> None:
    path = write_primary_guidance(tmp_path / PRIMARY_GUIDANCE_FILENAME, _guidance(), input_path="root.ply")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[key] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        read_primary_guidance(path)


def test_guidance_rejects_wrong_sample_and_coincident_endpoints(tmp_path: Path) -> None:
    path = write_primary_guidance(tmp_path / PRIMARY_GUIDANCE_FILENAME, _guidance(), input_path=r"C:\roots\first.ply")
    with pytest.raises(ValueError, match="first.ply, not second.ply"):
        read_primary_guidance(path, expected_input=tmp_path / "second.ply")
    invalid = PrimaryGuidance(np.ones(3), np.ones(3), None, np.empty((0, 3)))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="distinct"):
        write_primary_guidance(path, invalid, input_path="first.ply")
    assert path.read_bytes() == original


def test_pipeline_saves_manual_guidance_before_loading_geometry(tmp_path: Path, monkeypatch) -> None:
    selected = _guidance()
    source_file = write_primary_guidance(tmp_path / "selected.json", selected, input_path="root.ply")
    config = pipeline.PipelineConfig(
        Path("root.ply"), tmp_path / "manual-output",
        endpoint_file=source_file, guide_file=source_file, soil_z=selected.soil_z,
    )

    def stop_loading(*args, **kwargs):
        raise RuntimeError("test stopped after saving guidance")

    monkeypatch.setattr(pipeline, "load_root_geometry", stop_loading)
    with pytest.raises(RuntimeError, match="test stopped"):
        pipeline.run_pipeline(config)
    loaded = read_primary_guidance(config.output_dir / PRIMARY_GUIDANCE_FILENAME)
    np.testing.assert_array_equal(loaded.start, selected.start)
    np.testing.assert_array_equal(loaded.guides, selected.guides)

    automatic = pipeline.PipelineConfig(Path("root.ply"), tmp_path / "automatic-output", auto_endpoints="z")
    with pytest.raises(RuntimeError, match="test stopped"):
        pipeline.run_pipeline(automatic)
    assert not (automatic.output_dir / PRIMARY_GUIDANCE_FILENAME).exists()


def test_pipeline_uses_the_saved_snapshot_not_reread_files(tmp_path: Path, monkeypatch) -> None:
    selected = _guidance()
    source_file = write_primary_guidance(tmp_path / "selected.json", selected, input_path="root.ply")
    config = pipeline.PipelineConfig(Path("root.ply"), tmp_path / "out", endpoint_file=source_file, guide_file=source_file)
    snapshot = pipeline._manual_primary_guidance(config)
    source_file.unlink()
    received = {}

    def capture_path(points, start, end, **kwargs):
        received.update(start=start, end=end, guides=kwargs["waypoints"])
        return RootPath(root_id="primary", points=np.vstack([start, end]))

    monkeypatch.setattr(pipeline, "estimate_primary_path", capture_path)
    points = np.vstack([selected.start, selected.end])
    pipeline._resolve_primary_path(
        points, points, Normalization(np.zeros(3), 1.0), 0.01, config,
        manual_guidance=snapshot,
    )
    np.testing.assert_array_equal(received["start"], selected.start)
    np.testing.assert_array_equal(received["end"], selected.end)
    np.testing.assert_array_equal(received["guides"], selected.guides)
