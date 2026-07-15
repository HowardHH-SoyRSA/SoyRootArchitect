from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from soyrootbio.cli import build_parser
from soyrootbio.desktop_gui import (
    estimate_remaining_seconds,
    format_eta,
    validate_launcher_settings,
)
from soyrootbio.endpoint_picker import _parse_sample_count, write_endpoint_file
from soyrootbio.io import load_root_geometry_with_progress
from soyrootbio.pipeline import AnalysisCancelled, PipelineConfig, run_pipeline


def _root_file(tmp_path: Path) -> Path:
    source = tmp_path / "root.ply"
    source.write_text("ply", encoding="ascii")
    return source


def test_launcher_accepts_only_the_retained_settings(tmp_path: Path):
    source = _root_file(tmp_path)
    output = tmp_path / "outputs"

    interactive = validate_launcher_settings(source, output, "5,000", "3000")
    assert interactive.endpoint_mode == "interactive"
    assert interactive.sample_points == 5000
    assert interactive.display_points == 3000

    coordinates = validate_launcher_settings(
        source,
        output,
        5000,
        3000,
        endpoint_mode="coordinates",
        start_coordinates=("1", "2", "3"),
        end_coordinates=("4", "5", "6"),
    )
    assert coordinates.start == (1.0, 2.0, 3.0)
    assert coordinates.end == (4.0, 5.0, 6.0)

    automatic = validate_launcher_settings(source, output, 5000, 3000, endpoint_mode="auto")
    assert automatic.auto_endpoints == "z"


def test_launcher_rejects_invalid_endpoint_values(tmp_path: Path):
    source = _root_file(tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        validate_launcher_settings(
            source,
            tmp_path / "out",
            5000,
            3000,
            endpoint_mode="coordinates",
            start_coordinates=(1, 2, 3),
            end_coordinates=(1, 2, 3),
        )
    with pytest.raises(ValueError, match="interactive, coordinates, or auto"):
        validate_launcher_settings(source, tmp_path / "out", 5000, 3000, endpoint_mode="pca")


def test_gui_source_contains_approved_controls_only():
    import soyrootbio.desktop_gui as desktop_gui

    source = Path(desktop_gui.__file__).read_text(encoding="utf-8")
    for required in (
        "Input root file",
        "Output directory",
        "Mesh samples",
        "Display points",
        "Primary-root endpoints",
        "Automatic Z-axis extrema",
        "Tip direction vs downward Z (0, 0, -1)",
        "ETA",
        "Activity",
    ):
        assert required in source
    for rejected in (
        "PCA-axis extrema",
        "PCA-projected",
        "Maximum laterals",
        "Unit scale",
        "Root order:",
    ):
        assert rejected not in source


def test_progress_eta_helpers_and_pipeline_hooks(tmp_path: Path):
    assert estimate_remaining_seconds(10.0, 0.5) == pytest.approx(10.0)
    assert format_eta(65) == "ETA 01:05"
    signature = inspect.signature(run_pipeline)
    assert {"preloaded_cloud", "progress_callback", "cancel_check"}.issubset(signature.parameters)

    config = PipelineConfig(tmp_path / "missing.ply", tmp_path / "out", auto_endpoints="z")
    with pytest.raises(AnalysisCancelled):
        run_pipeline(config, cancel_check=lambda: True)


def test_progress_loader_and_endpoint_writer(tmp_path: Path):
    source = tmp_path / "points.csv"
    source.write_text("x,y,z\n" + "\n".join(f"{i},0,0" for i in range(10)), encoding="utf-8")
    events: list[tuple[str, float, float | None]] = []
    cloud = load_root_geometry_with_progress(source, sample_points=10, progress_callback=lambda *event: events.append(event))
    assert len(cloud.points) == 10
    assert events[-1] == ("Point cloud ready", 1.0, 0.0)

    endpoint_path = write_endpoint_file(
        tmp_path / "endpoints.csv",
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
    )
    assert endpoint_path.exists()
    assert _parse_sample_count("5,000") == 5000


def test_cli_and_shortcut_launcher_expose_gui(tmp_path: Path):
    parsed = build_parser().parse_args(["gui", "--input", "root.ply", "--output", "outputs"])
    assert parsed.command == "gui"

    launcher = Path(__file__).parents[1] / "scripts" / "launch_gui.pyw"
    assert launcher.exists()
    launcher_source = launcher.read_text(encoding="utf-8")
    assert 'SOURCE_DIRECTORY = PROJECT_DIRECTORY / "src"' in launcher_source
