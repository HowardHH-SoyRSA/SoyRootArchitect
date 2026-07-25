from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from soyrootbio.cli import build_parser
from soyrootbio.desktop_gui import (
    estimate_remaining_seconds,
    format_eta,
    validate_launcher_settings,
)
from soyrootbio.endpoint_picker import (
    _parse_sample_count,
    _wheel_zoom_factor,
    _zoom_3d_axis,
    write_endpoint_file,
)
from soyrootbio.io import load_root_geometry_with_progress
from soyrootbio.pipeline import AnalysisCancelled, PipelineConfig, run_pipeline


def _root_file(tmp_path: Path) -> Path:
    source = tmp_path / "root.ply"
    source.write_text("ply", encoding="ascii")
    return source


def test_primary_section_wheel_zoom_supports_backend_event_variants() -> None:
    from matplotlib.figure import Figure

    assert _wheel_zoom_factor("up", 0) == pytest.approx(0.75)
    assert _wheel_zoom_factor(None, 1) == pytest.approx(0.75)
    assert _wheel_zoom_factor("down", 0) == pytest.approx(1.35)
    assert _wheel_zoom_factor(None, -1) == pytest.approx(1.35)
    assert _wheel_zoom_factor(None, 0) is None

    axis = Figure().add_subplot(projection="3d")
    axis.set_xlim(0.0, 8.0)
    axis.set_ylim(-2.0, 2.0)
    axis.set_zlim(10.0, 14.0)
    _zoom_3d_axis(axis, 0.75)
    assert np.diff(axis.get_xlim())[0] == pytest.approx(6.0)
    assert np.diff(axis.get_ylim())[0] == pytest.approx(3.0)
    assert np.diff(axis.get_zlim())[0] == pytest.approx(3.0)


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


def test_batch_gui_source_exposes_required_controls():
    import soyrootbio.batch_gui as batch_gui

    source = Path(batch_gui.__file__).read_text(encoding="utf-8")
    for required in (
        "Add files",
        "Remove selected",
        "Set selected output",
        "Open output folder",
        "Scored automatic",
        "Manual soil line + scorer",
        "Interactive endpoints + sections",
        "Analysis vertex cap",
        "Display points",
        "Angle vector window (mesh units)",
        "Concurrent samples",
        "Threads / sample",
        "Start batch",
        "Pause selected",
        "Resume selected",
        "Cancel selected",
    ):
        assert required in source
    assert 'orient="horizontal"' in source
    assert "xscrollcommand=horizontal_scrollbar.set" in source
    assert "Voxel size (mm)" not in source
    assert "Mesh unit → mm" not in source
    assert "from threadpoolctl import threadpool_limits" not in source


def test_batch_gui_opens_only_one_existing_output_folder(tmp_path: Path, monkeypatch):
    import soyrootbio.batch_gui as batch_gui
    from soyrootbio.batch_gui import BioInsAlgoBatchApp, SampleEntry

    class Tree:
        selected: tuple[str, ...] = ()

        def selection(self) -> tuple[str, ...]:
            return self.selected

    output = tmp_path / "results"
    output.mkdir()
    app = BioInsAlgoBatchApp.__new__(BioInsAlgoBatchApp)
    app.tree = Tree()
    app.entries = {"sample": SampleEntry("sample", tmp_path / "root.ply", output)}
    opened: list[Path] = []
    notices: list[tuple[str, str]] = []
    monkeypatch.setattr(app, "_open_directory", opened.append)
    monkeypatch.setattr(batch_gui.messagebox, "showinfo", lambda title, text: notices.append((title, text)))

    app.open_output_folder()
    assert not opened
    assert notices[-1][0] == "Select one sample"

    app.tree.selected = ("sample",)
    app.open_output_folder()
    assert opened == [output.resolve()]

    output.rmdir()
    app.open_output_folder()
    assert opened == [output.resolve()]
    assert notices[-1][0] == "Output not available"


def test_batch_gui_blocks_additions_while_active_and_selects_only_new_entries(tmp_path: Path):
    from soyrootbio.batch_gui import BioInsAlgoBatchApp, SampleEntry

    class StatusVar:
        value = ""

        def set(self, value: str) -> None:
            self.value = value

    app = BioInsAlgoBatchApp.__new__(BioInsAlgoBatchApp)
    app.scheduler = SimpleNamespace(all_done=False)
    app.status_var = StatusVar()
    app.add_files([tmp_path / "ignored.ply"])
    assert "Batch active" in app.status_var.value

    app.entries = {
        "done": SampleEntry("done", tmp_path / "done.ply", tmp_path / "done-output", job_id="old-job"),
        "new": SampleEntry("new", tmp_path / "new.ply", tmp_path / "new-output"),
    }
    assert app._next_batch_items() == ("new",)
    app.entries["new"].job_id = "new-job"
    assert app._next_batch_items() == ("done", "new")


def test_batch_gui_rejects_resolved_duplicate_output_paths(tmp_path: Path):
    from soyrootbio.batch_gui import BioInsAlgoBatchApp, SampleEntry

    alias_parent = tmp_path / "alias"
    alias_parent.mkdir()
    app = BioInsAlgoBatchApp.__new__(BioInsAlgoBatchApp)
    app.entries = {
        "a": SampleEntry("a", tmp_path / "a.ply", tmp_path / "output"),
        "b": SampleEntry("b", tmp_path / "b.ply", alias_parent / ".." / "output"),
    }
    with pytest.raises(ValueError, match="assigned to both"):
        app._validate_output_ownership(("a", "b"))


def test_progress_eta_helpers_and_pipeline_hooks(tmp_path: Path):
    assert estimate_remaining_seconds(10.0, 0.5) == pytest.approx(10.0)
    assert format_eta(65) == "ETA 01:05"
    signature = inspect.signature(run_pipeline)
    assert {"preloaded_cloud", "progress_callback", "cancel_check", "pause_check"}.issubset(signature.parameters)

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
