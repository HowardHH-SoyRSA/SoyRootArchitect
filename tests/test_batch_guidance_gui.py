from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from soyrootbio import batch, batch_gui
from soyrootbio.batch_gui import BioInsAlgoBatchApp, SampleEntry
from soyrootbio.primary_guidance import PrimaryGuidance, write_primary_guidance


class Variable:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Tree:
    def __init__(self):
        self.selected = ()
        self.rows = {}
        self.cells = {}

    def insert(self, parent, position, *, values):
        item = f"item-{len(self.rows)}"
        self.rows[item] = values
        return item

    def selection(self):
        return self.selected

    def selection_set(self, items):
        self.selected = tuple(items)

    def see(self, item):
        pass

    def set(self, item, column, value):
        self.cells[item, column] = value

    def exists(self, item):
        return item in self.rows


def _app(tmp_path: Path) -> BioInsAlgoBatchApp:
    app = BioInsAlgoBatchApp.__new__(BioInsAlgoBatchApp)
    app.scheduler = None
    app.entries = {}
    app.tree = Tree()
    for name, value in {
        "output_root_var": str(tmp_path / "outputs"),
        "primary_method_var": "Scored automatic",
        "status_var": "", "soil_z_var": "", "sample_cap_var": "0",
        "max_order_var": "3", "runtime_limit_var": "30",
        "minimum_fraction_var": "25", "tip_window_var": "2.0",
    }.items():
        setattr(app, name, Variable(value))
    return app


def _add_sample(app, tmp_path):
    source = tmp_path / "root.ply"
    source.write_text("ply", encoding="ascii")
    app.add_files([source])
    return next(iter(app.entries.values()))


def test_start_batch_dispatches_module_level_process_runner(tmp_path, monkeypatch):
    app = _app(tmp_path)
    _add_sample(app, tmp_path)
    app.concurrency_var = Variable('2')
    app.threads_var = Variable('1')
    app.hardware_var = Variable()
    app.job_to_item = {}
    # Keep Tk out of spawn payloads and avoid launching a full analysis here.
    monkeypatch.setattr(batch_gui.BatchScheduler, 'start', lambda self: None)
    errors = []
    monkeypatch.setattr(batch_gui.messagebox, 'showerror', lambda *args: errors.append(args))
    app.start_batch()
    assert not errors
    assert app.scheduler.runner is batch_gui.run_pipeline_process
    assert app.scheduler.threads_per_sample == 1
    assert 'sample process' in app.status_var.get()
    app.scheduler.shutdown(cancel_pending=True)


def test_added_sample_can_load_guidance_as_a_per_sample_override(tmp_path, monkeypatch):
    app = _app(tmp_path)
    entry = _add_sample(app, tmp_path)
    assert app.tree.selection() == (entry.item_id,)
    assert len(app.tree.rows[entry.item_id]) == 6
    selected = PrimaryGuidance(np.array([0., 0., 1.]), np.zeros(3), 1.1, np.array([[0., 0., 0.6]]))
    selection_file = write_primary_guidance(tmp_path / "old-run" / "primary_guidance.json", selected, input_path=entry.input_path)
    monkeypatch.setattr(batch_gui.filedialog, "askopenfilename", lambda **kwargs: str(selection_file))
    app.load_selected_guidance()
    selection_file.unlink()  # Imported values must be self-contained in memory.
    config = app._pipeline_config(entry, 2)
    assert config.start == (0., 0., 1.)
    assert config.end == (0., 0., 0.)
    assert config.primary_guides == ((0., 0., 0.6),)
    assert config.soil_z == 1.1
    assert config.auto_endpoints is None
    assert config.guide_file is None
    assert "Loaded endpoints" in app.tree.cells[entry.item_id, "primary"]


def test_loading_wrong_sample_is_atomic_and_active_batches_cannot_change(tmp_path, monkeypatch):
    app = _app(tmp_path)
    entry = _add_sample(app, tmp_path)
    original = PrimaryGuidance(np.ones(3), np.zeros(3), None, np.empty((0, 3)))
    entry.guidance = original
    wrong = write_primary_guidance(tmp_path / "wrong.json", original, input_path="other.ply")
    dialogs = []
    errors = []
    monkeypatch.setattr(batch_gui.filedialog, "askopenfilename", lambda **kwargs: dialogs.append(kwargs) or str(wrong))
    monkeypatch.setattr(batch_gui.messagebox, "showerror", lambda title, text: errors.append(text))
    monkeypatch.setattr(batch_gui.messagebox, "showinfo", lambda *args: None)
    app.load_selected_guidance()
    assert "other.ply, not root.ply" in errors[-1]
    assert entry.guidance is original
    app.scheduler = SimpleNamespace(all_done=False)
    app.load_selected_guidance()
    assert len(dialogs) == 1


def test_runtime_updates_without_progress_events_and_freezes_on_pause_and_finish(tmp_path, monkeypatch):
    app = _app(tmp_path)
    entry = _add_sample(app, tmp_path)
    clock = [100.0]
    monkeypatch.setattr(batch.time, "monotonic", lambda: clock[0])
    job = batch.BatchJob("job", entry.input_path, entry.output_dir, 2)
    # An earlier pause while queued must not be subtracted from active time.
    job.control.pause()
    clock[0] = 130.0
    job.control.resume()
    job._paused_seconds_at_start = job.control.paused_seconds
    job.started_at = 130.0
    job._started_monotonic = 130.0
    app.job_to_item = {job.job_id: entry.item_id}
    app.scheduler = SimpleNamespace(jobs=(job,), all_done=False, drain_events=lambda: [])
    app.root = SimpleNamespace(after=lambda *args: None)

    clock[0] = 195.0
    app._poll_scheduler()
    assert app.tree.cells[entry.item_id, "runtime"] == "01:05"
    job.control.pause()
    clock[0] = 225.0
    app._poll_scheduler()
    assert app.tree.cells[entry.item_id, "runtime"] == "01:05"
    job.control.resume()
    clock[0] = 230.0
    job.finished_at = 230.0
    job._finished_monotonic = 230.0
    clock[0] = 260.0
    app._poll_scheduler()
    assert app.tree.cells[entry.item_id, "runtime"] == "01:10"
    assert {column for item, column in app.tree.cells} == {"runtime"}
    assert app._format_runtime(None) == "--"
    assert app._format_runtime(3661) == "1:01:01"


def test_batch_completion_notifies_once_only_after_every_sample_succeeds(tmp_path, monkeypatch):
    app = _app(tmp_path)
    first = _add_sample(app, tmp_path)
    second_path = tmp_path / "second.ply"
    second_path.write_text("ply", encoding="ascii")
    app.add_files([second_path])
    second = next(entry for entry in app.entries.values() if entry is not first)
    jobs = (
        batch.BatchJob("first-job", first.input_path, first.output_dir, 1, state=batch.BatchJobState.COMPLETED),
        batch.BatchJob("second-job", second.input_path, second.output_dir, 1, state=batch.BatchJobState.COMPLETED),
    )
    app.job_to_item = {
        jobs[0].job_id: first.item_id,
        jobs[1].job_id: second.item_id,
    }
    app.scheduler = SimpleNamespace(jobs=jobs, all_done=True, drain_events=lambda: [])
    app.root = SimpleNamespace(after=lambda *args: None)
    notices = []
    monkeypatch.setattr(
        batch_gui.messagebox,
        "showinfo",
        lambda title, text, **kwargs: notices.append((title, text, kwargs)),
    )

    app._poll_scheduler()
    app._poll_scheduler()

    assert notices == [
        (
            "Batch complete",
            "All 2 sample(s) completed successfully.",
            {"parent": app.root},
        )
    ]


def test_batch_completion_notification_excludes_failed_batches(tmp_path, monkeypatch):
    app = _app(tmp_path)
    entry = _add_sample(app, tmp_path)
    job = batch.BatchJob("failed-job", entry.input_path, entry.output_dir, 1, state=batch.BatchJobState.FAILED)
    app.job_to_item = {job.job_id: entry.item_id}
    app.scheduler = SimpleNamespace(jobs=(job,), all_done=True, drain_events=lambda: [])
    app.root = SimpleNamespace(after=lambda *args: None)
    notices = []
    monkeypatch.setattr(batch_gui.messagebox, "showinfo", lambda *args, **kwargs: notices.append(args))

    app._poll_scheduler()

    assert notices == []


def test_first_failed_sample_plays_one_native_alert_while_batch_continues(tmp_path, monkeypatch):
    app = _app(tmp_path)
    failed = _add_sample(app, tmp_path)
    running_path = tmp_path / "still-running.ply"
    running_path.write_text("ply", encoding="ascii")
    app.add_files([running_path])
    running = next(entry for entry in app.entries.values() if entry is not failed)
    failed_job = batch.BatchJob(
        "failed-job",
        failed.input_path,
        failed.output_dir,
        1,
        state=batch.BatchJobState.FAILED,
    )
    running_job = batch.BatchJob(
        "running-job",
        running.input_path,
        running.output_dir,
        1,
        state=batch.BatchJobState.RUNNING,
    )
    failed_event = batch.BatchEvent(batch.BatchEventType.FAILED, failed_job.snapshot())
    app.job_to_item = {
        failed_job.job_id: failed.item_id,
        running_job.job_id: running.item_id,
    }
    app.scheduler = SimpleNamespace(
        jobs=(failed_job, running_job),
        all_done=False,
        drain_events=lambda: [failed_event],
    )
    app.root = SimpleNamespace(after=lambda *args: None)
    sounds = []
    monkeypatch.setattr(
        batch_gui,
        "winsound",
        SimpleNamespace(MB_ICONHAND="error", MessageBeep=sounds.append),
    )

    app._poll_scheduler()
    app._poll_scheduler()

    assert sounds == ["error"]
    assert app.tree.cells[failed.item_id, "status"] == "Failed"
    assert running_job.state == batch.BatchJobState.RUNNING
