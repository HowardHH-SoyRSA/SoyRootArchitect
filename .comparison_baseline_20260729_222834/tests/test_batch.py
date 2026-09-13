from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import pytest

from soyrootbio.batch import (
    BatchCancelled,
    BatchEventType,
    BatchJobState,
    BatchScheduler,
    CooperativeToken,
    StepTimingHistory,
)
from soyrootbio.hardware import GIB, HardwareInfo, allocate_resources, detect_hardware


def test_hardware_detection_is_safe_without_gpu_dependencies():
    hardware = detect_hardware(include_gpus=False)

    assert hardware.logical_cpus >= 1
    assert 1 <= hardware.physical_cpus <= hardware.logical_cpus
    assert hardware.gpus == ()
    assert hardware.total_memory_bytes is None or hardware.total_memory_bytes >= 0
    assert hardware.available_memory_bytes is None or hardware.available_memory_bytes >= 0
    if hardware.total_memory_bytes is not None and hardware.available_memory_bytes is not None:
        assert hardware.available_memory_bytes <= hardware.total_memory_bytes


def test_resource_allocation_is_deterministic_and_honors_manual_values():
    hardware = HardwareInfo(
        logical_cpus=16,
        physical_cpus=8,
        total_memory_bytes=32 * GIB,
    )

    automatic = allocate_resources(hardware, sample_count=10)
    assert automatic.max_concurrent_samples == 7
    assert automatic.threads_per_sample == 2
    assert automatic.cpu_budget == 15
    assert automatic.memory_limited_samples == 7

    manual = allocate_resources(
        hardware,
        sample_count=10,
        max_concurrent_samples=3,
        threads_per_sample=6,
    )
    assert manual.max_concurrent_samples == 3
    assert manual.threads_per_sample == 6

    thread_override = allocate_resources(hardware, sample_count=10, threads_per_sample=4)
    assert thread_override.max_concurrent_samples == 3
    assert thread_override.threads_per_sample == 4

    # Concurrency is meaningful only up to the number of selected samples.
    one_sample = allocate_resources(hardware, sample_count=1, max_concurrent_samples=8)
    assert one_sample.max_concurrent_samples == 1

    busy_hardware = HardwareInfo(
        logical_cpus=16,
        physical_cpus=8,
        total_memory_bytes=32 * GIB,
        available_memory_bytes=6 * GIB,
    )
    memory_limited = allocate_resources(busy_hardware, sample_count=10)
    assert memory_limited.max_concurrent_samples == 1
    assert memory_limited.memory_limited_samples == 1


def test_timing_history_persists_bounded_step_samples_and_eta(tmp_path: Path):
    path = tmp_path / "timings.json"
    history = StepTimingHistory(path, max_samples_per_step=2)
    history.record("load", 1.0)
    history.record("load", 3.0)
    history.record("load", 5.0)
    history.record(StepTimingHistory.TOTAL_STEP, 20.0)

    restored = StepTimingHistory(path, max_samples_per_step=2)
    assert restored.samples("load") == (3.0, 5.0)
    assert restored.estimate("load") == pytest.approx(4.0)
    assert restored.estimate_eta(0.25) == pytest.approx(15.0)
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not json", encoding="utf-8")
    empty = StepTimingHistory(malformed)
    assert empty.load_error is not None
    assert empty.estimate("anything") is None


def test_cooperative_token_pauses_resumes_and_cancels():
    token = CooperativeToken()
    passed = threading.Event()
    token.pause()

    worker = threading.Thread(target=lambda: (token.checkpoint(), passed.set()))
    worker.start()
    assert not passed.wait(0.05)
    assert token.resume()
    assert passed.wait(1.0)
    worker.join(timeout=1.0)

    assert token.cancel()
    assert token.cancel_check()
    with pytest.raises(BatchCancelled):
        token.checkpoint()


def test_scheduler_bounds_concurrency_reports_events_and_records_timings(tmp_path: Path):
    active = 0
    peak = 0
    lock = threading.Lock()

    def runner(job, control, progress):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            progress("load", 0.25)
            time.sleep(0.03)
            control.checkpoint()
            progress("measure", 0.75)
            time.sleep(0.03)
            return job.output_dir.name
        finally:
            with lock:
                active -= 1

    timing_path = tmp_path / "history" / "steps.json"
    scheduler = BatchScheduler(
        runner,
        max_concurrent_samples=2,
        threads_per_sample=3,
        timing_history=StepTimingHistory(timing_path),
    )
    jobs = scheduler.submit_many(
        [tmp_path / "a" / "root.ply", tmp_path / "b" / "root.ply", tmp_path / "c.ply"],
        tmp_path / "outputs",
    )
    scheduler.start()
    assert scheduler.wait(timeout=3.0)

    assert peak == 2
    assert scheduler.all_done
    assert all(job.state == BatchJobState.COMPLETED for job in jobs)
    assert all(job.progress == 1.0 and job.eta_seconds == 0.0 for job in jobs)
    assert all(job.threads_per_sample == 3 for job in jobs)
    assert [job.output_dir.name for job in jobs] == ["root", "root_2", "c"]
    assert all(job.output_dir.is_dir() for job in jobs)
    assert timing_path.exists()

    events = scheduler.drain_events()
    kinds = {event.kind for event in events}
    assert {BatchEventType.SUBMITTED, BatchEventType.STARTED, BatchEventType.PROGRESS, BatchEventType.COMPLETED} <= kinds
    assert all(event.job.state == event.job.status for event in events)
    scheduler.shutdown()


def test_failed_job_writes_utf8_error_log_with_traceback_and_safe_config(tmp_path: Path):
    @dataclass
    class FailureConfig:
        sample_points: int = 1234
        endpoint_mode: str = "interactive"
        secret_token: str = "must-not-appear"

    def runner(job, control, progress):
        progress("segmentation", 0.4)
        raise RuntimeError("无法处理 SN14")

    output_dir = tmp_path / "failed-output"
    input_path = tmp_path / "SN14_6-2_20260405.ply"
    scheduler = BatchScheduler(runner)
    job = scheduler.submit(
        input_path,
        output_dir,
        payload=FailureConfig(),
    )
    scheduler.start()

    assert scheduler.wait(timeout=2.0)
    snapshot = job.snapshot()
    assert snapshot.state == BatchJobState.FAILED
    assert snapshot.error == "RuntimeError: 无法处理 SN14"
    assert snapshot.error_log_path == output_dir.resolve() / "processing_error.log"
    report = snapshot.error_log_path.read_text(encoding="utf-8")
    assert "timestamp:" in report
    assert "sample: SN14_6-2_20260405.ply" in report
    assert f"input_path: {input_path}" in report
    assert f"output_directory: {output_dir.resolve()}" in report
    assert "exception_type: RuntimeError" in report
    assert "exception_message: 无法处理 SN14" in report
    assert '"sample_points": 1234' in report
    assert '"endpoint_mode": "interactive"' in report
    assert '"secret_token": "<redacted>"' in report
    assert "must-not-appear" not in report
    assert '"last_reported_step": "segmentation"' in report
    assert '"progress_fraction": 0.4' in report
    assert '"threads_per_sample": 1' in report
    assert "traceback:" in report
    assert "RuntimeError: 无法处理 SN14" in report
    scheduler.shutdown()


def test_scheduler_cooperatively_pauses_and_cancels_jobs(tmp_path: Path):
    entered = threading.Event()
    release = threading.Event()

    def runner(job, control, progress):
        progress("first", 0.1)
        entered.set()
        assert release.wait(1.0)
        control.checkpoint()
        progress("second", 0.8)
        return "done"

    scheduler = BatchScheduler(runner)
    job = scheduler.submit(tmp_path / "sample.ply", tmp_path / "paused-output")
    scheduler.start()
    assert entered.wait(1.0)
    assert scheduler.pause(job.job_id)
    release.set()
    time.sleep(0.05)
    assert job.state == BatchJobState.PAUSED
    assert not scheduler.wait(timeout=0.02)
    assert scheduler.resume(job.job_id)
    assert scheduler.wait(timeout=1.0)
    assert job.state == BatchJobState.COMPLETED
    scheduler.shutdown()

    cancel_entered = threading.Event()
    cancel_release = threading.Event()

    def cancelled_runner(job, control, progress):
        progress("first", 0.2)
        cancel_entered.set()
        assert cancel_release.wait(1.0)
        control.checkpoint()

    cancelled_scheduler = BatchScheduler(cancelled_runner)
    cancelled_job = cancelled_scheduler.submit(
        tmp_path / "cancel.ply",
        tmp_path / "cancelled-output",
    )
    cancelled_scheduler.start()
    assert cancel_entered.wait(1.0)
    assert cancelled_scheduler.cancel(cancelled_job.job_id)
    cancel_release.set()
    assert cancelled_scheduler.wait(timeout=1.0)
    assert cancelled_job.state == BatchJobState.CANCELLED
    cancelled_scheduler.shutdown()


def test_pause_between_checkpoint_and_progress_state_update_can_resume(tmp_path: Path):
    class GateToken(CooperativeToken):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.checkpoint_passed = threading.Event()
            self.release = threading.Event()

        def checkpoint(self) -> None:
            self.calls += 1
            super().checkpoint()
            if self.calls == 2:
                self.checkpoint_passed.set()
                assert self.release.wait(2.0)

    progress_returned = threading.Event()

    def runner(job, control, progress):
        progress("work", 0.5)
        progress_returned.set()
        control.checkpoint()
        return "done"

    scheduler = BatchScheduler(runner)
    job = scheduler.submit(tmp_path / "input.ply", tmp_path / "pause-race-output")
    token = GateToken()
    job._control = token
    scheduler.start()

    assert token.checkpoint_passed.wait(2.0)
    assert scheduler.pause(job.job_id)
    token.release.set()
    assert progress_returned.wait(2.0)
    assert job.state == BatchJobState.PAUSED
    assert token.paused
    assert scheduler.resume(job.job_id)
    assert scheduler.wait(timeout=2.0)
    assert job.state == BatchJobState.COMPLETED
    scheduler.shutdown()


def test_cancel_during_completion_commit_cannot_revert_to_completed(tmp_path: Path):
    class BlockingHistory(StepTimingHistory):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def record_many(self, durations, *, save=True):
            self.entered.set()
            assert self.release.wait(2.0)
            return super().record_many(durations, save=save)

    history = BlockingHistory()
    scheduler = BatchScheduler(
        lambda job, control, progress: "done",
        timing_history=history,
    )
    job = scheduler.submit(tmp_path / "input.ply", tmp_path / "cancel-race-output")
    scheduler.start()

    assert history.entered.wait(2.0)
    assert scheduler.cancel(job.job_id)
    history.release.set()
    assert scheduler.wait(timeout=2.0)
    assert job.state == BatchJobState.CANCELLED
    assert job.result is None
    kinds = [event.kind for event in scheduler.drain_events()]
    assert BatchEventType.CANCEL_REQUESTED in kinds
    assert BatchEventType.CANCELLED in kinds
    assert BatchEventType.COMPLETED not in kinds
    scheduler.shutdown()


def test_pause_during_completion_commit_waits_for_resume(tmp_path: Path):
    class BlockingHistory(StepTimingHistory):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def record_many(self, durations, *, save=True):
            self.entered.set()
            assert self.release.wait(2.0)
            return super().record_many(durations, save=save)

    history = BlockingHistory()
    scheduler = BatchScheduler(
        lambda job, control, progress: "done",
        timing_history=history,
    )
    job = scheduler.submit(tmp_path / "input.ply", tmp_path / "pause-completion-output")
    scheduler.start()

    assert history.entered.wait(2.0)
    assert scheduler.pause(job.job_id)
    history.release.set()
    assert not scheduler.wait(timeout=0.05)
    assert job.state == BatchJobState.PAUSED
    assert scheduler.resume(job.job_id)
    assert scheduler.wait(timeout=2.0)
    assert job.state == BatchJobState.COMPLETED
    scheduler.shutdown()


def test_scheduler_rejects_canonical_output_aliases(tmp_path: Path):
    alias_parent = tmp_path / "alias"
    alias_parent.mkdir()
    output = tmp_path / "results"
    alias = alias_parent / ".." / "results"
    scheduler = BatchScheduler(lambda job, control, progress: None)

    first = scheduler.submit(tmp_path / "a.ply", output)
    with pytest.raises(ValueError, match="already assigned"):
        scheduler.submit(tmp_path / "b.ply", alias)

    assert first.output_dir == output.resolve()
    scheduler.shutdown(wait=False, cancel_pending=True)
