from __future__ import annotations

from functools import partial
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
import time
import threading
import json

import psutil
import pytest

from soyrootbio.batch import BatchEventType, BatchJobState, BatchScheduler, CooperativeToken, StepTimingHistory
from soyrootbio.batch_process import ProcessSampleResult, SampleProcessError, _run_sample_process, run_pipeline_process
from soyrootbio.pipeline import PipelineConfig
from soyrootbio.hardware import GIB
from soyrootbio.memory_budget import MEMORY_BUDGET_POLICY, MemoryAdmission, MemorySnapshot


def test_observed_private_peak_increases_admission_budget_with_headroom(tmp_path, monkeypatch):
    from soyrootbio import batch_process

    gate = MemoryAdmission(lambda: MemorySnapshot(11 * GIB, 30 * GIB, 32 * GIB, 64 * GIB))
    config = PipelineConfig(tmp_path / 'root.ply', tmp_path / 'out', worker_threads=2)
    with BatchScheduler(run_pipeline_process, memory_admission=gate) as scheduler:
        job = scheduler.submit(config.input_path, config.output_dir, payload=config)
        job.memory_estimate_bytes = 1_932_735_284  # Reduced 1.8 GiB initial estimate
        job._memory_admission = gate
        assert gate.try_acquire(job.job_id, job.memory_estimate_bytes)
        info = SimpleNamespace(private=4 * GIB, rss=GIB, peak_pagefile=4 * GIB, peak_wset=GIB)
        monkeypatch.setattr(psutil, 'Process', lambda pid: SimpleNamespace(memory_info=lambda: info))
        monkeypatch.setattr(batch_process, 'memory_snapshot', lambda: MemorySnapshot())

        def inspect_worker(config, control, progress, *, resource_callback):
            monitor = batch_process._ResourceMonitor(config.output_dir, resource_callback)
            monitor.decoded({'full_vertex_count':101102, 'triangle_count':202202,
                             'original_vertex_count':101102})
            monitor.sample(123, force=True)
            assert monitor.data['memory_budget_policy'] == MEMORY_BUDGET_POLICY
            assert monitor.data['geometry_estimated_peak_bytes'] == 1_932_735_284
            assert monitor.data['estimated_peak_bytes'] == 5 * GIB
            assert job.memory_estimate_bytes == 5 * GIB
            assert gate.try_acquire('neighbor', 3 * GIB)
            assert not gate.try_acquire('third', 3 * GIB)
            # Decreasing current use and re-reporting decoded geometry must
            # never erase the high-water budget for this active sample.
            info.private = info.rss = GIB // 2
            monitor.decoded({'full_vertex_count':101102, 'triangle_count':202202,
                             'original_vertex_count':101102})
            monitor.sample(123, force=True)
            assert monitor.data['estimated_peak_bytes'] == 5 * GIB
            assert job.memory_estimate_bytes == 5 * GIB
            monitor.save()
            saved = json.loads((config.output_dir / 'processing_resources.json').read_text())
            assert saved['estimated_peak_bytes'] == 5 * GIB
            return ProcessSampleResult(config.output_dir, 101102, 123)

        monkeypatch.setattr(batch_process, '_run_sample_process', inspect_worker)
        assert run_pipeline_process(job, CooperativeToken(), lambda *args: None).point_count == 101102


def _probe_analyze(config, control, progress):
    """Real spawn worker; files let tests observe work independently of GUI IPC."""
    (config.output_dir / 'pid').write_text(str(os.getpid()))
    progress('Ready', 0.1)
    if config.input_path.stem == 'crash':
        os._exit(17)
    if config.input_path.stem == 'fail':
        raise ValueError('sample failure 测试')
    if config.input_path.stem == 'ignore':
        time.sleep(60)
    tick = 0
    while not (config.output_dir / 'release').exists():
        control.checkpoint()
        tick += 1
        (config.output_dir / 'heartbeat').write_text(str(tick))
        progress('Working', min(0.9, 0.1 + tick / 1000))
        time.sleep(0.02)
    return ProcessSampleResult(config.output_dir, 123, os.getpid())


def _failure_with_slow_cleanup(config, control, progress):
    threading.Thread(target=time.sleep, args=(30,), daemon=False).start()
    progress('Tracing order-1 lateral roots', .51)
    raise MemoryError('original allocation failure')


def test_received_failure_survives_worker_cleanup_timeout(tmp_path):
    config = PipelineConfig(tmp_path / 'root.stl', tmp_path / 'out', worker_threads=1)
    with pytest.raises(SampleProcessError, match='MemoryError.*original allocation failure') as caught:
        _run_sample_process(config, CooperativeToken(), lambda *args: None,
                            analyze=_failure_with_slow_cleanup, exit_grace_seconds=.1)
    assert '_failure_with_slow_cleanup' in caught.value.remote_traceback
    assert caught.value.resource_context['cleanup_issue'] == 'worker did not exit after reporting failure'
    report = json.loads((config.output_dir / 'processing_resources.json').read_text())
    assert report['peak_private_bytes'] > 0
    assert report['terminal_state'] == 'failed'
    assert report['last_reported_step'] == 'Tracing order-1 lateral roots'
    assert not psutil.pid_exists(report['process_id'])


def test_unavailable_system_telemetry_does_not_hide_sample_error(tmp_path, monkeypatch):
    from soyrootbio import batch_process
    def unavailable():
        raise OSError('memory counters unavailable')
    monkeypatch.setattr(batch_process, 'memory_snapshot', unavailable)
    config = PipelineConfig(tmp_path / 'fail.ply', tmp_path / 'out', worker_threads=1)
    config.output_dir.mkdir()
    with pytest.raises(SampleProcessError, match='sample failure'):
        _run_sample_process(config, CooperativeToken(), lambda *args: None, analyze=_probe_analyze)
    report = json.loads((config.output_dir / 'processing_resources.json').read_text())
    assert report['system_memory_unavailable']


def _probe_runner(job, control, progress, *, grace=10.0):
    return _run_sample_process(job.payload, control, progress,
                               analyze=_probe_analyze, cancel_grace_seconds=grace)


def _until(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail('Timed out waiting for sample process')


def _job(scheduler, tmp_path, name):
    config = PipelineConfig(tmp_path / f'{name}.ply', tmp_path / name, worker_threads=1)
    return scheduler.submit(config.input_path, config.output_dir, payload=config)


def _pid(job):
    return int((job.output_dir / 'pid').read_text())


def test_processes_run_concurrently_bound_slots_and_report_compact_results(tmp_path):
    history = StepTimingHistory(tmp_path / 'history.json')
    with BatchScheduler(_probe_runner, max_concurrent_samples=2, timing_history=history) as scheduler:
        a, b, c = [_job(scheduler, tmp_path, name) for name in ('a', 'b', 'c')]
        scheduler.start()
        try:
            _until(lambda: a.progress >= 0.1 and b.progress >= 0.1)
            pids = {_pid(a), _pid(b)}
            assert len(pids) == 2 and os.getpid() not in pids
            assert all(psutil.pid_exists(pid) for pid in pids)
            assert c.started_at is None and not (c.output_dir / 'pid').exists()
            for job in (a, b):
                (job.output_dir / 'release').touch()
            _until(lambda: c.progress >= 0.1)
            (c.output_dir / 'release').touch()
            assert scheduler.wait(20)
            assert all(job.state == BatchJobState.COMPLETED for job in (a, b, c))
            assert all(isinstance(job.result, ProcessSampleResult) for job in (a, b, c))
            assert all(not psutil.pid_exists(_pid(job)) for job in (a, b, c))
            assert all(job.progress == 1 for job in (a, b, c))
            assert len(history.samples(StepTimingHistory.TOTAL_STEP)) == 3
            kinds = {event.kind for event in scheduler.drain_events()}
            assert {BatchEventType.STARTED, BatchEventType.PROGRESS, BatchEventType.COMPLETED} <= kinds
        finally:
            scheduler.cancel_all()


def test_process_pause_resume_freezes_work_and_elapsed_then_cancel_reaps_child(tmp_path):
    with BatchScheduler(_probe_runner) as scheduler:
        job = _job(scheduler, tmp_path, 'pause')
        scheduler.start()
        try:
            _until(lambda: job.progress > 0.102)
            pid = _pid(job)
            assert scheduler.pause(job.job_id)
            time.sleep(0.15)  # Allow the current bounded operation to finish.
            heartbeat = (job.output_dir / 'heartbeat').read_text()
            elapsed = job.elapsed_seconds
            time.sleep(0.2)
            assert (job.output_dir / 'heartbeat').read_text() == heartbeat
            assert job.elapsed_seconds == pytest.approx(elapsed, abs=0.025)
            assert job.state == BatchJobState.PAUSED
            assert scheduler.resume(job.job_id)
            _until(lambda: (job.output_dir / 'heartbeat').read_text() != heartbeat)
            assert scheduler.pause(job.job_id)
            assert scheduler.cancel(job.job_id)  # Cancellation wakes a paused child.
            assert scheduler.wait(20)
            assert job.state == BatchJobState.CANCELLED
            assert not psutil.pid_exists(pid)
            assert job.error_log_path is None
        finally:
            scheduler.cancel_all()


@pytest.mark.parametrize('name', ['fail', 'crash'])
def test_child_failure_preserves_traceback_or_exit_code_and_other_jobs_continue(tmp_path, name):
    with BatchScheduler(_probe_runner, max_concurrent_samples=2) as scheduler:
        bad = _job(scheduler, tmp_path, name)
        good = _job(scheduler, tmp_path, 'good')
        scheduler.start()
        try:
            # The error report is persisted after the mutable state changes,
            # before the FAILED event is published to the GUI.
            _until(lambda: bad.state == BatchJobState.FAILED
                   and bad.error_log_path is not None and good.progress >= 0.1)
            report = bad.error_log_path.read_text(encoding='utf-8')
            if name == 'fail':
                assert 'ValueError' in report and 'sample failure 测试' in report
                assert '_probe_analyze' in report and 'Sample process traceback:' in report
            else:
                assert 'exit code 17' in report
            assert not psutil.pid_exists(_pid(bad))
            (good.output_dir / 'release').touch()
            assert scheduler.wait(20)
            assert good.state == BatchJobState.COMPLETED
        finally:
            scheduler.cancel_all()


def test_cancel_queued_sample_does_not_start_process(tmp_path):
    with BatchScheduler(_probe_runner) as scheduler:
        active = _job(scheduler, tmp_path, 'active')
        queued = _job(scheduler, tmp_path, 'queued')
        scheduler.start()
        try:
            _until(lambda: active.progress >= 0.1)
            assert scheduler.pause(queued.job_id)
            assert scheduler.cancel(queued.job_id)
            (active.output_dir / 'release').touch()
            assert scheduler.wait(20)
            assert queued.state == BatchJobState.CANCELLED
            assert queued.started_at is None
            assert not queued.output_dir.exists()
        finally:
            scheduler.cancel_all()


def test_shutdown_reaps_unresponsive_child_and_cancels_queue(tmp_path):
    scheduler = BatchScheduler(partial(_probe_runner, grace=0.25))
    active = _job(scheduler, tmp_path, 'ignore')
    queued = _job(scheduler, tmp_path, 'queued')
    scheduler.start()
    try:
        _until(lambda: active.progress >= 0.1)
        pid = _pid(active)
        scheduler.shutdown(wait=False, cancel_pending=True)
        assert scheduler.wait(10)
        assert not psutil.pid_exists(pid)
        assert active.state == queued.state == BatchJobState.CANCELLED
        assert not queued.output_dir.exists()
    finally:
        scheduler.shutdown(wait=True, cancel_pending=True)


def test_process_start_failure_is_reported_and_event_mirror_detaches(tmp_path, monkeypatch):
    from multiprocessing.process import BaseProcess
    def fail_start(self):
        raise OSError('Cannot spawn sample')
    monkeypatch.setattr(BaseProcess, 'start', fail_start)
    with BatchScheduler(_probe_runner) as scheduler:
        job = _job(scheduler, tmp_path, 'start-failure')
        scheduler.start()
        assert scheduler.wait(5)
        assert job.state == BatchJobState.FAILED
        assert 'Cannot spawn sample' in job.error
        assert not job.control._event_mirrors


def test_control_mirror_initial_state_and_cleanup():
    token = CooperativeToken()
    token.pause()
    context = mp.get_context('spawn')
    paused, cancelled = context.Event(), context.Event()
    with token.mirror_to(paused, cancelled):
        assert paused.is_set() and not cancelled.is_set()
        token.resume()
        assert not paused.is_set()
        token.pause()
        token.cancel()
        assert cancelled.is_set() and not paused.is_set()
    assert not token._event_mirrors


def test_real_pipeline_failure_crosses_spawn_boundary(tmp_path):
    with BatchScheduler(run_pipeline_process) as scheduler:
        job = _job(scheduler, tmp_path, 'missing-file')
        scheduler.start()
        assert scheduler.wait(30)
        assert job.state == BatchJobState.FAILED
        report = job.error_log_path.read_text(encoding='utf-8')
        assert '_analyze_sample' in report and 'load_root_geometry' in report
        assert 'missing-file.ply' in report
