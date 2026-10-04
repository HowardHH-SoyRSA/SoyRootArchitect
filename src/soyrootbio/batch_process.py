"""Spawn one isolated analysis process per active desktop batch slot.

The scheduler's threads only supervise processes and dispatch small messages.
Tk objects, locks, point clouds and full PipelineResult arrays never cross the
process boundary.  Each child exits after its sample, releasing native memory.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import multiprocessing as mp
import json
from pathlib import Path
import os
import time
import traceback
from typing import Any, Callable

from .batch import BatchCancelled, BatchJob, CooperativeToken, ProgressCallback
from .pipeline import PipelineConfig, run_pipeline
from .memory_budget import estimate_decoded_memory, memory_snapshot


@dataclass(frozen=True)
class ProcessSampleResult:
    output_dir: Path
    point_count: int
    process_id: int


class SampleProcessError(RuntimeError):
    """A child exception or abnormal exit, with its original traceback."""

    def __init__(self, message: str, remote_traceback: str = "", resource_context=None) -> None:
        super().__init__(message)
        self.remote_traceback = remote_traceback
        self.resource_context = resource_context or {}


class _ProcessControl:
    def __init__(self, paused: Any, cancelled: Any, messages=None) -> None:
        self._paused = paused
        self._cancelled = cancelled
        self._messages = messages

    def report_resources(self, values: dict) -> None:
        if self._messages is not None:
            self._messages.send(("resources", values))

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def checkpoint(self) -> None:
        while self._paused.is_set() and not self._cancelled.is_set():
            self._cancelled.wait(0.05)
        if self._cancelled.is_set():
            raise BatchCancelled("Batch job cancelled")

    def cancel_check(self) -> bool:
        self.checkpoint()
        return False


def _analyze_sample(config: PipelineConfig, control: _ProcessControl, progress):
    # Unlike overlapping thread contexts, each process owns its native pools.
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=config.worker_threads):
        result = run_pipeline(
            config,
            progress_callback=progress,
            cancel_check=control.cancel_check,
            pause_check=lambda: control.paused,
            resource_callback=control.report_resources,
        )
    return ProcessSampleResult(result.output_dir, result.point_count, os.getpid())


def _sample_process_main(config, paused, cancelled, messages, analyze) -> None:
    """Importable Windows spawn target; never receives the Tk application."""
    control = _ProcessControl(paused, cancelled, messages)

    def progress(step: str, fraction: float) -> None:
        control.checkpoint()
        messages.send(("progress", str(step), float(fraction)))

    try:
        control.checkpoint()
        result = analyze(config, control, progress)
        control.checkpoint()
        messages.send(("completed", result))
    except BaseException as exc:
        if cancelled.is_set() or isinstance(exc, BatchCancelled):
            messages.send(("cancelled",))
        else:
            messages.send(("failed", type(exc).__name__, str(exc), traceback.format_exc()))
    finally:
        messages.close()


def run_pipeline_process(
    job: BatchJob,
    control: CooperativeToken,
    progress: ProgressCallback,
) -> ProcessSampleResult:
    """Batch runner using a dedicated sample process and compact IPC."""
    config = replace(
        job.payload,
        input_path=job.input_path,
        output_dir=job.output_dir,
        worker_threads=job.threads_per_sample,
    )
    def resources(values):
        job.memory_current_bytes = int(values.get("current_private_bytes", 0))
        job.memory_peak_bytes = int(values.get("peak_private_bytes", 0))
        job.memory_estimate_bytes = max(job.memory_estimate_bytes, int(values.get("estimated_peak_bytes", 0)))
        if job._memory_admission is not None:
            job._memory_admission.update(
                job.job_id,
                (min(job.memory_current_bytes, int(values.get("current_working_set_bytes", 0)))
                 if values.get("current_sample_available", False) else 0),
                max(job.memory_estimate_bytes, job.memory_peak_bytes),
            )
    return _run_sample_process(config, control, progress, resource_callback=resources)


class _ResourceMonitor:
    def __init__(self, output_dir, callback=None):
        self.output_dir = output_dir
        self.callback = callback
        self.last_sample = 0.0
        self.data = {"policy": "sample-process-memory-v1", "peak_private_bytes": 0,
                     "peak_working_set_bytes": 0}

    def decoded(self, values):
        self.data["decoded_geometry"] = values
        self.data["estimated_peak_bytes"] = estimate_decoded_memory(
            values["full_vertex_count"], values["triangle_count"], values["original_vertex_count"])

    def sample(self, pid, *, force=False):
        now = time.monotonic()
        if not force and now - self.last_sample < 0.5:
            return
        self.last_sample = now
        self.data["current_sample_available"] = False
        try:
            import psutil
            info = psutil.Process(pid).memory_info()
            private = int(getattr(info, "private", info.rss))
            self.data.update(current_private_bytes=private, current_working_set_bytes=int(info.rss))
            self.data["current_sample_available"] = True
            self.data["peak_private_bytes"] = max(self.data["peak_private_bytes"], private,
                                                 int(getattr(info, "peak_pagefile", 0)))
            self.data["peak_working_set_bytes"] = max(self.data["peak_working_set_bytes"], int(info.rss),
                                                     int(getattr(info, "peak_wset", 0)))
        except (ImportError, OSError):
            pass
        except Exception:
            # psutil.NoSuchProcess/AccessDenied must not hide the sample error.
            pass
        try:
            state = memory_snapshot()
            self.data["last_system_memory"] = vars(state)
            if state.commit_limit is not None and state.available_commit is not None:
                self.data["peak_observed_system_commit_bytes"] = max(
                    self.data.get("peak_observed_system_commit_bytes", 0), state.commit_limit - state.available_commit)
        except (OSError, RuntimeError):
            self.data["system_memory_unavailable"] = True
        if self.callback is not None:
            self.callback(self.data)

    def save(self):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "processing_resources.json").write_text(
                json.dumps(self.data, indent=2), encoding="utf-8")
        except OSError:
            pass


def _run_sample_process(
    config: PipelineConfig,
    control: CooperativeToken,
    progress: ProgressCallback,
    *,
    analyze: Callable = _analyze_sample,
    cancel_grace_seconds: float = 10.0,
    exit_grace_seconds: float = 5.0,
    resource_callback: Callable[[dict], None] | None = None,
) -> ProcessSampleResult:
    context = mp.get_context("spawn")
    paused, cancelled = context.Event(), context.Event()
    receive, send = context.Pipe(duplex=False)
    child = context.Process(
        target=_sample_process_main,
        args=(config, paused, cancelled, send, analyze),
        name=f"soyroot-sample-{config.input_path.stem}",
    )
    started = False
    terminal = None
    cancel_started = None
    monitor = _ResourceMonitor(config.output_dir, resource_callback)
    try:
        with control.mirror_to(paused, cancelled):
            control.checkpoint()
            child.start()
            started = True
            monitor.data["process_id"] = child.pid
            send.close()
            while True:
                monitor.sample(child.pid)
                if control.cancelled:
                    if cancel_started is None:
                        cancel_started = time.monotonic()
                    if time.monotonic() - cancel_started >= cancel_grace_seconds:
                        # A native call may never reach another checkpoint.  A
                        # cancelled sample must not keep the desktop alive.
                        raise BatchCancelled("Cancelled sample process exceeded cleanup grace period")
                if receive.poll(0.05):
                    try:
                        message = receive.recv()
                    except EOFError:
                        break
                    if message[0] == "resources":
                        monitor.decoded(message[1])
                        monitor.sample(child.pid, force=True)
                    elif message[0] == "progress":
                        monitor.data["last_reported_step"] = message[1]
                        monitor.data["progress_fraction"] = message[2]
                        if not control.cancelled:
                            try:
                                progress(message[1], message[2])
                            except BatchCancelled:
                                # Continue draining until the child observes
                                # cancellation and exits; never leave it orphaned.
                                control.cancel()
                    else:
                        terminal = message
                        break
                elif not child.is_alive():
                    # The pipe can still hold the final message after exit.
                    if not receive.poll():
                        break
            monitor.data["terminal_state"] = terminal[0] if terminal is not None else "missing"
            monitor.sample(child.pid, force=True)
            child.join(timeout=exit_grace_seconds)
            if control.cancelled:
                raise BatchCancelled("Batch job cancelled")
            # Preserve a received exception even when interpreter/native cleanup
            # stalls or exits abnormally. Cleanup is secondary evidence.
            if terminal is not None and terminal[0] == "failed":
                if child.is_alive():
                    monitor.data["cleanup_issue"] = "worker did not exit after reporting failure"
                elif child.exitcode != 0:
                    monitor.data["cleanup_exit_code"] = child.exitcode
                raise SampleProcessError(
                    f"{terminal[1]} in sample process {child.pid}: {terminal[2]}",
                    terminal[3], monitor.data.copy(),
                )
            if child.is_alive():
                monitor.data["cleanup_issue"] = "worker did not exit after reporting its result"
                raise SampleProcessError(f"Sample process {child.pid} did not exit after reporting its result",
                                         resource_context=monitor.data.copy())
            if child.exitcode != 0 or terminal is None:
                raise SampleProcessError(
                    f"Sample process {child.pid} exited without a result (exit code {child.exitcode})",
                    resource_context=monitor.data.copy(),
                )
            if terminal[0] == "cancelled":
                raise BatchCancelled("Batch job cancelled")
            if terminal[0] != "completed":
                raise SampleProcessError(f"Unknown sample process response: {terminal[0]}")
            return terminal[1]
    finally:
        # The mirror has detached before termination.  Do not touch shared
        # Events after killing a worker: it could die while holding their lock.
        if started:
            monitor.sample(child.pid, force=True)
            if child.is_alive():
                child.terminate()
            child.join(timeout=5.0)
            if child.is_alive():
                child.kill()
                child.join()
            monitor.data["exit_code"] = child.exitcode
            monitor.save()
        child.close()
        receive.close()
        send.close()
