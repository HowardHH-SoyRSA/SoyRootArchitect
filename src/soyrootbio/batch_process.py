"""Spawn one isolated analysis process per active desktop batch slot.

The scheduler's threads only supervise processes and dispatch small messages.
Tk objects, locks, point clouds and full PipelineResult arrays never cross the
process boundary.  Each child exits after its sample, releasing native memory.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import multiprocessing as mp
from pathlib import Path
import os
import time
import traceback
from typing import Any, Callable

from .batch import BatchCancelled, BatchJob, CooperativeToken, ProgressCallback
from .pipeline import PipelineConfig, run_pipeline


@dataclass(frozen=True)
class ProcessSampleResult:
    output_dir: Path
    point_count: int
    process_id: int


class SampleProcessError(RuntimeError):
    """A child exception or abnormal exit, with its original traceback."""

    def __init__(self, message: str, remote_traceback: str = "") -> None:
        super().__init__(message)
        self.remote_traceback = remote_traceback


class _ProcessControl:
    def __init__(self, paused: Any, cancelled: Any) -> None:
        self._paused = paused
        self._cancelled = cancelled

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
        )
    return ProcessSampleResult(result.output_dir, result.point_count, os.getpid())


def _sample_process_main(config, paused, cancelled, messages, analyze) -> None:
    """Importable Windows spawn target; never receives the Tk application."""
    control = _ProcessControl(paused, cancelled)

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
    return _run_sample_process(config, control, progress)


def _run_sample_process(
    config: PipelineConfig,
    control: CooperativeToken,
    progress: ProgressCallback,
    *,
    analyze: Callable = _analyze_sample,
    cancel_grace_seconds: float = 10.0,
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
    try:
        with control.mirror_to(paused, cancelled):
            control.checkpoint()
            child.start()
            started = True
            send.close()
            while True:
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
                    if message[0] == "progress":
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
            child.join(timeout=5.0)
            if control.cancelled:
                raise BatchCancelled("Batch job cancelled")
            if child.is_alive():
                raise SampleProcessError(f"Sample process {child.pid} did not exit after reporting its result")
            if child.exitcode != 0 or terminal is None:
                raise SampleProcessError(
                    f"Sample process {child.pid} exited without a result (exit code {child.exitcode})"
                )
            if terminal[0] == "failed":
                raise SampleProcessError(
                    f"{terminal[1]} in sample process {child.pid}: {terminal[2]}",
                    terminal[3],
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
            if child.is_alive():
                child.terminate()
            child.join(timeout=5.0)
            if child.is_alive():
                child.kill()
                child.join()
        child.close()
        receive.close()
        send.close()
