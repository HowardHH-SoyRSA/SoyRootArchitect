"""Thread-based batch scheduling primitives for the desktop application.

Workers never call Tkinter.  Instead, state changes are copied into immutable
``BatchEvent`` objects and placed in a thread-safe queue.  A GUI can poll
``drain_events`` from ``root.after`` and safely update widgets on its main
thread.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait as wait_futures
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
import json
import math
import os
from pathlib import Path
import queue
import re
import statistics
import threading
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
import uuid

from .hardware import HardwareInfo, ResourceAllocation, allocate_resources


ERROR_LOG_FILENAME = "processing_error.log"
_SENSITIVE_CONFIG_FIELD = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|credential)",
    re.IGNORECASE,
)


def _safe_error_config_value(value: Any, *, depth: int = 0) -> Any:
    """Convert a bounded, non-sensitive configuration value to JSON data."""

    if depth >= 5:
        return "<nested value omitted>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 4096 else value[:4096] + "... <truncated>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _safe_error_config_value(value.value, depth=depth + 1)
    if isinstance(value, (tuple, list)):
        converted = [
            _safe_error_config_value(item, depth=depth + 1)
            for item in value[:100]
        ]
        if len(value) > 100:
            converted.append(f"<{len(value) - 100} additional values omitted>")
        return converted
    if isinstance(value, Mapping):
        converted_mapping: dict[str, Any] = {}
        for key, item in list(value.items())[:100]:
            key_text = str(key)
            if _SENSITIVE_CONFIG_FIELD.search(key_text):
                converted_mapping[key_text] = "<redacted>"
            else:
                converted_mapping[key_text] = _safe_error_config_value(
                    item,
                    depth=depth + 1,
                )
        if len(value) > 100:
            converted_mapping["<omitted>"] = f"{len(value) - 100} additional entries"
        return converted_mapping
    return f"<{type(value).__name__} value omitted>"


def _safe_error_configuration(config: Any) -> dict[str, Any]:
    """Return public dataclass settings while redacting credential-like fields."""

    if config is None:
        return {}
    if not is_dataclass(config) or isinstance(config, type):
        return {
            "configuration_type": type(config).__name__,
            "details": "<non-dataclass configuration omitted>",
        }
    values: dict[str, Any] = {"configuration_type": type(config).__name__}
    for config_field in fields(config):
        name = config_field.name
        if name.startswith("_"):
            continue
        if _SENSITIVE_CONFIG_FIELD.search(name):
            values[name] = "<redacted>"
            continue
        try:
            value = getattr(config, name)
        except Exception:
            values[name] = "<unavailable>"
            continue
        values[name] = _safe_error_config_value(value)
    return values


def write_processing_error_log(
    *,
    output_dir: Path | str,
    input_path: Path | str,
    exception: BaseException,
    config: Any = None,
    context: Mapping[str, Any] | None = None,
) -> Path | None:
    """Persist a best-effort UTF-8 diagnostic report for a failed analysis.

    Error-report persistence must never hide or replace the original processing
    failure, so filesystem errors are deliberately reported by returning
    ``None``.  Configuration is limited to public dataclass fields, with
    credential-like names redacted and unsupported values omitted.
    """

    destination = Path(output_dir)
    report_path = destination / ERROR_LOG_FILENAME
    try:
        destination.mkdir(parents=True, exist_ok=True)
        occurred_at = datetime.now().astimezone().isoformat(timespec="seconds")
        trace = "".join(
            traceback.format_exception(
                type(exception),
                exception,
                exception.__traceback__,
            )
        ).rstrip()
        configuration = json.dumps(
            _safe_error_configuration(config),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        failure_context = json.dumps(
            _safe_error_config_value(dict(context or {})),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        report = (
            "BioInsAlgo processing failure\n"
            f"timestamp: {occurred_at}\n"
            f"sample: {Path(input_path).name}\n"
            f"input_path: {Path(input_path)}\n"
            f"output_directory: {destination}\n"
            f"exception_type: {type(exception).__name__}\n"
            f"exception_message: {exception}\n"
            "\nconfiguration:\n"
            f"{configuration}\n"
            "\nfailure_context:\n"
            f"{failure_context}\n"
            "\ntraceback:\n"
            f"{trace}\n"
        )
        report_path.write_text(report, encoding="utf-8")
    except Exception:
        return None
    return report_path


class BatchCancelled(RuntimeError):
    """Raised by :meth:`CooperativeToken.checkpoint` after cancellation."""


class BatchJobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {self.COMPLETED, self.CANCELLED, self.FAILED}


# Short name for code that already has a BatchJob context.
JobState = BatchJobState


class BatchEventType(str, Enum):
    SUBMITTED = "submitted"
    STARTED = "started"
    PROGRESS = "progress"
    PAUSED = "paused"
    RESUMED = "resumed"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CooperativeToken:
    """Thread-safe cooperative pause/resume/cancel control.

    Long-running code should call :meth:`checkpoint` between bounded pieces of
    work.  Existing APIs that accept a ``cancel_check: Callable[[], bool]`` can
    receive :meth:`cancel_check`; it also waits while a pause is requested.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pause_requested = False
        self._cancel_requested = False
        self._pause_started: float | None = None
        self._paused_seconds = 0.0

    @property
    def paused(self) -> bool:
        with self._condition:
            return self._pause_requested

    @property
    def cancelled(self) -> bool:
        with self._condition:
            return self._cancel_requested

    @property
    def paused_seconds(self) -> float:
        with self._condition:
            total = self._paused_seconds
            if self._pause_started is not None:
                total += time.monotonic() - self._pause_started
            return total

    def pause(self) -> bool:
        """Request a pause and return whether the request changed state."""

        with self._condition:
            if self._cancel_requested or self._pause_requested:
                return False
            self._pause_requested = True
            self._pause_started = time.monotonic()
            return True

    def resume(self) -> bool:
        """Release a pause and wake a worker waiting at a checkpoint."""

        with self._condition:
            if not self._pause_requested:
                return False
            self._finish_pause_locked()
            self._pause_requested = False
            self._condition.notify_all()
            return True

    def cancel(self) -> bool:
        """Request cancellation and wake a paused worker."""

        with self._condition:
            if self._cancel_requested:
                return False
            self._cancel_requested = True
            self._finish_pause_locked()
            self._pause_requested = False
            self._condition.notify_all()
            return True

    def checkpoint(self) -> None:
        """Block while paused, then raise if cancellation was requested."""

        with self._condition:
            while self._pause_requested and not self._cancel_requested:
                self._condition.wait()
            if self._cancel_requested:
                raise BatchCancelled("Batch job cancelled")

    def wait_if_paused(self) -> None:
        """Compatibility spelling for a cooperative checkpoint."""

        self.checkpoint()

    def cancel_check(self) -> bool:
        """Adapter for processing functions that poll a boolean callback."""

        try:
            self.checkpoint()
        except BatchCancelled:
            return True
        return False

    def _finish_pause_locked(self) -> None:
        if self._pause_started is not None:
            self._paused_seconds += time.monotonic() - self._pause_started
            self._pause_started = None


# Alternate descriptive names make the token easy to discover from either a
# pause/resume or a cancellation-oriented API.
BatchControlToken = CooperativeToken
CancellationToken = CooperativeToken


@dataclass(frozen=True, slots=True)
class BatchJobSnapshot:
    """Immutable job state safe to hand from a worker to a GUI thread."""

    job_id: str
    input_path: Path
    output_dir: Path
    state: BatchJobState
    progress: float
    step: str
    eta_seconds: float | None
    threads_per_sample: int
    error: str | None
    error_log_path: Path | None
    submitted_at: float
    started_at: float | None
    finished_at: float | None

    @property
    def progress_percent(self) -> float:
        return self.progress * 100.0

    @property
    def status(self) -> BatchJobState:
        return self.state


@dataclass(frozen=True, slots=True)
class BatchEvent:
    """One scheduler notification retrieved through :meth:`drain_events`."""

    kind: BatchEventType
    job: BatchJobSnapshot

    @property
    def job_id(self) -> str:
        return self.job.job_id


@dataclass(slots=True)
class BatchJob:
    """Mutable per-sample state owned by a :class:`BatchScheduler`."""

    job_id: str
    input_path: Path
    output_dir: Path
    threads_per_sample: int
    payload: Any = None
    state: BatchJobState = BatchJobState.QUEUED
    progress: float = 0.0
    step: str = "Queued"
    eta_seconds: float | None = None
    error: str | None = None
    error_log_path: Path | None = None
    result: Any = None
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    _control: CooperativeToken = field(default_factory=CooperativeToken, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _started_monotonic: float | None = field(default=None, init=False, repr=False)
    _finished_monotonic: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        self.output_dir = Path(self.output_dir)
        if self.threads_per_sample < 1:
            raise ValueError("threads_per_sample must be at least 1")

    @property
    def control(self) -> CooperativeToken:
        return self._control

    @property
    def status(self) -> BatchJobState:
        return self.state

    @property
    def progress_percent(self) -> float:
        return self.progress * 100.0

    @property
    def elapsed_seconds(self) -> float:
        with self._lock:
            if self._started_monotonic is None:
                return 0.0
            if self.finished_at is not None:
                end = self._finished_monotonic or time.monotonic()
            else:
                end = time.monotonic()
            return max(0.0, end - self._started_monotonic - self._control.paused_seconds)

    def snapshot(self) -> BatchJobSnapshot:
        with self._lock:
            return BatchJobSnapshot(
                job_id=self.job_id,
                input_path=self.input_path,
                output_dir=self.output_dir,
                state=self.state,
                progress=self.progress,
                step=self.step,
                eta_seconds=self.eta_seconds,
                threads_per_sample=self.threads_per_sample,
                error=self.error,
                error_log_path=self.error_log_path,
                submitted_at=self.submitted_at,
                started_at=self.started_at,
                finished_at=self.finished_at,
            )


class StepTimingHistory:
    """Bounded, persistent timing samples used to estimate future job ETAs."""

    VERSION = 1
    TOTAL_STEP = "__total__"

    def __init__(self, path: Path | str | None = None, *, max_samples_per_step: int = 50) -> None:
        if max_samples_per_step < 1:
            raise ValueError("max_samples_per_step must be at least 1")
        self.path = Path(path) if path is not None else None
        self.max_samples_per_step = max_samples_per_step
        self._samples: dict[str, list[float]] = {}
        self._lock = threading.RLock()
        self.load_error: str | None = None
        self.reload()

    @property
    def steps(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(step for step in self._samples if step != self.TOTAL_STEP)

    def samples(self, step: str) -> tuple[float, ...]:
        with self._lock:
            return tuple(self._samples.get(step, ()))

    def record(self, step: str, duration_seconds: float, *, save: bool = True) -> None:
        """Append a non-negative duration and optionally persist immediately."""

        normalized = step.strip()
        if not normalized:
            raise ValueError("step cannot be empty")
        duration = float(duration_seconds)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration_seconds must be finite and non-negative")
        with self._lock:
            values = self._samples.setdefault(normalized, [])
            values.append(duration)
            del values[: max(0, len(values) - self.max_samples_per_step)]
            if save:
                self._save_locked()

    def record_many(self, durations: Mapping[str, float], *, save: bool = True) -> None:
        """Record one completed run using a single atomic persistence write."""

        checked: list[tuple[str, float]] = []
        for step, duration in durations.items():
            normalized = step.strip()
            value = float(duration)
            if not normalized:
                raise ValueError("step cannot be empty")
            if not math.isfinite(value) or value < 0:
                raise ValueError("duration_seconds must be finite and non-negative")
            checked.append((normalized, value))
        with self._lock:
            for step, duration in checked:
                values = self._samples.setdefault(step, [])
                values.append(duration)
                del values[: max(0, len(values) - self.max_samples_per_step)]
            if save:
                self._save_locked()

    def estimate(self, step: str) -> float | None:
        """Return the median duration for a step, robust to occasional outliers."""

        with self._lock:
            values = self._samples.get(step)
            return float(statistics.median(values)) if values else None

    def estimate_total(self, steps: Sequence[str] | None = None) -> float | None:
        """Estimate a complete run, or the sum of a supplied sequence of steps."""

        if steps is None:
            explicit_total = self.estimate(self.TOTAL_STEP)
            if explicit_total is not None:
                return explicit_total
            steps = self.steps
        estimates = [self.estimate(step) for step in steps]
        known = [value for value in estimates if value is not None]
        return sum(known) if known else None

    def estimate_eta(self, progress: float, *, elapsed_seconds: float = 0.0) -> float | None:
        """Estimate remaining seconds from history, falling back to live rate."""

        fraction = min(1.0, max(0.0, float(progress)))
        historical_total = self.estimate_total()
        if historical_total is not None:
            return max(0.0, historical_total * (1.0 - fraction))
        if fraction > 0.0 and elapsed_seconds >= 0.0:
            return max(0.0, elapsed_seconds * (1.0 - fraction) / fraction)
        return None

    def reload(self) -> None:
        """Reload the JSON file.  Missing or malformed history is non-fatal."""

        with self._lock:
            self._samples = {}
            self.load_error = None
            if self.path is None or not self.path.exists():
                return
            try:
                document = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(document, dict):
                    raise ValueError("timing history must be a JSON object")
                raw_steps = document.get("steps", {})
                if not isinstance(raw_steps, dict):
                    raise ValueError("'steps' must be an object")
                loaded: dict[str, list[float]] = {}
                for step, raw_values in raw_steps.items():
                    if not isinstance(step, str) or not isinstance(raw_values, list):
                        continue
                    values = [
                        parsed
                        for value in raw_values
                        if math.isfinite(parsed := float(value)) and parsed >= 0.0
                    ]
                    if values:
                        loaded[step] = values[-self.max_samples_per_step :]
                self._samples = loaded
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.load_error = str(exc)
                self._samples = {}

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "version": self.VERSION,
            "steps": self._samples,
        }
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


ProgressCallback = Callable[[str, float], None]


class BatchRunner(Protocol):
    """Callable executed for each job by :class:`BatchScheduler`."""

    def __call__(
        self,
        job: BatchJob,
        control: CooperativeToken,
        progress_callback: ProgressCallback,
    ) -> Any: ...


class BatchScheduler:
    """Manage a bounded pool of per-sample analysis workers.

    ``runner`` receives the job (including its output directory and per-sample
    thread allowance), a cooperative token, and a progress callback accepting
    ``(step_name, overall_fraction)``.
    """

    def __init__(
        self,
        runner: BatchRunner,
        *,
        max_concurrent_samples: int = 1,
        threads_per_sample: int = 1,
        timing_history: StepTimingHistory | None = None,
        thread_name_prefix: str = "soyroot-batch",
    ) -> None:
        if max_concurrent_samples < 1:
            raise ValueError("max_concurrent_samples must be at least 1")
        if threads_per_sample < 1:
            raise ValueError("threads_per_sample must be at least 1")
        self.runner = runner
        self.max_concurrent_samples = max_concurrent_samples
        self.threads_per_sample = threads_per_sample
        self.timing_history = timing_history or StepTimingHistory()
        self.thread_name_prefix = thread_name_prefix
        self._jobs: dict[str, BatchJob] = {}
        self._output_owners: dict[Path, str] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._events: queue.Queue[BatchEvent] = queue.Queue()
        self._lock = threading.RLock()
        self._executor: ThreadPoolExecutor | None = None
        self._started = False
        self._closed = False

    @classmethod
    def automatic(
        cls,
        runner: BatchRunner,
        *,
        sample_count: int | None = None,
        hardware: HardwareInfo | None = None,
        max_concurrent_samples: int | None = None,
        threads_per_sample: int | None = None,
        timing_history: StepTimingHistory | None = None,
        **allocation_options: Any,
    ) -> tuple["BatchScheduler", ResourceAllocation]:
        """Create a scheduler using hardware-aware defaults and return its plan."""

        allocation = allocate_resources(
            hardware,
            sample_count=sample_count,
            max_concurrent_samples=max_concurrent_samples,
            threads_per_sample=threads_per_sample,
            **allocation_options,
        )
        scheduler = cls(
            runner,
            max_concurrent_samples=allocation.max_concurrent_samples,
            threads_per_sample=allocation.threads_per_sample,
            timing_history=timing_history,
        )
        return scheduler, allocation

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def jobs(self) -> tuple[BatchJob, ...]:
        with self._lock:
            return tuple(self._jobs.values())

    @property
    def snapshots(self) -> tuple[BatchJobSnapshot, ...]:
        return tuple(job.snapshot() for job in self.jobs)

    @property
    def all_done(self) -> bool:
        jobs = self.jobs
        return all(job.snapshot().state.terminal for job in jobs)

    @property
    def active_count(self) -> int:
        return sum(
            job.snapshot().state in {BatchJobState.RUNNING, BatchJobState.PAUSED, BatchJobState.CANCELLING}
            for job in self.jobs
        )

    def get_job(self, job_id: str) -> BatchJob:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError:
                raise KeyError(f"Unknown batch job: {job_id}") from None

    def submit(
        self,
        input_path: Path | str,
        output_dir: Path | str,
        *,
        payload: Any = None,
        job_id: str | None = None,
    ) -> BatchJob:
        """Add one sample.  Submissions after :meth:`start` begin automatically."""

        identifier = job_id or uuid.uuid4().hex
        canonical_output = canonical_output_directory(output_dir)
        with self._lock:
            if self._closed:
                raise RuntimeError("BatchScheduler is closed")
            if identifier in self._jobs:
                raise ValueError(f"Duplicate batch job id: {identifier}")
            owner = self._output_owners.get(canonical_output)
            if owner is not None:
                raise ValueError(
                    f"Output directory is already assigned to batch job {owner}: "
                    f"{canonical_output}"
                )
            job = BatchJob(
                job_id=identifier,
                input_path=Path(input_path),
                output_dir=canonical_output,
                threads_per_sample=self.threads_per_sample,
                payload=payload,
            )
            self._jobs[identifier] = job
            self._output_owners[canonical_output] = identifier
            should_schedule = self._started
        self._emit(BatchEventType.SUBMITTED, job)
        if should_schedule:
            self._submit_job(job)
        return job

    def submit_many(
        self,
        input_paths: Iterable[Path | str],
        output_root: Path | str,
        *,
        payload_factory: Callable[[Path], Any] | None = None,
    ) -> tuple[BatchJob, ...]:
        """Submit inputs with deterministic, collision-free per-sample folders."""

        root = Path(output_root)
        used = {job.output_dir for job in self.jobs}
        submitted: list[BatchJob] = []
        for input_value in input_paths:
            input_path = Path(input_value)
            output_dir = unique_output_directory(input_path, root, used=used)
            used.add(output_dir)
            payload = payload_factory(input_path) if payload_factory is not None else None
            submitted.append(self.submit(input_path, output_dir, payload=payload))
        return tuple(submitted)

    def start(self) -> None:
        """Start all queued jobs.  Calling this more than once is harmless."""

        with self._lock:
            if self._closed:
                raise RuntimeError("BatchScheduler is closed")
            if self._started:
                return
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_concurrent_samples,
                thread_name_prefix=self.thread_name_prefix,
            )
            self._started = True
            jobs = tuple(self._jobs.values())
        for job in jobs:
            if not job.snapshot().state.terminal:
                self._submit_job(job)

    def pause(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        with job._lock:
            if job.state.terminal or job.state == BatchJobState.CANCELLING:
                return False
            if not job._control.pause():
                return False
            job.state = BatchJobState.PAUSED
            job.step = "Paused"
            job.eta_seconds = None
        self._emit(BatchEventType.PAUSED, job)
        return True

    def resume(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        with job._lock:
            if job.state.terminal or job.state == BatchJobState.CANCELLING:
                return False
            # The token is the authoritative pause state.  A worker can be
            # between a checkpoint and its state update when the GUI requests
            # a pause, so relying only on the display state can strand a
            # genuinely paused worker with no usable resume action.
            if not job._control.paused:
                return False
            if not job._control.resume():
                return False
            if job.started_at is None:
                job.state = BatchJobState.QUEUED
                job.step = "Queued"
            else:
                job.state = BatchJobState.RUNNING
                job.step = "Resuming"
        self._emit(BatchEventType.RESUMED, job)
        return True

    def cancel(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        with self._lock:
            future = self._futures.get(job_id)
        with job._lock:
            if job.state.terminal:
                return False
            if not job._control.cancel():
                return False
            cancelled_before_start = job.started_at is None and (future is None or future.cancel())
            if cancelled_before_start:
                job.state = BatchJobState.CANCELLED
                job.step = "Cancelled"
                job.eta_seconds = None
                job.finished_at = time.time()
            else:
                job.state = BatchJobState.CANCELLING
                job.step = "Cancelling"
                job.eta_seconds = None
        self._emit(
            BatchEventType.CANCELLED if cancelled_before_start else BatchEventType.CANCEL_REQUESTED,
            job,
        )
        return True

    def pause_all(self) -> int:
        return sum(self.pause(job.job_id) for job in self.jobs)

    def resume_all(self) -> int:
        return sum(self.resume(job.job_id) for job in self.jobs)

    def cancel_all(self) -> int:
        return sum(self.cancel(job.job_id) for job in self.jobs)

    def drain_events(self, max_items: int | None = None) -> list[BatchEvent]:
        """Non-blockingly drain notifications (intended for a Tk ``after`` loop)."""

        if max_items is not None and max_items < 0:
            raise ValueError("max_items cannot be negative")
        events: list[BatchEvent] = []
        while max_items is None or len(events) < max_items:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        return events

    poll_events = drain_events

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for currently submitted jobs and report whether all finished."""

        if not self.started:
            self.start()
        with self._lock:
            futures = tuple(self._futures.values())
        if not futures:
            return True
        _, pending = wait_futures(futures, timeout=timeout)
        return not pending

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = False) -> None:
        """Stop accepting work and release the executor."""

        if cancel_pending:
            self.cancel_all()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=cancel_pending)

    def _submit_job(self, job: BatchJob) -> None:
        with self._lock:
            if self._executor is None:
                raise RuntimeError("BatchScheduler has not started")
            if job.job_id in self._futures:
                return
            if job.snapshot().state.terminal:
                return
            self._futures[job.job_id] = self._executor.submit(self._execute, job)

    def _execute(self, job: BatchJob) -> None:
        control = job._control
        try:
            control.checkpoint()
        except BatchCancelled:
            self._finish_cancelled(job)
            return

        started_monotonic = time.monotonic()
        paused_at_start = control.paused_seconds
        with job._lock:
            if control.cancelled:
                self._finish_cancelled(job)
                return
            job.started_at = time.time()
            job._started_monotonic = started_monotonic
            if control.paused:
                job.state = BatchJobState.PAUSED
                job.step = "Paused"
                job.eta_seconds = None
            else:
                job.state = BatchJobState.RUNNING
                job.step = "Starting"
        self._emit(BatchEventType.STARTED, job)

        step_name = "Starting"
        step_started = started_monotonic
        step_paused = paused_at_start
        step_durations: dict[str, float] = {}

        def progress_callback(step: str, fraction: float) -> None:
            nonlocal step_name, step_started, step_paused
            control.checkpoint()
            normalized_step = str(step).strip() or "Processing"
            now = time.monotonic()
            paused_now = control.paused_seconds
            if normalized_step != step_name:
                duration = max(0.0, now - step_started - (paused_now - step_paused))
                step_durations[step_name] = step_durations.get(step_name, 0.0) + duration
                step_name = normalized_step
                step_started = now
                step_paused = paused_now
            bounded = min(1.0, max(0.0, float(fraction)))
            elapsed = max(0.0, now - started_monotonic - (paused_now - paused_at_start))
            with job._lock:
                if job.state in {BatchJobState.CANCELLING, BatchJobState.CANCELLED}:
                    raise BatchCancelled("Batch job cancelled")
                effective_progress = max(job.progress, bounded)
                eta = self.timing_history.estimate_eta(effective_progress, elapsed_seconds=elapsed)
                job.progress = effective_progress
                if control.paused:
                    # Preserve the externally visible paused state if a pause
                    # landed after the checkpoint above.  This keeps resume()
                    # reachable while the token waits at the next checkpoint.
                    job.state = BatchJobState.PAUSED
                    job.step = "Paused"
                    job.eta_seconds = None
                else:
                    job.state = BatchJobState.RUNNING
                    job.step = normalized_step
                    job.eta_seconds = eta
            self._emit(BatchEventType.PROGRESS, job)

        try:
            job.output_dir.mkdir(parents=True, exist_ok=True)
            result = self.runner(job, control, progress_callback)
            control.checkpoint()
        except Exception as exc:
            now = time.monotonic()
            # Earlier transitioned steps completed successfully and remain
            # useful, but a partial current step and partial total would skew
            # future estimates downward.
            self._record_timings(step_durations, total=None)
            with job._lock:
                failure_context = {
                    "job_id": job.job_id,
                    "last_reported_step": job.step,
                    "progress_fraction": job.progress,
                    "threads_per_sample": job.threads_per_sample,
                }
                cancelled = (
                    isinstance(exc, BatchCancelled)
                    or control.cancelled
                    or job.state == BatchJobState.CANCELLING
                )
                if cancelled:
                    self._mark_cancelled_locked(job, now=now)
                else:
                    job.state = BatchJobState.FAILED
                    job.step = "Failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.eta_seconds = None
                    job.finished_at = time.time()
                    job._finished_monotonic = now
            if not cancelled:
                error_log_path = write_processing_error_log(
                    output_dir=job.output_dir,
                    input_path=job.input_path,
                    exception=exc,
                    config=job.payload,
                    context=failure_context,
                )
                with job._lock:
                    job.error_log_path = error_log_path
            self._emit(BatchEventType.CANCELLED if cancelled else BatchEventType.FAILED, job)
            return

        now = time.monotonic()
        paused_now = control.paused_seconds
        step_durations[step_name] = step_durations.get(step_name, 0.0) + max(
            0.0, now - step_started - (paused_now - step_paused)
        )
        total_duration = max(0.0, now - started_monotonic - (paused_now - paused_at_start))
        self._record_timings(step_durations, total_duration)
        while True:
            with job._lock:
                commit_now = time.monotonic()
                cancelled = control.cancelled or job.state == BatchJobState.CANCELLING
                paused = not cancelled and control.paused
                if cancelled:
                    self._mark_cancelled_locked(job, now=commit_now)
                elif not paused:
                    job.result = result
                    job.state = BatchJobState.COMPLETED
                    job.step = "Complete"
                    job.progress = 1.0
                    job.eta_seconds = 0.0
                    job.finished_at = time.time()
                    job._finished_monotonic = commit_now
            if not paused:
                break
            # A pause that wins the job lock immediately before completion is
            # a real pause.  Wait without holding the job lock so resume/cancel
            # can make progress, then retry the terminal commit atomically.
            try:
                control.checkpoint()
            except BatchCancelled:
                pass
        self._emit(BatchEventType.CANCELLED if cancelled else BatchEventType.COMPLETED, job)

    def _record_timings(self, durations: Mapping[str, float], total: float | None) -> None:
        cleaned = {step: duration for step, duration in durations.items() if step and duration >= 0.0}
        if total is not None:
            cleaned[StepTimingHistory.TOTAL_STEP] = total
        if not cleaned:
            return
        try:
            self.timing_history.record_many(cleaned)
        except OSError:
            # Output persistence should not turn a successful analysis into a
            # failed job.  In-memory samples remain available for this session.
            pass

    def _finish_cancelled(self, job: BatchJob) -> None:
        with job._lock:
            if job.state == BatchJobState.CANCELLED and job.finished_at is not None:
                return
            if job.state.terminal:
                return
            self._mark_cancelled_locked(job)
        self._emit(BatchEventType.CANCELLED, job)

    @staticmethod
    def _mark_cancelled_locked(job: BatchJob, *, now: float | None = None) -> None:
        """Commit cancellation while ``job._lock`` is held."""

        job.state = BatchJobState.CANCELLED
        job.step = "Cancelled"
        job.eta_seconds = None
        job.finished_at = time.time()
        if job._started_monotonic is not None:
            job._finished_monotonic = time.monotonic() if now is None else now

    def _emit(self, kind: BatchEventType, job: BatchJob) -> None:
        self._events.put(BatchEvent(kind=kind, job=job.snapshot()))

    def __enter__(self) -> "BatchScheduler":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.shutdown(wait=True, cancel_pending=exc is not None)


_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def canonical_output_directory(path: Path | str) -> Path:
    """Return an absolute, normalized output path suitable for ownership checks."""

    expanded = Path(path).expanduser()
    try:
        return expanded.resolve(strict=False)
    except OSError:
        # ``resolve`` can fail for malformed or temporarily unavailable network
        # parents.  ``abspath`` still normalizes ``.``/``..`` and preserves the
        # platform's case-folding Path equality semantics.
        return Path(os.path.abspath(expanded))


def unique_output_directory(
    input_path: Path | str,
    output_root: Path | str,
    *,
    used: Iterable[Path] = (),
) -> Path:
    """Return a deterministic per-sample directory, suffixing duplicate stems."""

    root = Path(output_root)
    stem = _INVALID_FILENAME.sub("_", Path(input_path).stem).strip(" .") or "sample"
    reserved = {canonical_output_directory(path) for path in used}
    candidate = root / stem
    suffix = 2
    while canonical_output_directory(candidate) in reserved or candidate.exists():
        candidate = root / f"{stem}_{suffix}"
        suffix += 1
    return candidate


__all__ = [
    "BatchCancelled",
    "BatchControlToken",
    "BatchEvent",
    "BatchEventType",
    "BatchJob",
    "BatchJobSnapshot",
    "BatchJobState",
    "BatchRunner",
    "BatchScheduler",
    "CancellationToken",
    "canonical_output_directory",
    "CooperativeToken",
    "ERROR_LOG_FILENAME",
    "JobState",
    "ProgressCallback",
    "StepTimingHistory",
    "unique_output_directory",
    "write_processing_error_log",
]
