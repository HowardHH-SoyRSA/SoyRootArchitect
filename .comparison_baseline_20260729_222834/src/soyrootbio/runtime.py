"""Per-analysis runtime controls that remain isolated across batch threads."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_WORKER_THREADS: ContextVar[int] = ContextVar("soyrootbio_worker_threads", default=-1)


def worker_threads() -> int:
    """Return the SciPy worker count for the current analysis context."""

    return int(_WORKER_THREADS.get())


@contextmanager
def worker_thread_limit(value: int | None) -> Iterator[None]:
    """Apply a cKDTree worker limit without leaking it to concurrent jobs."""

    workers = -1 if value is None else int(value)
    if workers == 0 or workers < -1:
        raise ValueError("worker thread limit must be a positive integer or None")
    token = _WORKER_THREADS.set(workers)
    try:
        yield
    finally:
        _WORKER_THREADS.reset(token)
