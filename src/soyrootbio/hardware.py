"""Hardware discovery and deterministic resource recommendations for batches.

The functions in this module deliberately depend only on the Python standard
library.  :mod:`psutil` is used when it happens to be installed, but it is not
required.  GPU discovery is likewise best-effort: a machine without a GPU (or
without vendor utilities on ``PATH``) simply reports an empty tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import importlib
import io
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any


GIB = 1024**3
DEFAULT_MEMORY_PER_SAMPLE_BYTES = 4 * GIB
DEFAULT_MEMORY_RESERVE_BYTES = 2 * GIB


@dataclass(frozen=True, slots=True)
class GPUInfo:
    """Information about one optional compute GPU."""

    index: int
    name: str
    memory_total_bytes: int | None = None
    backend: str = "CUDA"
    driver_version: str | None = None

    @property
    def total_memory_bytes(self) -> int | None:
        """Alias convenient for callers that use the host RAM field naming."""

        return self.memory_total_bytes


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    """A dependency-free snapshot of resources relevant to batch processing."""

    logical_cpus: int
    physical_cpus: int
    total_memory_bytes: int | None
    gpus: tuple[GPUInfo, ...] = ()
    available_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.logical_cpus < 1:
            raise ValueError("logical_cpus must be at least 1")
        if self.physical_cpus < 1:
            raise ValueError("physical_cpus must be at least 1")
        if self.total_memory_bytes is not None and self.total_memory_bytes < 0:
            raise ValueError("total_memory_bytes cannot be negative")
        if self.available_memory_bytes is not None and self.available_memory_bytes < 0:
            raise ValueError("available_memory_bytes cannot be negative")
        if (
            self.total_memory_bytes is not None
            and self.available_memory_bytes is not None
            and self.available_memory_bytes > self.total_memory_bytes
        ):
            raise ValueError("available_memory_bytes cannot exceed total_memory_bytes")

    @property
    def ram_bytes(self) -> int | None:
        """Short alias used by display code."""

        return self.total_memory_bytes

    @property
    def memory_total_bytes(self) -> int | None:
        """Compatibility alias matching :class:`GPUInfo`."""

        return self.total_memory_bytes

    @property
    def memory_available_bytes(self) -> int | None:
        """RAM currently available for new work, when discovery supports it."""

        return self.available_memory_bytes

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus)


@dataclass(frozen=True, slots=True)
class ResourceAllocation:
    """CPU and concurrency settings selected for one batch."""

    max_concurrent_samples: int
    threads_per_sample: int
    cpu_budget: int
    memory_limited_samples: int | None

    def __post_init__(self) -> None:
        if self.max_concurrent_samples < 1:
            raise ValueError("max_concurrent_samples must be at least 1")
        if self.threads_per_sample < 1:
            raise ValueError("threads_per_sample must be at least 1")

    @property
    def total_worker_threads(self) -> int:
        return self.max_concurrent_samples * self.threads_per_sample


def detect_hardware(*, include_gpus: bool = True) -> HardwareInfo:
    """Inspect CPU, RAM, and optional NVIDIA GPU resources.

    Discovery failures are intentionally non-fatal.  Physical CPU count falls
    back to the logical count, unknown RAM is represented by ``None``, and no
    detected GPU is represented by ``()``.
    """

    logical = max(1, int(os.cpu_count() or 1))
    physical: int | None = None
    total_memory: int | None = None
    available_memory: int | None = None

    psutil = _load_psutil()
    if psutil is not None:
        try:
            physical_value = psutil.cpu_count(logical=False)
            if physical_value:
                physical = int(physical_value)
        except Exception:
            pass
        try:
            memory = psutil.virtual_memory()
            memory_value = int(memory.total)
            available_value = int(memory.available)
            if memory_value >= 0:
                total_memory = memory_value
            if available_value >= 0:
                available_memory = available_value
        except Exception:
            pass

    if physical is None:
        physical = _physical_cpu_count() or logical
    if total_memory is None:
        total_memory = _total_memory_bytes()
    if available_memory is None:
        available_memory = _available_memory_bytes()
    if total_memory is not None and available_memory is not None:
        available_memory = min(total_memory, available_memory)

    # Virtualized systems occasionally expose inconsistent values.  Both are
    # useful, but a physical-core count above the logical count is not credible.
    physical = max(1, min(int(physical), logical))
    gpus = detect_gpus() if include_gpus else ()
    return HardwareInfo(logical, physical, total_memory, gpus, available_memory)


def detect_gpus() -> tuple[GPUInfo, ...]:
    """Return NVIDIA GPU information when ``nvidia-smi`` is available.

    The main application never requires this utility.  Avoiding imports of
    CUDA frameworks keeps startup fast and makes CPU-only installations safe.
    """

    executable = shutil.which("nvidia-smi")
    if not executable:
        return ()
    command = [
        executable,
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return ()

    detected: list[GPUInfo] = []
    try:
        rows = csv.reader(io.StringIO(completed.stdout))
        for row in rows:
            if not row or len(row) < 4:
                continue
            index_text, name, memory_mib_text, driver = (part.strip() for part in row[:4])
            memory_bytes: int | None
            try:
                memory_bytes = int(float(memory_mib_text)) * 1024**2
            except ValueError:
                memory_bytes = None
            detected.append(
                GPUInfo(
                    index=int(index_text),
                    name=name,
                    memory_total_bytes=memory_bytes,
                    driver_version=driver or None,
                )
            )
    except (TypeError, ValueError, csv.Error):
        return ()
    return tuple(detected)


def allocate_resources(
    hardware: HardwareInfo | None = None,
    *,
    sample_count: int | None = None,
    max_concurrent_samples: int | None = None,
    threads_per_sample: int | None = None,
    reserve_logical_cpus: int = 1,
    memory_per_sample_bytes: int = DEFAULT_MEMORY_PER_SAMPLE_BYTES,
    reserve_memory_bytes: int = DEFAULT_MEMORY_RESERVE_BYTES,
) -> ResourceAllocation:
    """Choose deterministic batch concurrency while honoring manual overrides.

    Automatic selection reserves one logical CPU for the desktop by default,
    aims to leave at least two worker threads per sample, and limits concurrency
    using a conservative per-sample RAM estimate.  A manual concurrency or
    thread value is honored exactly (apart from concurrency never exceeding a
    supplied ``sample_count``); this permits intentional oversubscription.
    """

    hardware = hardware or detect_hardware()
    _positive_or_none(sample_count, "sample_count")
    _positive_or_none(max_concurrent_samples, "max_concurrent_samples")
    _positive_or_none(threads_per_sample, "threads_per_sample")
    if reserve_logical_cpus < 0:
        raise ValueError("reserve_logical_cpus cannot be negative")
    if memory_per_sample_bytes <= 0:
        raise ValueError("memory_per_sample_bytes must be positive")
    if reserve_memory_bytes < 0:
        raise ValueError("reserve_memory_bytes cannot be negative")

    logical = hardware.logical_cpus
    cpu_budget = max(1, logical - reserve_logical_cpus) if logical > 1 else 1
    # Available RAM, rather than installed RAM, is the meaningful constraint
    # when a batch starts alongside the GUI and other desktop applications.
    # Fall back to total RAM for callers constructing legacy snapshots.
    memory_budget = (
        hardware.available_memory_bytes
        if hardware.available_memory_bytes is not None
        else hardware.total_memory_bytes
    )
    memory_limit = _memory_sample_limit(
        memory_budget,
        memory_per_sample_bytes,
        reserve_memory_bytes,
    )

    if max_concurrent_samples is not None:
        concurrent = max_concurrent_samples
    else:
        if threads_per_sample is not None:
            cpu_limit = max(1, cpu_budget // threads_per_sample)
        else:
            # Do not turn every physical core into a separate single-threaded
            # sample unless the device genuinely only has one worker per sample.
            cpu_limit = max(1, min(hardware.physical_cpus, cpu_budget // 2))
        concurrent = cpu_limit
        if memory_limit is not None:
            concurrent = min(concurrent, memory_limit)

    if sample_count is not None:
        concurrent = min(concurrent, sample_count)
    concurrent = max(1, concurrent)

    if threads_per_sample is None:
        selected_threads = max(1, cpu_budget // concurrent)
    else:
        selected_threads = threads_per_sample

    return ResourceAllocation(
        max_concurrent_samples=concurrent,
        threads_per_sample=selected_threads,
        cpu_budget=cpu_budget,
        memory_limited_samples=memory_limit,
    )


# A descriptive alias for callers that prefer recommendation terminology.
recommend_allocation = allocate_resources


def format_bytes(value: int | None) -> str:
    """Format a byte count for a compact hardware summary."""

    if value is None:
        return "Unknown"
    size = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or suffix == "TiB":
            return f"{size:.1f} {suffix}"
        size /= 1024.0
    return f"{size:.1f} TiB"  # pragma: no cover - loop always returns


def _positive_or_none(value: int | None, name: str) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be at least 1")


def _memory_sample_limit(total: int | None, per_sample: int, reserve: int) -> int | None:
    if total is None:
        return None
    usable = max(0, total - reserve)
    return max(1, usable // per_sample)


def _load_psutil() -> Any | None:
    try:
        return importlib.import_module("psutil")
    except (ImportError, OSError):
        return None


def _physical_cpu_count() -> int | None:
    system = platform.system()
    if system == "Linux":
        return _linux_physical_cpu_count()
    if system == "Windows":
        return _windows_physical_cpu_count()
    if system == "Darwin":
        try:
            completed = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            return int(completed.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    return None


def _linux_physical_cpu_count() -> int | None:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    packages_and_cores: set[tuple[str, str]] = set()
    for block in text.split("\n\n"):
        values: dict[str, str] = {}
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip()
        if "physical id" in values and "core id" in values:
            packages_and_cores.add((values["physical id"], values["core id"]))
    return len(packages_and_cores) or None


def _windows_physical_cpu_count() -> int | None:
    # Each RelationProcessorCore record describes one physical core.  Parsing
    # only the common record header avoids architecture-specific union layouts.
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        function = kernel32.GetLogicalProcessorInformationEx
        function.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        function.restype = wintypes.BOOL
        length = wintypes.DWORD(0)
        relation_processor_core = 0
        function(relation_processor_core, None, ctypes.byref(length))
        if length.value == 0:
            return None
        buffer = ctypes.create_string_buffer(length.value)
        if not function(relation_processor_core, buffer, ctypes.byref(length)):
            return None
        offset = 0
        count = 0
        header_size = ctypes.sizeof(wintypes.DWORD) * 2
        while offset + header_size <= length.value:
            relationship = wintypes.DWORD.from_buffer_copy(buffer, offset).value
            record_size = wintypes.DWORD.from_buffer_copy(
                buffer, offset + ctypes.sizeof(wintypes.DWORD)
            ).value
            if record_size < header_size or offset + record_size > length.value:
                break
            if relationship == relation_processor_core:
                count += 1
            offset += record_size
        return count or None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _total_memory_bytes() -> int | None:
    if platform.system() == "Windows":
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * pages
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _available_memory_bytes() -> int | None:
    if platform.system() == "Windows":
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * pages
    except (AttributeError, OSError, TypeError, ValueError):
        return None


__all__ = [
    "DEFAULT_MEMORY_PER_SAMPLE_BYTES",
    "DEFAULT_MEMORY_RESERVE_BYTES",
    "GIB",
    "GPUInfo",
    "HardwareInfo",
    "ResourceAllocation",
    "allocate_resources",
    "detect_gpus",
    "detect_hardware",
    "format_bytes",
    "recommend_allocation",
]
