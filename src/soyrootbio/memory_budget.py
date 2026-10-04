"""Live admission budgets for independent sample processes, not point caps."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import threading

from .hardware import GIB


@dataclass(frozen=True)
class MemorySnapshot:
    available_physical: int | None = None
    available_commit: int | None = None
    total_physical: int | None = None
    commit_limit: int | None = None


def memory_snapshot() -> MemorySnapshot:
    """Windows commit headroom matters even when some RAM is still free."""
    import os
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Status(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "physical", "free_physical", "commit", "free_commit",
                    "virtual", "free_virtual", "extended")]
        status = Status()
        status.size = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return MemorySnapshot(status.free_physical, status.free_commit, status.physical, status.commit)
    try:
        import psutil
        mem = psutil.virtual_memory()
        return MemorySnapshot(mem.available, None, mem.total, None)
    except (ImportError, OSError):
        return MemorySnapshot()


def estimate_decoded_memory(vertices: int, faces: int, original_vertices: int = 0) -> int:
    """Conservative planning heuristic including candidate sets and mesh audits.

    Not a hard allocation bound. Observed private memory and peaks also inform
    admission. Never reduce vertices or hypotheses to meet this estimate.
    """
    return max(4 * GIB, 2 * GIB + int(vertices) * 16384 + int(faces) * 192 + int(original_vertices) * 48)


def estimate_input_memory(path: Path) -> int:
    """Cheap pre-spawn estimate; the worker updates it from decoded geometry."""
    try:
        if path.suffix.lower() == ".stl":
            with path.open("rb") as stream:
                header = stream.read(84)
            if len(header) == 84:
                faces = struct.unpack_from("<I", header, 80)[0]
                if path.stat().st_size == 84 + 50 * faces:
                    # Typical closed STL has ~F/2 distinct positions. The raw
                    # 3F facet records are budgeted separately during decoding.
                    return estimate_decoded_memory((faces + 4) // 2, faces, 3 * faces)
            # ASCII STL: bounded-memory facet counting, no geometry allocations.
            with path.open("r", encoding="ascii", errors="replace") as stream:
                faces = sum(line.lstrip().lower().startswith("facet normal") for line in stream)
            if faces:
                return estimate_decoded_memory((faces + 4) // 2, faces, 3 * faces)
        elif path.suffix.lower() == ".ply":
            vertices = faces = 0
            with path.open("rb") as stream:
                for _ in range(256):
                    line = stream.readline(4096).decode("ascii", errors="replace").strip()
                    if line.startswith("element vertex "):
                        vertices = int(line.split()[-1])
                    elif line.startswith("element face "):
                        faces = int(line.split()[-1])
                    elif line == "end_header":
                        return estimate_decoded_memory(vertices, faces, vertices)
    except (OSError, ValueError):
        pass  # The pipeline reports invalid input normally.
    return 4 * GIB


class MemoryAdmission:
    """Reserve future growth atomically while accounting for resident workers.

    Available memory already excludes committed worker allocations, so only
    their unused reservation is subtracted. Paused workers retain reservations.
    Unknown telemetry retains the entire reservation (conservative).
    """
    def __init__(self, snapshot=memory_snapshot, reserve_bytes: int = 2 * GIB):
        self.snapshot = snapshot
        self.reserve_bytes = reserve_bytes
        self._active: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()

    def try_acquire(self, key: str, estimate: int) -> bool:
        with self._lock:
            if key in self._active:
                return True
            state = self.snapshot()
            totals = [x for x in (state.total_physical, state.commit_limit) if x is not None]
            if totals and estimate + self.reserve_bytes > min(totals):
                raise MemoryError(
                    f"Estimated sample memory {estimate / GIB:.1f} GiB plus "
                    f"{self.reserve_bytes / GIB:.1f} GiB reserve exceeds machine capacity. "
                    "Use a machine with more memory or review the input geometry; no analysis points were discarded."
                )
            available = [x for x in (state.available_physical, state.available_commit) if x is not None]
            unused = sum(max(0, budget - current) for budget, current in self._active.values())
            if not available:
                # Unknown capacity permits one sample, never an unbounded batch.
                ready = not self._active
            else:
                ready = min(available) - unused - self.reserve_bytes >= estimate
            if ready:
                self._active[key] = (estimate, 0)
            return ready

    def update(self, key: str, current: int, estimate: int) -> None:
        with self._lock:
            if key in self._active:
                previous, _ = self._active[key]
                self._active[key] = (max(previous, estimate, current), current)

    def release(self, key: str) -> None:
        with self._lock:
            self._active.pop(key, None)
