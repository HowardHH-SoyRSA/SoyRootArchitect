"""Bounded CUDA projection with CPU-exact final competition evidence.

Only sparse pairs from the existing CPU spatial index reach CUDA. No dense
point-by-point matrix, approximate search, topology or precision change is used.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import time

import numpy as np


_ACTIVE: ContextVar["CudaBackend | None"] = ContextVar("soyrootbio_cuda", default=None)
_KERNEL = r'''
extern "C" __global__ void project(
    const double* points, const double* starts, const double* deltas,
    const double* lengths, const long long* qi, const long long* si,
    double* distances, const long long n) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    long long p = 3 * qi[i], s = 3 * si[i];
    double x = points[p] - starts[s];
    double y = points[p+1] - starts[s+1];
    double z = points[p+2] - starts[s+2];
    double dot = (x*deltas[s] + y*deltas[s+1]) + z*deltas[s+2];
    double t = fmin(1.0, fmax(0.0, dot / fmax(lengths[si[i]], 1e-24)));
    x = x - t*deltas[s]; y = y - t*deltas[s+1]; z = z - t*deltas[s+2];
    distances[i] = sqrt((x*x + y*y) + z*z);
}
'''


def active_backend():
    return _ACTIVE.get()


class CudaBackend:
    """One pipeline's device stream, allocations and synchronized counters."""

    max_pairs = 262_144

    def __init__(self):
        started = time.perf_counter()
        try:
            import cupy as cp
            self.cp = cp
            self.device = cp.cuda.Device(0)
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.pool = cp.cuda.MemoryPool()
            self.kernel = cp.RawKernel(_KERNEL, "project", options=("--fmad=false",))
            self.kernel.compile()
            with self.device, self.stream, cp.cuda.using_allocator(self.pool.malloc):
                test = cp.asarray([1.0, 2.0], dtype=cp.float64)
                if float(cp.asnumpy(test.sum())) != 3.0:
                    raise RuntimeError("CUDA arithmetic self-check failed")
            self.stream.synchronize()
        except Exception as exc:
            raise RuntimeError(
                "CUDA backend requested but unavailable. Use the isolated GPU runtime "
                "with CuPy/CUDA installed, or explicitly select --backend cpu."
            ) from exc
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        self.stats = {
            "backend": "cuda", "implementation": "sparse-segment-fp64-v1",
            "cupy": cp.__version__, "numpy": np.__version__,
            "device": name.decode() if isinstance(name, bytes) else name,
            "driver_version": cp.cuda.runtime.driverGetVersion(),
            "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "precision": "float64", "fast_math": False,
            "initialization_seconds": time.perf_counter() - started,
            "host_to_device_seconds": 0.0, "kernel_seconds": 0.0,
            "device_to_host_seconds": 0.0, "cpu_refinement_seconds": 0.0,
            "sparse_pairs": 0, "cpu_refined_pairs": 0, "kernel_launches": 0,
            "peak_owned_device_pool_bytes": self.pool.total_bytes(), "max_pairs_per_launch": self.max_pairs,
            "policy": "CPU index; CUDA projection; CPU exact contenders and distinct-root ties",
        }

    def snapshot(self):
        self.stream.synchronize()
        return dict(self.stats)

    def projector(self, index):
        return SegmentProjector(self, index)

    def close(self):
        self.stream.synchronize()
        self.pool.free_all_blocks()


class SegmentProjector:
    def __init__(self, backend, index):
        self.backend, self.index = backend, index
        cp = backend.cp
        started = time.perf_counter()
        with backend.device, backend.stream, cp.cuda.using_allocator(backend.pool.malloc):
            self.start = cp.asarray(index.start)
            self.delta = cp.asarray(index.delta)
            self.length_squared = cp.asarray(index.length_squared)
        backend.stream.synchronize()
        backend.stats["host_to_device_seconds"] += time.perf_counter() - started
        backend.stats["peak_owned_device_pool_bytes"] = max(
            backend.stats["peak_owned_device_pool_bytes"], backend.pool.total_bytes())

    def contenders(self, points, query_index, segment_index):
        """Return a conservative shortlist with bitwise CPU reference distances.

        Keep every pair at or below the provisional second distinct root plus
        a coordinate-scaled rounding guard, then recompute those pairs with the
        unchanged NumPy oracle. Thus even an approximate CUDA tie cannot choose
        ownership. A sole root uses its nearest segment as the bound.
        """
        b, cp, index = self.backend, self.backend.cp, self.index
        distance = np.empty(len(query_index), dtype=np.float64)
        with b.device, b.stream, cp.cuda.using_allocator(b.pool.malloc):
            started = time.perf_counter()
            device_points = cp.asarray(points)
            b.stream.synchronize()
            b.stats["host_to_device_seconds"] += time.perf_counter() - started
            for begin in range(0, len(query_index), b.max_pairs):
                end = min(begin + b.max_pairs, len(query_index))
                started = time.perf_counter()
                qi = cp.asarray(query_index[begin:end], dtype=cp.int64)
                si = cp.asarray(segment_index[begin:end], dtype=cp.int64)
                output = cp.empty(end - begin, dtype=cp.float64)
                b.stream.synchronize()
                b.stats["host_to_device_seconds"] += time.perf_counter() - started
                started = time.perf_counter()
                b.kernel(((end-begin+255)//256,), (256,),
                         (device_points, self.start, self.delta, self.length_squared,
                          qi, si, output, np.int64(end-begin)))
                b.stream.synchronize()
                b.stats["kernel_seconds"] += time.perf_counter() - started
                started = time.perf_counter()
                distance[begin:end] = cp.asnumpy(output)
                b.stats["device_to_host_seconds"] += time.perf_counter() - started
                b.stats["kernel_launches"] += 1
                b.stats["peak_owned_device_pool_bytes"] = max(
                    b.stats["peak_owned_device_pool_bytes"], b.pool.total_bytes())
        b.stats["sparse_pairs"] += len(distance)
        started = time.perf_counter()
        labels = index.segment_labels[segment_index]
        order = np.lexsort((labels, distance, query_index))
        qp, lab, ds = query_index[order], labels[order], distance[order]
        first = np.r_[True, qp[1:] != qp[:-1]]
        owner = np.full(len(points), -1, dtype=int)
        bound = np.full(len(points), np.inf)
        owner[qp[first]], bound[qp[first]] = lab[first], ds[first]
        different = lab != owner[qp]
        other = qp[different]
        if len(other):
            second = np.r_[True, other[1:] != other[:-1]]
            bound[other[second]] = ds[different][second]
        scale = max(1.0, float(np.max(np.abs(points))),
                    float(np.max(np.abs(index.start))), float(np.max(np.abs(index.delta))))
        tolerance = 256 * np.finfo(np.float64).eps * scale
        keep = distance <= bound[query_index] + tolerance
        # Nonfinite arithmetic must never prune evidence.
        if not np.isfinite(distance).all():
            keep[:] = True
        qi, si = query_index[keep], segment_index[keep]
        relative = points[qi] - index.start[si]
        along = np.clip(np.einsum("ij,ij->i", relative, index.delta[si]) /
                        np.maximum(index.length_squared[si], 1e-24), 0.0, 1.0)
        residual = relative - along[:, None] * index.delta[si]
        exact = np.linalg.norm(residual, axis=1)
        b.stats["cpu_refined_pairs"] += len(exact)
        b.stats["cpu_refinement_seconds"] += time.perf_counter() - started
        return qi, si, exact


@contextmanager
def compute_backend(name: str):
    if name not in {"cpu", "cuda"}:
        raise ValueError("compute_backend must be cpu or cuda")
    backend = CudaBackend() if name == "cuda" else None
    token = _ACTIVE.set(backend)
    try:
        yield backend
    finally:
        _ACTIVE.reset(token)
        if backend is not None:
            backend.close()
