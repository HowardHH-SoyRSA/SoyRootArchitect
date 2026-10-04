# Isolated GPU build

This implementation follows the first prototype boundary in section 5 of the
October 2 performance assessment: exact distances for sparse point/segment
pairs already selected by the existing CPU spatial index. CuPy runs float64
projection kernels on CUDA, with fused multiply-add disabled. Geometry arrays
stay on the device through each index query. Pair transfers and launches are
bounded to 262,144 pairs; no dense all-pairs matrix is created.

The CPU retains every possible first/second distinct-root contender, including
ties within a conservative coordinate-scaled floating-point guard, recomputes
their distances with the unchanged NumPy expression, and applies the existing
stable label tie rule. Assignment radii, ambiguous labels, collar exclusions,
immutable primary top, native connectivity, topology and final measurements
are unchanged by the backend. An unavailable requested CUDA backend fails
clearly; it does not silently substitute CPU processing.

This is a hybrid CPU/CUDA implementation, not a conversion of the whole
pipeline. Mesh reductions, PCA, neighbor support batching, graph algorithms
and Python tracing control remain CPU work. The assessment marks several of
these as conditional or requiring redesign; this initial implementation does
not claim to accelerate them or improve segmentation quality. Real elapsed
speed depends on the measured workload, including transfers and CPU refinement.

## Separate installation

Use Python 3.12 and a compatible NVIDIA CUDA 13 driver on Windows. From an
independent checkout of this branch:

```powershell
git clone --branch GPU-version --single-branch https://github.com/HowardHH-SoyRSA/SoyRootArchitect.git SoyRootArchitect-GPU
cd SoyRootArchitect-GPU
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -e '.[gpu,test]'
$env:CUPY_CACHE_DIR = (Join-Path (Get-Location) '.cupy-cache')
.\.venv-gpu\Scripts\python.exe scripts\launch_gpu.py
```

The validated machine uses CuPy 14.2.0, CUDA Runtime/NVRTC 13.4.92, CUDA
Pathfinder 1.8.3 and NumPy 2.5.1. The application runtime was made as an
independent copy of the CPU site's packages, then installed with its own
editable source mapping and only these GPU dependencies added. No dependency
or source path points back to the CPU application. PyTorch, Numba and RAPIDS
are verified separately but unnecessary for this implementation.

The CUDA Runtime/NVRTC versions above are installed distribution versions.
The actual CuPy run reports runtime API version `13020` and driver API version
`13040` in backend provenance; those reported API values are retained separately.

Create the additional desktop shortcut with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_gpu_shortcut.ps1
```

`SoyRootArchitect GPU.lnk` targets this checkout's `.venv-gpu\Scripts\pythonw.exe`
and `scripts\launch_gpu.py`. The window is marked **GPU / CUDA**. Its initial
output root is this checkout's `outputs\gpu_gui`; choosing another root is
explicit. The CPU launcher and output folders are preserved.

Command-line use:

```powershell
.\.venv-gpu\Scripts\python.exe -m soyrootbio.cli run --backend cuda --input sample.ply --output outputs\gpu_sample
.\.venv-gpu\Scripts\python.exe -m soyrootbio.cli run --backend cpu --input sample.ply --output outputs\cpu_sample
```

`PipelineConfig.compute_backend` defaults to `cpu` for existing callers.
This branch's CLI and the GPU shortcut default to required `cuda`; use
`--backend cpu` explicitly for a command-line CPU run. Batch child processes receive
this field in their serialized configuration; context-local backend state
and an owned stream/memory pool prevent cross-job statistics or cache mixing.

## Evidence and reproducibility

See [validation](validation.md) for the 570 passing regression tests, package
checks and desktop shortcut/startup verification.

The [six-sample comparison](benchmark_comparison.md) found exact agreement in
180 scientific exports and 24 presentation figures. Aggregate pipeline elapsed
was CPU 2,900.95s versus CUDA 2,904.50s (CUDA 0.12% longer). The isolated
[warm helper](benchmark_warm_projection.md) had a CPU/CUDA median ratio of
1.095x; this narrower benefit did not produce an overall pipeline gain.
The [Baxi CPU repeat](benchmark_baxi_repeat.md) records a separate timing-drift
check. Existing unresolved child patches remain in all six samples; SN14 also
retains 244 native higher-order/primary contact edges. These are unresolved
violations in both versions, not compliant segmentation results.

Every completed GPU bundle includes `backend_provenance.json`, also listed
in `metadata.json`. It records device/runtime/package versions, float64 policy,
sparse pair and launch counts, exact CPU-refined pair counts, initialization,
synchronized transfer/kernel/refinement timings and peak owned device-pool
bytes. The latter measures this backend's allocations, not all GPU memory
used by the driver, display, or other applications.

Full pipeline elapsed time in the benchmark includes initialization and export.
The internal stage total begins after CUDA initialization. Fresh-process and
warmed measurements must be interpreted separately; a cached compiled kernel
does not remove CUDA context startup. See the [benchmark methodology](benchmark_methodology.md).

The CPU source snapshot includes the uncommitted centerline/topology changes
explicitly selected by the user, frozen before adding CUDA. The separate CPU
checkout is never edited during this work. The recorded source/input hashes
and baseline patch permit later reconstruction without relying on that folder
remaining unchanged.

`cpu_baseline.patch` reconstructs the frozen scientific source from base commit
`e149ba1fd95275ddfa3520b2fe772d220983b826`. On Windows, apply with
`git apply --ignore-space-change docs/gpu/cpu_baseline.patch` in a separate
checkout at that commit, then verify against `cpu_source_provenance.json` (raw hashes
include the captured Windows line endings). Do not apply it over this already
updated GPU source.

Run regression checks with:

```powershell
New-Item -ItemType Directory -Force .pytest-runs | Out-Null
$env:SOYROOTBIO_TEST_CUDA = '1'
$env:CUPY_CACHE_DIR = (Join-Path (Get-Location) '.cupy-cache')
.\.venv-gpu\Scripts\python.exe -m pytest -q tests --basetemp .pytest-runs\gpu
```

Publish with an explicit destination, then compare the reported SHA with HEAD:

```powershell
git push origin HEAD:refs/heads/GPU-version
git ls-remote origin refs/heads/GPU-version
```

The branch-specific rule is also recorded in `AGENTS.md` for future edits.

Implementation references: [CuPy RawKernel](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawKernel.html)
and [CuPy synchronized performance measurement](https://docs.cupy.dev/en/stable/user_guide/performance.html).
