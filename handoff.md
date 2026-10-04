# GPU-version handoff

Date: 2026-10-04 (Asia/Shanghai)

Continuation branch: [HowardHH-SoyRSA/SoyRootArchitect — GPU-version](https://github.com/HowardHH-SoyRSA/SoyRootArchitect/tree/GPU-version).
Implementation commit: `c9e4432e72b8fcf4c632860b43d2596649bcb629`.

## Intention

Explore whether this machine's NVIDIA GeForce RTX 5090 can reduce processing
time for SoyRootArchitect while preserving the CPU version's scientific
results. The starting proposal was section 5, “GPU/CUDA feasibility on this
machine,” in `E:\SoyRSA Build\performance_audit_20261002\astra_assessment.md`.

Development was isolated from the existing CPU application: separate worktree,
Python environment, launcher, output location and desktop shortcut. All GPU
development and publication belong on `GPU-version`, not `Current-build` or
the repository's default branch. The original CPU application and six source
PLY meshes remain the user's reference data.

The user explicitly selected the current uncommitted CPU centerline/topology
changes as the comparison baseline. Those changes were frozen before CUDA was
added and included in the GPU branch. The baseline is reconstructible from
commit `e149ba1fd95275ddfa3520b2fe772d220983b826`, the
[CPU source patch](docs/gpu/cpu_baseline.patch), and
[source hashes](docs/gpu/cpu_source_provenance.json).

## Work completed

### Dependencies and machine checks

GPT-6.1 Sol with Extra High reasoning reviewed
[Install GPU Python libraries](codex://threads/01a0fd65-6c4f-75d1-a799-72a50174a8cc)
and repeated actual GPU computations, rather than relying on successful imports.

- Native Windows: CuPy 14.2.0, PyTorch 2.13.0+cu130 and Numba 0.68.0 with
  numba-cuda 0.30.4 passed; both CuPy-first and PyTorch-first import orders worked.
- Ubuntu-24.04 under WSL2: CuPy 14.2.0, PyTorch 2.14.1+cu132, Numba 0.64.0,
  cuML 26.08.00 and cuGraph 26.08.00 passed actual device calculations.
- Package consistency checks passed. The original CPU application environment
  was preserved; a separate `.venv-gpu` application environment used CuPy and
  the CUDA Runtime/NVRTC wheels. PyTorch, Numba and RAPIDS were unnecessary for
  the implemented projection kernel.

These are historical verification results. See the local retirement section
before assuming any of these environments still exists. Detailed versions,
paths and smoke results are in [dependency verification](docs/gpu/dependency_verification.md)
and [the validated application package list](docs/gpu/validated_environment.txt).

### Implemented CUDA boundary

The first proposal boundary was implemented in
[`src/soyrootbio/gpu_backend.py`](src/soyrootbio/gpu_backend.py): sparse
point-to-segment projection using CuPy RawKernel and float64 arithmetic.

- The existing CPU spatial index selects candidate pairs; no dense all-pairs
  matrix or approximate neighbor search was introduced.
- GPU launches are bounded to 262,144 pairs. Segment geometry stays on the
  device through each query. Fused multiply-add is disabled.
- CPU refinement recalculates possible first and second distinct-root
  contenders with the original NumPy expression, retaining deterministic ties.
- Explicit CUDA selection fails clearly if CUDA is unavailable. A CPU backend
  remains available for comparisons. The GPU branch CLI and dedicated launcher
  default to required CUDA.
- Each completed GPU bundle includes backend provenance: device and runtime,
  synchronized timing, pair/launch counts and owned device-pool allocations.
- A separate desktop shortcut, `SoyRootArchitect GPU`, was created and its
  target, real Tk startup, CUDA configuration and output isolation were verified.

This is a hybrid implementation. Most tracing, spatial indexing, topology,
final fitting and export remain on the CPU. GPU mesh reductions, PCA,
neighborhood batching and graph redesign were not implemented in this pass.

### Validation and measurements

GPT-6.1 Sol with Extra High reasoning performed the comparison on the six PLY
samples in `E:\Seafile\Test files for BioInsAlgo`. Both backends used frozen
source/configuration, full-resolution vertex counts, seed 42, two SciPy worker
threads and one BLAS thread. Runs were sequential. Elapsed time includes
backend startup and export; the kernel disk cache could already be warm.

| Sample | Vertices | CPU seconds | CUDA seconds |
|---|---:|---:|---:|
| BaxiNo2_4-2_20260525 | 101,102 | 191.84 | 154.19 |
| Kaixinlv_3-2_20260525 | 345,297 | 629.28 | 644.39 |
| SN14_6-2_20260405 | 526,324 | 1,312.07 | 1,323.54 |
| w5168-3_m4-2_20260415 | 171,120 | 260.51 | 270.47 |
| W82_9cm_water_1-2_20260522 | 164,657 | 228.22 | 230.63 |
| W82_MS4-2_20260617 | 177,101 | 279.02 | 281.28 |
| **Total** | | **2,900.95** | **2,904.50** |

All **180 scientific exports and 24 rendered PNG images matched exactly**
after reload, excluding only explicitly documented operational fields such
as timings, backend selection and output paths. Labels, uncertainty, native
faces, hierarchy, centerlines, traits and QC were retained in the comparison.
All six CUDA runs executed real kernels: 6,984 launches and 263,492,169 sparse
pairs in total. The complete regression suite passed **570 tests**, including
opt-in real CUDA tests; package consistency and desktop startup also passed.

There was **no demonstrated whole-pipeline speed gain**. Aggregate CUDA time
was 0.12% longer. The isolated warm projection helper measured 0.610805 seconds
on CPU versus 0.557800 seconds on CUDA, an observed 1.095x ratio. A later Baxi
CPU replay took 143.403 seconds, versus the first CPU run's 191.840 seconds and
CUDA's 154.187 seconds, with identical outputs. This substantial timing drift
means the first Baxi result must not be advertised as a CUDA speedup.

Evidence:

- [Full comparison and machine-readable results](docs/gpu/benchmark_comparison.md)
- [Methodology](docs/gpu/benchmark_methodology.md) and [frozen manifest](docs/gpu/benchmark_manifest.json)
- [Warm helper measurements](docs/gpu/benchmark_warm_projection.md)
- [CPU timing repeat](docs/gpu/benchmark_baxi_repeat.md)
- [Regression and desktop validation](docs/gpu/validation.md)
- [Reproducible benchmark runner](scripts/benchmark_gpu_pipeline.py)

## Scientific constraints and remaining QC

Preserve the immutable primary-root top, above-collar exclusions, below-top
origins, distinct-root competition and native mesh connectivity. Higher-order
roots must not contact the primary; isolated child patches on a parent must
be corrected only with supporting evidence or reported unresolved. Child
length alone must never delete a branch or force hierarchy changes.

Exact CPU/CUDA parity is not biological compliance. Both versions retained
unresolved child patches in all six samples. SN14 retained 244 native
higher-order/primary contact edges, correctly reported as `unresolved_contacts`.
These violations were not resolved by the GPU implementation. Do not hide
them, invent geometry, or relax evidence requirements to obtain a passing result.

## Ideal advantages of further GPU development

The objective is identical scientific output with less elapsed time. Potential
benefits, which remain to be demonstrated for the full pipeline, are:

1. Faster full-resolution processing by batching many independent geometric
   calculations on the GPU.
2. Higher sample throughput when the expensive stages contain enough parallel
   work to benefit from acceleration.
3. Faster repeated analyses by reusing immutable geometry and intermediate
   arrays on the device across suitable operations.
4. Less pressure to downsample within a practical processing budget, subject
   to device memory and the remaining CPU workload.

GPU execution does not inherently improve biological accuracy. Startup,
transfers, synchronization, CPU refinement and sequential control work can
erase gains from a fast kernel. Keeping suitable data on the device and
batching operations are design requirements, not evidence of achieved speedup.

For continuation, profile representative complete workloads first. CPU lateral
tracing, point assignment and final fitting accounted for approximately 94%
of aggregate baseline stage time. Identify the genuinely parallel work within
those stages before choosing another CUDA boundary. Retain float64 and exact
tie/threshold behavior unless a separate scientific validation explicitly
supports a change. Use repeated, alternating CPU/CUDA runs to measure variance,
and continue checking complete exports and unresolved QC.

## Local retirement and future continuation

The user requested this handoff be published to `GPU-version` before removing
the local GPU development installation. Cleanup covers the GPU worktree,
`.venv-gpu`, generated benchmark bundles, test files/caches and GPU shortcut;
the standalone Windows GPU-library environment and its setup files; and the
Ubuntu-24.04 distribution installed solely for the referenced GPU-library task.
That Ubuntu inspection found only the GPU environment and caches, with no
additional user projects. Existing Docker/WSL infrastructure, the NVIDIA
driver, original CPU checkout/environment/shortcut and source samples are
outside this cleanup.

The source code and compact evidence remain in GitHub. Full local benchmark
bundles and environments are disposable and are not stored in this branch.
Older reports describe the validated historical installation, not continuing
local availability. Resume from a fresh `GPU-version` checkout, follow
[GPU setup](docs/gpu/README.md), reconstruct the frozen CPU baseline if needed,
and rerun validation on the new environment. Continue committing and pushing
to `GPU-version` and verify the remote commit after publication.
