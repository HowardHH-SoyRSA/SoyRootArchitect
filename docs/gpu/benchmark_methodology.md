# CPU/CUDA benchmark method

`scripts/benchmark_gpu_pipeline.py` runs a complete analysis in a fresh child
process for each sample and compares the resulting bundles after reloading their
scientific exports. It records both pipeline elapsed time and process elapsed
time; the latter additionally includes Python startup and imports. CUDA work is
synchronized before elapsed time is finalized. Samples run sequentially to
avoid CPU/GPU contention. One observation per sample and backend does not
estimate timing variance.

Elapsed ratios in the six-sample table are one-pass observations. They are not
isolated CUDA speedups: CPU-only tracing/fitting stages can also change with
hardware, cache and background conditions. The isolated warm helper and one
later Baxi CPU replay are used to assess backend costs and timing drift.

CUDA timings include fresh-process/context startup. Validation populated
`.cupy-cache`, so the compiled-kernel disk cache may already be warm. These
measurements must not be described as cold JIT compilation.

The CPU baseline is a frozen copy of the current CPU source, including the
uncommitted changes explicitly selected by the user. Source SHA-256 hashes are
saved in `outputs/gpu_benchmark/frozen_manifest.json` and every run result. CPU
runs refuse a baseline whose Python source hashes have changed. Every sample
also has a frozen source mesh hash and hashes for external guidance/correction
files, if present.

Settings come from the completed automatic October 4 bundles. For each sample,
the saved analysis vertex count is supplied as an explicit `sample_points` cap,
so runtime-dependent preflight estimates cannot change the analysis subset.
All six saved analysis counts are full-resolution source vertex counts. The
saved random seed is 42, both versions use two SciPy worker threads, and BLAS
thread pools are limited to one thread. Existing automatic endpoint selection
is retained. The same sample config is used for CPU and CUDA; output directory
and `compute_backend` are the only differences.

| Sample | Frozen analysis/source vertices |
|---|---:|
| BaxiNo2_4-2_20260525 | 101,102 |
| Kaixinlv_3-2_20260525 | 345,297 |
| SN14_6-2_20260405 | 526,324 |
| w5168-3_m4-2_20260415 | 171,120 |
| W82_9cm_water_1-2_20260522 | 164,657 |
| W82_MS4-2_20260617 | 177,101 |

Example PowerShell commands from the isolated GPU checkout:

```powershell
& 'E:\SoyRSA Build\.venv\Scripts\python.exe' scripts\benchmark_gpu_pipeline.py prepare --source-root outputs\gpu_benchmark\baseline\src
& 'E:\SoyRSA Build\.venv\Scripts\python.exe' scripts\benchmark_gpu_pipeline.py run --backend cpu --python 'E:\SoyRSA Build\.venv\Scripts\python.exe' --source-root outputs\gpu_benchmark\baseline\src --continue-on-failure
$env:CUPY_CACHE_DIR = (Join-Path (Get-Location) '.cupy-cache')
& 'E:\SoyRSA Build\.venv\Scripts\python.exe' scripts\benchmark_gpu_pipeline.py run --backend cuda --python .venv-gpu\Scripts\python.exe --source-root src --continue-on-failure
& 'E:\SoyRSA Build\.venv\Scripts\python.exe' scripts\benchmark_gpu_pipeline.py compare --source-root outputs\gpu_benchmark\baseline\src
& .venv-gpu\Scripts\python.exe scripts\benchmark_gpu_pipeline.py warm --source-root src --repeat 3
& 'E:\SoyRSA Build\.venv\Scripts\python.exe' scripts\benchmark_gpu_pipeline.py run --backend cpu --sample BaxiNo2_4-2_20260525 --python 'E:\SoyRSA Build\.venv\Scripts\python.exe' --source-root outputs\gpu_benchmark\baseline\src --workdir outputs\gpu_benchmark\repeat_baxi
& 'E:\SoyRSA Build\.venv\Scripts\python.exe' scripts\benchmark_gpu_pipeline.py repeat-check --source-root outputs\gpu_benchmark\baseline\src
```

The `prepare` command refuses an existing manifest. `run` refuses to overwrite
any nonempty output bundle. `--resume` skips only recorded completed runs.
`--sample <sample-stem>` selects one sample. Failed runs retain their traceback
and logs and do not produce a parity claim.

The comparison reloads every scientific JSON, CSV, NPZ, XLSX, PLY and RSML file.
For PLY it reads every scalar vertex property, including root labels, root
orders and assignment states, and native faces. NPZ arrays retain their exact
shape and dtype. XLSX comparisons include all sheet names and cell values or
formulas, avoiding ZIP/container timestamps. CSV row order and raw fields are
retained. All hierarchy, centerline, trait, collar and QC content remains in
the comparison. PNG presentation figures have a separate decoded-pixel
comparison, including image dimensions and color mode; container metadata is
outside that comparison.

Strict parity requires exact equality of all compared values. Numeric drift
is summarized by mismatch count, location and maximum absolute difference;
it is never silently accepted as an exact parity pass. Complete bundle checks
also independently recount assigned/unassigned/uncertain vertices, verify
root/order/parent correspondence and immutable-top origin bounds, and count
native higher-order contact edges with the primary. Existing unresolved
biological QC remains visible even when CPU/CUDA values match.

The only operational exclusions are:

- `metadata.json:stage_timings_seconds`.
- `metadata.json:config.output_dir`, normalized to a fixed placeholder.
- `metadata.json:config.compute_backend`.
- `metadata.json:source_geometry.projected_full_analysis_seconds`.
- `metadata.json:backend_provenance`, if present.
- `metadata.json:outputs` entries for the two operational JSON files below.
- `processing_resources.json`, which records process and memory observations.
- `backend_provenance.json`, which records backend/device/kernel observations.
- `root_system.rsml:metadata/last-modified`.

Process RSS/private memory is sampled every 0.1 seconds; the Windows process
peak working set is also included. These process figures are distinct from
CUDA memory. Actual CUDA calls, kernel counts, synchronization and allocation
statistics are retained separately in backend provenance.

After the complete pipeline runs, `warm` measures the exposed-segment helper
using all actual Baxi vertices and the final emitted hierarchy. CPU and CUDA
reuse one immutable CPU segment index; query-result caching is disabled. It
records the first CPU helper call, CUDA context initialization, first CUDA
helper call and three warm calls per backend, including transfers and exact
CPU contender refinement. Every returned distance/owner/competitor array must
match the CPU reference exactly. These helper timings have narrower scope than
the full pipeline measurements.

The later Baxi CPU run uses a separate output directory and identical frozen
source, runtime and settings. Its complete scientific output and presentation
pixels are compared to the first CPU run. This extra replay preserves the
original six CPU/CUDA pairs and provides a drift observation rather than a
statistical estimate of runtime variance.

The runner was checked against a completed Baxi bundle: all 30 scientific
exports reload and compare equal with themselves, a deliberately changed
full-resolution root label is detected, and the exported IDs/orders/collar
checks pass. This validates the comparison machinery; the CPU/CUDA results
are reported separately after execution.
