# CPU/CUDA six-sample comparison

All 180 scientific exports and 24 PNG images match exactly after reload. CPU source consistent: True; CUDA source consistent: True.

Aggregate pipeline elapsed: CPU 2900.95s; CUDA 2904.50s. CUDA was 0.12% longer. This one-pass test does not establish a whole-pipeline performance gain.

The isolated warm helper measured CPU 0.610805s versus CUDA 0.557800s, an observed 1.095× ratio with exact evidence parity. [Warm helper report](benchmark_warm_projection.md).

Later Baxi CPU replay: 143.403s versus first CPU 191.840s and CUDA 154.187s. All repeated outputs match exactly. [CPU repeat report](benchmark_baxi_repeat.md).

Existing unresolved child patches remain in 6 samples. SN14 retains 244 native higher-order contact edges with the primary; metadata correctly reports `unresolved_contacts`. Exact CPU/CUDA parity does not establish biological compliance.

Comparison performed by GPT-6.1 Sol, Extra High.

Completed runs: CPU 6/6; CUDA 6/6.

One fresh child process per sample, CPU/CUDA sequential, pipeline elapsed includes backend context startup and synchronized CUDA completion; subprocess elapsed also includes imports. Compiled-kernel disk cache may already be warm from validation; these are not cold-JIT timings.

One-pass elapsed ratios are observations, not isolated CUDA speedups. CPU-only stages also vary between runs; hardware/cache/background variation can affect the totals. The warm helper isolates backend work, and a later Baxi CPU repeat checks timing drift.

Both versions use the current CPU source snapshot, frozen Oct4 sample settings, two SciPy workers, one BLAS thread and the same explicit analysis count and seed. Each backend is run once per sample; the timings do not estimate run-to-run variance.

| Sample | Analysis vertices | CPU s | CUDA s | CPU/CUDA | Exact scientific parity |
|---|---:|---:|---:|---:|---|
| BaxiNo2_4-2_20260525 | 101,102 | 191.84 | 154.19 | 1.244× | True |
| Kaixinlv_3-2_20260525 | 345,297 | 629.28 | 644.39 | 0.977× | True |
| SN14_6-2_20260405 | 526,324 | 1312.07 | 1323.54 | 0.991× | True |
| w5168-3_m4-2_20260415 | 171,120 | 260.51 | 270.47 | 0.963× | True |
| W82_9cm_water_1-2_20260522 | 164,657 | 228.22 | 230.63 | 0.990× | True |
| W82_MS4-2_20260617 | 177,101 | 279.02 | 281.28 | 0.992× | True |

Aggregate measured pipeline elapsed: CPU 2900.95s, CUDA 2904.50s, ratio 0.999×.

| Sample | CPU peak RSS GiB | CUDA peak RSS GiB | CUDA owned pool MiB | Kernel launches | Sparse pairs |
|---|---:|---:|---:|---:|---:|
| BaxiNo2_4-2_20260525 | 0.809 | 1.044 | 22.55 | 440 | 16,364,482 |
| Kaixinlv_3-2_20260525 | 1.537 | 1.483 | 25.63 | 1773 | 61,578,101 |
| SN14_6-2_20260405 | 3.875 | 4.137 | 57.98 | 2429 | 104,241,037 |
| w5168-3_m4-2_20260415 | 1.117 | 1.305 | 26.02 | 666 | 26,947,125 |
| W82_9cm_water_1-2_20260522 | 1.206 | 1.403 | 31.82 | 691 | 23,201,700 |
| W82_MS4-2_20260617 | 1.138 | 1.524 | 22.57 | 985 | 31,159,724 |

CPU source identical across samples: True; CUDA source identical across samples: True.

| Pipeline stage, aggregate | CPU s | CUDA s |
|---|---:|---:|
| export | 16.76 | 15.42 |
| final_centerline_fitting | 573.12 | 556.65 |
| lateral_tracing | 1290.74 | 1327.70 |
| load_geometry | 7.92 | 9.45 |
| normalization | 7.73 | 6.98 |
| point_assignment | 864.10 | 851.69 |
| primary_detection | 11.83 | 10.85 |
| primary_segmentation | 27.85 | 28.16 |
| topology_repair | 61.02 | 55.73 |
| trait_measurement | 4.14 | 4.12 |
| validation_figures | 34.82 | 34.22 |

Internal stage timings exclude the wrapper's CUDA context startup and final backend-provenance write; the primary elapsed-time table includes them.
The comparison reloads all scientific JSON, CSV, NPZ, XLSX, PLY and RSML exports. It preserves IDs, labels, topology, centerlines, traits, native mesh faces, uncertainty and QC. XLSX ZIP/container timestamps are outside the reloaded cell comparison. PNG presentation figures have a separate decoded-pixel comparison. Numerical differences remain failures of exact parity.

Ignored operational fields:

- metadata.json:stage_timings_seconds
- metadata.json:config.output_dir (normalized to <benchmark-output>)
- metadata.json:config.compute_backend
- metadata.json:source_geometry.projected_full_analysis_seconds
- metadata.json:backend_provenance
- metadata.json:outputs entries for processing_resources.json/backend_provenance.json
- processing_resources.json (process/memory/timing observations)
- backend_provenance.json (device/backend/kernel observations)
- root_system.rsml:metadata/last-modified

## BaxiNo2_4-2_20260525

Scientific exports compared: 30. Roots: 82; order counts: {'1': 47, '2': 33, '3': 1}. Assigned/unassigned/uncertain vertices: 100077/169/856.
Native higher-order contact edges: 0; final discrete-child-patch status: unresolved_patches. Existing unresolved QC is retained and does not constitute biological compliance.
Export consistency failures: [].
Actual CUDA pipeline computation verified: True. Initialization 0.242s; host-to-device 0.086s; kernel 0.033s; device-to-host 0.029s; CPU contender refinement 1.739s (2,000,277 pairs).
Presentation figures compared: 4; exact decoded-pixel parity: True.

## Kaixinlv_3-2_20260525

Scientific exports compared: 30. Roots: 287; order counts: {'1': 73, '2': 211, '3': 2}. Assigned/unassigned/uncertain vertices: 334958/4854/5485.
Native higher-order contact edges: 0; final discrete-child-patch status: unresolved_patches. Existing unresolved QC is retained and does not constitute biological compliance.
Export consistency failures: [].
Actual CUDA pipeline computation verified: True. Initialization 0.233s; host-to-device 0.341s; kernel 0.150s; device-to-host 0.136s; CPU contender refinement 7.176s (9,005,781 pairs).
Presentation figures compared: 4; exact decoded-pixel parity: True.

## SN14_6-2_20260405

Scientific exports compared: 30. Roots: 247; order counts: {'1': 164, '2': 66, '3': 16}. Assigned/unassigned/uncertain vertices: 501170/7541/17613.
Native higher-order contact edges: 244; final discrete-child-patch status: unresolved_patches. Existing unresolved QC is retained and does not constitute biological compliance.
Export consistency failures: [].
Actual CUDA pipeline computation verified: True. Initialization 0.240s; host-to-device 0.503s; kernel 0.240s; device-to-host 0.330s; CPU contender refinement 13.709s (12,872,766 pairs).
Presentation figures compared: 4; exact decoded-pixel parity: True.

## w5168-3_m4-2_20260415

Scientific exports compared: 30. Roots: 120; order counts: {'1': 87, '2': 27, '3': 5}. Assigned/unassigned/uncertain vertices: 165188/1160/4772.
Native higher-order contact edges: 0; final discrete-child-patch status: unresolved_patches. Existing unresolved QC is retained and does not constitute biological compliance.
Export consistency failures: [].
Actual CUDA pipeline computation verified: True. Initialization 0.762s; host-to-device 0.128s; kernel 0.058s; device-to-host 0.046s; CPU contender refinement 2.867s (3,004,897 pairs).
Presentation figures compared: 4; exact decoded-pixel parity: True.

## W82_9cm_water_1-2_20260522

Scientific exports compared: 30. Roots: 127; order counts: {'1': 56, '2': 54, '3': 16}. Assigned/unassigned/uncertain vertices: 159495/782/4380.
Native higher-order contact edges: 0; final discrete-child-patch status: unresolved_patches. Existing unresolved QC is retained and does not constitute biological compliance.
Export consistency failures: [].
Actual CUDA pipeline computation verified: True. Initialization 0.226s; host-to-device 0.130s; kernel 0.058s; device-to-host 0.045s; CPU contender refinement 2.500s (3,080,258 pairs).
Presentation figures compared: 4; exact decoded-pixel parity: True.

## W82_MS4-2_20260617

Scientific exports compared: 30. Roots: 196; order counts: {'1': 55, '2': 117, '3': 23}. Assigned/unassigned/uncertain vertices: 168029/1533/7539.
Native higher-order contact edges: 0; final discrete-child-patch status: unresolved_patches. Existing unresolved QC is retained and does not constitute biological compliance.
Export consistency failures: [].
Actual CUDA pipeline computation verified: True. Initialization 0.229s; host-to-device 0.187s; kernel 0.090s; device-to-host 0.073s; CPU contender refinement 3.596s (4,997,186 pairs).
Presentation figures compared: 4; exact decoded-pixel parity: True.

## Baxi CPU timing repeat

Comparison performed by GPT-6.1 Sol, Extra High.

One later CPU replay after all twelve primary runs and the warm helper, using the identical frozen source/runtime/config; this checks drift and does not estimate variance.

First CPU elapsed: 191.840s; later CPU elapsed: 143.403s; CUDA elapsed: 154.187s. Later/first CPU ratio: 0.748. Observed first CPU/CUDA ratio: 1.244; later CPU/CUDA ratio: 0.930.

Exact scientific parity for the repeated CPU outputs: True (30 files). Exact presentation pixel parity: True (4 PNGs).

Changes in CPU-only tracing/fitting stages between the primary CPU and CUDA runs mean the whole observed elapsed difference cannot be attributed to CUDA. Use isolated warm helper timings for backend attribution.
