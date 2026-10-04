# Repeated geometry and ownership work: implementation and validation

Implemented on 2026-10-03 against a snapshot of the existing working tree,
using `performance_audit_20261002/astra_assessment.md` as the reference.
The baseline includes the user's pre-existing source changes; comparisons
are against that frozen source, rather than an older saved scientific run.

## Implementation

`MeshGeometryContext` now owns read-only copies of the ordered coordinates and
triangles. Raw native edges, exact lengths, edge incidence, face areas,
centroids, vertex area weights, vertex/face incidence, point and centroid trees,
and the existing ordered geometry SHA-256 are shared throughout one input run.
Caller mutation of its original arrays cannot silently change this context;
validation rejects changed or reordered geometry. The full and analysis tree
are shared only when their ordered coordinates and mapping coincide exactly.

`OwnershipGeometryGeneration` takes an immutable label snapshot, builds stable
vertex groups once, and lazily builds same-label and boundary-edge groups per
edge policy. Fitting, attachment restrictions, primary-contact preservation,
parent-contact assessment, and branch-facing transections use compact local
indices, including isolated owned vertices. Static native edge lengths survive
ownership changes. Graphs and component IDs do not: any changed label, including
an in-place change or a later return to an earlier label array, starts a fresh
generation. Component caches also include the exact edge policy and active mask.
Cleanup rebuilds its ownership generation after island transfers before
evaluating holes. Frozen before/after component comparisons use their respective
vertex mappings rather than transferring component numbers between snapshots.

`_ExposedSegmentIndexCache` retains exact subdivisions for individual exposed
bodies, keyed by geometry SHA-256, shape, body start, spacing, and indexing
policy. A sibling change can reuse those pieces while rebuilding the assembled
midpoint tree. The assembled index additionally includes ordered numeric root
labels. Exact query evidence includes the index, ordered point-coordinate
fingerprint, and exact midpoint search radius. Ownership masks and the analysis
and full-resolution assignment policies are applied separately after querying.
The two closest **distinct roots** and numeric-label tie order are retained.
Cached results are snapshots; returned copies cannot poison a later query.

The index cache holds four assembled indexes by default. Individual body
subdivisions have a 16 MiB/512-entry limit; exact query arrays have a 32 MiB limit.
Evicting an assembled index also removes its query entries. Static policy-length
views are bounded to eight entries. These limits describe retained cache arrays,
not total process memory or transient query allocations.

`select_non_overlapping_paths` retains its original stable initial ordering,
strict `value > best_value` choice, zero-value stopping rule, support preference,
and identity-based removal. Directed duplicate evidence and retained-path trees
are cached lazily for one selection invocation. A sparse vertex-to-candidate
index increments overlap counts only for newly claimed vertices. Static growth
lengths are computed once; score arithmetic and root renaming remain unchanged.
There is no global combinatorial solver or reduction in the candidate search.

Raw native contact and supported repair edges remain separate. The `4*d_bar`
and six-times-median support policies remain distinct. Collar exclusions,
immutable primary-top references, unresolved contacts/patches, and child-length
review rules retain their existing behavior.

## Validation

The final suite completed with **473 passed in 45.56 s**:

```powershell
.venv\Scripts\python.exe -m pytest -q tests --basetemp performance_reuse_20261003/tests-final
```

New regressions cover caller mutation, read-only geometry, label-generation
changes and reversions, active/edge policy changes, compact weighted graph
parity with isolated vertices, changed siblings/body starts, exact second-root
competition and numeric-label ties, query-cache poisoning, byte bounds, and
selection against the original greedy oracle. Existing whole-pipeline and
scientific-contract regressions passed as part of the suite.

Five unprofiled repetitions of the bounded helper workloads produced:

| Workload | Baseline median | Optimized median | Ratio |
|---|---:|---:|---:|
| Selection: 168 real Baxi candidates, identical 18 retained roots | 0.376 s | 0.066 s | 5.7x |
| Grouped native graphs: 120,000 vertices, 300 synthetic roots | 0.073 s | 0.031 s | 2.3x |
| Repeated identical exact query: 10,000 Baxi vertices, warm cache | 0.0235 s | 0.000153 s | 154x |

The assignment number measures an exact cache hit. Initial querying still has
its normal geometric cost (0.030 s for the optimized first call in this probe).
Changed geometry, candidate domains, body starts, spacing, labels, query points,
or search radius must compute fresh affected evidence. These helper ratios do
not predict whole-pipeline speed.

A complete BaxiNo2 replay used the unchanged September 23 saved configuration
and full native input, with the baseline and optimized source in separate
processes and output directories. Its **101,102 final labels** and **102 root
records** match exactly. All **31 scientific files** match, comparing JSON
content, NPZ arrays, XLSX cell values, CSV/PLY contents, and parsed RSML. This
includes hierarchy/centerlines, traits, collar/contact/patch QC, original
geometry, and ownership evidence arrays. Two capped point-only replays
(`surface_points` and `occupied_volume`, 270 full vertices/120 analysis vertices)
also match all 31 scientific files and final labels.

Only the following nondeterministic fields are excluded:

- `metadata.json:stage_timings_seconds`
- `metadata.json:config.output_dir`
- `metadata.json:source_geometry.projected_full_analysis_seconds`
- `root_system.rsml:metadata/last-modified`

The complete Baxi run took **166.6 s baseline versus 154.2 s optimized**, a
7.4% reduction in this single pair. The saved pipeline configuration uses two
SciPy workers; the benchmark script holds native numerical thread pools at one.
Helper benchmarks use one SciPy worker. The complete-run timing is a single
observation, not a repeated throughput measurement or a six-sample speed claim.
Full six-sample replay, peak-memory measurement, and batch throughput have not
been certified by this change.

## Reproduction and artifacts

`scripts/benchmark_repeated_work.py` prepares the real eight-start Baxi capture
once and then compares helper workloads or complete saved-config pipelines
against an explicit `--source-root`. Source hashes and scientific fingerprints
are recorded in each JSON artifact. `performance_reuse_20261003/summarize.py`
fails if the compared scientific fingerprints differ or the current optimized
source no longer matches the measured source.

Artifacts are in `performance_reuse_20261003/`: the initial source snapshot under
`baseline/`, `changes.diff`, captured `baxi_candidates.pkl`, baseline/optimized
benchmark JSON, the complete Baxi and point-only replay directories/JSON, and
`summary.json`. Existing Seafile inputs and scientific output bundles were read
as sources; replay outputs are separate local artifacts.
