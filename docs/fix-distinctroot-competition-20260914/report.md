# Distinct-root segment assignment validation

14 September 2026. Implementation and three full-resolution dataset comparisons completed. **266 tests passed.**

The assignment engine now finds the minimum point-to-segment distance for each relevant root, then ranks the closest two distinct root identities by `abs(centerline_distance - interpolated_local_radius)`. Missing radius evidence falls back to zero. Final assignment freezes radii from segmented primary support and unshared traced lateral support before making ownership decisions. The indexed neighborhood includes every segment that can meet the surface-residual bound; there is no fixed-k truncation.

Endpoint projections are clamped; singleton paths and zero-length segments are supported; empty paths are ignored. Numerical distances are quantized at 1e-12 normalized units for deterministic identity/segment tie-breaking. Length-bucketed midpoint KD trees and query chunks avoid a full-resolution points-by-segments matrix.

Competition is used by sampled and full assignment, parent-owned junction resolution, collar counts, and patch correction. Unresolved pairs cannot be erased by primary patch absorption, and unrelated competitor identities veto primary-to-child protrusion transfers. Between tracing orders, uncertain competition remains occupied support. Exported `root_competition.npz` retains vertex indices and both root labels, with the policy, radii and collar counts in `metadata.json`. Evidence describes geometry before the existing final support-based centerline refit.

## Comparison controls

The baseline is a copy of the workspace source taken before these edits. Both end-to-end runs use the saved input and endpoint/guide configuration, full vertex counts, seed 42, `PYTHONHASHSEED=0`, and two workers. Mesh positions and triangles match exactly in each comparison. Earlier exploratory runs are excluded from the tables below.

Two comparisons separate the effects:

- **Whole pipeline:** old and new policies run from input geometry through tracing, topology, assignment, fitting, traits and export. Assignment is also called inside tracing, so retained roots can change.
- **Fixed tracing:** the old downstream pipeline is replayed from the new run's saved pre-assignment state. Root geometry, topology, input points and primary mask start identical. Both pipelines then run their assignment, junction/patch corrections, final fitting, trait calculation and exports. Replaying BaxiNo2 with the same new code reproduces raw labels, competitor pairs, segmented PLY, hierarchy JSON and trait CSV byte-for-byte, validating the replay method.

## Whole-pipeline changes

“New competition” counts vertices absent from the old raw full-assignment competitor map and present in the new map. “Reassigned” counts vertices with different nonnegative output root labels. Whole-pipeline IDs may be renumbered when tracing retains different roots; use the fixed-tracing table for ownership changes with consistent identities.

| Sample | Vertices | New competition | Reassigned | Uncertain before → after | Unassigned before → after | Roots incl. primary |
|---|---:|---:|---:|---:|---:|---:|
| BaxiNo2 | 101,102 | 3,840 | 2,590 | 571 → 2,898 | 1,239 → 1,184 | 90 → 70 |
| Kaixinlv | 345,297 | 14,087 | 16,965 | 2,120 → 8,387 | 105,335 → 104,403 | 204 → 173 |
| w5168-3 | 171,120 | 7,261 | 2,143 | 93 → 5,580 | 14,307 → 14,159 | 82 → 81 |

## Fixed-tracing assignment changes

| Sample | New competition | Reassigned | Newly assigned | Uncertain before → after | Assigned before → after |
|---|---:|---:|---:|---:|---:|
| BaxiNo2 | 4,096 | 1,190 | 176 | 247 → 2,898 | 99,607 → 97,020 |
| Kaixinlv | 13,879 | 3,440 | 1,937 | 1,719 → 8,387 | 238,180 → 232,507 |
| w5168-3 | 5,389 | 1,863 | 263 | 1,994 → 5,580 | 154,876 → 151,381 |

In fixed-tracing comparisons, root membership, parent identities and root orders remain unchanged. Final centerline fitting changes geometric attachments and path traits in response to the changed ownership. Detailed insertion-index/coordinate changes are retained in the comparison JSON.

## Traits and topology

Totals below sum finite values across all roots, including primary. Length is in source mesh units, area in squared mesh units and volume in cubed mesh units. The CSV artifacts contain every changed per-root trait, including point counts, diameters, tips, angles and QC fields.

| Sample | Measure | Whole baseline | Updated | Fixed-tracing baseline |
|---|---|---:|---:|---:|
| BaxiNo2 | length | 919.628 | 892.010 | 912.220 |
| BaxiNo2 | surface_area | 1910.406 | 1867.511 | 1916.379 |
| BaxiNo2 | volume | 454.139 | 448.226 | 451.500 |
| Kaixinlv | length | 1941.631 | 1881.445 | 1939.773 |
| Kaixinlv | surface_area | 4766.516 | 4666.287 | 4777.383 |
| Kaixinlv | volume | 1237.692 | 1228.511 | 1224.874 |
| w5168-3 | length | 1701.362 | 1658.767 | 1685.810 |
| w5168-3 | surface_area | 3817.911 | 3679.000 | 3766.616 |
| w5168-3 | volume | 901.633 | 931.254 | 981.769 |

| Sample | O1/O2/O3 whole baseline → updated | Fixed-tracing centerlines changed | Fixed-tracing roots with trait changes | Unmeasurable lengths whole baseline → updated |
|---|---|---:|---:|---:|
| BaxiNo2 | 53/36/0 → 46/23/0 | 70 | 70 | 12 → 2 |
| Kaixinlv | 75/125/3 → 72/98/2 | 173 | 173 | 0 → 1 |
| w5168-3 | 76/5/0 → 74/6/0 | 80 | 81 | 2 → 1 |

The reduction or redistribution of retained roots is a material whole-pipeline change. More competition evidence and fewer root candidates do not establish biological accuracy; these outputs require inspection before treating the changed counts or traits as improved measurements.

## Runtime and collar evidence

The indexed full assignment alone processed 171,120 vertices against 81 roots in **2.35 seconds**, including index construction, using two workers and frozen radius profiles. Whole-stage times below additionally include radius estimation, junction resolution, surface connectivity and patch corrections; concurrent validation workloads affect timing.

| Sample | Assignment stage before → after (s) | Collar competitors | Collar still uncertain |
|---|---:|---:|---:|
| BaxiNo2 | 16.64 → 26.11 | 74 | 31 |
| Kaixinlv | 38.48 → 49.03 | 510 | 300 |
| w5168-3 | 25.76 → 36.07 | 521 | 338 |

## Validation artifacts

The large replay bundles and per-vertex arrays are intentionally excluded from Git. They remain in the local ignored directory `outputs/distinct_root_segments_20260914/`. This report records their aggregate results; the committed validation scripts reproduce the comparisons from the saved input bundles.

The local artifact set contains:

- whole-pipeline and fixed-tracing comparison JSON;
- byte-for-byte replay-equivalence evidence;
- complete pytest output and the segment-index timing result;
- complete new output bundles for BaxiNo2, Kaixinlv and w5168;
- per-root trait changes, label-transition tables and newly competing vertex indices.
