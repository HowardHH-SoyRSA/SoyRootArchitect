# SoyRootBio / BioInsAlgo 0.2

SoyRootBio is a desktop and command-line application for topology-aware measurement of reconstructed soybean root system architecture (RSA). Its primary inputs are root-only micro-CT surface meshes in STL or PLY format. The software detects a primary root, traces lateral roots recursively, repairs the result into a rooted hierarchy, measures traits in source mesh units, and writes editable and validation-ready outputs.

The project prioritizes the measurement objectives in this repository over exact reproduction of any one publication. It evolved from the MIT-licensed [BioInsAlgo baseline](https://github.com/HowardHH-SoyRSA/BioInsAlgo/tree/agent/refine-primary-centerline-log) and remains MIT licensed. Zhou et al. (2025) motivated the original bio-inspired workflow, but this is not the authors' implementation and is not a bit-for-bit reproduction. See [Research lineage and licensing](#research-lineage-and-licensing) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Capabilities

- Reads STL and PLY meshes while preserving their vertices and faces. OBJ meshes and XYZ/CSV point clouds are also supported.
- Uses a ranked automatic primary-root detector by default, with manual endpoints, a soil-line constraint, and optional guide sections as overrides.
- Represents the primary as order 0, its children as order 1, and recursively assigns every descendant `parent order + 1`.
- Enforces non-increasing centreline length down the hierarchy: an automatic child longer than its parent is removed with its dependent subtree, and equivalent manual edits are rejected atomically.
- Produces an oriented centreline tree, stable root IDs, insertion locations, confidence values, and QC flags.
- Measures per-root length, diameter, tortuosity, surface area, volume estimate, hierarchy, and three explicitly directional angles.
- Writes CSV, multi-sheet XLSX, hierarchy-preserving RSML, editable JSON, labelled full-resolution PLY files, skeleton overlays, and separate 600-dpi angle figures.
- Provides a batch-first Windows desktop GUI with drag/drop, per-sample outputs and primary guidance, ETA/progress, hardware-aware concurrency, pause/resume, and cancellation.
- Provides a local, GPU-accelerated 3D result viewer and graph editor with per-root inspection, nine topology/geometry tools, durable undo/redo, and non-destructive materialisation.
- Keeps the full mesh by default. Automatic analysis reduction is allowed only when the runtime/memory preflight crosses the configured limit; the default runtime limit is 30 minutes.

## Installation

Python 3.10 or newer is required. On Windows PowerShell:

```powershell
git clone https://github.com/HowardHH-SoyRSA/BioInsAlgo.git
cd BioInsAlgo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The main dependencies are Open3D, NumPy, SciPy, HDBSCAN, scikit-learn, NetworkX, pandas, Matplotlib, openpyxl, psutil, Flask, and tkinterdnd2. A normal Python.org Windows installation includes Tcl/Tk for the desktop interface.

For development and tests:

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
```

Large repository data use Git LFS. If a clone contains pointer files rather than meshes, install Git LFS and run:

```powershell
git lfs install
git lfs pull
```

## Desktop batch GUI

Launch the application with:

```powershell
soyrootbio gui
```

Optionally prefill one input and an output root:

```powershell
soyrootbio gui --input "D:\roots\sample.ply" --output "D:\results"
```

The GUI supports these primary-root modes:

- **Scored automatic**: the measurement default.
- **Z-axis extrema**: a simple fallback for consistently oriented, clean roots.
- **Manual soil line + scorer**: constrains the collar search around a supplied soil-surface Z value.
- **Interactive endpoints + sections**: configure a selected sample in the 3D picker; select collar/tip endpoints and optional sections that the primary centreline must cross. The picker can also record a horizontal soil line, and the mouse wheel zooms both the endpoint and primary-section views.
- **XYZ endpoints**: enter collar/base and tip coordinates, with an optional semicolon-separated guide list (`x,y,z; x,y,z`).

Add multiple STL/PLY files with **Add files** or drag/drop them onto the queue. Each sample receives a unique output directory, which can be changed individually. Select one sample and use **Open output folder** to open that directory. The queue/status table has both vertical and horizontal scrollbars for long paths and status messages. Automatic resource allocation reserves a logical CPU for the desktop, considers physical/logical CPU counts and available RAM, and assigns concurrent samples and threads per sample. Both values can be overridden.

If a sample fails, its output directory contains `processing_error.log` with the failure time, input/output paths, exception, traceback, and non-sensitive pipeline configuration. Cancelled jobs are not reported as processing failures. GUI batch clustering avoids nested HDBSCAN worker pools, so sample-level concurrency remains within the scheduler's resource allocation.

The displayed memory summary distinguishes currently available RAM from installed RAM, and automatic concurrency is budgeted against the available value with a reserve for the desktop. Manual concurrency/thread overrides are honored and can intentionally oversubscribe the machine. Each queued output directory must be absent or empty when its run starts. Automatically generated names are suffixed when a path is already present; a manually selected non-empty directory is rejected so stale files cannot be mixed into a new result.

Pause and cancel are cooperative. A running numerical stage may finish its current operation before it observes the request. Step timings are stored under the batch output root in `.soyrootbio_step_timings.json` and are reused for future ETAs. GPU hardware is detected and displayed when available, but the current analysis pipeline is CPU based; there is no required CUDA path.

## Interactive 3D result editor

Open any completed SoyRootBio output bundle with one command:

```powershell
soyrootbio editor --output "D:\results\sample"
```

The command starts a loopback-only local server and opens the editor in the browser. The viewer downloads and parses `segmented_root_structure.ply` off the UI thread, retains every vertex and face, requests the browser's high-performance GPU, computes smooth normals in a worker, and builds a background BVH for fast picking. It does not downsample the result. The status area reports the detected host GPU, the browser WebGL renderer, vertex/face counts, and full-resolution state.

Hovering the surface shows root ID, length, mean diameter, and tip–gravity angle. Selecting a root from the surface or hierarchy highlights its complete centreline, direct parent, direct children, insertion point, and tip, then frames the full path in the 3D view. The hierarchy also lists every triangle-connected uncertain and unassigned mesh patch for direct highlighting, framing, and patch-level inspection. The inspector shows path and chord length, all three directional angles, mean diameter, surface area, volume, tortuosity, parent ID, and a direct-children list that is collapsed by default.

The surface, centreline, hierarchy, hover card, and inspector use the same root-order colours written to the exported PLY and `csv/root_label_map.csv`. The hierarchy side panel includes their exact hex-code legend and retains compact colour swatches when collapsed. Selection preserves those base colours and uses line width to distinguish the selected root and its direct relationships. Mouse-wheel and trackpad scrolling zoom continuously in both directions.

The toolbar provides:

- creation of a new root by drawing a base-to-tip path through grey unassigned points, with a configurable claim radius and automatic attachment to the selected parent;
- split;
- merge using a scale-checked, directed tip-to-start join;
- assign surface points with a 3D radius brush;
- reconnect;
- reparent;
- delete;
- redraw; and
- root-order correction.

Every accepted edit is validated for missing parents, cycles, label integrity, and exact geometric attachment before it becomes visible. `Ctrl+Z`/`Ctrl+Shift+Z` and the toolbar buttons provide undo/redo. Point assignments are resolved against the full vertex array on the local Python process; Shift + left-dragging the assignment brush creates one continuous region and one atomic history operation, while ordinary left-drag retains rotation.

After a successful edit, undo, or redo, the camera keeps the same orbit, zoom, and target used to make the edit while retaining the current root selection and highlight. The initial full-system orbit target is the geometric centre of the primary root.

The automatic output files are opened read-only in memory. By default, editor state is written under:

```text
<automatic output>\.soyrootbio-editor\
  session.json
  operations.jsonl
  blobs\
  materialised\
```

`operations.jsonl` is append-only; large point selections use hashed NumPy sidecar blobs so the log remains compact and replayable. Undo and redo are also recorded as events. **Export edits** writes an edited hierarchy, traits, numeric-label map, full-resolution labelled PLY, nested RSML, operation log, and manifest under the session directory. It never overwrites the automatic hierarchy, traits, label map, or PLY.

Use a separate session location when desired:

```powershell
soyrootbio editor `
  --output "D:\results\sample" `
  --session-dir "D:\root-edit-sessions\sample-review"
```

The editor deliberately binds only to `localhost`/loopback addresses. Its mutation API uses a per-launch, same-site session cookie and rejects cross-origin requests. Run `soyrootbio editor --help` for port and browser-launch options.

## Command-line use

The scored automatic detector and full-mesh analysis are the defaults:

```powershell
soyrootbio run `
  --input "D:\roots\sample.ply" `
  --output "D:\results\sample"
```

The output path may be new or an existing empty directory. A non-empty output directory is rejected; choose a fresh path for every run, including correction reruns.

Constrain the automatic collar detector with a soil-line Z coordinate:

```powershell
soyrootbio run `
  --input "D:\roots\sample.stl" `
  --output "D:\results\sample" `
  --auto-endpoints scored `
  --soil-z 92.5 `
  --max-root-order 3
```

Use endpoints in original input coordinates and force the centreline through guide sections:

```powershell
soyrootbio run `
  --input "D:\roots\sample.ply" `
  --output "D:\results\sample_manual" `
  --start 3.7 59.2 94.6 `
  --end -19.7 67.3 -55.3 `
  --guide-file "D:\roots\sample_primary_guides.csv"
```

An endpoint file may be JSON (`{"start":[x,y,z],"end":[x,y,z]}`), a CSV with `x,y,z` columns and two rows, or text containing six numbers. A guide file may be JSON, CSV with `x,y,z` columns, or whitespace/comma-delimited XYZ rows.

Re-run after editing the exported hierarchy:

```powershell
soyrootbio run `
  --input "D:\roots\sample.ply" `
  --output "D:\results\sample_corrected" `
  --correction-file "D:\results\sample\root_hierarchy.json"
```

In `root_hierarchy.json`, change `parent_id`, edit a `polyline`, or set `"valid": false` on a lateral root. Corrections with missing parents, cycles, invalid polylines, a child centreline longer than its parent, or inconsistent topology are rejected. Root order is recalculated from the corrected parent links.

The primary row is immutable in a correction file; change the primary with endpoints, soil line, or guide sections instead. Lateral root IDs are treated as immutable provenance keys while corrections are applied: deleting a root does not renumber the survivors or reuse the deleted ID. Unknown/stale IDs, duplicate IDs, and stale geometry fingerprints are rejected. If a lateral's parent or polyline is actually changed, its automatic attachment confidence is set to `0` and the QC flags `manual_correction`, `attachment_confidence_invalidated`, and `low_confidence` are added pending review.

Useful options include:

- `--auto-endpoints scored|z|pca` (default `scored`)
- `--sample-points N` for an explicit analysis cap; `0` means no user cap
- `--max-root-order N` (default 3)
- `--max-laterals N` for a deliberate branch-count cap
- `--graph-k N` (default 14)
- `--runtime-limit-minutes M` (default 30)
- `--minimum-retained-fraction F` (default 0.25)
- `--tip-window-mesh-units W` (default 2.0), the source-mesh arc-length window used for local tip, lateral-start, and primary-reference directions

Run `soyrootbio run --help` for the authoritative option list.

For a self-contained smoke test, generate a synthetic taproot and its endpoint file:

```powershell
soyrootbio generate-synthetic `
  --output data/synthetic/synthetic_taproot.csv `
  --lateral-count 3 `
  --seed 21

soyrootbio run `
  --input data/synthetic/synthetic_taproot.csv `
  --endpoint-file data/synthetic/synthetic_taproot_endpoints.csv `
  --output outputs/synthetic_run `
  --max-root-order 2
```

## Primary-root detection

The automatic detector does not hide a single endpoint heuristic. It generates collar-to-tip candidates on a local sparse graph and ranks them using five recorded score components:

1. basal/collar location, relative to gravity or the supplied soil line;
2. local radius and thickness continuity;
3. downward extent;
4. geodesic path length; and
5. graph centrality/reachability.

Candidate rank, score, confidence, endpoints, component values, and QC flags are preserved in `metadata.json`. The winning path is refined from the segmented primary surface. Manual endpoints supersede automatic endpoints; optional guide sections create constrained shortest-path segments so the centreline crosses biologically identified primary-root regions.

The selected or detected base is also the upper assignment boundary. Within a sampling-scaled collar neighbourhood, the boundary follows the local primary cross-section so both walls of a tilted collar are preserved. Outside that neighbourhood, “above” follows gravity so a long sideways lateral is not cut by an infinite oblique plane. Shoot-side points are excluded from primary segmentation, lateral tracing, and full-resolution root assignment. `metadata.json` records both directions, the collar radius, tolerance, rule, counts, and assignment-state explanations under `point_assignment`.

Automatic confidence is evidence for review, not a probability of biological correctness. Inspect the labelled PLY, skeleton overlay, hierarchy, and angle figures, especially for merged roots, broken mesh components, pot/soil remnants, or an ambiguous collar.

## Topology and root order

All paths are oriented from collar/insertion toward the root tip. Before traits are computed, the topology stage:

- attaches each lateral to an already established lower-order parent;
- records insertion point and parent-centreline index;
- reorients reversed paths;
- repairs cycles by reattaching the weakest edge;
- recomputes orders recursively; and
- assigns stable IDs such as `root-o1-001`.

The invariant is `primary = order 0`; a child of the primary is order 1; every other child is `parent order + 1`. The final graph must be connected to the primary, acyclic, and free of missing parents.

## Angle definitions

Gravity is fixed to the directional vector **g = (0, 0, -1)**. Angles use normalized vectors and `acos(clip(a · b, -1, 1))`, so their range is 0–180°; they are not folded into an acute 0–90° axis angle.

For an oriented lateral centreline `p[0] ... p[n-1]`:

- **Tip–gravity** (`tip_gravity_angle_deg`): angle between the lateral tip tangent and **g**. The tangent points toward `p[n-1]` and is interpolated over the final source-mesh arc-length window selected by `--tip-window-mesh-units`.
- **Tip-start–gravity** (`tip_start_gravity_angle_deg`): angle between the vector `p[n-1] - p[0]` and **g**.
- **Lateral-start–primary** (`tip_primary_angle_deg`, historical column name): angle between the lateral tangent over the first mesh-unit window from `p[0]` and the ordered primary-root tangent interpolated around the lateral insertion location.

The requested mesh-unit window is capped at 25% of the referenced path's total length on short paths, making the direction independent of centreline sampling density. `tip_vector_*` records the lateral-tip window and `base_vector_*` records the lateral-start window; both use `mesh_unit`. The CSV/XLSX vector table also includes root start/tip, vector start/end XYZ coordinates, vector components in source mesh units, the local primary reference vector, and gravity components.

Each requested angle has its own 600-dpi X–Z front-view PNG. Every lateral uses the compact `oN-NNN angle°` pattern without the storage-only `root-` prefix. Order-1 laterals shorter than 3.5 mesh units and order-2 laterals shorter than 1.3 mesh units remain visible as segmented roots, but their angle labels, indicatrices, and related vectors are omitted from all angle figures; the numerical traits and vector tables remain unchanged. Labels are vertically distributed in side columns and connected to their corresponding tips by ordered, three-segment polyline indicatrices routed through two outside rails to avoid leader-line crossings. Label font size adapts independently to the density of each side so adjacent text remains separated without unnecessarily shrinking the sparser column. The measured rays and arrowheads remain shortened, and the compact legends use `Tip`, `S–T`, `Start`, and `Primary` as applicable. In the lateral-start–primary view, both rays originate at the lateral insertion; the name and angle label remains connected to the root tip. The gravity reference is intentionally omitted from the two gravity-angle images, while its numerical vector and angles remain in the CSV/XLSX exports and metadata. Figure height grows for large hierarchies, but a front projection can still compress Y-directed geometry. Use the numeric 3D vector table as the measurement record.

## Trait definitions and units

Physical calibration is temporarily disabled. Output lengths, coordinates, and vector components use the source mesh coordinate unit (`mesh_unit`); area uses `mesh_unit^2` and volume uses `mesh_unit^3`. Internal unit-box normalization is only a numerical transform and is reversed before reporting. No millimetre/voxel conversion value is accepted by the current CLI or GUI, and no physical distance is used for a threshold.

| Trait | Method | Interpretation / limitation |
|---|---|---|
| Root length | Sum of consecutive centreline segment lengths | Per primary and lateral root. Accuracy depends on centreline and resolution. |
| Lateral count | Count of validated hierarchy nodes grouped by order | Deliberate `--max-laterals` caps make this a selected count, not a complete biological count. |
| Tortuosity | Centreline length / endpoint chord length | Undefined when the chord is effectively zero. |
| Diameter | Twice a smoothed local radius from assigned surface-to-centreline distances | A geometric estimate; touching roots and assignment errors can bias it. Mean, median, minimum, and maximum are reported. |
| Per-root surface area | Mesh-triangle area partitioned by majority vertex label | Falls back to a centreline-frustum estimate for point clouds or roots lacking assigned faces. |
| Whole-system surface area | Sum of original mesh triangle areas, or sum of per-root estimates when no mesh area is available | `root_system_surface_area_method` records `full_mesh_triangle_area` or `sum_per_root_surface_estimates`; biological accuracy still depends on reconstruction quality. |
| Per-root volume | Consecutive centreline frustums using the estimated radius profile | Always labelled `centerline_frustum_estimate`; per-root values are not guaranteed to sum to the whole-mesh volume. |
| Whole-system volume | Sum of absolute component-centred tetrahedral volumes for an edge-clean, consistently oriented closed mesh; otherwise the sum of per-root centreline-frustum estimates | The exact mesh value is used only when there are no boundary, non-manifold, degenerate, or orientation-inconsistent faces. `root_system_volume_method`, `root_system_volume_reliable`, and its reason distinguish the two cases. |
| Confidence and QC | Attachment, tangent continuity, length/support, and tracing evidence | Review aids, not calibrated biological probabilities. |

Disconnected components, boundary/non-manifold edges, degenerate faces, signed/absolute volume, retained fraction, and reduction reason are recorded in `metadata.json`.

## Outputs

Every run writes a self-contained output directory. Class-specific PLY files are omitted only when that class has no points.

### Measurements and topology

- `root_traits.csv`: one row per root with all measurements, units, methods, vectors, confidence, and QC.
- `traits.xlsx`: sheets for Root traits, System summary, Length, Counts by order, Angles, Tortuosity, Surface area, Volume, Diameter, Vectors, Topology, QC, and Label map.
- `csv/system_summary.csv`
- `csv/root_lengths.csv`
- `csv/lateral_counts_by_order.csv`
- `csv/root_angles.csv`
- `csv/root_tortuosity.csv`
- `csv/root_surface_area.csv`
- `csv/root_volume.csv`
- `csv/root_diameter.csv`
- `csv/angle_vectors.csv`
- `csv/root_topology.csv`
- `csv/root_qc.csv`
- `csv/root_label_map.csv`: numeric PLY labels mapped to stable root IDs, parent IDs, orders, and colours, including `-2` uncertain and `-1` unassigned rows.
- `primary_skeleton.csv` and `lateral_skeletons.csv`: centreline nodes with source-mesh `x,y,z` coordinates and explicit `mesh_unit` labels.
- `root_hierarchy.json`: editable topology and polylines.
- `root_system.rsml`: nested, hierarchy-preserving RSML with source-mesh geometry, `mesh_unit` labels, and selected properties.
- `metadata.json`: configuration, provenance, primary candidates, mesh audit, topology report, stage timings, system summary, and output inventory.

### Geometry and validation figures

- `segmented_root_structure.ply`: full-resolution labelled vertices and original triangle faces.
- `segmented_points.ply`: compatibility coloured point cloud for viewers that ignore custom PLY properties.
- `primary_points.ply`, `lateral_points.ply`, `unassigned_points.ply`, and `uncertain_points.ply`.
- `skeleton_original_overlay.ply`: side-by-side validation PLY with the complete original grey structure in source coordinates on the left and the color-coded skeleton translated to the right in the standard X–Z view (+X rightward). Mesh inputs retain the original faces and skeleton tube faces; point-cloud inputs remain face-free so viewers do not discard the original points. The X translation, minimum gap, and representation are recorded under `skeleton_original_overlay_layout` in `metadata.json`; use the skeleton CSV files for unshifted measurement coordinates.
- `overview.png`: 3D segmentation and skeleton overview.
- `tip_gravity_front_view_600dpi.png`
- `tip_start_gravity_front_view_600dpi.png`
- `tip_primary_front_view_600dpi.png`: lateral-start vector versus the local primary-root tangent; the filename is retained for compatibility.

The labelled PLY stores RGB plus scalar properties `root_id`, `root_order`, and `assignment_state`. Numeric `root_id` is `0` for primary, positive for selected laterals, `-1` for unassigned, and `-2` for uncertain. `assignment_state` is `0` unassigned, `1` assigned, and `2` uncertain. For the unsigned `root_order` PLY field, `255` denotes unassigned and `254` denotes uncertain; use `csv/root_label_map.csv` rather than treating those sentinels as biological orders. The colours are:

An **unassigned** point is either deliberately above the selected base or was not claimed by the segmented primary or by the support radius of any selected lateral. This commonly includes shoot/stem remnants, soil or pot fragments, disconnected noise, distant surface regions, or real roots that tracing did not retain. An **uncertain** point is different: it lies within assignment range but is nearly equally close to competing selected roots, so ownership is intentionally withheld. Counts for both states and the two unassigned categories are written to `metadata.json`; their fractions are included in the system summary.

After full-resolution assignment, a mesh-connectivity cleanup absorbs small, discrete order-1, uncertain, or unassigned islands that are enclosed by primary-labelled surface and lie within the measured primary-root envelope. It preserves every order-1 root's main surface component and never reclaims points excluded above the selected base. The applied policy and absorbed patch/vertex counts are recorded under `point_assignment.primary_surface_patch_cleanup` in `metadata.json`.

| Class | Colour |
|---|---|
| Primary / order 0 | Blue (`#0D3BE0`) |
| Order 1 | Magenta (`#FF00FF`) |
| Order 2 | Green (`#009E73`) |
| Order 3 | Purple (`#8C33D1`) |
| Order 4+ | Gold (`#F2A614`) |
| Uncertain | Orange (`#FA7A0D`) |
| Unassigned/original background | Grey (`#8C8C8C`) |

## Resolution, runtime, and reduction policy

Mesh loading uses the original mesh vertices rather than unconditional uniform resampling. With `--sample-points 0` (the default), all vertices are analysed unless a pilot calculation projects that full analysis will exceed the configured runtime or available-memory policy. If automatic reduction is required, the retained fraction cannot fall below `--minimum-retained-fraction` (25% by default). An explicit positive `--sample-points` value is a user-requested cap and can reduce earlier.

Even when the analysis vertices are reduced, the original full vertices/faces are preserved for geometric audit and export, and labels are mapped back to them. Always inspect these metadata fields before comparing samples:

- `source_geometry.analysis_reduced`
- `source_geometry.retained_fraction`
- `source_geometry.reduction_reason`
- `source_geometry.projected_full_analysis_seconds`
- `stage_timings_seconds`

The 30-minute policy is a projection, not a hard deadline. Topology complexity, disconnected components, and lateral-candidate count can make an individual stage slower than the pilot estimate.

## Validation status and supplied samples

The external test folder `E:\Seafile\Test files for BioInsAlgo` contains six matched STL/PLY soybean-root pairs (12 files, about 229 MiB): BaxiNo2 4-2, Kaixinlv 3-2, SN14 6-2, w5168-3 m4-2, W82 9cm water 1-2, and W82 MS4-2. The meshes span roughly 101,000–526,000 vertices and include disconnected and, in some samples, non-manifold components. Both binary STL and binary PLY ingestion paths were audited.

A final full-resolution suite ran all six PLY samples without reduction or a lateral-count cap. Per-run metadata recorded 11.91–59.83 seconds on the development machine for 101,102–526,324 vertices; the six runs completed in 156 seconds wall time including all 600-dpi figures and exports. Every run produced all 30 declared artefacts, and all six RSML files passed the official [RootSystemML XSD](https://raw.githubusercontent.com/RootSystemML/RSMLValidator/master/rsml.xsd). A separate matched BaxiNo2 STL run produced the same per-root rows as its PLY input. These are engineering/performance and contract checks, not biological ground truth: automatic counts, orders, assigned fraction, and any future physical calibration still require expert validation.

The automated suite covers synthetic end-to-end export, above-base assignment exclusion, directional angles, primary candidate ranking, higher-order junction ground truth, hierarchy invariants and corrections, editor history/replay and source immutability, RSML nesting, XLSX/CSV contracts, labelled PLY properties, 600-dpi figures, metadata, scheduling, and hardware allocation. A production-path smoke test also completed two real samples concurrently through `BatchScheduler` with four threads per sample and verified both complete export bundles. Measurement-quality validation still requires expert annotation or manual measurements for each genotype, age, scan protocol, and reconstruction workflow.

## Known limitations

- Inputs should contain reconstructed root tissue only. Soil, pot, shoot, labels, and reconstruction artefacts can be mistaken for roots.
- Root touching/merging and gaps can create ambiguous topology; use confidence/QC and the editable hierarchy.
- The automatic primary scorer assumes the coordinate system has meaningful Z orientation and gravity is `(0,0,-1)`.
- Fine laterals below the reconstruction/mesh resolution cannot be recovered reliably.
- Per-root diameter and volume are centreline/assignment estimates, not voxel-exact organ measurements.
- The batch GUI is currently Tk-based and the scientific analysis path is CPU based. The 3D editor uses the browser GPU for rendering and worker/BVH acceleration for parsing and picking; trait recomputation remains CPU based.
- STL contains no portable unit standard. Source-unit hints may be recorded as provenance, but this temporary build does not apply them to traits or thresholds.

## Research lineage and licensing

The only direct source-code lineage claimed by this tree is the MIT-licensed BioInsAlgo baseline. P3D is also MIT licensed, but was used as a design/algorithm reference rather than copied into this Python implementation. Repositories without an explicit software license and GPL-licensed projects were treated as reference-only; their code is not incorporated here. Publication open-access terms do not grant a software license.

The methods considered include Zhou et al. (2025), 4DRoot, TopoRoot/TopoRoot+, P3D, DIRT/3D, Rootine, archiDART, DynamicRoots, GiA Roots/Gia3D, RooTh/RooTrak, VRoot, RootForce, open_iA, and the 2025 3D plant-root skeleton preprint. Their concrete relevance and license boundaries are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Citation metadata for the original BioInsAlgo paper focus is also available in [CITATION.cff](CITATION.cff) and [references.bib](references.bib).

## License

SoyRootBio/BioInsAlgo is distributed under the [MIT License](LICENSE). Scientific validity and fitness for a particular experiment are not guaranteed; retain metadata and validation artefacts with every reported result.
