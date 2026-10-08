# Noise reduction before primary selection

The **Noise reduction** checkbox is on by default in the batch GUI and legacy
single-sample launcher. It applies when opening the endpoint/soil-line picker
and when processing a sample with recorded or automatic primary guidance.
The CLI accepts `--noise-reduction` (default) and `--no-noise-reduction`.
The corresponding `PipelineConfig` field is `noise_reduction`.

## Removal policy

The stage runs on full finite source geometry after conservative STL indexing,
before the analysis cap, normalization, spacing estimation and primary detection.
It never builds proximity edges, welds new seams, cuts connected root branches,
or changes a recorded primary top. Only whole native connected components can
be excluded. The component with the greatest triangle area is the reference
structure; ties use native vertex order.

Automatic exclusion requires all of the following:

- The reference component has at least 90% of the total native triangle area.
- The fragment has at most 0.1% of the reference area.
- Its bounding-box diagonal is at most 2% of the reference diagonal.
- Its longest PCA extent is at most four times the second-longest extent.
- Native faces have closed manifold edges and vertex fans, consistent winding,
  no degenerate or duplicate faces, and no unresolved STL vertices or exact
  positions shared with a different component.
- Six deterministic ray directions place it outside the dominant surface:
  every vertex needs at least five outside votes, at least 99% need all six,
  and no vertex may have all six inside votes. Exact triangle intersection
  and near-zero surface-distance checks additionally preserve contacting pieces.

The `native-small-exterior-components-v2` policy adds this containment safeguard.
Separate internal shells/cavities are retained as review geometry. Ray queries
use unit-box coordinates for numerical stability; they are conservative geometric
evidence, not biological ground truth. Ambiguous containment is retained.

Unused isolated vertices can also be excluded under the dominant-structure
policy. Large, elongated or ambiguous components remain and receive an explicit
review reason. Meshes without a dominant structure retain all components.
Point-only inputs retain all points with `unresolved_no_native_connectivity`;
this stage does not infer biological disconnection from point density.

These thresholds define a conservative geometric filter, not biological ground
truth. A short detached real root can resemble noise. Switch the filter off to
compare the same input and recorded guidance. No root/descendant is removed for
being longer than its parent, and retained fragments are not reported as proven
noise or as resolved biological violations.

Changing the filter can change automatic branch ownership beyond the excluded
vertices. A previously saved scoped surface reference may then fail its existing
bounded-difference check. That failure must remain explicit: the filter does not
relax the reference tolerance or silently ignore the override. For an automatic
ON/OFF comparison, use the same primary guidance and omit the scoped surface
reference in both runs; report that choice separately from comparisons with a
historical bundle that applied the reference.

## Geometry and measurement contracts

Analysis points are an ordered subset of retained native vertices. Full source
coordinates, vertex order, duplicates, triangles and original decoded arrays
are preserved. The pipeline carries the full noise mask as a hard exclusion
through ownership cleanup, surface references, nodule analysis and final fitting.
Excluded vertices remain `-1` (unassigned) in full-resolution exports, and cannot
contribute support to measured roots. Above-collar exclusion remains a separate
mask, and the selected primary top remains immutable.

Whole-mesh area and reliable closed-mesh volume totals omit excluded components;
the original unfiltered geometry audit remains in `source_geometry`. Filtered
area uses the method `noise_filtered_mesh_triangle_area`. Root-specific area and
frustum measurements continue to use their final assigned support and existing
measurement/QC rules. The canonical source mesh still contains excluded dots.
For MeshLab, CloudCompare and other general PLY viewers, open
`presentation_root_structure.ply`: excluded vertices and their whole faces are
physically omitted, with exact source correspondence in
`presentation_vertex_mapping.npz`. Scientific labels, source arrays and trait
tables remain unchanged. SoyRootEditor hides these faces by default while
keeping source vertex IDs for picking; roots supported entirely by a reviewed
presentation mask have their centerlines hidden too.

A geometry-bound `presentation_noise_masks.npz` may also include individually
reviewed external fragments retained by the conservative automatic filter. Every
hidden vertex must be unassigned and excluded from measurement support, mesh
area/volume and summary-fraction denominators. Presentation export rejects a
display-only mask or an assigned hidden vertex; scientific export rejects traits
that were not recomputed with the complete mask. Both editors reject reassignment
of hidden vertices and preserve the mask through edited exports. Internal
surfaces/cavities must not be called external noise merely because they form
separate native components. Raw geometry remains in `segmented_root_structure.ply`.

To upgrade an older presentation-only bundle, run
`python scripts/exclude_hidden_noise.py SOURCE NEW_OUTPUT` with `PYTHONPATH=src`.
The command preserves the source bundle and biological ownership outside the mask,
removes noise-only leaf roots, refits bodies whose fitting component changed, and
regenerates CSV/XLSX/RSML and PLY outputs. A noise-only parent needed by surviving
descendants is retained as an unresolved topology placeholder, with all geometric
traits unavailable and excluded from root counts and totals. The selected primary
top stays fixed. Contact and disconnected-child-patch audits remain explicit.
Earlier presentation-only bundles with assigned hidden vertices are rejected on
editor load until rebuilt; old scientific files are not silently treated as valid.

Enabled runs export:

- `noise_reduction.json`: versioned policy, limits, counts and per-component reasons.
- `noise_reduction_masks.npz`: `excluded_full_vertices` and exact `analysis_to_full`
  mapping (a `-1` mapping sentinel denotes unavailable correspondence for a
  caller-supplied cloud; correspondence is never guessed).
- `metadata.json`: saved switch value and noise reduction report.
- Existing full-resolution mesh and `original_input_geometry.npz`, unchanged in
  geometry by this stage. Reduced analysis maps are saved with the input mapping.

The low-level geometry reader still defaults to filtering off so inspecting or
reloading a saved mesh preserves its full contents. GUI/pipeline entry points
explicitly pass the switch. A preloaded cloud may be filtered without mutating
the caller's source; changing an already-filtered cloud to off requires reloading.

Regression coverage includes connected-body protection, open/nonmanifold and
coincident components, no dominant structure, thin detached fragments, scale and
vertex-order invariance, source/mapping conservation, OFF bypass, GUI/CLI switch
propagation, pre-primary timing and end-to-end measurement/export exclusion.
