# Physical coordinates and measurements

New analysis bundles use millimeters for distance and coordinates, `mm^2` for
area, `mm^3` for volume, and `mm^-1` for linear density. Angles stay in degrees;
ratios, scores, counts, and unit direction vectors keep their original meaning.

STL has no mandatory unit field. The reader accepts only explicit `Unit:` or
`Units:` declarations in an 80-byte header whose binary record count agrees
with the file size. Supported spellings include mm/millimeter/millimetre,
cm/centimeter/centimetre, m/meter/metre, um/micrometer/micrometre (also µm/μm),
nm/nanometer/nanometre, and in/inch/inches, with plural SI names accepted.
Their millimeters-per-mesh-unit factors are respectively 1, 10, 1000, 0.001,
0.000001, and 25.4. Filename hints, arbitrary prose, ASCII STL, unsupported
declarations, and conflicting declarations establish no scale.

The desktop asks for a finite positive scale before opening a primary picker
or starting processing when no supported declaration exists. Its optional
batch checkbox applies to remaining unresolved samples. A declared scale always
takes precedence; an explicitly supplied conflicting scale is an error.
Cancelling prevents the entire launch. Each calibration records its source
(`stl_header`, `user`, `batch_rule`, or `cli`) and the raw declaration. Cached
header evidence is refreshed against the current file.

For noninteractive inputs without a declaration, specify the ratio explicitly:

```powershell
soyrootbio run --input root.ply --output new-results --mm-per-mesh-unit 1
```

The source arrays and internal segmentation arithmetic remain in their existing
frames. `Normalization.inverse_points` retains its raw-source meaning;
`to_mm_points`, `from_mm_points`, and `to_mm_length` perform the physical
conversion explicitly. For a normalized point `p`, raw position is
`p * source_scale + source_minimum`, and millimeter position is that result
times the resolved ratio. Length, area, and volume use the first, second, and
third powers of the ratio. No source vertex is moved, reordered, deduplicated,
or dropped because of calibration. Existing supported STL seam indexing retains
its reversible mapping.

`original_input_geometry.npz` is an explicit raw-source provenance exception.
The segmented and class PLYs, skeleton overlay, centerlines, insertion points,
CSV/XLSX tables, hierarchy JSON, RSML, nodules, figures, and edited/reopened
bundles use canonical output coordinates. Metadata names both the raw and mm
normalization and supplies physical whole-source audit equivalents. Technical
unit-box evidence retains explicit local normalized context and its conversion
factor; it is not an additional biological measurement.

Version-2 primary guidance stores mm coordinates. Version-1 guidance stores
raw mesh coordinates and converts exactly once. Endpoint and guide files with
explicit different frames are reconciled using the same calibration. Hierarchy,
surface-reference, and nodule-review readers check their recorded frame and
exact geometry/fingerprint contract. A legacy bundle with no calibration stays
in `mesh_unit` when inspected or edited; an edited mm bundle uses a physical
identity transform and never reapplies the source ratio.

The angle window is a physical 2 mm by default (`--tip-window-mm`). The older
CLI spelling and Python field remain compatibility aliases; new metadata uses
`tip_vector_window_mm`. The GUI removes the soil-line fields/modes and runtime
limit input. CPU/RAM admission and the existing automatic sampling policy remain
in effect, and GPU identification/status controls are disabled.

# Radius, diameter, volume, and completeness

Radius estimates use surface support projected onto the final fitted body.
Support beyond a terminal slab of twice the local positive sampling distance
is excluded; admissible endpoint-slab support uses transverse distance rather
than distance to the clipped endpoint. Duplicate source rows are preserved,
while a private distinct-position fallback can estimate sampling when repeated
positions exhaust the nearest-neighbor query.

Missing stations interpolate local radius quantiles in actual arc length.
An insufficient number of measured stations permits a clearly flagged global
estimate, which cannot certify a complete root. Unsupported endpoints retain
partial coverage; finite supported profile segments supply the reported frustum
estimate. Mean radius/diameter uses arc-length weighting, and the median uses
the arc-length CDF of the piecewise-linear radius profile. This avoids bias from
uneven or dense centerline station placement.

Completeness requires sufficient admissible radial support, endpoint coverage
within the sampling slab, no rejected out-of-body support, and no negative final
fit/connector evidence. Missing station count alone does not change completeness.
The table records the method, interpolation/fallback, support counts, sampling
spacing, and length/volume completeness. System summaries retain unavailable
values and report partial counts; finite fitted-body estimates cannot imply a
complete biological root. Native surface-area measurement can remain available
when a centerline volume is incomplete. No volume is forced to match mesh volume.
RSML completeness is true only for an actual true Boolean with an available
corresponding length or diameter. Unknown/unavailable quality uses false to mean
not complete; it cannot certify an unavailable topology placeholder.

The selected primary top stays separate from a shortened final fit. Editor
origin validation uses that immutable reference for every order, and protected
above-collar vertices remain unassigned. Edited materializations rerun native
higher-order/primary-contact and discrete child-patch audits. They retain
unresolved violations or missing native evidence and save previous automatic
stage reports only as historical provenance.
Read, reopen, and no-op export preserve historical child-length junction review
when final fitted bodies are shorter or parent measurements are unavailable.
Actual path/parent edits refresh length review only for affected roots and children
of changed parent paths. Validation may add a newly observed warning but cannot
discharge or reorder existing review evidence during inspection.

Existing dimensionless and local-sampling/radius policies remain intact. Optional
physical assignment floors support controlled experiments; they are not production
defaults without complete-bundle calibration evidence. The six-sample calibration
artifacts separate source/label/topology parity from intentional measurement
changes and from any assignment-floor experiment.
