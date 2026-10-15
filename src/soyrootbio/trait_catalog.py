"""Exported measurement definitions shared by tables and the illustrated guide."""
import re
import pandas as pd

DEFINITIONS = {
    "surface_area_partitioned_sum": "Sum of eligible per-root partitioned mesh areas (or frustum surface fallbacks), distinct from the strict all-corners segmented mesh area.",
    "root_count_total": "Eligible root count including the primary; excludes missing/nonpositive length and noise placeholders.",
    "lateral_root_count_total": "Eligible lateral count across every detected order above zero.",
    "maximum_root_order": "Highest order among eligible lateral roots; zero if there are none.",
    "root_system_length": "Sum of positive finite lengths across eligible primary and lateral roots; not an imputed complete-system length.",
    "primary_root_length": "Eligible measured primary-root path length.",
    "lateral_root_length_sum": "Sum of lengths across eligible lateral roots of every order.",
    "root_system_surface_area": "Native triangle area of eligible segmented roots only. Missing for point-only input; individual frustum surface estimates remain available.",
    "primary_root_surface_area": "Primary-root partitioned surface estimate when that root is eligible.",
    "lateral_root_surface_area_sum": "Sum of eligible laterals' partitioned surface estimates across all orders.",
    "root_system_volume": "Compatibility alias for the eligible centerline/frustum volume sum. The independent mesh integral has its own column.",
    "root_system_mesh_volume": "Direct native-mesh integral for eligible segmented roots; inspect mesh reliability and boundary diagnostics in architecture_summary.",
    "root_system_frustum_volume": "Sum of eligible exposed-body circular-frustum estimates.",
    "root_system_volume_reliable": "False for the compatibility frustum estimate, which is model-based rather than an enclosed-mesh measurement.",
    "root_system_mesh_volume_reliable": "Whether the eligible segmented mesh passes closed-surface and orientation checks.",
    "root_system_volume_reliability_reason": "Historical source-mesh diagnostic; current selected-patch status is mesh_volume_status in architecture_summary.",
    "primary_root_volume_estimate": "Primary exposed-body frustum volume when that root is eligible.",
    "lateral_root_volume_estimate_sum": "Sum of exposed-body frustum volumes across eligible lateral roots of all orders.",
    "assigned_vertex_fraction": "Fraction of retained native vertices with a nonnegative root label, including roots excluded from measurement by length.",
    "uncertain_vertex_fraction": "Fraction of retained native vertices with label -2; excluded from biological geometry measurements.",
    "unassigned_vertex_fraction": "Fraction of retained native vertices with label -1; excluded from biological geometry measurements.",
    "nodule_vertex_fraction": "Fraction of retained vertices assigned to separate nodule objects; excluded from root geometry measurements.",
    "nodule_excluded_from_root_totals": "True when separately classified nodule vertices are excluded from root measurements.",
    "noise_excluded_from_root_totals": "True when the noise mask has been applied to root measurements.",
    "measurement_vertex_count": "Number of non-noise native vertices before the eligible ownership filter.",
    "noise_excluded_vertex_count": "Number of native vertices removed by the noise mask; their source geometry remains archived.",
    "scope": "system: primary and all lateral orders; laterals: all orders above zero; order_N: that order only. All use finite positive-length roots.",
    "root_id": "Stable root identity; primary identifies the primary root.",
    "parent_id": "Recorded direct parent identity; empty for primary. Manual parent records are respected.",
    "root_order": "Primary is order 0; lateral generations are order 1, 2 and higher.",
    "numeric_label": "Source vertex ownership label: 0 primary, positive lateral, -1 unassigned, -2 uncertain, <=-3 nodule.",
    "length": "Arc length along the supported measured root path, including a supported parent connector. Missing if the preserved topology path is unmeasurable.",
    "chord_length": "Euclidean distance from root start to tip.",
    "tortuosity": "Measured path arc length divided by start-to-tip chord length; undefined for zero chord.",
    "mean_radius": "Arc-length-weighted mean of the piecewise-linear exposed-body radius profile; a station fallback is explicitly flagged.",
    "mean_diameter": "Twice the per-root mean radius. This is a within-root statistic, not the across-root diameter distribution.",
    "median_diameter": "Twice the median of the arc-length-weighted within-root radius distribution; fallback status is recorded.",
    "minimum_diameter": "Twice the minimum available exposed-body radius station.",
    "maximum_diameter": "Twice the maximum available exposed-body radius station.",
    "surface_area": "Per-root native triangle area partitioned by the fraction of triangle vertices owned by this root; frustum fallback without native faces. Consult surface_area_method.",
    "volume": "Legacy alias of the exposed-body centerline/frustum volume estimate, never a mesh integral.",
    "volume_frustum": "Sum of exposed-body circular frusta: pi*h*(r1^2+r1*r2+r2^2)/3. Parent connectors are excluded; incomplete coverage remains flagged.",
    "mesh_volume": "Sum of absolute signed tetrahedral integrals per native connected component, using a component vertex as origin; only triangles whose three corners are in eligible segmented roots of this scope. No caps or repair.",
    "mesh_surface_area": "Sum of native triangle areas with all three corners in eligible segmented roots of this scope; excludes unassigned, uncertain, nodule, noise and ineligible-root faces.",
    "mesh_volume_reliable": "True only for a nonempty closed, consistently oriented patch without nonmanifold edges, degenerate faces or duplicate index faces. This does not validate segmentation biology.",
    "mesh_volume_status": "Closed oriented mesh, provisional open/invalid patch, absent mesh, or no eligible triangles.",
    "mesh_volume_origin_sensitivity": "Range of mesh-volume estimates after shifting each integration origin by +/- half the patch extent along each coordinate axis; diagnostic, not a confidence interval.",
    "mesh_triangle_count": "Number of retained native triangles in this measurement scope.",
    "mesh_boundary_edges": "Edges used by exactly one retained face; nonzero values imply an open surface.",
    "mesh_nonmanifold_edges": "Edges used by more than two retained faces.",
    "mesh_inconsistent_edges": "Two-face edges with inconsistent winding.",
    "mesh_duplicate_faces": "Repeated triangles with the same three vertex indices, independent of winding.",
    "root_count": "Number of roots with finite positive length in this scope; includes primary for system/order_0.",
    "excluded_root_count": "Topology roots excluded from measurements because length is missing/nonpositive or the root is noise-excluded.",
    "length_sum": "Sum of eligible root arc lengths in this scope; partial finite lengths are retained and counted in QC.",
    "distinct_parent_count": "Number of distinct eligible recorded parents of eligible lateral children in this scope.",
    "parent_length_sum": "Sum of positive finite lengths of those distinct parents, counting each parent once; parent order need not equal child order minus one in manually edited data.",
    "branching_child_count": "Eligible non-primary children in this scope, before checking whether their parent is measurable.",
    "branching_unusable_parent_count": "Eligible children whose recorded parent is missing, ineligible or lacks a usable path; density is undefined if any denominator is missing.",
    "lateral_density": "Eligible lateral children divided by total length of their distinct eligible recorded parents. Units are roots per output length unit (roots/mm for calibrated data).",
    "mean_interbranch": "Mean consecutive insertion-arc gap within each recorded parent, pooled over gaps. No cross-parent intervals or terminal margins; zero gaps are retained.",
    "interbranch_cv": "Sample standard deviation divided by mean of pooled consecutive gaps; undefined with fewer than two gaps or zero mean.",
    "interbranch_gap_count": "Number of within-parent consecutive insertion intervals.",
    "coincident_gap_count": "Intervals <= 1e-9 output length units; these zero/near-zero gaps remain in the main spacing measurement.",
    "distinct_site_mean_interbranch": "Spacing sensitivity: mean after merging consecutive insertion positions within 1e-9 output length units on each parent.",
    "distinct_site_interbranch_cv": "Spacing sensitivity: sample SD/mean after merging coincident sites.",
    "distinct_site_gap_count": "Number of intervals in the distinct-site spacing sensitivity.",
    "insertion_arc": "Closest-segment projection of saved insertion onto the current parent polyline, expressed as arc distance from parent start. First-segment ties are deterministic.",
    "parent_eligible": "Whether the recorded parent has positive finite measured length and a usable path.",
    "insertion_source": "saved_insertion when available, otherwise explicit path_start_fallback.",
    "proximal_root_id": "Child at the lower arc coordinate of this same-parent interval.",
    "distal_root_id": "Child at the higher arc coordinate of this same-parent interval.",
    "distance": "Difference between consecutive insertion arc coordinates on the same parent.",
    "coincident": "Whether this interval is <= 1e-9 output length units.",
    "max_depth": "Maximum downward projection from the immutable primary top to eligible assigned vertices; lower bounded by zero. Uses the recorded gravity direction.",
    "max_width": "Maximum pairwise distance among eligible vertices projected perpendicular to gravity; evaluated on the 2D convex hull, independent of azimuth.",
    "depth_width": "Maximum depth divided by maximum horizontal width; undefined for zero width.",
    "horizontal_reach": "Magnitude of the start-to-tip vector projected perpendicular to gravity; curved roots may extend farther.",
    "relative_horizontal_reach": "Per-root horizontal start-to-tip displacement divided by eligible primary-root length.",
    "relative_horizontal_reach_p90": "90th percentile of per-root relative horizontal reach in this scope; linear interpolation (R type 7).",
    "hull_volume": "3D convex-hull volume enclosing eligible assigned vertices, including empty space. Zero for a degenerate envelope; missing without vertices.",
    "hull_status": "three_dimensional, degenerate_envelope, or no_eligible_vertices; no jitter or artificial geometry is added.",
    "envelope_length_density": "Sum of eligible root lengths divided by their convex-hull volume; undefined for a zero/missing envelope.",
    "solidity_mesh": "Segmented mesh integral divided by convex-hull volume. Provisional when the mesh is open/invalid; never silently clipped to [0,1].",
    "solidity_frustum": "Sum of exposed-body frustum estimates divided by convex-hull volume; estimate validity depends on radius and centerline coverage.",
    "diameter_mean": "Arithmetic mean across eligible roots' per-root mean diameters; each available root receives equal weight.",
    "diameter_median": "Median across eligible roots' per-root mean diameters (linear/R type-7 quantile).",
    "diameter_p10": "10th percentile across eligible roots' per-root mean diameters (linear/R type-7 quantile).",
    "diameter_p90": "90th percentile across eligible roots' per-root mean diameters (linear/R type-7 quantile).",
    "diameter_iqr": "75th minus 25th percentile across per-root mean diameters (linear/R type-7 quantiles).",
    "diameter_cv": "Sample SD divided by mean across available per-root mean diameters; undefined with fewer than two roots or zero mean.",
    "diameter_available_root_count": "Number of eligible roots with a finite per-root mean diameter.",
    "cumulative_fraction": "Empirical cumulative fraction of eligible roots with mean diameter <= this value; each root counts once.",
    "segmented_vertex_count": "Number of final native vertices assigned to eligible roots in this scope.",
    "angle_deg": "Legacy alias of base_parent_angle_deg.",
    "base_parent_angle_deg": "Undirected axial angle (0-90 degrees) between child start tangent and local direct-parent tangent.",
    "tip_angle_parent_deg": "Directed angle (0-180 degrees) between child tip tangent and direct-parent tangent at insertion.",
    "tip_angle_primary_deg": "Legacy alias of tip_primary_angle_deg: child BASE tangent versus primary tangent, not a tip-to-primary angle.",
    "tip_primary_angle_deg": "Directed angle (0-180 degrees) between child start tangent and primary tangent near its insertion. For higher orders this remains a primary reference, not the direct parent.",
    "tip_angle_z_deg": "Legacy alias of tip_gravity_angle_deg, relative to downward gravity.",
    "tip_gravity_angle_deg": "Directed angle (0-180 degrees) between tip-window tangent and downward gravity; smaller values point more downward.",
    "tip_start_gravity_angle_deg": "Directed angle (0-180 degrees) between root start-to-tip chord and downward gravity.",
    "exposed_body_length": "Measured arc length of the exposed root body after body_start_index, excluding the parent connector.",
    "parent_connector_length": "Arc length of the internal parent connection preceding the exposed body.",
    "parent_connector_mode": "Recorded support mode for the parent connection; retained/preserved unmeasurable geometry is explicitly identified.",
    "preserved_topology_length": "Length of unsupported retained topology connection; not treated as measured root length.",
    "topology_path_length": "Full stored topology path length, possibly including a retained unsupported connection.",
    "confidence": "Algorithmic root confidence score; not a statistical confidence interval.",
    "qc_flags": "Semicolon-separated unresolved measurement, support and topology findings.",
    "point_count": "Number of final assigned support vertices for this root.",
    "measurement_eligible": "True for positive finite length and no noise-exclusion flag; only these roots enter biological measurements.",
    "measurement_excluded": "Noise-only topology placeholder retained to preserve hierarchy; excluded from measurements.",
    "measurement_exclusion_reason": "Explicit reason a topology root is absent from the eligible measurement table.",
    "centerline_fit_status": "Status of the final centerline fit against assigned support.",
    "lateral_start_count": "Algorithmic start candidates, not the measured root count.",
    "selected_lateral_count": "Selected topology laterals, before positive-length eligibility filtering.",
    "node_id": "Ordered centerline station index from root start toward tip.",
    "centerline_region": "Exposed body, supported parent connector, preserved topology, or retained prior geometry.",
    "color_rgb": "Red, green and blue display values for this ownership label.",
}


def describe(column):
    if column in DEFINITIONS:
        return DEFINITIONS[column]
    if column.startswith("mean_") and column[5:] in DEFINITIONS:
        return "Arithmetic mean of finite values over eligible roots in this scope. " + DEFINITIONS[column[5:]]
    if column.endswith("_complete_root_count"):
        return "Number of eligible roots whose " + column[:-11] + " flag is true. False/missing flags do not count as complete."
    if column.endswith("_unavailable_root_count"):
        return "Number of topology roots in this scope with missing/nonfinite " + column.removesuffix('_unavailable_root_count') + "; includes roots excluded by the length filter."
    if column.endswith("_partial_root_count"):
        return "Number of topology roots with a finite but incomplete " + column.removesuffix('_partial_root_count').replace('_', ' ') + " measurement."
    if column.endswith("_complete"):
        return "Recorded completeness flag for " + column[:-9].replace("_", " ") + "; false means available values cover only a partial or unsupported measurement."
    if column.endswith("_unit"):
        return "Explicit unit of the associated measurement; calibrated lengths are mm, areas mm^2 and volumes mm^3."
    if column.endswith("_method"):
        return "Named calculation method/provenance for " + column[:-7].replace("_", " ") + "."
    if re.search(r"_(?:start|end)_[xyz]$", column) or re.search(r"^(?:root_start|root_tip|insertion)_[xyz]$", column):
        return "Output-coordinate component of " + column.rsplit("_", 1)[0].replace("_", " ") + "; uses the recorded coordinate unit."
    if re.search(r"_d[xyz]$", column):
        return "Signed component of " + column.rsplit("_", 1)[0].replace("_", " ") + "; gravity is a unit direction, measurement vectors are end minus start."
    if column.endswith("_requested_window"):
        return "Requested arc-length window for the local direction vector."
    if column.endswith("_arc_window"):
        return "Actual available arc-length window used for the local direction vector."
    if column.startswith("radius_"):
        return "Radius-support diagnostic: " + column[7:].replace("_", " ") + ". Uses final exposed-body support; flags/counts identify sparse stations and fallback measurements."
    if column in {"x", "y", "z"}:
        return "Centerline coordinate in the recorded coordinate unit."
    return column.replace("_", " ").capitalize() + "; see associated scope, unit, method and QC fields."


def catalog_tables(tables):
    rows = []
    for table, frame in tables.items():
        for column in frame.columns:
            rows.append(dict(table=table, field=column, definition=describe(column)))
        if table == "system_summary":
            for field in frame.trait:
                rows.append(dict(table=table, field=str(field), definition=describe(str(field))))
    return pd.DataFrame(rows)
