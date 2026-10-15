"""Read-only architecture measurements from final ownership and calibrated paths.

No geometry repair, relabelling, or synthetic mesh closure is performed here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError, distance


def eligible_mask(frame):
    length = pd.to_numeric(frame["length"], errors="coerce")
    keep = np.isfinite(length) & (length > 0)
    if "measurement_excluded" in frame:
        keep &= ~frame.measurement_excluded.eq(True)
    return keep


def _finite(values):
    a = np.asarray(values, float)
    return a[np.isfinite(a)]


def _mean(values):
    a = _finite(values)
    return float(a.mean()) if len(a) else np.nan


def _cv(values):
    a = _finite(values)
    return float(a.std(ddof=1) / a.mean()) if len(a) > 1 and a.mean() > 0 else np.nan


def _ratio(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan


def project_arc(point, path):
    """Closest segment projection, with deterministic first-segment ties."""
    path = np.asarray(path, float)
    if len(path) < 2 or not np.isfinite(path).all():
        return np.nan
    v = np.diff(path, axis=0)
    lengths = np.linalg.norm(v, axis=1)
    t = np.clip(np.einsum("ij,ij->i", point - path[:-1], v) /
                np.maximum(lengths**2, 1e-300), 0, 1)
    projected = path[:-1] + t[:, None] * v
    i = int(np.argmin(np.sum((projected - point)**2, axis=1)))
    return float(np.sum(lengths[:i]) + t[i] * lengths[i])


def mesh_measurement(points, faces):
    """Measure only supplied triangles; report open patches without closing them."""
    from .io import _mesh_audit
    if faces is None:
        return dict(mesh_surface_area=np.nan, mesh_volume=np.nan, mesh_volume_reliable=False,
                    mesh_volume_status="native_mesh_unavailable", mesh_triangle_count=0,
                    mesh_boundary_edges=0, mesh_nonmanifold_edges=0,
                    mesh_inconsistent_edges=0, mesh_duplicate_faces=0,
                    mesh_volume_origin_sensitivity=np.nan)
    faces = np.asarray(faces, dtype=np.int64)
    if not len(faces):
        result = mesh_measurement(points, None)
        result.update(mesh_surface_area=0., mesh_volume=np.nan, mesh_volume_status="no_segmented_triangles")
        return result
    ids, inverse = np.unique(faces, return_inverse=True)
    xyz, local_faces = points[ids], inverse.reshape(-1, 3)
    audit = _mesh_audit(xyz, local_faces)
    duplicates = len(local_faces) - len(np.unique(np.sort(local_faces, axis=1), axis=0))
    reliable = bool(audit["volume_reliable"]) and duplicates == 0
    # For an open surface, translation of the integration origin changes the
    # integral. Record an explicit span of estimates as a diagnostic, not CI.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e = np.vstack((local_faces[:, :2], local_faces[:, 1:], local_faces[:, [2, 0]]))
    graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(xyz), len(xyz)))
    _, component = connected_components(graph, directed=False)
    _, first = np.unique(component, return_index=True)
    fc = component[local_faces[:, 0]]
    origin = xyz[first][fc]
    a, b, c = (xyz[local_faces[:, i]] for i in range(3))
    normals = np.cross(b-a, c-a)
    estimates = []
    for shift in np.vstack((np.zeros(3), np.diag(np.ptp(xyz, axis=0)/2), -np.diag(np.ptp(xyz, axis=0)/2))):
        signed = np.einsum("ij,ij->i", a-origin-shift, normals)/6
        estimates.append(float(np.abs(np.bincount(fc, weights=signed)).sum()))
    return dict(mesh_surface_area=audit["surface_area_source_units2"],
                mesh_volume=audit["absolute_volume_source_units3"],
                mesh_volume_reliable=reliable,
                mesh_volume_status="closed_oriented_mesh" if reliable else "provisional_open_or_invalid_patch",
                mesh_triangle_count=len(faces), mesh_boundary_edges=audit["boundary_edge_count"],
                mesh_nonmanifold_edges=audit["nonmanifold_edge_count"],
                mesh_inconsistent_edges=audit["orientation_inconsistent_edge_count"],
                mesh_duplicate_faces=duplicates,
                mesh_volume_origin_sensitivity=float(np.ptp(estimates)))


def _extent(points, top, gravity):
    result = dict(max_depth=np.nan, max_width=np.nan, depth_width=np.nan,
                  hull_volume=np.nan, hull_status="no_eligible_vertices")
    if not len(points):
        return result
    result["max_depth"] = max(0., float(np.max((points-top) @ gravity)))
    helper = np.eye(3)[int(np.argmin(np.abs(gravity)))]
    u = np.cross(gravity, helper); u /= np.linalg.norm(u)
    v = np.cross(gravity, u)
    xy = np.unique(np.column_stack((points @ u, points @ v)), axis=0)
    try:
        xy = xy[ConvexHull(xy).vertices] if len(xy) >= 3 else xy
    except QhullError:
        # Collinear projection: the end points are among the original points.
        _, _, axes = np.linalg.svd(xy-xy.mean(axis=0), full_matrices=False)
        t = xy @ axes[0]
        xy = xy[[int(t.argmin()), int(t.argmax())]]
    width = 0.
    for start in range(0, len(xy), 256):
        width = max(width, float(distance.cdist(xy[start:start+256], xy).max()))
    result["max_width"] = width
    result["depth_width"] = _ratio(result["max_depth"], width)
    try:
        result.update(hull_volume=float(ConvexHull(points).volume), hull_status="three_dimensional")
    except QhullError:
        result.update(hull_volume=0., hull_status="degenerate_envelope")
    return result


def add_architecture_traits(frame, primary, laterals, points, labels, faces, top, gravity, unit):
    """Add per-root measurements and nonoverlapping tables to ``frame.attrs``.

    Coordinates, insertion points and top are already in the output unit.
    Numeric labels are 0 for primary, then 1..N in lateral list order.
    """
    frame["measurement_eligible"] = eligible_mask(frame)
    frame["measurement_exclusion_reason"] = np.where(frame.measurement_eligible, "", "missing_or_nonpositive_length")
    if "measurement_excluded" in frame:
        frame.loc[frame.measurement_excluded.eq(True), "measurement_exclusion_reason"] = "excluded_noise"
    paths = {"primary": np.asarray(primary)}
    paths.update({r.root_id: np.asarray(r.points) for r in laterals})
    label_for = {"primary": 0, **{r.root_id: i for i, r in enumerate(laterals, 1)}}
    row_for = frame.set_index("root_id")
    eligible = set(frame.loc[frame.measurement_eligible, "root_id"])
    primary_length = float(row_for.loc["primary", "length"]) if "primary" in eligible else np.nan
    branches = []
    for r in laterals:
        if r.root_id not in eligible:
            continue
        valid_parent = r.parent_id in eligible and r.parent_id in paths
        insertion = np.asarray(r.insertion_point if r.insertion_point is not None else r.points[0])
        arc = project_arc(insertion, paths[r.parent_id]) if valid_parent else np.nan
        valid_parent = valid_parent and np.isfinite(arc)
        branches.append(dict(root_id=r.root_id, parent_id=r.parent_id, root_order=r.order,
                             insertion_arc=arc, parent_eligible=valid_parent,
                             insertion_source="saved_insertion" if r.insertion_point is not None else "path_start_fallback",
                             length_unit=unit))
    branch_frame = pd.DataFrame(branches, columns=["root_id", "parent_id", "root_order", "insertion_arc", "parent_eligible", "insertion_source", "length_unit"])
    frame["insertion_arc"] = frame.root_id.map({b["root_id"]: b["insertion_arc"] for b in branches})
    frame["horizontal_reach"] = np.nan
    frame["relative_horizontal_reach"] = np.nan
    for key, value in mesh_measurement(points, None).items():
        if isinstance(value, str):
            frame[key] = "ineligible_root"
        elif isinstance(value, bool):
            frame[key] = False
        else:
            frame[key] = np.nan
    mesh_cache = {}
    def selected_mesh(ids):
        key = tuple(sorted(label_for[r] for r in ids))
        if key not in mesh_cache:
            selected = None if faces is None else faces[np.all(np.isin(labels[faces], key), axis=1)]
            mesh_cache[key] = mesh_measurement(points, selected)
        return mesh_cache[key]
    for idx, row in frame.iterrows():
        if row.root_id not in eligible:
            continue
        chord = paths[row.root_id][-1] - paths[row.root_id][0]
        reach = float(np.linalg.norm(chord - np.dot(chord, gravity)*gravity))
        frame.loc[idx, "horizontal_reach"] = reach
        frame.loc[idx, "relative_horizontal_reach"] = _ratio(reach, primary_length)
        for key, value in selected_mesh([row.root_id]).items():
            frame.loc[idx, key] = value
    frame["volume_frustum"] = frame.volume.where(frame.measurement_eligible)
    groups = [("system", frame), ("laterals", frame[frame.root_order > 0])]
    groups += [(f"order_{order}", frame[frame.root_order == order]) for order in sorted(frame.root_order.unique())]
    summaries, gaps, distributions = [], [], []
    form_columns = ["length", "chord_length", "tortuosity", "median_diameter", "minimum_diameter", "maximum_diameter",
                    "surface_area", "volume_frustum", "base_parent_angle_deg", "tip_angle_parent_deg",
                    "tip_primary_angle_deg", "tip_gravity_angle_deg", "tip_start_gravity_angle_deg"]
    for scope, candidates in groups:
        selected = candidates[candidates.measurement_eligible]
        ids = set(selected.root_id)
        chosen = branch_frame[branch_frame.root_id.isin(ids)]
        valid = chosen[chosen.parent_eligible.eq(True)]
        parents = set(valid.parent_id)
        parent_sum = float(sum(row_for.loc[p, "length"] for p in parents))
        raw_gaps, distinct_gaps = [], []
        for parent_id, group in valid.groupby("parent_id", sort=True):
            group = group.sort_values(["insertion_arc", "root_id"])
            pos = group.insertion_arc.to_numpy(float)
            root_ids = group.root_id.to_list()
            for i, gap in enumerate(np.diff(pos)):
                raw_gaps.append(float(gap))
                gaps.append(dict(scope=scope, parent_id=parent_id, proximal_root_id=root_ids[i],
                                 distal_root_id=root_ids[i+1], distance=float(gap),
                                 coincident=bool(gap <= 1e-9), length_unit=unit))
            distinct = pos[np.r_[True, np.diff(pos) > 1e-9]]
            distinct_gaps.extend(np.diff(distinct).tolist())
        vertices = points[np.isin(labels, [label_for[r] for r in ids])]
        extent = _extent(vertices, top, gravity)
        mesh = selected_mesh(ids)
        volume = float(selected.volume_frustum.sum(min_count=1)) if len(selected) else (np.nan if len(candidates) else 0.)
        length = float(selected.length.sum()) if len(selected) else (np.nan if len(candidates) else 0.)
        row = dict(scope=scope, root_count=len(selected), excluded_root_count=len(candidates)-len(selected),
                   length_sum=length, length_unit=unit, area_unit=f"{unit}^2", volume_unit=f"{unit}^3",
                   distinct_parent_count=len(parents), parent_length_sum=parent_sum,
                   branching_child_count=len(chosen), branching_unusable_parent_count=len(chosen)-len(valid),
                   lateral_density=_ratio(len(chosen), parent_sum) if len(valid)==len(chosen) else np.nan,
                   lateral_density_unit=f"roots/{unit}", mean_interbranch=_mean(raw_gaps),
                   interbranch_cv=_cv(raw_gaps), interbranch_gap_count=len(raw_gaps),
                   coincident_gap_count=int(np.sum(np.asarray(raw_gaps) <= 1e-9)),
                   distinct_site_mean_interbranch=_mean(distinct_gaps), distinct_site_interbranch_cv=_cv(distinct_gaps),
                   distinct_site_gap_count=len(distinct_gaps),
                   relative_horizontal_reach_p90=float(np.quantile(_finite(selected.relative_horizontal_reach), .9)) if len(_finite(selected.relative_horizontal_reach)) else np.nan,
                   volume_frustum=volume,
                   volume_frustum_complete=bool(len(selected) and selected.volume_estimate_complete.eq(True).all()) if "volume_estimate_complete" in selected else False,
                   segmented_vertex_count=len(vertices), **extent, **mesh)
        row.update(envelope_length_density=_ratio(length, extent["hull_volume"]),
                   envelope_length_density_unit=f"{unit}/{unit}^3",
                   solidity_mesh=_ratio(mesh["mesh_volume"], extent["hull_volume"]),
                   solidity_frustum=_ratio(volume, extent["hull_volume"]))
        row["surface_area_partitioned_sum"] = float(selected.surface_area.sum(min_count=1)) if len(selected) else (np.nan if len(candidates) else 0.)
        for column in form_columns:
            if column in selected:
                row[f"mean_{column}"] = _mean(selected[column])
        diam = _finite(selected.mean_diameter)
        row.update(diameter_mean=_mean(diam), diameter_cv=_cv(diam), diameter_available_root_count=len(diam))
        for name, q in (("median", .5), ("p10", .1), ("p90", .9)):
            row[f"diameter_{name}"] = float(np.quantile(diam, q, method="linear")) if len(diam) else np.nan
        row["diameter_iqr"] = float(np.diff(np.quantile(diam, [.25, .75], method="linear"))[0]) if len(diam) else np.nan
        for flag in ("length_estimate_complete", "volume_estimate_complete", "radius_profile_complete", "centerline_measurement_support_complete"):
            row[f"{flag}_root_count"] = int(selected[flag].eq(True).sum()) if flag in selected else 0
        for column in ("length", "mean_diameter", "volume", "surface_area"):
            row[f"{column}_unavailable_root_count"] = int((~np.isfinite(pd.to_numeric(candidates[column], errors="coerce"))).sum())
        for value in np.unique(diam):
            distributions.append(dict(scope=scope, mean_diameter=float(value), root_count=int(np.sum(diam==value)),
                                      cumulative_fraction=float(np.mean(diam<=value)), length_unit=unit))
        summaries.append(row)
    frame.attrs["architecture_summary"] = summaries
    frame.attrs["branch_positions"] = branch_frame.to_dict("records")
    frame.attrs["branch_spacing"] = gaps
    frame.attrs["diameter_distribution"] = distributions
    frame.attrs["measurement_policy"] = dict(version="all-orders-v1", eligibility="finite positive length; noise excluded",
        geometry="final assigned vertices of eligible roots only; no nodules, unassigned or uncertain points",
        mesh_faces="all three corners in the selected eligible roots; no artificial caps",
        immutable_primary_top=np.asarray(top).tolist(), coordinate_unit=unit,
        primary_top_source="explicit_immutable_reference", quantiles="linear (R type 7)",
        spacing="within recorded eligible parents; coincident sites retained; distinct-site sensitivity provided",
        solidity="mesh and frustum volumes divided by hull; validity follows numerator and coverage")
