"""Versioned, geometry-bound support shared by fitting and measurement.

Vertex IDs refer to the unmodified full native vertex array.  Evidence is not a
permission to join disconnected bodies, and a coordinate conversion is allowed
only after the old evidence has been validated.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

POLICY = "native-measurement-support-v1"


def assess_terminal_direction(path, *, spacing):
    """Require a real exposed terminal segment before certifying its direction."""
    path = np.asarray(path, float)
    spacing = float(spacing)
    tolerance = spacing * 1e-7 if np.isfinite(spacing) and spacing > 0 else None
    result = {"terminal_direction_support_version": 1,
              "terminal_direction_supported": False,
              "terminal_segment_length": None,
              "terminal_direction_tolerance": tolerance,
              "terminal_direction_reason": "terminal_direction_unavailable"}
    if (tolerance is None or path.ndim != 2 or path.shape[1:] != (3,)
            or len(path) < 2):
        return result
    if not np.isfinite(path[-2:]).all():
        result["terminal_direction_reason"] = "nonfinite_terminal_segment"
        return result
    length = float(np.linalg.norm(path[-1] - path[-2]))
    if not np.isfinite(length):
        result["terminal_direction_reason"] = "nonfinite_terminal_segment"
        return result
    supported = length > tolerance
    result.update(terminal_direction_supported=bool(supported), terminal_segment_length=length,
                  terminal_direction_reason=("positive_exposed_terminal_segment" if supported else
                                             "degenerate_terminal_segment"))
    return result


def assess_origin_coverage(path, origin, *, reference_path=None, spacing,
                           attachment_supported=True):
    """Check axial basal coverage against an independent biological origin.

    A fitted cross-section center may move radially from a surface selection.
    That is not lost axial root length. Tolerance follows sampling spacing,
    never the radius or a shortened fitted endpoint.
    """
    path = np.asarray(path, float)
    reference = np.asarray([] if reference_path is None else reference_path, float)
    tolerance = 1.5 * float(spacing)
    result = {"origin_coverage_complete": False, "axial_loss": None,
              "axial_displacement": None, "radial_offset": None,
              "tolerance": tolerance, "reason": "independent_origin_reference_unavailable",
              "origin_reference": [], "tipward_direction": []}
    if (not np.isfinite(tolerance) or tolerance <= 0 or path.ndim != 2
            or path.shape[1:] != (3,) or len(path) == 0 or not np.isfinite(path).all()
            or reference.ndim != 2 or reference.shape[1:] != (3,)
            or len(reference) < 2 or not np.isfinite(reference).all() or origin is None):
        return result
    origin = np.asarray(origin, float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        return result
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(reference, axis=0), axis=1))]
    index = min(max(1, int(np.searchsorted(arc, 4 * spacing))), len(reference) - 1)
    direction = reference[index] - reference[0]
    if np.linalg.norm(direction) <= 1e-12:
        return result
    direction /= np.linalg.norm(direction)
    delta = path[0] - origin
    axial = float(delta @ direction)
    radial = float(np.linalg.norm(delta - axial * direction))
    covered = abs(axial) <= tolerance + max(1e-12, spacing * 1e-7)
    result.update(origin_coverage_complete=bool(covered and attachment_supported),
                  axial_loss=max(0., axial), axial_displacement=axial,
                  radial_offset=radial, origin_reference=origin.tolist(),
                  tipward_direction=direction.tolist(),
                  reason=("supported_origin_axial_coverage" if covered and attachment_supported else
                          "attachment_not_supported" if not attachment_supported else
                          "fitted_body_omits_origin" if axial > 0 else "fitted_path_extends_before_origin"))
    return result


def array_fingerprint(value, *, integer=False):
    a = np.asarray(value, dtype=np.int64 if integer else np.float64)
    if not integer:
        a = np.round(a, 10)
        a[a == 0] = 0.  # affine round trips may change the sign bit of zero
    digest = hashlib.sha256()
    digest.update(str(a.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def make_evidence_context(points, triangles=None):
    """Compute once per immutable geometry generation, never across edits."""
    result = {
        "point_count": len(points),
        "geometry_fingerprint": array_fingerprint(points),
        "topology_fingerprint": array_fingerprint(
            np.empty((0, 3), dtype=np.int64) if triangles is None else triangles,
            integer=True,
        ),
        "native_mesh": triangles is not None and len(triangles) > 0,
    }
    if result["native_mesh"]:
        faces = np.asarray(triangles, int)
        edges, incidence = np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0, return_counts=True)
        boundary = np.zeros(len(points), bool)
        boundary[np.unique(edges[incidence != 2])] = True
        result["_native_boundary_vertices"] = boundary
        result["_native_full_graph"] = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                                  shape=(len(points), len(points))).tocsr()
        result["_native_full_graph_reverse"] = result["_native_full_graph"].T.tocsr()
        lengths = np.linalg.norm(np.asarray(points)[edges[:, 0]] - np.asarray(points)[edges[:, 1]], axis=1)
        positive = lengths[lengths > 0]
        if len(positive):
            edges = edges[lengths <= 6 * np.median(positive)]
        result["_native_graph"] = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                             shape=(len(points), len(points))).tocsr()
    return result


def _projection(points, path):
    """Exact bounded-memory segment projection; deterministic first tie."""
    points, path = np.asarray(points, float), np.asarray(path, float)
    if len(path) < 2:
        return np.zeros(len(points)), np.linalg.norm(points - path[0], axis=1), np.zeros(len(points), int)
    vector = np.diff(path, axis=0)
    lengths = np.linalg.norm(vector, axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    out_arc, out_distance, out_station = [], [], []
    # Limit temporary query x segment arrays, including long recovered roots.
    chunk_size = max(1, min(2048, 200000 // max(1, len(vector))))
    for start in range(0, len(points), chunk_size):
        q = points[start:start + chunk_size]
        delta = q[:, None, :] - path[:-1]
        t = np.clip(np.einsum("qsi,si->qs", delta, vector) /
                    np.maximum(lengths**2, 1e-300), 0., 1.)
        residual = delta - t[..., None] * vector
        squared = np.einsum("qsi,qsi->qs", residual, residual)
        best = np.argmin(squared, axis=1)
        rows = np.arange(len(q))
        along = arc[best] + t[rows, best] * lengths[best]
        station = best + (t[rows, best] > .5)
        out_arc.append(along)
        out_distance.append(np.sqrt(squared[rows, best]))
        out_station.append(station)
    return tuple(np.concatenate(a) if a else np.empty(0, int if i == 2 else float)
                 for i, a in enumerate((out_arc, out_distance, out_station)))


def build_measurement_evidence(points, labels, label, path, *, triangles=None,
                               selected_vertex_indices=None, spacing,
                               body_start_index=0, identity_resolved=True,
                               attachment_supported=True, tip_supported=False,
                               origin_coverage_complete=False, origin_coverage_evidence=None,
                               source="final_fit", context=None):
    points, labels, path = np.asarray(points, float), np.asarray(labels), np.asarray(path, float)
    context = make_evidence_context(points, triangles) if context is None else context
    owned = np.flatnonzero(labels == label)
    ids = owned if selected_vertex_indices is None else np.unique(np.asarray(selected_vertex_indices, int))
    if len(ids) and (ids[0] < 0 or ids[-1] >= len(points) or np.any(labels[ids] != label)):
        raise ValueError("measurement evidence support must be currently owned native vertices")
    start = min(max(int(body_start_index), 0), max(len(path) - 1, 0))
    body = path[start:]
    if not len(body):
        raise ValueError("measurement evidence requires one finite body location")
    support_arc, distance, station = _projection(points[ids], body)
    station = np.asarray(station, int) + start
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    station_ok = np.zeros(len(path), bool)
    station_counts = np.zeros(len(path), int)
    if len(ids) and len(body) > 1:
        # A station is supported only by its local longitudinal neighborhood.
        # This does not claim that its angular sampling is sufficient for a
        # radius; the estimator checks that independently.
        for i in range(start, len(path)):
            width = max(2. * spacing,
                        (arc[min(i + 1, len(path) - 1)] - arc[max(start, i - 1)]) * .6)
            local = np.abs(support_arc - (arc[i] - arc[start])) <= width
            station_counts[i] = int(np.count_nonzero(local))
            # A native terminal can be one cap vertex. Length follows real
            # edges; radius needs several transverse samples independently.
            station_ok[i] = station_counts[i] >= (1 if context["native_mesh"] else 3)
    interval_ok = station_ok[:-1] & station_ok[1:]
    if start:
        interval_ok[:start] = False
    # Native edges must cover the measured interval. Endpoint samples alone
    # never certify a long empty middle, while coarse but real mesh edges do.
    spans = []
    graph = context.get("_native_full_graph" if source == "legacy_component_measurement" else "_native_graph")
    if graph is not None and len(ids):
        local = graph[ids][:, ids].tocoo()
        edge_length = np.linalg.norm(points[ids[local.row]] - points[ids[local.col]], axis=1)
        lo = np.minimum(support_arc[local.row], support_arc[local.col])
        hi = np.maximum(support_arc[local.row], support_arc[local.col])
        compatible = (hi - lo) <= 3 * edge_length + 2 * spacing
        spans = sorted(zip(lo[compatible], hi[compatible]))
    elif len(support_arc):
        positions = np.unique(support_arc)
        # Ring samples can have a coarser axial spacing than their circumferential
        # nearest-neighbor spacing. Estimate it only from several distinct axial
        # stations; two isolated end rings cannot justify the empty middle.
        groups = np.split(positions, np.flatnonzero(np.diff(positions) > .2 * spacing) + 1)
        axial = np.asarray([np.median(g) for g in groups if len(g)])
        axial_step = float(np.median(np.diff(axial))) if len(axial) >= 4 else 0.
        gap_limit = max(2 * spacing, 1.5 * axial_step)
        spans = [(a, b) for a, b in zip(positions[:-1], positions[1:]) if b - a <= gap_limit]
    merged = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1] + 1e-10:
            merged[-1][1] = max(merged[-1][1], float(hi))
        else:
            merged.append([float(lo), float(hi)])
    for i in np.flatnonzero(interval_ok):
        left, right = arc[i] - arc[start], arc[i + 1] - arc[start]
        tolerance = max(1e-12, spacing * 1e-7)
        interval_ok[i] = (right - left > tolerance and any(
            hi - lo > tolerance and lo <= left + tolerance and hi >= right - tolerance
            for lo, hi in merged))
    terminal_direction = assess_terminal_direction(body, spacing=spacing)
    terminal = bool(terminal_direction["terminal_direction_supported"]
                    and len(station_ok) and station_ok[-1])
    return {
        "policy": POLICY, "interval_support_version": 2, "status": "current", "source": source,
        **terminal_direction,
        "coordinate_unit": "analysis_normalized", **{k: v for k, v in context.items() if not k.startswith("_")},
        "ownership_fingerprint": array_fingerprint(owned, integer=True),
        "support_fingerprint": array_fingerprint(ids, integer=True),
        "path_fingerprint": array_fingerprint(path),
        "body_start_index": start, "spacing": float(spacing),
        "native_vertex_indices": ids.tolist(),
        "support_station_indices": station.tolist(),
        "support_arc_positions": support_arc.tolist(),
        "supported_station_mask": station_ok.tolist(),
        "station_support_counts": station_counts.tolist(),
        "supported_interval_mask": interval_ok.tolist(),
        "identity_resolved": bool(identity_resolved and context["native_mesh"]),
        "attachment_supported": bool(attachment_supported),
        "origin_coverage_complete": bool(origin_coverage_complete),
        "origin_coverage_evidence": copy.deepcopy(origin_coverage_evidence),
        "biological_tip_verified": bool(tip_supported and identity_resolved and terminal and context["native_mesh"]),
        "component_measurement_only": not bool(identity_resolved and context["native_mesh"]),
    }


def validate_measurement_evidence(evidence, points, labels, label, path, *,
                                  triangles=None, body_start_index=0, context=None):
    if not isinstance(evidence, dict) or evidence.get("policy") != POLICY:
        return False, "measurement_evidence_missing_or_unknown"
    if evidence.get("status") != "current":
        return False, str(evidence.get("invalid_reason", "measurement_evidence_invalidated"))
    context = make_evidence_context(points, triangles) if context is None else context
    for field in ("point_count", "geometry_fingerprint", "topology_fingerprint"):
        if evidence.get(field) != context.get(field):
            return False, "stale_measurement_" + field
    owned = np.flatnonzero(np.asarray(labels) == label)
    if evidence.get("ownership_fingerprint") != array_fingerprint(owned, integer=True):
        return False, "stale_measurement_ownership"
    if evidence.get("path_fingerprint") != array_fingerprint(path):
        return False, "stale_measurement_path"
    if evidence.get("body_start_index") != int(body_start_index):
        return False, "stale_measurement_body_start"
    ids = np.asarray(evidence.get("native_vertex_indices", []), int)
    if len(ids) and (ids.min() < 0 or ids.max() >= len(points) or np.any(np.asarray(labels)[ids] != label)):
        return False, "stale_measurement_support"
    if evidence.get("support_fingerprint") != array_fingerprint(ids, integer=True):
        return False, "stale_measurement_support_selection"
    if len(evidence.get("support_station_indices", [])) != len(ids):
        return False, "invalid_measurement_station_support"
    stations = np.asarray(evidence.get("support_station_indices", []), int)
    if len(stations) and (stations.min() < int(body_start_index) or stations.max() >= len(path)):
        return False, "invalid_measurement_station_support"
    if len(evidence.get("supported_station_mask", [])) != len(path):
        return False, "invalid_measurement_station_support"
    if len(evidence.get("supported_interval_mask", [])) != max(0, len(path) - 1):
        return False, "invalid_measurement_interval_support"
    return True, "current"


def resolve_measurement_support(evidence, points, labels, label, path, **kwargs):
    valid, reason = validate_measurement_evidence(evidence, points, labels, label, path, **kwargs)
    return (np.asarray(evidence["native_vertex_indices"], int) if valid else None), reason


def upgrade_measurement_evidence(evidence, points, labels, label, path, *, triangles=None,
                                 body_start_index=0, context=None):
    """Reassess old interval/terminal policies after validating frozen support.

    This migrates the measurement interpretation of exactly the same accepted
    surface. It never refreshes a stale selection or edits geometry/ownership.
    """
    valid, reason = validate_measurement_evidence(evidence, points, labels, label, path,
        triangles=triangles, body_start_index=body_start_index, context=context)
    if not valid:
        return None, reason
    interval_current = evidence.get("interval_support_version", 1) >= 2
    terminal_current = evidence.get("terminal_direction_support_version", 0) >= 1
    if interval_current and terminal_current:
        return evidence, "current"
    upgraded = build_measurement_evidence(points, labels, label, path,
        triangles=triangles, selected_vertex_indices=evidence["native_vertex_indices"],
        spacing=evidence["spacing"], body_start_index=body_start_index,
        identity_resolved=evidence.get("identity_resolved", False),
        attachment_supported=evidence.get("attachment_supported", False),
        tip_supported=evidence.get("biological_tip_verified", False),
        origin_coverage_complete=evidence.get("origin_coverage_complete", False),
        origin_coverage_evidence=evidence.get("origin_coverage_evidence"),
        source=evidence.get("source", "final_fit"), context=context)
    upgraded["coordinate_unit"] = evidence.get("coordinate_unit", "analysis_normalized")
    upgraded["previous_interval_support_version"] = evidence.get("interval_support_version", 1)
    upgraded["previous_terminal_direction_support_version"] = evidence.get("terminal_direction_support_version", 0)
    return upgraded, ("evidence_interval_policy_upgraded" if not interval_current else
                      "evidence_terminal_direction_policy_upgraded")


def infer_component_indices(points, labels, label, triangles, assessment, *, spacing=None, context=None):
    """Conservative legacy import, never a certificate of resolved identity.

    Component numbering follows final fitting's bounded native edge graph. A
    missing selection for fragmented ownership is explicitly unavailable.
    """
    from .centerline import _support_edges
    points, labels = np.asarray(points, float), np.asarray(labels)
    ids = np.flatnonzero(labels == label)
    if not len(ids):
        return ids, "no_owned_support"
    if triangles is None or not len(triangles):
        return None, "unresolved_no_native_component_evidence"
    spacing = float(spacing or (assessment or {}).get("fit_spacing", 1e-6))
    native_graph = (context or {}).get("_native_graph")
    if native_graph is not None:
        graph = native_graph[ids][:, ids]
    else:
        edges = _support_edges(points, triangles, spacing)
        member = np.zeros(len(points), bool)
        member[ids] = True
        edges = edges[member[edges].all(axis=1)]
        local = np.searchsorted(ids, edges)
        graph = coo_matrix((np.ones(len(local)), (local[:, 0], local[:, 1])), shape=(len(ids), len(ids)))
    count, components = connected_components(graph, directed=False)
    recorded_count = (assessment or {}).get("component_count")
    if recorded_count is not None and int(recorded_count) != int(count):
        return None, "legacy_component_topology_changed"
    selection = (assessment or {}).get("selected_component")
    if selection is None:
        # Older records without a fitted component index may still establish
        # one actual native component. A global edge-length heuristic can split
        # a coarse root merely because another component is finely sampled.
        full_graph = (context or {}).get("_native_full_graph")
        if recorded_count is None and full_graph is not None:
            full_count, _ = connected_components(full_graph[ids][:, ids], directed=False)
            if full_count == 1:
                return ids, "legacy_single_native_component"
        if count == 1:
            return ids, "legacy_single_native_component"
        return None, "legacy_fragmented_component_selection_missing"
    if not isinstance(selection, (int, np.integer)) or selection < 0 or selection >= count:
        return None, "legacy_component_selection_invalid"
    return ids[components == int(selection)], "legacy_selected_native_component_unverified"


def rebase_assessment(assessment, *, old_points, old_labels, old_label, old_path,
                      new_points, new_labels, new_label, new_path, scale, offset,
                      triangles=None, body_start_index=0, old_context=None,
                      new_context=None):
    """Rebase a verified record after an explicit uniform coordinate change.

    Label IDs may change; native membership must not.  This is not a way to
    refresh stale evidence after editing a path, ownership, or native geometry.
    """
    result = copy.deepcopy(assessment or {})
    scale, offset = float(scale), np.asarray(offset, float)
    if not np.isfinite(scale) or scale <= 0 or offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("evidence rebase needs a positive scale and finite XYZ offset")
    evidence = result.get("measurement_evidence")
    valid, reason = validate_measurement_evidence(evidence, old_points, old_labels, old_label,
                                                old_path, triangles=triangles,
                                                body_start_index=body_start_index, context=old_context)
    if valid and (not np.array_equal(np.flatnonzero(np.asarray(old_labels) == old_label),
                                     np.flatnonzero(np.asarray(new_labels) == new_label))
                  or not np.allclose(np.asarray(new_points), np.asarray(old_points) * scale + offset, rtol=0, atol=1e-8)
                  or not np.allclose(np.asarray(new_path), np.asarray(old_path) * scale + offset, rtol=0, atol=1e-8)):
        valid, reason = False, "non_affine_or_membership_changed_rebase"
    if not valid:
        if isinstance(evidence, dict):
            evidence.update(status="invalidated", invalid_reason=reason)
        result["measurement_evidence_invalid_reason"] = reason
        if isinstance(result.get("prefit_identity_evidence"), dict):
            result["prefit_identity_evidence"].update(reference_valid=False,
                                                       invalid_reason="unverified_coordinate_rebase")
        return result
    reference = result.get("prefit_identity_evidence")
    if isinstance(reference, dict):
        for key in ("prior_path", "origin"):
            if key in reference:
                value = np.asarray(reference[key], float)
                reference[key] = (value * scale + offset).tolist() if value.size else []
        if reference.get("mean_radius") is not None:
            reference["mean_radius"] *= scale
        reference["path_fingerprint"] = array_fingerprint(reference["prior_path"])
        reference["coordinate_unit"] = "rebased_coordinates"
    rebuilt = build_measurement_evidence(new_points, new_labels, new_label, new_path,
        triangles=triangles, selected_vertex_indices=evidence["native_vertex_indices"],
        spacing=evidence["spacing"] * scale, body_start_index=body_start_index,
        identity_resolved=evidence["identity_resolved"],
        attachment_supported=evidence["attachment_supported"],
        tip_supported=evidence["biological_tip_verified"],
        origin_coverage_complete=evidence.get("origin_coverage_complete", False),
        origin_coverage_evidence=evidence.get("origin_coverage_evidence"),
        source=evidence["source"], context=new_context)
    origin_detail = rebuilt.get("origin_coverage_evidence")
    if isinstance(origin_detail, dict):
        for key in ("axial_loss", "axial_displacement", "radial_offset", "tolerance"):
            if origin_detail.get(key) is not None:
                origin_detail[key] *= scale
        if origin_detail.get("origin_reference"):
            origin_detail["origin_reference"] = (np.asarray(origin_detail["origin_reference"]) * scale + offset).tolist()
    rebuilt["coordinate_unit"] = "rebased_coordinates"
    result["measurement_evidence"] = rebuilt
    return result
