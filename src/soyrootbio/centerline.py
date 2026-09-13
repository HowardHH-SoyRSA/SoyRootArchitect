"""Conservative final-ownership centreline assessment (all distances normalized).

Ownership and hierarchy are inputs, never inferred or changed here. Mesh edges
are the connectivity authority; point clouds use bounded local neighbours.
Disconnected components are never bridged. A root with no measurable body is
represented by a single location and flagged, not by an unsupported polyline.
"""
from __future__ import annotations

from collections.abc import Callable
import hashlib

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .geometry import child_length_exceeds_parent, path_length, resample_polyline, tangent_vectors
from .primary import _plane_basis, _robust_cross_section_center
from .types import RootPath


POLICY = {
    "method": "final-owned-transverse-centers-v1",
    "coordinate_unit": "normalized",
    "iterations": 2,
    "section_half_width_spacing": 2.0,
    "curvature_turn_limit_radians": 0.30,
    "minimum_section_points": 8,
    "smoothing_neighbor_weight": 0.125,
    "smoothing_max_displacement_spacing": 0.25,
    "component_policy": "largest_connected_support_no_bridges",
    "sparse_policy": "longest_contiguous_measurable_body_or_single_location",
    "tip_policy": "assigned_terminal_sections_no_extrapolation",
    "connector_policy": "owned_junction_transverse_hulls_no_free_space_bridges",
    "connector_section_half_width_spacing": 1.5,
    "connector_hull_tolerance_spacing": 0.25,
}


def _support_edges(points: np.ndarray, triangles: np.ndarray | None, spacing: float) -> np.ndarray:
    if triangles is not None and len(triangles):
        faces = np.asarray(triangles, dtype=int)
        edges = np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
        lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
        # Reject anomalously long triangles across holes too.
        positive = lengths[lengths > 0]
        limit = 6 * float(np.median(positive)) if len(positive) else spacing
        return edges[lengths <= limit]
    if len(points) < 2:
        return np.empty((0, 2), dtype=int)
    distances, neighbours = cKDTree(points).query(points, k=min(9, len(points)))
    local = np.maximum(distances[:, 1], spacing)
    rows = np.broadcast_to(np.arange(len(points))[:, None], neighbours[:, 1:].shape)
    keep = distances[:, 1:] <= np.minimum(4 * spacing, 4 * np.minimum(local[:, None], local[neighbours[:, 1:]]))
    return np.unique(np.sort(np.column_stack([rows[keep], neighbours[:, 1:][keep]]), axis=1), axis=0)


def _graph(points: np.ndarray, edges: np.ndarray):
    lengths = np.maximum(np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1), 1e-15)
    return coo_matrix((np.r_[lengths, lengths], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(len(points), len(points))).tocsr()


def _axis(points: np.ndarray, graph, hint: np.ndarray, spacing: float) -> np.ndarray:
    seed = int(np.argmin(np.linalg.norm(points - hint, axis=1)))
    end = int(np.argmax(dijkstra(graph, indices=seed)))
    de = dijkstra(graph, indices=end)
    start = int(np.argmax(de))
    if np.linalg.norm(points[end] - hint) < np.linalg.norm(points[start] - hint):
        start, end = end, start
        de = dijkstra(graph, indices=end)
    ds = dijkstra(graph, indices=start)
    length = float(ds[end])
    axial = np.clip((ds - de + length) * 0.5, 0, length)
    # Avoid circumference walks on straight, elongated support.
    origin = points.mean(axis=0)
    _, singular, basis = np.linalg.svd(points - origin, full_matrices=False)
    principal = (points - origin) @ basis[0]
    if singular[0] > 3 * max(singular[1], 1e-12) and length < 1.5 * np.ptp(principal):
        if principal[start] > principal[end]:
            principal = -principal
        axial = principal - principal.min()
        length = float(axial.max())
    step = max(2 * spacing, length / 700)
    stations = np.linspace(0, length, max(2, int(np.ceil(length / step)) + 1))
    centers = []
    for x in stations:
        support = points[np.abs(axial - x) <= step]
        centers.append(np.mean(support, axis=0) if len(support) else points[np.argmin(np.abs(axial - x))])
    line = np.asarray(centers)
    radius = float(np.median(cKDTree(points).query(line)[0]))
    # Geodesic extrema are surface vertices, often on a cap rim. Replace the
    # circumference walk at each end with a terminal transverse centre, using
    # the nearby interior axis only to orient the observed terminal slab.
    for reverse in (False, True):
        if reverse:
            line = line[::-1]
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
        k = int(np.searchsorted(arc, max(2 * radius, 3 * spacing)))
        k = min(max(1, k), max(1, (len(line) - 1) // 3))
        if 2 * k < len(line):
            tangent = line[2 * k] - line[k]
            tangent /= max(np.linalg.norm(tangent), 1e-12)
            local = points[np.linalg.norm(points - line[k], axis=1) <= max(4 * radius, 6 * spacing)]
            if len(local) >= 8:
                longitudinal = (local - line[k]) @ tangent
                terminal = float(np.quantile(longitudinal, .01))
                cap = local[longitudinal <= terminal + spacing]
                if len(cap) >= 3:
                    basis = _plane_basis(tangent)
                    offsets = cap - line[k]
                    center = line[k] + float(np.median(offsets @ tangent)) * tangent + _robust_cross_section_center(offsets @ basis.T) @ basis
                    line = np.vstack([center, line[k:]])
        if reverse:
            line = line[::-1]
    return line


def _section(points, tree, line, tangents, arc, assigned, i, spacing):
    station, tangent = line[i], tangents[i]
    left, right = max(0, i - 1), min(len(line) - 1, i + 1)
    turn = np.arccos(np.clip(np.dot(tangents[left], tangents[right]), -1, 1))
    curvature = turn / max(arc[right] - arc[left], spacing)
    half_width = min(2 * spacing / (1 + 2 * spacing * curvature), 0.30 / max(curvature, 1e-12))
    half_width = max(0.5 * spacing, half_width)
    near = np.atleast_1d(tree.query(station, k=min(16, len(points)))[0])
    radius = max(4 * spacing, 2.5 * float(np.median(near)))
    local = np.asarray(tree.query_ball_point(station, radius), dtype=int)
    delta = points[local] - station
    mask = (np.abs(delta @ tangent) <= half_width) & (np.abs(arc[assigned[local]] - arc[i]) <= max(3 * half_width, 2 * spacing))
    delta = delta[mask]
    if len(delta) < 3:
        # For a sampled thin axis, a nearby observed point is safer than a
        # fabricated cross section. Wide surface walls cannot use this fallback.
        nearby = points[local][np.linalg.norm(points[local] - station, axis=1) <= 2 * spacing]
        if len(nearby):
            return np.mean(nearby, axis=0), half_width, False
        return None, half_width, False
    basis = _plane_basis(tangent)
    transverse = delta @ basis.T
    center = _robust_cross_section_center(transverse)
    radial = transverse - center
    # Thin sampled axes are supported too; wide surfaces require a surrounding
    # transverse section, never an extrapolated circle fitted to a wall.
    thin = float(np.max(np.linalg.norm(radial, axis=1))) <= 2 * spacing
    if not thin:
        if len(delta) < POLICY["minimum_section_points"] or np.linalg.matrix_rank(radial) < 2:
            return station + np.mean(delta, axis=0), half_width, False
        angles = np.sort(np.arctan2(radial[:, 1], radial[:, 0]))
        if np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi])) > np.pi:
            return station + np.mean(delta, axis=0), half_width, False
    candidate = station + center @ basis
    # Endpoints are cap/terminal-section centres in all three coordinates.
    # No furthest-vertex tip and no tangent extrapolation beyond observations.
    if i in (0, len(line) - 1):
        candidate += float(np.median(delta @ tangent)) * tangent
    return candidate, half_width, True


def _fit_body(points, initial, spacing):
    line = resample_polyline(initial, max(spacing, path_length(initial) / 700))
    if len(line) < 2:
        return line, {"rejected_section_count": 0, "status": "insufficient_support"}
    tree = cKDTree(points)
    widths = []
    for _ in range(POLICY["iterations"]):
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
        # A one-node tangent at a surface cap follows the circumference. Use a
        # short radius-scaled chord for section orientation, without smoothing
        # the line itself or averaging support across distant turns.
        radius = float(np.median(tree.query(line)[0]))
        span = max(3 * spacing, 2 * radius)
        left = np.maximum(0, arc - span)
        right = np.minimum(arc[-1], arc + span)
        tangent_delta = np.column_stack([np.interp(right, arc, line[:, j]) - np.interp(left, arc, line[:, j]) for j in range(3)])
        tangents = tangent_delta / np.maximum(np.linalg.norm(tangent_delta, axis=1), 1e-12)[:, None]
        assigned = cKDTree(line).query(points)[1]
        updated = line.copy()
        valid = np.zeros(len(line), dtype=bool)
        reliable = np.zeros(len(line), dtype=bool)
        widths = []
        for i in range(len(line)):
            candidate, width, measured = _section(points, tree, line, tangents, arc, assigned, i, spacing)
            widths.append(width)
            if candidate is not None:
                updated[i] = candidate
                valid[i] = True
                reliable[i] = measured
        line = updated
    # Never interpolate across a section that has no measurable support.
    groups = np.split(np.flatnonzero(valid), np.flatnonzero(np.diff(np.flatnonzero(valid)) > 1) + 1)
    run = max(groups, key=lambda group: path_length(line[group]) if len(group) else -1)
    rejected = int(np.count_nonzero(~valid))
    if len(run) < 2:
        return np.median(points, axis=0)[None, :], {"status": "insufficient_support", "rejected_section_count": rejected}
    line = line[run]
    # One local pass only. Its displacement bound shrinks at bends; endpoints
    # and the support-derived tip are never smoothed.
    if len(line) > 2:
        delta = 0.125 * (line[:-2] + line[2:] - 2 * line[1:-1])
        limit = 0.25 * np.asarray(widths)[run[1:-1]] / 2
        delta *= np.minimum(1, limit / np.maximum(np.linalg.norm(delta, axis=1), 1e-12))[:, None]
        line[1:-1] += delta
    return line, {"status": "fitted" if rejected == 0 and np.all(reliable) else "partial_support", "rejected_section_count": rejected,
                  "sparse_section_count": int(np.count_nonzero(valid & ~reliable)),
                  "excluded_section_count": int(len(valid) - len(run)),
                  "section_half_width_min": float(min(widths)), "section_half_width_max": float(max(widths))}


def _supported_connector(candidate, parent_support, child_support, spacing):
    """Test actual owned surface along a junction, including parent protrusions.

    A main-axis radius sphere misses parent-owned junction surfaces. Local
    transverse hulls can certify them without assuming empty space is filled.
    Parent and child support are used only for this attachment, never the body.
    """
    support = np.vstack([parent_support, child_support])
    if len(support) < 8:
        return False
    samples = resample_polyline(candidate, spacing)
    tree = cKDTree(support)
    for station, tangent in zip(samples, tangent_vectors(samples)):
        nearest = np.atleast_1d(tree.query(station, k=min(16, len(support)))[0])
        radius = max(4 * spacing, 2.5 * float(np.median(nearest)))
        local = support[tree.query_ball_point(station, radius)] - station
        slab = local[np.abs(local @ tangent) <= 1.5 * spacing]
        if len(slab) < 3:
            return False
        radial = slab @ _plane_basis(tangent).T
        if float(np.max(np.linalg.norm(radial, axis=1))) <= 2 * spacing:
            continue
        if len(slab) < 8:
            return False
        try:
            hull = ConvexHull(radial)
        except QhullError:
            return False
        if np.max(hull.equations[:, -1]) > .25 * spacing:
            return False
    return True


def refit_final_centerlines(
    points: np.ndarray, labels: np.ndarray, primary: np.ndarray,
    roots: list[RootPath], *, d_bar: float, triangles: np.ndarray | None = None,
    cooperate: Callable[[], None] | None = None,
) -> tuple[np.ndarray, dict]:
    """Refit all roots in parent-first order using full-resolution final labels.

    Labels use 0 for primary and one-based positions in ``roots``. The list is
    not reordered. Reports and body indices are saved on RootPath for export.
    An unsupported attachment keeps its topology metadata but no spatial bridge.
    """
    points, labels = np.asarray(points, dtype=float), np.asarray(labels)
    if points.ndim != 2 or points.shape[1] != 3 or labels.shape != (len(points),):
        raise ValueError("final fitting requires XYZ points and one label per point")
    if not np.all(np.isfinite(points)) or not np.isfinite(d_bar) or d_bar <= 0:
        raise ValueError("final fitting requires finite points and positive spacing")
    edges = _support_edges(points, triangles, d_bar)
    same = labels[edges[:, 0]] == labels[edges[:, 1]]
    owned_edges = edges[same]
    by_id = {root.root_id: (i, root) for i, root in enumerate(roots, 1)}
    if len(by_id) != len(roots) or "primary" in by_id:
        raise ValueError("root IDs must be unique and distinct from primary")
    pending = list(roots)
    ordered = []
    seen = {"primary"}
    while pending:
        ready = [r for r in pending if r.parent_id in seen]
        if not ready:
            raise ValueError("final fitting requires a repaired acyclic hierarchy")
        ordered.extend(ready)
        seen.update(r.root_id for r in ready)
        pending = [r for r in pending if r not in ready]
    curves, supports, reports = {}, {}, []
    primary_flags = []
    for root in [None, *ordered]:
        if cooperate:
            cooperate()
        rid = "primary" if root is None else root.root_id
        label = 0 if root is None else by_id[rid][0]
        old = primary if root is None else root.points
        indices = np.flatnonzero(labels == label)
        flags = []
        detail = {"root_id": rid, "numeric_label": label, "assigned_point_count": int(len(indices)), "length_before": path_length(old),
                  "coordinate_unit": "normalized", "input_geometry_fingerprint": hashlib.sha256(np.round(np.asarray(old, dtype=np.float64), decimals=8).tobytes(order="C")).hexdigest()[:20]}
        if not len(indices):
            # Preserve the record for label mapping, but export no line segment.
            body = np.asarray(old[:1], dtype=float).copy()
            support = np.empty((0, 3))
            detail.update(status="no_support", component_count=0, excluded_fragment_points=0)
        else:
            lookup = np.full(len(points), -1, dtype=int)
            lookup[indices] = np.arange(len(indices))
            local_edges = lookup[owned_edges[labels[owned_edges[:, 0]] == label]]
            cloud = points[indices]
            graph = _graph(cloud, local_edges)
            count, components = connected_components(graph, directed=False)
            sizes = np.bincount(components)
            keep = components == int(np.argmax(sizes))
            support = cloud[keep]
            # Full resolution spacing, bounded by the analysis spacing. This
            # prevents a reduced analysis cloud from producing wide bend slabs.
            spacing = d_bar
            if len(support) > 1:
                nn = cKDTree(support).query(support, k=2)[0][:, 1]
                positive = nn[nn > 1e-12]
                if len(positive):
                    spacing = min(d_bar, float(np.median(positive)))
            detail.update(component_count=int(count), excluded_fragment_points=int(np.count_nonzero(~keep)), fit_spacing=spacing)
            if count > 1:
                flags.append("centerline_disconnected_support")
            if len(support) < 8:
                body = np.median(support, axis=0)[None, :]
                detail["status"] = "insufficient_support"
            else:
                hint = old[0]
                initial = _axis(support, graph[keep][:, keep], hint, spacing)
                body, fit_report = _fit_body(support, initial, spacing)
                detail.update(fit_report)
        if detail["status"] != "fitted":
            flags.append("centerline_" + detail["status"])
        body_start = 0
        line = body
        if root is not None:
            parent = curves[root.parent_id]
            # Retain the repaired insertion neighbourhood, not an arbitrary
            # nearest turn of a curved or contacting parent.
            hint = root.insertion_point if root.insertion_point is not None else old[0]
            insertion_index = int(cKDTree(parent).query(hint)[1])
            anchor = parent[insertion_index]
            parent_support = supports[root.parent_id]
            gap = float(np.linalg.norm(body[0] - anchor))
            connector_ok = gap <= 1e-12
            connector = np.vstack([anchor, body[0]])
            if len(parent_support) >= 8 and len(body) >= 2:
                # The parent can be sampled more coarsely than the child.
                # Use the analysis spacing at their shared junction.
                connector_spacing = d_bar
                connector_ok = _supported_connector(connector, parent_support, support, connector_spacing)
                if not connector_ok:
                    # Preserve a curved original junction only if every local
                    # section is supported after the final ownership changes.
                    end = int(cKDTree(old).query(body[0])[1])
                    curved = np.vstack([anchor, old[1:end], body[0]])
                    if len(curved) > 2 and _supported_connector(curved, parent_support, support, connector_spacing):
                        connector, connector_ok = curved, True
            if connector_ok and gap > 1e-12 and len(body) >= 2:
                line = np.vstack([connector[:-1], body])
                body_start = len(connector) - 1
            elif not connector_ok:
                flags.append("centerline_unsupported_parent_connector")
            root.insertion_index = insertion_index
            root.insertion_point = anchor.copy()
            root.parent_points = parent
            root.points = line
            root.body_start_index = body_start
            root.node_indices = None
            root.start_index = None
            detail["parent_connector_supported"] = bool(connector_ok)
            detail["parent_connector_length"] = path_length(connector) if body_start else 0.0
            detail["attachment_gap"] = 0.0 if connector_ok else gap
            if child_length_exceeds_parent(path_length(line), path_length(parent)):
                flags.append("centerline_refit_child_longer_than_parent")
            root.qc_flags = list(dict.fromkeys([*root.qc_flags, *flags]))
            root.centerline_assessment = detail
        else:
            primary_flags = flags
        detail.update(length_after=path_length(line), exposed_body_length=path_length(body), body_start_index=body_start,
                      tip_displacement=float(np.linalg.norm(line[-1] - old[-1])), qc_flags=flags)
        curves[rid], supports[rid] = line, support
        reports.append(detail)
    return curves["primary"], {**POLICY, "primary_qc_flags": primary_flags, "roots": reports}
