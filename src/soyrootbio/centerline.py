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
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .geometry import (
    child_length_exceeds_parent,
    is_above_primary_top,
    path_length,
    primary_top_excess,
    resample_polyline,
    tangent_vectors,
)
from .primary import _plane_basis, _robust_cross_section_center
from .types import RootPath


POLICY = {
    "method": "final-owned-transverse-centers-v2",
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
    "descendant_attachment_policy": "fit_bodies_then_resolve_and_commit_parent_families_with_repaired_prefix_or_prior_fallback",
    "connector_section_half_width_spacing": 1.5,
    "connector_hull_tolerance_spacing": 0.25,
    "origin_policy": "every lateral insertion is at or below the immutable selected primary top along configured gravity",
    "primary_method": "prior-guided-area-angular-sections-curvature-spline-v2",
    "primary_prior_policy": "soft_only_where_locally_supported",
    "primary_section_angular_bins": 16,
    "primary_minimum_angular_coverage": 0.55,
    "primary_station_spacing_radius_fraction": 0.5,
    "primary_station_spacing_floor": 2.0,
    "primary_length_inflation_limit": 1.15,
    "primary_qc_failure_policy": "retain_prior_axis_and_do_not_apply_rejected_fit",
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


def _vertex_area_weights(points: np.ndarray, triangles: np.ndarray | None) -> np.ndarray:
    """Return per-vertex surface area, or neutral weights for point clouds."""
    weights = np.ones(len(points), dtype=float)
    if triangles is None or not len(triangles):
        return weights
    faces = np.asarray(triangles, dtype=int)
    valid = np.all((faces >= 0) & (faces < len(points)), axis=1)
    faces = faces[valid]
    if not len(faces):
        return weights
    area = 0.5 * np.linalg.norm(
        np.cross(
            points[faces[:, 1]] - points[faces[:, 0]],
            points[faces[:, 2]] - points[faces[:, 0]],
        ),
        axis=1,
    )
    weights.fill(0.0)
    np.add.at(weights, faces[:, 0], area / 3.0)
    np.add.at(weights, faces[:, 1], area / 3.0)
    np.add.at(weights, faces[:, 2], area / 3.0)
    positive = weights[weights > 1e-15]
    replacement = float(np.median(positive)) if len(positive) else 1.0
    weights[weights <= 1e-15] = replacement
    return weights


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    ordered_values = np.asarray(values, dtype=float)[order]
    ordered_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(ordered_weights)
    if not len(cumulative) or cumulative[-1] <= 0:
        return float(np.quantile(ordered_values, quantile))
    return float(ordered_values[np.searchsorted(cumulative, quantile * cumulative[-1], side="left")])


def _weighted_geometric_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    estimate = np.average(values, axis=0, weights=weights)
    for _ in range(20):
        distances = np.linalg.norm(values - estimate, axis=1)
        if np.any(distances <= 1e-12):
            return values[int(np.argmin(distances))].copy()
        adjusted = weights / np.maximum(distances, 1e-12)
        updated = np.average(values, axis=0, weights=adjusted)
        if np.linalg.norm(updated - estimate) <= 1e-10:
            return updated
        estimate = updated
    return estimate


def _weighted_circle_center(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Fit a robust transverse circle without letting dense wall samples win."""
    retained = np.arange(len(values))
    fallback = _robust_cross_section_center(values, fit_circle=True)
    center = fallback
    for _ in range(3):
        local = values[retained]
        local_weights = weights[retained]
        design = np.column_stack([2.0 * local[:, 0], 2.0 * local[:, 1], np.ones(len(local))])
        target = np.sum(local * local, axis=1)
        weighted_design = design * np.sqrt(local_weights)[:, None]
        weighted_target = target * np.sqrt(local_weights)
        if np.linalg.matrix_rank(weighted_design) < 3:
            return fallback
        coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
        center = np.asarray(coefficients[:2], dtype=float)
        radii = np.linalg.norm(local - center, axis=1)
        residual = np.abs(radii - _weighted_quantile(radii, local_weights, 0.5))
        cutoff = _weighted_quantile(residual, local_weights, 0.85)
        keep = residual <= max(cutoff, 1e-12)
        if int(np.count_nonzero(keep)) < 8 or np.all(keep):
            break
        retained = retained[keep]
    span = max(float(np.linalg.norm(np.ptp(values, axis=0))), 1e-12)
    if not np.all(np.isfinite(center)) or np.linalg.norm(center - fallback) > span:
        return fallback
    return center


def _project_to_polygon(point: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Project a 2-D point to a convex polygon when it lies outside it."""
    if len(polygon) < 3:
        return point
    hull = ConvexHull(polygon)
    if np.all(hull.equations[:, :2] @ point + hull.equations[:, 2] <= 1e-12):
        return point
    vertices = polygon[hull.vertices]
    closest = None
    closest_distance = np.inf
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        edge = end - start
        fraction = float(np.dot(point - start, edge) / max(np.dot(edge, edge), 1e-15))
        candidate = start + np.clip(fraction, 0.0, 1.0) * edge
        distance = float(np.linalg.norm(point - candidate))
        if distance < closest_distance:
            closest = candidate
            closest_distance = distance
    return np.asarray(closest, dtype=float)


def _polyline_projection(query: np.ndarray, line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact point-to-segment distance and arc coordinate."""
    query = np.atleast_2d(np.asarray(query, dtype=float))
    line = np.asarray(line, dtype=float)
    if len(line) < 2:
        return np.linalg.norm(query - line[0], axis=1), np.zeros(len(query))
    segment = np.diff(line, axis=0)
    squared = np.sum(segment * segment, axis=1)
    arc = np.r_[0.0, np.cumsum(np.sqrt(squared))]
    distances = np.empty(len(query), dtype=float)
    coordinates = np.empty(len(query), dtype=float)
    for index, point in enumerate(query):
        fraction = np.sum((point - line[:-1]) * segment, axis=1) / np.maximum(squared, 1e-15)
        fraction = np.clip(fraction, 0.0, 1.0)
        projected = line[:-1] + fraction[:, None] * segment
        local_distance = np.linalg.norm(projected - point, axis=1)
        best = int(np.argmin(local_distance))
        distances[index] = local_distance[best]
        coordinates[index] = arc[best] + fraction[best] * np.sqrt(squared[best])
    return distances, coordinates


def _primary_section(
    points: np.ndarray,
    point_weights: np.ndarray,
    tree: cKDTree,
    line: np.ndarray,
    tangents: np.ndarray,
    arc: np.ndarray,
    assigned: np.ndarray,
    local_radius: np.ndarray,
    index: int,
    spacing: float,
    station_step: float,
) -> dict:
    station = line[index]
    tangent = tangents[index]
    left, right = max(0, index - 1), min(len(line) - 1, index + 1)
    turn = np.arccos(np.clip(np.dot(tangents[left], tangents[right]), -1.0, 1.0))
    curvature = turn / max(arc[right] - arc[left], spacing)
    radius = float(local_radius[index])
    half_width = max(
        1.5 * spacing,
        min(0.65 * radius, 0.35 / max(curvature, 1e-12)),
    )
    search_radius = max(3.0 * radius, 6.0 * spacing)
    local = np.asarray(tree.query_ball_point(station, search_radius), dtype=int)
    record = {
        "index": int(index),
        "accepted": False,
        "coverage": 0.0,
        "half_width": float(half_width),
        "radius": radius,
        "curvature": float(curvature),
    }
    if not len(local):
        record["rejection_reason"] = "empty"
        return record
    delta = points[local] - station
    longitudinal = delta @ tangent
    station_distance = np.abs(arc[assigned[local]] - arc[index])
    keep = (np.abs(longitudinal) <= half_width) & (
        station_distance <= max(2.0 * half_width, 1.5 * station_step)
    )
    local = local[keep]
    delta = delta[keep]
    longitudinal = longitudinal[keep]
    weights = point_weights[local]
    if len(delta) < 3:
        record["rejection_reason"] = "too_few_points"
        return record
    basis = _plane_basis(tangent)
    transverse = delta @ basis.T
    initial = _weighted_geometric_median(transverse, weights)
    radial = transverse - initial
    radial_distance = np.linalg.norm(radial, axis=1)
    if _weighted_quantile(radial_distance, weights, 0.9) <= 2.0 * spacing:
        candidate = station + initial @ basis
        record.update(
            accepted=True,
            coverage=1.0,
            candidate=candidate,
            basis=basis,
            tangent=tangent,
            envelope=None,
            uncertainty=float(spacing),
            prior_inside=True,
            thin_support=True,
        )
        return record
    if len(delta) < POLICY["minimum_section_points"] or np.linalg.matrix_rank(radial) < 2:
        record["rejection_reason"] = "degenerate"
        return record
    # Coverage is measured around the retained prior station. Recentring on a
    # partial wall can make that wall appear to surround its own biased centre.
    angles = np.mod(np.arctan2(transverse[:, 1], transverse[:, 0]), 2.0 * np.pi)
    shape_center = _weighted_circle_center(transverse, weights)
    radial_distance = np.linalg.norm(transverse - shape_center, axis=1)
    bin_count = int(POLICY["primary_section_angular_bins"])
    bins = np.minimum((angles * bin_count / (2.0 * np.pi)).astype(int), bin_count - 1)
    occupied = np.unique(bins)
    ordered_angles = np.sort(angles)
    largest_gap = float(np.max(np.diff(np.r_[ordered_angles, ordered_angles[0] + 2.0 * np.pi])))
    coverage = min(float(len(occupied) / bin_count), float(1.0 - largest_gap / (2.0 * np.pi)))
    record["coverage"] = coverage
    record["largest_angular_gap_radians"] = largest_gap
    if coverage < float(POLICY["primary_minimum_angular_coverage"]) or largest_gap >= np.pi:
        record["rejection_reason"] = "one_sided_wall"
        return record
    angular_weights = np.zeros(len(weights), dtype=float)
    bin_radii = []
    for bin_index in occupied:
        members = bins == bin_index
        total = float(np.sum(weights[members]))
        angular_weights[members] = weights[members] / max(total, 1e-15)
        bin_radii.append(_weighted_quantile(radial_distance[members], weights[members], 0.5))
    bin_radii = np.asarray(bin_radii, dtype=float)
    median_bin_radius = float(np.median(bin_radii))
    flare_radius = float(np.quantile(bin_radii, 0.9))
    record["radial_bin_ratio_p90_median"] = flare_radius / max(median_bin_radius, spacing)
    if flare_radius > max(2.4 * median_bin_radius, median_bin_radius + 4.0 * spacing):
        record["rejection_reason"] = "mixed_flare"
        return record
    center = _weighted_circle_center(transverse, angular_weights)
    centered_radius = np.linalg.norm(transverse - center, axis=1)
    median_radius = _weighted_quantile(centered_radius, angular_weights, 0.5)
    radial_spread = (
        _weighted_quantile(centered_radius, angular_weights, 0.9)
        - _weighted_quantile(centered_radius, angular_weights, 0.1)
    ) / max(median_radius, spacing)
    uncertainty = max(
        0.25 * spacing,
        median_radius * (0.06 + 0.6 * (1.0 - coverage) + 0.15 * min(radial_spread, 2.0)),
    )
    longitudinal_center = np.clip(
        _weighted_quantile(longitudinal, angular_weights, 0.5),
        -half_width,
        half_width,
    )
    envelope = transverse
    try:
        hull = ConvexHull(envelope)
        prior_inside = bool(np.all(hull.equations[:, 2] <= 0.1 * spacing))
        envelope = envelope[hull.vertices]
    except QhullError:
        record["rejection_reason"] = "degenerate_envelope"
        return record
    candidate = station + center @ basis + longitudinal_center * tangent
    record.update(
        accepted=True,
        candidate=candidate,
        basis=basis,
        tangent=tangent,
        envelope=envelope,
        uncertainty=float(uncertainty),
        prior_inside=prior_inside,
        thin_support=False,
    )
    return record


def _regularized_primary_curve(
    prior: np.ndarray,
    sections: list[dict],
    spacing: float,
) -> np.ndarray:
    count = len(prior)
    accepted = np.array([bool(section["accepted"]) for section in sections])
    observations = prior.copy()
    uncertainties = np.full(count, max(spacing, 1e-12), dtype=float)
    coverage = np.zeros(count, dtype=float)
    prior_inside = np.zeros(count, dtype=bool)
    for index, section in enumerate(sections):
        if not section["accepted"]:
            continue
        observations[index] = section["candidate"]
        uncertainties[index] = section["uncertainty"]
        coverage[index] = section["coverage"]
        prior_inside[index] = section["prior_inside"]
    reference_uncertainty = float(np.median(uncertainties[accepted]))
    data_weight = np.zeros(count, dtype=float)
    data_weight[accepted] = (
        reference_uncertainty / np.maximum(uncertainties[accepted], 0.25 * spacing)
    ) ** 2 * np.maximum(coverage[accepted], 0.25)
    prior_weight = np.zeros(count, dtype=float)
    prior_weight[accepted & prior_inside] = 0.001 + 0.01 * (
        1.0 - coverage[accepted & prior_inside]
    )
    prior_weight[accepted & ~prior_inside] = 0.0
    data_weight[[0, -1]] *= 12.0
    rows = np.repeat(np.arange(max(0, count - 2)), 3)
    columns = np.column_stack(
        [np.arange(max(0, count - 2)), np.arange(1, max(1, count - 1)), np.arange(2, count)]
    ).ravel()
    values = np.tile([1.0, -2.0, 1.0], max(0, count - 2))
    second_difference = coo_matrix((values, (rows, columns)), shape=(max(0, count - 2), count)).tocsr()
    if count > 2:
        bend = np.linalg.norm(second_difference @ prior, axis=1) / np.maximum(
            np.array([sections[index]["radius"] for index in range(1, count - 1)]),
            spacing,
        )
        local_uncertainty = np.array(
            [sections[index]["uncertainty"] if sections[index]["accepted"] else reference_uncertainty for index in range(1, count - 1)]
        ) / max(reference_uncertainty, 1e-12)
        curvature_weight = 12.0 * np.clip(local_uncertainty, 0.5, 3.0) / (1.0 + 40.0 * bend * bend)
        regularizer = second_difference.T @ diags(curvature_weight) @ second_difference
    else:
        regularizer = diags(np.zeros(count))
    system = diags(data_weight + prior_weight + 1e-8) + regularizer
    right = data_weight[:, None] * observations + prior_weight[:, None] * prior
    fitted = np.column_stack([spsolve(system, right[:, coordinate]) for coordinate in range(3)])
    fitted[0] = observations[0]
    fitted[-1] = observations[-1]
    for index, section in enumerate(sections):
        if section["accepted"]:
            delta = fitted[index] - prior[index]
            longitudinal = float(np.clip(
                np.dot(delta, section["tangent"]),
                -section["half_width"],
                section["half_width"],
            ))
            transverse = delta @ section["basis"].T
            if section["envelope"] is not None:
                transverse = _project_to_polygon(transverse, section["envelope"])
            fitted[index] = (
                prior[index]
                + longitudinal * section["tangent"]
                + transverse @ section["basis"]
            )
    return fitted


def _primary_fit_qc(
    candidate: np.ndarray,
    prior: np.ndarray,
    sections: list[dict],
    support: np.ndarray,
    spacing: float,
) -> tuple[dict, list[str]]:
    radii = np.asarray([section["radius"] for section in sections], dtype=float)
    scale = max(4.0 * spacing, float(np.median(radii)))
    sampled = resample_polyline(candidate, scale)
    turn = np.empty(0, dtype=float)
    if len(sampled) > 2:
        segment = np.diff(sampled, axis=0)
        segment /= np.maximum(np.linalg.norm(segment, axis=1), 1e-12)[:, None]
        turn = np.arccos(np.clip(np.sum(segment[:-1] * segment[1:], axis=1), -1.0, 1.0))
    distance_from_prior, prior_arc = _polyline_projection(candidate, prior)
    prior_length = path_length(prior)
    candidate_length = path_length(candidate)
    length_ratio = candidate_length / max(prior_length, 1e-12)
    backward = np.diff(prior_arc)
    accepted_sections = [section for section in sections if section["accepted"]]
    coverage = np.asarray([section["coverage"] for section in accepted_sections], dtype=float)
    support_tree = cKDTree(support)
    endpoint_radius = np.asarray([sections[0]["radius"], sections[-1]["radius"]])
    endpoint_distance = support_tree.query(prior[[0, -1]])[0]
    endpoint_supported = endpoint_distance <= np.maximum(1.5 * endpoint_radius, 3.0 * spacing)
    start_retreat = float(prior_arc[0])
    end_retreat = float(max(0.0, prior_length - prior_arc[-1]))
    turn_p95 = float(np.quantile(turn, 0.95)) if len(turn) else 0.0
    high_turn_fraction = float(np.mean(turn > np.deg2rad(45.0))) if len(turn) else 0.0
    metrics = {
        "fit_qc_passed": True,
        "candidate_length": candidate_length,
        "candidate_to_prior_length_ratio": length_ratio,
        "turn_p95_degrees_radius_scale": float(np.rad2deg(turn_p95)),
        "high_turn_fraction_radius_scale": high_turn_fraction,
        "backtracking_step_count": int(np.count_nonzero(backward < -0.25 * scale)),
        "maximum_backtracking": float(max(0.0, -np.min(backward))) if len(backward) else 0.0,
        "start_endpoint_retreat": start_retreat,
        "end_endpoint_retreat": end_retreat,
        "prior_endpoint_supported": endpoint_supported.tolist(),
        "accepted_section_fraction": float(len(accepted_sections) / max(len(sections), 1)),
        "angular_coverage_min": float(np.min(coverage)) if len(coverage) else 0.0,
        "angular_coverage_p10": float(np.quantile(coverage, 0.1)) if len(coverage) else 0.0,
        "prior_displacement_p95": float(np.quantile(distance_from_prior, 0.95)),
        "prior_displacement_max": float(np.max(distance_from_prior)),
    }
    flags = []
    if turn_p95 > np.deg2rad(70.0) or high_turn_fraction > 0.20:
        flags.append("centerline_primary_oscillation")
    if metrics["backtracking_step_count"]:
        flags.append("centerline_primary_backtracking")
    if length_ratio > float(POLICY["primary_length_inflation_limit"]) and candidate_length - prior_length > 2.0 * scale:
        flags.append("centerline_primary_length_inflation")
    if (endpoint_supported[0] and start_retreat > 2.0 * scale) or (
        endpoint_supported[1] and end_retreat > 2.0 * scale
    ):
        flags.append("centerline_primary_endpoint_retreat")
    if not len(coverage) or metrics["accepted_section_fraction"] < 0.50 or metrics["angular_coverage_p10"] < float(POLICY["primary_minimum_angular_coverage"]):
        flags.append("centerline_primary_low_angular_coverage")
    if metrics["prior_displacement_p95"] > max(1.5 * float(np.median(radii)), 4.0 * spacing):
        flags.append("centerline_primary_prior_displacement")
    metrics["fit_qc_passed"] = not flags
    return metrics, flags


def _fit_primary_body(
    points: np.ndarray,
    prior: np.ndarray,
    spacing: float,
    point_weights: np.ndarray,
) -> tuple[np.ndarray, dict]:
    prior = np.asarray(prior, dtype=float)
    prior_length = path_length(prior)
    if len(prior) < 2 or prior_length <= 0:
        return prior.copy(), {
            "status": "insufficient_support",
            "fit_qc_passed": False,
            "fit_applied": False,
            "traits_geometry_source": "retained_prior",
            "rejected_section_count": 0,
            "qc_flags": ["centerline_primary_low_angular_coverage"],
        }
    tree = cKDTree(points)
    pilot = resample_polyline(prior, max(2.0 * spacing, prior_length / 700.0))
    pilot_near = np.atleast_2d(tree.query(pilot, k=min(24, len(points)))[0])
    if pilot_near.shape[0] != len(pilot):
        pilot_near = pilot_near.T
    pilot_radius = np.maximum(2.0 * spacing, np.median(pilot_near, axis=1))
    station_step = max(
        float(POLICY["primary_station_spacing_floor"]) * spacing,
        float(POLICY["primary_station_spacing_radius_fraction"]) * float(np.median(pilot_radius)),
        prior_length / 700.0,
    )
    line = resample_polyline(prior, station_step)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    near = np.atleast_2d(tree.query(line, k=min(24, len(points)))[0])
    if near.shape[0] != len(line):
        near = near.T
    local_radius = np.maximum(2.0 * spacing, np.median(near, axis=1))
    if len(local_radius) > 4:
        padded = np.pad(local_radius, 2, mode="edge")
        local_radius = np.asarray([np.median(padded[index:index + 5]) for index in range(len(local_radius))])
    left_arc = np.maximum(0.0, arc - np.maximum(3.0 * spacing, 2.0 * local_radius))
    right_arc = np.minimum(arc[-1], arc + np.maximum(3.0 * spacing, 2.0 * local_radius))
    tangent_delta = np.column_stack(
        [np.interp(right_arc, arc, line[:, coordinate]) - np.interp(left_arc, arc, line[:, coordinate]) for coordinate in range(3)]
    )
    tangents = tangent_delta / np.maximum(np.linalg.norm(tangent_delta, axis=1), 1e-12)[:, None]
    assigned = cKDTree(line).query(points)[1]
    sections = [
        _primary_section(
            points,
            point_weights,
            tree,
            line,
            tangents,
            arc,
            assigned,
            local_radius,
            index,
            spacing,
            station_step,
        )
        for index in range(len(line))
    ]
    accepted = np.array([section["accepted"] for section in sections], dtype=bool)
    accepted_indices = np.flatnonzero(accepted)
    rejection_counts = {}
    for section in sections:
        if section["accepted"]:
            continue
        reason = section.get("rejection_reason", "unknown")
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    base_report = {
        "station_spacing": float(station_step),
        "section_radius_median": float(np.median(local_radius)),
        "section_radius_min": float(np.min(local_radius)),
        "section_radius_max": float(np.max(local_radius)),
        "rejected_section_count": int(np.count_nonzero(~accepted)),
        "section_rejection_counts": rejection_counts,
        "section_half_width_min": float(min(section["half_width"] for section in sections)),
        "section_half_width_max": float(max(section["half_width"] for section in sections)),
    }
    if len(accepted_indices) < 3:
        flags = ["centerline_primary_low_angular_coverage"]
        return prior.copy(), {
            **base_report,
            "status": "qc_rejected_prior_retained",
            "fit_qc_passed": False,
            "fit_applied": False,
            "traits_geometry_source": "retained_prior",
            "accepted_section_fraction": float(len(accepted_indices) / max(len(sections), 1)),
            "qc_flags": flags,
        }
    breaks = []
    for position in range(len(accepted_indices) - 1):
        first, second = accepted_indices[position:position + 2]
        gap_limit = max(3.0 * station_step, 2.0 * max(local_radius[first], local_radius[second]))
        if arc[second] - arc[first] > gap_limit:
            breaks.append(position + 1)
    groups = np.split(accepted_indices, breaks)
    run = max(groups, key=lambda group: arc[group[-1]] - arc[group[0]] if len(group) > 1 else -1.0)
    start, stop = int(run[0]), int(run[-1]) + 1
    candidate = _regularized_primary_curve(line[start:stop], sections[start:stop], spacing)
    qc, flags = _primary_fit_qc(candidate, prior, sections[start:stop], points, spacing)
    report = {
        **base_report,
        **qc,
        "excluded_section_count": int(len(sections) - (stop - start)),
        "sparse_section_count": int(np.count_nonzero(~accepted[start:stop])),
        "qc_flags": flags,
    }
    if flags:
        report.update(
            status="qc_rejected_prior_retained",
            fit_applied=False,
            traits_geometry_source="retained_prior",
        )
        return prior.copy(), report
    report.update(
        status="fitted" if np.all(accepted[start:stop]) else "partial_support",
        fit_applied=True,
        traits_geometry_source="accepted_fit",
    )
    return candidate, report


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


def _legal_attachment_indices(line, top_reference, gravity_direction):
    return np.asarray(
        [
            index for index, point in enumerate(line)
            if not is_above_primary_top(point, top_reference, gravity=gravity_direction)
        ],
        dtype=int,
    )


def _retained_attachment_prefix(old, body, support, child_hints, top_reference, gravity_direction, spacing):
    """Keep a repaired basal path only when it joins the fitted body locally.

    The old path is topology evidence, not new owned surface. A distant join
    would invent a spatial bridge, so callers retain the whole prior path
    instead when the candidate cannot be joined within the analysis spacing.
    """
    if len(old) < 2 or len(body) < 2 or not child_hints:
        return None
    required = []
    for hint in child_hints:
        distances = np.linalg.norm(old - hint, axis=1)
        index = int(np.argmin(distances))
        if distances[index] > max(2.0 * spacing, 1e-7):
            return None
        if is_above_primary_top(old[index], top_reference, gravity=gravity_direction):
            return None
        required.append(index)
    last_required = max(required)
    if last_required >= len(old) - 1:
        return None
    tail = old[last_required + 1:]
    join_index = last_required + 1 + int(np.argmin(np.linalg.norm(tail - body[0], axis=1)))
    join_gap = float(np.linalg.norm(old[join_index] - body[0]))
    if join_gap > 3.0 * spacing:
        return None
    if join_gap > 1e-10:
        if not len(support):
            return None
        bridge_samples = np.linspace(old[join_index], body[0], 5)
        if np.max(cKDTree(support).query(bridge_samples)[0]) > 2.0 * spacing:
            return None
    incoming = old[join_index] - old[join_index - 1]
    outgoing = body[1] - body[0]
    if float(np.dot(incoming, outgoing)) < -0.25 * float(np.linalg.norm(incoming) * np.linalg.norm(outgoing)):
        return None
    prefix = old[:join_index + 1].copy()
    if join_gap <= 1e-10:
        line = np.vstack([prefix[:-1], body])
        body_start = len(prefix) - 1
    else:
        line = np.vstack([prefix, body])
        body_start = len(prefix)
    if path_length(line) > 1.25 * max(path_length(old), path_length(body)) + 3.0 * spacing:
        return None
    for hint in child_hints:
        indices = _legal_attachment_indices(line[:body_start], top_reference, gravity_direction)
        if not len(indices) or np.min(np.linalg.norm(line[indices] - hint, axis=1)) > max(2.0 * spacing, 1e-7):
            return None
    return line, body_start, join_gap


def refit_final_centerlines(
    points: np.ndarray, labels: np.ndarray, primary: np.ndarray,
    roots: list[RootPath], *, d_bar: float, triangles: np.ndarray | None = None,
    gravity: np.ndarray | tuple[float, float, float] = (0.0, 0.0, -1.0),
    primary_top_reference: np.ndarray | None = None,
    cooperate: Callable[[], None] | None = None,
) -> tuple[np.ndarray, dict]:
    """Refit all roots in parent-first order using full-resolution final labels.

    Labels use 0 for primary and one-based positions in ``roots``. The list is
    not reordered. Reports and body indices are saved on RootPath for export.
    An unsupported attachment keeps its topology metadata but no spatial bridge.
    """
    points, labels = np.asarray(points, dtype=float), np.asarray(labels)
    primary = np.asarray(primary, dtype=float)
    gravity_direction = np.asarray(gravity, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or labels.shape != (len(points),):
        raise ValueError("final fitting requires XYZ points and one label per point")
    if not np.all(np.isfinite(points)) or not np.isfinite(d_bar) or d_bar <= 0:
        raise ValueError("final fitting requires finite points and positive spacing")
    if (
        primary.ndim != 2
        or primary.shape[1] != 3
        or not len(primary)
        or not np.all(np.isfinite(primary))
    ):
        raise ValueError("final fitting requires a finite primary path")
    if primary_top_reference is None:
        top_reference = primary.copy()
    else:
        top_point = np.asarray(primary_top_reference, dtype=float)
        if top_point.shape != (3,) or not np.all(np.isfinite(top_point)):
            raise ValueError("primary_top_reference must contain one finite XYZ coordinate")
        top_reference = top_point[None, :].copy()
    if (
        gravity_direction.shape != (3,)
        or not np.all(np.isfinite(gravity_direction))
        or np.linalg.norm(gravity_direction) <= 1e-12
    ):
        raise ValueError("gravity must contain three finite values and have non-zero length")
    gravity_direction /= np.linalg.norm(gravity_direction)
    for root in roots:
        origin = (
            np.asarray(root.insertion_point, dtype=float)
            if root.insertion_point is not None
            else np.asarray(root.points[0], dtype=float)
        )
        if is_above_primary_top(
            origin,
            top_reference,
            gravity=gravity_direction,
        ):
            excess, _ = primary_top_excess(
                origin,
                top_reference,
                gravity=gravity_direction,
            )
            raise ValueError(
                f"{root.root_id}: lateral origin is {excess:.9g} above "
                "the primary-root top"
            )
    edges = _support_edges(points, triangles, d_bar)
    vertex_area_weights = _vertex_area_weights(points, triangles)
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
    # Fit every exposed body against the same frozen ownership before changing
    # any parent path or insertion. Attachment resolution follows below.
    prior = {"primary": primary.copy()}
    prior.update({root.root_id: np.asarray(root.points, dtype=float).copy() for root in roots})
    child_hints: dict[str, list[tuple[str, np.ndarray]]] = {}
    for root in roots:
        hint = root.insertion_point if root.insertion_point is not None else prior[root.root_id][0]
        child_hints.setdefault(root.parent_id, []).append((root.root_id, np.asarray(hint, dtype=float).copy()))
    staged = {}
    for root in [None, *ordered]:
        if cooperate:
            cooperate()
        rid = "primary" if root is None else root.root_id
        label = 0 if root is None else by_id[rid][0]
        old = prior[rid]
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
                if root is None:
                    body, fit_report = _fit_primary_body(
                        support,
                        old,
                        spacing,
                        vertex_area_weights[indices][keep],
                    )
                else:
                    hint = old[0]
                    initial = _axis(support, graph[keep][:, keep], hint, spacing)
                    body, fit_report = _fit_body(support, initial, spacing)
                detail.update(fit_report)
                flags.extend(fit_report.get("qc_flags", []))
        if detail["status"] != "fitted":
            flags.append("centerline_" + detail["status"])
        staged[rid] = (body, support, detail, flags)

    curves, supports, reports = {}, {}, []
    resolved = {}
    primary_flags = []
    for root in [None, *ordered]:
        if cooperate:
            cooperate()
        rid = "primary" if root is None else root.root_id
        old = prior[rid]
        body, support, detail, flags = staged[rid]
        body_start = 0
        line = body
        insertion_index = None
        anchor = None
        if root is not None:
            parent = curves[root.parent_id]
            # Retain the repaired insertion neighbourhood, not an arbitrary
            # nearest turn of a curved or contacting parent.
            hint = next(value for child_id, value in child_hints[root.parent_id] if child_id == rid)
            allowed_parent_indices = _legal_attachment_indices(parent, top_reference, gravity_direction)
            if not len(allowed_parent_indices):
                raise ValueError(
                    f"{root.root_id}: parent has no insertion point at or below "
                    "the primary-root top"
                )
            insertion_index = int(
                allowed_parent_indices[
                    np.argmin(
                        np.linalg.norm(
                            parent[allowed_parent_indices] - hint,
                            axis=1,
                        )
                    )
                ]
            )
            anchor = parent[insertion_index]
            parent_support = supports[root.parent_id]
            gap = float(np.linalg.norm(body[0] - anchor))
            connector_ok = gap <= 1e-12
            connector_reason = "coincident" if connector_ok else "insufficient_parent_or_body_support"
            connector = np.vstack([anchor, body[0]])
            if len(parent_support) >= 8 and len(body) >= 2:
                # The parent can be sampled more coarsely than the child.
                # Use the analysis spacing at their shared junction.
                connector_spacing = d_bar
                connector_ok = _supported_connector(connector, parent_support, support, connector_spacing)
                connector_reason = "supported" if connector_ok else "owned_junction_support_rejected"
                if not connector_ok:
                    # Preserve a curved original junction only if every local
                    # section is supported after the final ownership changes.
                    end = int(cKDTree(old).query(body[0])[1])
                    curved = np.vstack([anchor, old[1:end], body[0]])
                    if len(curved) > 2 and _supported_connector(curved, parent_support, support, connector_spacing):
                        connector, connector_ok = curved, True
                        connector_reason = "curved_repaired_connector_supported"
            if connector_ok and gap > 1e-12 and len(body) >= 2:
                line = np.vstack([connector[:-1], body])
                body_start = len(connector) - 1
            elif not connector_ok:
                flags.append("centerline_unsupported_parent_connector")
            detail["parent_connector_mode"] = "supported" if connector_ok else "unsupported_gap"
            detail["parent_connector_supported"] = bool(connector_ok)
            detail["parent_connector_support_decision"] = connector_reason
            detail["parent_connector_length"] = path_length(connector) if body_start else 0.0
            detail["attachment_gap"] = 0.0 if connector_ok else gap

        # A fitted parent is only committed if its direct children can still
        # attach below the immutable top. A repaired prior is the conservative
        # rollback; a locally joined basal prefix retains the accepted body.
        required = child_hints.get(rid, [])
        allowed = _legal_attachment_indices(line, top_reference, gravity_direction)
        largest_shift = max(
            (
                float(np.min(np.linalg.norm(line[allowed] - child_hint, axis=1)))
                for _, child_hint in required
            ),
            default=0.0,
        ) if len(allowed) else np.inf
        if required and (not len(allowed) or root is not None and largest_shift > 4.0 * d_bar):
            preservation_reason = (
                "no_legal_below_top_point" if not len(allowed)
                else "descendant_attachment_shift_exceeds_4_spacings"
            )
            retained = None if root is None else _retained_attachment_prefix(
                old,
                body,
                support,
                [hint for _, hint in required],
                top_reference,
                gravity_direction,
                d_bar,
            )
            if retained is not None:
                line, body_start, join_gap = retained
                flags.append("centerline_preserved_topology_prefix")
                detail.update(
                    parent_connector_mode="preserved_topology",
                    parent_connector_supported=False,
                    parent_connector_rejection_reason=(
                        connector_reason if not connector_ok else "accepted_connector_replaced_for_descendant"
                    ),
                    attachment_preservation_reason=preservation_reason,
                    preserved_topology_length=path_length(line[:body_start + 1]),
                    preserved_topology_join_gap=join_gap,
                    preserved_topology_required_by=sorted(child_id for child_id, _ in required),
                    exposed_body_length=path_length(body),
                    exposed_body_supported=True,
                )
                detail["parent_connector_length"] = detail["preserved_topology_length"]
                detail["attachment_gap"] = float(np.linalg.norm(line[0] - anchor))
            else:
                line = old.copy()
                body_start = 0 if root is None else int(root.body_start_index)
                flags.append("centerline_attachment_preserving_prior_retained")
                retained_status = (
                    detail["status"]
                    if detail["status"] in {"no_support", "insufficient_support"}
                    else "attachment_preserving_prior_retained"
                )
                detail.update(
                    status=retained_status,
                    fit_applied=False,
                    traits_geometry_source="retained_prior",
                    parent_connector_mode="retained_prior" if root is not None else "none",
                    parent_connector_supported=False if root is not None else None,
                    parent_connector_rejection_reason=(
                        connector_reason if root is not None and not connector_ok
                        else "accepted_connector_replaced_for_descendant"
                        if root is not None else "none"
                    ),
                    attachment_preservation_reason=preservation_reason,
                    preserved_topology_required_by=sorted(child_id for child_id, _ in required),
                    exposed_body_supported=False,
                    rejected_fitted_body_length=path_length(body),
                    rejected_fitted_body_start_normalized=body[0].tolist(),
                    nearest_prior_to_fitted_body_start_gap=float(
                        np.min(np.linalg.norm(old - body[0], axis=1))
                    ),
                )
            allowed = _legal_attachment_indices(line, top_reference, gravity_direction)
            if not len(allowed):
                raise ValueError(f"{rid}: neither fitted nor repaired path preserves a legal descendant attachment")
        for child_id, child_hint in required:
            if not len(allowed):
                raise ValueError(f"{rid}: no legal attachment for {child_id}")
            nearest = float(np.min(np.linalg.norm(line[allowed] - child_hint, axis=1)))
            if nearest > 4.0 * d_bar:
                detail.setdefault("descendant_attachment_shift", {})[child_id] = nearest
                flags.append("centerline_descendant_attachment_shift")

        if root is not None:
            if child_length_exceeds_parent(path_length(line), path_length(parent)):
                flags.append("centerline_refit_child_longer_than_parent")
        else:
            primary_flags = flags
        detail.setdefault("fit_qc_passed", detail["status"] in {"fitted", "partial_support"})
        detail.setdefault("fit_applied", detail["fit_qc_passed"])
        detail.setdefault(
            "traits_geometry_source",
            "accepted_fit" if detail["fit_applied"] else "unmeasurable_support",
        )
        detail.setdefault("exposed_body_supported", detail["status"] in {"fitted", "partial_support"})
        detail.update(length_after=path_length(line), exposed_body_length=path_length(line[body_start:]), body_start_index=body_start,
                      tip_displacement=float(np.linalg.norm(line[-1] - old[-1])), qc_flags=flags)
        if root is not None:
            resolved[rid] = (line, body_start, insertion_index, anchor.copy(), parent, detail, flags)
        curves[rid], supports[rid] = line, support
        reports.append(detail)
    fitted_primary = curves["primary"]
    up = -gravity_direction
    fitted_primary_top_index = int(np.argmax(fitted_primary @ up))
    reference_top_index = int(np.argmax(top_reference @ up))
    for root in roots:
        line, body_start, insertion_index, anchor, parent, detail, flags = resolved[root.root_id]
        if not np.array_equal(anchor, parent[insertion_index]):
            raise AssertionError(f"{root.root_id}: stale resolved parent insertion")
        if is_above_primary_top(
            anchor,
            top_reference,
            gravity=gravity_direction,
        ):
            raise AssertionError(
                f"{root.root_id}: final insertion is above the primary-root top"
            )
        if not 0 <= body_start < len(line):
            raise AssertionError(f"{root.root_id}: invalid resolved exposed-body index")
    # A failed candidate leaves every input RootPath unchanged. Commit only
    # after the complete hierarchy passes the final attachment checks.
    for root in roots:
        line, body_start, insertion_index, anchor, parent, detail, flags = resolved[root.root_id]
        root.insertion_index = insertion_index
        root.insertion_point = anchor
        root.parent_points = parent
        root.points = line
        root.body_start_index = body_start
        root.node_indices = None
        root.start_index = None
        root.qc_flags = list(dict.fromkeys([*root.qc_flags, *flags]))
        root.centerline_assessment = detail
    return fitted_primary, {
        **POLICY,
        "primary_qc_flags": primary_flags,
        "primary_top_point_normalized": top_reference[reference_top_index].tolist(),
        "primary_top_reference_policy": "immutable_preprocessing_selection",
        "fitted_primary_top_point_normalized": fitted_primary[
            fitted_primary_top_index
        ].tolist(),
        "gravity_direction": gravity_direction.tolist(),
        "origins_above_primary_top": 0,
        "primary_fit_qc_passed": bool(reports[0].get("fit_qc_passed", False)),
        "primary_fit_applied": bool(reports[0].get("fit_applied", False)),
        "roots": reports,
    }
