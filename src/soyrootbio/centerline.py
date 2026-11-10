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

from .mesh_geometry import MeshGeometryContext, OwnershipGeometryGeneration
from .native_sections import recovered_wall_sections
from .measurement_evidence import (
    array_fingerprint, assess_origin_coverage, assess_terminal_direction,
    build_measurement_evidence, make_evidence_context,
)

from .geometry import (
    is_above_primary_top,
    path_length,
    primary_top_excess,
    resample_polyline,
    tangent_vectors,
    update_child_length_qc,
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
    "component_policy": "whole_prior_supported_body_identity_no_bridges",
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
    "lateral_method": "validated-prior-axis-regularized-candidates-v1",
    "lateral_station_spacing_floor": 2.0,
    "lateral_station_spacing_radius_fraction": 0.5,
    "lateral_minimum_section_fraction": 0.60,
    "lateral_minimum_surface_coverage": 0.80,
    "lateral_sampling_length_ratio_limit": 1.08,
    "lateral_length_inflation_limit": 1.15,
    "lateral_centering_improvement_margin": 0.025,
    "lateral_maximum_backtrack_fraction": 0.01,
    "lateral_maximum_center_error_radius_fraction": 0.35,
    "lateral_extent_common_body_fraction": 0.95,
    "lateral_extent_surface_coverage_margin": 0.05,
    "lateral_extent_transverse_tolerance_radius_fraction": 0.75,
    "lateral_extent_minimum_extension_spacing": 2.0,
    "lateral_extent_endpoint_tolerance_spacing": 1.0,
}


def _support_edges(points: np.ndarray, triangles: np.ndarray | None, spacing: float,
                   point_tree: cKDTree | None = None) -> np.ndarray:
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
    tree = point_tree if point_tree is not None else cKDTree(points)
    distances, neighbours = tree.query(points, k=min(9, len(points)))
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
    width_scale = float(POLICY["section_half_width_spacing"]) * spacing
    half_width = min(width_scale / (1 + width_scale * curvature),
                     float(POLICY["curvature_turn_limit_radians"]) / max(curvature, 1e-12))
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
        delta = float(POLICY["smoothing_neighbor_weight"]) * (line[:-2] + line[2:] - 2 * line[1:-1])
        limit = float(POLICY["smoothing_max_displacement_spacing"]) * np.asarray(widths)[run[1:-1]] / 2
        delta *= np.minimum(1, limit / np.maximum(np.linalg.norm(delta, axis=1), 1e-12))[:, None]
        line[1:-1] += delta
    return line, {"status": "fitted" if rejected == 0 and np.all(reliable) else "partial_support", "rejected_section_count": rejected,
                  "sparse_section_count": int(np.count_nonzero(valid & ~reliable)),
                  "excluded_section_count": int(len(valid) - len(run)),
                  "section_half_width_min": float(min(widths)), "section_half_width_max": float(max(widths))}


def _lateral_sections(points, line, spacing, weights, tree):
    """Independent radius-scale sections; never feed noisy station updates back."""
    length = path_length(line)
    pilot = resample_polyline(line, max(2 * spacing, length / 700.0))
    distances = np.asarray(tree.query(pilot, k=min(24, len(points)))[0])
    radius = max(2 * spacing, float(np.median(distances)))
    step = max(float(POLICY["lateral_station_spacing_floor"]) * spacing,
               float(POLICY["lateral_station_spacing_radius_fraction"]) * radius,
               length / 700.0)
    sampled = resample_polyline(line, step)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(sampled, axis=0), axis=1))]
    near = np.asarray(tree.query(sampled, k=min(24, len(points)))[0])
    radii = np.maximum(2 * spacing, np.median(near, axis=1))
    if len(radii) > 4:
        padded = np.pad(radii, 2, mode="edge")
        radii = np.array([np.median(padded[i:i+5]) for i in range(len(radii))])
    left = np.maximum(0., arc - np.maximum(3 * spacing, 2 * radii))
    right = np.minimum(arc[-1], arc + np.maximum(3 * spacing, 2 * radii))
    tangent = np.column_stack([np.interp(right, arc, sampled[:, j]) - np.interp(left, arc, sampled[:, j])
                               for j in range(3)])
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1), 1e-12)[:, None]
    assigned = cKDTree(sampled).query(points)[1]
    sections = [_primary_section(points, weights, tree, sampled, tangent, arc, assigned,
                                 radii, i, spacing, step) for i in range(len(sampled))]
    for i, section in enumerate(sections):
        if not section["accepted"] or section.get("thin_support"):
            continue
        # A one-sided wall can surround its own biased midpoint. Coverage must
        # also surround the independently measured section centre, not only the
        # candidate's current station. Do not manufacture an axis from a wall.
        local = np.asarray(tree.query_ball_point(sampled[i], max(3*radii[i], 6*spacing)), int)
        delta = points[local] - sampled[i]
        keep = ((np.abs(delta @ tangent[i]) <= section["half_width"])
                & (np.abs(arc[assigned[local]]-arc[i]) <= max(2*section["half_width"], 1.5*step)))
        transverse = (points[local[keep]] - section["candidate"]) @ section["basis"].T
        angles = np.sort(np.arctan2(transverse[:, 1], transverse[:, 0]))
        gap = float(np.max(np.diff(np.r_[angles, angles[0]+2*np.pi]))) if len(angles) else 2*np.pi
        section["measured_center_angular_coverage"] = 1-gap/(2*np.pi)
        if gap >= np.pi or 1-gap/(2*np.pi) < float(POLICY["primary_minimum_angular_coverage"]):
            section.update(accepted=False, rejection_reason="one_sided_about_measured_center")
    for i, section in enumerate(sections):
        if section["accepted"]:
            # Lateral sections supply transverse observations. Uneven sampling
            # along a sleeve must not move its stations towards a dense ring or
            # shorten both ends at every fitting pass.
            displacement = section["candidate"]-sampled[i]
            section["candidate"] = sampled[i]+displacement-np.dot(displacement, section["tangent"])*section["tangent"]
    return sampled, arc, sections, step


def _supported_lateral_reference(line, sampled, arc, sections, step):
    """Crop unsupported ends/gaps without replacing the reference's bends."""
    accepted = np.array([s["accepted"] for s in sections], bool)
    indices = np.flatnonzero(accepted)
    if len(indices) < 2:
        return None, None
    breaks = []
    for k, (a, b) in enumerate(zip(indices[:-1], indices[1:])):
        limit = max(3 * step, 2 * max(sections[a]["radius"], sections[b]["radius"]))
        if arc[b] - arc[a] > limit:
            breaks.append(k + 1)
    run = max(np.split(indices, breaks), key=lambda x: arc[x[-1]] - arc[x[0]])
    if len(run) < 2:
        return None, None
    a, b = int(run[0]), int(run[-1])
    # Sampled points lie on the original polyline; projection recovers exact
    # endpoints while retaining all original internal vertices, including bends.
    _, position = _polyline_projection(sampled[[a, b]], line)
    original_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    inside = (original_arc > position[0] + 1e-12) & (original_arc < position[1] - 1e-12)
    retained = np.vstack([sampled[a], line[inside], sampled[b]])
    retained = retained[np.r_[True, np.linalg.norm(np.diff(retained, axis=0), axis=1) > 1e-12]]
    return retained, (a, b + 1)


def _lateral_candidate_metrics(points, line, spacing, weights, tree):
    """Curve quality and whole selected-body coverage, independent of its source."""
    sampled, arc, sections, step = _lateral_sections(points, line, spacing, weights, tree)
    accepted = [i for i, s in enumerate(sections) if s["accepted"]]
    radii = np.array([s["radius"] for s in sections])
    scale = max(4 * spacing, float(np.median(radii)))
    coarse = resample_polyline(line, scale)
    length = path_length(line)
    ratio = length / max(path_length(coarse), 1e-12)
    _, progress = _polyline_projection(line, coarse)
    backward = np.diff(progress)
    backtrack = float(-np.sum(backward[backward < 0]))
    # Measure spatial coverage, not vertex-count coverage: dense mesh rings at
    # the base must not conceal a long, sparsely sampled distal body. Local
    # radii account for taper; a global median radius rejects thick basal walls.
    dense = resample_polyline(line, spacing)
    _, representatives = np.unique(np.floor((points-points.min(axis=0))/(2*spacing)).astype(np.int64),
                                   axis=0, return_index=True)
    spatial_points = points[representatives]
    distance, dense_index = cKDTree(dense).query(spatial_points)
    _, station_index = cKDTree(sampled).query(dense)
    # Nearest-wall distance describes the minor radius of a flattened root.
    # Use its measured transverse envelope for the major radius as well.
    envelope_radius = np.array([max(s["radius"], float(np.max(np.linalg.norm(s["envelope"], axis=1))))
                                if s.get("envelope") is not None else s["radius"] for s in sections])
    local_limit = np.maximum(1.25*envelope_radius[station_index[dense_index]], 4*spacing)
    surface_coverage = float(np.mean(distance <= local_limit))
    center_error = []
    for i in accepted:
        section = sections[i]
        shift = (section["candidate"]-sampled[i]) @ section["basis"].T
        magnitude = float(np.linalg.norm(shift))
        normalizing_radius = max(section["radius"], spacing)
        if section.get("envelope") is not None and magnitude > 1e-12:
            # The nearest wall is only a pilot/search radius. An off-centre
            # station or flattened shaft has a smaller nearest distance than
            # its measured cross-section. Use the observed half-width in the
            # displacement direction; retain the independent angular gates.
            projection = section["envelope"] @ (shift/magnitude)
            normalizing_radius = max(spacing, .5*float(np.ptp(projection)))
        center_error.append(magnitude/normalizing_radius)
    error = float(np.quantile(center_error, .75)) if center_error else None
    rejection = []
    section_fraction = len(accepted) / max(len(sections), 1)
    if section_fraction < float(POLICY["lateral_minimum_section_fraction"]):
        rejection.append("insufficient_surrounding_sections")
    if not sections[0]["accepted"] or not sections[-1]["accepted"]:
        rejection.append("unsupported_endpoint_section")
    for endpoint in (0, len(sampled)-1):
        section = sections[endpoint]
        if not section["accepted"]:
            continue
        local = np.asarray(tree.query_ball_point(sampled[endpoint], max(3*radii[endpoint], 6*spacing)), int)
        forward = (points[local]-sampled[endpoint]) @ section["tangent"]
        # A wide slab may see the first/last ring from empty space beyond it.
        # That supplies a transverse fit, but cannot certify an extrapolated
        # endpoint. Retain longitudinal locations only inside observed extent.
        if (len(forward) and ((endpoint == 0 and np.min(forward) > .5*spacing)
                             or (endpoint != 0 and np.max(forward) < -.5*spacing))):
            rejection.append("extrapolated_endpoint")
            break
    if surface_coverage < float(POLICY["lateral_minimum_surface_coverage"]):
        rejection.append("incomplete_selected_body_extent")
    if error is not None and error > float(POLICY["lateral_maximum_center_error_radius_fraction"]):
        rejection.append("poor_transverse_centering")
    if ratio > float(POLICY["lateral_sampling_length_ratio_limit"]) and length - path_length(coarse) > 2 * spacing:
        rejection.append("sampling_scale_oscillation")
    if backtrack > max(spacing, float(POLICY["lateral_maximum_backtrack_fraction"]) * length):
        rejection.append("longitudinal_backtracking")
    record = {"length_normalized": float(length), "section_fraction": section_fraction,
              "surface_coverage_fraction": surface_coverage, "centering_error_radius_fraction": error,
              "centering_scale": "measured_transverse_directional_half_width",
              "transverse_extent_radius_normalized": float(np.median(envelope_radius)),
              "sampling_scale_length_ratio": float(ratio), "backtrack_length_normalized": backtrack,
              "section_count": len(sections), "rejected_section_count": len(sections)-len(accepted),
              "rejections": rejection, "valid": not rejection,
              "score": error + 0.5 * max(0., ratio-1.) if error is not None else None}
    return record


def _assess_lateral_candidate(points, line, spacing, weights, tree):
    """Allow one local endpoint crop, then repeat every acceptance check.

    Transverse regularization changes the endpoint frame. A supported interior
    must not be lost solely because that new endpoint slab is one-sided. This
    is a bounded crop, never repeated shrinking or extrapolation to force a fit.
    """
    original = _lateral_candidate_metrics(points, line, spacing, weights, tree)
    endpoint_reasons = {"unsupported_endpoint_section", "extrapolated_endpoint"}
    if not endpoint_reasons.intersection(original["rejections"]):
        return line, original
    sampled, arc, sections, step = _lateral_sections(points, line, spacing, weights, tree)
    accepted = np.array([s["accepted"] for s in sections], bool)
    start_window = max(3*step, 2*sections[0]["radius"])
    end_window = max(3*step, 2*sections[-1]["radius"])
    for at_start in (True, False):
        window = start_window if at_start else end_window
        order = range(len(sampled)) if at_start else range(len(sampled)-1, -1, -1)
        for index in order:
            section = sections[index]
            end_distance = arc[index] if at_start else arc[-1]-arc[index]
            if end_distance > window:
                break
            if not accepted[index]:
                continue
            local = np.asarray(tree.query_ball_point(sampled[index], max(3*section["radius"], 6*spacing)), int)
            forward = (points[local]-sampled[index]) @ section["tangent"]
            if len(forward) and ((at_start and np.min(forward) > .5*spacing)
                                 or (not at_start and np.max(forward) < -.5*spacing)):
                accepted[index] = False
            else:
                break
    indices = np.flatnonzero(accepted)
    if len(indices) < 2:
        return line, original
    first, last = int(indices[0]), int(indices[-1])
    if arc[first] > start_window or arc[-1]-arc[last] > end_window:
        return line, original
    _, ends = _polyline_projection(sampled[[first, last]], line)
    original_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    inside = (original_arc > ends[0]+1e-12) & (original_arc < ends[1]-1e-12)
    cropped = np.vstack([sampled[first], line[inside], sampled[last]])
    if path_length(cropped) >= path_length(line)-1e-12:
        return line, original
    reassessed = _lateral_candidate_metrics(points, cropped, spacing, weights, tree)
    reassessed.update(endpoint_trim_applied=True,
                      endpoint_trim_removed_length_normalized=float(path_length(line)-path_length(cropped)),
                      endpoint_trim_start_limit_normalized=float(start_window),
                      endpoint_trim_end_limit_normalized=float(end_window),
                      pretrim_rejections=original["rejections"],
                      pretrim_length_normalized=float(path_length(line)))
    return cropped, reassessed


def _lateral_extent_preference(candidates, spacing):
    """Prefer an observed extension of the same body over a centred crop.

    Only independently admissible curves participate. More arc length alone is
    insufficient: the other curve must cover the common body and contain a
    sustained, supported extension beyond a candidate's endpoint. Partial bodies
    remain valid fallback options when no complete attachment-compatible body
    can be committed.
    """
    probes = {name: resample_polyline(line, max(2*spacing, path_length(line)/300.))
              for name, line, _ in candidates}
    for name, line, record in candidates:
        record["selection_extent_dominated_by"] = []
        record["selection_extent_deficit_normalized"] = 0.0
        length = path_length(line)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
        window = max(4*spacing, record["transverse_extent_radius_normalized"])
        first = np.array([np.interp(min(window, length), arc, line[:, j]) for j in range(3)])-line[0]
        last = line[-1]-np.array([np.interp(max(0., length-window), arc, line[:, j]) for j in range(3)])
        first /= max(np.linalg.norm(first), 1e-12)
        last /= max(np.linalg.norm(last), 1e-12)
        for other_name, other, other_record in candidates:
            if other_name == name or path_length(other) <= length+2*spacing:
                continue
            if other_record["surface_coverage_fraction"] < record["surface_coverage_fraction"]-float(POLICY["lateral_extent_surface_coverage_margin"]):
                continue
            tolerance = max(2*spacing, float(POLICY["lateral_extent_transverse_tolerance_radius_fraction"])
                            * min(record["transverse_extent_radius_normalized"], other_record["transverse_extent_radius_normalized"]))
            common_distance, _ = _polyline_projection(probes[name], other)
            if np.mean(common_distance <= tolerance) < float(POLICY["lateral_extent_common_body_fraction"]):
                continue
            endpoint_tolerance = float(POLICY["lateral_extent_endpoint_tolerance_spacing"])*spacing
            _, endpoint_position = _polyline_projection(line[[0, -1]], other)
            other_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(other, axis=0), axis=1))]
            nearest = np.column_stack([np.interp(endpoint_position, other_arc, other[:, j]) for j in range(3)])
            if (np.dot(line[0]-nearest[0], first) < -endpoint_tolerance
                    or np.dot(line[-1]-nearest[-1], last) > endpoint_tolerance):
                # Extending one end while losing the opposite end is not
                # containment, even when the shared middle exceeds 95%.
                continue
            sampled = probes[other_name]
            distance, position = _polyline_projection(sampled, line)
            outside = (distance > endpoint_tolerance) & (
                ((position <= endpoint_tolerance) & ((sampled-line[0]) @ first < -endpoint_tolerance))
                | ((position >= length-endpoint_tolerance) & ((sampled-line[-1]) @ last > endpoint_tolerance)))
            indices = np.flatnonzero(outside)
            runs = np.split(indices, np.flatnonzero(np.diff(indices)>1)+1)
            extension = max((path_length(sampled[run]) for run in runs if len(run)>1), default=0.)
            if extension+1e-12 < float(POLICY["lateral_extent_minimum_extension_spacing"])*spacing:
                continue
            record["selection_extent_dominated_by"].append(other_name)
            record["selection_extent_deficit_normalized"] = max(record["selection_extent_deficit_normalized"], float(extension))
    return [x for x in candidates if not x[2]["selection_extent_dominated_by"]]


def _fit_lateral_body(points, initial, prior, spacing, point_weights=None, *, candidate_output=None):
    """Select a validated prior/axis/regularized curve, preserving real support.

    These are geometry candidates, not hierarchy hypotheses. Length evidence
    only rejects a refinement; it never deletes a root or changes its parent.
    Existing attachment validation and measurement-evidence rebuilding still run.
    """
    points, prior = np.asarray(points, float), np.asarray(prior, float)
    weights = np.ones(len(points)) if point_weights is None else np.asarray(point_weights, float)
    tree = cKDTree(points)
    candidates = []; reports = []
    for name, reference in (("prior", prior), ("starting_axis", np.asarray(initial, float))):
        if len(reference) < 2 or path_length(reference) <= 1e-12:
            continue
        sampled, arc, sections, step = _lateral_sections(points, reference, spacing, weights, tree)
        cropped, interval = _supported_lateral_reference(reference, sampled, arc, sections, step)
        if cropped is None:
            reports.append({"name": name, "valid": False, "rejections": ["no_supported_contiguous_extent"]})
            continue
        reference_length = path_length(cropped)
        cropped, record = _assess_lateral_candidate(points, cropped, spacing, weights, tree)
        record.update(name=name, input_length_normalized=float(path_length(reference)),
                      supported_crop_length_normalized=float(path_length(cropped)))
        candidates.append((name, cropped, record));reports.append(record)
        a, b = interval
        refined = _regularized_primary_curve(sampled[a:b], sections[a:b], spacing)
        refined, refined_record = _assess_lateral_candidate(points, refined, spacing, weights, tree)
        refined_record.update(name="regularized_"+name, reference=name,
                              candidate_to_reference_length_ratio=path_length(refined)/max(reference_length,1e-12))
        scale = max(4*spacing, float(np.median([s["radius"] for s in sections[a:b]])))
        if (path_length(refined) > float(POLICY["lateral_length_inflation_limit"])*reference_length
                and path_length(refined)-reference_length > 2*scale):
            refined_record["rejections"].append("unexplained_refinement_length_inflation")
            refined_record["valid"] = False
        candidates.append(("regularized_"+name, refined, refined_record));reports.append(refined_record)
    admissible = [x for x in candidates if x[2]["valid"]]
    preferred_extent = _lateral_extent_preference(admissible, spacing)
    if candidate_output is not None:
        # Internal candidate geometry is kept out of exported scalar reports.
        # Family attachment validation can try another independently accepted
        # body before retaining an unmeasurable topology-only prior.
        candidate_output.extend(sorted(admissible, key=lambda x: (bool(x[2]["selection_extent_dominated_by"]),
                                                                  x[0] != "prior", x[2]["score"])))
    base = {"method": POLICY["lateral_method"], "lateral_candidate_assessments": reports,
            "candidate_metric_coordinate_unit": "analysis_normalized",
            "candidate_selection_rule": "admissible_supported_extent_dominance_then_centering_and_prior_tie"}
    if not admissible:
        return prior.copy(), {**base, "status": "qc_rejected_prior_retained", "fit_qc_passed": False,
            "fit_applied": False, "traits_geometry_source": "retained_prior", "exposed_body_supported": False,
            "selected_candidate": None, "rejected_section_count": 0,
            "qc_flags": ["centerline_lateral_no_admissible_candidate"]}
    winner = min(preferred_extent, key=lambda x: x[2]["score"])
    # A tiny centering improvement does not justify replacing an admissible
    # earlier curve. Tie preference is deterministic and independent of IDs.
    for preferred in ("prior", "starting_axis"):
        alternative = next((x for x in preferred_extent if x[0] == preferred), None)
        if alternative is not None and alternative[2]["score"] <= winner[2]["score"] + float(POLICY["lateral_centering_improvement_margin"]):
            winner = alternative
            break
    name, body, record = winner
    return body, {**base, "status": "fitted" if record["rejected_section_count"] == 0 else "partial_support",
        "fit_qc_passed": True, "fit_applied": True, "exposed_body_supported": True,
        "traits_geometry_source": "validated_prior" if name == "prior" else "accepted_fit",
        "selected_candidate": name, "rejected_section_count": record["rejected_section_count"],
        "qc_flags": []}


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


def _native_junction_footprint(points, labels, edges, parent_label, child_label,
                               hint, radius, spacing, top_reference, gravity):
    """Recover contact evidence across a deliberately unassigned junction cut.

    This certifies topology only. It neither assigns those vertices nor creates
    a centerline connector. Uncertain/foreign/above-top vertices are barriers.
    """
    limit = max(8*spacing, 5*radius)
    local = np.linalg.norm(points-hint,axis=1) <= limit
    local &= np.isin(labels, [parent_label, child_label, -1])
    up = -np.asarray(gravity,dtype=float)
    up /= max(np.linalg.norm(up),1e-12)
    local &= (points-np.asarray(top_reference).reshape(3)) @ up <= 1e-12
    vertices = np.flatnonzero(local)
    if not np.any(labels[vertices]==parent_label) or not np.any(labels[vertices]==child_label):
        return np.empty((0,3))
    native = edges[local[edges].all(axis=1)]
    links = np.searchsorted(vertices,native)
    length = np.linalg.norm(points[native[:,0]]-points[native[:,1]],axis=1)
    graph = coo_matrix((np.r_[length,length], (np.r_[links[:,0],links[:,1]],
        np.r_[links[:,1],links[:,0]])),shape=(len(vertices),len(vertices))).tocsr()
    distance = dijkstra(graph,directed=False,indices=np.flatnonzero(labels[vertices]==parent_label),
                        min_only=True,limit=max(4*spacing,3*radius))
    reached = (labels[vertices]==child_label) & np.isfinite(distance)
    return points[vertices[reached]]


def _retained_attachment_prefix(old, body, support, child_hints, top_reference, gravity_direction, spacing,
                                *, measured_radius=None):
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
    # A basal cross-section can move radially while the next few fitted
    # sections converge on the retained attachment. Search only a short local
    # overlap; joining a later turn would silently skip the supported body.
    old_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(old, axis=0), axis=1))]
    body_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(body, axis=0), axis=1))]
    nearest_start = last_required + 1 + int(np.argmin(
        np.linalg.norm(old[last_required + 1:] - body[0], axis=1)))
    # A recovered thick shaft may need several transverse radii to converge
    # after a basal branch junction. Bound the overlap independently of root
    # length, and retain the same small, surface-supported bridge limit.
    overlap = 8.0 * spacing
    if measured_radius is not None and np.isfinite(measured_radius):
        overlap = max(overlap, min(24.0 * spacing, 6.0 * measured_radius))
    old_indices = np.flatnonzero((np.arange(len(old)) > last_required)
                                & (np.abs(old_arc - old_arc[nearest_start]) <= overlap))
    body_indices = np.flatnonzero(body_arc <= overlap)
    join = None
    for body_index in body_indices:
        distances = np.linalg.norm(old[old_indices] - body[body_index], axis=1)
        for offset in np.argsort(distances, kind='stable'):
            if distances[offset] > 3.0 * spacing:
                break
            index = int(old_indices[offset])
            bridge = np.vstack((old[index], body[body_index]))
            if distances[offset] > 1e-10 and not _supported_connector(bridge, support, support, spacing):
                continue
            join = index, int(body_index), float(distances[offset])
            break
        if join is not None:
            break
    if join is None:
        return None
    join_index, body_index, join_gap = join
    body = body[body_index:]
    if len(body) < 2:
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


def _origin_supported_component(cloud, components, old, root, spacing, *,
                                reference=None, return_evidence=False):
    """Assess correspondence to the whole independent pre-fit body.

    Vertex population is deliberately not the ranking score. A sustained
    proximal tube can win over a larger stolen tail, but a few proximal wall
    vertices cannot displace a long supported body. Selection remains a partial
    component measurement whenever native ownership is disconnected.
    """
    sizes = np.bincount(components)
    largest = int(np.argmax(sizes))
    if root is None or len(sizes) == 1:
        result = (largest, "single_or_primary_component")
        return (*result, []) if return_evidence else result
    source = old
    if reference and reference.get("reference_valid"):
        source = np.asarray(reference["prior_path"], float)[int(reference.get("body_start_index", 0)):]
    step = max(2 * spacing, path_length(source) / 1000)
    line = resample_polyline(source, step)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    nearest_all = cKDTree(cloud).query(line)[0]
    radius = float(np.quantile(nearest_all, .35))
    expected = (reference or {}).get("mean_radius")
    if expected is not None and np.isfinite(expected) and expected > 0:
        radius = min(radius, float(expected))
    limit = max(4 * spacing, 1.6 * radius)
    evidence = []
    rows = []
    for component, size in enumerate(sizes):
        if size < 8:
            continue
        distance = cKDTree(cloud[components == component]).query(line)[0]
        supported = distance <= limit
        hits = np.flatnonzero(supported)
        interval = supported[:-1] & supported[1:]
        supported_length = float(np.sum(np.diff(arc)[interval]))
        runs = np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1)
        longest = max((float(arc[r[-1]] - arc[r[0]]) for r in runs if len(r)), default=0.)
        # Rooted continuity is independent evidence with a bounded benefit;
        # its influence vanishes for a short isolated basal patch.
        early = next((float(arc[r[-1]] - arc[r[0]]) for r in runs
                      if len(r) and arc[r[0]] <= max(4 * radius, 8 * spacing)), 0.)
        score = supported_length + .25 * min(early, max(4 * radius, 8 * spacing))
        qualifies = len(hits) >= 3 and longest >= max(4 * spacing, 1.5 * radius)
        rows.append({"component": component, "vertex_count": int(size),
                     "matched_prior_length": supported_length,
                     "longest_supported_prior_interval": longest,
                     "early_supported_prior_length": early,
                     "first_supported_prior_arc": float(arc[hits[0]]) if len(hits) else None,
                     "tube_distance_limit": limit, "score": score, "qualified": bool(qualifies)})
        if qualifies:
            evidence.append((score, early, -float(np.median(distance[hits])), -component, component))
    evidence.sort(reverse=True)
    if not evidence:
        result = (largest, "origin_component_unresolved")
    else:
        chosen = evidence[0][-1]
        ambiguous = (len(evidence) > 1 and evidence[1][0] >= .95 * evidence[0][0]
                     and abs(evidence[1][1] - evidence[0][1]) <= 2 * spacing)
        reason = ("origin_component_ambiguous" if ambiguous else
                  "early_body_supported_component" if evidence[0][1] >= max(4 * spacing, 1.5 * radius)
                  else "whole_prior_supported_component")
        result = chosen, reason
    return (*result, rows) if return_evidence else result


def _identity_reference(root, old):
    existing = (root.centerline_assessment or {}) if root is not None else {}
    reference = existing.get("prefit_identity_evidence")
    if isinstance(reference, dict):
        return reference
    # A legacy fitted path is not an independent pre-fit identity reference.
    already_fitted = bool(existing.get("status") or existing.get("traits_geometry_source"))
    return {"policy": "pre-fit-exposed-body-v1", "prior_path": [] if already_fitted else old.tolist(),
            "body_start_index": int(root.body_start_index) if root is not None else 0,
            "mean_radius": root.mean_radius if root is not None else None,
            "coordinate_unit": "analysis_normalized", "path_fingerprint": array_fingerprint(old),
            "origin": (np.asarray(root.insertion_point).tolist() if root is not None and root.insertion_point is not None else old[0].tolist()),
            "reference_valid": not already_fitted}


def _verified_terminal_support(points, triangles, ids, line, reference, spacing, context):
    """Conservative observed native terminal, not an arbitrary fitted endpoint."""
    if triangles is None or not len(triangles) or len(line) < 2 or len(ids) < 8 or not reference.get("reference_valid"):
        return False
    if not assess_terminal_direction(line, spacing=spacing)["terminal_direction_supported"]:
        return False
    prior = np.asarray(reference["prior_path"], float)
    if not len(prior) or np.linalg.norm(line[-1] - prior[-1]) > 4 * spacing:
        return False
    tangent = line[-1] - line[-2]
    tangent /= np.linalg.norm(tangent)
    surface = points[ids]
    delta = surface - line[-1]
    forward = delta @ tangent
    radial = np.linalg.norm(delta - forward[:, None] * tangent, axis=1)
    local_radius = max(spacing, float(np.quantile(np.linalg.norm(delta, axis=1), .05)))
    local = np.linalg.norm(delta, axis=1) <= max(8 * spacing, 2 * local_radius)
    if np.count_nonzero(local) < 3 or np.any((forward > 3 * spacing) & (radial < max(4 * spacing, 2 * local_radius))):
        return False
    graph = context.get("_native_full_graph")
    if graph is not None:
        terminal_ids = ids[local & (forward >= -max(4 * spacing, local_radius))]
        owned = np.zeros(len(points), bool)
        owned[ids] = True
        # Edges are stored once in index order, so inspect both orientations.
        adjacency = graph[terminal_ids].tocoo()
        reverse = context["_native_full_graph_reverse"][terminal_ids].tocoo()
        first = np.r_[terminal_ids[adjacency.row], terminal_ids[reverse.row]]
        second = np.r_[adjacency.col, reverse.col]
        crosses = ~owned[second]
        if np.any(crosses):
            vector = points[second[crosses]] - points[first[crosses]]
            continuation = vector @ tangent > .25 * np.linalg.norm(vector, axis=1)
            distal = (points[second[crosses]] - line[-1]) @ tangent >= -2 * spacing
            if np.any(continuation & distal):
                return False
    # An open scan boundary is not a biological terminal. Identity and current
    # ownership are independently checked by the caller.
    return not bool(np.any(context["_native_boundary_vertices"][ids[local]]))


def refit_final_centerlines(
    points: np.ndarray, labels: np.ndarray, primary: np.ndarray,
    roots: list[RootPath], *, d_bar: float, triangles: np.ndarray | None = None,
    gravity: np.ndarray | tuple[float, float, float] = (0.0, 0.0, -1.0),
    primary_top_reference: np.ndarray | None = None,
    cooperate: Callable[[], None] | None = None,
    mesh_context: MeshGeometryContext | None = None,
    nodule_labels: np.ndarray | None = None,
    preserved_bodies: dict[str, dict] | None = None,
    identity_references: dict[str, dict] | None = None,
) -> tuple[np.ndarray, dict]:
    """Refit all roots in parent-first order using full-resolution final labels.

    Labels use 0 for primary and one-based positions in ``roots``. The list is
    not reordered. Reports and body indices are saved on RootPath for export.
    An unsupported attachment keeps its topology metadata but no spatial bridge.
    """
    points, labels = np.asarray(points, dtype=float), np.asarray(labels)
    if nodule_labels is not None and np.asarray(nodule_labels).shape != labels.shape:
        raise ValueError("nodule labels must match final-support vertices")
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
    if mesh_context is not None and triangles is not None and len(triangles):
        mesh_context.validate(points, triangles)
        edges = mesh_context.support_edges(d_bar)
        vertex_area_weights = mesh_context.vertex_area_weights
    else:
        if mesh_context is not None:
            mesh_context.validate(points, triangles)
        edges = _support_edges(points, triangles, d_bar,
                               point_tree=mesh_context.point_tree if mesh_context is not None else None)
        vertex_area_weights = _vertex_area_weights(points, triangles)
    edges.setflags(write=False)
    measurement_context = make_evidence_context(points, triangles)
    ownership = (mesh_context.ownership(labels) if mesh_context is not None
                 else OwnershipGeometryGeneration(labels, points))
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
    original_child_hints = {parent_id: list(hints) for parent_id, hints in child_hints.items()}
    staged = {}
    staged_support_indices = {}
    staged_lateral_candidates = {}
    for root in [None, *ordered]:
        if cooperate:
            cooperate()
        rid = "primary" if root is None else root.root_id
        label = 0 if root is None else by_id[rid][0]
        old = prior[rid]
        indices = ownership.vertices(label)
        flags = []
        detail = {"root_id": rid, "numeric_label": label, "assigned_point_count": int(len(indices)), "length_before": path_length(old),
                  "coordinate_unit": "normalized", "input_geometry_fingerprint": hashlib.sha256(np.round(np.asarray(old, dtype=np.float64), decimals=8).tobytes(order="C")).hexdigest()[:20]}
        reference = (identity_references or {}).get(rid) or _identity_reference(root, old)
        if root is None and (preserved_bodies or {}).get("primary"):
            reference = preserved_bodies["primary"].get("assessment", {}).get("prefit_identity_evidence", reference)
        detail["prefit_identity_evidence"] = reference
        if not reference.get("reference_valid"):
            flags.append("centerline_identity_reference_unavailable")
        if not len(indices):
            # Preserve the record for label mapping, but export no line segment.
            body = np.asarray(old[:1], dtype=float).copy()
            support = np.empty((0, 3))
            staged_support_indices[rid] = np.empty(0, int)
            detail.update(status="no_support", component_count=0, excluded_fragment_points=0)
        else:
            cloud = points[indices]
            graph = ownership.local_graph(label, edges)
            count, components = connected_components(graph, directed=False)
            sizes = np.bincount(components)
            selected_component, component_decision, component_evidence = _origin_supported_component(
                cloud, components, old, root, d_bar, reference=reference, return_evidence=True)
            keep = components == selected_component
            detail.update(component_selection=component_decision,
                          selected_component=int(selected_component),
                          largest_component=int(np.argmax(sizes)),
                          component_identity_evidence=component_evidence)
            support = cloud[keep]
            staged_support_indices[rid] = indices[keep]
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
            preserved = (preserved_bodies or {}).get(rid)
            if preserved is not None:
                # A scoped ownership correction need not smooth unrelated
                # bodies again. Reuse only an exact owned-vertex snapshot;
                # attachments are still checked against the current parents.
                if not np.array_equal(indices, preserved["vertices"]):
                    raise ValueError(f"{rid}: preserved body ownership changed")
                if not np.array_equal(old, preserved["points"]):
                    raise ValueError(f"{rid}: preserved body geometry changed")
                dynamic = dict(detail)
                detail.update(preserved.get("assessment", {}))
                detail.update(dynamic)
                body = old[int(preserved["body_start_index"]):].copy()
                detail["body_fit_reused"] = True
                flags.extend(preserved.get("assessment", {}).get("qc_flags", []))
            elif len(support) < 8:
                body = np.median(support, axis=0)[None, :]
                detail["status"] = "insufficient_support"
            else:
                if root is not None and root.score_components.get("native_recovered_extension", 0) > 0 and triangles is not None and len(triangles):
                    section_context = mesh_context or MeshGeometryContext.build(points, triangles)
                    _, _, initial, measured = recovered_wall_sections(points, labels, old, label,
                        section_context, spacing, labels != label)
                    # Missing contour stations at child junctions must not
                    # alternate between a measured centre and an off-centre
                    # prior. This is only a pilot: the subsequent section fit
                    # independently requires final owned surface throughout.
                    measured_indices = np.flatnonzero(measured)
                    for first, last in zip(measured_indices[:-1], measured_indices[1:]):
                        if last - first <= 12:  # at most 24 full-resolution spacings
                            initial[first:last+1] = np.linspace(initial[first], initial[last], last-first+1)
                    for _ in range(2):
                        initial[1:-1] = (initial[:-2] + 2*initial[1:-1] + initial[2:])/4
                    body, fit_report = _fit_primary_body(support, initial, spacing,
                        vertex_area_weights[indices][keep])
                    if not fit_report.get('fit_applied', False):
                        body = old.copy()
                    fit_report['method'] = 'native-contour-guided-recovered-body'
                    fit_report['native_measured_section_count'] = int(measured.sum())
                    fit_report['qc_flags'] = [flag.replace('centerline_primary_', 'centerline_recovered_')
                                             for flag in fit_report.get('qc_flags', [])]
                elif root is None:
                    body, fit_report = _fit_primary_body(
                        support,
                        old,
                        spacing,
                        vertex_area_weights[indices][keep],
                    )
                else:
                    hint = old[0]
                    initial = _axis(support, graph[keep][:, keep], hint, spacing)
                    reference_body = np.asarray(reference.get("prior_path", old), float)
                    reference_start = int(reference.get("body_start_index", root.body_start_index))
                    staged_lateral_candidates[rid] = []
                    body, fit_report = _fit_lateral_body(
                        support, initial, reference_body[reference_start:], spacing,
                        vertex_area_weights[indices][keep], candidate_output=staged_lateral_candidates[rid])
                detail.update(fit_report)
                flags.extend(fit_report.get("qc_flags", []))
            if root is not None and count > 1:
                # A good local fit does not certify a disconnected identity.
                detail["identity_qc_passed"] = False
                detail["component_measurement_only"] = True
                flags.append("centerline_identity_unresolved")
                if component_decision.startswith("origin_component_"):
                    flags.append("centerline_origin_component_unresolved")
                    detail.update(fit_qc_passed=False, exposed_body_supported=False)
                elif selected_component != int(np.argmax(sizes)):
                    flags.append("centerline_disconnected_distal_identity_review")
        if nodule_labels is not None and len(indices) and detail.get("component_count", 0) > 1:
            native_nodules = np.asarray(nodule_labels)
            crossing = edges[((labels[edges[:, 0]] == label) & (native_nodules[edges[:, 1]] <= -3))
                             | ((labels[edges[:, 1]] == label) & (native_nodules[edges[:, 0]] <= -3))]
            obscured = False
            if len(crossing):
                root_vertex = np.where(labels[crossing[:, 0]] == label, crossing[:, 0], crossing[:, 1])
                nodule_vertex = np.where(labels[crossing[:, 0]] == label, crossing[:, 1], crossing[:, 0])
                local_component = components[np.searchsorted(indices, root_vertex)]
                for nodule_id in np.unique(native_nodules[nodule_vertex]):
                    if len(np.unique(local_component[native_nodules[nodule_vertex] == nodule_id])) > 1:
                        obscured = True
                        break
            if obscured:
                body, support = old.copy(), points[indices]
                detail.update(status="nodule_obscured", fit_applied=False, fit_qc_passed=False,
                              traits_geometry_source="retained_prior_nodule_obscured",
                              exposed_body_supported=False,
                              nodule_gap_policy="retain_existing_topology_without_measuring_hidden_root")
        if detail["status"] != "fitted":
            flags.append("centerline_" + detail["status"])
        staged[rid] = (body, support, detail, flags)

    # A late recovered shaft can have an off-centre prior. Resolve its child
    # origins against frozen native junction footprints before the parent
    # family is committed; stale radial hint positions are not immutable tops.
    if triangles is not None and len(triangles):
        for parent_root in ordered:
            parent_id = parent_root.root_id
            if parent_root.score_components.get("native_recovered_extension", 0) <= 0:
                continue
            parent_body, parent_surface, parent_detail, _ = staged[parent_id]
            if not parent_detail.get("fit_applied", False) or len(parent_surface) < 8:
                continue
            parent_tree = cKDTree(parent_surface)
            revised = []
            for child_id, hint in child_hints.get(parent_id, []):
                nearest = int(np.argmin(np.linalg.norm(parent_body-hint, axis=1)))
                radius = max(d_bar, float(np.median(parent_tree.query(parent_body[nearest], k=16)[0])))
                # Oblique emergence has a longitudinal footprint several
                # parent radii long; use the attachment survey's five-radius
                # neighbourhood, still requiring observed native contact.
                limit = max(8*d_bar, 5*radius)
                boundary = ownership.boundary_edges(by_id[parent_id][0], by_id[child_id][0], edges)
                contacts = points[np.unique(boundary)]
                contact_mode = "direct_native_contact"
                if not len(contacts):
                    contacts = _native_junction_footprint(points, labels, edges,
                        by_id[parent_id][0], by_id[child_id][0], hint, radius, d_bar,
                        top_reference, gravity_direction)
                    contact_mode = "bounded_native_path_across_unassigned_cut"
                replacement = None
                connector_supported = False
                if len(contacts):
                    candidates = _legal_attachment_indices(parent_body, top_reference, gravity_direction)
                    candidates = candidates[np.linalg.norm(parent_body[candidates]-hint,axis=1)<=limit]
                    near_contact = cKDTree(contacts).query(parent_body[candidates])[0]<=limit
                    if not np.any(near_contact) and contact_mode == "direct_native_contact":
                        contacts = _native_junction_footprint(points, labels, edges,
                            by_id[parent_id][0], by_id[child_id][0], hint, radius, d_bar,
                            top_reference, gravity_direction)
                        contact_mode = "bounded_native_path_across_unassigned_cut"
                        near_contact = (cKDTree(contacts).query(parent_body[candidates])[0]<=limit
                                        if len(contacts) else np.zeros(len(candidates),bool))
                    candidates = candidates[near_contact]
                    child_body, child_surface, _, _ = staged[child_id]
                    for index in candidates[np.argsort(np.linalg.norm(parent_body[candidates]-child_body[0],axis=1),kind='stable')]:
                        if _supported_connector(np.vstack((parent_body[index],child_body[0])),
                                                parent_surface,child_surface,d_bar):
                            replacement = parent_body[index].copy()
                            connector_supported = True
                            break
                    if replacement is None and len(candidates):
                        # Native contact certifies the parent relationship.
                        # A missing straight internal connector is reported on
                        # the child below; it must not displace an independently
                        # supported parent body back onto its old surface wall.
                        index = candidates[int(np.argmin(np.linalg.norm(parent_body[candidates]-child_body[0],axis=1)))]
                        replacement = parent_body[index].copy()
                revised.append((child_id, hint if replacement is None else replacement))
                if replacement is not None:
                    parent_detail.setdefault("native_recentered_child_origins", {})[child_id] = {
                        "prior_hint": hint.tolist(), "resolved_hint": replacement.tolist(),
                        "displacement": float(np.linalg.norm(replacement-hint)),
                        "native_contact_vertex_count": int(len(contacts)),
                        "straight_connector_supported": connector_supported,
                        "native_contact_mode": contact_mode,
                    }
            child_hints[parent_id] = revised

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
        shifted = [
            (child_id, hint) for child_id, hint in required
            if not len(allowed) or np.min(np.linalg.norm(line[allowed] - hint, axis=1)) > 4.0 * d_bar
        ]
        largest_shift = max(
            (float(np.min(np.linalg.norm(line[allowed] - hint, axis=1))) for _, hint in shifted),
            default=0.0,
        ) if len(allowed) else np.inf
        if (root is not None and detail.get("fit_qc_passed") and required
                and (not len(allowed) or largest_shift > 4.0*d_bar)):
            for candidate_name, candidate_body, candidate_record in staged_lateral_candidates.get(rid, []):
                if candidate_name == detail.get("selected_candidate"):
                    continue
                candidate_line = candidate_body
                candidate_start = 0
                candidate_connector = np.vstack([anchor, candidate_body[0]])
                candidate_gap = float(np.linalg.norm(anchor-candidate_body[0]))
                candidate_connector_ok = candidate_gap <= 1e-12
                candidate_reason = "coincident" if candidate_connector_ok else "owned_junction_support_rejected"
                if len(parent_support) >= 8 and len(candidate_body) >= 2:
                    candidate_connector_ok = _supported_connector(candidate_connector, parent_support, support, d_bar)
                    if candidate_connector_ok:
                        candidate_reason = "supported"
                    else:
                        end = int(cKDTree(old).query(candidate_body[0])[1])
                        curved = np.vstack([anchor, old[1:end], candidate_body[0]])
                        if len(curved) > 2 and _supported_connector(curved, parent_support, support, d_bar):
                            candidate_connector, candidate_connector_ok = curved, True
                            candidate_reason = "curved_repaired_connector_supported"
                if candidate_connector_ok and candidate_gap > 1e-12:
                    candidate_line = np.vstack([candidate_connector[:-1], candidate_body])
                    candidate_start = len(candidate_connector)-1
                candidate_allowed = _legal_attachment_indices(candidate_line, top_reference, gravity_direction)
                candidate_shifted = [(child_id, hint) for child_id, hint in required
                    if not len(candidate_allowed) or np.min(np.linalg.norm(candidate_line[candidate_allowed]-hint, axis=1)) > 4.0*d_bar]
                candidate_prefix = None
                if candidate_shifted:
                    candidate_prefix = _retained_attachment_prefix(
                        old, candidate_body, support, [hint for _, hint in candidate_shifted],
                        top_reference, gravity_direction, d_bar)
                    if candidate_prefix is None:
                        continue
                    candidate_line, candidate_start, candidate_join_gap = candidate_prefix
                    candidate_allowed = _legal_attachment_indices(candidate_line, top_reference, gravity_direction)
                    if not len(candidate_allowed) or any(
                        np.min(np.linalg.norm(candidate_line[candidate_allowed]-hint, axis=1)) > 4.0*d_bar
                        for _, hint in required
                    ):
                        continue
                previous_candidate = detail.get("selected_candidate")
                body, line, body_start = candidate_body, candidate_line, candidate_start
                connector, connector_ok, connector_reason = candidate_connector, candidate_connector_ok, candidate_reason
                gap = candidate_gap
                allowed, shifted, largest_shift = candidate_allowed, [], 0.0
                flags[:] = [f for f in flags if f != "centerline_unsupported_parent_connector"]
                if not connector_ok:
                    flags.append("centerline_unsupported_parent_connector")
                detail.update(
                    selected_candidate=candidate_name,
                    attachment_candidate_fallback_from=previous_candidate,
                    attachment_candidate_selection="validated_candidate_preserves_descendant_attachments",
                    status="fitted" if candidate_record["rejected_section_count"] == 0 else "partial_support",
                    fit_applied=True, fit_qc_passed=True, exposed_body_supported=True,
                    traits_geometry_source="validated_prior" if candidate_name == "prior" else "accepted_fit",
                    rejected_section_count=candidate_record["rejected_section_count"],
                    parent_connector_mode="supported" if connector_ok else "unsupported_gap",
                    parent_connector_supported=bool(connector_ok), parent_connector_support_decision=connector_reason,
                    parent_connector_length=path_length(connector) if body_start else 0.0,
                    attachment_gap=0.0 if connector_ok else gap,
                )
                if candidate_prefix is not None:
                    flags.append("centerline_preserved_topology_prefix")
                    detail.update(
                        parent_connector_mode="preserved_topology", parent_connector_supported=False,
                        parent_connector_rejection_reason=(connector_reason if not connector_ok
                                                           else "accepted_connector_replaced_for_descendant"),
                        attachment_preservation_reason="descendant_attachment_shift_exceeds_4_spacings",
                        preserved_topology_length=path_length(line[:body_start+1]),
                        preserved_topology_join_gap=candidate_join_gap,
                        preserved_topology_required_by=sorted(child_id for child_id, _ in candidate_shifted),
                        attachment_candidate_fallback_prefix=True,
                        parent_connector_length=path_length(line[:body_start+1]),
                        attachment_gap=float(np.linalg.norm(line[0]-anchor)),
                    )
                break
        if required and (not len(allowed) or root is not None and largest_shift > 4.0 * d_bar):
            preservation_reason = (
                "no_legal_below_top_point" if not len(allowed)
                else "descendant_attachment_shift_exceeds_4_spacings"
            )
            retained = None if root is None else _retained_attachment_prefix(
                old,
                body,
                support,
                [hint for _, hint in shifted],
                top_reference,
                gravity_direction,
                d_bar,
                measured_radius=(detail.get("section_radius_median")
                                 if detail.get("method") == "native-contour-guided-recovered-body" else None),
            )
            if retained is not None:
                candidate_allowed = _legal_attachment_indices(retained[0], top_reference, gravity_direction)
                if not len(candidate_allowed) or any(
                    np.min(np.linalg.norm(retained[0][candidate_allowed] - hint, axis=1)) > 4.0 * d_bar
                    for child_id, hint in required
                ):
                    retained = None
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
                    preserved_topology_required_by=sorted(child_id for child_id, _ in shifted),
                    exposed_body_length=path_length(body),
                    exposed_body_supported=True,
                )
                detail["parent_connector_length"] = detail["preserved_topology_length"]
                detail["attachment_gap"] = float(np.linalg.norm(line[0] - anchor))
            else:
                line = old.copy()
                required = original_child_hints.get(rid, [])
                child_hints[rid] = required
                if "native_recentered_child_origins" in detail:
                    detail["rejected_native_child_origin_proposals"] = detail.pop("native_recentered_child_origins")
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
                    rejected_fitted_body_qc_passed=bool(detail.get("fit_qc_passed")),
                    fit_qc_passed=False,
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

        preserved = (preserved_bodies or {}).get(rid)
        # If this parent also retained its body, and its stored insertion is
        # still exactly on that parent, keep the existing supported connector
        # byte-for-byte. Changed parents use the attachment checks above.
        if preserved is not None and (root is None or root.parent_id in (preserved_bodies or {})):
            if root is None or (root.insertion_index is not None
                    and root.insertion_index < len(curves[root.parent_id])
                    and root.insertion_point is not None
                    and np.array_equal(root.insertion_point, curves[root.parent_id][root.insertion_index])):
                line = old.copy()
                body_start = int(preserved["body_start_index"])
                if root is not None:
                    insertion_index = root.insertion_index
                    anchor = root.insertion_point.copy()
                    for key in ("parent_connector_mode", "parent_connector_supported",
                                "parent_connector_support_decision", "parent_connector_length", "attachment_gap"):
                        if key in preserved.get("assessment", {}):
                            detail[key] = preserved["assessment"][key]
        if root is not None:
            flags = [flag for flag in flags if flag != "centerline_refit_child_longer_than_parent"]
            if update_child_length_qc(flags, path_length(line), path_length(parent)):
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
        identity_resolved = bool(detail.get("component_count", 0) == 1
                                 and detail["prefit_identity_evidence"].get("reference_valid")
                                 and detail.get("fit_qc_passed")
                                 and detail.get("exposed_body_supported")
                                 and not (root is not None and "native_body_identity_unresolved" in root.qc_flags))
        detail["identity_qc_passed"] = identity_resolved
        support_ids = staged_support_indices[rid]
        tip_verified = bool(identity_resolved and _verified_terminal_support(
            points, triangles, support_ids, line[body_start:],
            detail["prefit_identity_evidence"], d_bar, measurement_context))
        detail["biological_tip_verified"] = tip_verified
        reference = detail["prefit_identity_evidence"]
        origin_evidence = assess_origin_coverage(
            line, top_reference[int(np.argmax(top_reference @ -gravity_direction))] if root is None else anchor,
            reference_path=reference.get("prior_path") if reference.get("reference_valid") else None,
            spacing=detail.get("fit_spacing", d_bar),
            attachment_supported=root is None or detail.get("parent_connector_supported", False))
        detail["origin_coverage_complete"] = origin_evidence["origin_coverage_complete"]
        detail["origin_coverage_evidence"] = origin_evidence
        detail["measurement_evidence"] = build_measurement_evidence(
            points, labels, 0 if root is None else by_id[rid][0], line,
            triangles=triangles, selected_vertex_indices=support_ids,
            spacing=detail.get("fit_spacing", d_bar), body_start_index=body_start,
            identity_resolved=identity_resolved,
            attachment_supported=root is None or detail.get("parent_connector_supported", False),
            tip_supported=tip_verified,
            origin_coverage_complete=origin_evidence["origin_coverage_complete"],
            origin_coverage_evidence=origin_evidence, context=measurement_context)
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
