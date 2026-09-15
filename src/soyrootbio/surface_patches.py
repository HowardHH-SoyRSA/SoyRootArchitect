"""Symmetric correction of assigned surface components on an observed mesh."""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .types import RootPath


def correct_surface_patches(
    points: np.ndarray,
    labels: np.ndarray,
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    *,
    d_bar: float,
    triangles: np.ndarray | None = None,
    excluded_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Minimize a symmetric component energy using observed mesh support.

    Initial same-label components remain atomic during optimization: merging
    two corrected patches must not erase the evidence needed to reconsider
    them. Fixed geometry/radii and symmetric boundary costs make each accepted
    move strictly decrease one finite-state energy. Sweeps stop only when no
    admissible move improves it. This is a component-local minimum, not a claim
    of globally optimal or anatomically certain segmentation.

    No root order, main-component exemption, primary mask, or kNN connectivity
    is used. Negative/excluded labels are barriers. Root paths are references;
    this stage neither fills gaps nor edits geometry or topology.
    """
    source = np.asarray(points, dtype=float)
    before = np.asarray(labels, dtype=int)
    spacing = float(d_bar)
    paths = [np.asarray(primary_path, dtype=float)] + [
        np.asarray(p.points, dtype=float) for p in lateral_paths
    ]
    if source.ndim != 2 or source.shape[1] != 3 or not np.all(np.isfinite(source)):
        raise ValueError("points must contain finite XYZ coordinates")
    if before.shape != (len(source),):
        raise ValueError("labels must contain one value per point")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("d_bar must be positive and finite")
    for path in paths:
        if path.ndim != 2 or path.shape[1] != 3 or not np.all(np.isfinite(path)):
            raise ValueError("root paths must contain finite XYZ coordinates")
    # Accepted internal connectors are attachment geometry, not exposed body
    # evidence. Respect this existing distinction at every lateral order.
    for label, root in enumerate(lateral_paths, 1):
        start = int(root.body_start_index)
        if start < 0 or (len(paths[label]) and start >= len(paths[label])):
            raise ValueError("body_start_index must reference the root path")
        paths[label] = paths[label][start:]
    if np.any(before >= len(paths)):
        raise ValueError("assigned labels must reference a root path")
    excluded = (np.zeros(len(source), dtype=bool) if excluded_mask is None
                else np.asarray(excluded_mask, dtype=bool))
    if excluded.shape != before.shape:
        raise ValueError("excluded_mask must contain one value per point")
    result = before.copy()
    report = {
        "policy": "symmetric-mesh-surface-patches-v1",
        "connectivity": "triangle_edges",
        "status": "stable",
        "converged": True,
        "component_partition": "initial same-label mesh components retained as atomic patches",
        "tie_rule": "retain current owner within 1e-9 per vertex; visit patches by minimum vertex index",
        "radius_estimator": "median segment distances of competitively supported owned points in 4*d_bar bins",
        "score_weights": {"segment_distance": 1.0, "radius_residual": 0.25,
                          "direction": 0.6, "body_connection": 0.4, "boundary": 0.35},
        "size_rule": "unary cost times vertex count; incident boundary cost capped at 0.35 times vertex count",
        "support_rule": "every patch vertex within 1.75*local_radius + 1.5*d_bar; axial alignment >= 0.5 when measurable",
        "body_rule": "mesh path through geometrically supported patches to an originally owned, axially supported body patch",
        "internal_connectors": "excluded using body_start_index at every lateral order",
        "maximum_mesh_edge_length": 4.0 * spacing,
        "reassigned_patch_count": 0,
        "reassigned_vertex_count": 0,
        "passes": 0,
        "moves": [],
        "components": [],
    }
    faces = (np.empty((0, 3), dtype=np.int64) if triangles is None
             else np.asarray(triangles, dtype=np.int64))
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("triangles must have shape (m, 3)")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(source)):
        raise ValueError("triangles contain an out-of-range vertex index")
    if not len(faces):
        report.update(connectivity="no_mesh", status="skipped_no_mesh")
        return result, report
    edges = np.unique(np.sort(np.vstack([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
    ]), axis=1), axis=0)
    # Even a malformed triangle spanning a scan gap must not provide a bridge.
    lengths = np.linalg.norm(source[edges[:, 0]] - source[edges[:, 1]], axis=1)
    edges = edges[(edges[:, 0] != edges[:, 1]) & (lengths <= 4.0 * spacing)]
    assigned = (before >= 0) & ~excluded
    edges = edges[assigned[edges[:, 0]] & assigned[edges[:, 1]]]
    same = edges[before[edges[:, 0]] == before[edges[:, 1]]]
    graph = coo_matrix((np.ones(len(same)), (same[:, 0], same[:, 1])),
                       shape=(len(source), len(source))).tocsr()
    _, partition = connected_components(graph, directed=False)
    ids = np.flatnonzero(assigned)
    if not len(ids):
        return result, report
    _, compact = np.unique(partition[ids], return_inverse=True)
    count = int(compact.max()) + 1
    by_vertex = np.full(len(source), -1, dtype=int)
    by_vertex[ids] = compact
    grouped = ids[np.argsort(compact, kind="stable")]
    members = np.split(grouped, np.cumsum(np.bincount(compact))[:-1])
    owners = np.array([before[ix[0]] for ix in members])
    original = owners.copy()
    sizes = np.array([len(ix) for ix in members])
    pairs = by_vertex[edges]
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    pairs, contacts = np.unique(np.sort(pairs, axis=1), axis=0, return_counts=True)
    neighbors: list[dict[int, int]] = [{} for _ in members]
    for (a, b), contact in zip(pairs, contacts):
        neighbors[a][int(b)] = int(contact)
        neighbors[b][int(a)] = int(contact)
    degree = np.array([sum(n.values()) for n in neighbors])
    boundary_weights = [{} for _ in members]
    for (a, b), contact in zip(pairs, contacts):
        weight = 0.35 * float(contact) / max(degree[a] / sizes[a], degree[b] / sizes[b])
        boundary_weights[a][int(b)] = weight
        boundary_weights[b][int(a)] = weight

    # A wrong, even largest, component cannot inflate its root's radius using
    # points closer to a competing neighboring centerline. Freeze these profiles
    # before moving labels so radius inflation cannot reward a growing invasion.
    projections = {}

    def projection(component, label):
        key = (component, label)
        if key not in projections:
            projections[key] = _polyline_projection_distance_and_arc(
                source[members[component]], paths[label])
        return projections[key]

    profiles = {}
    for label in np.unique(original):
        support = []
        for c in np.flatnonzero(original == label):
            own, _ = projection(c, label)
            best = own.copy()
            for other in sorted({int(original[n]) for n in neighbors[c]}):
                best = np.minimum(best, projection(c, other)[0])
            support.append(members[c][np.isfinite(own) & (own <= best + spacing)])
        support_ids = np.concatenate(support)
        if len(paths[label]) >= 2 and len(support_ids) >= 3:
            profiles[int(label)] = _segment_radius_profile(source[support_ids], paths[label], spacing)
        else:
            profiles[int(label)] = (np.array([0.0]), np.array([spacing]))

    metrics = {}
    evaluated_labels = [set() for _ in members]

    def evidence(c, label):
        key = (c, label)
        if key in metrics:
            return metrics[key]
        distance, arc = projection(c, label)
        radius = np.interp(arc, *profiles[label])
        normalized = distance / (radius + spacing)
        residual = np.abs(distance - radius) / (radius + spacing)
        path = paths[label]
        alignment = 1.0
        measured = False
        span = float(np.ptp(arc))
        # Measure longitudinal surface continuity only on elongated patches;
        # a tiny island or a transverse ring has no reliable axial direction.
        if len(members[c]) >= 3 and len(path) >= 2:
            xyz = source[members[c]]
            covariance = (xyz - xyz.mean(axis=0)).T @ (xyz - xyz.mean(axis=0)) / len(xyz)
            values, vectors = np.linalg.eigh(covariance)
            path_arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            # Use the chord over this patch's projected extent, so curved paths
            # are compared at the same scale as the patch covariance.
            ends = np.array([[np.interp(a, path_arc, path[:, k]) for k in range(3)]
                             for a in [float(arc.min()), float(arc.max())]])
            direction = ends[1] - ends[0]
            norm = np.linalg.norm(direction)
            if norm <= 2.0 * spacing:
                # A perpendicular strip can project to a single station. It
                # still has a measurable direction relative to the local axis.
                segment = int(np.clip(np.searchsorted(path_arc, np.median(arc), side="right") - 1,
                                      0, len(path) - 2))
                direction = path[segment + 1] - path[segment]
                norm = np.linalg.norm(direction)
            if values[-1] > 2.0 * max(values[-2], spacing ** 2) and norm > 1e-12:
                measured = True
                alignment = float(abs(np.dot(vectors[:, -1], direction / norm)))
        supported = bool(len(path) >= 2 and np.all(distance <= 1.75 * radius + 1.5 * spacing)
                         and (not measured or alignment >= 0.5))
        body = bool(supported and sizes[c] >= 3 and span >= max(2 * spacing, float(np.median(radius))))
        metric = {
            "segment_distance": float(np.mean(np.minimum(normalized, 8.0))),
            "radius_residual": float(np.mean(np.minimum(residual, 8.0))),
            "direction_penalty": 1.0 - alignment,
            "direction_measured": measured,
            "supported": supported,
            "body_anchor": body and int(original[c]) == label,
            "median_local_radius": float(np.median(radius)),
        }
        metrics[key] = metric
        evaluated_labels[c].add(label)
        return metric

    # Grow candidate domains over real mesh contacts only. Every traversed
    # patch must have support at ALL its vertices. Alternating mislabeled
    # patches can therefore reach a body anchor without a label-filling bridge.
    domains = [set([int(owner)]) for owner in owners]
    connected = set()
    for label in sorted(profiles):
        seeds = [int(c) for c in np.flatnonzero(original == label)
                 if evidence(int(c), label)["body_anchor"]]
        queue = deque(seeds)
        seen = set(seeds)
        while queue:
            c = queue.popleft()
            connected.add((c, label))
            domains[c].add(label)
            for n in sorted(neighbors[c]):
                if n not in seen:
                    seen.add(n)
                    if evidence(n, label)["supported"]:
                        queue.append(n)
    # Evaluate every originally neighboring label even if its claim cannot
    # reach supported body. These rejected alternatives remain auditable.
    for c in range(count):
        for n in neighbors[c]:
            evidence(c, int(original[n]))

    def unary(c, label):
        m = evidence(c, label)
        return sizes[c] * (m["segment_distance"] + 0.25 * m["radius_residual"]
                           + 0.6 * m["direction_penalty"]
                           + 0.4 * ((c, label) not in connected))

    def local_cost(c, label):
        return float(unary(c, label) + sum(
            weight for n, weight in boundary_weights[c].items() if owners[n] != label))

    def energy():
        return float(sum(unary(c, int(owners[c])) for c in range(count)) + sum(
            boundary_weights[a][int(b)] for a, b in pairs if owners[a] != owners[b]))

    report["energy"] = [energy()]
    visit = sorted(range(count), key=lambda c: int(members[c][0]))
    while True:
        moves = 0
        for c in visit:
            current = int(owners[c])
            candidates = domains[c] & {int(owners[n]) for n in neighbors[c]}
            candidates.add(current)
            scores = sorted((local_cost(c, label), label) for label in candidates)
            tolerance = 1e-9 * sizes[c]
            best_cost, target = scores[0]
            old_cost = local_cost(c, current)
            tied = len(scores) > 1 and scores[1][0] - best_cost <= tolerance
            if target == current or tied or old_cost - best_cost <= tolerance:
                continue
            owners[c] = target
            moves += 1
            report["moves"].append({"patch": c, "source_label": current, "target_label": target,
                                    "vertex_count": int(sizes[c]), "energy_decrease": old_cost - best_cost})
        report["passes"] += 1
        if not moves:
            break
        report["energy"].append(energy())
    for c, ix in enumerate(members):
        result[ix] = owners[c]
        alternatives = sorted(evaluated_labels[c])
        scores = []
        for label in alternatives:
            scores.append({"label": label, **evidence(c, label),
                           "connected_to_supported_body": (c, label) in connected,
                           "boundary_contact_edges": sum(contact for n, contact in neighbors[c].items()
                                                         if owners[n] == label),
                           "cost": local_cost(c, label),
                           "admissible": label in domains[c]})
        report["components"].append({"patch": c, "first_vertex": int(ix[0]),
                                     "vertex_count": int(sizes[c]), "source_label": int(original[c]),
                                     "target_label": int(owners[c]), "candidates": scores})
    report["initial_component_count"] = count
    report["reassigned_patch_count"] = int(np.sum(owners != original))
    report["reassigned_vertex_count"] = int(np.sum(result != before))
    return result, report


def _polyline_projection_distance_and_arc(
    query_points: np.ndarray,
    path: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact nearest-segment distance and projected path arc."""

    query = np.asarray(query_points, dtype=float)
    polyline = np.asarray(path, dtype=float)
    if query.ndim != 2 or query.shape[1] != 3:
        raise ValueError("query_points must have shape (n, 3)")
    if len(polyline) == 0:
        return (
            np.full(len(query), np.inf, dtype=float),
            np.zeros(len(query), dtype=float),
        )
    if len(polyline) == 1:
        return (
            np.linalg.norm(query - polyline[0], axis=1),
            np.zeros(len(query), dtype=float),
        )
    starts = polyline[:-1]
    vectors = np.diff(polyline, axis=0)
    squared_lengths = np.sum(vectors * vectors, axis=1)
    segment_lengths = np.sqrt(squared_lengths)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    distances = np.full(len(query), np.inf, dtype=float)
    arcs = np.zeros(len(query), dtype=float)
    chunk_size = max(128, min(8192, 1_000_000 // max(1, len(starts))))
    for chunk_start in range(0, len(query), chunk_size):
        chunk = query[chunk_start : chunk_start + chunk_size]
        offsets = chunk[:, None, :] - starts[None, :, :]
        parameters = np.zeros((len(chunk), len(starts)), dtype=float)
        valid = squared_lengths > 1e-20
        if np.any(valid):
            parameters[:, valid] = np.clip(
                np.einsum(
                    "nkj,kj->nk",
                    offsets[:, valid, :],
                    vectors[valid],
                )
                / squared_lengths[valid][None, :],
                0.0,
                1.0,
            )
        projections = starts[None, :, :] + parameters[:, :, None] * vectors[None, :, :]
        candidate_distances = np.linalg.norm(
            chunk[:, None, :] - projections,
            axis=2,
        )
        best_segment = np.argmin(candidate_distances, axis=1)
        rows = np.arange(len(chunk))
        best_distance = candidate_distances[rows, best_segment]
        best_arc = (
            cumulative[best_segment]
            + parameters[rows, best_segment] * segment_lengths[best_segment]
        )
        distances[chunk_start : chunk_start + len(chunk)] = best_distance
        arcs[chunk_start : chunk_start + len(chunk)] = best_arc
    return distances, arcs


def _segment_radius_profile(points, path, spacing):
    """Measure radii against segments, robust to sparse skeleton stations.

    Median radial support suppresses short mislabeled side protrusions.
    Bins have a physical arc width; adding collinear path nodes changes neither
    the distances nor the radius estimate.
    """
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    stations = np.linspace(0.0, arc[-1], max(2, int(np.ceil(arc[-1] / (4 * spacing))) + 1))
    distances, projected = _polyline_projection_distance_and_arc(points, path)
    bins = np.clip(np.searchsorted((stations[:-1] + stations[1:]) / 2, projected), 0, len(stations) - 1)
    values = np.full(len(stations), np.nan)
    for index in np.unique(bins):
        support = distances[bins == index]
        if len(support) >= 3:
            values[index] = np.median(support)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        values[:] = np.median(distances) if len(distances) else spacing
    else:
        values = np.interp(stations, stations[valid], values[valid])
    padded = np.pad(values, (2, 2), mode="edge")
    values = np.median(np.lib.stride_tricks.sliding_window_view(padded, 5), axis=1)
    return stations, np.maximum(values, spacing)

