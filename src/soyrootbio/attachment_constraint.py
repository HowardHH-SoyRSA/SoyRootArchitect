"""Conservative, mesh-native evidence for a lateral's parent footprint.

The inferred footprint is a set of observed triangles near the extrapolated
exposed tube.  It is evidence about an attachment hypothesis, not a licence to
turn every exterior contact into a parent-child relationship.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from .surface_patches import _polyline_projection_distance_and_arc as project
from .types import RootPath


@dataclass(frozen=True)
class AttachmentBounds:
    # These are explicit, conservative working limits in units of exposed
    # child radius.  Corrected biological junctions are needed for calibration.
    area_multiplier: float = 1.5
    angle_factor_cap: float = 1.5
    longitudinal_radii: float = 5.0
    geodesic_diameter_radii: float = 6.0
    maximum_inverse_compactness: float = 6.0


def _edge_rows(faces: np.ndarray) -> np.ndarray:
    return np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]],
                              faces[:, [2, 0]])), axis=1)


def _angular_coverage(vectors: np.ndarray, tangent: np.ndarray) -> float:
    transverse = vectors - (vectors @ tangent)[:, None] * tangent
    length = np.linalg.norm(transverse, axis=1)
    transverse = transverse[length > 1e-12 * max(float(length.max(initial=0)),
                                                  np.finfo(float).tiny)]
    if len(transverse) < 5:
        return 0.0
    reference = transverse[0] / np.linalg.norm(transverse[0])
    other = np.cross(tangent, reference)
    angles = np.sort(np.arctan2(transverse @ other, transverse @ reference))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    return float(2 * np.pi - gaps.max())


def _manifold_vertex_links(faces: np.ndarray, incidence, vertices: np.ndarray) -> bool:
    """Check each observed vertex fan, including edge-manifold vertex splits."""
    for vertex in vertices:
        adjacent: dict[int, set[int]] = {}
        incident = incidence.indices[incidence.indptr[vertex]:incidence.indptr[vertex + 1]]
        for face_index in incident:
            other = faces[face_index][faces[face_index] != vertex]
            if len(other) != 2:
                return False
            a, b = map(int, other)
            adjacent.setdefault(a, set()).add(b)
            adjacent.setdefault(b, set()).add(a)
        if not adjacent:
            return False
        degree = [len(neighbors) for neighbors in adjacent.values()]
        if max(degree) > 2 or degree.count(1) not in (0, 2):
            return False
        seen = {next(iter(adjacent))}
        queue = list(seen)
        while queue:
            for neighbor in adjacent[queue.pop()]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        if len(seen) != len(adjacent):
            return False
    return True


def _exposed_radius(points: np.ndarray, labels: np.ndarray, label: int,
                    root: RootPath, spacing: float) -> tuple[float | None, str, dict]:
    path = np.asarray(root.points, float)
    start = int(root.body_start_index)
    if start < 0 or start >= len(path) or len(path[start:]) < 3:
        return None, "insufficient_exposed_path", {}
    body = path[start:]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(body, axis=0), axis=1))]
    if arc[-1] < 12 * spacing:
        return None, "insufficient_stable_length", {}
    support = points[labels == label]
    if len(support) < 24:
        return None, "insufficient_exposed_surface", {}
    distance, station = project(support, body)
    # A preliminary distal radius is used only to locate sections beyond the
    # flare.  The actual radius is remeasured there, never at the collar.
    distal = (station >= .35 * arc[-1]) & (station <= .85 * arc[-1])
    if np.count_nonzero(distal) < 16:
        return None, "insufficient_distal_surface", {}
    preliminary = float(np.median(distance[distal]))
    begin = max(.20 * arc[-1], 3 * preliminary, 6 * spacing)
    end = .85 * arc[-1]
    if begin >= end:
        return None, "flare_consumes_exposed_body", {}
    radii: list[float] = []
    coverages: list[float] = []
    for left, right in zip(np.linspace(begin, end, 5)[:-1],
                           np.linspace(begin, end, 5)[1:]):
        section = (station >= left) & (station < right)
        if np.count_nonzero(section) < 6:
            continue
        middle = (left + right) / 2
        segment = int(np.clip(np.searchsorted(arc, middle) - 1, 0, len(body) - 2))
        tangent = body[segment + 1] - body[segment]
        tangent /= max(float(np.linalg.norm(tangent)), np.finfo(float).tiny)
        center = np.array([np.interp(middle, arc, body[:, k]) for k in range(3)])
        coverage = _angular_coverage(support[section] - center, tangent)
        if coverage < np.pi:
            continue
        radii.append(float(np.median(distance[section])))
        coverages.append(coverage)
    if len(radii) < 2 or min(radii) <= spacing * .25:
        return None, "insufficient_stable_sections", {
            "stable_section_count": len(radii), "distal_support_count": int(distal.sum())}
    radius = float(np.median(radii))
    if max(radii) > 2.0 * min(radii):
        return None, "unstable_exposed_radius", {"section_radii": radii}
    return radius, "measured", {"section_radii": radii,
                                "section_angular_coverage_radians": coverages,
                                "excluded_body_prefix_nodes": start,
                                "stable_arc_start": float(begin)}


def assess_attachment_footprints(
    points: np.ndarray, labels: np.ndarray, primary_path: np.ndarray,
    roots: list[RootPath], *, triangles: np.ndarray | None, d_bar: float,
    excluded_mask: np.ndarray | None = None,
    bounds: AttachmentBounds = AttachmentBounds(),
    include_competitors: bool = False,
) -> dict:
    """Assess every current parent from the same frozen mesh and labels.

    No labels or hierarchy are changed.  Rejected footprints identify bounded
    proximal regions for reassessment; unresolved evidence never becomes a
    fabricated disk or a guessed alternative parent.
    """
    p = np.asarray(points, float)
    owner = np.asarray(labels, int)
    faces = np.empty((0, 3), int) if triangles is None else np.asarray(triangles, int)
    excluded = np.zeros(len(p), bool) if excluded_mask is None else np.asarray(excluded_mask, bool)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("points must contain finite XYZ coordinates")
    if owner.shape != (len(p),) or excluded.shape != (len(p),):
        raise ValueError("labels and excluded_mask must match points")
    if faces.ndim != 2 or faces.shape[1] != 3 or (len(faces) and
            (faces.min() < 0 or faces.max() >= len(p))):
        raise ValueError("triangles must have valid vertex indices")
    if not np.isfinite(d_bar) or d_bar <= 0:
        raise ValueError("d_bar must be positive and finite")
    if any(not np.isfinite(v) or v <= 0 for v in vars(bounds).values()):
        raise ValueError("attachment bounds must be positive and finite")
    if bounds.angle_factor_cap < 1 or bounds.maximum_inverse_compactness < 1:
        raise ValueError("angle and compactness caps must be at least one")
    report = {"policy": "observed-local-attachment-footprint-v1",
              "coordinate_system": "input coordinates",
              "evidence": "frozen labels, native triangle edges, exposed-child sections",
              "bounds": vars(bounds).copy(), "junctions": [],
              "status": "evaluated" if len(faces) else "unresolved_no_mesh"}
    if not len(faces):
        for root in roots:
            report["junctions"].append({"root_id": str(root.root_id),
                                        "parent_id": str(root.parent_id),
                                        "status": "unresolved_no_mesh"})
        return report
    good_face = ~excluded[faces].any(axis=1) & (owner[faces] >= 0).all(axis=1)
    area = .5 * np.linalg.norm(np.cross(p[faces[:, 1]] - p[faces[:, 0]],
                                       p[faces[:, 2]] - p[faces[:, 0]]), axis=1)
    good_face &= area > 1e-12 * d_bar**2
    centroids = p[faces].mean(axis=1)
    tree = cKDTree(centroids)
    face_incidence = coo_matrix(
        (np.ones(3*len(faces)),
         (faces.ravel(), np.repeat(np.arange(len(faces)), 3))),
        shape=(len(p), len(faces))).tocsr()
    all_edges = _edge_rows(faces)
    mesh_edges, mesh_incidence = np.unique(all_edges, axis=0, return_counts=True)
    mesh_edge_length = np.linalg.norm(p[mesh_edges[:, 0]] - p[mesh_edges[:, 1]], axis=1)
    parent_paths = {"primary": np.asarray(primary_path, float)}
    parent_paths.update({str(root.root_id): np.asarray(root.points, float) for root in roots})
    root_by_id = {str(root.root_id): root for root in roots}
    label_by_id = {"primary": 0, **{str(root.root_id): i for i, root in enumerate(roots, 1)}}
    frozen = owner.copy()
    frozen[excluded] = -1
    support_edges = mesh_edges[(mesh_edge_length <= 4*d_bar)
                               & (frozen[mesh_edges[:, 0]] == frozen[mesh_edges[:, 1]])
                               & (frozen[mesh_edges[:, 0]] >= 0)]
    support_graph = coo_matrix(
        (np.ones(2*len(support_edges)),
         (np.r_[support_edges[:, 0], support_edges[:, 1]],
          np.r_[support_edges[:, 1], support_edges[:, 0]])),
        shape=(len(p), len(p))).tocsr()
    _, support_component = connected_components(support_graph, directed=False)
    for child_label, child in enumerate(roots, 1):
        row = {"root_id": str(child.root_id), "parent_id": str(child.parent_id)}
        report["junctions"].append(row)
        parent_id = str(child.parent_id)
        if parent_id not in parent_paths or parent_id == str(child.root_id):
            row["status"] = "unresolved_parent_missing"
            continue
        radius, radius_status, radius_meta = _exposed_radius(p, frozen, child_label, child, d_bar)
        row["radius_evidence"] = radius_meta
        if radius is None:
            row["status"] = "unresolved_" + radius_status
            continue
        row["child_radius"] = radius
        body = np.asarray(child.points, float)[int(child.body_start_index):]
        body_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(body, axis=0), axis=1))]
        step = min(len(body) - 1, max(1, int(np.searchsorted(body_arc, max(radius, 3*d_bar)))))
        tangent = body[step] - body[0]
        tangent /= max(float(np.linalg.norm(tangent)), np.finfo(float).tiny)
        # Require an actually observed proximal child cross-section.
        child_surface = p[frozen == child_label]
        _, child_station = project(child_surface, body)
        section_at = min(max(radius, 3*d_bar), .25 * body_arc[-1])
        near_section = np.abs(child_station - section_at) <= max(.5 * radius, 2*d_bar)
        section_center = np.array([np.interp(section_at, body_arc, body[:, k]) for k in range(3)])
        section_coverage = _angular_coverage(child_surface[near_section] - section_center, tangent)
        row["junction_section_support"] = int(near_section.sum())
        row["junction_section_coverage_radians"] = section_coverage
        if near_section.sum() < 5 or section_coverage < 2*np.pi/3:
            row["status"] = "unresolved_junction_cross_section"
            continue
        parent = parent_paths[parent_id]
        if len(parent) < 2:
            row["status"] = "unresolved_parent_path"
            continue
        parent_label = label_by_id[parent_id]
        origin = (np.asarray(child.raw_start_point, float) if child.raw_start_point is not None
                  else np.asarray(child.points[0], float))
        _, parent_station = project(origin[None, :], parent)
        parent_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(parent, axis=0), axis=1))]
        insertion_arc = float(parent_station[0])
        k = int(np.clip(np.searchsorted(parent_arc, insertion_arc) - 1, 0, len(parent) - 2))
        parent_tangent = parent[k + 1] - parent[k]
        parent_tangent /= max(float(np.linalg.norm(parent_tangent)), np.finfo(float).tiny)
        sine = float(np.linalg.norm(np.cross(tangent, parent_tangent)))
        angle_factor = min(bounds.angle_factor_cap, 1 / max(sine, 1e-6))
        row["angle_factor"] = angle_factor
        row["area_limit"] = float(bounds.area_multiplier * np.pi * radius**2 * angle_factor)
        row["longitudinal_limit"] = float(bounds.longitudinal_radii * radius * angle_factor + 2*d_bar)
        row["geodesic_diameter_limit"] = float(bounds.geodesic_diameter_radii * radius * angle_factor + 2*d_bar)
        row["inverse_compactness_limit"] = float(bounds.maximum_inverse_compactness * angle_factor)
        neighborhood_radius = (bounds.longitudinal_radii * angle_factor + 2) * radius + 4*d_bar
        local = np.asarray(tree.query_ball_point(origin, neighborhood_radius), int)
        local = local[good_face[local]]
        local = local[(np.isin(owner[faces[local]], [parent_label, child_label])).all(axis=1)
                      & (owner[faces[local]] == parent_label).any(axis=1)]
        if not len(local):
            row["status"] = "unresolved_no_parent_triangles"
            continue
        center = centroids[local]
        axis_distance = np.linalg.norm(np.cross(center - body[0], tangent), axis=1)
        _, face_arc = project(center, parent)
        selected = local[(axis_distance <= 1.5*radius + 2*d_bar)
                         & (np.abs(face_arc - insertion_arc) <= 2*radius*angle_factor + 2*d_bar)]
        if not len(selected):
            row["status"] = "unresolved_no_reconstructed_patch"
            continue
        # Only the edge-connected component with observed child-parent
        # junction triangles can be an accepted footprint. Nearby surfaces
        # with no such contact are incidental and cannot enlarge it.
        candidate_faces = faces[selected]
        edge_to_faces: dict[tuple[int, int], list[int]] = {}
        for local_face, triangle in enumerate(candidate_faces):
            for a, b in ((0, 1), (1, 2), (2, 0)):
                edge = tuple(sorted((int(triangle[a]), int(triangle[b]))))
                edge_to_faces.setdefault(edge, []).append(local_face)
        links = [(members[0], other) for members in edge_to_faces.values()
                 for other in members[1:]]
        if links:
            pairs = np.asarray(links, int)
            component_graph = coo_matrix(
                (np.ones(2*len(pairs)),
                 (np.r_[pairs[:, 0], pairs[:, 1]],
                  np.r_[pairs[:, 1], pairs[:, 0]])),
                shape=(len(selected), len(selected))).tocsr()
            candidate_component_count, candidate_components = connected_components(
                component_graph, directed=False)
        else:
            candidate_component_count = len(selected)
            candidate_components = np.arange(len(selected))
        face_owners = owner[candidate_faces]
        contact_faces = ((face_owners == parent_label).any(axis=1) &
                         (face_owners == child_label).any(axis=1))
        contact_components = np.unique(candidate_components[contact_faces])
        row["candidate_component_count"] = int(candidate_component_count)
        row["native_contact_component_count"] = int(len(contact_components))
        if len(contact_components) == 0:
            row["status"] = "unresolved_no_native_child_parent_contact"
            continue
        if len(contact_components) != 1:
            row["status"] = "unresolved_multiple_contact_patches"
            continue
        chosen = candidate_components == contact_components[0]
        row["discarded_noncontact_face_count"] = int((~chosen).sum())
        selected = selected[chosen]
        patch_faces = faces[selected]
        patch_vertices = np.unique(patch_faces)
        row["patch_face_count"] = int(len(selected))
        row["patch_vertex_count"] = int(len(patch_vertices))
        # Verify an observed parent-child interface in this very footprint.
        patch_edges = _edge_rows(patch_faces)
        contact = ((owner[patch_edges[:, 0]] == parent_label) &
                   (owner[patch_edges[:, 1]] == child_label)) | (
                   (owner[patch_edges[:, 1]] == parent_label) &
                   (owner[patch_edges[:, 0]] == child_label))
        row["native_contact_edge_count"] = int(contact.sum())
        if not contact.any():
            row["status"] = "unresolved_no_native_child_parent_contact"
            continue
        contact_child_vertices = np.unique(patch_edges[contact][
            owner[patch_edges[contact]] == child_label])
        child_indices = np.flatnonzero(frozen == child_label)
        distal_indices = child_indices[child_station >= float(radius_meta["stable_arc_start"])]
        if not len(distal_indices) or not np.intersect1d(
            support_component[contact_child_vertices],
            support_component[distal_indices], assume_unique=False).size:
            row["status"] = "unresolved_disconnected_exposed_body"
            continue
        unique_edges, incidence = np.unique(patch_edges, axis=0, return_counts=True)
        mesh_at_patch = np.searchsorted(
            np.ravel_multi_index(mesh_edges.T, (len(p), len(p))),
            np.ravel_multi_index(unique_edges.T, (len(p), len(p))))
        native_incidence = mesh_incidence[mesh_at_patch]
        if np.any(mesh_edge_length[mesh_at_patch] > 4*d_bar):
            row["status"] = "unresolved_long_mesh_edge"
            continue
        if np.any(native_incidence != 2) or np.any(incidence > 2):
            row["status"] = "unresolved_open_or_nonmanifold_junction"
            continue
        if not _manifold_vertex_links(faces, face_incidence, patch_vertices):
            row["status"] = "unresolved_nonmanifold_vertex_link"
            continue
        vertex_index = np.searchsorted(patch_vertices, unique_edges)
        graph = coo_matrix((np.ones(2*len(unique_edges)),
                            (np.r_[vertex_index[:, 0], vertex_index[:, 1]],
                             np.r_[vertex_index[:, 1], vertex_index[:, 0]])),
                           shape=(len(patch_vertices), len(patch_vertices))).tocsr()
        component_count = connected_components(graph, directed=False)[0]
        boundary = unique_edges[incidence == 1]
        boundary_vertices, degrees = np.unique(boundary, return_counts=True)
        boundary_local = np.searchsorted(boundary_vertices, boundary)
        boundary_graph = coo_matrix(
            (np.ones(2*len(boundary)),
             (np.r_[boundary_local[:, 0], boundary_local[:, 1]],
              np.r_[boundary_local[:, 1], boundary_local[:, 0]])),
            shape=(len(boundary_vertices), len(boundary_vertices))).tocsr()
        boundary_components = (connected_components(boundary_graph, directed=False)[0]
                               if len(boundary_vertices) else 0)
        euler = len(patch_vertices) - len(unique_edges) + len(selected)
        row.update(patch_component_count=int(component_count),
                   boundary_component_count=int(boundary_components),
                   boundary_closed=bool(len(boundary) and np.all(degrees == 2)),
                   euler_characteristic=int(euler))
        if component_count != 1:
            row["status"] = "rejected_disconnected_footprint"
            row["patch_vertex_indices"] = patch_vertices.tolist()
            continue
        if boundary_components != 1 or not len(boundary) or not np.all(degrees == 2) or euler != 1:
            # A fused exterior can have a hole where the child emerges.  The
            # unobserved interior is not filled in to make a convenient disk.
            row["status"] = "unresolved_non_disk_observation"
            continue
        edge_length = np.linalg.norm(p[unique_edges[:, 0]] - p[unique_edges[:, 1]], axis=1)
        perimeter = float(np.linalg.norm(p[boundary[:, 0]] - p[boundary[:, 1]], axis=1).sum())
        patch_area = float(area[selected].sum())
        _, vertex_arc = project(p[patch_vertices], parent)
        longitudinal = float(np.ptp(vertex_arc))
        compactness = perimeter**2 / max(4*np.pi*patch_area, np.finfo(float).tiny)
        area_limit = bounds.area_multiplier * np.pi * radius**2 * angle_factor
        longitudinal_limit = bounds.longitudinal_radii * radius * angle_factor + 2*d_bar
        diameter_limit = bounds.geodesic_diameter_radii * radius * angle_factor + 2*d_bar
        compactness_limit = bounds.maximum_inverse_compactness * angle_factor
        weighted = coo_matrix((np.r_[edge_length, edge_length],
                               (np.r_[vertex_index[:, 0], vertex_index[:, 1]],
                                np.r_[vertex_index[:, 1], vertex_index[:, 0]])),
                              shape=graph.shape).tocsr()
        # Two sweeps give a safe lower bound.  Exact all-source distances are
        # needed to accept a patch; large patches remain unresolved.
        first = dijkstra(weighted, indices=0, directed=False)
        second = dijkstra(weighted, indices=int(np.argmax(first)), directed=False)
        diameter_lower = float(np.max(second))
        if diameter_lower > diameter_limit:
            diameter = diameter_lower
        elif len(patch_vertices) <= 400:
            diameter = float(np.max(dijkstra(weighted, directed=False)))
        else:
            row["status"] = "unresolved_geodesic_diameter_limit"
            continue
        row.update(area=patch_area, area_limit=float(area_limit),
                   longitudinal_extent=longitudinal,
                   longitudinal_limit=float(longitudinal_limit),
                   geodesic_diameter=diameter,
                   geodesic_diameter_limit=float(diameter_limit),
                   inverse_compactness=float(compactness),
                   inverse_compactness_limit=float(compactness_limit),
                   boundary_perimeter=perimeter)
        row["status"] = ("accepted" if patch_area <= area_limit and
                         longitudinal <= longitudinal_limit and
                         diameter <= diameter_limit and
                         compactness <= compactness_limit else
                         "rejected_oversized_or_elongated")
        if row["status"].startswith("rejected"):
            row["patch_vertex_indices"] = patch_vertices.tolist()
    # Every root was measured from one frozen snapshot.  The neighboring rows
    # below are reviewed as the same batch, before any label restriction.
    rows_by_id = {row["root_id"]: row for row in report["junctions"]}
    for child_index, row in enumerate(report["junctions"]):
        if not str(row["status"]).startswith("rejected"):
            continue
        root = root_by_id[row["root_id"]]
        footprint = p[np.asarray(row.get("patch_vertex_indices", []), int)]
        if not len(footprint):
            row["neighboring_insertions_for_reassessment"] = []
            continue
        radius = float(row.get("child_radius", 0))
        footprint_tree = cKDTree(footprint)
        neighbors = []
        for other in roots:
            if other is root:
                continue
            insertion = np.asarray(other.raw_start_point if other.raw_start_point is not None
                                   else other.points[0], float)
            if float(footprint_tree.query(insertion)[0]) <= 2*radius + 2*d_bar:
                neighbors.append(str(other.root_id))
        row["neighboring_insertions_for_reassessment"] = sorted(neighbors)
        row["neighbor_reassessment"] = [
            {"root_id": other_id,
             "parent_id": rows_by_id[other_id]["parent_id"],
             "attachment_status": rows_by_id[other_id]["status"],
             "root_order": int(root_by_id[other_id].order)}
            for other_id in sorted(neighbors)]
        if include_competitors:
            seed = np.asarray(root.raw_start_point if root.raw_start_point is not None
                              else root.points[0], float)
            candidates = []
            for candidate_id, candidate_path in sorted(parent_paths.items()):
                if candidate_id in {str(root.root_id), str(root.parent_id)}:
                    continue
                if candidate_id != "primary" and root_by_id[candidate_id].order >= root.order:
                    continue
                distance, _ = project(seed[None, :], candidate_path)
                if distance[0] > 3*radius + 2*d_bar:
                    continue
                alternatives = list(roots)
                alternatives[child_index] = replace(root, parent_id=candidate_id)
                alternative_report = assess_attachment_footprints(
                    p, frozen, primary_path, alternatives, triangles=faces,
                    d_bar=d_bar, excluded_mask=excluded, bounds=bounds,
                    include_competitors=False)
                candidate_row = alternative_report["junctions"][child_index]
                candidates.append({"parent_id": candidate_id,
                                   "status": candidate_row["status"],
                                   "raw_seed_axis_distance": float(distance[0])})
            row["competing_parent_hypotheses"] = candidates
    return report


def mark_final_attachment_qc(roots: list[RootPath], report: dict) -> None:
    """Export attachment uncertainty without changing tree or exposed body."""
    for root, row in zip(roots, report.get("junctions", [])):
        status = str(row.get("status", "unresolved_unknown"))
        root.qc_flags = [flag for flag in root.qc_flags if flag not in
                         {"attachment_footprint_rejected", "attachment_unresolved"}]
        if status.startswith("rejected"):
            root.qc_flags.append("attachment_footprint_rejected")
        elif status.startswith("unresolved"):
            root.qc_flags.append("attachment_unresolved")


def restrict_rejected_contacts(labels: np.ndarray, report: dict,
                              roots: list[RootPath], points: np.ndarray, *,
                              triangles: np.ndarray | None = None,
                              d_bar: float | None = None) -> tuple[np.ndarray, dict]:
    """Leave only implicated proximal child interface vertices uncertain.

    The distal exposed body, parent surface, other roots, and all pre-existing
    ambiguous/unassigned vertices are immutable under this operation.
    """
    result = np.asarray(labels, int).copy()
    p = np.asarray(points, float)
    if result.shape != (len(p),):
        raise ValueError("labels must match points")
    changed: dict[str, list[int]] = {}
    connectivity_rollbacks: list[str] = []
    mesh_edges = None
    if triangles is not None and len(triangles):
        mesh_edges = np.unique(_edge_rows(np.asarray(triangles, int)), axis=0)
        if d_bar is not None:
            lengths = np.linalg.norm(p[mesh_edges[:, 0]] - p[mesh_edges[:, 1]], axis=1)
            mesh_edges = mesh_edges[lengths <= 4*float(d_bar)]

    def support_components(current: np.ndarray, label: int) -> np.ndarray:
        assert mesh_edges is not None
        edges = mesh_edges[(current[mesh_edges[:, 0]] == label) &
                           (current[mesh_edges[:, 1]] == label)]
        graph = coo_matrix((np.ones(2*len(edges)),
                            (np.r_[edges[:, 0], edges[:, 1]],
                             np.r_[edges[:, 1], edges[:, 0]])),
                           shape=(len(p), len(p))).tocsr()
        return connected_components(graph, directed=False)[1]
    for label, (root, row) in enumerate(zip(roots, report.get("junctions", [])), 1):
        if not str(row.get("status", "")).startswith("rejected"):
            continue
        vertices = np.asarray(row.get("patch_vertex_indices", []), int)
        vertices = vertices[(vertices >= 0) & (vertices < len(p))]
        vertices = vertices[result[vertices] == label]
        if not len(vertices):
            continue
        start = int(root.body_start_index)
        if start >= len(root.points) - 1:
            continue
        _, station = project(p[vertices], np.asarray(root.points, float)[start:])
        stable_start = float(row.get("radius_evidence", {}).get("stable_arc_start", 0))
        vertices = vertices[station < stable_start]
        if not len(vertices):
            continue
        if mesh_edges is not None:
            owned = np.flatnonzero(result == label)
            retained = np.setdiff1d(owned, vertices, assume_unique=True)
            before_component = support_components(result, label)[retained]
            proposed = result.copy()
            proposed[vertices] = -2
            after_component = support_components(proposed, label)[retained]
            splits_body = any(
                np.unique(after_component[before_component == component]).size > 1
                for component in np.unique(before_component))
            if splits_body:
                connectivity_rollbacks.append(str(root.root_id))
                continue
        result[vertices] = -2
        changed[str(root.root_id)] = vertices.tolist()
    return result, {"policy": "rejected-proximal-contact-to-uncertain-v1",
                    "changed_vertex_count": int(sum(map(len, changed.values()))),
                    "changed_by_root": changed,
                    "connectivity_rollbacks": connectivity_rollbacks}
