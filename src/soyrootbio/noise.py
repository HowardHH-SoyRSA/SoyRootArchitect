"""Conservative pre-primary exclusions, using full native connectivity only.

This is a size/shape policy, not a biological classifier. Large, elongated,
open or ambiguous components remain available with an explicit review reason.
Source vertices and faces are never deleted or welded by this stage.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .types import PointCloudData


POLICY = "native-small-exterior-components-v2"
MIN_DOMINANT_AREA_FRACTION = 0.90
MAX_COMPONENT_AREA_FRACTION = 0.001
MAX_COMPONENT_EXTENT_FRACTION = 0.02
MAX_COMPONENT_ELONGATION = 4.0


class _ExteriorEvidence:
    """Conservative containment/contact evidence; never modifies native geometry."""

    def __init__(self, points, faces):
        from .io import require_open3d
        self.o3d = require_open3d()
        minimum = points.min(axis=0)
        scale = max(float(np.ptp(points, axis=0).max()), np.finfo(float).tiny)
        # Unit-box coordinates make ray precision independent of source units.
        self.points = (points - minimum) / scale
        self.scale = scale
        self.faces = faces
        corners = self.points[faces]
        self.low, self.high = corners.min(axis=1), corners.max(axis=1)
        centers = corners.mean(axis=1)
        self.maximum_radius = float(np.linalg.norm(corners-centers[:, None, :], axis=2).max())
        self.tree = cKDTree(centers)
        self.scene = self.o3d.t.geometry.RaycastingScene(nthreads=2)
        self.scene.add_triangles(self.o3d.t.geometry.TriangleMesh.from_legacy(self.mesh(faces)))

    def mesh(self, faces):
        indices, inverse = np.unique(faces, return_inverse=True)
        return self.o3d.geometry.TriangleMesh(
            self.o3d.utility.Vector3dVector(self.points[indices]),
            self.o3d.utility.Vector3iVector(inverse.reshape(-1, 3)))

    def inspect(self, indices, faces):
        points = self.points[indices]
        low, high = points.min(axis=0), points.max(axis=0)
        nearby = np.asarray(self.tree.query_ball_point(
            (low+high)/2, np.linalg.norm(high-low)/2+self.maximum_radius), dtype=int)
        local = self.faces[nearby[np.all(self.high[nearby] >= low, axis=1)
                                 & np.all(self.low[nearby] <= high, axis=1)]]
        intersects = bool(len(local) and len(faces) and self.mesh(local).is_intersecting(self.mesh(faces)))
        directions = np.array([[1,.123,.217],[-1,.311,.127],[.241,1,.179],
                               [.317,-1,.113],[.223,.137,1],[.179,.271,-1]])
        odd = np.stack([self.scene.count_intersections(self.o3d.core.Tensor(
            np.hstack([points, np.tile(direction, (len(points),1))]).astype(np.float32))).numpy()%2
                        for direction in directions], axis=1)
        outside_votes = (odd == 0).sum(axis=1)
        distance = self.scene.compute_distance(
            self.o3d.core.Tensor(points.astype(np.float32))).numpy()
        details = {'main_triangle_intersection': intersects,
                   'all_rays_outside_vertices': int(np.sum(outside_votes == 6)),
                   'all_rays_inside_vertices': int(np.sum(outside_votes == 0)),
                   'minimum_outside_votes': int(outside_votes.min()),
                   'minimum_main_triangle_distance': float(distance.min()*self.scale),
                   'ray_count': 6}
        if intersects or float(distance.min()) <= 32*np.finfo(np.float32).eps:
            reason = 'retained_geometric_contact_requires_review'
        elif np.any(outside_votes == 0):
            reason = 'retained_internal_surface_requires_review'
        elif outside_votes.min() < 5 or np.mean(outside_votes == 6) < .99:
            reason = 'retained_ambiguous_exterior_requires_review'
        else:
            reason = 'excluded_small_disconnected_component'
        return reason, details


def noise_report(enabled: bool) -> dict:
    return {
        "policy": POLICY, "enabled": bool(enabled),
        "status": "pending" if enabled else "disabled",
        "excluded_vertex_count": 0, "excluded_face_count": 0,
        "excluded_component_count": 0, "retained_review_component_count": 0,
        "source_geometry_preserved": True,
        "point_only_policy": "retain_unresolved_no_native_connectivity",
    }


def detect_disconnected_noise(
    points: np.ndarray, triangles: np.ndarray | None,
    *, unresolved_vertices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Classify whole small closed components before sampling/normalization.

The dominant component is chosen by surface area (tie: first native vertex).
Both area and extent must be small relative to it. An independent PCA extent
ratio preserves thin root-like fragments. No nearest-neighbor edges are added.
"""
    points = np.asarray(points, dtype=float)
    excluded = np.zeros(len(points), dtype=bool)
    report = noise_report(True)
    if triangles is None or not len(triangles):
        report["status"] = "unresolved_no_native_connectivity"
        return excluded, report
    faces = np.asarray(triangles, dtype=np.int64)
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
    graph = coo_matrix((np.ones(len(edges), np.uint8), (edges[:, 0], edges[:, 1])),
                       shape=(len(points), len(points))).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    areas = .5 * np.linalg.norm(np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                                        points[faces[:, 2]] - points[faces[:, 0]]), axis=1)
    face_labels = labels[faces[:, 0]]
    component_area = np.bincount(face_labels, weights=areas, minlength=component_count)
    main = int(np.argmax(component_area))
    dominance = float(component_area[main] / max(float(areas.sum()), np.finfo(float).tiny))
    sizes = np.bincount(labels, minlength=component_count)
    first = np.full(component_count, len(points), dtype=np.int64)
    np.minimum.at(first, labels, np.arange(len(points)))
    low = np.full((component_count, 3), np.inf)
    high = np.full((component_count, 3), -np.inf)
    np.minimum.at(low, labels, points)
    np.maximum.at(high, labels, points)
    extent = np.linalg.norm(high - low, axis=1)
    area_limit = float(component_area[main] * MAX_COMPONENT_AREA_FRACTION)
    extent_limit = float(extent[main] * MAX_COMPONENT_EXTENT_FRACTION)
    # Reject ambiguous native topology, winding, degeneracy and split seams.
    unsafe = np.zeros(component_count, dtype=bool)
    winding = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
    unsafe[labels[edges[(counts != 2) | (winding != 0), 0]]] = True
    unsafe[face_labels[areas <= 0]] = True
    distinct_faces, face_counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    unsafe[labels[distinct_faces[face_counts > 1, 0]]] = True
    if unresolved_vertices is not None:
        unsafe[labels[np.asarray(unresolved_vertices, dtype=int)]] = True
    # Coincident but separately indexed positions do not prove disconnection.
    _, position_groups = np.unique(points, axis=0, return_inverse=True)
    minimum_label = np.full(int(position_groups.max()) + 1, component_count)
    maximum_label = np.full_like(minimum_label, -1)
    np.minimum.at(minimum_label, position_groups, labels)
    np.maximum.at(maximum_label, position_groups, labels)
    unsafe[labels[(minimum_label != maximum_label)[position_groups]]] = True
    candidates = ((np.arange(component_count) != main) & ~unsafe
                  & (component_area <= area_limit) & (extent <= extent_limit))
    # Bow-tie vertices can have two incident faces on every edge. Check native
    # vertex fans too, only when there are candidates to remove.
    if np.any(candidates):
        from .io import require_open3d
        o3d = require_open3d()
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(points),
                                       o3d.utility.Vector3iVector(faces))
        unsafe[labels[np.asarray(mesh.get_non_manifold_vertices(), dtype=int)]] = True
    grouped = np.argsort(labels, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    component_excluded = np.zeros(component_count, dtype=bool)
    rows = []
    exterior = None
    for cid in range(component_count):
        if cid == main:
            continue
        indices = grouped[offsets[cid]:offsets[cid + 1]]
        elongation = None
        exterior_details = None
        if dominance < MIN_DOMINANT_AREA_FRACTION:
            reason = "retained_no_dominant_structure"
        elif unsafe[cid]:
            reason = "retained_ambiguous_native_connectivity"
        elif component_area[cid] > area_limit or extent[cid] > extent_limit:
            reason = "retained_size_requires_review"
        else:
            if len(indices) >= 3:
                local = points[indices] - points[indices].mean(axis=0)
                _, _, axes = np.linalg.svd(local, full_matrices=False)
                spans = np.sort(np.ptp(local @ axes.T, axis=0))[::-1]
                elongation = float(spans[0] / max(spans[1], np.finfo(float).tiny))
            else:
                elongation = 1.0 if len(indices) == 1 else float("inf")
            if elongation > MAX_COMPONENT_ELONGATION:
                reason = "retained_elongated_fragment_requires_review"
            else:
                if exterior is None:
                    exterior = _ExteriorEvidence(points, faces[face_labels == main])
                reason, exterior_details = exterior.inspect(indices, faces[face_labels == cid])
                component_excluded[cid] = reason == 'excluded_small_disconnected_component'
        rows.append({"component": cid, "first_vertex": int(first[cid]),
                     "vertex_count": int(sizes[cid]), "surface_area": float(component_area[cid]),
                     "extent_diagonal": float(extent[cid]),
                     "elongation": elongation if elongation is not None and np.isfinite(elongation) else None,
                     "reason": reason, "exterior_evidence": exterior_details})
    excluded = component_excluded[labels]
    report.update({
        "status": "applied" if excluded.any() else "retained_all",
        "component_count": int(component_count), "dominant_component": main,
        "dominant_component_area_fraction": dominance,
        "thresholds": {"minimum_dominant_area_fraction": MIN_DOMINANT_AREA_FRACTION,
                       "maximum_relative_area": MAX_COMPONENT_AREA_FRACTION,
                       "maximum_relative_extent": MAX_COMPONENT_EXTENT_FRACTION,
                       "maximum_elongation": MAX_COMPONENT_ELONGATION,
                       "area_limit_source_units2": area_limit,
                       "extent_limit_source_units": extent_limit},
        "excluded_vertex_count": int(excluded.sum()),
        "excluded_face_count": int(component_excluded[face_labels].sum()),
        "excluded_component_count": int(component_excluded.sum()),
        "retained_review_component_count": int(component_count - 1 - component_excluded.sum()),
        "components": rows,
    })
    return excluded, report


def filter_preloaded_cloud(cloud: PointCloudData, *, enabled: bool) -> PointCloudData:
    """Filter a caller's existing subset without mutating it or guessing a map."""
    previous = cloud.source_metadata.get("noise_reduction")
    if previous is not None and bool(previous["enabled"]) == enabled:
        return cloud
    if previous is not None and previous["enabled"]:
        raise ValueError("Reload the preloaded geometry after changing noise reduction")
    metadata = dict(cloud.source_metadata)
    if not enabled:
        metadata["noise_reduction"] = noise_report(False)
        return replace(cloud, source_metadata=metadata)
    mask, report = detect_disconnected_noise(
        cloud.export_points, cloud.triangles,
        unresolved_vertices=cloud.geometry_mapping.get("unresolved_full_vertex_indices"),
    )
    indices = cloud.analysis_indices
    if indices is None:
        if len(cloud.points) == len(cloud.export_points) and np.array_equal(cloud.points, cloud.export_points):
            indices = np.arange(len(cloud.points))
        elif mask.any():
            raise ValueError("Noise reduction needs an exact analysis-to-source mapping")
    if indices is None or not mask.any():
        metadata["noise_reduction"] = report
        return replace(cloud, source_metadata=metadata, noise_mask=mask)
    keep = ~mask[indices]
    indices = np.asarray(indices)[keep]
    mapping = dict(cloud.geometry_mapping)
    mapping["analysis_to_full"] = indices.copy()
    metadata.update(noise_reduction=report, analysis_point_count=int(keep.sum()),
                    analysis_reduced=True, retained_fraction=float(keep.sum()/len(mask)))
    return replace(cloud, points=cloud.points[keep], full_points=cloud.export_points,
                   colors=None if cloud.colors is None else cloud.colors[keep],
                   analysis_indices=indices, source_metadata=metadata,
                   geometry_mapping=mapping, noise_mask=mask)
