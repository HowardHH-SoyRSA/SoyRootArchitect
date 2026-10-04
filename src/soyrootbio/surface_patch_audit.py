"""Read-only final audit of disconnected child-owned patches on parent mesh."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .mesh_geometry import MeshGeometryContext
from .types import RootPath


def audit_discrete_child_patches(
    points: np.ndarray,
    labels: np.ndarray,
    roots: list[RootPath],
    *,
    triangles: np.ndarray | None,
    d_bar: float,
    excluded_mask: np.ndarray | None = None,
    mesh_context: MeshGeometryContext | None = None,
    nonroot_labels: tuple[int, ...] = (),
) -> dict:
    """Report native parent-contact patches detached from the exposed body.

    Connectivity uses only bounded native edges. Raw parent contact, including
    anomalously long edges, still enters the violation report. No ownership or
    root geometry is changed by this audit.
    """

    source = np.asarray(points, dtype=float)
    owner = np.asarray(labels, dtype=int)
    faces = (np.empty((0, 3), dtype=np.int64) if triangles is None
             else np.asarray(triangles, dtype=np.int64))
    excluded = (np.zeros(len(source), dtype=bool) if excluded_mask is None
                else np.asarray(excluded_mask, dtype=bool))
    if source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all():
        raise ValueError("points must contain finite XYZ coordinates")
    if owner.shape != (len(source),) or excluded.shape != owner.shape or \
            np.any(((owner < -2) & ~np.isin(owner, nonroot_labels)) | (owner > len(roots))):
        raise ValueError("labels and exclusion must match native vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or (
        len(faces) and (faces.min() < 0 or faces.max() >= len(source))
    ):
        raise ValueError("triangles must have valid native vertex indices")
    if not np.isfinite(d_bar) or d_bar <= 0:
        raise ValueError("d_bar must be positive and finite")
    report = {
        "policy": "final-native-discrete-child-patch-audit-v1",
        "status": "unresolved_no_mesh" if not len(faces) else "clear",
        "unresolved_patch_count": 0,
        "unresolved_root_count": 0,
        "patches": [],
    }
    if not len(faces):
        return report
    if mesh_context is not None:
        mesh_context.validate(source, faces)
        raw_edges = mesh_context.edges[
            mesh_context.edges[:, 0] != mesh_context.edges[:, 1]
        ]
        safe_edges = mesh_context.bounded_edges(d_bar)
    else:
        raw_edges = np.unique(np.sort(np.vstack((
            faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
        )), axis=1), axis=0)
        raw_edges = raw_edges[raw_edges[:, 0] != raw_edges[:, 1]]
        lengths = np.linalg.norm(
            source[raw_edges[:, 0]] - source[raw_edges[:, 1]], axis=1
        )
        safe_edges = raw_edges[lengths <= 4.0 * d_bar]
    safe_edges = safe_edges[~excluded[safe_edges].any(axis=1)]
    if mesh_context is not None:
        ownership = mesh_context.ownership(owner)
        component = ownership.components(safe_edges)
    else:
        same = safe_edges[
            (owner[safe_edges[:, 0]] == owner[safe_edges[:, 1]]) &
            (owner[safe_edges[:, 0]] >= 0)
        ]
        graph = coo_matrix((
            np.ones(2 * len(same)),
            (np.r_[same[:, 0], same[:, 1]],
             np.r_[same[:, 1], same[:, 0]]),
        ), shape=(len(source), len(source))).tocsr()
        _, component = connected_components(graph, directed=False)
    label_by_id = {"primary": 0}
    label_by_id.update({str(root.root_id): i for i, root in enumerate(roots, 1)})
    unresolved_roots = set()
    for child_label, root in enumerate(roots, 1):
        parent_label = label_by_id.get(str(root.parent_id))
        if parent_label is None:
            continue
        owned = (ownership.vertices(child_label) if mesh_context is not None
                 else np.flatnonzero(owner == child_label))
        owned = owned[~excluded[owned]]
        if not len(owned):
            continue
        parts = np.unique(component[owned])
        body = np.asarray(root.points, dtype=float)[max(0, int(root.body_start_index)):]
        if len(body) < 2:
            body_part = None
        else:
            distance, node = cKDTree(body).query(source[owned], k=1)
            distal = node >= max(1, int(np.ceil(.20 * len(body))))
            radius = float(root.mean_radius or 0.0)
            anchor = distal & (distance <= max(2.0 * d_bar, radius))
            anchored_parts, counts = np.unique(component[owned[anchor]],
                                               return_counts=True)
            body_part = (int(anchored_parts[np.argmax(counts)])
                         if len(counts) and np.count_nonzero(counts == counts.max()) == 1
                         else None)
        left, right = owner[raw_edges[:, 0]], owner[raw_edges[:, 1]]
        contact = (((left == child_label) & (right == parent_label)) |
                   ((right == child_label) & (left == parent_label)))
        touching = raw_edges[contact]
        child_end = np.where(owner[touching[:, 0]] == child_label,
                             touching[:, 0], touching[:, 1])
        for part in parts:
            if body_part is not None and int(part) == body_part:
                continue
            vertices = owned[component[owned] == part]
            incident = touching[component[child_end] == part]
            if not len(incident):
                continue
            row = {
                "root_id": str(root.root_id),
                "parent_id": str(root.parent_id),
                "root_order": int(root.order),
                "vertex_indices": vertices.tolist(),
                "native_parent_contact_edges": incident.tolist(),
                "body_anchor_component_identified": body_part is not None,
                "status": ("unresolved_discrete_child_patch" if body_part is not None
                           else "unresolved_exposed_body_anchor_ambiguous"),
            }
            report["patches"].append(row)
            unresolved_roots.add(str(root.root_id))
    report["unresolved_patch_count"] = len(report["patches"])
    report["unresolved_root_count"] = len(unresolved_roots)
    if report["patches"]:
        report["status"] = "unresolved_patches"
    return report
