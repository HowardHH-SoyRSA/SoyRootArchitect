"""Read-only final audit of disconnected child-owned patches on parent mesh."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .mesh_geometry import MeshGeometryContext
from .identity_repair import identity_reference, _sample
from .types import RootPath


def native_body_anchor_evidence(points, owned_ids, components, prior, spacing, *,
                                normals=None, radius_hint=None):
    """Locate an exposed body using sections around the independent prior.

    A missing stored radius must not turn a thick shaft into a false patch.
    Radius queries come from native vertices projected onto the preserved
    pre-fit path; circumferential surface/normal support then has to establish
    each section independently. No final fitted centerline supplies evidence.
    Components are examined separately, so disconnected patches cannot pool
    their points to fabricate a complete ring.
    """
    from .junction_tubes import _measure_ring
    from .surface_patches import _segment_radius_profile, _polyline_projection_distance_and_arc

    prior = np.asarray(prior, float)
    ids = np.asarray(owned_ids, int)
    result = {"policy": "independent-native-transverse-body-anchor-v1",
              "body_component": None, "supported_components": [],
              "section_counts": {}, "sections": [],
              "radius_source": "native_sections_against_preserved_prefit_path",
              "status": "unresolved_no_native_transverse_body"}
    if normals is None or len(prior) < 2 or len(ids) < 12:
        return result
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(prior, axis=0), axis=1))]
    if arc[-1] <= 0:
        return result
    stations = np.linspace(.2 * arc[-1], arc[-1],
                           max(2, int(np.ceil(.8 * arc[-1] / (2 * spacing))) + 1))
    centers, directions, _ = _sample(prior, stations)
    accepted = {}
    for component in np.unique(components[ids]):
        native = ids[components[ids] == component]
        if len(native) < 12:
            continue
        _, projected = _polyline_projection_distance_and_arc(points[native], prior)
        if np.ptp(projected) < 4 * spacing:
            continue
        positions, radii = _segment_radius_profile(points[native], prior, spacing)
        tree = cKDTree(points[native])
        records = []
        for index in np.flatnonzero((stations >= projected.min()) & (stations <= projected.max())):
            radius = float(np.interp(stations[index], positions, radii))
            if radius_hint is not None:
                radius = min(radius, max(2 * float(radius_hint), spacing))
            ring = _measure_ring(points, normals, native, tree, centers[index],
                                 directions[index], radius, spacing)
            if ring is None:
                continue
            records.append({"station_index": int(index), "arc": float(stations[index]),
                            "radius": float(ring[2]), "support_vertex_count": int(len(ring[3]))})
        if len(records) < 3:
            continue
        span = records[-1]["arc"] - records[0]["arc"]
        median_radius = float(np.median([r["radius"] for r in records]))
        if span < max(4 * spacing, 2 * median_radius):
            continue
        key = int(component)
        accepted[key] = len(records)
        result["section_counts"][str(key)] = len(records)
        result["sections"].append({"component": key, "accepted_section_count": len(records),
                                   "axial_span": span, "median_radius": median_radius,
                                   "records": records})
    result["supported_components"] = sorted(accepted)
    ranked = sorted(accepted, key=lambda part: (-accepted[part], part))
    if ranked and (len(ranked) == 1 or accepted[ranked[0]] > 1.5 * accepted[ranked[1]]):
        result["body_component"] = ranked[0]
        result["status"] = "supported_native_transverse_body"
    elif ranked:
        result["status"] = "unresolved_multiple_supported_bodies"
    return result


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
    identity_references: dict | None = None,
) -> dict:
    """Report native parent-contact patches detached from the exposed body.

    Body support uses bounded native edges. Discrete patches additionally need
    actual raw native disconnection; a long native connection is retained as a
    connectivity ambiguity, not silently omitted or called an isolated patch.
    Raw parent contact always enters the report. No ownership or root geometry
    is changed by this audit. Body anchors use persistent pre-fit
    evidence, never a shortened final fit. Assessed legacy roots without this
    evidence remain unresolved instead of certifying their fitted fragment.
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
        "policy": "final-native-discrete-child-patch-audit-v3",
        "status": "unresolved_no_mesh" if not len(faces) else "clear",
        "unresolved_patch_count": 0,
        "unresolved_root_count": 0,
        "patches": [],
        "body_anchor_ambiguities": [],
        "body_anchor_evidence": [],
        "unresolved_body_anchor_root_count": 0,
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
    active_raw_edges = raw_edges[~excluded[raw_edges].any(axis=1)]
    if mesh_context is not None:
        ownership = mesh_context.ownership(owner)
        component = ownership.components(safe_edges)
        native_component = ownership.components(active_raw_edges)
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
        same_native = active_raw_edges[
            (owner[active_raw_edges[:, 0]] == owner[active_raw_edges[:, 1]]) &
            (owner[active_raw_edges[:, 0]] >= 0)]
        native_graph = coo_matrix((np.ones(len(same_native)),
                                  (same_native[:, 0], same_native[:, 1])),
                                 shape=(len(source), len(source))).tocsr()
        _, native_component = connected_components(native_graph, directed=False)
    label_by_id = {"primary": 0}
    label_by_id.update({str(root.root_id): i for i, root in enumerate(roots, 1)})
    from .junction_tubes import vertex_normals
    normals = vertex_normals(source, faces)
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
        native_parts = np.unique(native_component[owned])
        body, radius, reference_source = identity_reference(root, identity_references)
        body_part = None
        independently_supported = set()
        anchor_evidence = None
        if body is not None:
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(body, axis=0), axis=1))]
            if arc[-1] > 0:
                stations = np.linspace(.2 * arc[-1], arc[-1],
                                       max(2, int(np.ceil(.8 * arc[-1] / (2 * d_bar))) + 1))
                centers = _sample(body, stations)[0]
                # Equal arc stations, rather than vertex density or component
                # size, supply evidence against the independent retained path.
                distance, node = cKDTree(source[owned]).query(centers)
                supported = distance <= max(2.0 * d_bar, float(radius or 0.) + d_bar)
                anchored_parts, counts = np.unique(component[owned[node[supported]]],
                                                   return_counts=True)
                if len(counts):
                    ranked = np.argsort(-counts, kind="stable")
                    minimum = min(3, len(stations))
                    independently_supported = set(map(int, anchored_parts[counts >= minimum]))
                    strongest = ranked[0]
                    if (counts[strongest] >= minimum and
                            (len(ranked) == 1 or counts[strongest] > 1.5 * counts[ranked[1]])):
                        body_part = int(anchored_parts[strongest])
                if radius is None:
                    anchor_evidence = native_body_anchor_evidence(
                        source, owned, component, body, d_bar, normals=normals)
                    # Measured sections supersede the radius-free point-distance
                    # heuristic, which systematically rejects thick shafts.
                    if anchor_evidence["supported_components"]:
                        independently_supported = set(anchor_evidence["supported_components"])
                        body_part = anchor_evidence["body_component"]
        if anchor_evidence is not None:
            report["body_anchor_evidence"].append({"root_id": str(root.root_id), **anchor_evidence})
        left, right = owner[raw_edges[:, 0]], owner[raw_edges[:, 1]]
        contact = (((left == child_label) & (right == parent_label)) |
                   ((right == child_label) & (left == parent_label)))
        touching = raw_edges[contact]
        child_end = np.where(owner[touching[:, 0]] == child_label,
                             touching[:, 0], touching[:, 1])
        if len(native_parts) == 1 and body_part is None and len(touching):
            # There is no separate owned component to call a discrete patch.
            # Keep uncertainty explicit instead of certifying this root or
            # treating its entire connected body as removable parent tissue.
            report["body_anchor_ambiguities"].append({
                "root_id": str(root.root_id), "parent_id": str(root.parent_id),
                "root_order": int(root.order), "vertex_indices": owned.tolist(),
                "native_parent_contact_edges": touching.tolist(),
                "status": ("unresolved_single_component_body_identity" if len(parts) == 1
                           else "unresolved_long_native_edge_body_identity"),
                "body_anchor_reference_source": reference_source,
                "native_owned_component_count": 1,
                "bounded_owned_component_count": int(len(parts)),
            })
            unresolved_roots.add(str(root.root_id))
            continue
        for part in parts:
            if body_part is not None and int(part) == body_part:
                continue
            vertices = owned[component[owned] == part]
            incident = touching[component[child_end] == part]
            if not len(incident):
                continue
            body_native_part = (native_component[owned[component[owned] == body_part][0]]
                                if body_part is not None else None)
            if body_native_part is not None and np.all(native_component[vertices] == body_native_part):
                report["body_anchor_ambiguities"].append({
                    "root_id": str(root.root_id), "parent_id": str(root.parent_id),
                    "root_order": int(root.order), "vertex_indices": vertices.tolist(),
                    "native_parent_contact_edges": incident.tolist(),
                    "status": "unresolved_long_native_edge_body_connection",
                    "body_anchor_reference_source": reference_source,
                    "native_owned_component_count": int(len(native_parts)),
                    "bounded_owned_component_count": int(len(parts)),
                })
                unresolved_roots.add(str(root.root_id))
                continue
            row = {
                "root_id": str(root.root_id),
                "parent_id": str(root.parent_id),
                "root_order": int(root.order),
                "vertex_indices": vertices.tolist(),
                "native_parent_contact_edges": incident.tolist(),
                "body_anchor_component_identified": body_part is not None,
                "body_anchor_reference_source": reference_source,
                "component_has_independent_body_support": int(part) in independently_supported,
                "body_anchor_evidence_policy": (anchor_evidence["policy"] if anchor_evidence is not None
                                                  else "preserved_prior_radius_proximity"),
                "native_owned_component_count": int(len(native_parts)),
                "bounded_owned_component_count": int(len(parts)),
                "disconnection_basis": "distinct_raw_native_owned_components",
                "status": ("unresolved_disconnected_exposed_body"
                           if int(part) in independently_supported else
                           "unresolved_discrete_child_patch" if body_part is not None
                           else "unresolved_exposed_body_anchor_ambiguous"),
            }
            report["patches"].append(row)
            unresolved_roots.add(str(root.root_id))
    report["unresolved_patch_count"] = len(report["patches"])
    report["unresolved_root_count"] = len(unresolved_roots)
    report["unresolved_body_anchor_root_count"] = len({r["root_id"] for r in report["body_anchor_ambiguities"]})
    if report["patches"]:
        report["status"] = "unresolved_patches"
    elif report["body_anchor_ambiguities"]:
        report["status"] = "unresolved_body_identity"
    return report
