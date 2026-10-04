"""Diagnostic point neighborhoods; never a substitute for native mesh edges."""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .runtime import worker_threads
from .types import RootPath


def assess_point_only_evidence(
    points: np.ndarray, labels: np.ndarray, primary: np.ndarray,
    roots: list[RootPath], *, input_mode: str = "surface_points",
    excluded_mask: np.ndarray | None = None,
    neighborhood_k: int = 16, distance_multiplier: float = 2.5,
    cooperate: Callable[[], None] | None = None,
) -> dict:
    """Audit bounded mutual neighbors, local PCA frames and possible contacts.

    Negative labels and collar exclusions are barriers. Edge distances are
    bounded at both ends by local nearest *distinct* point spacing, preventing
    a sparse region from reaching across a densely sampled neighbor or gap.
    Surface frames use same-owner PCA neighborhoods; reliable tangent planes
    must agree with the edge direction at both ends. Volume frames instead
    expose local axial evidence when PCA has a dominant principal direction.
    No points, labels, paths or triangles are modified. Even an empty contact
    list cannot establish the absence of native contact or discrete patches.
    """
    p = np.asarray(points, float)
    owner = np.asarray(labels, int)
    excluded = np.zeros(len(p), bool) if excluded_mask is None else np.asarray(excluded_mask, bool)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("points must contain finite XYZ coordinates")
    if owner.shape != (len(p),) or excluded.shape != (len(p),):
        raise ValueError("labels and excluded_mask must match points")
    if np.any((owner < -2) | (owner > len(roots))):
        raise ValueError("labels must reference an existing root")
    if input_mode not in {"surface_points", "occupied_volume"}:
        raise ValueError("Point evidence requires surface_points or occupied_volume")
    if neighborhood_k < 6 or not np.isfinite(distance_multiplier) or distance_multiplier <= 0:
        raise ValueError("neighborhood_k must be at least 6 and distance_multiplier must be positive")
    active = np.flatnonzero((owner >= 0) & ~excluded)
    report = {
        "policy": "bounded-mutual-point-tangent-evidence-v1",
        "status": "unresolved_no_mesh", "input_mode": input_mode,
        "native_contact_compliance": "unresolved_no_mesh",
        "native_patch_compliance": "unresolved_no_mesh",
        "mesh_generated": False, "changed_vertex_count": 0,
        "active_point_count": int(len(active)),
        "excluded_or_negative_point_count": int(len(p) - len(active)),
        "neighborhood_k": int(neighborhood_k),
        "distance_multiplier": float(distance_multiplier),
        "frame_definition": "same-owner local PCA tangent plane" if input_mode == "surface_points" else "same-owner local PCA principal axis",
        "surface_normal_max_edge_cosine": 0.35,
        "minimum_frame_alignment_cosine": 0.5,
        "minimum_frame_neighbors": 6,
        "minimum_spacing_support_neighbors": 6,
        "duplicate_neighbor_pair_count": 0,
        "neighborhood_edge_count": 0, "tangent_compatible_edge_count": 0,
        "barrier_rejected_edge_count": 0,
        "reliable_frame_point_count": 0, "roots": [], "contacts": [],
        "higher_order_primary_possible_contact_count": 0,
        "discrete_patch_candidate_count": 0,
        "limitation": "Proximity and inferred point connectivity do not prove native contact, attachment area or surface patch connectivity; no ownership correction is authorized by this evidence.",
    }
    if len(active) < 2:
        report["evidence_status"] = "insufficient_assigned_points"
        return report
    q, lab = p[active], owner[active]
    distances, neighbors = cKDTree(q).query(q, k=min(neighborhood_k + 1, len(q)), workers=worker_threads())
    distinct = distances > 0.0
    report["duplicate_neighbor_pair_count"] = int(np.count_nonzero(~distinct[:, 1:]))
    local_spacing = np.min(np.where(distinct, distances, np.inf), axis=1)
    valid_spacing = np.isfinite(local_spacing)
    spacing_support = np.count_nonzero(distinct & (distances <= 3.0 * local_spacing[:, None]), axis=1) >= 6
    vectors = np.zeros((len(q), 3))
    reliable = np.zeros(len(q), bool)
    # Batches bound PCA temporaries and provide GUI cancellation checkpoints.
    for begin in range(0, len(q), 4096):
        if cooperate is not None:
            cooperate()
        stop = min(begin + 4096, len(q))
        ids = np.arange(begin, stop)
        nearby = neighbors[begin:stop]
        mask = (lab[nearby] == lab[ids, None]) & (distances[begin:stop] <= 3.0 * local_spacing[ids, None])
        mask &= valid_spacing[ids, None]
        count = mask.sum(axis=1)
        offsets = q[nearby] - q[ids, None, :]
        weight = mask.astype(float)
        mean = np.einsum("nk,nkj->nj", weight, offsets) / np.maximum(count, 1)[:, None]
        centered = offsets - mean[:, None, :]
        covariance = np.einsum("nk,nki,nkj->nij", weight, centered, centered) / np.maximum(count, 1)[:, None, None]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if input_mode == "surface_points":
            good = (count >= 6) & (eigenvalues[:, 1] > 1e-15) & (eigenvalues[:, 0] <= 0.2 * eigenvalues[:, 1])
            vectors[ids] = eigenvectors[:, :, 0]
        else:
            good = (count >= 6) & (eigenvalues[:, 2] > 1e-15) & (eigenvalues[:, 2] >= 4.0 * eigenvalues[:, 1])
            vectors[ids] = eigenvectors[:, :, 2]
        reliable[ids] = good
    report["reliable_frame_point_count"] = int(reliable.sum())
    # Mutual kNN candidates, without any unbounded nearest-neighbor fallback.
    first = np.repeat(np.arange(len(q)), neighbors.shape[1] - 1)
    second = neighbors[:, 1:].ravel()
    directed = first.astype(np.int64) * len(q) + second
    mutual = np.isin(second.astype(np.int64) * len(q) + first, directed)
    edges = np.column_stack([first[mutual & (first < second)], second[mutual & (first < second)]])
    delta = q[edges[:, 1]] - q[edges[:, 0]]
    length = np.linalg.norm(delta, axis=1)
    bound = distance_multiplier * np.minimum(local_spacing[edges[:, 0]], local_spacing[edges[:, 1]])
    keep = (length > 0) & (length <= bound) & np.isfinite(bound)
    keep &= spacing_support[edges[:, 0]] & spacing_support[edges[:, 1]]
    edges, delta, length = edges[keep], delta[keep], length[keep]
    barriers = np.flatnonzero(excluded | (owner < 0))
    if len(barriers) and len(edges):
        # Check interior samples against observed excluded/negative support.
        # This withholds shortcuts across a sampled collar/uncertain strip.
        barrier_tree = cKDTree(p[barriers])
        blocked = np.zeros(len(edges), bool)
        tolerance = 0.5 * np.minimum(local_spacing[edges[:, 0]], local_spacing[edges[:, 1]])
        for fraction in (0.25, 0.5, 0.75):
            sample = q[edges[:, 0]] + fraction * delta
            _, nearest_barrier = barrier_tree.query(sample, workers=worker_threads())
            offset = p[barriers[nearest_barrier]] - q[edges[:, 0]]
            arc_fraction = np.einsum("ij,ij->i", offset, delta) / length**2
            residual = np.linalg.norm(offset - np.clip(arc_fraction, 0, 1)[:, None] * delta, axis=1)
            blocked |= (arc_fraction > 0) & (arc_fraction < 1) & (residual < tolerance)
        report["barrier_rejected_edge_count"] = int(blocked.sum())
        edges, delta, length = edges[~blocked], delta[~blocked], length[~blocked]
    direction = delta / length[:, None]
    left, right = edges.T
    frames_known = reliable[left] & reliable[right]
    first_cos = np.abs(np.einsum("ij,ij->i", direction, vectors[left]))
    second_cos = np.abs(np.einsum("ij,ij->i", direction, vectors[right]))
    alignment = np.abs(np.einsum("ij,ij->i", vectors[left], vectors[right]))
    tangent = frames_known & (alignment >= 0.5)
    tangent &= ((first_cos <= 0.35) & (second_cos <= 0.35) if input_mode == "surface_points" else
                (first_cos >= 0.5) & (second_cos >= 0.5))
    report["neighborhood_edge_count"] = int(len(edges))
    report["tangent_compatible_edge_count"] = int(tangent.sum())
    same = tangent & (lab[left] == lab[right])
    graph = coo_matrix((np.ones(2 * int(same.sum()), dtype=np.uint8),
                        (np.r_[left[same], right[same]], np.r_[right[same], left[same]])), shape=(len(q), len(q))).tocsr()
    _, components = connected_components(graph, directed=False)
    by_id = {"primary": 0, **{root.root_id: i for i, root in enumerate(roots, 1)}}
    for label in range(len(roots) + 1):
        members = lab == label
        sizes = np.unique(components[members], return_counts=True)[1]
        supported_components = int(np.count_nonzero(sizes >= 3))
        report["roots"].append({
            "root_id": "primary" if label == 0 else roots[int(label) - 1].root_id,
            "assigned_point_count": int(members.sum()),
            "reliable_frame_point_count": int(reliable[members].sum()),
            "inferred_component_count": int(len(sizes)),
            "multi_point_component_count": supported_components,
            "discrete_patch_candidate_count": max(0, supported_components - 1),
            "status": "unresolved_point_connectivity" if members.any() else "insufficient_assigned_points",
        })
        report["discrete_patch_candidate_count"] += max(0, supported_components - 1)
    cross = lab[left] != lab[right]
    pairs = np.sort(np.column_stack([lab[left[cross]], lab[right[cross]]]), axis=1)
    cross_edges = np.flatnonzero(cross)
    for pair in np.unique(pairs, axis=0):
        use = cross_edges[np.all(pairs == pair, axis=1)]
        a, b = map(int, pair)
        root_a = None if a == 0 else roots[a - 1]
        root_b = roots[b - 1]
        expected_parent = by_id.get(root_b.parent_id) == a or (root_a is not None and by_id.get(root_a.parent_id) == b)
        higher_primary = a == 0 and root_b.order >= 2
        row = {
            "root_ids": ["primary" if a == 0 else root_a.root_id, root_b.root_id],
            "expected_parent_pair": bool(expected_parent),
            "higher_order_primary_pair": bool(higher_primary),
            "proximity_edge_count": int(len(use)),
            "tangent_compatible_edge_count": int(tangent[use].sum()),
            "unknown_frame_edge_count": int((~frames_known[use]).sum()),
            "minimum_gap": float(length[use].min()),
            "first_source_vertex_indices": active[edges[use[0]]].tolist(),
            "status": "possible_contact_unresolved_no_mesh",
        }
        report["contacts"].append(row)
        report["higher_order_primary_possible_contact_count"] += int(higher_primary)
    for row in report["roots"]:
        root_id = row["root_id"]
        root = next((item for item in roots if item.root_id == root_id), None)
        row["parent_id"] = None if root is None else root.parent_id
        contacts = [item for item in report["contacts"] if root is not None and set(item["root_ids"]) == {root_id, root.parent_id}]
        row["parent_proximity_edge_count"] = int(sum(item["proximity_edge_count"] for item in contacts))
        row["parent_tangent_compatible_edge_count"] = int(sum(item["tangent_compatible_edge_count"] for item in contacts))
        row["parent_contact_evidence"] = (
            "not_applicable_primary" if root is None else
            "proximity_observed_unresolved_no_mesh" if contacts else
            "not_observed_unresolved_no_mesh"
        )
    report["evidence_status"] = "possible_contacts_observed" if report["contacts"] else "no_contact_observed_in_bounded_neighborhoods"
    return report
