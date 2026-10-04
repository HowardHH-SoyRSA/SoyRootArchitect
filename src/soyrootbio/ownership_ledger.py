"""Frozen provisional ownership evidence and conservative interface reuse.

The ledger records uncertainty separately from labels.  In particular, an
uncertain vertex is a barrier unless a caller explicitly marks it eligible for
supervised reconsideration. This module never invents mesh edges or edits root
paths; it considers only vertices just released by guarded native
higher-order/primary-contact or rejected-attachment restrictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .mesh_geometry import MeshGeometryContext
from .surface_patches import _polyline_projection_distance_and_arc
from .types import RootPath


@dataclass(frozen=True)
class OwnershipEvidenceSnapshot:
    """One read-only generation from which all simultaneous proposals derive."""

    labels: np.ndarray
    excluded: np.ndarray
    exposed_body_anchors: np.ndarray
    generation: int


@dataclass
class ProvisionalEvidenceLedger:
    """Per-vertex provisional evidence without changing mesh vertex identity.

    ``excluded`` and ``exposed_body_anchors`` are immutable within a ledger.
    ``records`` contains an entry for every uncertain vertex, including older
    uncertainty that is not eligible for this contact-seam reconsideration.
    Each record stores owner candidates, the generation that supplied them,
    the reason for uncertainty, and whether supervised reconsideration is
    allowed.  Records remain in ``history`` after a supported commit.
    """

    labels: np.ndarray
    excluded: np.ndarray
    exposed_body_anchors: np.ndarray
    generation: int = 0
    records: dict[int, dict] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels, dtype=int).copy()
        self.excluded = np.asarray(self.excluded, dtype=bool).copy()
        self.exposed_body_anchors = np.asarray(
            self.exposed_body_anchors, dtype=bool
        ).copy()
        if self.labels.ndim != 1 or self.excluded.shape != self.labels.shape or \
                self.exposed_body_anchors.shape != self.labels.shape:
            raise ValueError("ledger arrays must have one entry per vertex")
        if np.any(self.excluded & self.exposed_body_anchors):
            raise ValueError("excluded vertices cannot be exposed-body anchors")
        if np.any(self.exposed_body_anchors & (self.labels < 0)):
            raise ValueError("exposed-body anchors must have an owner")
        if self.generation < 0:
            raise ValueError("generation must be nonnegative")
        self.excluded.flags.writeable = False
        self.exposed_body_anchors.flags.writeable = False
        for vertex in np.flatnonzero(self.labels == -2):
            self.records.setdefault(int(vertex), {
                "owner_candidates": [],
                "generation": int(self.generation),
                "reason": "preexisting_uncertainty",
                "eligible_for_reconsideration": False,
            })

    def snapshot(self) -> OwnershipEvidenceSnapshot:
        """Copy the current labels and permanent barriers for a frozen pass."""
        arrays = [self.labels.copy(), self.excluded.copy(),
                  self.exposed_body_anchors.copy()]
        for array in arrays:
            array.flags.writeable = False
        return OwnershipEvidenceSnapshot(*arrays, int(self.generation))

    def record_interface(
        self, vertices: Sequence[int] | np.ndarray, *,
        owner_candidates: Sequence[int], reason: str,
        eligible_for_reconsideration: bool,
    ) -> None:
        ids = np.unique(np.asarray(vertices, dtype=int))
        if np.any((ids < 0) | (ids >= len(self.labels))):
            raise ValueError("interface vertex index out of range")
        if np.any(self.labels[ids] != -2) or np.any(self.excluded[ids]) or \
                np.any(self.exposed_body_anchors[ids]):
            raise ValueError("only unprotected uncertain vertices are interfaces")
        candidates = sorted({int(owner) for owner in owner_candidates})
        if any(owner < 0 for owner in candidates):
            raise ValueError("owner candidates must be nonnegative labels")
        for vertex in ids:
            self.records[int(vertex)] = {
                "owner_candidates": candidates.copy(),
                "generation": int(self.generation),
                "reason": str(reason),
                "eligible_for_reconsideration": bool(eligible_for_reconsideration),
            }

    def commit(
        self, proposals: Sequence[tuple[np.ndarray, int, str]], *,
        expected_generation: int,
    ) -> np.ndarray:
        """Apply disjoint supported regions together against one generation."""
        if expected_generation != self.generation:
            raise ValueError("stale ownership generation")
        result = self.labels.copy()
        claimed = np.zeros(len(result), dtype=bool)
        decisions = []
        for vertices, owner, reason in proposals:
            ids = np.unique(np.asarray(vertices, dtype=int))
            if not len(ids) or np.any((ids < 0) | (ids >= len(result))):
                raise ValueError("proposal must contain valid vertices")
            if np.any(claimed[ids]) or np.any(self.excluded[ids]) or \
                    np.any(self.exposed_body_anchors[ids]) or \
                    np.any(self.labels[ids] != -2):
                raise ValueError("proposal overlaps or changes a protected vertex")
            if int(owner) < 0 or any(
                int(owner) not in self.records[int(vertex)]["owner_candidates"] or
                not self.records[int(vertex)]["eligible_for_reconsideration"] or
                self.records[int(vertex)]["generation"] != self.generation
                for vertex in ids
            ):
                raise ValueError("proposal lacks frozen eligible candidate evidence")
            claimed[ids] = True
            result[ids] = int(owner)
            decisions.append({"vertices": ids.tolist(), "owner": int(owner),
                              "reason": str(reason),
                              "evidence_generation": int(self.generation)})
        if decisions:
            for decision in decisions:
                for vertex in decision["vertices"]:
                    self.history.append({"vertex": vertex,
                                         "prior_record": self.records.pop(vertex),
                                         **{key: value for key, value in decision.items()
                                            if key != "vertices"}})
            self.labels = result
            self.generation += 1
        return self.labels.copy()

    def summary(self) -> dict:
        return {
            "generation": int(self.generation),
            "excluded_vertex_count": int(np.count_nonzero(self.excluded)),
            "exposed_body_anchor_count": int(np.count_nonzero(self.exposed_body_anchors)),
            "uncertain_vertex_count": int(np.count_nonzero(self.labels == -2)),
            "reconsiderable_vertex_count": sum(
                row["eligible_for_reconsideration"] for row in self.records.values()),
            "committed_vertex_count": len(self.history),
            "interfaces": [
                {"vertex": vertex, **row} for vertex, row in sorted(self.records.items())
            ],
        }


class ProvisionalTraceEvidenceLedger:
    """Keep the two tracing assignment checkpoints for each root order.

    Candidate pairs are calculated by the assignment from its frozen path
    snapshot. This recorder preserves them with generation numbers and keeps
    spatial indexes only while the matching exposed path geometry is current.
    It does not turn an uncertain tip barrier into a permanent owner.
    """

    def __init__(self, points: np.ndarray, excluded_mask: np.ndarray) -> None:
        self.points = np.asarray(points, dtype=float)
        self.excluded = np.asarray(excluded_mask, dtype=bool).copy()
        if self.points.ndim != 2 or self.points.shape[1] != 3 or \
                self.excluded.shape != (len(self.points),):
            raise ValueError("trace ledger geometry and exclusion must align")
        self.excluded.flags.writeable = False
        self.generation = 0
        self.checkpoints: list[dict] = []
        self.interfaces: list[dict] = []
        self.commits: list[dict] = []
        self.label_snapshots: list[np.ndarray] = []
        self.latest_anchor_owner = np.full(len(self.points), -1, dtype=np.int32)
        self.anchor_snapshots: list[np.ndarray] = []
        self.native_snapshots: dict[int, dict[str, np.ndarray]] = {}
        self._path_versions: dict[str, str] = {}
        self._body_trees: dict[str, cKDTree] = {}

    def capture(
        self,
        labels: np.ndarray,
        paths: list[RootPath],
        *,
        stage: str,
        root_order: int,
        d_bar: float,
        competitor_labels: Mapping[int, Sequence[int]] | None = None,
        released_interfaces: Mapping[int, Sequence[int]] | None = None,
        released_interface_reasons: Mapping[int, str] | None = None,
        released_interface_origins: Mapping[int, int] | None = None,
        supported_commits: Sequence[dict] | None = None,
        native_reconciliation: dict | None = None,
        native_evidence: Mapping[str, np.ndarray] | None = None,
        primary_path: np.ndarray | None = None,
    ) -> dict:
        frozen = np.asarray(labels, dtype=int)
        if frozen.shape != (len(self.points),) or np.any(frozen[self.excluded] != -1):
            raise ValueError("excluded analysis vertices must stay unassigned")
        if not np.isfinite(d_bar) or d_bar <= 0:
            raise ValueError("d_bar must be positive and finite")
        if np.any((frozen < -2) | (frozen > len(paths))):
            raise ValueError("provisional labels must reference selected paths")
        anchor_owner = np.full(len(frozen), -1, dtype=np.int32)
        versions = {}
        changed = []
        reused = 0
        body_trees = dict(self._body_trees)
        if primary_path is not None:
            primary = np.asarray(primary_path, dtype=float)
            versions["primary"] = hashlib.sha256(
                np.ascontiguousarray(primary.astype("<f8")).tobytes()
            ).hexdigest()
        for label, path in enumerate(paths, start=1):
            root_id = str(path.root_id)
            body_start = max(0, int(path.body_start_index))
            body = np.asarray(path.points, dtype=float)[body_start:]
            if len(body) < 2:
                continue
            digest = hashlib.sha256()
            digest.update(np.ascontiguousarray(body.astype("<f8")).tobytes())
            digest.update(np.asarray([body_start], dtype="<i8").tobytes())
            version = digest.hexdigest()
            versions[root_id] = version
            if self._path_versions.get(root_id) != version:
                changed.append(root_id)
            tree = body_trees.get(version)
            if tree is None:
                tree = cKDTree(body)
                body_trees[version] = tree
            else:
                reused += 1
            covered = np.asarray(sorted(path.novel_support_indices
                                        if path.novel_support_indices is not None
                                        else path.covered_indices), dtype=int)
            covered = covered[(covered >= 0) & (covered < len(frozen))]
            covered = covered[frozen[covered] == label]
            if not len(covered):
                continue
            distance, nearest_node = tree.query(self.points[covered], k=1)
            local_radius = float(path.mean_radius or 0.0)
            body_threshold = max(1, int(np.ceil(0.20 * len(body))))
            confidence = (nearest_node >= body_threshold) & (
                distance <= max(2.0 * d_bar, local_radius)
            )
            anchor_owner[covered[confidence]] = label
        changed.extend(sorted(set(self._path_versions) - set(versions)))
        candidate_sources = (
            (competitor_labels or {}, "competing_exposed_segments", False,
             "global_assignment_qc_only"),
            (released_interfaces or {}, "rejected_attachment_interface", True,
             "frozen_parent_order_native_family"),
        )
        record_by_vertex: dict[int, dict] = {}
        for mapping, default_reason, reconsiderable, scope in candidate_sources:
            for vertex, owners in mapping.items():
                vertex = int(vertex)
                if vertex < 0 or vertex >= len(frozen) or frozen[vertex] != -2 or \
                        self.excluded[vertex]:
                    continue
                owner_candidates = sorted({int(owner) for owner in owners
                                           if 0 <= int(owner) <= len(paths)})
                reason = (str((released_interface_reasons or {}).get(vertex, default_reason))
                          if reconsiderable else default_reason)
                record = {
                    "vertex": vertex,
                    "generation": self.generation,
                    "root_order": int(root_order),
                    "stage": str(stage),
                    "owner_candidates": owner_candidates,
                    "reason": reason,
                    "eligible_for_reconsideration": reconsiderable,
                    "candidate_scope": scope,
                }
                if reconsiderable and vertex in (released_interface_origins or {}):
                    origin = int(released_interface_origins[vertex])
                    if origin <= 0 or origin > len(paths):
                        raise ValueError("released source owner must be a selected lateral")
                    source = paths[origin - 1]
                    family = {int(label) for label, path in enumerate(paths, 1)
                              if int(path.order) == int(source.order)
                              and str(path.parent_id) == str(source.parent_id)}
                    parent = next((int(label) for label, path in enumerate(paths, 1)
                                   if str(path.root_id) == str(source.parent_id)), None)
                    if parent is not None:
                        family.add(parent)
                    family.discard(origin)
                    if not set(owner_candidates).issubset(family):
                        raise ValueError("released candidates leave frozen parent/order family")
                    record.update(source_owner=origin,
                                  source_parent_id=str(source.parent_id),
                                  source_order=int(source.order))
                record_by_vertex[vertex] = record
        checkpoint_commits = []
        for decision in supported_commits or ():
            vertex = int(decision["analysis_vertex"])
            owner = int(decision["owner"])
            if vertex < 0 or vertex >= len(frozen) or self.excluded[vertex] or \
                    owner <= 0 or owner > len(paths) or frozen[vertex] != owner or \
                    int(decision["generation"]) != self.generation:
                raise ValueError("provisional commit does not match frozen checkpoint")
            checkpoint_commits.append({**decision, "analysis_vertex": vertex,
                                       "owner": owner})
        if len({int(row["analysis_vertex"]) for row in checkpoint_commits}) != len(checkpoint_commits):
            raise ValueError("provisional commits must address disjoint vertices")
        native_archive_hashes = {}
        native_arrays = None
        if native_evidence is not None:
            required = {"analysis_to_mesh", "excluded", "exposed_body_anchors",
                        "before_labels", "restricted_labels", "committed_labels",
                        "released_vertices", "source_owners", "final_owners"}
            if set(native_evidence) != required:
                raise ValueError("native evidence archive has missing or extra arrays")
            arrays = {key: np.asarray(value).copy()
                      for key, value in native_evidence.items()}
            native_size = len(arrays["excluded"])
            if arrays["analysis_to_mesh"].shape != (len(frozen),) or \
                    arrays["exposed_body_anchors"].shape != (native_size,) or \
                    any(arrays[key].shape != (native_size,)
                        for key in ("before_labels", "restricted_labels",
                                    "committed_labels")) or \
                    any(arrays[key].shape != (len(arrays["released_vertices"]),)
                        for key in ("source_owners", "final_owners")):
                raise ValueError("native evidence archive shapes do not align")
            if np.any(arrays["excluded"] & arrays["exposed_body_anchors"]):
                raise ValueError("native excluded vertices cannot be anchors")
            ids = arrays["released_vertices"]
            if np.any((ids < 0) | (ids >= native_size)) or \
                    len(np.unique(ids)) != len(ids):
                raise ValueError("native released vertices must be distinct and in range")
            if np.any(arrays["before_labels"][ids] != arrays["source_owners"]) or \
                    np.any(arrays["restricted_labels"][ids] != -2) or \
                    np.any(arrays["committed_labels"][ids] != arrays["final_owners"]):
                raise ValueError("native released-vertex record disagrees with frozen labels")
            anchors = arrays["exposed_body_anchors"]
            if np.any(arrays["restricted_labels"][anchors] < 0) or \
                    np.any(arrays["committed_labels"][anchors] !=
                           arrays["restricted_labels"][anchors]):
                raise ValueError("native exposed-body anchor owner changed")
            for key, value in arrays.items():
                value.flags.writeable = False
                native_archive_hashes[key] = hashlib.sha256(
                    np.ascontiguousarray(value).tobytes()
                ).hexdigest()
            native_arrays = arrays
        checkpoint = {
            "generation": self.generation,
            "root_order": int(root_order),
            "stage": str(stage),
            "owner_roots": [
                {"label": 0, "root_id": "primary", "root_order": 0,
                 "parent_id": None},
                *({"label": label, "root_id": str(path.root_id),
                   "root_order": int(path.order), "parent_id": str(path.parent_id),
                   "body_start_index": int(path.body_start_index)}
                  for label, path in enumerate(paths, 1)),
            ],
            "path_versions_sha256": dict(versions),
            "labels_sha256": hashlib.sha256(
                np.ascontiguousarray(frozen.astype("<i8")).tobytes()
            ).hexdigest(),
            "excluded_vertex_count": int(np.count_nonzero(self.excluded)),
            "exposed_body_anchor_count": int(np.count_nonzero(anchor_owner >= 0)),
            "uncertain_vertex_count": int(np.count_nonzero(frozen == -2)),
            "interface_record_count": len(record_by_vertex),
            "supported_native_commit_count": len(checkpoint_commits),
            "native_reconciliation": native_reconciliation,
            "native_archive_sha256": native_archive_hashes,
            "changed_path_domains": sorted(changed),
            "reused_body_index_count": reused,
            "candidate_derivation": "frozen_selected_paths_and_assignment_competitors",
            "commit": "whole_checkpoint_labels_committed_together",
        }
        # Publish the complete checkpoint only after every frozen candidate,
        # proposed commit, and native archive array has passed validation.
        self._path_versions = versions
        self._body_trees = {
            version: body_trees[version]
            for version in versions.values() if version in body_trees
        }
        self.latest_anchor_owner = anchor_owner
        self.label_snapshots.append(frozen.astype(np.int32, copy=True))
        self.anchor_snapshots.append(anchor_owner.copy())
        self.interfaces.extend(record_by_vertex[key]
                               for key in sorted(record_by_vertex))
        self.commits.extend(sorted(checkpoint_commits,
                                   key=lambda row: int(row["analysis_vertex"])))
        if native_arrays is not None:
            self.native_snapshots[len(self.checkpoints)] = native_arrays
        self.checkpoints.append(checkpoint)
        self.generation += 1
        return checkpoint

    def payload(self) -> dict:
        return {
            "policy": "provisional-tracing-evidence-ledger-v1",
            "generation_count": self.generation,
            "checkpoints": self.checkpoints,
            "interfaces": self.interfaces,
            "commits": self.commits,
        }


def _native_edges(points: np.ndarray, triangles: np.ndarray,
                  d_bar: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    listed = np.sort(np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                triangles[:, [2, 0]])), axis=1)
    listed = listed[listed[:, 0] != listed[:, 1]]
    edges, incidences = np.unique(listed, axis=0, return_counts=True)
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    return edges, edges[lengths <= 4.0 * d_bar], incidences


def _graph(size: int, edges: np.ndarray):
    return coo_matrix((np.ones(2 * len(edges), dtype=np.int8),
                       (np.r_[edges[:, 0], edges[:, 1]],
                        np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(size, size)).tocsr()


def _released_components(released: np.ndarray, edges: np.ndarray) -> list[np.ndarray]:
    vertices = np.flatnonzero(released)
    if not len(vertices):
        return []
    selected = edges[released[edges[:, 0]] & released[edges[:, 1]]]
    local = np.full(len(released), -1, dtype=int)
    local[vertices] = np.arange(len(vertices))
    graph = _graph(len(vertices), local[selected])
    _, membership = connected_components(graph, directed=False)
    return [vertices[membership == index] for index in np.unique(membership)]


def _contact_edge_set(edges: np.ndarray, labels: np.ndarray,
                      orders: np.ndarray) -> set[tuple[int, int]]:
    left, right = labels[edges[:, 0]], labels[edges[:, 1]]
    contact = (((left == 0) & (right > 0)) |
               ((right == 0) & (left > 0)))
    other = np.maximum(left[contact], right[contact])
    return {tuple(map(int, edge)) for edge in edges[contact][orders[other] >= 2]}


def _candidate_family(origin: int, roots: list[RootPath],
                      by_id: dict[str, int]) -> set[int]:
    if origin == 0:
        return {index for index, root in enumerate(roots, 1)
                if int(root.order) == 1 and str(root.parent_id) == "primary"}
    root = roots[origin - 1]
    parent = by_id.get(str(root.parent_id))
    return {origin, *([parent] if parent is not None else []),
            *(index for index, peer in enumerate(roots, 1)
              if int(peer.order) == int(root.order) and
              str(peer.parent_id) == str(root.parent_id))}


def reconcile_released_primary_contact_vertices(
    points: np.ndarray,
    labels_before_contact: np.ndarray,
    labels_after_contact: np.ndarray,
    primary_path: np.ndarray,
    roots: list[RootPath],
    *,
    triangles: np.ndarray | None,
    d_bar: float,
    excluded_mask: np.ndarray | None = None,
    competitor_labels: Mapping[int, Sequence[int]] | None = None,
    generation: int = 0,
    mesh_context: MeshGeometryContext | None = None,
    allow_released_owner: bool = True,
    allow_primary_owner: bool = True,
    release_kind: str = "higher_order_primary_contact",
) -> tuple[np.ndarray, ProvisionalEvidenceLedger, dict]:
    """Reassign newly released contact or attachment regions conservatively.

    Candidates are derived from the frozen post-restriction labels, never from
    sequentially accepted claims.  Native mesh continuity to an already owned,
    geometrically supported exposed body and a separate centerline fit are both
    required.  Open/nonmanifold or long-edge-only evidence stays uncertain.
    Every accepted region is committed in one generation; a final raw-edge
    audit rejects the batch if it would recreate any O2+ primary contact.
    """
    source = np.asarray(points, dtype=float)
    before = np.asarray(labels_before_contact, dtype=int)
    after = np.asarray(labels_after_contact, dtype=int)
    primary = np.asarray(primary_path, dtype=float)
    faces = (np.empty((0, 3), dtype=int) if triangles is None
             else np.asarray(triangles, dtype=int))
    excluded = (np.zeros(len(source), dtype=bool) if excluded_mask is None
                else np.asarray(excluded_mask, dtype=bool))
    if source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all():
        raise ValueError("points must contain finite XYZ coordinates")
    if before.shape != (len(source),) or after.shape != before.shape or \
            excluded.shape != before.shape:
        raise ValueError("labels and excluded_mask must match points")
    if np.any(before < -2) or np.any(after < -2) or \
            np.any(before > len(roots)) or np.any(after > len(roots)):
        raise ValueError("labels must reference a root or assignment state")
    if faces.ndim != 2 or faces.shape[1] != 3 or \
            (len(faces) and (faces.min() < 0 or faces.max() >= len(source))):
        raise ValueError("triangles must have valid vertex indices")
    if not np.isfinite(d_bar) or d_bar <= 0:
        raise ValueError("d_bar must be positive and finite")
    if primary.ndim != 2 or primary.shape[1] != 3 or not np.isfinite(primary).all():
        raise ValueError("primary_path must contain finite XYZ coordinates")
    if len({str(root.root_id) for root in roots}) != len(roots):
        raise ValueError("root IDs must be unique")
    if np.any(before[excluded] != after[excluded]):
        raise ValueError("contact restriction changed immutable excluded vertices")

    released = (before >= 0) & (after == -2) & ~excluded
    anchor_mask = np.zeros(len(source), dtype=bool)
    ledger = ProvisionalEvidenceLedger(after, excluded, anchor_mask, generation)
    for vertex, candidates in (competitor_labels or {}).items():
        vertex = int(vertex)
        if 0 <= vertex < len(after) and after[vertex] == -2 and not excluded[vertex]:
            ledger.record_interface([vertex], owner_candidates=candidates,
                                    reason="preexisting_competitor_interface",
                                    eligible_for_reconsideration=False)
    report = {
        "policy": ("frozen-contact-release-evidence-v1"
                   if release_kind == "higher_order_primary_contact"
                   else "frozen-attachment-release-evidence-v1"),
        "status": "evaluated",
        "evidence_generation": int(generation),
        "released_vertex_count": int(np.count_nonzero(released)),
        "reassigned_vertex_count": 0,
        "unresolved_region_count": 0,
        "regions": [],
        "raw_higher_order_primary_contacts_before": 0,
        "raw_higher_order_primary_contacts_after": 0,
    }
    if not np.any(released):
        report["status"] = "no_contact_releases"
        return after.copy(), ledger, report
    if not len(faces):
        for vertex in np.flatnonzero(released):
            ledger.record_interface([vertex], owner_candidates=[int(before[vertex])],
                                    reason="unresolved_no_native_mesh",
                                    eligible_for_reconsideration=False)
        report["status"] = "unresolved_no_native_mesh"
        report["unresolved_region_count"] = int(np.count_nonzero(released))
        return after.copy(), ledger, report

    if mesh_context is not None:
        mesh_context.validate(source, faces)
        raw_edges = mesh_context.edges[
            mesh_context.edges[:, 0] != mesh_context.edges[:, 1]
        ]
        incidences = mesh_context.edge_incidence[
            mesh_context.edges[:, 0] != mesh_context.edges[:, 1]
        ]
        edges = mesh_context.bounded_edges(d_bar)
    else:
        raw_edges, edges, incidences = _native_edges(source, faces, d_bar)
    mesh = _graph(len(source), edges)
    raw_mesh = _graph(len(source), raw_edges)
    open_or_nonmanifold = np.zeros(len(source), dtype=bool)
    unsafe_edges = raw_edges[incidences != 2]
    open_or_nonmanifold[unsafe_edges.ravel()] = True
    orders = np.array([0, *(int(root.order) for root in roots)])
    paths = [primary] + [np.asarray(root.points, dtype=float)[
        int(root.body_start_index):] for root in roots]
    for path in paths:
        if path.ndim != 2 or path.shape[1] != 3 or not np.isfinite(path).all():
            raise ValueError("root paths must contain finite XYZ coordinates")
    by_id = {str(root.root_id): label for label, root in enumerate(roots, 1)}
    by_id["primary"] = 0
    original_contacts = _contact_edge_set(raw_edges, after, orders)
    report["raw_higher_order_primary_contacts_before"] = len(original_contacts)
    snapshot = ledger.snapshot()
    # Identify each retained exposed body independently of a released seam.
    # A geometrically close, disconnected island must not authenticate itself
    # as the body that supports another ownership change.
    same_label_edges = edges[
        (snapshot.labels[edges[:, 0]] == snapshot.labels[edges[:, 1]]) &
        (snapshot.labels[edges[:, 0]] >= 0) &
        ~snapshot.excluded[edges].any(axis=1)
    ]
    _, body_components = connected_components(
        _graph(len(source), same_label_edges), directed=False)
    body_component_cache: dict[int, tuple[int | None, float, str]] = {}

    def retained_body_component(label: int) -> tuple[int | None, float, str]:
        cached = body_component_cache.get(label)
        if cached is not None:
            return cached
        mean_radius = None if label == 0 else roots[label - 1].mean_radius
        radius_bound = (3.0 * d_bar if mean_radius is None or
                        not np.isfinite(mean_radius) or mean_radius <= 0
                        else max(3.0 * d_bar, 1.5 * mean_radius + d_bar))
        owned = np.flatnonzero((snapshot.labels == label) & ~snapshot.excluded)
        path = paths[label]
        if len(path) < 2 or len(owned) < 2:
            result = (None, radius_bound, "insufficient_retained_exposed_body")
        else:
            distance, node = cKDTree(path).query(source[owned], k=1)
            distal = node >= max(1, int(np.ceil(0.20 * len(path))))
            anchored = owned[distal & (distance <= radius_bound)]
            components, counts = np.unique(body_components[anchored],
                                           return_counts=True)
            if not len(counts) or counts.max() < 2:
                result = (None, radius_bound, "insufficient_retained_exposed_body")
            elif np.count_nonzero(counts == counts.max()) != 1:
                result = (None, radius_bound, "ambiguous_retained_body_component")
            else:
                result = (int(components[np.argmax(counts)]), radius_bound,
                          "connected_retained_exposed_body")
        body_component_cache[label] = result
        return result

    proposals: list[tuple[np.ndarray, int, str]] = []
    proposal_rows: list[dict] = []
    for component in _released_components(released, edges):
        origins = np.unique(before[component])
        boundary = np.unique(mesh[component].indices)
        boundary = boundary[~np.isin(boundary, component)]
        touching = sorted({int(label) for label in snapshot.labels[boundary]
                           if label >= 0})
        candidate_labels: list[int] = []
        row = {"vertices": component.tolist(), "origin_labels": origins.tolist(),
               "candidate_labels": [], "candidate_evidence": [],
               "decision": "unresolved_no_supported_candidate",
               "target_label": None,
               "generation": int(snapshot.generation)}
        if len(origins) != 1:
            row["decision"] = "unresolved_mixed_origin_component"
        elif np.any(open_or_nonmanifold[component]):
            row["decision"] = "unresolved_open_or_nonmanifold_region"
        else:
            family = _candidate_family(int(origins[0]), roots, by_id)
            if not allow_released_owner:
                family.discard(int(origins[0]))
            if not allow_primary_owner:
                family.discard(0)
            candidate_labels = sorted(set(touching) & family)
            row["candidate_labels"] = candidate_labels
            supported = []
            for label in candidate_labels:
                owned = boundary[(snapshot.labels[boundary] == label) &
                                 ~snapshot.excluded[boundary]]
                evidence = {"label": label, "native_boundary_vertex_count": int(len(owned)),
                            "supported": False}
                body_component, radius_bound, body_reason = retained_body_component(label)
                if body_component is not None:
                    owned = owned[body_components[owned] == body_component]
                evidence["connected_exposed_boundary_vertex_count"] = int(len(owned))
                if body_component is None:
                    evidence["reason"] = body_reason
                elif len(owned) < 2:
                    evidence["reason"] = "insufficient_owned_exposed_boundary"
                elif label >= 1 and orders[label] >= 2 and \
                        np.any(snapshot.labels[np.unique(raw_mesh[component].indices)] == 0):
                    evidence["reason"] = "would_recreate_higher_order_primary_contact"
                elif label == 0 and np.any(
                    (snapshot.labels[np.unique(raw_mesh[component].indices)] > 0) &
                    (orders[np.maximum(snapshot.labels[
                        np.unique(raw_mesh[component].indices)], 0)] >= 2)
                ):
                    evidence["reason"] = "would_recreate_higher_order_primary_contact"
                else:
                    owned_distance, owned_arc = _polyline_projection_distance_and_arc(
                        source[owned], paths[label])
                    component_distance, component_arc = _polyline_projection_distance_and_arc(
                        source[component], paths[label])
                    anchor = owned_distance <= radius_bound
                    anchor_ids = owned[anchor]
                    evidence["exposed_body_anchor_vertex_count"] = int(len(anchor_ids))
                    if len(anchor_ids) < 2:
                        evidence["reason"] = "insufficient_geometric_body_anchors"
                    else:
                        anchor_mask[anchor_ids] = True
                        local_limit = min(radius_bound,
                                          float(np.median(owned_distance[anchor])) +
                                          1.5 * d_bar)
                        evidence["maximum_centerline_distance"] = float(
                            np.max(component_distance))
                        evidence["distance_limit"] = float(local_limit)
                        evidence["arc_gap"] = float(max(
                            0.0, np.min(component_arc) - np.max(owned_arc[anchor]),
                            np.min(owned_arc[anchor]) - np.max(component_arc)))
                        if np.any(component_distance > local_limit):
                            evidence["reason"] = "insufficient_exposed_body_geometry"
                        elif evidence["arc_gap"] > 2.0 * d_bar:
                            evidence["reason"] = "disjoint_exposed_body_arc"
                        else:
                            evidence["supported"] = True
                            evidence["reason"] = "native_connection_and_exposed_geometry"
                            evidence["mean_centerline_distance"] = float(
                                np.mean(component_distance))
                            supported.append(evidence)
                row["candidate_evidence"].append(evidence)
            if len(supported) == 1:
                row["target_label"] = int(supported[0]["label"])
                row["decision"] = "supported_reassignment"
            elif len(supported) > 1:
                ranked = sorted(supported,
                                key=lambda item: (item["mean_centerline_distance"],
                                                  item["label"]))
                if (ranked[1]["mean_centerline_distance"] -
                        ranked[0]["mean_centerline_distance"] > 0.5 * d_bar):
                    row["target_label"] = int(ranked[0]["label"])
                    row["decision"] = "supported_reassignment"
                else:
                    row["decision"] = "unresolved_competing_supported_owners"
        ledger.record_interface(component, owner_candidates=candidate_labels,
                                reason=row["decision"],
                                eligible_for_reconsideration=(
                                    row["decision"] == "supported_reassignment"))
        if row["target_label"] is not None:
            proposals.append((component, row["target_label"],
                              "native_connection_and_exposed_geometry"))
            proposal_rows.append(row)
        report["regions"].append(row)

    # An anchor is established only from the frozen, already owned body.
    ledger.exposed_body_anchors = anchor_mask.copy()
    ledger.exposed_body_anchors.flags.writeable = False
    proposed = snapshot.labels.copy()
    for vertices, label, _ in proposals:
        proposed[vertices] = label
    novel_contacts = _contact_edge_set(raw_edges, proposed, orders) - original_contacts
    if novel_contacts:
        for row in proposal_rows:
            row["decision"] = "unresolved_batch_would_recreate_higher_order_primary_contact"
            row["target_label"] = None
            ledger.record_interface(row["vertices"],
                                    owner_candidates=row["candidate_labels"],
                                    reason=row["decision"],
                                    eligible_for_reconsideration=False)
        proposals = []
    result = ledger.commit(proposals, expected_generation=snapshot.generation)
    report["reassigned_vertex_count"] = int(np.count_nonzero(result != after))
    report["unresolved_region_count"] = sum(
        row["decision"] != "supported_reassignment" for row in report["regions"])
    report["raw_higher_order_primary_contacts_after"] = len(
        _contact_edge_set(raw_edges, result, orders))
    report["status"] = ("unresolved_regions" if report["unresolved_region_count"]
                        else "reassigned_or_clear")
    return result, ledger, report


def reconcile_provisional_attachment_interfaces(
    analysis_points: np.ndarray,
    analysis_labels: np.ndarray,
    analysis_to_mesh: np.ndarray,
    mesh_points: np.ndarray,
    mesh_labels_before: np.ndarray,
    mesh_labels_restricted: np.ndarray,
    primary_path: np.ndarray,
    roots: list[RootPath],
    *,
    triangles: np.ndarray,
    d_bar: float,
    analysis_excluded_mask: np.ndarray,
    mesh_excluded_mask: np.ndarray,
    generation: int,
    mesh_context: MeshGeometryContext | None = None,
) -> tuple[np.ndarray, dict[int, tuple[int, ...]], dict[int, str],
           dict[int, int], list[dict], dict, dict[str, np.ndarray]]:
    """Reconsider sampled rejected interfaces on the real full mesh only.

    The supplied analysis-to-mesh index map preserves duplicate coordinates as
    distinct vertices. The restriction has already protected every retained
    child component and descendant contact. A rejected child cannot reclaim
    its own released interface; only its lateral parent or same-parent/order
    peer can receive a region when native connectivity and exposed-body
    geometry independently support that recipient. All proposals derive from
    one frozen full-resolution label snapshot and commit together.
    """
    analysis = np.asarray(analysis_points, dtype=float)
    labels = np.asarray(analysis_labels, dtype=int)
    mapping = np.asarray(analysis_to_mesh, dtype=int)
    source = np.asarray(mesh_points, dtype=float)
    before = np.asarray(mesh_labels_before, dtype=int)
    restricted = np.asarray(mesh_labels_restricted, dtype=int)
    analysis_excluded = np.asarray(analysis_excluded_mask, dtype=bool)
    mesh_excluded = np.asarray(mesh_excluded_mask, dtype=bool)
    if analysis.ndim != 2 or analysis.shape[1] != 3 or \
            source.ndim != 2 or source.shape[1] != 3 or \
            labels.shape != (len(analysis),) or \
            mapping.shape != (len(analysis),) or \
            analysis_excluded.shape != labels.shape or \
            before.shape != (len(source),) or restricted.shape != before.shape or \
            mesh_excluded.shape != before.shape:
        raise ValueError("analysis/full geometry, labels, and mapping must align")
    if np.any((mapping < 0) | (mapping >= len(source))) or \
            len(np.unique(mapping)) != len(mapping):
        raise ValueError("analysis-to-mesh indices must be distinct and in range")
    if not np.array_equal(analysis, source[mapping]):
        raise ValueError("analysis-to-mesh mapping must preserve exact ordered coordinates")
    if np.any((before != restricted) & ((before <= 0) | (restricted != -2))):
        raise ValueError("provisional restriction may only release lateral owners")
    if np.any(before[mesh_excluded] != restricted[mesh_excluded]):
        raise ValueError("provisional restriction changed immutable mesh exclusions")
    if np.any(analysis_excluded & (labels > 0)):
        raise ValueError("excluded analysis vertices cannot have lateral owners")

    native_labels, native_ledger, native_report = (
        reconcile_released_primary_contact_vertices(
            source, before, restricted, primary_path, roots,
            triangles=triangles, d_bar=d_bar,
            excluded_mask=mesh_excluded, generation=generation,
            mesh_context=mesh_context,
            allow_released_owner=False, allow_primary_owner=False,
            release_kind="rejected_attachment",
        )
    )
    affected = ((before[mapping] > 0) & (restricted[mapping] == -2) &
                (labels == before[mapping]) & ~analysis_excluded)
    result = labels.copy()
    interfaces: dict[int, tuple[int, ...]] = {}
    reasons: dict[int, str] = {}
    origins: dict[int, int] = {}
    commits: list[dict] = []
    history = {int(row["vertex"]): row for row in native_ledger.history}
    for analysis_vertex in np.flatnonzero(affected):
        mesh_vertex = int(mapping[analysis_vertex])
        origin = int(before[mesh_vertex])
        owner = int(native_labels[mesh_vertex])
        if owner == -2:
            record = native_ledger.records[mesh_vertex]
            interfaces[int(analysis_vertex)] = tuple(int(label) for label in
                                                      record["owner_candidates"])
            reasons[int(analysis_vertex)] = str(record["reason"])
            origins[int(analysis_vertex)] = origin
        elif owner > 0 and owner != origin:
            decision = history[mesh_vertex]
            commits.append({
                "analysis_vertex": int(analysis_vertex),
                "mesh_vertex": mesh_vertex,
                "source_owner": origin,
                "owner": owner,
                "reason": str(decision["reason"]),
                "generation": int(generation),
                "source_parent_id": str(roots[origin - 1].parent_id),
                "source_order": int(roots[origin - 1].order),
            })
        else:
            raise AssertionError("native provisional decision escaped its candidate family")
        result[analysis_vertex] = owner
    released_native = np.flatnonzero((before > 0) & (restricted == -2) & ~mesh_excluded)
    native_records = []
    for mesh_vertex in released_native:
        mesh_vertex = int(mesh_vertex)
        committed = history.get(mesh_vertex)
        evidence = (committed["prior_record"] if committed is not None
                    else native_ledger.records[mesh_vertex])
        native_records.append({
            "mesh_vertex": mesh_vertex,
            "source_owner": int(before[mesh_vertex]),
            "final_owner": int(native_labels[mesh_vertex]),
            "owner_candidates": [int(label) for label in evidence["owner_candidates"]],
            "reason": str(committed["reason"] if committed is not None
                          else evidence["reason"]),
            "evidence_generation": int(evidence["generation"]),
        })
    native_evidence = {
        "analysis_to_mesh": mapping.copy(),
        "excluded": mesh_excluded.copy(),
        "exposed_body_anchors": native_ledger.exposed_body_anchors.copy(),
        "before_labels": before.copy(),
        "restricted_labels": restricted.copy(),
        "committed_labels": native_labels.copy(),
        "released_vertices": released_native.astype(np.int64, copy=True),
        "source_owners": before[released_native].astype(np.int32, copy=True),
        "final_owners": native_labels[released_native].astype(np.int32, copy=True),
    }
    frozen_hashes = {
        key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for key, value in (
            ("before_labels", before),
            ("restricted_labels", restricted),
            ("committed_labels", native_labels),
            ("excluded", mesh_excluded),
            ("exposed_body_anchors", native_ledger.exposed_body_anchors),
        )
    }
    report = {
        **native_report,
        "candidate_scope": "frozen_lateral_parent_and_same_parent_order_peers",
        "mapping_policy": "supplied_analysis_to_full_vertex_indices",
        "mapped_released_analysis_vertex_count": int(np.count_nonzero(affected)),
        "mapped_supported_commit_count": len(commits),
        "mapped_unresolved_interface_count": len(interfaces),
        "native_commits": native_ledger.history,
        "native_released_vertex_records": native_records,
        "native_frozen_snapshot_sha256": frozen_hashes,
    }
    return result, interfaces, reasons, origins, commits, report, native_evidence
