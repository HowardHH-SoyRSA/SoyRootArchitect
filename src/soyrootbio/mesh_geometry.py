"""Run-scoped facts derived once from immutable native mesh geometry."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _edge_key(edges: np.ndarray) -> tuple:
    return (edges.shape, hashlib.sha256(np.ascontiguousarray(edges).tobytes()).digest())


class OwnershipGeometryGeneration:
    """Frozen ownership, with grouped support and policy-specific graphs.

    Component numbers belong to this snapshot and edge/active policy only.
    Geometry caches never carry them into the next ownership generation.
    """

    def __init__(self, labels: np.ndarray, points: np.ndarray, number: int = 0,
                 geometry: MeshGeometryContext | None = None):
        values = np.array(labels, dtype=np.int64, copy=True)
        if values.shape != (len(points),):
            raise ValueError("labels must contain one value per point")
        self.labels = _readonly(values)
        self.points = points
        self.number = number
        self._geometry = geometry
        order = np.argsort(values, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(values[order])) + 1] if len(values) else []
        ends = np.r_[starts[1:], len(values)] if len(values) else []
        self._vertices = {
            int(values[order[start]]): _readonly(order[start:end])
            for start, end in zip(starts, ends)
        }
        self._empty = _readonly(np.empty(0, dtype=np.int64))
        self._edge_groups: dict[tuple, dict[int, np.ndarray]] = {}
        self._edge_group_lengths: dict[tuple, dict[int, np.ndarray]] = {}
        self._boundary_edges: dict[tuple, dict[tuple[int, int], np.ndarray]] = {}
        self._local_graphs: dict[tuple, csr_matrix] = {}
        self._components: dict[tuple, np.ndarray] = {}
        self._edge_keys: dict[int, tuple[np.ndarray, tuple]] = {}

    def _key(self, edges: np.ndarray) -> tuple:
        # An owning read-only policy array can retain its content key. A
        # read-only view may still change through a writable base, so hash
        # views again even when their own write flag is disabled. Retain
        # owning references so object-id recycling cannot reuse a key.
        if not edges.flags.writeable and edges.flags.owndata:
            prior = self._edge_keys.get(id(edges))
            if prior is None:
                prior = (edges, _edge_key(edges))
                self._edge_keys[id(edges)] = prior
            return prior[1]
        return _edge_key(edges)

    def vertices(self, label: int) -> np.ndarray:
        return self._vertices.get(int(label), self._empty)

    def edges(self, label: int, edges: np.ndarray) -> np.ndarray:
        """Group native edges once, retaining their original order."""
        key = self._key(edges)
        if key not in self._edge_groups:
            left, right = self.labels[edges[:, 0]], self.labels[edges[:, 1]]
            ids = np.flatnonzero(left == right)
            ids = ids[np.argsort(left[ids], kind="stable")]
            starts = np.r_[0, np.flatnonzero(np.diff(left[ids])) + 1] if len(ids) else []
            ends = np.r_[starts[1:], len(ids)] if len(ids) else []
            self._edge_groups[key] = {
                int(left[ids[start]]): _readonly(edges[ids[start:end]].copy())
                for start, end in zip(starts, ends)
            }
            lengths = (self._geometry.lengths_for(edges) if self._geometry is not None else
                       np.linalg.norm(self.points[edges[:, 0]] - self.points[edges[:, 1]], axis=1))
            self._edge_group_lengths[key] = {
                int(left[ids[start]]): _readonly(lengths[ids[start:end]].copy())
                for start, end in zip(starts, ends)
            }
        return self._edge_groups[key].get(int(label), np.empty((0, 2), dtype=np.int64))

    def local_graph(self, label: int, edges: np.ndarray) -> csr_matrix:
        key = (int(label), self._key(edges))
        if key not in self._local_graphs:
            vertices = self.vertices(label)
            owned = self.edges(label, edges)
            local = np.searchsorted(vertices, owned)
            lengths = np.maximum(self._edge_group_lengths[key[1]].get(int(label), self._empty), 1e-15)
            graph = coo_matrix((np.r_[lengths, lengths],
                                (np.r_[local[:, 0], local[:, 1]],
                                 np.r_[local[:, 1], local[:, 0]])),
                               shape=(len(vertices), len(vertices))).tocsr()
            for buffer in (graph.data, graph.indices, graph.indptr):
                _readonly(buffer)
            self._local_graphs[key] = graph
        return self._local_graphs[key]

    def boundary_edges(self, first: int, second: int, edges: np.ndarray) -> np.ndarray:
        """Group crossing native edges once for this frozen ownership."""
        key = self._key(edges)
        if key not in self._boundary_edges:
            owners = self.labels[edges]
            crossing = np.flatnonzero(owners[:, 0] != owners[:, 1])
            pairs = np.sort(owners[crossing], axis=1)
            order = np.lexsort((pairs[:, 1], pairs[:, 0]))
            pairs, crossing = pairs[order], crossing[order]
            starts = (np.r_[0, np.flatnonzero(np.any(pairs[1:] != pairs[:-1], axis=1)) + 1]
                      if len(pairs) else [])
            ends = np.r_[starts[1:], len(pairs)] if len(pairs) else []
            self._boundary_edges[key] = {
                tuple(map(int, pairs[start])): _readonly(edges[crossing[start:end]].copy())
                for start, end in zip(starts, ends)
            }
        return self._boundary_edges[key].get(tuple(sorted((int(first), int(second)))),
                                             np.empty((0, 2), dtype=np.int64))

    def components(self, edges: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
        if active is None:
            active = self.labels >= 0
        active = np.asarray(active, dtype=bool)
        if active.shape != self.labels.shape:
            raise ValueError("active must match labels")
        key = (self._key(edges), np.packbits(active).tobytes())
        if key not in self._components:
            same = edges[active[edges[:, 0]] & active[edges[:, 1]] &
                         (self.labels[edges[:, 0]] == self.labels[edges[:, 1]])]
            graph = coo_matrix((np.ones(len(same)), (same[:, 0], same[:, 1])),
                               shape=(len(self.labels), len(self.labels))).tocsr()
            self._components[key] = _readonly(connected_components(graph, directed=False)[1])
        return self._components[key]


@dataclass(frozen=True)
class MeshGeometryContext:
    points: np.ndarray
    triangles: np.ndarray
    edges: np.ndarray
    edge_lengths: np.ndarray
    edge_incidence: np.ndarray
    face_areas: np.ndarray
    vertex_area_weights: np.ndarray
    face_centroids: np.ndarray
    centroid_tree: cKDTree | None
    vertex_face_incidence: csr_matrix
    coordinate_sha256: str
    point_tree: cKDTree
    _bounded_edges: dict[float, np.ndarray] = field(default_factory=dict)
    _support_edges: dict[float, np.ndarray] = field(default_factory=dict)
    _ownership: list[OwnershipGeometryGeneration] = field(default_factory=list)
    _policy_lengths: dict[tuple, np.ndarray] = field(default_factory=dict)
    _edge_codes: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def build(
        cls, points: np.ndarray, triangles: np.ndarray | None,
    ) -> MeshGeometryContext:
        # Own the snapshots: a caller mutating its arrays cannot silently
        # invalidate an already constructed tree or coordinate fingerprint.
        source = np.array(points, dtype=float, copy=True, order="C")
        faces = (np.empty((0, 3), dtype=np.int64) if triangles is None
                 else np.array(triangles, dtype=np.int64, copy=True, order="C"))
        if source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all():
            raise ValueError("points must contain finite XYZ coordinates")
        if faces.ndim != 2 or faces.shape[1] != 3 or (
            len(faces) and (faces.min() < 0 or faces.max() >= len(source))
        ):
            raise ValueError("triangles must have valid vertex indices")
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(source.astype("<f8")).tobytes())
        digest.update(np.ascontiguousarray(faces.astype("<i8")).tobytes())
        if len(faces):
            listed = np.sort(np.vstack((faces[:, [0, 1]],
                                        faces[:, [1, 2]],
                                        faces[:, [2, 0]])), axis=1)
            edges, incidence = np.unique(listed, axis=0, return_counts=True)
            lengths = np.linalg.norm(source[edges[:, 0]] - source[edges[:, 1]], axis=1)
            areas = .5 * np.linalg.norm(np.cross(
                source[faces[:, 1]] - source[faces[:, 0]],
                source[faces[:, 2]] - source[faces[:, 0]],
            ), axis=1)
            vertex_weights = np.zeros(len(source), dtype=float)
            np.add.at(vertex_weights, faces[:, 0], areas / 3.0)
            np.add.at(vertex_weights, faces[:, 1], areas / 3.0)
            np.add.at(vertex_weights, faces[:, 2], areas / 3.0)
            positive = vertex_weights[vertex_weights > 1e-15]
            replacement = float(np.median(positive)) if len(positive) else 1.0
            vertex_weights[vertex_weights <= 1e-15] = replacement
            centroids = source[faces].mean(axis=1)
            tree = cKDTree(centroids)
            vertex_faces = coo_matrix(
                (np.ones(3 * len(faces)),
                 (faces.ravel(), np.repeat(np.arange(len(faces)), 3))),
                shape=(len(source), len(faces)),
            ).tocsr()
        else:
            edges = np.empty((0, 2), dtype=np.int64)
            incidence = np.empty(0, dtype=np.int64)
            lengths = np.empty(0, dtype=float)
            areas = np.empty(0, dtype=float)
            vertex_weights = np.ones(len(source), dtype=float)
            centroids = np.empty((0, 3), dtype=float)
            tree = None
            vertex_faces = csr_matrix((len(source), 0))
        point_tree = cKDTree(source)
        for array in (source, faces, edges, lengths, incidence, areas, vertex_weights,
                      centroids, point_tree.data, vertex_faces.data,
                      vertex_faces.indices, vertex_faces.indptr):
            _readonly(array)
        if tree is not None:
            _readonly(tree.data)
        return cls(source, faces, edges, lengths, incidence, areas, vertex_weights,
                   centroids, tree, vertex_faces, digest.hexdigest(), point_tree)

    def validate(self, points: np.ndarray, triangles: np.ndarray | None) -> None:
        source = np.asarray(points, dtype=float)
        faces = (np.empty((0, 3), dtype=np.int64) if triangles is None
                 else np.asarray(triangles, dtype=np.int64))
        if not ((source is self.points or np.array_equal(source, self.points)) and
                (faces is self.triangles or np.array_equal(faces, self.triangles))):
            raise ValueError("mesh_context must use the same ordered native geometry")

    def bounded_edges(self, d_bar: float) -> np.ndarray:
        spacing = float(d_bar)
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("d_bar must be positive and finite")
        if spacing not in self._bounded_edges:
            self._bounded_edges[spacing] = _readonly(self.edges[
                (self.edges[:, 0] != self.edges[:, 1]) &
                (self.edge_lengths <= 4.0 * spacing)
            ])
        return self._bounded_edges[spacing]

    def support_edges(self, spacing: float) -> np.ndarray:
        """Match the centerline/transection anomalous-edge threshold."""
        if spacing not in self._support_edges:
            positive = self.edge_lengths[self.edge_lengths > 0]
            limit = 6.0 * float(np.median(positive)) if len(positive) else float(spacing)
            self._support_edges[spacing] = _readonly(self.edges[self.edge_lengths <= limit])
        return self._support_edges[spacing]

    def ownership(self, labels: np.ndarray) -> OwnershipGeometryGeneration:
        """Return the current generation; a changed label starts a fresh one."""
        if not self._ownership or not np.array_equal(labels, self._ownership[0].labels):
            number = self._ownership[0].number + 1 if self._ownership else 0
            self._ownership[:] = [OwnershipGeometryGeneration(labels, self.points, number, self)]
        return self._ownership[0]

    def lengths_for(self, edges: np.ndarray) -> np.ndarray:
        """Reuse exact native lengths independently of ownership generations."""
        if edges is self.edges:
            return self.edge_lengths
        key = _edge_key(edges)
        if key not in self._policy_lengths:
            lengths = None
            if len(self.edges):
                if not self._edge_codes:
                    self._edge_codes.append(_readonly(self.edges[:, 0] * len(self.points) + self.edges[:, 1]))
                codes = edges[:, 0] * len(self.points) + edges[:, 1]
                index = np.searchsorted(self._edge_codes[0], codes)
                if np.all(index < len(self.edges)) and np.array_equal(self.edges[index], edges):
                    lengths = self.edge_lengths[index]
            if lengths is None:
                # Point-only diagnostic edges remain separate from native facts.
                lengths = np.linalg.norm(self.points[edges[:, 0]] - self.points[edges[:, 1]], axis=1)
            if len(self._policy_lengths) >= 8:
                self._policy_lengths.pop(next(iter(self._policy_lengths)))
            self._policy_lengths[key] = _readonly(lengths)
        return self._policy_lengths[key]
