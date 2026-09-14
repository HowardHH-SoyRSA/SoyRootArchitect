"""Exact, bounded distinct-root competition against polyline segments.

Candidate balls are conservative: a segment within D of a point has its
midpoint within D + half_length. Length buckets keep a sparse, long segment
from enlarging every query. There is no fixed-k truncation. Reduction first
finds the closest centerline projection per root, then ranks roots by the
absolute residual to their local surface radius. Unknown radii are zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .runtime import worker_threads


TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class RootCompetition:
    labels: np.ndarray
    distances: np.ndarray
    centerline_distances: np.ndarray
    radii: np.ndarray

    def ambiguous(self, margin: float) -> np.ndarray:
        valid = self.labels[:, 1] >= 0
        result = np.zeros(len(valid), dtype=bool)
        result[valid] = (self.distances[valid, 1] - self.distances[valid, 0]
                         <= margin + TIE_TOLERANCE)
        return result


class RootSegmentIndex:
    """Index paths with unique numeric identities and optional node radii.

    Empty paths are ignored; singleton paths are point segments. Endpoint
    projections are clamped. Repeated coordinates use the first endpoint's
    radius. Ties are quantized at 1e-12 coordinate units and resolved by root
    label, then segment order. Memory is bounded by a query chunk's candidate
    neighborhood, never a full points-by-segments matrix.
    """

    def __init__(self, paths, *, labels=None, radii=None):
        paths = list(paths)
        labels = list(range(len(paths))) if labels is None else list(labels)
        if len(labels) != len(paths) or len(set(labels)) != len(labels) or any(x < 0 for x in labels):
            raise ValueError("paths require unique non-negative root labels")
        radii = [None] * len(paths) if radii is None else list(radii)
        if len(radii) != len(paths):
            raise ValueError("one radius profile is required per root")
        starts, ends, owners, r0, r1 = [], [], [], [], []
        for path, label, profile in zip(paths, labels, radii, strict=True):
            path = np.asarray(path, dtype=float)
            if path.ndim != 2 or path.shape[1] != 3 or not np.all(np.isfinite(path)):
                raise ValueError("paths must contain finite XYZ coordinates")
            rr = np.zeros(len(path)) if profile is None else np.broadcast_to(profile, (len(path),))
            if not np.all(np.isfinite(rr)) or np.any(rr < 0):
                raise ValueError("radii must be finite and non-negative")
            if not len(path):
                continue
            a = path[:-1] if len(path) > 1 else path
            b = path[1:] if len(path) > 1 else path
            starts.extend(a)
            ends.extend(b)
            owners.extend([int(label)] * len(a))
            r0.extend(rr[:-1] if len(path) > 1 else rr)
            r1.extend(rr[1:] if len(path) > 1 else rr)
        self.starts = np.asarray(starts, dtype=float).reshape(-1, 3)
        self.vectors = np.asarray(ends, dtype=float).reshape(-1, 3) - self.starts
        self.length2 = np.einsum("ij,ij->i", self.vectors, self.vectors)
        self.labels = np.asarray(owners, dtype=int)
        self.r0, self.r1 = np.asarray(r0), np.asarray(r1)
        self.groups = []
        if not len(starts):
            return
        half = np.sqrt(self.length2) / 2
        # The global radius maximum ensures a closer segment is included even if
        # it has a smaller local radius than the initially relevant segment.
        max_radius = max(float(self.r0.max()), float(self.r1.max()))
        bucket = np.floor(np.log2(np.maximum(half, 1e-12))).astype(int)
        midpoints = self.starts + self.vectors / 2
        for key in np.unique(bucket):
            ids = np.flatnonzero(bucket == key)
            self.groups.append((ids, cKDTree(midpoints[ids]), float(half[ids].max()) + max_radius))

    def query(self, points, *, max_distance: float, chunk_size: int = 2048) -> RootCompetition:
        """Return two distinct roots within the surface-residual bound.

        Missing alternatives are (-1, inf). All roots that can meet the bound
        are evaluated, including sparse segments whose endpoints are far away.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("points must contain finite XYZ coordinates")
        if not np.isfinite(max_distance) or max_distance < 0 or chunk_size < 1:
            raise ValueError("query bound must be finite and non-negative; chunk_size must be positive")
        labels = np.full((len(points), 2), -1, dtype=int)
        distances = np.full((len(points), 2), np.inf)
        centers = distances.copy()
        radii = np.zeros_like(distances)
        for start in range(0, len(points), chunk_size):
            chunk = points[start:start + chunk_size]
            rows, segments = [], []
            for ids, tree, padding in self.groups:
                neighbors = tree.query_ball_point(chunk, max_distance + padding + TIE_TOLERANCE,
                                                 workers=worker_threads())
                counts = np.fromiter(map(len, neighbors), dtype=int, count=len(chunk))
                if counts.sum():
                    rows.append(np.repeat(np.arange(len(chunk)), counts))
                    segments.append(ids[np.concatenate(neighbors).astype(int)])
            if not rows:
                continue
            row, seg = np.concatenate(rows), np.concatenate(segments)
            delta = chunk[row] - self.starts[seg]
            parameter = np.divide(np.einsum("ij,ij->i", delta, self.vectors[seg]),
                                  self.length2[seg], out=np.zeros(len(seg)), where=self.length2[seg] > 0)
            parameter = np.clip(parameter, 0, 1)
            distance = np.linalg.norm(delta - parameter[:, None] * self.vectors[seg], axis=1)
            radius = self.r0[seg] + parameter * (self.r1[seg] - self.r0[seg])
            owner = self.labels[seg]
            order = np.lexsort((seg, np.rint(distance / TIE_TOLERANCE), owner, row))
            first = np.r_[True, (row[order[1:]] != row[order[:-1]]) | (owner[order[1:]] != owner[order[:-1]])]
            chosen = order[first]
            residual = np.abs(distance[chosen] - radius[chosen])
            chosen = chosen[residual <= max_distance + TIE_TOLERANCE]
            residual = np.abs(distance[chosen] - radius[chosen])
            order = np.lexsort((owner[chosen], np.rint(residual / TIE_TOLERANCE), row[chosen]))
            chosen = chosen[order]
            if not len(chosen):
                continue
            local_rows = row[chosen]
            offsets = np.maximum.accumulate(np.where(np.r_[True, np.diff(local_rows) != 0], np.arange(len(chosen)), 0))
            rank = np.arange(len(chosen)) - offsets
            keep = rank < 2
            ix, slot, chosen = local_rows[keep] + start, rank[keep], chosen[keep]
            labels[ix, slot] = owner[chosen]
            centers[ix, slot] = distance[chosen]
            radii[ix, slot] = radius[chosen]
            distances[ix, slot] = np.abs(distance[chosen] - radius[chosen])
        return RootCompetition(labels, distances, centers, radii)
