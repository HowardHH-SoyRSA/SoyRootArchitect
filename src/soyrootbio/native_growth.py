"""Bound lateral growth to local native surface routes, without mesh bridges."""
from collections import OrderedDict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


class NativeStepGuard:
    """Frozen geometry/occupancy snapshot shared by one tracing order.

    A route must stay in the local search ball and cannot traverse occupied
    parent surface. Global connectivity through the primary is insufficient.
    Missing native correspondence is an error, never a proximity mapping.
    """

    def __init__(self, context, analysis_points, analysis_to_mesh, blocked, excluded, spacing):
        context.validate(context.points, context.triangles)
        self.context = context
        self.spacing = float(spacing)
        mapping = (np.arange(len(analysis_points)) if analysis_to_mesh is None
                   and len(analysis_points) == len(context.points) else analysis_to_mesh)
        if mapping is None:
            raise ValueError("Native growth requires analysis vertex correspondence")
        self.mapping = np.asarray(mapping, dtype=int)
        if (self.mapping.shape != (len(analysis_points),) or np.any(self.mapping < 0)
                or np.any(self.mapping >= len(context.points))
                or not np.array_equal(analysis_points, context.points[self.mapping])):
            raise ValueError("Native growth requires exact analysis vertex correspondence")
        self.excluded = np.asarray(excluded, bool).copy()
        if self.excluded.shape != (len(context.points),):
            raise ValueError("Native growth exclusions must match native vertices")
        self.excluded[np.unique(context.edges[context.edge_incidence != 2])] = True
        self.blocked = self.excluded.copy()
        self.blocked[self.mapping[np.asarray(blocked, bool)]] = True
        e = context.edges
        w = np.maximum(context.edge_lengths, 1e-15)
        self.graph = coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]],
                                             np.r_[e[:, 1], e[:, 0]])),
                                shape=(len(context.points), len(context.points))).tocsr()
        self.cache = OrderedDict()
        self.report = {"policy": "bounded-native-growth-v1", "queries": 0,
                       "rejected_proposals": 0, "unresolved_anchors": 0}

    def reachable(self, current, candidates, radius, *, junction_origin=None,
                  junction_seed=None, analysis_anchor=None):
        """Return which analysis vertices have a bounded native route."""
        candidates = np.asarray(candidates, int)
        self.report["queries"] += 1
        if analysis_anchor is None:
            gap, anchor = self.context.point_tree.query(current)
            anchor = int(anchor)
        else:
            anchor = int(self.mapping[int(analysis_anchor)])
            gap = np.linalg.norm(self.context.points[anchor] - current)
        if self.excluded[anchor] or gap > max(4 * self.spacing, .5 * radius):
            self.report["unresolved_anchors"] += 1
            self.report["rejected_proposals"] += len(candidates)
            return np.zeros(len(candidates), bool)
        seed = self.context.points[anchor] if junction_seed is None else np.asarray(junction_seed, float)
        key = (anchor, float(radius), None if junction_origin is None
               else (tuple(np.asarray(junction_origin, float)), tuple(seed)))
        if key in self.cache:
            reachable = self.cache.pop(key)
        else:
            # Center on the immutable native anchor, so the cache is exact.
            local = np.asarray(self.context.point_tree.query_ball_point(
                self.context.points[anchor], 1.25 * radius), int)
            allowed = ~self.blocked[local]
            if junction_origin is not None:
                # The provisional parent mask can absorb a branch neck.
                # Only a bounded outward corridor from the original seed may
                # traverse that neck: never route through the parent core.
                outward = seed - junction_origin
                distance = float(np.linalg.norm(outward))
                outward /= max(distance, 1e-12)
                radial = np.linalg.norm(self.context.points[local] - junction_origin, axis=1)
                progress = (self.context.points[local] - seed) @ outward
                allowed |= ((radial >= distance - 2 * self.spacing)
                            & (progress >= -2 * self.spacing)
                            & (np.linalg.norm(self.context.points[local] - seed, axis=1) <= 2 * radius)
                            & ~self.excluded[local])
            local = np.sort(local[allowed | (local == anchor)])
            graph = self.graph[local][:, local]
            distance = dijkstra(graph, directed=False,
                                indices=int(np.searchsorted(local, anchor)),
                                limit=3 * radius)
            reachable = local[np.isfinite(distance)]
        self.cache[key] = reachable
        while len(self.cache) > 2048:
            self.cache.popitem(last=False)
        result = np.isin(self.mapping[candidates], reachable)
        self.report["rejected_proposals"] += int(np.count_nonzero(~result))
        return result
