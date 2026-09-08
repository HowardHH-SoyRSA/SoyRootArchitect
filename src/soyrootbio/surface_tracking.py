"""Local surface continuity and persistent cross-section tube identities."""
from __future__ import annotations

from itertools import permutations
from collections import OrderedDict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


class LocalMeshConnectivity:
    """Query induced mesh patches, including full-mesh links after reduction.

    Connectivity is evaluated inside a ball, so a distant connection through
    the crown cannot authorize a lateral-to-lateral jump.
    """

    def __init__(self, points, triangles, *, full_points=None):
        self.points = np.asarray(points, dtype=float)
        self.full_points = self.points if full_points is None else np.asarray(full_points, dtype=float)
        self.tree = cKDTree(self.full_points)
        self.analysis_tree = cKDTree(self.points)
        faces = np.asarray(triangles, dtype=int)
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            raise ValueError("Local connectivity requires triangular mesh faces")
        if faces.min() < 0 or faces.max() >= len(self.full_points):
            raise ValueError("Mesh face index outside full-resolution points")
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        self.graph = coo_matrix((np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
                               shape=(len(self.full_points), len(self.full_points))).tocsr()
        same_vertices = (self.points.shape == self.full_points.shape and
                         np.array_equal(self.points, self.full_points))
        self.mapping = (np.arange(len(self.points)) if same_vertices else
                        np.asarray(self.tree.query(self.points)[1], dtype=int))
        self.rejected_points = 0
        self._patches = OrderedDict()

    def retain(self, anchor, indices, radius, *, anchor_index=None, direction=None):
        indices = np.asarray(indices, dtype=int)
        if not len(indices):
            return indices
        # A modest surrounding collar permits paths around a curved tube wall
        # without allowing a connection far upstream through the root system.
        anchor_vertex = (int(self.tree.query(anchor)[1]) if anchor_index is None else
                         int(self.mapping[anchor_index]))
        direction_key = () if direction is None else tuple(np.round(direction, 8))
        key = (anchor_vertex, float(radius), *map(float, anchor), direction_key)
        if key in self._patches:
            patch = self._patches.pop(key)
            self._patches[key] = patch
            keep = np.isin(self.mapping[indices], patch, assume_unique=False)
            self.rejected_points += int(np.count_nonzero(~keep))
            return indices[keep]
        vertices = np.asarray(self.tree.query_ball_point(anchor, 1.6 * float(radius)), dtype=int)
        if not len(vertices):
            return indices[:0]
        vertices.sort()
        anchor_pos = np.searchsorted(vertices, anchor_vertex)
        if anchor_pos == len(vertices) or vertices[anchor_pos] != anchor_vertex:
            return indices[:0]
        _, labels = connected_components(self.graph[vertices][:, vertices], directed=False)
        selected_labels = {int(labels[anchor_pos])}
        if direction is not None and len(np.unique(labels)) > 1:
            selected_labels.update(self._nested_sheets(anchor, vertices, labels, labels[anchor_pos], direction, radius))
        selected = np.isin(labels, list(selected_labels))
        self._patches[key] = vertices[selected]
        if len(self._patches) > 2048:
            self._patches.popitem(last=False)
        mapped = self.mapping[indices]
        positions = np.searchsorted(vertices, mapped)
        contained = positions < len(vertices)
        contained[contained] &= vertices[positions[contained]] == mapped[contained]
        keep = np.zeros(len(indices), dtype=bool)
        keep[contained] = selected[positions[contained]]
        self.rejected_points += int(np.count_nonzero(~keep))
        return indices[keep]

    def _nested_sheets(self, anchor, vertices, labels, anchor_label, direction, radius):
        """Include a coaxial cavity wall inside the same biological tube.

        A cavity cap can be locally disconnected from its enclosing exterior.
        Requiring nested cross-section circles with nearly coincident centers
        distinguishes these sheets from a nearby root's separate tube.
        """
        axis = np.asarray(direction, dtype=float)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        helper = np.eye(3)[int(np.argmin(np.abs(axis)))]
        u = np.cross(axis, helper)
        u /= max(np.linalg.norm(u), 1e-12)
        basis = np.vstack([u, np.cross(axis, u)])
        offsets = self.full_points[vertices] - anchor
        axial = offsets @ axis
        votes = {}
        # Follow nesting across sections behind and at the cap. A cap fills
        # the inner circle, so its radial residual can exceed a wall's; the
        # enclosing exterior must still be well fit in multiple sections.
        for station in (-.40, -.20, 0.):
            slab = np.abs(axial-station*radius) <= .20 * float(radius)
            fits = {}
            for label in np.unique(labels[slab]):
                samples = offsets[slab & (labels == label)] @ basis.T
                if len(samples) >= 10:
                    fit = _circle(samples)
                    if fit is not None:
                        fits[int(label)] = fit
            own = fits.get(int(anchor_label))
            if own is None:
                continue
            for label, other in fits.items():
                inner, outer = sorted([own, other], key=lambda fit: fit[1])
                distance = np.linalg.norm(own[0]-other[0])
                if (outer[2] <= .25*outer[1] and inner[2] <= .45*inner[1] and
                        distance <= .35*outer[1] and distance+inner[1] < .90*outer[1]):
                    votes[label] = votes.get(label, 0)+1
        return {label for label, count in votes.items() if count >= 2}


def _circle(points):
    if len(points) < 6:
        return None
    a = np.column_stack((2 * points, np.ones(len(points))))
    if np.linalg.matrix_rank(a) < 3:
        return None
    values = np.linalg.lstsq(a, np.sum(points**2, axis=1), rcond=None)[0]
    center = values[:2]
    distances = np.linalg.norm(points-center, axis=1)
    radius = float(np.median(distances))
    if not np.isfinite(radius) or radius <= 1e-12:
        return None
    return center, radius, float(np.median(np.abs(distances-radius)))


def section_modes(points):
    """Fit one or two boundary circles; do not split an ordinary single ring."""
    points = np.asarray(points, dtype=float)
    one = _circle(points)
    if one is None:
        return np.asarray([np.median(points, axis=0)]), np.array([0.])
    center, radius, residual = one
    if len(points) < 24:
        return center[None], np.array([radius])
    _, _, axes = np.linalg.svd(points-np.mean(points, axis=0), full_matrices=False)
    projection = points @ axes[0]
    groups = projection > np.median(projection)
    fits = None
    for _ in range(8):
        fits = [_circle(points[groups == group]) for group in (False, True)]
        if any(fit is None for fit in fits):
            return center[None], np.array([radius])
        errors = np.column_stack([np.abs(np.linalg.norm(points-fit[0], axis=1)-fit[1]) for fit in fits])
        new_groups = np.argmin(errors, axis=1).astype(bool)
        if np.array_equal(groups, new_groups):
            break
        groups = new_groups
    centers = np.array([fit[0] for fit in fits])
    radii = np.array([fit[1] for fit in fits])
    separation = np.linalg.norm(centers[0]-centers[1])
    dual_residual = float(np.median(np.min(np.column_stack([
        np.abs(np.linalg.norm(points-c, axis=1)-r) for c, r in zip(centers, radii)]), axis=1)))
    if (min(np.count_nonzero(groups), np.count_nonzero(~groups)) < 10 or
            separation < .65 * radii.sum() or radii.min() < .35*radii.max() or
            dual_residual >= .55 * max(residual, 1e-12)):
        return center[None], np.array([radius])
    return centers, radii


class TubeSectionTracker:
    """Track both fitted tube centers with prediction and joint association.

    Index zero always denotes the incoming root. Unresolved merged sections
    preserve predictions instead of collapsing two biological identities.
    """

    def __init__(self):
        self.centers = None
        self.velocity = None
        self.station = None
        self.radii = None
        self.multimode_sections = 0
        self.ambiguous_sections = 0

    def update(self, radial, station, basis, fallback):
        offsets, radii = section_modes(radial)
        modes = station + offsets @ basis
        drift = np.zeros(3) if self.station is None else station-self.station
        predicted = None if self.centers is None else self.centers + drift + self.velocity
        if predicted is not None and len(predicted) == 2 and len(modes) == 1:
            # Preserve two identities only while the incoming prediction still
            # has observed boundary support. Missing tube support is flagged.
            distances = np.linalg.norm(radial - (predicted[0]-station) @ basis.T, axis=1)
            if np.median(np.abs(distances-self.radii[0])) < .4*self.radii[0]:
                self.centers = predicted
                self.station = station.copy()
                self.velocity *= .5
                self.ambiguous_sections += 1
                return (predicted[0]-station) @ basis.T
        if len(modes) == 2:
            self.multimode_sections += 1
            if predicted is None:
                order = np.argsort(np.linalg.norm(modes-station, axis=1))
            else:
                order = min(permutations(range(2)), key=lambda order:
                            sum(np.linalg.norm(modes[j]-predicted[i]) for i, j in enumerate(order[:len(predicted)])))
            modes, radii = modes[list(order)], radii[list(order)]
        if predicted is not None and len(predicted) == len(modes):
            innovation = modes - (self.centers + drift)
            self.velocity = .5*self.velocity + .5*innovation
        else:
            self.velocity = np.zeros_like(modes)
        self.centers, self.radii, self.station = modes, radii, station.copy()
        return (modes[0]-station) @ basis.T if len(modes) == 2 else fallback


def resolve_endpoint_fragments(established, fragments, surface, d_bar):
    """Jointly associate short outgoing fragments with established endpoints.

    Existing basal identities are never removed. Competing assignments remain
    provisional; true departing branches fail the continuation-angle gate.
    """
    if surface is None or not established or not fragments:
        return fragments
    proposals = []
    for root in established:
        if len(root.points) < 4:
            continue
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(root.points, axis=0), axis=1))]
        window = max(10*d_bar, .02*root.length)
        i = max(0, np.searchsorted(arc, arc[-1]-window)-1)
        incoming = root.points[-1]-root.points[i]
        incoming /= max(np.linalg.norm(incoming), 1e-12)
        limit = max(18*d_bar, .008)
        for fragment in fragments:
            if fragment is root or len(fragment.points) < 4 or fragment.parent_id == 'primary':
                continue
            # The first node is a synthetic attachment, not surface evidence.
            start = fragment.points[1]
            gap = np.linalg.norm(start-root.points[-1])
            if gap > limit:
                continue
            j = min(len(fragment.points)-1, 4)
            outgoing = fragment.points[j]-start
            outgoing /= max(np.linalg.norm(outgoing), 1e-12)
            alignment = float(incoming @ outgoing)
            bridge = start-root.points[-1]
            if alignment < np.cos(np.deg2rad(30)) or incoming @ bridge < -.5*d_bar:
                continue
            _, node = surface.analysis_tree.query(start)
            if not len(surface.retain(root.points[-1], [node], limit, direction=incoming)):
                continue
            score = alignment - .25*gap/limit
            proposals.append((score, root, fragment))
    consumed = set()
    extended = set()
    for score, root, fragment in sorted(proposals, key=lambda x: (-x[0], x[1].root_id, x[2].root_id)):
        if id(fragment) in consumed or id(root) in extended:
            continue
        alternatives = [s for s, r, f in proposals if (r is root or f is fragment) and not (r is root and f is fragment)]
        if alternatives and score-max(alternatives) < .12:
            if 'endpoint_continuation_ambiguous' not in root.qc_flags:
                root.qc_flags.append('endpoint_continuation_ambiguous')
            continue
        root.points = np.vstack((root.points, fragment.points[1:]))
        root.covered_indices.update(fragment.covered_indices)
        if root.novel_support_indices is not None:
            root.novel_support_indices.update(fragment.novel_support_indices or ())
        root.node_indices = None
        root.score_components['endpoint_fragments_joined'] = root.score_components.get('endpoint_fragments_joined', 0.)+1
        for child in fragments:
            if child.parent_id == fragment.root_id:
                child.parent_id = root.root_id
                child.parent_points = root.points
        consumed.add(id(fragment))
        extended.add(id(root))
    return [f for f in fragments if id(f) not in consumed]
