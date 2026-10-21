"""Resolve shared O1 suffixes using frozen, native transverse tube evidence.

No reference labels, sample coordinates, or length-based deletion are used.
Original hypotheses remain in the decision report; unsupported sections stop
tracking instead of bridging a gap. Geometry and exclusion masks are read-only.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from soyrootbio.primary import _plane_basis
from soyrootbio.geometry import resample_polyline, tangent_vectors
from soyrootbio.junction_tubes import vertex_normals
from soyrootbio.primary import refine_primary_centerline
from soyrootbio.geometry import path_length

class NativeTubeTracker:
    """Follow one tube from closed native contours and measured wall arcs.

    All distances are in analysis coordinates; bounds use measured radius or
    point spacing. Open contours and nonmanifold edges never supply a partial
    section. A missing section stops the trace rather than being interpolated.
    """

    def __init__(self, context, excluded, spacing):
        self.c = context
        self.p = context.points
        self.f = context.triangles
        self.s = float(spacing)
        if not np.isfinite(self.s) or self.s <= 0:
            raise ValueError("spacing must be finite and positive")
        if self.f is None or not len(self.f):
            raise ValueError("Native tube tracking requires triangles")
        self.normals = vertex_normals(self.p, self.f)
        self.incident = context.vertex_face_incidence
        self.good = ~np.asarray(excluded, bool).copy()
        if self.good.shape != (len(self.p),):
            raise ValueError('excluded must match native vertices')
        unsafe = np.unique(context.edges[context.edge_incidence != 2])
        self.good[unsafe] = False

    def cut(self, station, tangent, reach):
        """Intersect native faces; retain complete closed components only."""
        station = station + tangent * self.s * 1e-06
        nearby = self.c.point_tree.query_ball_point(station, reach)
        face_ids = np.unique(self.incident[nearby].indices)
        tri = self.f[face_ids]
        tri = tri[self.good[tri].all(axis=1)]
        heights = (self.p[tri] - station) @ tangent
        cross = (heights.min(axis=1) < 0) & (heights.max(axis=1) > 0)
        tri, heights = (tri[cross], heights[cross])
        if len(tri) < 6:
            return ([], None)
        e = tri[:, [[0, 1], [1, 2], [2, 0]]]
        h = heights[:, [[0, 1], [1, 2], [2, 0]]]
        cut = h[:, :, 0] * h[:, :, 1] < 0
        if np.any(cut.sum(axis=1) != 2):
            return ([], None)
        native, node = np.unique(np.sort(e[cut], axis=1), axis=0, return_inverse=True)
        seg = node.reshape(-1, 2)
        g = coo_matrix((np.ones(len(seg) * 2), (np.r_[seg[:, 0], seg[:, 1]], np.r_[seg[:, 1], seg[:, 0]])), shape=(len(native), len(native))).tocsr()
        n, part = connected_components(g, directed=False)
        a, b = (self.p[native[:, 0]], self.p[native[:, 1]])
        ha, hb = ((a - station) @ tangent, (b - station) @ tangent)
        xyz = a + (ha / (ha - hb))[:, None] * (b - a)
        nn = self.normals[native].mean(axis=1)
        nn /= np.maximum(np.linalg.norm(nn, axis=1)[:, None], 1e-12)
        basis = _plane_basis(tangent)
        uv = (xyz - station) @ basis.T
        choices = []
        closed = np.zeros(len(native), bool)
        for component in range(n):
            nodes = np.flatnonzero(part == component)
            if len(nodes) < 6 or np.any(np.diff(g.indptr)[nodes] != 2) or np.any(g[nodes].data != 1):
                continue
            sequence = [int(nodes[0])]
            prev = -1
            cur = sequence[0]
            while True:
                ns = g.indices[g.indptr[cur]:g.indptr[cur + 1]]
                nxt = int(ns[0] if ns[0] != prev else ns[1])
                if nxt == sequence[0]:
                    break
                if nxt in sequence:
                    sequence = []
                    break
                sequence.append(nxt)
                prev, cur = (cur, nxt)
            if len(sequence) != len(nodes):
                continue
            closed[nodes] = True
            polygon = uv[sequence]
            nxt = np.roll(polygon, -1, axis=0)
            cross = polygon[:, 0] * nxt[:, 1] - nxt[:, 0] * polygon[:, 1]
            area = 0.5 * cross.sum()
            if abs(area) < 0.15 * self.s ** 2:
                continue
            center = ((polygon + nxt) * cross[:, None]).sum(axis=0) / (6 * area)
            radial = np.linalg.norm(polygon - center, axis=1)
            perim = np.linalg.norm(nxt - polygon, axis=1).sum()
            compact = perim ** 2 / (4 * np.pi * abs(area))
            if compact > 1.25 or radial.max() > 2 * np.median(radial):
                continue
            choices.append(dict(center=station + center @ basis, radius=float(np.median(radial)), compact=compact, vertices=np.unique(native[nodes]), normals=nn[nodes], partial=False))
        return (choices, (station, uv[closed], nn[closed], native[closed], basis))

    def partial(self, raw, radius):
        """Fit a wall arc of a merged closed section, with independent normals.

        More than half a circumference (198 degrees), bounded center/radius
        changes and radial-normal agreement are required. This supplies local
        center evidence, never permission to relabel a missing contact wall.
        """
        if raw is None or len(raw[1]) < 8:
            return None
        station, uv, nn, native, basis = raw
        n = nn @ basis.T
        n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-12)
        rad = np.linalg.norm(uv, axis=1)
        keep = (abs(rad - radius) < max(0.4 * radius, self.s)) & (np.abs(nn @ np.cross(basis[0], basis[1])) < 0.7)
        keep &= np.abs(np.einsum('ij,ij->i', n, uv) / np.maximum(rad, 1e-12)) > 0.7
        if keep.sum() < 8:
            return None
        trans = np.column_stack([-n[:, 1], n[:, 0]])
        target = np.einsum('ij,ij->i', trans, uv)
        for _ in range(3):
            shift, *_ = np.linalg.lstsq(trans[keep], target[keep], rcond=None)
            radial = np.linalg.norm(uv - shift, axis=1)
            measured = float(np.median(radial[keep]))
            new = (abs(radial - measured) < max(0.2 * measured, 0.65 * self.s)) & (np.abs(nn @ np.cross(basis[0], basis[1])) < 0.7)
            new &= np.abs(np.einsum('ij,ij->i', n, uv - shift) / np.maximum(radial, 1e-12)) > 0.8
            keep = new & (rad < 1.7 * radius)
            if keep.sum() < 8:
                return None
        angles = np.sort(np.arctan2((uv - shift)[keep, 1], (uv - shift)[keep, 0]))
        coverage = 1 - np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max() / (2 * np.pi)
        if coverage < 0.55 or np.linalg.norm(shift) > 0.55 * radius or abs(measured - radius) > 0.35 * radius:
            return None
        return dict(center=station + shift @ basis, radius=measured, compact=0, vertices=np.unique(native[keep]), normals=nn[keep], partial=True, coverage=coverage)

    def track(self, guide, seed=None, cooperate=None):
        """Use the guide to seed a tube, then follow measured sections."""
        s = self.s
        guide = resample_polyline(guide, 2 * s)
        tangents = tangent_vectors(guide)
        first = None
        diagnostics = []
        begin = 6 if seed is None else max(6, int(np.argmin(np.linalg.norm(guide - seed, axis=1))) + 3)
        for i in range(begin, min(len(guide) - 2, begin + 30)):
            choices, raw = self.cut(guide[i], tangents[i], max(12 * s, 0.012))
            choices = [x for x in choices if np.linalg.norm(x['center'] - guide[i]) < max(4 * s, x['radius']) and x['radius'] > s]
            if not choices:
                continue
            choices.sort(key=lambda x: np.linalg.norm(x['center'] - guide[i]))
            first = choices[0]
            start_i = i
            break
        if first is None:
            return (guide, [], {'status': 'no_seed'})
        center = first['center']
        radius = first['radius']
        direction = tangents[start_i]
        centers = [center.copy()]
        radii = [radius]
        walls = [first['vertices']]
        partial_count = 0
        maximum = max(1.2 * np.linalg.norm(np.diff(guide, axis=0), axis=1).sum() + 10 * s, np.linalg.norm(np.ptp(self.p, axis=0)))
        for step in range(min(1600, int(maximum / (2 * s)))):
            if cooperate is not None and step % 16 == 0:
                cooperate()
            prediction = center + 2 * s * direction
            reach = max(3 * radius, 10 * s)
            choices, raw = self.cut(prediction, direction, reach)
            stable = float(np.median(radii[-8:]))
            choices = [x for x in choices if np.linalg.norm(x['center'] - prediction) < max(0.9 * stable, 1.5 * s) and 0.45 * stable < x['radius'] < 1.4 * stable]
            for x in choices:
                x['cost'] = np.linalg.norm(x['center'] - prediction) / stable + abs(np.log(x['radius'] / stable))
            choices.sort(key=lambda x: x['cost'])
            chosen = choices[0] if choices else self.partial(raw, stable)
            if chosen is None:
                status = 'missing_or_merged_contour'
                diagnostics.append(dict(failed_prediction=prediction.tolist(), direction=direction.tolist(), radius=stable))
                break
            if len(choices) > 1 and choices[1]['cost'] - choices[0]['cost'] < 0.25:
                status = 'ambiguous_contours'
                break
            new_center = chosen['center']
            radius = chosen['radius']
            nn = chosen['normals']
            eig, vec = np.linalg.eigh(nn.T @ nn)
            axis = vec[:, 0]
            if axis @ direction < 0:
                axis = -axis
            if not chosen['partial'] and eig[1] > 0.15 * eig[2] and (eig[0] < 0.6 * eig[1]) and (axis @ direction > 0.7):
                direction = 0.5 * direction + 0.5 * axis
                direction /= np.linalg.norm(direction)
            elif len(centers) > 3:
                axis = new_center - centers[-3]
                axis /= max(np.linalg.norm(axis), 1e-12)
                direction = 0.7 * direction + 0.3 * axis
                direction /= np.linalg.norm(direction)
            center = new_center
            centers.append(center.copy())
            radii.append(radius)
            walls.append(chosen['vertices'])
            partial_count += int(chosen['partial'])
            diagnostics.append(dict(point=center.tolist(), radius=radius, partial=chosen['partial']))
            if radius < 0.8 * s:
                status = 'tapered_tip'
                break
        else:
            status = 'limit'
        return (np.vstack([guide[:start_i], centers]), np.unique(np.concatenate(walls)), dict(status=status, partial_sections=partial_count, stations=len(centers), diagnostics=diagnostics, seed_index=start_i, radius_profile=np.r_[np.full(start_i, radii[0]), radii].tolist()))

    def sustained(self, trace):
        """Accept long witnesses or a compact, closed, independently capped tube.

        A short witness needs a native cap, not just an early tracking failure.
        Excluded/open/nonmanifold boundaries cannot act as fabricated tips.
        """
        line, wall, detail = trace
        if detail.get("stations", 0) >= 20:
            return True, "sustained_sections"
        if (detail.get("stations", 0) < 6 or detail.get("partial_sections", 0)
                or detail.get("status") not in ("missing_or_merged_contour", "tapered_tip")):
            return False, "insufficient_closed_sections"
        body = line[detail["seed_index"]:]
        radius = float(np.median(detail["radius_profile"][detail["seed_index"]:]))
        if path_length(body) < max(6 * self.s, 4 * radius):
            return False, "insufficient_radius_scaled_arc"
        axis = body[-1] - body[-min(4, len(body))]
        axis /= max(np.linalg.norm(axis), 1e-12)
        local = np.asarray(self.c.point_tree.query_ball_point(body[-1], 3 * radius + 4 * self.s), int)
        # Identify the native component of the observed wall in this local ball.
        edges = self.c.edges[np.isin(self.c.edges, local).all(axis=1)]
        graph = coo_matrix((np.ones(len(edges)), (np.searchsorted(np.sort(local), edges[:, 0]),
                                                np.searchsorted(np.sort(local), edges[:, 1]))),
                           shape=(len(local), len(local))).tocsr()
        local = np.sort(local)
        _, components = connected_components(graph, directed=False)
        witnesses = np.isin(local, wall)
        if not np.any(witnesses):
            return False, "no_terminal_native_component"
        component = int(np.argmax(np.bincount(components[witnesses])))
        vertices = local[components == component]
        axial = (self.p[vertices] - body[-1]) @ axis
        ahead = axial >= -self.s
        if np.any(~self.good[vertices[ahead]]):
            return False, "terminal_boundary_unresolved"
        cap = ahead & (np.abs(self.normals[vertices] @ axis) >= .65)
        accepted = (np.count_nonzero(cap) >= 3 and
                    np.max(axial) <= max(3 * self.s, .85 * radius))
        return bool(accepted), "closed_native_cap" if accepted else "terminal_cap_unproven"

def reconcile_shared_o1_tubes(candidates, analysis_points, primary_mask, *,
                              context, analysis_to_mesh, excluded, spacing,
                              cooperate=None):
    """Repair a mixed basal/distal identity before greedy O1 selection.

    A shared support set is only a trigger. A correction additionally needs a
    backwards native-contour trace that misses its own surface seed, reaches
    another independently primary-attached seed, and a separate sustained
    forward tube from the first seed. Both roots are prepared from the same
    immutable mesh before any candidate is changed. Reference labels are never
    consumed. Point-only, open/nonmanifold and ambiguous evidence abstain.
    """
    report = {"policy": "native-reciprocal-o1-tube-identity-v1", "pairs": [],
              "corrected_candidate_ids": [], "status": "no_shared_suffix"}
    if context is None or context.triangles is None or not len(context.triangles):
        report["status"] = "unresolved_no_mesh"
        return report
    context.validate(context.points, context.triangles)
    mapping = (np.arange(len(analysis_points)) if analysis_to_mesh is None
               and len(analysis_points) == len(context.points) else analysis_to_mesh)
    if mapping is None:
        report["status"] = "unresolved_no_analysis_mapping"
        return report
    mapping = np.asarray(mapping, dtype=int)
    if (mapping.shape != (len(analysis_points),) or np.any(mapping < 0)
            or np.any(mapping >= len(context.points))
            or not np.array_equal(analysis_points, context.points[mapping])):
        raise ValueError("Native tube tracing requires exact analysis vertex correspondence")
    eligible = [i for i, root in enumerate(candidates)
                if root.order == 1 and root.parent_id == "primary"
                and root.raw_start_point is not None and len(root.points) >= 8
                and root.score_components.get("surface_aware_seed", 0) > 0
                and root.score_components.get("surface_seed_contact_count", 0) >= 3
                and root.score_components.get("surface_seed_gap", np.inf) <= 2.5 * spacing]
    support = {i: set(candidates[i].novel_support_indices
                      if candidates[i].novel_support_indices is not None
                      else candidates[i].covered_indices) for i in eligible}
    pairs = []
    for offset, i in enumerate(eligible):
        for j in eligible[:offset]:
            if np.linalg.norm(candidates[i].points[0] - candidates[j].points[0]) <= 8 * spacing:
                continue
            shared = len(support[i] & support[j])
            fraction = shared / max(1, min(len(support[i]), len(support[j])))
            if shared >= 40 and fraction > .40:
                pairs.append((j, i, shared, fraction))
    if not pairs:
        return report
    tracker = NativeTubeTracker(context, excluded, spacing)
    guides, reverse = {}, {}
    for i in sorted({i for pair in pairs for i in pair[:2]}, key=lambda i: candidates[i].root_id):
        if cooperate is not None:
            cooperate()
        root = candidates[i]
        mask = np.zeros(len(analysis_points), dtype=bool)
        ids = np.asarray(sorted(root.covered_indices), dtype=int)
        mask[ids] = True
        guide = refine_primary_centerline(analysis_points, mask, root.points, spacing,
                                         max_stations=240, min_slice_points=6,
                                         cooperate=cooperate)
        guides[i] = guide
        reverse[i] = tracker.track(guide[::-1], cooperate=cooperate)

    def seed_fraction(wall, root):
        nearby = np.asarray(context.point_tree.query_ball_point(root.raw_start_point, 4 * spacing), int)
        nearby = nearby[tracker.good[nearby]]
        if len(nearby) < 6:
            return 0.0
        return float(np.mean(np.isin(nearby, wall)))

    proposals = {}
    conflicts = {}
    supported_prefixes = {}
    for i, j, shared, fraction in sorted(pairs, key=lambda row: tuple(sorted(
            (candidates[row[0]].root_id, candidates[row[1]].root_id)))):
        row = {"candidate_ids": [candidates[i].root_id, candidates[j].root_id],
               "shared_support_count": shared, "shared_support_fraction": fraction,
               "action": "unchanged"}
        hits = [[seed_fraction(reverse[a][1], candidates[b]) for b in (i, j)] for a in (i, j)]
        row["reverse_seed_support_fraction"] = hits
        report["pairs"].append(row)
        # Reciprocal evidence distinguishes a duplicate seed on one real tube
        # from a trace whose distal arm belongs to a different primary origin.
        if hits[0][0] < .15 and hits[0][1] >= .70 and hits[1][1] >= .70:
            mixed, owner = i, j
        elif hits[1][1] < .15 and hits[1][0] >= .70 and hits[0][0] >= .70:
            mixed, owner = j, i
        else:
            row["reason"] = "no_reciprocal_identity_conflict"
            continue
        conflicts.setdefault(mixed, set()).update(support[mixed] & support[owner])
        row.update(action="identity_conflict_unresolved",
                   mixed_candidate=candidates[mixed].root_id,
                   distal_owner_candidate=candidates[owner].root_id,
                   original_mixed_polyline_normalized=candidates[mixed].points.tolist())
        if mixed in proposals:
            row["reason"] = "same_mixed_candidate_already_prepared"
            continue
        basal = tracker.track(guides[mixed], seed=candidates[mixed].raw_start_point,
                              cooperate=cooperate)
        if (basal[2].get("stations", 0) >= 6 and len(basal[1]) >= 30
                and basal[2]["status"] not in ("limit", "ambiguous_contours")):
            prefix = basal[0].copy()
            prefix[0] = candidates[mixed].points[0]
            supported_prefixes[mixed] = (prefix, basal[1], basal[2])
        back_line, back_wall, back_report = reverse[owner]
        if basal[2]["status"] == "limit" or back_report["status"] == "limit":
            row["reason"] = "native_trace_limit_unresolved"
            continue
        basal_ok, basal_reason = tracker.sustained(basal)
        back_ok, back_reason = tracker.sustained(reverse[owner])
        row.update(mixed_native_support=basal_reason, owner_native_support=back_reason,
                   mixed_native_stations=basal[2].get("stations", 0))
        if not basal_ok or not back_ok:
            row["reason"] = "insufficient_sustained_native_tube"
            # A disproved mixed suffix must not suppress the independent
            # owner while its basal reconstruction is still unresolved.
            # Retain the original path for QC but remove only shared support
            # from its selection score. No surface labels or geometry change.
            row["action"] = "reserve_independently_supported_suffix"
            continue
        # Trace the stable distal tube forwards as well, so a capped greedy
        # trace does not become a newly truncated counterpart.
        if len(guides[owner]) < 14:
            row["reason"] = "insufficient_distal_guide"
            continue
        distal = tracker.track(guides[owner], seed=guides[owner][-12], cooperate=cooperate)
        if distal[2].get("stations", 0) < 3 or distal[2]["status"] == "limit":
            row["reason"] = "no_supported_distal_extension"
            continue
        left = back_line[back_report["seed_index"]:][::-1]
        right = distal[0][distal[2]["seed_index"]:]
        distances, nearest = cKDTree(right).query(left)
        a = int(np.argmin(distances)); b = int(nearest[a])
        if distances[a] > 2 * spacing:
            row["reason"] = "native_forward_reverse_sections_do_not_meet"
            continue
        if a < 2 or len(right) - b < 2:
            row["reason"] = "insufficient_join_flanks"
            continue
        first_direction = left[a] - left[a - 2]
        second_direction = right[min(b + 2, len(right) - 1)] - right[b]
        cosine = float(first_direction @ second_direction / max(
            np.linalg.norm(first_direction) * np.linalg.norm(second_direction), 1e-12))
        if cosine < np.cos(np.deg2rad(30)):
            row["reason"] = "native_forward_reverse_direction_disagrees"
            continue
        # The prefix is existing primary attachment topology. Only the exposed
        # body is claimed by the native tube witnesses and final fitting still
        # independently checks this connector.
        owner_line = np.vstack([candidates[owner].points[0], left[:a + 1], right[b + 1:]])
        owner_wall = np.union1d(back_wall, distal[1])
        if np.mean(np.isin(basal[1], owner_wall)) > .35:
            row["reason"] = "basal_tubes_not_independently_supported"
            continue
        basal_line = basal[0].copy()
        basal_line[0] = candidates[mixed].points[0]
        proposals[mixed] = (basal_line, basal[1], basal[2])
        proposals.setdefault(owner, (owner_line, owner_wall, distal[2]))
        row.update(action="reconstruct_two_primary_tubes", mixed_candidate=candidates[mixed].root_id,
                   distal_owner_candidate=candidates[owner].root_id,
                   join_gap_normalized=float(distances[a]), join_direction_cosine=cosine,
                   mixed_forward_status=basal[2]["status"],
                   mixed_partial_sections=basal[2]["partial_sections"],
                   original_mixed_polyline_normalized=candidates[mixed].points.tolist())
    # Applying prepared proposals in root-ID order makes decisions independent
    # of incidental candidate enumeration. No surface labels are modified here.
    prepared = []
    incomplete = set(supported_prefixes) - set(proposals)
    for i in incomplete:
        proposals[i] = supported_prefixes[i]
        for row in report["pairs"]:
            if row.get("mixed_candidate") == candidates[i].root_id:
                row["action"] = "retain_supported_basal_prefix"
    for i in sorted(proposals, key=lambda i: candidates[i].root_id):
        line, wall, details = proposals[i]
        root = candidates[i]
        full_mask = np.zeros(len(context.points), bool)
        full_mask[wall] = True
        analysis_wall = np.flatnonzero(full_mask[mapping] & ~np.asarray(primary_mask, bool)
                                      & ~np.asarray(excluded, bool)[mapping])
        if len(analysis_wall) < 30:
            raise ValueError("A prepared native tube lost its analysis correspondence")
        prepared.append((root, line, analysis_wall, details))
    for root, line, analysis_wall, details in prepared:
        root.points = line
        root.node_indices = None
        root.covered_indices = set(map(int, analysis_wall))
        root.novel_support_indices = set(root.covered_indices)
        root.score_components.update(native_tube_identity_corrected=1.,
                                     native_tube_partial_sections=float(details["partial_sections"]),
                                     trace_growth_arc=path_length(line))
        if "native_tube_identity_corrected" not in root.qc_flags:
            root.qc_flags.append("native_tube_identity_corrected")
        if details["partial_sections"]:
            root.qc_flags.append("native_tube_contact_review")
        if any(root is candidates[i] for i in incomplete):
            root.qc_flags.append("native_tube_terminal_unresolved")
        report["corrected_candidate_ids"].append(root.root_id)
    for i, reserved in conflicts.items():
        if i in proposals:
            continue
        root = candidates[i]
        # Selection quarantine preserves the hypothesis for review, without
        # letting its unsupported tail continue into assignment or topology.
        root.score_components["native_identity_conflict_unresolved"] = 1.
        root.qc_flags.append("native_identity_conflict_unresolved")
        root.novel_support_indices = set(support[i]) - reserved
    report["status"] = ("corrected" if proposals else "unresolved_identity_conflict"
                        if conflicts else "evaluated_no_supported_change")
    return report

