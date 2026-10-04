"""Independent transverse surface evidence for competing fork continuations."""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree


def _at(points, arc, station):
    return np.array([np.interp(station, arc, points[:, axis]) for axis in range(3)])


def _sections(support, path, stations, spacing, radius, native_tree, unsafe, *, allow_offcenter=False):
    from .centerline import _plane_basis

    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    tree = cKDTree(support)
    measured = []
    for station in stations:
        center = _at(path, arc, station)
        tangent = _at(path, arc, station + 6 * spacing) - _at(path, arc, station - 6 * spacing)
        norm = np.linalg.norm(tangent)
        if norm <= spacing:
            continue
        tangent /= norm
        indices = np.asarray(tree.query_ball_point(center, max(3 * radius, 8 * spacing)), int)
        delta = support[indices] - center
        slab = np.abs(delta @ tangent) <= 2 * spacing
        indices, delta = indices[slab], delta[slab]
        if len(delta) < 8 or np.any(unsafe[native_tree.query(support[indices])[1]]):
            continue
        uv = delta @ _plane_basis(tangent).T
        angles = np.sort(np.arctan2(uv[:, 1], uv[:, 0]))
        coverage = 1 - np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max() / (2 * np.pi)
        if coverage < .70 and not allow_offcenter:
            continue
        try:
            hull = ConvexHull(uv)
        except QhullError:
            continue
        centered = coverage >= .70 and np.max(hull.equations[:, -1]) <= .25 * spacing
        if not centered and not allow_offcenter:
            continue
        if not centered:
            # Missing ownership around a valid native tube is not evidence of
            # a bad continuation. Penalize a surface-following trace only when
            # the native section also fails to enclose the candidate center.
            native = native_tree.data[native_tree.query_ball_point(center, max(3 * radius, 8 * spacing))] - center
            native = native[np.abs(native @ tangent) <= 2 * spacing]
            native_uv = native @ _plane_basis(tangent).T
            try:
                native_hull = ConvexHull(native_uv)
            except QhullError:
                continue
            native_angles = np.sort(np.arctan2(native_uv[:, 1], native_uv[:, 0]))
            native_coverage = 1 - np.diff(np.r_[native_angles, native_angles[0] + 2*np.pi]).max() / (2*np.pi)
            if native_coverage >= .70 and np.max(native_hull.equations[:, -1]) <= .25 * spacing:
                continue
        radial = np.linalg.norm(uv, axis=1)
        median = float(np.median(radial))
        spread = float((np.quantile(radial, .9) - np.quantile(radial, .1)) / max(median, spacing))
        measured.append((median, float(np.clip(1 - spread / 1.5, 0, 1)), float(centered)))
    if len(measured) < 3:
        return None
    values = np.asarray(measured)
    taper = float(np.clip(1 - values[-1, 0] / max(values[0, 0], spacing), 0, 1))
    return (float(np.median(values[:, 0])), float(np.median(values[:, 1])),
            len(measured), taper, float(values[-1, 0]), float(values[:, 2].mean()))


def native_fork_surface_evidence(parent, child, insertion_index, points, spacing, native_tree, unsafe):
    """Compare caliber and transverse regularity without rewarding length.

    The same frozen support is used for the incoming tube and both arms.
    Incomplete/open sections cannot authorize a swap. A thin terminal arm is
    preserved by the caller even when the other tube continues the caliber.
    """
    p, c = np.asarray(parent.points), np.asarray(child.points)
    pa = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    ca = np.r_[0., np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    insertion = pa[insertion_index]
    suffix = pa[-1] - insertion
    if insertion < 24 * spacing or suffix < 2 * spacing or ca[-1] < 32 * spacing:
        return None
    pi = sorted(i for i in parent.covered_indices if 0 <= i < len(points))
    ci = sorted(i for i in child.covered_indices if 0 <= i < len(points))
    if min(len(pi), len(ci)) < 30:
        return None
    radius = max(float(parent.score_components.get('trace_local_radius', 3 * spacing)),
                 float(child.score_components.get('trace_local_radius', 3 * spacing)), 2 * spacing)
    before = _sections(points[pi], p,
        np.linspace(max(0, insertion - 48 * spacing), insertion - 12 * spacing, 5),
        spacing, radius, native_tree, unsafe)
    short = _sections(points[pi], p,
        insertion + np.linspace(min(4 * spacing, .25 * suffix), suffix - min(2 * spacing, .15 * suffix), 5),
        spacing, radius, native_tree, unsafe, allow_offcenter=True)
    long = _sections(points[ci], c,
        np.linspace(12 * spacing, min(ca[-1] - 4 * spacing, 64 * spacing), 7),
        spacing, radius, native_tree, unsafe)
    if before is None or short is None or long is None or long[2] < 4:
        return None
    rp, rs, rl = before[0], short[0], long[0]
    short_similarity = min(rp, rs) / max(rp, rs, spacing)
    long_similarity = min(rp, rl) / max(rp, rl, spacing)
    current = .85 * short_similarity + .15 * short[1]
    alternate = .85 * long_similarity + .15 * long[1]
    # A closed shrinking cap can inherit the junction's width in its median.
    # Compare the measured contraction with the other arm; extent contributes
    # nothing to this evidence and the cap remains a preserved child surface.
    cap_evidence = (short[3] >= .40 and short[4] <= .85 * rp
                    and long[4] >= .80 * rp and long[3] <= .20)
    if cap_evidence:
        current = .40 * short_similarity + .10 * short[1] + .50 * (1 - short[3])
        alternate = .40 * long_similarity + .10 * long[1] + .50 * (1 - long[3])
    current *= short[5]
    return {
        'incoming_surface_radius': rp, 'short_arm_surface_radius': rs, 'long_arm_surface_radius': rl,
        'short_arm_radius_similarity': short_similarity, 'long_arm_radius_similarity': long_similarity,
        'short_arm_section_quality': short[1], 'long_arm_section_quality': long[1],
        'short_arm_surface_score': current, 'long_arm_surface_score': alternate,
        'surface_evidence_gain': alternate - current,
        'short_arm_surface_contraction': short[3], 'long_arm_surface_contraction': long[3],
        'closed_cap_evidence': float(cap_evidence),
        'short_arm_centered_section_fraction': short[5],
        'incoming_section_count': before[2], 'short_arm_section_count': short[2], 'long_arm_section_count': long[2],
    }
