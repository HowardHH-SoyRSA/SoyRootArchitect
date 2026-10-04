"""Re-examine native unassigned arms after a fork identity correction."""
from copy import deepcopy

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .lateral import LateralStart, grow_lateral_candidates, reduce_similar_paths, resume_lateral_tip_in_batches
from .topology import _path_arc, _point_at_arc, _reconcile_overlong_forks, validate_root_tree


def recover_unassigned_fork_arms(points, labels, primary, roots, *, d_bar, mesh_context,
                                 excluded_mask, analysis_points, analysis_to_mesh,
                                 primary_top_reference, gravity):
    """Propose growth only inside an existing unassigned native component.

    Corrected forks and unresolved traced fork alternatives are re-examined. A new
    arm is committed together with the preserved short arm and descendants
    only if the ordinary fork review accepts it. Every proposal uses the same
    label snapshot; contested components are left unassigned.
    """
    report = {"policy": "native-unassigned-arm-resurvey-v1", "decisions": [], "accepted_count": 0}
    edges = mesh_context.bounded_edges(d_bar)
    free = (labels == -1) & ~excluded_mask
    same = edges[free[edges].all(axis=1)]
    graph = coo_matrix((np.ones(2 * len(same)), (np.r_[same[:, 0], same[:, 1]],
                       np.r_[same[:, 1], same[:, 0]])), shape=(len(points), len(points))).tocsr()
    _, component = connected_components(graph, directed=False)
    parts, counts = np.unique(component[free], return_counts=True)
    large = set(parts[counts >= 300].tolist())
    proposals = []
    for label, root in enumerate(roots, 1):
        if (root.score_components.get("fork_long_arm_reconciled", 0) <= 0
                and root.score_components.get("fork_hypothesis_count", 0) < 2):
            continue
        boundary = edges[((labels[edges[:, 0]] == label) & free[edges[:, 1]])
                         | ((labels[edges[:, 1]] == label) & free[edges[:, 0]])]
        contact = np.unique(boundary[free[boundary]])
        root_arc = _path_arc(root.points)
        root_tree = cKDTree(root.points)
        for part in sorted(set(component[contact].tolist()) & large):
            members = np.flatnonzero(free & (component == part))
            dist, node = root_tree.query(points[members])
            seeds = members[dist <= dist.min() + 2 * d_bar]
            point = points[seeds].mean(axis=0)
            insertion = int(root_tree.query(point)[1])
            if root_arc[insertion] < .5 * root_arc[-1]:
                continue
            direction = root.points[insertion] - _point_at_arc(root.points, root_arc, root_arc[insertion] - 24 * d_bar)
            start = LateralStart(0, point, root.points[insertion], insertion, seeds,
                                 direction=direction, surface_contact=True)
            blocked = np.ones(len(points), bool)
            blocked[members] = False
            candidates = reduce_similar_paths(grow_lateral_candidates(
                points, [start], root.points, blocked, d_bar, max_steps=100,
                point_tree=mesh_context.point_tree,
            ))
            for candidate in candidates:
                resume_lateral_tip_in_batches(points, candidate, blocked, d_bar, point_tree=mesh_context.point_tree)
            if not candidates:
                continue
            candidate = max(candidates, key=lambda r: (r.score, len(r.covered_indices)))
            candidate.parent_id = root.root_id
            candidate.order = root.order + 1
            candidate.insertion_index = insertion
            candidate.insertion_point = root.points[insertion].copy()
            candidate.points = np.vstack((candidate.insertion_point, candidate.points))
            candidate.raw_start_point = point.copy()
            candidate.score_components["novel_density_support"] = float(len(candidate.covered_indices))
            candidate.novel_support_indices = set(candidate.covered_indices)
            proposals.append((root.root_id, part, candidate))
    usage = {}
    for _, part, _ in proposals:
        usage[part] = usage.get(part, 0) + 1
    retained = roots
    used_ids = {str(r.root_id) for r in roots}
    for parent_id, part, candidate in proposals:
        if usage[part] != 1:
            continue
        order = candidate.order
        suffix = 1
        while f"root-o{order}-{suffix:03d}" in used_ids:
            suffix += 1
        candidate.root_id = f"root-o{order}-{suffix:03d}"
        used_ids.add(candidate.root_id)
        if analysis_to_mesh is not None:
            full_support = np.zeros(len(points), bool)
            full_support[list(candidate.covered_indices)] = True
            candidate.covered_indices = set(np.flatnonzero(full_support[analysis_to_mesh]).tolist())
            candidate.novel_support_indices = set(candidate.covered_indices)
        trial = deepcopy(retained) + [candidate]
        decisions = []
        reconciled, _ = _reconcile_overlong_forks(
            primary, trial, d_bar=d_bar, support_points=analysis_points,
            mesh_points=points, mesh_triangles=mesh_context.triangles,
            mesh_excluded_mask=excluded_mask, mesh_context=mesh_context,
            decision_log=decisions,
        )
        accepted = any(item.short_child.root_id == candidate.root_id for item in reconciled)
        errors = validate_root_tree(trial, primary_path=primary, primary_top_reference=primary_top_reference,
                                    gravity=gravity)
        row = {"parent_id": parent_id, "candidate_id": candidate.root_id,
               "unassigned_component_first_vertex": int(np.flatnonzero(free & (component == part))[0]),
               "candidate_support_count": len(candidate.covered_indices),
               "action": "accepted" if accepted and not errors else "retained_unassigned",
               "fork_decisions": decisions, "hierarchy_errors": errors}
        report["decisions"].append(row)
        if accepted and not errors:
            retained = trial
            report["accepted_count"] += 1
    return retained, report
