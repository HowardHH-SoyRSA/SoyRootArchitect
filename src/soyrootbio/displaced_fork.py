"""Resolve unsupported neighbour attachments using a supported native junction."""
from collections import Counter
from copy import deepcopy

import numpy as np
from scipy.spatial import cKDTree


def reroute_displaced_forks(paths, *, spacing, support_points, mesh_points,
                            mesh_triangles, excluded, mesh_context, attachment_status):
    from .centerline import _supported_connector
    from .fork_evidence import native_fork_surface_evidence
    from .primary_surface import _unsafe_native_vertices
    from .topology import _path_arc, _terminal_mesh_connection

    if support_points is None or mesh_points is None or mesh_triangles is None or not len(mesh_triangles):
        return []
    cloud = np.asarray(support_points)
    native_tree = mesh_context.point_tree if mesh_context is not None else cKDTree(mesh_points)
    unsafe = _unsafe_native_vertices(mesh_triangles, len(mesh_points))
    if excluded is not None:
        unsafe |= excluded
    by_id = {r.root_id: r for r in paths}
    proposals = []
    for child in sorted(paths, key=lambda r: str(r.root_id)):
        former = by_id.get(child.parent_id)
        if (child.order < 2 or former is None or len(child.points) < 12
                or attachment_status.get(str(child.root_id)) == 'accepted'
                or child.body_start_index > 1):
            continue
        if any(r.parent_id == child.root_id and (r.insertion_index or 0) == 0 for r in paths):
            # A descendant at the old anchor needs its own attachment review;
            # moving that anchor must not silently move the descendant body.
            continue
        cp = np.asarray(child.points)
        child_support = cloud[sorted(child.covered_indices)]
        old_bridge = np.linspace(cp[0], cp[1], max(3, int(np.ceil(np.linalg.norm(cp[1]-cp[0])/spacing))))
        if _supported_connector(old_bridge, cloud[sorted(former.covered_indices)], child_support, spacing):
            continue
        for parent in paths:
            if parent is former or parent.order != child.order - 1 or len(parent.points) < 12:
                continue
            pp = np.asarray(parent.points)
            gap, index = cKDTree(pp).query(cp[1]); index = int(index)
            arc = _path_arc(pp)
            radius = max(float(parent.score_components.get('trace_local_radius', 2*spacing)),
                         float(child.score_components.get('trace_local_radius', 2*spacing)))
            if gap > max(16*spacing, 3*radius) or arc[index] < .5*arc[-1]:
                continue
            bridge = np.linspace(pp[index], cp[1], max(3, int(np.ceil(gap/spacing))))
            if not _supported_connector(bridge, cloud[sorted(parent.covered_indices)], child_support, spacing):
                continue
            trial = deepcopy(child)
            trial.points = np.vstack((pp[index], cp[1:]))
            evidence = native_fork_surface_evidence(parent, trial, index, cloud, spacing, native_tree, unsafe)
            if evidence is None or evidence['surface_evidence_gain'] < .20 or evidence['long_arm_radius_similarity'] < .72:
                continue
            connected, _ = _terminal_mesh_connection(pp[:index+1], trial.points,
                window=32*spacing, radius=radius, spacing=spacing, mesh_points=mesh_points,
                mesh_triangles=mesh_triangles, mesh_excluded_mask=excluded,
                mesh_tree=native_tree, mesh_context=mesh_context)
            if connected:
                proposals.append((parent, child, index, evidence))
    counts = Counter(str(root.root_id) for parent, child, *_ in proposals for root in (parent, child))
    decisions = []
    for parent, child, index, evidence in proposals:
        if counts[str(parent.root_id)] != 1 or counts[str(child.root_id)] != 1:
            continue
        former_id = child.parent_id
        child.parent_id = parent.root_id
        child.insertion_index = index
        child.insertion_point = parent.points[index].copy()
        child.points[0] = child.insertion_point
        child.node_indices = None
        child.qc_flags.append('native_displaced_fork_attachment_corrected')
        decisions.append({'parent_id': str(parent.root_id), 'candidate_arm_id': str(child.root_id),
            'former_parent_id': str(former_id), 'action': 'attachment_corrected',
            'former_connector_supported': False, 'transverse_connector_supported': True,
            'native_mesh_connected': True, 'surface_evidence': evidence, 'parent_ref': parent})
    return decisions
