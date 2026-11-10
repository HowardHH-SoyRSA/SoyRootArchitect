"""Parameterised recovery of explicitly reviewed basal parent ownership.

This is a supervised correction primitive, not an automatic branch detector.
Its caller supplies reviewed parent identities and frozen pre-higher-order
ownership. No root ID, specimen name, or physical calibration is embedded here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .mesh_geometry import MeshGeometryContext
from .surface_patches import _polyline_projection_distance_and_arc as project


@dataclass(frozen=True)
class BasalRecoveryParameters:
    proximal_length: float
    radial_multiplier: float = 1.5
    radial_padding_spacings: float = 1.5
    minimum_native_contact_edges: int = 2


def _components(count, edges):
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(count, count)).tocsr()
    return connected_components(graph, directed=False)[1]


def recover_reviewed_basal_ownership(
    points, labels, triangles, *, reviewed_preorder_labels,
    reviewed_parent_paths, parent_body_masks, parent_radii, label_orders, independent_body_masks,
    excluded_mask, immutable_primary_top, gravity, spacing,
    parameters: BasalRecoveryParameters, mesh_context=None, supersedable_labels=(),
):
    """Restore a reviewed O1 neck on existing native edges only.

    Candidate points must have the same frozen O1 owner, lie inside the bounded
    proximal envelope, and currently be unassigned or higher order. Uncertain
    vertices are deliberately left to a separate uncertainty decision. The raw
    graph retains protected-neighbour evidence; the exclusion mask forbids
    writes. Independently supported donor bodies are never consumed or split.
    Entire labels with no independent body may be superseded by user-reviewed
    parent ownership, but their inventory records must remain explicit QC.
    """
    p = np.asarray(points, float); before = np.asarray(labels, int)
    prior = np.asarray(reviewed_preorder_labels, int)
    excluded = np.array(excluded_mask, bool, copy=True)
    top = np.asarray(immutable_primary_top, float); down = np.asarray(gravity, float)
    if (p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all()
            or before.shape != (len(p),) or prior.shape != before.shape
            or excluded.shape != before.shape):
        raise ValueError('finite XYZ points and matching ownership/mask arrays required')
    if (top.shape != (3,) or down.shape != (3,) or not np.isfinite(top).all()
            or not np.isfinite(down).all() or np.linalg.norm(down) <= 0
            or not np.isfinite(spacing) or spacing <= 0):
        raise ValueError('valid immutable top, gravity and spacing required')
    if (not np.isfinite(parameters.proximal_length) or parameters.proximal_length <= 0
            or not np.isfinite(parameters.radial_multiplier) or parameters.radial_multiplier <= 0
            or not np.isfinite(parameters.radial_padding_spacings) or parameters.radial_padding_spacings < 0
            or parameters.minimum_native_contact_edges < 1):
        raise ValueError('positive finite bounds and a native contact gate required')
    excluded |= (p-top) @ (-down/np.linalg.norm(down)) > 1e-9*spacing
    context = mesh_context or MeshGeometryContext.build(p, triangles)
    context.validate(p, triangles)
    edges = context.edges
    result = before.copy(); decisions = []; proposals = {}
    independent = np.zeros(len(p), bool)
    for label, mask in independent_body_masks.items():
        mask = np.asarray(mask, bool)
        if mask.shape != before.shape or np.any(mask & (before != label)):
            raise ValueError('independent body masks must match their frozen owner')
        independent |= mask
    order = np.array([label_orders.get(int(v), -1) for v in before])
    allowed_source = (before == -1) | (order >= 2)
    for label, path in sorted(reviewed_parent_paths.items()):
        if label_orders.get(label) != 1:
            raise ValueError('reviewed receiving roots must be O1')
        body = np.asarray(parent_body_masks.get(label), bool)
        if body.shape != before.shape or not body.any() or np.any(body & (before != label)):
            raise ValueError('each reviewed parent requires a frozen owned body witness')
        path = np.asarray(path, float); radius = float(parent_radii[label])
        if (path.ndim != 2 or path.shape[1] != 3 or len(path) < 2
                or not np.isfinite(path).all() or not np.isfinite(radius) or radius <= 0):
            raise ValueError('valid reviewed parent path and frozen radius required')
        ids = np.flatnonzero((prior == label) & allowed_source & ~excluded & ~independent)
        distance, arc = project(p[ids], path)
        envelope = parameters.radial_multiplier*radius + parameters.radial_padding_spacings*spacing
        selected = ids[(distance <= envelope) & (arc <= parameters.proximal_length)]
        active = np.zeros(len(p), bool); active[selected] = True
        local_edges = edges[active[edges].all(axis=1)]
        components = _components(len(p), local_edges)
        accepted = []
        for part in np.unique(components[selected]):
            members = selected[components[selected] == part]
            member_mask = np.zeros(len(p), bool); member_mask[members] = True
            boundary = edges[member_mask[edges].sum(axis=1) == 1]
            contacts = (before[boundary] == label).any(axis=1)
            count = int(contacts.sum())
            action = 'recover_reviewed_parent_surface' if count >= parameters.minimum_native_contact_edges else 'retain_no_native_parent_anchor'
            decisions.append({'recipient_label':int(label),'first_vertex':int(members[0]),
                              'vertices':len(members),'native_parent_contact_edges':count,
                              'decision':action,'vertex_indices':members.tolist()})
            if action == 'recover_reviewed_parent_surface':accepted.extend(members.tolist())
        proposals[label] = np.asarray(accepted, int)
        result[proposals[label]] = label
    # Preserve each surviving donor component. A disconnected island may be
    # wholly transferred, but a surviving independent body must not be cut.
    donor_rejections = []
    old_components = _components(len(p), edges[before[edges[:,0]] == before[edges[:,1]]])
    new_components = _components(len(p), edges[result[edges[:,0]] == result[edges[:,1]]])
    for donor in np.unique(before[(before != result) & (before >= 0)]):
        for part in np.unique(old_components[before == donor]):
            members = np.flatnonzero((before == donor) & (old_components == part))
            retained = members[result[members] == donor]
            if len(retained) and len(np.unique(new_components[retained])) > 1:
                moved = members[result[members] != donor]
                result[moved] = donor
                donor_rejections.append({'donor_label':int(donor),'vertices':len(moved),
                                         'reason':'retained_to_avoid_splitting_surviving_donor',
                                         'vertex_indices':moved.tolist()})
    # Rollbacks only restore old labels. Re-evaluate recipient connectivity
    # after every rollback: a rejected donor transfer may have been a bridge.
    recipient_rejections = []
    allowed_empty = set(int(x) for x in supersedable_labels)
    while True:
        old = result.copy()
        current_components = _components(len(p), edges[result[edges[:,0]] == result[edges[:,1]]])
        for donor in np.unique(before[(before != result) & (before >= 0)]):
            for part in np.unique(old_components[before == donor]):
                members = np.flatnonzero((before == donor) & (old_components == part))
                retained = members[result[members] == donor]
                if len(retained) and len(np.unique(current_components[retained])) > 1:
                    moved = members[result[members] != donor]
                    result[moved] = donor
                    donor_rejections.append({'donor_label':int(donor),'vertices':len(moved),
                        'reason':'retained_donor_after_recipient_rollback','vertex_indices':moved.tolist()})
        for donor in np.unique(before[(before != result) & (before >= 0)]):
            if donor not in allowed_empty and not np.any(result == donor):
                moved = np.flatnonzero((before == donor) & (result != donor))
                result[moved] = donor
                donor_rejections.append({'donor_label':int(donor),'vertices':len(moved),
                    'reason':'retain_unreviewed_whole_label','vertex_indices':moved.tolist()})
        comp = _components(len(p), edges[result[edges[:,0]] == result[edges[:,1]]])
        for label in sorted(reviewed_parent_paths):
            added = np.flatnonzero((result == label) & (before != label))
            anchored = np.unique(comp[np.asarray(parent_body_masks[label],bool)])
            rejected = added[~np.isin(comp[added],anchored)]
            if len(rejected):
                result[rejected] = before[rejected]
                recipient_rejections.append({'recipient_label':int(label),'vertices':len(rejected),
                    'reason':'no_final_native_route_to_frozen_parent_body','vertex_indices':rejected.tolist()})
        if np.array_equal(old,result):break
    assert np.array_equal(result[excluded | independent | (before == -2)], before[excluded | independent | (before == -2)])
    changed = np.flatnonzero(result != before)
    for row in decisions:
        members = np.asarray(row['vertex_indices'],int)
        committed = members[result[members] == row['recipient_label']]
        row['proposal_decision'] = row.pop('decision')
        row['committed_vertex_indices'] = committed.tolist()
        row['committed_vertex_count'] = len(committed)
    emptied = [int(l) for l in np.unique(before[changed]) if l >= 0 and not np.any(result == l)]
    report = {'policy':'reviewed-basal-parent-recovery-v1','parameters':asdict(parameters),
              'parameter_length_unit':'same as points','parent_radii':{int(k):float(v) for k,v in parent_radii.items()},
              'changed_vertex_count':len(changed),'changed_vertex_indices':changed.tolist(),
              'decisions':decisions,'donor_rejections':donor_rejections,'recipient_rejections':recipient_rejections,
              'superseded_surface_labels':emptied,
              'superseded_label_policy':'retain inventory ID; invalidate old measurement and request identity review',
              'independent_body_vertices_preserved':int(independent.sum()),
              'protected_vertices_preserved':int(excluded.sum()),
              'geometry_or_hierarchy_changed':False,'uncertain_vertices_changed':0,
              'automatic_global_parameter_validation':False}
    return result, report
