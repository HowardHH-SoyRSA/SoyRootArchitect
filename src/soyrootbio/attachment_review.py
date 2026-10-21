"""Reconsider provisional parents using connected bodies and native tube evidence.

Detached labels cannot certify a biological parent. A bounded uncertain basal
region can be assigned only after measured transverse rings connect it to an
existing child body. Every proposal uses the same ownership snapshot; no mesh
edges, root identities, or exposed bodies are invented or removed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .centerline import _axis, _fit_body, _legal_attachment_indices
from .attachment_constraint import _manifold_vertex_links
from .junction_tubes import child_tube_witnesses, vertex_normals
from .mesh_geometry import MeshGeometryContext
from .parent_contact import _distal_body_support


def connected_body_evidence(points, labels, primary, roots, *, d_bar, mesh_context,
                            excluded_mask=None):
    """Return body-supported components, without treating size as proof."""
    labels = np.asarray(labels, int)
    excluded = (np.zeros(len(points), bool) if excluded_mask is None
                else np.asarray(excluded_mask, bool))
    if excluded.shape != labels.shape:
        raise ValueError("exclusions must match ownership")
    edges = mesh_context.bounded_edges(d_bar)
    edges = edges[~excluded[edges].any(axis=1)]
    ownership = mesh_context.ownership(labels)
    component = ownership.components(edges, active=~excluded & (labels >= 0))
    body_owner = np.full(len(points), -1, int)
    rows = {}
    paths = [np.asarray(primary)] + [np.asarray(r.points)[r.body_start_index:] for r in roots]
    for label, path in enumerate(paths):
        owned = ownership.vertices(label)
        owned = owned[~excluded[owned]]
        parts, sizes = np.unique(component[owned], return_counts=True)
        supported = []
        for part, size in zip(parts, sizes):
            members = owned[component[owned] == part]
            if label == 0:
                okay = part == parts[np.argmax(sizes)]
            else:
                root = roots[label - 1]
                radius = max(float(root.mean_radius or 0.), 2 * d_bar)
                okay = size >= 8 and _distal_body_support(
                    points[members], path, radius, d_bar)["supported"]
            if okay:
                supported.append(int(part))
                body_owner[members] = label
        rows[label] = {"component_count": len(parts),
                       "body_component_count": len(supported),
                       "body_vertex_count": int(np.count_nonzero(body_owner[owned] == label)),
                       "detached_vertex_count": int(np.count_nonzero(body_owner[owned] != label))}
    return body_owner, component, rows


def review_basal_attachments(points, labels, primary, roots, *, d_bar, triangles,
                             primary_top_reference, gravity=(0., 0., -1.),
                             excluded_mask=None, mesh_context=None):
    """Jointly review ownership/parentage before final attachment preservation.

    A new parent requires a unique body-supported native basal contact, no
    competing contact to the old parent's body, a sustained measured tube, and
    manifold observed surface. Ambiguity leaves both labels and hierarchy as-is.
    Only uncertain vertices and detached, unsupported parent fragments may move.
    Ordinary parent-owned tube reconciliation handles any measured sleeve later.
    """
    p, before = np.asarray(points, float), np.asarray(labels, int)
    result, updated = before.copy(), deepcopy(roots)
    report = {"policy": "connected-body-basal-parent-review-v1", "status": "clear",
              "evidence_labels_sha256": hashlib.sha256(before.astype('<i8').tobytes()).hexdigest(),
              "changed_vertex_count": 0, "changed_parent_count": 0, "decisions": [],
              "changed_vertex_indices": [], "refreshed_origin_root_ids": []}
    if triangles is None or not len(triangles):
        report["status"] = "unresolved_no_native_mesh"
        return result, updated, report
    context = mesh_context or MeshGeometryContext.build(p, triangles)
    context.validate(p, triangles)
    if before.shape != (len(p),) or np.any(before >= len(roots) + 1):
        raise ValueError("ownership must reference roots")
    excluded = (np.zeros(len(p), bool) if excluded_mask is None
                else np.array(excluded_mask, bool, copy=True))
    if excluded.shape != before.shape or not np.isfinite(d_bar) or d_bar <= 0:
        raise ValueError("invalid exclusions or spacing")
    gravity = np.asarray(gravity, float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all() or np.linalg.norm(gravity) == 0:
        raise ValueError("gravity must be a finite nonzero vector")
    gravity = gravity / np.linalg.norm(gravity)
    top = np.asarray(primary_top_reference, float)
    if top.shape != (3,) or not np.isfinite(top).all():
        raise ValueError("primary top must be a finite XYZ reference")
    excluded |= (p - top) @ -gravity > 1e-12
    unsafe = np.zeros(len(p), bool)
    unsafe[np.unique(context.edges[context.edge_incidence != 2])] = True
    edges = context.bounded_edges(d_bar)
    edges = edges[~excluded[edges].any(axis=1)]
    body_owner, component, body_rows = connected_body_evidence(
        p, before, primary, roots, d_bar=d_bar, mesh_context=context, excluded_mask=excluded)
    ownership = context.ownership(before)
    all_components = ownership.components(edges, active=~excluded)
    lookup = {"primary": 0, **{r.root_id: i for i, r in enumerate(roots, 1)}}
    reverse = {v:k for k,v in lookup.items()}
    paths = [np.asarray(primary)] + [np.asarray(r.points) for r in roots]
    bodies = {label:np.flatnonzero(body_owner == label) for label in range(len(roots)+1)}
    axes = {0: np.asarray(primary)}
    def axis(label):
        if label not in axes:
            ids = bodies[label]
            all_ids = ownership.vertices(label)
            local = np.searchsorted(all_ids, ids)
            graph = ownership.local_graph(label, edges)[local][:, local]
            initial = _axis(p[ids], graph, paths[label][0], d_bar)
            axes[label], _ = _fit_body(p[ids], initial, d_bar)
        return axes[label]
    normals = None
    proposals = []
    for label in sorted(range(1, len(roots)+1), key=lambda k: roots[k-1].root_id):
        root = roots[label-1]
        parent_label = lookup.get(root.parent_id)
        if root.order < 2 or parent_label in (None, 0) or body_rows[label]['body_component_count'] != 1:
            continue
        child_ids = bodies[label]
        if len(child_ids) < 12:
            continue
        child = axis(label)
        if len(child) < 2:
            continue
        radius = max(d_bar, float(np.median(cKDTree(p[child_ids]).query(child)[0])))
        limit = max(8*d_bar, 5*radius)
        near = np.linalg.norm(p-child[0], axis=1) <= limit
        # Only whole bounded components may be reconsidered, never an arbitrary
        # slice through an uncertain band or a supported exposed root.
        eligible = ~excluded & ((before == -2) |
            ((before == parent_label) & (body_owner != parent_label)))
        domain = np.zeros(len(p), bool)
        for part in np.unique(all_components[eligible & near]):
            members = np.flatnonzero(all_components == part)
            if np.all(eligible[members]) and np.all(near[members]):
                domain[members] = True
        child_mask = np.zeros(len(p), bool); child_mask[child_ids] = True
        local_child = child_mask & near
        # Keep only domain components actually connected to the basal child.
        active = domain | local_child
        ee = edges[active[edges].all(axis=1)]
        graph = coo_matrix((np.ones(len(ee)), (ee[:,0],ee[:,1])),shape=(len(p),len(p))).tocsr()
        groups = connected_components(graph,directed=False)[1]
        domain &= np.isin(groups,np.unique(groups[local_child]))
        active = domain | local_child
        boundary = edges[active[edges[:,0]] ^ active[edges[:,1]]]
        outside = np.where(active[boundary[:,0]],boundary[:,1],boundary[:,0])
        native_candidates = {}
        for candidate in np.unique(body_owner[outside]):
            candidate = int(candidate)
            if candidate < 0 or candidate == label or body_rows[candidate]['body_component_count'] != 1:
                continue
            contact = boundary[body_owner[outside] == candidate]
            if len(contact) >= 3:
                native_candidates[candidate] = contact
        if parent_label in native_candidates:
            # A genuine parent-body attachment is not disproved by an incidental
            # primary contact. Keep that case for ordinary contact correction.
            continue
        if not native_candidates:
            continue
        row = {"root_id": root.root_id, "previous_parent": root.parent_id,
               "candidate_parents": {reverse[k]:len(v) for k,v in native_candidates.items()},
               "domain_vertex_count": int(domain.sum()),
               "detached_parent_vertex_count": int(np.count_nonzero(domain & (before==parent_label))),
               "status": "unresolved_competing_body_contacts"}
        report['decisions'].append(row)
        if len(native_candidates) != 1:
            continue
        candidate, contact = next(iter(native_candidates.items()))
        # A same/later-order neighbour cannot become a parent in this frozen
        # review. It also cannot be ignored as a competing contact above.
        if candidate and roots[candidate-1].order >= root.order:
            row['status'] = 'unresolved_parent_order_or_cycle'
            continue
        junction_vertices=np.unique(np.r_[np.flatnonzero(domain),contact.ravel()])
        if (np.any(unsafe[junction_vertices]) or not _manifold_vertex_links(
                context.triangles, context.vertex_face_incidence, junction_vertices)):
            row['status'] = 'unresolved_open_or_nonmanifold_junction'
            continue
        hypothetical = before.copy()
        hypothetical[domain] = candidate
        # Uncertain surface is only a candidate donor for the measurement. It
        # is committed below solely where closed-ring witnesses certify it.
        if normals is None:
            normals = vertex_normals(p, np.asarray(triangles,int))
        witness, _, evidence = child_tube_witnesses(
            p, hypothetical, candidate, label, axis(candidate), child,
            spacing=d_bar, context=context, normals=normals,
            excluded=excluded | unsafe | (hypothetical < 0))
        row['tube_evidence'] = evidence
        moved = np.intersect1d(witness,np.flatnonzero(domain))
        # Require the entire reconsidered patch. Partial evidence must not
        # silently make a hole or leave half a detached parent fragment.
        if len(moved) != np.count_nonzero(domain):
            row['status'] = 'unresolved_incomplete_measured_basal_surface'
            row['measured_domain_vertex_count'] = len(moved)
            continue
        if evidence['status'] != 'supported':
            row['status'] = 'unresolved_no_measured_basal_tube'
            continue
        proposed = before.copy(); proposed[moved] = label
        junction = edges[((proposed[edges[:,0]] == label) & (proposed[edges[:,1]] == candidate)) |
                         ((proposed[edges[:,1]] == label) & (proposed[edges[:,0]] == candidate))]
        junction = junction[near[junction].any(axis=1)]
        if len(junction) < 3:
            row['status'] = 'unresolved_no_measured_native_parent_contact'
            continue
        parent = axis(candidate)
        legal = _legal_attachment_indices(parent, top[None], gravity)
        footprint = p[np.unique(junction)].mean(axis=0)
        if not len(legal) or (footprint-top) @ -gravity > 1e-12:
            row['status'] = 'unresolved_above_primary_top'
            continue
        insertion = int(legal[np.argmin(np.linalg.norm(parent[legal]-footprint,axis=1))])
        row.update(status='proposed', proposed_parent=reverse[candidate],
                   changed_vertex_indices=moved.tolist(), native_contact_edge_count=len(junction),
                   body_start=child[0].tolist(), insertion_hint=parent[insertion].tolist())
        proposals.append((label,candidate,moved,parent[insertion].copy(),child[0].copy(),row))
    # No last-writer wins. Any shared domain makes all its claimants unresolved.
    claims = np.zeros(len(p),int)
    for _,_,moved,_,_,_ in proposals:claims[moved] += 1
    accepted=[]
    for proposal in proposals:
        label,candidate,moved,hint,body_start,row=proposal
        if np.any(claims[moved]>1):
            row['status']='unresolved_competing_surface_claims'
            continue
        result[moved]=label
        root=updated[label-1]
        root.parent_id=reverse[candidate]
        root.insertion_point=hint
        root.insertion_index=None
        root.raw_start_point=body_start
        root.parent_points=paths[candidate]
        root.qc_flags=[f for f in root.qc_flags if f not in
                       {'attachment_parent_unresolved','attachment_parent_component_unresolved'}]
        row['status']='reparented_measured_native_body'
        accepted.append(proposal)
    # Lower-order candidates make simultaneous cycles impossible. Recompute
    # descendant orders without renaming, deleting, or reshaping any root.
    orders={'primary':0}
    pending=list(updated)
    while pending:
        ready=[r for r in pending if r.parent_id in orders]
        if not ready:raise ValueError('basal review requires an acyclic input tree')
        for root in ready:
            root.order=orders[root.parent_id]+1;orders[root.root_id]=root.order
        pending=[r for r in pending if r.root_id not in orders]
    # With the false child link removed, refresh the old parent's stale origin
    # from its *connected body* contact. This never edits the primary reference.
    affected={lookup[row['previous_parent']] for *_,row in accepted}
    for label in sorted(affected):
        root=updated[label-1]; parent_label=lookup.get(root.parent_id)
        if parent_label is None or body_rows[label]['body_component_count'] != 1:
            continue
        body=axis(label)
        crossing=ownership.boundary_edges(label,parent_label,edges)
        crossing=crossing[np.isin(crossing,bodies[label]).any(axis=1)]
        if len(crossing)<3:continue
        radius=max(d_bar,float(np.median(cKDTree(p[bodies[label]]).query(body)[0])))
        crossing=crossing[np.any(np.linalg.norm(p[crossing]-body[0],axis=2)<=max(8*d_bar,5*radius),axis=1)]
        if len(crossing)<3 or np.any(unsafe[crossing]):continue
        parent=paths[parent_label]
        legal=_legal_attachment_indices(parent,top[None],gravity)
        if not len(legal):continue
        footprint=p[np.unique(crossing)].mean(axis=0)
        index=int(legal[np.argmin(np.linalg.norm(parent[legal]-footprint,axis=1))])
        root.insertion_index=index;root.insertion_point=parent[index].copy()
        root.raw_start_point=body[0].copy()
        root.qc_flags=[f for f in root.qc_flags if f != 'attachment_parent_unresolved']
        report['refreshed_origin_root_ids'].append(root.root_id)
    report['changed_parent_count']=len(accepted)
    for row in report['decisions']:
        if row['status'].startswith('unresolved'):
            root=updated[lookup[row['root_id']]-1]
            root.qc_flags=list(dict.fromkeys([*root.qc_flags,'basal_parent_review_unresolved']))
    report['changed_vertex_indices']=np.flatnonzero(result!=before).tolist()
    report['changed_vertex_count']=len(report['changed_vertex_indices'])
    report['status']='corrected' if accepted else 'unresolved' if report['decisions'] else 'clear'
    return result,updated,report
