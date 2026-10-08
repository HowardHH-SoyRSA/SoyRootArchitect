"""Materialise reviewed noise exclusions without retracing root ownership."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .centerline import _support_edges, refit_final_centerlines
from .editor.ply import read_labeled_ply
from .export import export_results
from .presentation import geometry_fingerprint
from .primary_contact import audit_higher_order_primary_contacts
from .surface_patch_audit import audit_discrete_child_patches
from .traits import compute_traits
from .types import Normalization, RootPath


def exclude_hidden_noise(source: str | Path, target: str | Path) -> dict:
    """Rebuild a separate complete bundle with its entire display mask excluded.

    Numeric labels are compacted for export; biological IDs/ownership outside
    the mask are preserved. A noise-only parent with surviving descendants is
    an explicitly unmeasured topology placeholder, never silently reparented.
    """
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists() or source == target or source in target.parents:
        raise ValueError('Use a new output directory outside the source bundle')
    files = [p for p in source.rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(source).parts)]
    hashes = {str(p.relative_to(source)): sha256(p.read_bytes()).hexdigest() for p in files}
    mesh = read_labeled_ply(source / 'segmented_root_structure.ply')
    with np.load(source / 'presentation_noise_masks.npz', allow_pickle=False) as data:
        mask = np.asarray(data['excluded_full_vertices'], bool)
        fingerprint = str(data['source_geometry_sha256'].item())
    if fingerprint != geometry_fingerprint(mesh.positions, mesh.triangles) or mask.shape != (mesh.vertex_count,):
        raise ValueError('Reviewed noise mask does not match source geometry')
    touched = mask[mesh.triangles].sum(axis=1)
    if np.any((touched != 0) & (touched != 3)):
        raise ValueError('Noise exclusions must contain whole native components')
    old_meta = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
    meta = deepcopy(old_meta)
    rows = {r['root_id']: r for r in json.loads((source / 'root_hierarchy.json').read_text())['roots']}
    old_label = {r['root_id']: int(r['numeric_label']) for r in meta['root_label_map'] if int(r['numeric_label']) >= 0}
    if np.any(mesh.root_labels <= -3):
        raise ValueError('Bundles with accepted nodules require nodule-aware reanalysis; this saved-bundle helper must not discard nodule exports')
    norm = Normalization(np.asarray(meta['normalization_minimum']), float(meta['normalization_scale']))
    points = norm.transform_points(mesh.positions)
    spacing = float(meta['d_bar_normalized'])
    gravity = np.asarray(meta.get('gravity_vector', [0, 0, -1.]), float)
    top_source = np.asarray(meta['point_assignment']['base_point_source_coordinates'])
    top = norm.transform_points(top_source[None])[0]
    before = mesh.root_labels.copy()
    labels = before.copy()
    labels[mask] = -1
    affected = {rid for rid, label in old_label.items() if np.any(mask & (before == label))}
    noise_only = {rid for rid in affected if not np.any(labels == old_label[rid])}
    if 'primary' in noise_only:
        raise ValueError('Noise exclusion cannot remove the primary root')
    removed = set(noise_only)
    # Keep any noise-only ancestor needed by surviving topology. Its traits
    # are explicitly excluded, and its surface/line are hidden in the editor.
    for rid, row in rows.items():
        if rid in noise_only:
            continue
        parent = row.get('parent_id')
        while parent in rows:
            removed.discard(parent)
            parent = rows[parent].get('parent_id')
    placeholders = noise_only - removed
    retained = sorted((rid for rid in rows if rid != 'primary' and rid not in removed), key=old_label.get)
    remapped = labels.copy()
    paths = []
    for label, rid in enumerate(retained, 1):
        row = rows[rid]
        remapped[labels == old_label[rid]] = label
        paths.append(RootPath(
            root_id=rid, parent_id=row['parent_id'], order=int(row['root_order']),
            points=norm.transform_points(np.asarray(row['polyline'])),
            parent_points=norm.transform_points(np.asarray(rows[row['parent_id']]['polyline'])),
            insertion_point=None if row.get('insertion_point') is None else norm.transform_points(np.asarray(row['insertion_point'])[None])[0],
            insertion_index=row.get('insertion_index'), body_start_index=int(row.get('body_start_index', 0)),
            confidence=float(row.get('confidence', 0)), qc_flags=list(row.get('qc_flags', [])),
            centerline_assessment=deepcopy(row.get('centerline_assessment', {})),
        ))
    labels = remapped
    primary = norm.transform_points(np.asarray(rows['primary']['polyline']))
    edges = _support_edges(points, mesh.triangles, spacing)
    graph = coo_matrix((np.ones(len(edges), np.uint8), (edges[:, 0], edges[:, 1])), shape=(len(points), len(points))).tocsr()
    preserved, need_refit, details = {}, set(), []
    for rid, path, body_start in [('primary', primary, 0)] + [(r.root_id, r.points, r.body_start_index) for r in paths]:
        old_indices = np.flatnonzero(before == old_label[rid])
        assessment = deepcopy(rows[rid].get('centerline_assessment', {}))
        label = 0 if rid == 'primary' else retained.index(rid) + 1
        indices = np.flatnonzero(labels == label)
        largest_changed = False
        if len(old_indices):
            _, components = connected_components(graph[old_indices][:, old_indices], directed=False)
            largest = components == np.argmax(np.bincount(components))
            largest_changed = bool(np.any(mask[old_indices[largest]]))
        if largest_changed:
            need_refit.add(rid)
        else:
            preserved[rid] = {'vertices': indices, 'points': path.copy(), 'body_start_index': body_start, 'assessment': assessment}
        details.append({'root_id': rid, 'numeric_label': label, 'removed_support_vertices': int(np.sum(mask[old_indices])),
                        'assigned_point_count': len(indices), 'original_fitting_component_changed': largest_changed})
    retained_faces = mesh.triangles[touched == 0]
    if need_refit:
        primary, final_fit = refit_final_centerlines(
            points, labels, primary, paths, d_bar=spacing, triangles=retained_faces,
            gravity=gravity, primary_top_reference=top, preserved_bodies=preserved,
        )
    else:
        final_fit = deepcopy(meta['final_centerline_fitting'])
        final_fit['roots'] = [r for r in final_fit['roots'] if r['root_id'] not in removed]
        updates = {d['root_id']: d for d in details}
        for row in final_fit['roots']:
            row.update(updates[row['root_id']])
            row['noise_exclusion_body_reused'] = True
        for root in paths:
            root.centerline_assessment.update(updates[root.root_id])
    for root in paths:
        if root.root_id in placeholders:
            root.qc_flags = list(dict.fromkeys([*root.qc_flags, 'noise_excluded_root', 'noise_parent_topology_unresolved']))
    primary_flags = list(dict.fromkeys([*rows['primary'].get('qc_flags', []), *final_fit.get('primary_qc_flags', [])]))
    excluded = mask.copy()
    from .pipeline import _selected_base_exclusion_mask
    assignment = meta['point_assignment']
    excluded |= _selected_base_exclusion_mask(points, top, np.asarray(assignment['base_tipward_direction']),
        gravity=np.asarray(assignment['gravity_direction']), collar_neighborhood_radius=assignment['base_collar_neighborhood_radius_normalized'],
        tolerance=assignment['above_base_tolerance_normalized'])
    if np.any(labels[excluded] != -1):
        raise AssertionError('Noise/collar exclusion must remain unassigned')
    contacts = audit_higher_order_primary_contacts(points, labels, paths, triangles=mesh.triangles, d_bar=spacing)
    patches = audit_discrete_child_patches(points, labels, paths, triangles=mesh.triangles, d_bar=spacing, excluded_mask=excluded)
    traits = compute_traits(primary, paths, points, labels == 0, labels, norm,
        full_points=mesh.positions, triangles=mesh.triangles, full_root_labels=labels,
        mesh_metadata=meta.get('source_geometry', {}), primary_confidence=rows['primary'].get('confidence', 1.),
        primary_qc_flags=primary_flags, primary_centerline_assessment=final_fit['roots'][0],
        gravity=gravity, tip_vector_window=meta['config'].get('tip_vector_window_mesh_units', 2.), noise_mask=mask)
    report = {'policy': 'all-hidden-noise-excluded-v1', 'source_bundle': str(source),
        'excluded_vertex_count': int(mask.sum()), 'hidden_assigned_vertex_count': int(np.sum(labels[mask] != -1)),
        'removed_noise_only_root_ids': sorted(removed), 'unmeasured_topology_placeholder_ids': sorted(placeholders),
        'fitting_component_changed_root_ids': sorted(need_refit), 'per_root_support_changes': details,
        'selected_primary_top_source_coordinates': top_source.tolist(), 'source_geometry_preserved': True,
        'outside_mask_biological_ownership_preserved': True, 'system_summary': traits.attrs['system_summary'],
        'contact_audit': contacts, 'child_patch_audit': patches}
    # Recompute current audits; earlier stage evidence is saved separately.
    for key in ('final_compliance_audit', 'attachment_constraint', 'parent_contact_reconciliation', 'ownership_evidence_ledger', 'topology_report'):
        meta.pop(key, None)
    meta.update(noise_exclusion=report, final_centerline_fitting=final_fit,
        higher_order_primary_contact={'status': contacts['status'], 'requires_correction': contacts['requires_correction'], 'final_native_audit': contacts},
        final_surface_patch_audit=patches, selected_lateral_count=len(paths)-len(placeholders),
        selected_order_counts=dict(Counter(r.order for r in paths if r.root_id not in placeholders)),
        inherited_stage_evidence='source_metadata_before_noise_exclusion.json; historical stages are not a new tracing run')
    assignment.update(primary_assigned_vertex_count=int(np.sum(labels==0)), lateral_assigned_vertex_count=int(np.sum(labels>0)),
        assigned_vertex_count=int(np.sum(labels>=0)), uncertain_vertex_count=int(np.sum(labels==-2)), unassigned_vertex_count=int(np.sum(labels==-1)))
    target.mkdir(parents=True)
    shutil.copy2(source/'metadata.json', target/'source_metadata_before_noise_exclusion.json')
    for name in ('original_input_geometry.npz',):
        if (source/name).exists(): shutil.copy2(source/name, target/name)
    analysis_indices = np.flatnonzero(~mask)
    if (source/'noise_reduction_masks.npz').exists():
        with np.load(source/'noise_reduction_masks.npz') as data:
            old_map=data['analysis_to_full']
            if np.all(old_map>=0): analysis_indices=old_map[~mask[old_map]]
    np.savez_compressed(target/'noise_reduction_masks.npz', excluded_full_vertices=mask, analysis_to_full=analysis_indices)
    mapping = {}
    if (source/'input_geometry_mapping.npz').exists():
        with np.load(source/'input_geometry_mapping.npz') as data:
            mapping = {key: data[key] for key in data.files}
    mapping['analysis_to_full'] = analysis_indices
    np.savez_compressed(target/'input_geometry_mapping.npz', **mapping)
    meta['point_count'] = len(analysis_indices)
    native_edges = np.vstack((mesh.triangles[:,[0,1]],mesh.triangles[:,[1,2]],mesh.triangles[:,[2,0]]))
    native_graph = coo_matrix((np.ones(len(native_edges),np.uint8),(native_edges[:,0],native_edges[:,1])),shape=(len(points),len(points))).tocsr()
    component_count, native_components = connected_components(native_graph,directed=False)
    excluded_components = np.unique(native_components[mask])
    meta['noise_reduction'] = {'enabled': True, 'status': 'applied', 'policy': report['policy'],
        'excluded_vertex_count': int(mask.sum()), 'excluded_face_count': int(np.sum(touched==3)),
        'excluded_component_count': len(excluded_components), 'component_count': component_count,
        'source_geometry_preserved': True, 'reviewed_mask_geometry_sha256': fingerprint,
        'excluded_native_components': excluded_components.tolist(),
        'report_file': 'noise_exclusion.json', 'masks_file': 'noise_reduction_masks.npz'}
    (target/'noise_reduction.json').write_text(json.dumps(meta['noise_reduction'], indent=2), encoding='utf-8')
    (target/'noise_exclusion.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    export_results(target, mesh.positions[analysis_indices], primary, paths, labels[analysis_indices]==0,
        np.where(labels[analysis_indices]>0,labels[analysis_indices],0), traits, norm, meta,
        full_points=mesh.positions, triangles=mesh.triangles, full_root_labels=labels, noise_mask=mask)
    assert hashes == {name: sha256((source/name).read_bytes()).hexdigest() for name in hashes}
    return report
