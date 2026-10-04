"""Compare complete ownership runs by native vertex overlap, never edited axes."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from soyrootbio.editor.ply import read_labeled_ply
from soyrootbio.traits import compute_traits
from soyrootbio.types import Normalization, RootPath


def render_manual_junctions(reference, candidate, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    original = read_labeled_ply(reference/'segmented_root_structure.ply')
    edited = read_labeled_ply(reference/'.soyrootbio-editor/materialised/edited_segmented_root_structure.ply')
    current = read_labeled_ply(candidate/'segmented_root_structure.ply')
    labels, _, _ = correspondence(original.root_labels, current.root_labels)
    rows = read(reference/'root_hierarchy.json')['roots']
    erows = read(reference/'.soyrootbio-editor/materialised/edited_root_hierarchy.json')['roots']
    names = {r['root_id']: i for i, r in enumerate(rows)}
    emap = {r['numeric_label']: names.get(r['root_id'], -3) for r in erows}
    manual = np.array([emap.get(int(v), -3) if v >= 0 else v for v in edited.root_labels])
    mask = (original.root_labels == 0) & (manual > 0)
    largest = sorted(Counter(manual[mask]).items(), key=lambda x: (-x[1], x[0]))[:6]
    primary = np.asarray(rows[0]['polyline'])
    primary_tree = cKDTree(primary)
    fig, axes = plt.subplots(len(largest), 3, figsize=(10, 2.9*len(largest)), squeeze=False)
    for row_index, (label, count) in enumerate(largest):
        points = original.positions
        region = points[mask & (manual == label)]
        center = region.mean(axis=0)
        j = primary_tree.query(center)[1]
        z = primary[min(j+4, len(primary)-1)] - primary[max(j-4, 0)]
        z /= max(np.linalg.norm(z), 1e-12)
        x = center - primary[j]
        x -= np.dot(x, z)*z
        x /= max(np.linalg.norm(x), 1e-12)
        radius = max(float(np.linalg.norm(region-center, axis=1).max())*1.8, 2.)
        local = np.flatnonzero(np.linalg.norm(points-center, axis=1) <= radius)
        u, v = (points[local]-center) @ x, (points[local]-center) @ z
        for col, owner in enumerate((original.root_labels, manual, labels)):
            ax = axes[row_index, col]
            other = (owner[local] != 0) & (owner[local] != label)
            ax.scatter(u[other], v[other], s=1, c='#d1d5db', rasterized=True)
            for k, color in [(0, '#dc9c30'), (label, '#148a9c')]:
                chosen = owner[local] == k
                ax.scatter(u[chosen], v[chosen], s=2, c=color, rasterized=True)
            ax.set_aspect('equal'); ax.set_xlim(-radius, radius); ax.set_ylim(-radius, radius)
            ax.set_xticks([]); ax.set_yticks([])
            if row_index == 0:
                ax.set_title(['Original automatic', 'Manual ownership reference', 'Candidate automatic'][col])
            if col == 0:
                ax.set_ylabel(rows[label]['root_id'] + f'\n{count} edited primary vertices')
    fig.suptitle(reference.name + '\nGold: primary   Teal: target child   Gray: other ownership; identical geometry/views', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def render_higher_order_junctions(baseline, candidate, destination):
    """Show the largest changed higher-order interfaces in identical views."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    original = read_labeled_ply(baseline/'segmented_root_structure.ply')
    current = read_labeled_ply(candidate/'segmented_root_structure.ply')
    after, _, _ = correspondence(original.root_labels, current.root_labels)
    rows = read(baseline/'root_hierarchy.json')['roots']
    lookup = {row['root_id']: index for index,row in enumerate(rows)}
    interfaces = []
    for child, row in enumerate(rows):
        if row['root_order'] < 2:
            continue
        parent = lookup[row['parent_id']]
        gained = (original.root_labels == parent) & (after == child)
        if np.any(gained):
            interfaces.append((int(gained.sum()), child, parent, gained))
    interfaces.sort(key=lambda item: (-item[0], item[1]))
    if not interfaces:
        return
    fig, axes = plt.subplots(min(4, len(interfaces)), 2,
        figsize=(9, 3.4*min(4, len(interfaces))), squeeze=False)
    points = original.positions
    for row_index, (count, child, parent, gained) in enumerate(interfaces[:4]):
        region = points[gained]
        center = region.mean(axis=0)
        path = np.asarray(rows[parent]['polyline'])
        index = cKDTree(path).query(center)[1]
        z = path[min(index+4, len(path)-1)] - path[max(index-4, 0)]
        z /= max(np.linalg.norm(z), 1e-12)
        x = center - path[index]
        x -= np.dot(x, z)*z
        if np.linalg.norm(x) < 1e-12:
            x = np.cross(z, [1., 0., 0.])
        x /= max(np.linalg.norm(x), 1e-12)
        radius = max(float(np.linalg.norm(region-center, axis=1).max())*2, .75)
        local = np.flatnonzero(np.linalg.norm(points-center, axis=1) <= radius)
        u, v = (points[local]-center) @ x, (points[local]-center) @ z
        for column, owner in enumerate((original.root_labels, after)):
            ax = axes[row_index, column]
            other = (owner[local] != parent) & (owner[local] != child)
            ax.scatter(u[other], v[other], s=2, c='#d1d5db', rasterized=True)
            for label, color in [(parent, '#dc9c30'), (child, '#148a9c')]:
                selected = owner[local] == label
                ax.scatter(u[selected], v[selected], s=3, c=color, rasterized=True)
            ax.set_aspect('equal'); ax.set_xlim(-radius, radius); ax.set_ylim(-radius, radius)
            ax.set_xticks([]); ax.set_yticks([])
            if row_index == 0:
                ax.set_title(['Baseline automatic', 'Corrected automatic'][column])
            if column == 0:
                ax.set_ylabel(f"{rows[child]['root_id']} (order {rows[child]['root_order']})\n{count} parent-to-child vertices")
    fig.suptitle(baseline.name+'\nHigher-order interfaces — gold: direct parent; teal: child; gray: other', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_full_analysis_mapping(bundle, meta, positions):
    """Verify the identity mapping used by this six-sample, unreduced PLY study."""
    from soyrootbio.io import _analysis_vertex_indices

    geometry = meta['source_geometry']
    assert Path(meta['source']).suffix.lower() == '.ply', 'mapping audit requires a PLY source'
    assert geometry['nonfinite_vertex_count'] == 0, 'mapping audit requires finite source vertices'
    assert not geometry['analysis_reduced'], 'reduced analyses need their own mapping audit'
    assert geometry['original_point_count'] == geometry['full_point_count'] == geometry['analysis_point_count'] == len(positions)
    with np.load(bundle/'original_input_geometry.npz') as original:
        np.testing.assert_array_equal(original['points'], positions)
    indices = _analysis_vertex_indices(positions, target_count=geometry['analysis_point_count'],
                                      random_seed=meta['config']['random_seed'])
    np.testing.assert_array_equal(indices, np.arange(len(positions), dtype=np.int64))
    return dict(all_source_vertices_retained_in_order=True, analysis_to_full_identity=True,
                vertex_count=len(indices), indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                evidence='Exact source/export vertex equality, full analysis count, and replay of the unreduced PLY selection rule')


def correspondence(a, b):
    """Map b numeric labels onto a, maximizing total surface intersection/union."""
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    valid = (a >= 0) & (b >= 0)
    overlap = np.bincount(a[valid] * nb + b[valid], minlength=na*nb).reshape(na, nb)
    ca, cb = np.bincount(a[a >= 0], minlength=na), np.bincount(b[b >= 0], minlength=nb)
    iou = overlap / np.maximum(ca[:, None] + cb[None, :] - overlap, 1)
    # Primary identity is fixed; never match it to a lateral based on size.
    ia, ib = linear_sum_assignment(-iou[1:, 1:])
    matches = [(0, 0, float(iou[0, 0]))] + [
        (int(x+1), int(y+1), float(iou[x+1, y+1])) for x, y in zip(ia, ib)
        if overlap[x+1, y+1] > 0]
    mapping = {-2: -2, -1: -1, **{y: x for x, y, _ in matches}}
    normalized = np.array([mapping.get(int(x), na + int(x)) for x in b], int)
    return normalized, matches, mapping


def components(labels, edges):
    use = edges[(labels[edges[:, 0]] >= 0) & (labels[edges[:, 0]] == labels[edges[:, 1]])]
    graph = coo_matrix((np.ones(len(use)), (use[:, 0], use[:, 1])), shape=(len(labels), len(labels))).tocsr()
    groups = connected_components(graph, directed=False)[1]
    return {int(k): len(np.unique(groups[labels == k])) for k in np.unique(labels[labels >= 0])}


def verify_export(bundle, mesh, meta, hierarchy):
    """Recompute numerical measurements from exported final labels and axes."""
    norm = Normalization(np.asarray(meta['normalization_minimum']), meta['normalization_scale'])
    point = norm.transform_points(mesh.positions)
    paths = {r['root_id']: norm.transform_points(np.asarray(r['polyline'])) for r in hierarchy}
    roots = [RootPath(root_id=r['root_id'], parent_id=r['parent_id'], order=r['root_order'],
        points=paths[r['root_id']], parent_points=paths[r['parent_id']],
        insertion_point=norm.transform_points(np.asarray(r['insertion_point'])[None])[0] if r.get('insertion_point') is not None else None,
        insertion_index=r.get('insertion_index'), body_start_index=r.get('body_start_index', 0),
        centerline_assessment=r.get('centerline_assessment') or {}, confidence=r.get('confidence', 0.),
        qc_flags=r.get('qc_flags', [])) for r in hierarchy[1:]]
    labels = mesh.root_labels
    recalculated = compute_traits(paths['primary'], roots, point, labels == 0, labels, norm,
        lateral_start_count=meta['lateral_start_count'], full_points=mesh.positions,
        triangles=mesh.triangles, full_root_labels=labels, mesh_metadata=meta['source_geometry'],
        primary_confidence=hierarchy[0].get('confidence', 1.),
        primary_qc_flags=hierarchy[0].get('qc_flags', []),
        primary_centerline_assessment=hierarchy[0].get('centerline_assessment') or {},
        tip_vector_window=meta['config']['tip_vector_window_mesh_units'],
        gravity=np.array(meta['gravity_vector'])).set_index('root_id').sort_index()
    exported = pd.read_csv(bundle/'root_traits.csv').set_index('root_id').sort_index()
    numeric = sorted(set(recalculated.select_dtypes(include='number').columns) & set(exported.select_dtypes(include='number').columns))
    mismatches = {}
    for col in numeric:
        okay = np.isclose(recalculated[col], exported[col], rtol=1e-8, atol=1e-8, equal_nan=True)
        if not np.all(okay):
            mismatches[col] = exported.index[~okay].tolist()
    assert not mismatches, f"trait recomputation mismatch: {mismatches}"
    workbook = pd.read_excel(bundle/'traits.xlsx', sheet_name='Root traits').set_index('root_id').sort_index()
    assert workbook.index.equals(exported.index)
    for col in numeric:
        np.testing.assert_allclose(workbook[col], exported[col], rtol=1e-8, atol=1e-8,
                                   equal_nan=True, err_msg='Excel/CSV mismatch: '+col)
    skeleton = pd.concat([pd.read_csv(bundle/'primary_skeleton.csv'),
                          pd.read_csv(bundle/'lateral_skeletons.csv')])
    for row in hierarchy:
        coordinates = skeleton[skeleton.root_id == row['root_id']].sort_values('node_id')[['x','y','z']].to_numpy()
        np.testing.assert_allclose(coordinates, row['polyline'], rtol=1e-12, atol=1e-10)
    for filename, selected in [('primary_points.ply', labels == 0),
                               ('lateral_points.ply', labels > 0),
                               ('unassigned_points.ply', labels == -1),
                               ('uncertain_points.ply', labels == -2)]:
        subset = read_labeled_ply(bundle/filename)
        np.testing.assert_array_equal(subset.positions, mesh.positions[selected])
    # Reload through the standard editor reader in a separate audit directory.
    from soyrootbio.editor.session import EditorSession
    session = EditorSession(bundle, session_dir=bundle.parent/'audit_reload'/bundle.name, load_existing_log=False)
    np.testing.assert_array_equal(session.mesh.root_labels, labels)
    assert set(session.roots) == set(paths)
    import xml.etree.ElementTree as ET
    rsml = ET.parse(bundle/'root_system.rsml')
    rsml_ids = {r.attrib['id'] for r in rsml.findall('.//root')}
    assert rsml_ids == set(paths), "RSML root identities differ from hierarchy"
    by_id = {r['root_id']: r for r in hierarchy}
    for element in rsml.findall('.//root'):
        coords = np.asarray([[float(point.attrib[axis]) for axis in ('x', 'y', 'z')]
                             for point in element.findall('./geometry/polyline/point')])
        np.testing.assert_allclose(coords, by_id[element.attrib['id']]['polyline'],
                                   rtol=1e-12, atol=1e-10)
    return {'numeric_trait_columns_recomputed': len(numeric), 'all_numeric_traits_match': True,
            'standard_bundle_reload_matches': True, 'rsml_root_ids_match': True,
            'rsml_polylines_match_final_hierarchy': True, 'excel_traits_match_csv': True,
            'skeleton_csvs_match_final_hierarchy': True, 'all_point_subsets_match_final_labels': True}


def compare(baseline, candidate, reference=None):
    bm, cm = read(baseline/'metadata.json'), read(candidate/'metadata.json')
    b = read_labeled_ply(baseline/'segmented_root_structure.ply')
    c = read_labeled_ply(candidate/'segmented_root_structure.ply')
    assert np.array_equal(b.positions, c.positions), "native vertices changed"
    assert np.array_equal(b.triangles, c.triangles), "native triangle topology changed"
    with np.load(baseline/'original_input_geometry.npz') as original_b, np.load(candidate/'original_input_geometry.npz') as original_c:
        assert set(original_b.files) == set(original_c.files)
        assert all(np.array_equal(original_b[k], original_c[k], equal_nan=True) for k in original_b.files)
    mappings = {side: verify_full_analysis_mapping(bundle, meta, mesh.positions)
                for side, bundle, meta, mesh in [('baseline', baseline, bm, b), ('candidate', candidate, cm, c)]}
    assert mappings['baseline']['indices_sha256'] == mappings['candidate']['indices_sha256']
    bh, ch = read(baseline/'root_hierarchy.json')['roots'], read(candidate/'root_hierarchy.json')['roots']
    after, matches, mapping = correspondence(b.root_labels, c.root_labels)
    changed = after != b.root_labels
    edges = np.unique(np.sort(np.vstack([b.triangles[:, [0, 1]], b.triangles[:, [1, 2]],
                                       b.triangles[:, [2, 0]]]), axis=1), axis=0)
    bc, cc = components(b.root_labels, edges), components(after, edges)
    broots, croots = {i:r for i,r in enumerate(bh)}, {i:r for i,r in enumerate(ch)}
    bparents = {r['root_id']: i for i,r in broots.items()}
    cparents = {r['root_id']: i for i,r in croots.items()}
    # Empty roots have no surface IoU. Preserve their identity for parent
    # checks only when both the saved ID and automatic polyline agree.
    for candidate_label, row in croots.items():
        old_label = bparents.get(row['root_id'])
        if (old_label is not None and candidate_label not in mapping
                and not np.any(c.root_labels == candidate_label)
                and not np.any(b.root_labels == old_label)
                and np.array_equal(row['polyline'], broots[old_label]['polyline'])):
            mapping[candidate_label] = old_label
    root_rows = []
    for old, new, iou in matches:
        br, cr = broots[old], croots[new]
        unchanged = np.array_equal(b.root_labels == old, c.root_labels == new)
        parent_matches = (bparents.get(br['parent_id'], -1) == mapping.get(cparents.get(cr['parent_id'], -1), -999))
        bp, cp = np.asarray(br['polyline']), np.asarray(cr['polyline'])
        bl, cl = np.linalg.norm(np.diff(bp, axis=0), axis=1).sum(), np.linalg.norm(np.diff(cp, axis=0), axis=1).sum()
        root_rows.append(dict(baseline_id=br['root_id'], candidate_id=cr['root_id'], iou=iou,
            ownership_identical=unchanged, baseline_order=br['root_order'], candidate_order=cr['root_order'],
            parent_matches=parent_matches, baseline_vertices=int(np.sum(b.root_labels == old)),
            candidate_vertices=int(np.sum(c.root_labels == new)),
            baseline_components=bc.get(old, 0), candidate_components=cc.get(old, 0),
            centerline_identical=bp.shape == cp.shape and np.array_equal(bp, cp),
            baseline_length=float(bl), candidate_length=float(cl)))
    # Public exports must faithfully reflect final ownership.
    primary = read_labeled_ply(candidate/'primary_points.ply')
    np.testing.assert_array_equal(primary.positions, c.positions[c.root_labels == 0])
    for side, meta in (("baseline", bm), ("candidate", cm)):
        assert meta['point_assignment']['assigned_vertex_count'] >= 0
    base_point = bm['point_assignment']['base_point_source_coordinates']
    assert base_point == cm['point_assignment']['base_point_source_coordinates'], "immutable primary top changed"
    assignment = cm['point_assignment']
    delta = (c.positions - np.asarray(base_point)) / cm['normalization_scale']
    near = np.linalg.norm(delta, axis=1) <= assignment['base_collar_neighborhood_radius_normalized']
    tolerance = assignment['above_base_tolerance_normalized']
    above = np.where(near, delta @ np.asarray(assignment['base_tipward_direction']) < -tolerance,
                     delta @ np.asarray(assignment['gravity_direction']) < -tolerance)
    assert np.all(c.root_labels[above] == -1), "above-collar ownership was assigned"
    report = dict(sample=baseline.name, baseline=str(baseline), candidate=str(candidate),
        vertex_count=len(after), changed_vertices=int(changed.sum()), changed_fraction=float(changed.mean()),
        baseline_roots=len(bh), candidate_roots=len(ch),
        baseline_roots_with_surface=int(len(np.unique(b.root_labels[b.root_labels >= 0]))),
        candidate_roots_with_surface=int(len(np.unique(c.root_labels[c.root_labels >= 0]))),
        baseline_assignment_counts={k:int(mask.sum()) for k,mask in dict(primary=b.root_labels == 0,
            lateral=b.root_labels > 0, uncertain=b.root_labels == -2, unassigned=b.root_labels == -1).items()},
        candidate_assignment_counts={k:int(mask.sum()) for k,mask in dict(primary=c.root_labels == 0,
            lateral=c.root_labels > 0, uncertain=c.root_labels == -2, unassigned=c.root_labels == -1).items()},
        baseline_order_counts=dict(Counter(r['root_order'] for r in bh)),
        candidate_order_counts=dict(Counter(r['root_order'] for r in ch)),
        unchanged_ownership_roots=sum(r['ownership_identical'] for r in root_rows),
        matched_roots=len(matches), changed_parent_or_order=[r for r in root_rows if not r['parent_matches'] or r['baseline_order'] != r['candidate_order']],
        native_geometry_identical=True, primary_export_matches_final_labels=True, immutable_primary_top_identical=True,
        analysis_mapping_validation=mappings,
        recomputed_above_collar_mask_unassigned=True,
        complete_root_hierarchy_identical=(
            {r['root_id']:(r['parent_id'],r['root_order']) for r in bh} ==
            {r['root_id']:(r['parent_id'],r['root_order']) for r in ch}),
        baseline_compliance=bm.get('final_compliance_audit'), candidate_compliance=cm.get('final_compliance_audit'),
        baseline_unresolved_patches=bm['final_surface_patch_audit']['unresolved_patch_count'],
        candidate_unresolved_patches=cm['final_surface_patch_audit']['unresolved_patch_count'],
        baseline_fitting=bm.get('final_centerline_fitting', {}).get('status'),
        candidate_fitting=cm.get('final_centerline_fitting', {}).get('status'),
        baseline_seconds=bm.get('stage_timings_seconds',{}).get('total'), candidate_seconds=cm.get('stage_timings_seconds',{}).get('total'),
        trimmer_transfers=cm['branch_facing_transection_trimming']['transferred_vertex_count'],
        child_tube_transfers=cm.get('parent_owned_tube_reconciliation', {}).get('transferred_vertex_count', 0),
        child_tube_unresolved=cm.get('parent_owned_tube_reconciliation', {}).get('unresolved_junction_count', 0),
        higher_order_treated_roots=sum(row['root_order'] >= 2 and row['transferred_vertex_count'] > 0
            for row in cm.get('parent_owned_tube_reconciliation', {}).get('junctions', [])),
        surface_iou_at_least_099=sum(row['iou'] >= .99 for row in root_rows),
        component_increases=[r for r in root_rows if r['candidate_components'] > r['baseline_components']],
        roots=root_rows)
    report['export_validation'] = verify_export(candidate, c, cm, ch)
    bt = pd.read_csv(baseline/'root_traits.csv').set_index('root_id')
    ct = pd.read_csv(candidate/'root_traits.csv').set_index('root_id')
    trait_fields = ['length', 'mean_diameter', 'surface_area', 'volume', 'point_count']
    report['trait_changes'] = []
    for row in root_rows:
        old, new = bt.loc[row['baseline_id']], ct.loc[row['candidate_id']]
        delta = {key: {'before': float(old[key]), 'after': float(new[key])}
                 for key in trait_fields if not np.isclose(old[key], new[key], rtol=1e-10, atol=1e-10, equal_nan=True)}
        if delta:
            report['trait_changes'].append(dict(baseline_id=row['baseline_id'], candidate_id=row['candidate_id'],
                                               ownership_identical=row['ownership_identical'], changes=delta))
    if reference and (reference/'.soyrootbio-editor/materialised/edited_segmented_root_structure.ply').exists():
        original = read_labeled_ply(reference/'segmented_root_structure.ply')
        manual_path = reference/'.soyrootbio-editor/materialised/edited_segmented_root_structure.ply'
        manual = read_labeled_ply(manual_path)
        assert np.array_equal(original.positions, c.positions)
        oh = read(reference/'root_hierarchy.json')['roots']
        eh = read(reference/'.soyrootbio-editor/materialised/edited_root_hierarchy.json')['roots']
        names = {r['numeric_label']:r['root_id'] for r in eh}
        original_names = {r['root_id']:i for i,r in enumerate(oh)}
        manual_labels = np.array([original_names.get(names.get(int(v)), len(oh)+int(v)) if v >= 0 else int(v)
                                  for v in manual.root_labels])
        baseline_ref, _, _ = correspondence(original.root_labels, b.root_labels)
        candidate_ref, _, reference_map = correspondence(original.root_labels, c.root_labels)
        baseline_manual, baseline_matches, _ = correspondence(manual.root_labels, b.root_labels)
        candidate_manual, candidate_matches, _ = correspondence(manual.root_labels, c.root_labels)
        manual_changed = original.root_labels != manual.root_labels
        target = (original.root_labels == 0) & (manual_labels > 0) & (manual_labels < len(oh))
        target_rows = []
        tube_rows = {r['root_id']: r for r in cm.get('parent_owned_tube_reconciliation', {}).get('junctions', [])}
        candidate_for_reference = {value:key for key,value in reference_map.items() if key >= 0}
        for label in np.unique(manual_labels[target]):
            mask = target & (manual_labels == label)
            candidate_label = candidate_for_reference.get(int(label))
            candidate_id = ch[candidate_label]['root_id'] if candidate_label is not None else None
            tube = tube_rows.get(candidate_id, {})
            target_rows.append(dict(root_id=oh[label]['root_id'],manual_vertices=int(mask.sum()),
                baseline_agree=int(np.sum(baseline_ref[mask] == label)), candidate_agree=int(np.sum(candidate_ref[mask] == label)),
                candidate_root_id=candidate_id, correction_status=tube.get('status', 'no_matched_tube_record'),
                proposed_vertices=tube.get('proposed_vertex_count', 0),
                transferred_vertices=tube.get('transferred_vertex_count', 0)))
        report['manual_ownership_reference'] = dict(sha256=fingerprint(manual_path),
            edited_centerlines_used=False, total_edited_vertices=int(np.sum(original.root_labels != manual.root_labels)),
            all_edited_vertices_baseline_agree=int(np.sum(manual_changed & (baseline_manual == manual.root_labels))),
            all_edited_vertices_candidate_agree=int(np.sum(manual_changed & (candidate_manual == manual.root_labels))),
            primary_to_existing_child_target_vertices=int(target.sum()),
            baseline_agree=int(np.sum(target & (baseline_ref == manual_labels))),
            candidate_agree=int(np.sum(target & (candidate_ref == manual_labels))), roots=target_rows)
        additions = []
        baseline_by_manual = {x:(y,iou) for x,y,iou in baseline_matches}
        candidate_by_manual = {x:(y,iou) for x,y,iou in candidate_matches}
        for row in eh:
            if row['root_id'] in original_names:
                continue
            label = row['numeric_label']
            bo, biou = baseline_by_manual.get(label, (-1, 0.))
            co, ciou = candidate_by_manual.get(label, (-1, 0.))
            additions.append(dict(manual_id=row['root_id'], order=row['root_order'],
                surface_vertices=int(np.sum(manual.root_labels == label)),
                baseline_match=bh[bo]['root_id'] if bo >= 0 else None, baseline_iou=biou,
                candidate_match=ch[co]['root_id'] if co >= 0 else None, candidate_iou=ciou))
        report['manual_ownership_reference']['manually_added_roots'] = additions
    return report


def write_html(reports, destination):
    esc = html.escape
    table, details = [], []
    for r in reports:
        m = r.get('manual_ownership_reference')
        recovery = (f"{m['candidate_agree']:,} / {m['primary_to_existing_child_target_vertices']:,} "
                    f"({m['candidate_agree']/max(m['primary_to_existing_child_target_vertices'],1):.1%})") if m else 'No manual reference'
        qc = r['candidate_compliance']
        table.append('<tr>' + ''.join(f'<td>{v}</td>' for v in [
            esc(r['sample']), f"{r['baseline_roots']} → {r['candidate_roots']}",
            f"{r['baseline_roots_with_surface']} → {r['candidate_roots_with_surface']}",
            f"{r['changed_vertices']:,} ({r['changed_fraction']:.2%})",
            f"{r['unchanged_ownership_roots']} / {r['matched_roots']}", recovery,
            f"{r['baseline_unresolved_patches']} → {r['candidate_unresolved_patches']}",
            esc(qc['higher_order_primary_contact_status'])]) + '</tr>')
        links = f'<a href="{esc(r["sample"])}.json">Complete audit</a> · <a href="{esc(r["sample"])}_roots.csv">Every matched root</a>'
        detail = f'<h2>{esc(r["sample"])}</h2><p>{links}</p>'
        detail += (f'<p>Export validation: {r["export_validation"]["numeric_trait_columns_recomputed"]} numerical trait columns '
                   'recomputed successfully; standard bundle reload and RSML identities verified. '
                   f'Native component increases in the final bundle: {len(r["component_increases"])} matched roots. '
                   f'Changed matched parent/order records: {len(r["changed_parent_or_order"])}.</p>')
        detail += f'<p>Above-collar assigned vertices: {qc["above_collar_assigned_vertex_count"]}; origin/hierarchy errors: {qc["origin_or_hierarchy_error_count"]}. '
        detail += f'Discrete-child-patch QC: {esc(qc["discrete_child_patch_status"])}. See the JSON for retained unresolved cases and root-level changes.</p>'
        detail += f'<p>Roots treated at order 2 or higher: {r.get("higher_order_treated_roots", 0)}. '
        detail += f'Surface IoU at least 0.99: {r.get("surface_iou_at_least_099", 0)} / {r["matched_roots"]} matched roots. '
        detail += f'Exact complete hierarchy preserved: {r["complete_root_hierarchy_identical"]}.</p>'
        if 'native_primary_contacts' in r:
            contacts = r['native_primary_contacts']
            detail += (f'<p>Native higher-order/primary contact edges: {contacts["before_edges"]} → '
                       f'{contacts["after_edges"]}; newly introduced edges: {contacts["new_edges"]}. '
                       f'Affected roots: {contacts["before_roots"]} → {contacts["after_roots"]}.</p>')
        if 'additional_qc_review' in r:
            detail += '<p class="note">' + esc(r['additional_qc_review']['explanation']) + '</p>'
        if m:
            detail += (f'<p>Agreement on all manually edited vertices (including deletions, additions, unassigned points and other hierarchy edits): '
                       f'{m["all_edited_vertices_baseline_agree"]:,} → {m["all_edited_vertices_candidate_agree"]:,} '
                       f'of {m["total_edited_vertices"]:,}. Edited centerlines were never used for fitting or comparison targets.</p>')
            detail += f'<img src="{esc(r["sample"])}_junctions.png" alt="Original, manual and candidate ownership at the six largest edited junctions">'
            detail += '<h3>Every edited primary-to-child target</h3><table><tr><th>Reference child</th><th>Edited vertices</th><th>Baseline agreement</th><th>Candidate agreement</th><th>Correction decision</th></tr>'
            for row in m['roots']:
                detail += '<tr>' + ''.join(f'<td>{esc(str(value))}</td>' for value in [
                    row['root_id'], row['manual_vertices'], row['baseline_agree'], row['candidate_agree'],
                    row['correction_status']]) + '</tr>'
            detail += '</table><p>Agreement is measured on edited surface vertices, not edited polylines. A retained or unmeasurable junction is not evidence that the original ownership was correct.</p>'
            if m['manually_added_roots']:
                detail += '<h3>Manually added roots: automatic correspondence by surface overlap</h3><table><tr><th>Manual root</th><th>Vertices</th><th>Candidate root</th><th>Surface IoU</th></tr>'
                for row in m['manually_added_roots']:
                    detail += '<tr>' + ''.join(f'<td>{esc(str(v))}</td>' for v in [row['manual_id'], row['surface_vertices'], row['candidate_match'], f"{row['candidate_iou']:.3f}"]) + '</tr>'
                detail += '</table><p>Low overlap remains a missed or differently segmented manual branch; the ownership correction does not invent a new root from a manual centerline.</p>'
        if (destination.parent/(r['sample']+'_higher_order.png')).exists():
            detail += f'<h3>Higher-order ownership changes</h3><img src="{esc(r["sample"])}_higher_order.png" alt="Baseline and corrected ownership at higher-order interfaces">'
        details.append(detail)
    content = '''<!doctype html><meta charset="utf-8"><title>Junction ownership comparison</title>
<style>body{font:15px/1.5 system-ui,sans-serif;max-width:1450px;margin:35px auto;padding:0 28px;color:#17252b}h1,h2{color:#145968}h2{margin-top:44px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d6e2e5;padding:9px;text-align:left}th{background:#e8f1f3}tr:nth-child(even){background:#f7fafb}img{display:block;max-width:1050px;width:100%;margin:24px auto}.note{background:#edf5f4;border-left:4px solid #268b7e;padding:15px}a{color:#0b677c}</style>
<h1>Parent–child ownership: six-sample comparison</h1>
<p>Full raw-PLY pipeline runs with frozen source snapshots, saved primary guidance, fixed analysis counts and seed. The manual segmentation supplies ownership evidence only. Final centerlines and traits come from each automatic run.</p>
<p class="note">The correction measures a connected exposed child tube from native cross-sections and surface normals, then follows its observed surface back toward its direct parent at every root order. All claims use frozen ownership; ambiguous cuts, parent disconnections and higher-order primary contact are rejected. Native geometry and the immutable primary top are checked. Unresolved patches and contacts remain explicit QC; this report does not label them compliant.</p>
<table><tr><th>Sample</th><th>Root records before → after</th><th>Roots with owned surface</th><th>Changed native vertices</th><th>Identical / matched surfaces</th><th>Edited primary → existing child recovered</th><th>Unresolved child patches before → after</th><th>Higher-order primary contact</th></tr>'''
    content += ''.join(table) + '</table>' + ''.join(details)
    references = [r for r in reports if 'manual_ownership_reference' in r]
    if references:
        recovery = '; '.join(
            f"{esc(r['sample'])}: {r['manual_ownership_reference']['candidate_agree']:,} / "
            f"{r['manual_ownership_reference']['primary_to_existing_child_target_vertices']:,} target vertices"
            for r in references)
        notice = ('<p class="note"><strong>Outcome: partial ownership correction.</strong> '
                  + recovery + '. Remaining manually edited regions are not all recovered. '
                  'The method requires an existing connected exposed child body; absent or '
                  'unmeasurable branches remain unresolved. No edited centerlines were used.</p>')
        content = content.replace('<h1>Parent–child ownership: six-sample comparison</h1>',
            '<h1>Parent–child ownership: six-sample comparison</h1>' + notice)
    destination.write_text(content, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    samples = [args.sample] if args.sample else sorted(p.name for p in args.baseline.iterdir() if (p/'metadata.json').exists())
    reports = []
    for sample in samples:
        report = compare(args.baseline/sample, args.candidate/sample, args.reference/sample if args.reference else None)
        (args.output/(sample+'.json')).write_text(json.dumps(report, indent=2))
        if 'manual_ownership_reference' in report:
            render_manual_junctions(args.reference/sample, args.candidate/sample, args.output/(sample+'_junctions.png'))
        render_higher_order_junctions(args.baseline/sample, args.candidate/sample, args.output/(sample+'_higher_order.png'))
        with (args.output/(sample+'_roots.csv')).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=report['roots'][0])
            writer.writeheader(); writer.writerows(report['roots'])
        print(json.dumps({k:v for k,v in report.items() if k not in ('roots','trait_changes','changed_parent_or_order','component_increases')}, indent=2), flush=True)
        reports.append(report)
    (args.output/'comparison.json').write_text(json.dumps(reports, indent=2))
    write_html(reports, args.output/'comparison.html')


if __name__ == '__main__':
    main()
