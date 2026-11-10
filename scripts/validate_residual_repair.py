"""Compare a residual-repair bundle with its immutable pre-repair baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from soyrootbio.bundle_layout import read_bundle_mesh
from soyrootbio.measurement_evidence import make_evidence_context, validate_measurement_evidence


def _finite_value(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def compare(before: Path, after: Path) -> dict:
    old_mesh, new_mesh = read_bundle_mesh(before), read_bundle_mesh(after)
    np.testing.assert_array_equal(old_mesh.positions, new_mesh.positions)
    np.testing.assert_array_equal(old_mesh.triangles, new_mesh.triangles)
    read = lambda directory, name: json.loads((directory/'support'/name).read_text(encoding='utf-8'))
    old_meta, new_meta = read(before, 'metadata.json'), read(after, 'metadata.json')
    assert old_meta['coordinate_unit'] == new_meta['coordinate_unit'], 'Coordinate units changed'
    old_roots = {r['root_id']: r for r in read(before, 'root_hierarchy.json')['roots']}
    new_roots = {r['root_id']: r for r in read(after, 'root_hierarchy.json')['roots']}
    assert old_roots.keys() == new_roots.keys(), 'Root identities changed'
    np.testing.assert_array_equal(old_meta['primary_top_reference_mm'], new_meta['primary_top_reference_mm'])
    old_map = {r['root_id']: int(r['numeric_label']) for r in old_meta['root_label_map'] if int(r['numeric_label']) >= 0}
    new_map = {r['root_id']: int(r['numeric_label']) for r in new_meta['root_label_map'] if int(r['numeric_label']) >= 0}
    with np.load(before/'support/ownership_evidence_masks.npz') as archive:
        above = archive['above_base_excluded']
    assert np.all(new_mesh.root_labels[above] == -1), 'Above-collar ownership changed'
    with np.load(before/'support/presentation_noise_masks.npz') as archive:
        noise = archive['excluded_full_vertices']
    assert np.all(new_mesh.root_labels[noise] == -1), 'Excluded noise acquired ownership'
    context = make_evidence_context(old_mesh.positions, old_mesh.triangles)
    rows = []
    inventory = pd.read_csv(before/'tables/root_inventory.csv').set_index('root_id')
    new_inventory = pd.read_csv(after/'tables/root_inventory.csv').set_index('root_id')
    assert inventory.index.is_unique and new_inventory.index.is_unique, 'Duplicate inventory identities'
    assert set(inventory.index) == set(old_roots) and set(new_inventory.index) == set(new_roots)
    for rid, root in old_roots.items():
        current = new_roots[rid]
        assert (root['parent_id'], root['root_order']) == (current['parent_id'], current['root_order']), rid
        owned = np.flatnonzero(old_mesh.root_labels == old_map[rid])
        retained = int(np.count_nonzero(new_mesh.root_labels[owned] == new_map[rid]))
        assert not len(owned) or retained > 0, (rid, 'Root surface consumed')
        evidence = root.get('centerline_assessment', {}).get('measurement_evidence')
        valid, reason = validate_measurement_evidence(evidence, old_mesh.positions, old_mesh.root_labels,
            old_map[rid], np.asarray(root['polyline']), triangles=old_mesh.triangles,
            body_start_index=root.get('body_start_index', 0), context=context)
        assert valid, (rid, reason)
        supported = np.asarray(evidence.get('native_vertex_indices', []), dtype=int)
        lost = supported[new_mesh.root_labels[supported] != new_map[rid]]
        # These targeted repairs promise to preserve exposed lateral bodies.
        # Primary-side uncertainty seams are reported separately and must pass
        # the pipeline's exact connectivity and first-order attachment guards.
        if rid != 'primary' and inventory.loc[rid, 'length_measurement_eligible'] == True:
            assert not len(lost), (rid, 'Previously measured lateral support changed ownership', len(lost))
        current_evidence = current.get('centerline_assessment', {}).get('measurement_evidence')
        if current_evidence:
            valid, reason = validate_measurement_evidence(current_evidence, new_mesh.positions, new_mesh.root_labels,
                new_map[rid], np.asarray(current['polyline']), triangles=new_mesh.triangles,
                body_start_index=current.get('body_start_index', 0), context=context)
            assert valid, (rid, 'New measurement evidence invalid', reason)
        current_supported = np.asarray((current_evidence or {}).get('native_vertex_indices', []), dtype=int)
        no_longer_selected = np.setdiff1d(supported, current_supported)
        newly_selected = np.setdiff1d(current_supported, supported)
        measurements = {}
        for metric in ('length', 'mean_diameter', 'volume'):
            previous_value = _finite_value(inventory.loc[rid, metric])
            current_value = _finite_value(new_inventory.loc[rid, metric])
            measurements[metric] = {'before': previous_value, 'after': current_value,
                'delta': current_value - previous_value if previous_value is not None and current_value is not None else None,
                'status_before': str(inventory.loc[rid, metric + '_measurement_status']),
                'status_after': str(new_inventory.loc[rid, metric + '_measurement_status'])}
        rows.append({'root_id': rid, 'owned_before': len(owned), 'owned_retained': retained,
                     'measurement_support_before': len(supported), 'measurement_support_reassigned': len(lost),
                     'reassigned_support_vertex_indices': lost.tolist(),
                     'measurement_support_after': len(current_supported),
                     'measurement_support_retained_in_new_selection': len(supported) - len(no_longer_selected),
                     'measurement_support_not_selected_after': len(no_longer_selected),
                     'not_selected_support_vertex_indices': no_longer_selected.tolist(),
                     'measurement_support_added_to_selection': len(newly_selected),
                     'newly_selected_support_vertex_indices': newly_selected.tolist(),
                     'measurements': measurements})
    mapped_labels = np.asarray(new_mesh.root_labels).copy()
    for rid in new_roots:
        mapped_labels[new_mesh.root_labels == new_map[rid]] = old_map[rid]
    changed = np.flatnonzero(old_mesh.root_labels != mapped_labels)
    result = {'sample': after.name, 'passed': True, 'native_geometry_unchanged': True,
              'immutable_top_unchanged': True, 'root_ids_and_hierarchy_preserved': True,
              'previously_measured_lateral_support_preserved': True,
              'changed_ownership_vertices': len(changed),
              'changed_vertex_indices': changed.tolist(),
              'new_uncertain_vertices': int(np.count_nonzero((mapped_labels == -2) & (old_mesh.root_labels != -2))),
              'roots_with_selected_support_loss': [row['root_id'] for row in rows if row['measurement_support_not_selected_after']],
              'selected_support_not_retained_count': sum(row['measurement_support_not_selected_after'] for row in rows),
              'selected_support_added_count': sum(row['measurement_support_added_to_selection'] for row in rows),
              'selected_support_changes_require_review': any(row['measurement_support_not_selected_after'] for row in rows),
              'measurement_change_policy': 'Report new fit selection and trait deltas; ownership preservation does not certify unchanged measurements.',
              'roots': rows,
              'baseline_metadata_sha256': hashlib.sha256((before/'support/metadata.json').read_bytes()).hexdigest()}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.before, args.after)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in {'roots', 'changed_vertex_indices'}}, indent=2))
