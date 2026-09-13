"""Reproducible audit of the specified immutable bundle and its editor log."""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from soyrootbio.editor.ply import read_labeled_ply
from soyrootbio.editor.session import EditorSession

SOURCE = Path(r'E:\Seafile\Test files for BioInsAlgo\SoyRootBio_outputs\w5168-3_m4-2_20260415_2')
OUT = Path(r'E:\SoyRSA Build\outputs\w5168_assignment_repair_20260912')


class BatchSession(EditorSession):
    """Defer presentation and derived traits; retain every edit validation."""
    def public_state(self):
        return {}

    def _recompute_traits(self):
        pass


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def edges_of(triangles):
    edges = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)


def component_report(labels, edges, roots):
    same = labels[edges[:, 0]] == labels[edges[:, 1]]
    e = edges[same]
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(labels), len(labels))).tocsr()
    _, cc = connected_components(g, directed=False)
    report = []
    for root in roots.values():
        ix = np.flatnonzero(labels == root.numeric_label)
        sizes = np.sort(np.unique(cc[ix], return_counts=True)[1])[::-1]
        report.append({'root_id': root.root_id, 'label': root.numeric_label, 'points': len(ix),
                       'components': len(sizes), 'largest': int(sizes[0]) if len(sizes) else 0,
                       'outside_largest': int(sizes[1:].sum()), 'sizes': sizes[:12].tolist()})
    return report


def load_session():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'session'
    if not target.exists():
        shutil.copytree(SOURCE / '.soyrootbio-editor', target,
                        ignore=shutil.ignore_patterns('materialised'))
    return BatchSession(SOURCE, session_dir=target)


def main():
    session = load_session()
    edited = read_labeled_ply(SOURCE / '.soyrootbio-editor/materialised/edited_segmented_root_structure.ply')
    assert np.array_equal(edited.positions, session.mesh.positions)
    assert np.array_equal(edited.triangles, session.mesh.triangles)
    assert np.array_equal(edited.root_labels, session.mesh.root_labels), 'Export differs from live log replay'
    exported_hierarchy = json.loads((SOURCE / '.soyrootbio-editor/materialised/edited_root_hierarchy.json').read_text())
    for row in exported_hierarchy['roots']:
        root = session.roots[row['root_id']]
        assert np.allclose(root.points, row['polyline'], atol=1e-10, rtol=0)
        assert root.parent_id == row['parent_id'] and root.order == row['root_order']
    baseline, labels = session._baseline_labels, session.mesh.root_labels
    events = [json.loads(x) for x in session.log_path.read_text().splitlines() if x.strip()]
    pairs, counts = np.unique(np.column_stack([baseline, labels]), axis=0, return_counts=True)
    before_ids = {r.numeric_label: r.root_id for r in session._baseline_roots.values()}
    after_ids = {r.numeric_label: r.root_id for r in session.roots.values()}
    edges = edges_of(session.mesh.triangles)
    meta = json.loads((SOURCE / 'metadata.json').read_text())
    report = {
        'source': str(SOURCE), 'replay_matches_export': True, 'sequence': session._sequence,
        'log_events': dict(Counter(e['event'] for e in events)),
        'active_operations': dict(Counter(f.operation.type for f in session._history)),
        'before_roots': len(session._baseline_roots), 'edited_roots': len(session.roots),
        'before_orders': dict(Counter(r.order for r in session._baseline_roots.values())),
        'edited_orders': dict(Counter(r.order for r in session.roots.values())),
        'before_states': {'assigned': int((baseline >= 0).sum()), 'uncertain': int((baseline == -2).sum()), 'unassigned': int((baseline == -1).sum())},
        'edited_states': {'assigned': int((labels >= 0).sum()), 'uncertain': int((labels == -2).sum()), 'unassigned': int((labels == -1).sum())},
        'changed_vertices': int((baseline != labels).sum()),
        'transitions': sorted([{'before': before_ids.get(int(a), str(a)), 'after': after_ids.get(int(b), str(b)), 'count': int(n)} for (a,b),n in zip(pairs, counts) if a != b], key=lambda x: -x['count']),
        'baseline_components': component_report(baseline, edges, session._baseline_roots),
        'edited_components': component_report(labels, edges, session.roots),
        'point_assignment_metadata': meta['point_assignment'],
        'new_roots': [r.root_id for r in session.roots.values() if r.root_id not in session._baseline_roots],
        'deleted_roots': [r for r in session._baseline_roots if r not in session.roots],
    }
    save_json(OUT / 'audit.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('baseline_components', 'edited_components', 'point_assignment_metadata', 'transitions')}, indent=2), flush=True)
    print('Largest ownership changes:', json.dumps(report['transitions'][:18], indent=2), flush=True)
    print('Edited components:', json.dumps(sorted(report['edited_components'], key=lambda x:-x['outside_largest'])[:12], indent=2), flush=True)


if __name__ == '__main__':
    main()
