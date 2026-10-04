"""Audit trusted local pre-trimming snapshots produced by the comparison runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from soyrootbio.junction_tubes import reconcile_parent_owned_tubes
from soyrootbio.mesh_geometry import MeshGeometryContext
from audit_junction_comparison import components


def audit(bundle):
    # These are locally generated pipeline checkpoints, never external pickles.
    with (bundle/'before_child_tubes.pkl').open('rb') as stream:
        args, kwargs = pickle.load(stream)
    points, before, primary, roots = args
    input_labels = before.copy()
    input_primary = primary.copy()
    tree = [(r.root_id, r.parent_id, r.order) for r in roots]
    paths = [r.points.copy() for r in roots]
    context = MeshGeometryContext.build(points, kwargs['triangles'])
    after, report = reconcile_parent_owned_tubes(*args, **kwargs, mesh_context=context)
    assert [r.points.tolist() for r in roots] == [p.tolist() for p in paths]
    assert tree == [(r.root_id, r.parent_id, r.order) for r in roots]
    np.testing.assert_array_equal(before, input_labels)
    np.testing.assert_array_equal(primary, input_primary)
    changed = before != after
    excluded = kwargs['excluded_mask']
    np.testing.assert_array_equal(before[excluded | (before < 0)], after[excluded | (before < 0)])
    unsafe = np.unique(context.edges[context.edge_incidence != 2])
    np.testing.assert_array_equal(before[unsafe], after[unsafe])
    lookup = {'primary': 0, **{r.root_id:i for i,r in enumerate(roots, 1)}}
    for label in np.unique(after[changed]):
        ids = changed & (after == label)
        assert np.all(before[ids] == lookup[roots[label-1].parent_id]), "non-parent donor modified"
        if roots[label-1].order >= 2:
            boundary = context.edges[(after[context.edges[:, 0]] == label) ^
                                     (after[context.edges[:, 1]] == label)]
            assert not np.any(after[boundary] == 0), "proposal introduced primary contact"
    original_components, final_components = components(before, context.edges), components(after, context.edges)
    assert all(final_components.get(k, 0) <= count for k,count in original_components.items()), "native components increased"
    # Reverse numeric labels and root traversal order while retaining biological
    # identities. This checks frozen arbitration on the complete real mesh.
    permutation = np.r_[0, np.arange(len(roots), 0, -1)]
    permuted_before = before.copy()
    permuted_before[before >= 0] = permutation[before[before >= 0]]
    permuted, _ = reconcile_parent_owned_tubes(points, permuted_before, primary, list(reversed(roots)),
                                      **kwargs, mesh_context=context)
    permuted[permuted >= 0] = permutation[permuted[permuted >= 0]]
    np.testing.assert_array_equal(permuted, after)
    expected = json.loads((bundle/'metadata.json').read_text())['parent_owned_tube_reconciliation']
    assert report['changed_vertex_indices'] == expected['changed_vertex_indices']
    assert report['transferred_vertex_count'] == expected['transferred_vertex_count']
    return dict(sample=bundle.name, moved_vertices=int(changed.sum()),
                replay_matches_pipeline_decisions=True, reverse_root_order_identical=True,
                paths_and_hierarchy_unchanged=True, changes_only_parent_to_direct_child=True,
                native_component_counts_do_not_increase=True, exclusions_and_unsafe_seams_preserved=True,
                higher_order_proposals_do_not_contact_primary=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--sample')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    samples = [args.sample] if args.sample else sorted(p.name for p in args.candidate.iterdir() if (p/'metadata.json').exists())
    reports = []
    for sample in samples:
        report = audit(args.candidate/sample)
        reports.append(report)
        print(json.dumps(report), flush=True)
    args.output.write_text(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
