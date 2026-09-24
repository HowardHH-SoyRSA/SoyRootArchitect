"""Post-refit diagnostic, not a replay of the pre-refit restriction decision.

Exported hierarchy polylines are refitted after restriction, so their arc
stations can differ from the paths used by the in-process connectivity guard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soyrootbio.attachment_constraint import restrict_rejected_contacts
from soyrootbio.editor.ply import read_labeled_ply
from soyrootbio.types import RootPath


def main(bundle: Path) -> None:
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    hierarchy = json.loads((bundle / "root_hierarchy.json").read_text(encoding="utf-8"))["roots"]
    mesh = read_labeled_ply(bundle / "segmented_root_structure.ply")
    scale = float(metadata["normalization_scale"])
    minimum = np.asarray(metadata["normalization_minimum"], float)
    points = (mesh.positions - minimum) / scale
    roots = [RootPath(row["root_id"], (np.asarray(row["polyline"], float)-minimum)/scale,
                      parent_id=row["parent_id"], order=int(row["root_order"]),
                      body_start_index=int(row.get("body_start_index", 0)))
             for row in hierarchy[1:]]
    restriction = metadata["attachment_constraint"]["final_contact_restriction"]
    before = np.asarray(mesh.root_labels, int).copy()
    labels_by_id = {root.root_id: label for label, root in enumerate(roots, 1)}
    for root_id, vertices in restriction["changed_by_root"].items():
        before[np.asarray(vertices, int)] = labels_by_id[root_id]
    report = metadata["attachment_constraint"]["after_ownership_cleanup"]
    after, guarded = restrict_rejected_contacts(
        before, report, roots, points, triangles=mesh.triangles,
        d_bar=float(metadata["d_bar_normalized"]))
    expected = {root_id: set(vertices) for root_id, vertices in
                restriction["changed_by_root"].items()}
    actual = {root_id: set(vertices) for root_id, vertices in
              guarded["changed_by_root"].items()}
    edges = np.sort(np.vstack((mesh.triangles[:, [0, 1]],
                               mesh.triangles[:, [1, 2]],
                               mesh.triangles[:, [2, 0]])), axis=1)
    edges = np.unique(edges, axis=0)
    edges = edges[np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
                  <= 4*float(metadata["d_bar_normalized"])]
    newly_split_components = {}
    for root_id in restriction["changed_by_root"]:
        label = labels_by_id[root_id]
        old_edges = edges[(before[edges[:, 0]] == label) &
                          (before[edges[:, 1]] == label)]
        new_edges = edges[(mesh.root_labels[edges[:, 0]] == label) &
                          (mesh.root_labels[edges[:, 1]] == label)]
        def components(selected_edges):
            return connected_components(coo_matrix(
                (np.ones(2*len(selected_edges)),
                 (np.r_[selected_edges[:, 0], selected_edges[:, 1]],
                  np.r_[selected_edges[:, 1], selected_edges[:, 0]])),
                shape=(len(points), len(points))).tocsr(), directed=False)[1]
        old_component = components(old_edges)
        new_component = components(new_edges)
        retained = np.flatnonzero(mesh.root_labels == label)
        split_count = sum(np.unique(new_component[retained[
            old_component[retained] == component]]).size > 1
            for component in np.unique(old_component[retained]))
        newly_split_components[root_id] = int(split_count)
    result = {"diagnostic": "post_refit_centerlines_not_authoritative_for_pre_refit_guard",
              "in_process_connectivity_rollbacks": restriction["connectivity_rollbacks"],
              "original_restriction_vertices": restriction["changed_vertex_count"],
              "guarded_restriction_vertices": guarded["changed_vertex_count"],
              "connectivity_rollbacks": guarded["connectivity_rollbacks"],
              "newly_split_native_components": newly_split_components,
              "native_connectivity_preserved": not any(newly_split_components.values()),
              "identical_changed_vertices": expected == actual,
              "identical_exported_labels": bool(np.array_equal(after, mesh.root_labels))}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
