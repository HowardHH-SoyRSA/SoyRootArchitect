"""Read-only audit of higher-order primary contacts in an exported bundle."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soyrootbio.editor.ply import read_labeled_ply
from soyrootbio.primary_contact import restrict_higher_order_primary_contacts
from soyrootbio.types import RootPath


def main(bundle: Path) -> None:
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    hierarchy = json.loads((bundle / "root_hierarchy.json").read_text(encoding="utf-8"))
    mesh = read_labeled_ply(bundle / "segmented_root_structure.ply")
    roots = [RootPath(row["root_id"], np.asarray(row["polyline"], float),
                      order=int(row["root_order"]), parent_id=str(row["parent_id"]))
             for row in hierarchy["roots"][1:]]
    spacing = float(metadata["d_bar_normalized"]) * float(metadata["normalization_scale"])
    corrected, report = restrict_higher_order_primary_contacts(
        mesh.positions, mesh.root_labels, roots, triangles=mesh.triangles,
        d_bar=spacing)
    before = np.asarray(mesh.root_labels, int)
    changed = corrected != before
    assert np.all(corrected[changed] == -2)
    assert np.all((before[changed] == 0) |
                  np.isin(before[changed], [i for i, root in enumerate(roots, 1)
                                           if root.order > 1]))
    edges = np.unique(np.sort(np.vstack((mesh.triangles[:, [0, 1]],
                                          mesh.triangles[:, [1, 2]],
                                          mesh.triangles[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(mesh.positions[edges[:, 0]] - mesh.positions[edges[:, 1]], axis=1)
    edges = edges[lengths <= 4 * spacing]
    orders = np.array([0, *(root.order for root in roots)])

    def forbidden_contact_count(owner: np.ndarray) -> int:
        left, right = owner[edges[:, 0]], owner[edges[:, 1]]
        other = np.maximum(left, right)
        contact = (((left == 0) & (right > 0)) |
                   ((right == 0) & (left > 0)))
        return int(np.count_nonzero(orders[other[contact]] > 1))

    before_contact_edges = forbidden_contact_count(before)
    after_contact_edges = forbidden_contact_count(corrected)
    assert after_contact_edges == sum(row["remaining_contact_edge_count"]
                                      for row in report["contacts"])
    print(json.dumps({"bundle": str(bundle), "simulation_only": True,
                      "root_count": len(roots), "status": report["status"],
                      "contact_root_count": report["contact_root_count"],
                      "unresolved_root_count": report["unresolved_root_count"],
                      "changed_vertex_count": report["changed_vertex_count"],
                      "changed_primary_vertex_count": int(np.count_nonzero(changed & (before == 0))),
                      "changed_higher_order_vertex_count": int(np.count_nonzero(changed & (before > 0))),
                      "forbidden_contact_edges_before": before_contact_edges,
                      "forbidden_contact_edges_after": after_contact_edges,
                      "contacts": report["contacts"]}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
