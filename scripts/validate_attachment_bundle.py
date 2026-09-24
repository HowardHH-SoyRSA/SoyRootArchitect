"""Check hierarchy, labels, QC and exports of an attachment replay bundle."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soyrootbio.editor.ply import read_labeled_ply
from soyrootbio.topology import validate_root_tree
from soyrootbio.types import RootPath


def main(bundle: Path, source: Path | None) -> None:
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    hierarchy = json.loads((bundle / "root_hierarchy.json").read_text(encoding="utf-8"))
    rows = hierarchy["roots"]
    mesh = read_labeled_ply(bundle / "segmented_root_structure.ply")
    primary = np.asarray(rows[0]["polyline"], float)
    paths = [RootPath(
        root_id=row["root_id"], parent_id=row["parent_id"],
        order=int(row["root_order"]), points=np.asarray(row["polyline"], float),
        insertion_index=row["insertion_index"],
        insertion_point=np.asarray(row["insertion_point"], float),
        body_start_index=int(row.get("body_start_index", 0)),
    ) for row in rows[1:]]
    assert len(paths) == metadata["selected_lateral_count"]
    top = np.asarray(metadata["lateral_origin_constraint"]["primary_top_point_normalized"], float)
    top = top * float(metadata["normalization_scale"]) + np.asarray(
        metadata["normalization_minimum"], float)
    errors = validate_root_tree(paths, primary_path=primary,
                                primary_top_reference=top,
                                gravity=np.asarray(metadata["config"]["gravity"], float))
    row_by_id = {row["root_id"]: row for row in rows}
    flagged_postfit_length = [error for error in errors
                              if "centreline length" in error and
                              "centerline_refit_child_longer_than_parent" in
                              row_by_id.get(error.split(":", 1)[0], {}).get("qc_flags", [])]
    structural_errors = [error for error in errors if error not in flagged_postfit_length]
    assert not structural_errors, structural_errors
    traits = pd.read_csv(bundle / "root_traits.csv")
    assert set(traits.root_id) == {row["root_id"] for row in rows}
    assert len(traits) == len(rows)
    assert len(mesh.root_labels) == mesh.vertex_count
    assert np.all((mesh.root_labels >= -2) & (mesh.root_labels <= len(paths)))
    assert np.max(mesh.triangles) < len(mesh.positions)
    counts = {"assigned": int(np.sum(mesh.root_labels >= 0)),
              "unassigned": int(np.sum(mesh.root_labels == -1)),
              "uncertain": int(np.sum(mesh.root_labels == -2))}
    summary = metadata["point_assignment"]
    assert counts["assigned"] == summary["assigned_vertex_count"]
    assert counts["unassigned"] == summary["unassigned_vertex_count"]
    assert counts["uncertain"] == summary["uncertain_vertex_count"]
    restriction = metadata["attachment_constraint"]["final_contact_restriction"]
    final_junctions = metadata["attachment_constraint"].get(
        "after_centerline_reconciliation",
        metadata["attachment_constraint"]["after_ownership_cleanup"],
    )["junctions"]
    if "after_topology_repair" in metadata["attachment_constraint"]:
        for row, junction in zip(rows[1:], final_junctions):
            status = junction["status"]
            if status.startswith("rejected"):
                assert "attachment_footprint_rejected" in row["qc_flags"]
            elif status.startswith("unresolved"):
                assert "attachment_unresolved" in row["qc_flags"]
    restricted = [v for vertices in restriction["changed_by_root"].values() for v in vertices]
    assert len(restricted) == restriction["changed_vertex_count"]
    assert np.all(mesh.root_labels[restricted] == -2)
    postfit_restricted = [
        v
        for item in metadata["attachment_constraint"].get("postfit_reconciliation_passes", [])
        for vertices in item["restriction"]["changed_by_root"].values()
        for v in vertices
    ]
    assert len(postfit_restricted) == len(set(postfit_restricted))
    assert np.all(mesh.root_labels[postfit_restricted] == -2)
    rsml = ElementTree.parse(bundle / "root_system.rsml").getroot()
    rsml_roots = sum(element.tag.rsplit("}", 1)[-1] == "root" for element in rsml.iter())
    assert rsml_roots == len(rows)
    source_counts = None
    if source is not None:
        original = read_labeled_ply(source / "segmented_root_structure.ply")
        np.testing.assert_array_equal(mesh.positions, original.positions)
        np.testing.assert_array_equal(mesh.triangles, original.triangles)
        source_counts = {"assigned": int(np.sum(original.root_labels >= 0)),
                         "unassigned": int(np.sum(original.root_labels == -1)),
                         "uncertain": int(np.sum(original.root_labels == -2))}
    print(json.dumps({"bundle": str(bundle), "root_count": len(rows),
                      "mesh_vertices": mesh.vertex_count, "mesh_faces": mesh.face_count,
                      "label_counts": counts, "source_label_counts": source_counts,
                      "restricted_proximal_vertices": len(restricted),
                      "postfit_restricted_proximal_vertices": len(postfit_restricted),
                      "rsml_root_count": rsml_roots,
                      "hierarchy_errors": structural_errors,
                      "flagged_postfit_length_exceptions": flagged_postfit_length},
                     indent=2), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
