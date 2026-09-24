"""Check a complete exported bundle after parent-contact reconciliation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from replay_parent_contact_bundles import replay
from soyrootbio.editor.ply import read_labeled_ply
from soyrootbio.geometry import is_above_primary_top


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate(bundle: Path) -> dict:
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    hierarchy = json.loads((bundle / "root_hierarchy.json").read_text(encoding="utf-8"))
    roots = hierarchy["roots"]
    by_id = {root["root_id"]: root for root in roots}
    assert len(by_id) == len(roots) and roots[0]["root_id"] == "primary"
    assert metadata["selected_lateral_count"] == len(roots) - 1
    mesh = read_labeled_ply(bundle / "segmented_root_structure.ply")
    labels = mesh.root_labels
    assert np.all((labels >= -2) & (labels < len(roots)))
    assert metadata["point_assignment"]["total_vertex_count"] == len(labels)
    table = _csv_rows(bundle / "csv" / "root_label_map.csv")
    numeric = {int(row["numeric_label"]): row for row in table}
    for label, root in enumerate(roots):
        assert numeric[label]["root_id"] == root["root_id"]
        assert int(numeric[label]["root_order"]) == root["root_order"]
        assert np.all(mesh.root_orders[labels == label] == int(root["root_order"]))
    for name in ("root_topology.csv", "root_lengths.csv", "root_diameter.csv"):
        rows = _csv_rows(bundle / "csv" / name)
        assert {row["root_id"] for row in rows} == set(by_id), name
    assert {row["root_id"] for row in _csv_rows(bundle / "lateral_skeletons.csv")} == set(by_id) - {"primary"}
    assert {row["root_id"] for row in _csv_rows(bundle / "primary_skeleton.csv")} == {"primary"}
    rsml = ElementTree.parse(bundle / "root_system.rsml")
    assert {node.get("id") for node in rsml.findall(".//root")} == set(by_id)
    for root in roots[1:]:
        parent = by_id[root["parent_id"]]
        assert root["root_order"] == parent["root_order"] + 1
    top = np.asarray(metadata["final_centerline_fitting"]["primary_top_point_normalized"], float)
    gravity = np.asarray(metadata["gravity_vector"], float)
    minimum = np.asarray(metadata["normalization_minimum"], float)
    scale = float(metadata["normalization_scale"])
    missing_origins = []
    for root in roots[1:]:
        origin = root.get("insertion_point")
        if origin is None:
            missing_origins.append(root["root_id"])
            continue
        normalized = (np.asarray(origin, float) - minimum) / scale
        assert not is_above_primary_top(normalized, top[None, :], gravity=gravity), root["root_id"]
    assert not missing_origins, f"missing lateral origins: {missing_origins}"
    stage = metadata["parent_contact_reconciliation"]
    assert stage["passes"]
    assert stage["status"] == stage["passes"][-1]["status"]
    audit = replay(bundle)
    assert audit["changed_vertex_count"] == 0, "exported parent contacts are not stable"
    assert audit["above_collar_assigned_after"] == 0
    # The export stores final surface diameter, not the trace-time mean_radius
    # used by this pass. Check QC against the recorded final pipeline decision;
    # the independent reconstruction may classify extra bodies as uncertain.
    unresolved = {row["root_id"] for row in stage["passes"][-1]["contacts"]
                  if row["status"].startswith("unresolved")}
    assert all("parent_contact_unresolved" in by_id[root_id]["qc_flags"]
               for root_id in unresolved), "unresolved parent contact lacks root QC"
    replay_unresolved = {row["root_id"] for row in audit["final_audit"]["contacts"]
                         if row["status"].startswith("unresolved")}
    primary_contact = metadata["higher_order_primary_contact"]
    if audit["higher_order_primary_contact_edges_after"]:
        assert primary_contact["requires_correction"], "higher-order primary contact is unreported"
    return {
        "bundle": str(bundle), "vertex_count": len(labels), "root_count": len(roots),
        "parent_contact_reconciliation_status": stage["status"],
        "parent_contact_reassigned_vertices": stage["changed_vertex_count"],
        "remaining_unresolved_parent_contact_roots": sorted(unresolved),
        "replay_only_uncertain_root_ids": sorted(replay_unresolved - unresolved),
        "remaining_higher_order_primary_contact_edges": audit["higher_order_primary_contact_edges_after"],
        "above_collar_assigned_count": audit["above_collar_assigned_after"],
        "repeat_parent_contact_changed_vertices": audit["changed_vertex_count"],
        "origin_top_invariant": "passed", "label_order_trait_export_consistency": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
