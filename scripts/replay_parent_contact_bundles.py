"""Replay parent-contact reconciliation against immutable exported mesh bundles.

This is a stage audit, not a full retrace. It reads source bundles and writes
only a compact report to the requested workspace output path.
"""
from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from time import perf_counter
from pathlib import Path

import numpy as np

from soyrootbio.attachment_constraint import assess_attachment_footprints, restrict_rejected_contacts
from soyrootbio.editor.ply import read_labeled_ply
from soyrootbio.parent_contact import _native_edges, reconcile_parent_contacts
from soyrootbio.pipeline import (
    _assign_full_root_labels, _merge_final_contact_proposals,
    _selected_base_exclusion_mask,
)
from soyrootbio.types import RootPath


def replay(bundle: Path, *, attachment_audit: bool = False,
           assignment_benchmark: bool = False,
           simultaneous_audit: bool = False) -> dict:
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    hierarchy = json.loads((bundle / "root_hierarchy.json").read_text(encoding="utf-8"))
    mesh = read_labeled_ply(bundle / "segmented_root_structure.ply")
    minimum = np.asarray(metadata["normalization_minimum"], float)
    scale = float(metadata["normalization_scale"])
    spacing = float(metadata["d_bar_normalized"])
    points = (mesh.positions - minimum) / scale
    before = mesh.root_labels.copy()
    diameter = {}
    with (bundle / "csv" / "root_diameter.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["mean_diameter"]:
                diameter[row["root_id"]] = float(row["mean_diameter"]) / (2.0 * scale)
    paths = hierarchy["roots"]
    primary = (np.asarray(paths[0]["polyline"], float) - minimum) / scale
    roots = [
        RootPath(
            row["root_id"], (np.asarray(row["polyline"], float) - minimum) / scale,
            order=int(row["root_order"]), parent_id=row["parent_id"],
            body_start_index=int(row.get("body_start_index", 0)),
            centerline_assessment=row.get("centerline_assessment") or {},
            mean_radius=diameter.get(row["root_id"]),
            raw_start_point=((np.asarray(row["insertion_point"], float) - minimum) / scale
                             if row.get("insertion_point") is not None else None),
        ) for row in paths[1:]
    ]
    assignment = metadata["point_assignment"]
    excluded = _selected_base_exclusion_mask(
        points,
        (np.asarray(assignment["base_point_source_coordinates"], float) - minimum) / scale,
        np.asarray(assignment["base_tipward_direction"], float),
        gravity=np.asarray(assignment["gravity_direction"], float),
        collar_neighborhood_radius=float(assignment["base_collar_neighborhood_radius_normalized"]),
        tolerance=float(assignment["above_base_tolerance_normalized"]),
    )
    above_collar = excluded.copy()
    collar = np.asarray(metadata["joint_root_collar"]["neighborhood_vertex_indices"], int)
    excluded[collar] = True
    after, first = reconcile_parent_contacts(
        points, before, primary, roots, triangles=mesh.triangles, d_bar=spacing,
        cleanup_report=metadata["final_surface_cleanup"], excluded_mask=excluded,
    )
    repeated, final = reconcile_parent_contacts(
        points, after, primary, roots, triangles=mesh.triangles, d_bar=spacing,
        cleanup_report=metadata["final_surface_cleanup"], excluded_mask=excluded,
    )
    assert np.array_equal(after, repeated), "one stage replay must reach a stable label snapshot"
    changed = np.flatnonzero(before != after)
    assert not np.any(excluded[changed]), "protected collar or above-base point was changed"
    parent_label = {row.root_id: i for i, row in enumerate(roots, 1)}
    parent_label["primary"] = 0
    for row in first["contacts"]:
        if row["status"] != "reassigned_to_supported_parent":
            continue
        indices = np.flatnonzero(before == row["label"])
        assert np.count_nonzero(after[indices] == parent_label[row["parent_id"]]) >= row["changed_vertex_count"]
    all_edges = _native_edges(points, mesh.triangles, spacing)
    edges = all_edges[~excluded[all_edges[:, 0]] & ~excluded[all_edges[:, 1]]]
    orders = np.r_[0, [root.order for root in roots]]
    by_id = {"primary": 0, **{root.root_id: i for i, root in enumerate(roots, 1)}}
    parents = np.r_[-1, [by_id.get(root.parent_id, -1) for root in roots]]

    def parent_contacts(labels: np.ndarray) -> tuple[int, dict[str, int]]:
        left, right = labels[edges[:, 0]], labels[edges[:, 1]]
        child = np.where((left > 0) & (right == parents[left.clip(min=0)]), left, -1)
        child = np.where((right > 0) & (left == parents[right.clip(min=0)]), right, child)
        counts = np.bincount(child[child > 0], minlength=len(roots) + 1)
        return int(counts.sum()), {root_id: int(counts[by_id[root_id]]) if root_id in by_id else 0
                                   for root_id in sorted(selected_ids)}

    def higher_order_contact(labels: np.ndarray) -> int:
        left, right = labels[all_edges[:, 0]], labels[all_edges[:, 1]]
        return int(np.count_nonzero(((left == 0) & (right > 0) & (orders[right.clip(min=0)] >= 2)) |
                                    ((right == 0) & (left > 0) & (orders[left.clip(min=0)] >= 2))))

    selected_ids = {"root-o1-043", "root-o1-046", "root-o1-049", "root-o1-064",
                    "root-o1-075", "root-o1-076"}
    parent_before, selected_contact_before = parent_contacts(before)
    parent_after, selected_contact_after = parent_contacts(after)
    higher_before = higher_order_contact(before)
    higher_after = higher_order_contact(after)
    assert higher_after <= higher_before, "parent transfer introduced higher-order primary contact"
    selected = {
        root_id: [row for row in first["contacts"] if row["root_id"] == root_id]
        for root_id in sorted(selected_ids)
    }
    result = {
        "bundle": str(bundle), "vertex_count": int(len(points)),
        "root_count": len(roots) + 1, "above_collar_vertex_count": int(above_collar.sum()),
        "above_collar_assigned_before": int(np.count_nonzero(before[above_collar] >= 0)),
        "above_collar_assigned_after": int(np.count_nonzero(after[above_collar] >= 0)),
        "higher_order_primary_contact_edges_before": higher_before,
        "higher_order_primary_contact_edges_after": higher_after,
        "parent_contact_edges_before": parent_before,
        "parent_contact_edges_after": parent_after,
        "selected_parent_contact_edges_before": selected_contact_before,
        "selected_parent_contact_edges_after": selected_contact_after,
        "changed_vertex_count": int(len(changed)),
        "changed_source_root_ids": sorted({roots[int(label) - 1].root_id for label in before[changed]}),
        "remaining_contact_status_counts": {
            status: sum(row["status"] == status for row in final["contacts"])
            for status in sorted({row["status"] for row in final["contacts"]})
        },
        "first_pass": first, "final_audit": final,
        "selected_sn14_roots": selected,
    }
    if assignment_benchmark:
        started = perf_counter()
        assigned = _assign_full_root_labels(
            points, primary, roots, d_bar=spacing, excluded_mask=above_collar,
        )
        result["assignment_benchmark"] = {
            "elapsed_seconds": round(perf_counter() - started, 3),
            "assigned_vertex_count": int(np.count_nonzero(assigned >= 0)),
            "uncertain_vertex_count": int(np.count_nonzero(assigned == -2)),
            "above_collar_assigned_count": int(np.count_nonzero(assigned[above_collar] >= 0)),
        }
    if simultaneous_audit:
        attachment = assess_attachment_footprints(
            points, before, primary, roots, triangles=mesh.triangles,
            d_bar=spacing, excluded_mask=above_collar,
        )
        attachment_proposal, restriction = restrict_rejected_contacts(
            before, attachment, roots, points, triangles=mesh.triangles,
            d_bar=spacing,
        )
        parent_report = deepcopy(first)
        combined = _merge_final_contact_proposals(
            before, after, parent_report, attachment_proposal, restriction, roots,
        )
        result["simultaneous_audit"] = {
            "parent_changed_vertex_count": parent_report["changed_vertex_count"],
            "attachment_changed_vertex_count": restriction["changed_vertex_count"],
            "conflicting_root_ids": parent_report["competing_restriction_root_ids"],
            "recipient_restriction_root_ids": parent_report["recipient_restriction_root_ids"],
            "combined_changed_vertex_count": int(np.count_nonzero(combined != before)),
            "selected_parent_contact_edges_after": parent_contacts(combined)[1],
            "selected_attachment_status": {
                row["root_id"]: row["status"] for row in attachment["junctions"]
                if row["root_id"] in selected_ids
            },
            "above_collar_assigned_after": int(np.count_nonzero(combined[above_collar] >= 0)),
            "higher_order_primary_contact_edges_after": higher_order_contact(combined),
        }
    if attachment_audit:
        attachment = assess_attachment_footprints(
            points, after, primary, roots, triangles=mesh.triangles,
            d_bar=spacing, excluded_mask=above_collar,
        )
        restricted, restriction = restrict_rejected_contacts(
            after, attachment, roots, points, triangles=mesh.triangles,
            d_bar=spacing,
        )
        result["attachment_audit"] = {
            "status_counts": {
                status: sum(row["status"] == status for row in attachment["junctions"])
                for status in sorted({row["status"] for row in attachment["junctions"]})
            },
            "selected_roots": {
                row["root_id"]: {"status": row["status"],
                                 "contact_edge_count": row.get("contact_edge_count"),
                                 "patch_area": row.get("patch_area"),
                                 "area_limit": row.get("area_limit")}
                for row in attachment["junctions"] if row["root_id"] in selected_ids
            },
            "restricted_vertex_count": int(np.count_nonzero(restricted != after)),
            "restricted_by_root": {root_id: len(vertices) for root_id, vertices
                                   in restriction["changed_by_root"].items()},
            "connectivity_rollbacks": restriction["connectivity_rollbacks"],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attachment-audit", action="store_true")
    parser.add_argument("--assignment-benchmark", action="store_true")
    parser.add_argument("--simultaneous-audit", action="store_true")
    args = parser.parse_args()
    results = [replay(bundle, attachment_audit=args.attachment_audit,
                      assignment_benchmark=args.assignment_benchmark,
                      simultaneous_audit=args.simultaneous_audit)
               for bundle in args.bundle]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    for result in results:
        print(json.dumps({key: value for key, value in result.items()
                          if key not in {"first_pass", "final_audit", "selected_sn14_roots"}},
                         allow_nan=False))


if __name__ == "__main__":
    main()
