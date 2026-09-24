"""Read-only attachment evidence sweep across six established bundles."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soyrootbio.attachment_constraint import assess_attachment_footprints
from soyrootbio.editor.ply import read_labeled_ply
from soyrootbio.types import RootPath


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "final_centerline_six_sample_assessment"
DESTINATION = ROOT / "outputs" / "attachment_validation_20260923" / "six_bundle_probe.json"


def main() -> None:
    results = {}
    for sample_dir in sorted(path for path in SOURCE.iterdir() if path.is_dir()):
        bundle = sample_dir / "bundle"
        if not bundle.is_dir():
            continue
        hierarchy_file = bundle / "root_hierarchy.json"
        mesh_file = bundle / "segmented_root_structure.ply"
        metadata_file = bundle / "metadata.json"
        rows = json.loads(hierarchy_file.read_text(encoding="utf-8"))["roots"]
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        mesh = read_labeled_ply(mesh_file)
        curves = np.load(sample_dir / "curves.npz")
        scale = float(metadata["normalization_scale"])
        minimum = np.asarray(metadata["normalization_minimum"], float)
        primary = np.asarray(curves["primary"], float) * scale + minimum
        labels = np.asarray(mesh.root_labels, int).copy()
        roots = []
        for label, row in enumerate(rows[1:], 1):
            numeric = int(row["centerline_assessment"]["numeric_label"])
            labels[mesh.root_labels == numeric] = label
            roots.append(RootPath(
                root_id=row["root_id"], parent_id=row["parent_id"],
                order=int(row["root_order"]),
                points=np.asarray(row["polyline"], float),
                body_start_index=int(row.get("body_start_index", 0)),
                raw_start_point=(np.asarray(row["insertion_point"], float)
                                 if row.get("insertion_point") is not None else None),
            ))
        report = assess_attachment_footprints(
            mesh.positions, labels, primary, roots,
            triangles=mesh.triangles,
            d_bar=float(metadata["d_bar_normalized"]) * scale,
        )
        counts = dict(sorted(Counter(row["status"] for row in report["junctions"]).items()))
        noteworthy = {row["root_id"]: row["status"] for row in report["junctions"]
                      if row["status"].startswith("rejected")}
        rejected_metrics = {
            row["root_id"]: {
                "area_ratio": row.get("area", 0) / row.get("area_limit", 1),
                "longitudinal_ratio": row.get("longitudinal_extent", 0) /
                                      row.get("longitudinal_limit", 1),
                "geodesic_diameter_ratio": row.get("geodesic_diameter", 0) /
                                           row.get("geodesic_diameter_limit", 1),
                "inverse_compactness_ratio": row.get("inverse_compactness", 0) /
                                             row.get("inverse_compactness_limit", 1),
                "neighboring_insertions": row.get(
                    "neighboring_insertions_for_reassessment", []),
            }
            for row in report["junctions"]
            if row["status"].startswith("rejected")}
        target_ids = {"root-o1-010", "root-o1-002", "root-o2-008",
                      "root-o1-005", "root-o1-027", "root-o1-066", "root-o1-077"}
        targets = {row["root_id"]: row["status"] for row in report["junctions"]
                   if row["root_id"] in target_ids}
        results[sample_dir.name] = {
            "root_count": len(rows), "vertex_count": mesh.vertex_count,
            "face_count": mesh.face_count, "status_counts": counts,
            "rejected_roots": noteworthy, "rejected_metrics": rejected_metrics,
            "named_targets": targets,
            "source_sha256": {
                "root_hierarchy.json": hashlib.sha256(hierarchy_file.read_bytes()).hexdigest(),
                "segmented_root_structure.ply": hashlib.sha256(mesh_file.read_bytes()).hexdigest(),
            },
        }
        print(sample_dir.name, len(rows), counts, flush=True)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(DESTINATION, flush=True)


if __name__ == "__main__":
    main()
