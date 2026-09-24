"""Read-only probe of proposed attachment geometry on the three audit bundles."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soyrootbio.attachment_constraint import assess_attachment_footprints
from soyrootbio.types import RootPath


AUDIT = Path(__file__).resolve().parents[1] / "outputs" / "three_sample_audit_20260922"
BUNDLES = Path(r"E:\Seafile\Test files for BioInsAlgo\SoyRootBio_outputs_20260922")
TARGETS = {
    "Kaixinlv_3-2_20260525": ["root-o1-010"],
    "W82_9cm_water_1-2_20260522": ["root-o1-002", "root-o2-008"],
    "W82_MS4-2_20260617": ["root-o1-005", "root-o1-002"],
}


def _root(row: dict) -> RootPath:
    return RootPath(
        root_id=row["root_id"], parent_id=row["parent_id"],
        order=int(row["root_order"]), points=np.asarray(row["polyline"], float),
        body_start_index=int(row.get("body_start_index", 0)),
    )


def main() -> None:
    results = {}
    for sample, target_ids in TARGETS.items():
        with (AUDIT / sample / "geometry.pkl").open("rb") as stream:
            points, saved_labels, faces, _, _ = pickle.load(stream)
        bundle = BUNDLES / sample
        metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        with (AUDIT / sample / "historical" / "capture.pkl").open("rb") as stream:
            capture = pickle.load(stream)
        primary_reference = (
            np.asarray(capture["prefit_primary"], float) *
            float(metadata["normalization_scale"]) +
            np.asarray(metadata["normalization_minimum"], float))
        rows = {row["root_id"]: row for row in json.loads(
            (bundle / "root_hierarchy.json").read_text(encoding="utf-8"))["roots"]}
        spacing = float(metadata["d_bar_normalized"] * metadata["normalization_scale"])
        sample_result = {}
        for target_id in target_ids:
            row = rows[target_id]
            parent_id = row["parent_id"]
            selected = ([rows[parent_id]] if parent_id != "primary" else []) + [row]
            roots = [_root(item) for item in selected]
            labels = np.full(len(saved_labels), -2, int)
            labels[saved_labels == 0] = 0
            for label, item in enumerate(selected, 1):
                numeric = int(item["centerline_assessment"]["numeric_label"])
                labels[saved_labels == numeric] = label
            labels[saved_labels == -1] = -1
            report = assess_attachment_footprints(
                points, labels, primary_reference,
                roots, triangles=faces, d_bar=spacing, include_competitors=False)
            result = report["junctions"][-1]
            sample_result[target_id] = result
            print(sample, target_id, result["status"],
                  "radius", result.get("child_radius"),
                  "faces", result.get("patch_face_count"), flush=True)
        results[sample] = sample_result
    output = AUDIT / "attachment_probe.json"
    output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
