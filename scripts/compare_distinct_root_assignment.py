"""Compare every vertex, root, trait and hierarchy relation in paired replays."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soyrootbio.editor.ply import read_labeled_ply


def states(labels):
    return dict(assigned=int(np.sum(labels >= 0)), primary=int(np.sum(labels == 0)),
                lateral=int(np.sum(labels > 0)), uncertain=int(np.sum(labels == -2)),
                unassigned=int(np.sum(labels == -1)))


def compare(before, after, destination):
    destination.mkdir(parents=True, exist_ok=True)
    b = read_labeled_ply(before / "segmented_root_structure.ply")
    a = read_labeled_ply(after / "segmented_root_structure.ply")
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.triangles, b.triangles)
    old, new = b.root_labels, a.root_labels
    bm, am = [json.loads((p / "metadata.json").read_text()) for p in (before, after)]
    bh, ah = [{r["root_id"]: r for r in json.loads((p / "root_hierarchy.json").read_text())["roots"]}
              for p in (before, after)]
    bc, ac = [np.load(p / "competition.npy") for p in (before, after)]
    newly = np.setdiff1d(ac[:, 0], bc[:, 0])
    removed = np.setdiff1d(bc[:, 0], ac[:, 0])
    np.save(destination / "newly_competing_indices.npy", newly)
    np.save(destination / "changed_vertex_indices.npy", np.flatnonzero(old != new))
    transitions = Counter(zip(old[old != new].tolist(), new[old != new].tolist()))
    pd.DataFrame([dict(before=x, after=y, count=n) for (x, y), n in sorted(transitions.items())]).to_csv(
        destination / "label_transitions.csv", index=False)
    bt, at = [pd.read_csv(p / "root_traits.csv").set_index("root_id") for p in (before, after)]
    changes = []
    for rid in sorted(set(bt.index) & set(at.index)):
        for column in bt.columns.intersection(at.columns):
            x, y = bt.loc[rid, column], at.loc[rid, column]
            if (pd.isna(x) and pd.isna(y)) or x == y:
                continue
            changes.append(dict(root_id=rid, trait=column, before=x, after=y))
    pd.DataFrame(changes).to_csv(destination / "trait_changes.csv", index=False)
    common = sorted(set(bh) & set(ah))
    hierarchy = [{"root_id": rid, "field": key, "before": bh[rid].get(key), "after": ah[rid].get(key)}
        for rid in common for key in ("parent_id", "root_order", "insertion_index", "insertion_point", "body_start_index")
        if bh[rid].get(key) != ah[rid].get(key)]
    geometry_changed = [rid for rid in common if bh[rid]["polyline"] != ah[rid]["polyline"]]
    correction_changed = [rid for rid in common if
        bh[rid].get("centerline_assessment", {}).get("correction_input_geometry_fingerprint") !=
        ah[rid].get("centerline_assessment", {}).get("correction_input_geometry_fingerprint")]
    totals = {col: {"before": float(bt[col].sum()), "after": float(at[col].sum())}
              for col in ("length", "surface_area", "volume", "point_count")}
    identity_changes = [r for r in hierarchy if r["field"] in ("parent_id", "root_order")]
    order_counts = lambda roots: dict(sorted(Counter(r["root_order"] for r in roots.values()).items()))
    report = dict(sample=before.name, vertices=len(old), before=states(old), after=states(new),
        changed_vertices=int(np.sum(old != new)),
        reassigned_between_roots=int(np.sum((old >= 0) & (new >= 0) & (old != new))),
        newly_assigned=int(np.sum((old < 0) & (new >= 0))),
        newly_uncertain=int(np.sum((old != -2) & (new == -2))),
        resolved_uncertain=int(np.sum((old == -2) & (new >= 0))),
        raw_competing_before=len(bc), raw_competing_after=len(ac),
        newly_competing=len(newly), no_longer_competing=len(removed),
        roots_before=len(bh), roots_after=len(ah),
        added_roots=sorted(set(ah) - set(bh)), removed_roots=sorted(set(bh) - set(ah)),
        hierarchy_changes=hierarchy, changed_centerlines=geometry_changed,
        parent_or_order_changes=identity_changes,
        order_counts_before=order_counts(bh), order_counts_after=order_counts(ah),
        upstream_correction_input_geometry_changes=correction_changed,
        trait_change_count=len(changes), roots_with_trait_changes=len({c["root_id"] for c in changes}),
        unmeasurable_length_before=int(bt["length"].isna().sum()),
        unmeasurable_length_after=int(at["length"].isna().sum()),
        topology_report_before=bm.get("topology_report"), topology_report_after=am.get("topology_report"),
        trait_totals=totals, mesh_identical=True,
        analysis_points_before=bm["point_count"], analysis_points_after=am["point_count"],
        timings_before=bm["stage_timings_seconds"], timings_after=am["stage_timings_seconds"],
        competition_summary={k: v for k, v in am["distinct_root_competition"].items() if k != "radius_profiles_normalized"})
    (destination / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    before, after, out = map(Path, sys.argv[1:])
    rows = []
    for sample in sorted(before.iterdir()):
        if (sample / "metadata.json").exists() and (after / sample.name / "metadata.json").exists():
            row = compare(sample, after / sample.name, out / sample.name)
            rows.append(row)
            print(sample.name, row["before"], "->", row["after"], "new competition", row["newly_competing"], flush=True)
    (out / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
