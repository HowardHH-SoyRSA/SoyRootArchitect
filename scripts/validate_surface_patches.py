"""Audit symmetric patch correction on a frozen complete result bundle.

No tracing or centerline refit is performed: label and support-trait changes
are attributable to this stage alone. Source files are hashed and untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from soyrootbio.editor.session import EditorSession
from soyrootbio.pipeline import _selected_base_exclusion_mask, _surface_connectivity_edges
from soyrootbio.surface_patches import correct_surface_patches
from soyrootbio.types import Normalization, RootPath
from validate_primary_o1_ownership import BatchSession, component_report, digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if out == source or source in out.parents:
        raise ValueError("output must be outside the immutable source bundle")
    out.mkdir(parents=True, exist_ok=False)
    protected = [p for p in source.rglob("*") if p.is_file()]
    hashes = {str(p): digest(p) for p in protected}
    meta = json.loads((source / "metadata.json").read_text())
    provenance = Path(meta.get("automatic_metadata_provenance", source / "metadata.json"))
    automatic = json.loads(provenance.read_text())
    session = BatchSession(source, session_dir=out / "session", load_existing_log=False)
    roots = {rid: r.clone() for rid, r in session.roots.items()}
    norm = Normalization(np.array(meta["normalization_minimum"]), meta["normalization_scale"])
    points = norm.transform_points(session.mesh.positions)
    primary = norm.transform_points(roots["primary"].points)
    laterals = sorted((r for r in roots.values() if r.root_id != "primary"), key=lambda r: r.numeric_label)
    numeric = np.array([0] + [r.numeric_label for r in laterals])
    before = session.mesh.root_labels.copy()
    mapped = before.copy()
    for label, original in enumerate(numeric):
        mapped[before == original] = label
    paths = [RootPath(root_id=r.root_id, parent_id=r.parent_id, order=r.order,
                      points=norm.transform_points(r.points), insertion_index=r.insertion_index,
                      body_start_index=r.body_start_index) for r in laterals]
    a = automatic["point_assignment"]
    excluded = _selected_base_exclusion_mask(
        points, norm.transform_points(np.array(a["base_point_source_coordinates"])[None])[0],
        np.array(a["base_tipward_direction"]), gravity=np.array(a["gravity_direction"]),
        collar_neighborhood_radius=a["base_collar_neighborhood_radius_normalized"],
        tolerance=a["above_base_tolerance_normalized"])
    started = time.perf_counter()
    after, stage = correct_surface_patches(points, mapped, primary, paths,
        d_bar=meta["d_bar_normalized"], triangles=session.mesh.triangles, excluded_mask=excluded)
    elapsed = time.perf_counter() - started
    save(out / "surface_patch_stage.json", stage)
    np.save(out / "labels_before.npy", before)
    compact_after = after.copy()
    after[after >= 0] = numeric[after[after >= 0]]
    np.save(out / "labels_after.npy", after)
    changed = before != after
    assert np.array_equal(after[excluded | (before < 0)], before[excluded | (before < 0)])
    assert stage["converged"] and np.all(np.diff(stage.get("energy", [])) < 0)
    EditorSession._recompute_traits(session)
    traits_before = pd.DataFrame([r.traits for r in session.roots.values()]).set_index("root_id").sort_index()
    for label in np.unique(after[changed]):
        root = next(r for r in session.roots.values() if r.numeric_label == label)
        session.apply_operation("assign_points", {"root_id": root.root_id,
            "indices": np.flatnonzero(changed & (after == label)).tolist(),
            "reason": "Symmetric mesh-component correction against frozen centerlines and observed body support."})
    session._validate_state()
    EditorSession._recompute_traits(session)
    traits_after = pd.DataFrame([r.traits for r in session.roots.values()]).set_index("root_id").sort_index()
    traits_before.to_csv(out / "traits_before.csv")
    traits_after.to_csv(out / "traits_after.csv")
    affected = {r.root_id for r in roots.values() if r.numeric_label in set(before[changed]) | set(after[changed])}
    unaffected = sorted(set(roots) - affected)
    pd.testing.assert_frame_equal(traits_before.loc[unaffected], traits_after.loc[unaffected], check_exact=True)
    trait_changes = []
    support_traits = {"mean_radius", "mean_diameter", "median_diameter", "minimum_diameter",
                      "maximum_diameter", "surface_area", "volume", "point_count"}
    for rid in sorted(affected):
        for col in traits_before.columns:
            b, a = traits_before.loc[rid, col], traits_after.loc[rid, col]
            if (pd.isna(b) and pd.isna(a)) or b == a:
                continue
            assert col in support_traits, (rid, col)
            trait_changes.append({"root_id": rid, "trait": col, "before": b, "after": a})
    save(out / "trait_changes.json", trait_changes)
    replay = BatchSession(source, session_dir=out / "session")
    np.testing.assert_array_equal(replay.mesh.root_labels, after)
    for rid, root in roots.items():
        for check in [session, replay]:
            now = check.roots[rid]
            np.testing.assert_array_equal(now.points, root.points)
            assert (now.order, now.parent_id, now.insertion_index, now.body_start_index) == (
                root.order, root.parent_id, root.insertion_index, root.body_start_index)
    edges, _ = _surface_connectivity_edges(points, session.mesh.triangles, meta["d_bar_normalized"])
    states = lambda values: {"assigned": int(np.sum(values >= 0)), "primary": int(np.sum(values == 0)),
                            "lateral": int(np.sum(values > 0)), "uncertain": int(np.sum(values == -2)),
                            "unassigned": int(np.sum(values == -1))}
    # A fresh invocation re-estimates evidence. Record its effect separately
    # from convergence under the first invocation's fixed component energy.
    repeated, _ = correct_surface_patches(points, compact_after, primary, paths,
        d_bar=meta["d_bar_normalized"], triangles=session.mesh.triangles, excluded_mask=excluded)
    assert hashes == {str(p): digest(p) for p in protected}
    report = {
        "source": str(source), "vertices": len(points), "triangles": len(session.mesh.triangles),
        "elapsed_seconds": elapsed, "changed_vertices": int(changed.sum()),
        "changed_patches": stage["reassigned_patch_count"], "passes": stage["passes"],
        "converged": stage["converged"], "energy": stage.get("energy", []),
        "before": states(before), "after": states(after), "root_count": len(roots),
        "order_counts": dict(Counter(r.order for r in roots.values())),
        "source_hashes_unchanged": True, "negative_and_excluded_labels_unchanged": True,
        "hierarchy_centerlines_attachments_unchanged": True, "operation_replay_matches": True,
        "unaffected_root_traits_unchanged": True, "only_support_traits_changed": True,
        "fresh_invocation_changed_vertices": int(np.sum(repeated != compact_after)),
        "affected_roots": sorted(affected),
        "components_before": component_report(before, edges, roots),
        "components_after": component_report(after, edges, roots),
    }
    save(out / "validation.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {
        "components_before", "components_after", "affected_roots"}}, indent=2))


if __name__ == "__main__":
    main()
