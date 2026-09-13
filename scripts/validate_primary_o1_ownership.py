"""Apply and audit ownership on a complete edited bundle without tracing anew.

Usage: python scripts/validate_primary_o1_ownership.py --source BUNDLE --output NEW_DIR
The source is immutable; centerlines and hierarchy are frozen to isolate labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from soyrootbio.editor.session import EditorSession
from soyrootbio.editor.ply import read_labeled_ply
from soyrootbio.pipeline import _resolve_primary_o1_ownership, _selected_base_exclusion_mask, _surface_connectivity_edges
from soyrootbio.types import Normalization, RootPath


class BatchSession(EditorSession):
    """Replay validated operations, computing derived traits once per stage."""
    def public_state(self):
        return {}

    def _recompute_traits(self):
        pass


def component_report(labels, edges, roots):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e = edges[labels[edges[:, 0]] == labels[edges[:, 1]]]
    graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(labels), len(labels))).tocsr()
    _, components = connected_components(graph, directed=False)
    rows = []
    for root in roots.values():
        ids = np.flatnonzero(labels == root.numeric_label)
        sizes = np.sort(np.unique(components[ids], return_counts=True)[1])[::-1]
        rows.append(dict(root_id=root.root_id, label=root.numeric_label, points=len(ids),
                         components=len(sizes), largest=int(sizes[0]) if len(sizes) else 0,
                         outside_largest=int(sizes[1:].sum()), sizes=sizes[:12].tolist()))
    return rows


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False,
                              default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist()), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    protected = [p for p in source.rglob("*") if p.is_file()]
    hashes = {str(p): digest(p) for p in protected}
    meta = json.loads((source / "metadata.json").read_text())
    provenance = Path(meta.get("automatic_metadata_provenance", source / "metadata.json"))
    automatic = json.loads(provenance.read_text())
    s = BatchSession(source, session_dir=out / "session", load_existing_log=False)
    before = s.mesh.root_labels.copy()
    roots = {rid: r.clone() for rid, r in s.roots.items()}
    norm = Normalization(np.array(meta["normalization_minimum"]), meta["normalization_scale"])
    points = norm.transform_points(s.mesh.positions)
    primary = norm.transform_points(s.roots["primary"].points)
    laterals = sorted((r for r in s.roots.values() if r.root_id != "primary"), key=lambda r: r.numeric_label)
    numeric = np.array([0] + [r.numeric_label for r in laterals])
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
    EditorSession._recompute_traits(s)
    traits_before = pd.DataFrame([r.traits for r in s.roots.values()]).set_index("root_id").sort_index()
    traits_before.to_csv(out / "traits_before.csv")
    started = time.perf_counter()
    after, stage = _resolve_primary_o1_ownership(points, mapped, primary, paths,
        d_bar=meta["d_bar_normalized"], triangles=s.mesh.triangles, excluded_mask=excluded)
    elapsed = time.perf_counter() - started
    save(out / "ownership_stage.json", stage)
    positive = after >= 0
    after[positive] = numeric[after[positive]]
    np.save(out / "labels_before.npy", before)
    np.save(out / "labels_after.npy", after)
    changed = before != after
    np.save(out / "changed_vertex_indices.npy", np.flatnonzero(changed))
    assert np.all(((before[changed] == 0) & (after[changed] > 0)) |
                  ((before[changed] == -2) & excluded[changed] & (after[changed] == -1)))
    regions = np.zeros(len(points), dtype=bool)
    for row in stage["junctions"]:
        if "region_radius" in row:
            regions |= np.linalg.norm(points - row["region_center"], axis=1) <= row["region_radius"]
    assert np.array_equal(after[~regions & ~excluded], before[~regions & ~excluded])
    assert np.array_equal(after[excluded & (before != -2)], before[excluded & (before != -2)])
    # This editor API logs assignments to roots. Exclusion-only demotions are
    # retained in the separate audit arrays if a supplied bundle needs them.
    if np.any(after[changed] < 0):
        raise RuntimeError("Bundle requires exclusion demotions; audit saved, no partial export written")
    for label in np.unique(after[changed]):
        r = next(r for r in s.roots.values() if r.numeric_label == label)
        s.apply_operation("assign_points", {"root_id": r.root_id,
            "indices": np.flatnonzero(changed & (after == label)).tolist(),
            "reason": "Bounded primary-O1 competition: exposed, segment-supported surface connected to existing child ownership."})
    s._validate_state()
    assert np.array_equal(s.mesh.root_labels, after)
    for rid, r in roots.items():
        now = s.roots[rid]
        assert r.parent_id == now.parent_id and r.order == now.order and r.insertion_index == now.insertion_index
        np.testing.assert_array_equal(r.points, now.points)
    EditorSession._recompute_traits(s)
    traits_after = pd.DataFrame([r.traits for r in s.roots.values()]).set_index("root_id").sort_index()
    traits_after.to_csv(out / "traits_after.csv")
    affected = {r.root_id for r in s.roots.values() if r.numeric_label in set(before[changed]) | set(after[changed])}
    unaffected = sorted(set(roots) - affected)
    pd.testing.assert_frame_equal(traits_before.loc[unaffected], traits_after.loc[unaffected], check_exact=True)
    changed_traits = []
    for rid in sorted(affected):
        for col in traits_before.columns:
            b, a = traits_before.loc[rid, col], traits_after.loc[rid, col]
            if (pd.isna(b) and pd.isna(a)) or b == a:
                continue
            changed_traits.append({"root_id": rid, "trait": col, "before": b, "after": a})
    support_traits = {"mean_radius", "mean_diameter", "median_diameter", "minimum_diameter",
                      "maximum_diameter", "surface_area", "volume", "point_count"}
    assert all(row["trait"] in support_traits for row in changed_traits)
    for root in s.roots.values():
        assert traits_after.loc[root.root_id, "point_count"] == np.sum(after == root.numeric_label)
    save(out / "trait_changes.json", changed_traits)
    edges, _ = _surface_connectivity_edges(points, s.mesh.triangles, meta["d_bar_normalized"])
    counts_before = component_report(before, edges, roots)
    counts_after = component_report(after, edges, s.roots)
    for b, a in zip(counts_before, counts_after):
        assert b["root_id"] == a["root_id"]
        if a["root_id"] != "primary":
            assert a["components"] <= b["components"], a["root_id"]
    bundle = s.export_materialised(out / "bundle")
    for src, dst in {"edited_segmented_root_structure.ply": "segmented_root_structure.ply",
                     "edited_root_hierarchy.json": "root_hierarchy.json",
                     "edited_root_traits.csv": "root_traits.csv", "edited_root_system.rsml": "root_system.rsml"}.items():
        shutil.copy2(bundle / src, bundle / dst)
    (bundle / "csv").mkdir(exist_ok=True)
    shutil.copy2(bundle / "edited_root_label_map.csv", bundle / "csv/root_label_map.csv")
    for name in ["primary_guidance.json", "primary_skeleton.csv", "lateral_skeletons.csv"]:
        if (source / name).exists():
            shutil.copy2(source / name, bundle / name)
    metadata = dict(meta)
    metadata["primary_o1_ownership"] = stage
    metadata["ownership_source_bundle"] = str(source)
    metadata["selected_lateral_count"] = len(laterals)
    metadata["selected_order_counts"] = dict(Counter(r.order for r in laterals))
    metadata["point_assignment"] = dict(meta["point_assignment"],
        repair_policy="Primary protrusions reassigned by bounded exposed-child surface competition; other labels and geometry preserved.",
        assigned_vertex_count=int(np.sum(after >= 0)), uncertain_vertex_count=int(np.sum(after == -2)),
        unassigned_vertex_count=int(np.sum(after == -1)), primary_assigned_vertex_count=int(np.sum(after == 0)),
        lateral_assigned_vertex_count=int(np.sum(after > 0)))
    save(bundle / "metadata.json", metadata)
    replay = BatchSession(source, session_dir=out / "session")
    loaded = BatchSession(bundle, session_dir=out / "reload", load_existing_log=False)
    exported = read_labeled_ply(bundle / "segmented_root_structure.ply")
    for checked in [replay, loaded]:
        np.testing.assert_array_equal(checked.mesh.root_labels, after)
        assert set(checked.roots) == set(roots)
        for rid, r in roots.items():
            np.testing.assert_array_equal(checked.roots[rid].points, r.points)
            assert checked.roots[rid].parent_id == r.parent_id
            assert checked.roots[rid].order == r.order
    np.testing.assert_array_equal(exported.positions, s.mesh.positions)
    np.testing.assert_array_equal(exported.triangles, s.mesh.triangles)
    assert hashes == {str(p): digest(p) for p in protected}
    states = lambda labels: {"primary": int(np.sum(labels == 0)), "lateral": int(np.sum(labels > 0)),
        "assigned": int(np.sum(labels >= 0)), "uncertain": int(np.sum(labels == -2)), "unassigned": int(np.sum(labels == -1))}
    report = dict(source=str(source), output=str(bundle), source_hashes_unchanged=hashes,
        vertices=len(points), triangles=len(s.mesh.triangles), elapsed_seconds=elapsed,
        before=states(before), after=states(after), changed_vertices=int(changed.sum()),
        root_count_before=len(roots), root_count_after=len(s.roots), orders=dict(Counter(r.order for r in roots.values())),
        hierarchy_and_all_centerlines_unchanged=True, all_unaffected_traits_identical=True,
        unaffected_roots=unaffected, affected_roots=sorted(affected),
        outside_competition_regions_unchanged=True, collar_exclusions_preserved=True,
        full_mesh_unchanged=True, replay_and_standard_bundle_reload_match=True,
        components_before=counts_before, components_after=counts_after)
    save(out / "validation.json", report)
    preview(points, before, after, stage, out)
    print(json.dumps({k: v for k, v in report.items() if k not in ["source_hashes_unchanged", "components_before", "components_after", "unaffected_roots"]}, indent=2))


def preview(points, before, after, stage, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(stage["junctions"], key=lambda r: -r["transferred_vertex_count"])[:4]
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 4 * len(rows)), squeeze=False, layout="constrained")
    for row, item in enumerate(rows):
        if "region_center" not in item:
            continue
        center = np.array(item["region_center"])
        ids = np.flatnonzero(np.linalg.norm(points - center, axis=1) <= item["region_radius"])
        cloud = points[ids] - center
        _, _, basis = np.linalg.svd(cloud, full_matrices=False)
        q = cloud @ basis.T
        for col, labels in enumerate([before, after]):
            colors = np.where(labels[ids] == 0, "#1684bc", np.where(labels[ids] > 0, "#cb329a", "#aaaaaa"))
            axes[row, col].scatter(q[:, 0], q[:, 1], s=3, c=colors)
            axes[row, col].set_aspect("equal")
            axes[row, col].set_title(item["root_id"] + (" before" if col == 0 else f" after ({item['transferred_vertex_count']} moved)"))
    fig.savefig(out / "junction_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
