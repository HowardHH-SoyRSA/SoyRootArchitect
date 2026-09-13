"""Replay final fitting on a saved bundle without changing its assignments."""
from pathlib import Path
import argparse
import json
import time
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from soyrootbio.centerline import refit_final_centerlines
import soyrootbio.centerline as fitting_module
from soyrootbio.editor.ply import read_labeled_ply
from soyrootbio.export import export_results
from soyrootbio.traits import compute_traits
from soyrootbio.types import Normalization, RootPath
from soyrootbio.geometry import path_length, resample_polyline, tangent_vectors
from soyrootbio.primary import _plane_basis, _robust_cross_section_center


def section_metrics(points, path, spacing):
    """Independent hull/offset check with the same sampling for both outputs."""
    line = resample_polyline(path, max(3 * spacing, path_length(path) / 500))
    tree = cKDTree(points)
    outside = tested = 0
    offsets_out = []
    for i, tangent in enumerate(tangent_vectors(line)):
        if i < 2 or i >= len(line) - 2:
            continue
        nearest = np.atleast_1d(tree.query(line[i], k=min(16, len(points)))[0])
        radius = max(6 * spacing, 2.5 * float(np.median(nearest)))
        offsets = points[tree.query_ball_point(line[i], radius)] - line[i]
        slab = offsets[np.abs(offsets @ tangent) <= 2 * spacing]
        if len(slab) < 8:
            continue
        radial = slab @ _plane_basis(tangent).T
        try:
            hull = ConvexHull(radial)
        except QhullError:
            continue
        outside += int(np.max(hull.equations[:, -1]) > spacing * .1)
        tested += 1
        center = _robust_cross_section_center(radial)
        radius = max(spacing, float(np.median(np.linalg.norm(radial - center, axis=1))))
        offsets_out.append(float(np.linalg.norm(center) / radius))
    return {"tested_sections": tested, "outside_transverse_hull": outside, "outside_fraction": outside / max(1, tested),
            "median_offset_over_radius": float(np.median(offsets_out)) if offsets_out else None,
            "p90_offset_over_radius": float(np.quantile(offsets_out, .9)) if offsets_out else None,
            "length": path_length(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("assessment output must be separate from source")
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.source / "metadata.json").read_text())
    hierarchy = json.loads((args.source / "root_hierarchy.json").read_text())["roots"]
    mesh = read_labeled_ply(args.source / "segmented_root_structure.ply")
    norm = Normalization(np.array(meta["normalization_minimum"]), meta["normalization_scale"])
    points = norm.transform_points(mesh.positions)
    primary = norm.transform_points(np.array(next(r for r in hierarchy if r["root_id"] == "primary")["polyline"]))
    roots = []
    labels = mesh.root_labels.copy()
    label_map = {r["root_id"]: r.get("numeric_label") for r in hierarchy}
    if any(v is None for k, v in label_map.items() if k != "primary"):
        label_map.update({r["root_id"]: r["numeric_label"] for r in meta["root_label_map"]})
    for row in hierarchy:
        if row["root_id"] == "primary":
            continue
        root = RootPath(root_id=row["root_id"], parent_id=row["parent_id"], order=row["root_order"],
                        points=norm.transform_points(np.array(row["polyline"])),
                        insertion_point=None if row.get("insertion_point") is None else norm.transform_points(np.array([row["insertion_point"]]))[0])
        roots.append(root)
        labels[mesh.root_labels == label_map[root.root_id]] = len(roots)
    before = {"primary": primary.copy(), **{r.root_id: r.points.copy() for r in roots}}
    labels_before = labels.copy()
    topology_before = [(r.root_id, r.parent_id, r.order) for r in roots]
    started = time.perf_counter()
    fitted, report = refit_final_centerlines(points, labels, primary, roots, d_bar=meta["d_bar_normalized"], triangles=mesh.triangles)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["implementation_sha256"] = hashlib.sha256(Path(fitting_module.__file__).read_bytes()).hexdigest()
    np.testing.assert_array_equal(labels, labels_before)
    assert topology_before == [(r.root_id, r.parent_id, r.order) for r in roots]
    report["ownership_and_hierarchy_preserved"] = True
    report["total_length_before"] = sum(r["length_before"] for r in report["roots"])
    report["total_length_after"] = sum(r["length_after"] for r in report["roots"])
    report["shortened_over_10_percent"] = [r["root_id"] for r in report["roots"] if r["length_after"] < .9 * r["length_before"]]
    after = {"primary": fitted, **{r.root_id: r.points for r in roots}}
    metrics = []
    for detail in report["roots"]:
        rid = detail["root_id"]
        cloud = points[labels == detail["numeric_label"]]
        if len(cloud) < 8:
            continue
        row = {"root_id": rid, "before": section_metrics(cloud, before[rid], meta["d_bar_normalized"]),
               "after": section_metrics(cloud, after[rid][detail["body_start_index"]:], meta["d_bar_normalized"])}
        metrics.append(row)
    report["comparison"] = metrics
    for key in ("before", "after"):
        tested = sum(r[key]["tested_sections"] for r in metrics)
        outside = sum(r[key]["outside_transverse_hull"] for r in metrics)
        report[key] = {"tested_sections": tested, "outside_transverse_hull": outside, "outside_fraction": outside / max(1, tested)}
    (args.output / "assessment.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    traits = compute_traits(fitted, roots, points, labels == 0, labels, norm,
                            full_points=mesh.positions, full_root_labels=labels, triangles=mesh.triangles,
                            primary_qc_flags=report["primary_qc_flags"], primary_centerline_assessment=report["roots"][0])
    export_results(args.output / "bundle", mesh.positions, fitted, roots, labels == 0, labels, traits, norm,
                   {**meta, "final_centerline_fitting": report, "assessment_source": str(args.source)},
                   full_points=mesh.positions, triangles=mesh.triangles, full_root_labels=labels)
    exported = read_labeled_ply(args.output / "bundle/segmented_root_structure.ply")
    np.testing.assert_array_equal(exported.positions, mesh.positions)
    np.testing.assert_array_equal(exported.root_labels, labels)
    np.testing.assert_array_equal(exported.triangles, mesh.triangles)
    report["export_geometry_and_labels_verified"] = True
    (args.output / "assessment.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(args.output / "curves.npz", **after)
    interesting = sorted(metrics, key=lambda r: r["before"]["outside_fraction"], reverse=True)[:8]
    fig, axes = plt.subplots(4, 2, figsize=(14, 18), layout="constrained")
    for ax, row in zip(axes.flat, interesting):
        rid = row["root_id"]
        detail = next(d for d in report["roots"] if d["root_id"] == rid)
        cloud = points[labels == detail["numeric_label"]]
        origin = cloud.mean(axis=0)
        _, _, basis = np.linalg.svd(cloud - origin, full_matrices=False)
        xy = (cloud - origin) @ basis[:2].T
        ax.scatter(*xy.T, s=1, color="silver", alpha=.4)
        for curves, color, name in ((before, "#d95f02", "Before"), (after, "#1b66ad", "Final support fit")):
            xy = (curves[rid] - origin) @ basis[:2].T
            ax.plot(*xy.T, color=color, lw=1, label=name)
        ax.set_title(rid + " | " + detail["status"])
        ax.set_aspect("equal")
        ax.legend()
    fig.savefig(args.output / "comparison.png", dpi=130)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ("elapsed_seconds", "before", "after")}))


if __name__ == "__main__":
    main()
