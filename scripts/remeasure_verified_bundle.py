"""Recompute measurements from verified final support without refitting a bundle.

Use a fresh output directory. Native geometry, ownership, immutable top, hierarchy
and every path are retained exactly. Old evidence is validated before its support
intervals are upgraded; an invalid record is never silently refreshed.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from soyrootbio.bundle_layout import bundle_file, read_bundle_mesh
from soyrootbio.export import export_results
from soyrootbio.measurement_evidence import (
    assess_origin_coverage, make_evidence_context, upgrade_measurement_evidence,
)
from soyrootbio.provenance import file_sha256
from soyrootbio.traits import compute_traits
from soyrootbio.types import Normalization, RootPath
from soyrootbio.visualize import save_angle_front_views, save_overview_plot


def remeasure(bundle: Path, output: Path) -> dict:
    bundle, output = bundle.resolve(), output.resolve()
    if output == bundle or bundle in output.parents or output in bundle.parents:
        raise ValueError("Output must be separate from the source bundle")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    metadata_path = bundle_file(bundle, "metadata.json")
    hierarchy_path = bundle_file(bundle, "root_hierarchy.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    if hierarchy.get("coordinate_unit") != "mm":
        raise ValueError("This migration requires calibrated mm bundle coordinates")
    if "primary_top_reference_mm" not in metadata:
        raise ValueError("The immutable selected top must be recorded explicitly")
    mesh = read_bundle_mesh(bundle)
    points, faces, labels = mesh.positions, mesh.triangles, mesh.root_labels
    if np.any(labels <= -3):
        raise ValueError("Nodule-bearing bundles require a nodule-aware remeasurement")
    original_hashes = {str(p.relative_to(bundle)): file_sha256(p)
                       for p in bundle.rglob("*") if p.is_file()}
    rows = copy.deepcopy(hierarchy["roots"])
    mapping = {str(row["root_id"]): int(row["numeric_label"])
               for row in metadata["root_label_map"] if int(row["numeric_label"]) >= 0}
    rows.sort(key=lambda row: mapping[row["root_id"]])
    if [mapping[r["root_id"]] for r in rows] != list(range(len(rows))) or rows[0]["root_id"] != "primary":
        raise ValueError("Expected contiguous numeric labels with primary at zero")
    if not all(row.get("valid", True) for row in rows):
        raise ValueError("Invalid hierarchy entries must be resolved before remeasurement")
    paths = {row["root_id"]: np.asarray(row["polyline"], dtype=float) for row in rows}
    top = np.asarray(metadata["primary_top_reference_mm"], dtype=float)
    context = make_evidence_context(points, faces)
    upgrades = []
    for row in rows:
        assessment = row.setdefault("centerline_assessment", {})
        evidence = assessment.get("measurement_evidence")
        if evidence is None:
            # A fit without persisted support cannot be certified by this stage.
            raise ValueError(f"Missing final support evidence for {row['root_id']}")
        upgraded, reason = upgrade_measurement_evidence(
            evidence, points, labels, mapping[row["root_id"]], paths[row["root_id"]],
            triangles=faces, body_start_index=int(row.get("body_start_index", 0)), context=context,
        )
        if upgraded is None:
            raise ValueError(f"Stale support for {row['root_id']}: {reason}")
        np.testing.assert_array_equal(evidence["native_vertex_indices"], upgraded["native_vertex_indices"])
        reference = assessment.get("prefit_identity_evidence", {})
        insertion = row.get("insertion_point")
        origin = top if row["root_id"] == "primary" else insertion if insertion is not None else reference.get("origin")
        origin_detail = assess_origin_coverage(
            paths[row["root_id"]], origin,
            reference_path=reference.get("prior_path") if reference.get("reference_valid") else None,
            spacing=upgraded["spacing"], attachment_supported=upgraded["attachment_supported"],
        )
        upgraded["origin_coverage_complete"] = origin_detail["origin_coverage_complete"]
        upgraded["origin_coverage_evidence"] = origin_detail
        assessment["measurement_evidence"] = upgraded
        assessment["biological_tip_verified"] = upgraded.get("biological_tip_verified", False)
        assessment["origin_coverage_evidence"] = origin_detail
        assessment["origin_coverage_complete"] = origin_detail["origin_coverage_complete"]
        upgrades.append({"root_id": row["root_id"], "reason": reason,
                         "previous_interval_support_version": evidence.get("interval_support_version", 1),
                         "interval_support_version": upgraded["interval_support_version"],
                         "terminal_direction_support_version": upgraded.get("terminal_direction_support_version"),
                         "previous_biological_tip_verified": evidence.get("biological_tip_verified", False),
                         "biological_tip_verified": upgraded.get("biological_tip_verified", False),
                         "origin_coverage_complete": origin_detail["origin_coverage_complete"]})
    primary = rows[0]
    laterals = [RootPath(
        root_id=row["root_id"], parent_id=row["parent_id"], order=int(row["root_order"]),
        points=paths[row["root_id"]], parent_points=paths[row["parent_id"]],
        insertion_point=None if row.get("insertion_point") is None else np.asarray(row["insertion_point"]),
        insertion_index=row.get("insertion_index"), body_start_index=int(row.get("body_start_index", 0)),
        confidence=float(row.get("confidence", 0)), qc_flags=list(row.get("qc_flags", [])),
        centerline_assessment=row["centerline_assessment"],
    ) for row in rows[1:]]
    noise_path = bundle_file(bundle, "presentation_noise_masks.npz")
    with np.load(noise_path) as archive:
        noise = archive["excluded_full_vertices"].copy()
    gravity = np.asarray(metadata.get("gravity_vector", [0., 0., -1.]), dtype=float)
    normalization = Normalization(np.zeros(3), 1., mm_per_mesh_unit=1.)
    config = metadata.get("config", {})
    traits = compute_traits(
        paths["primary"], laterals, points, labels == 0, labels, normalization,
        lateral_start_count=int(metadata.get("lateral_start_count", 0)),
        full_points=points, triangles=faces, full_root_labels=labels,
        primary_confidence=float(primary.get("confidence", 1.)),
        primary_qc_flags=list(primary.get("qc_flags", [])),
        primary_centerline_assessment=primary["centerline_assessment"],
        tip_vector_window=float(config.get("tip_vector_window_mm", config.get("tip_vector_window_mesh_units", 2.))),
        gravity=gravity, noise_mask=noise, primary_top_reference=top,
    )
    output.mkdir(parents=True, exist_ok=True)
    # Keep acquisition, fitting and ownership evidence as upstream provenance.
    # Generated tables, views, geometry and measurement files are written anew.
    excluded = {"metadata.json", "root_hierarchy.json", "root_system.rsml", "analysis_geometry.npz",
                "presentation.json", "presentation_noise_masks.npz", "presentation_vertex_mapping.npz",
                "independent_audit.json"}
    target_support = output / "support"
    target_support.mkdir(exist_ok=True)
    for path in (bundle / "support").iterdir():
        if path.name in excluded:
            continue
        if path.is_dir():
            shutil.copytree(path, target_support / path.name)
        else:
            shutil.copy2(path, target_support / path.name)
    shutil.copy2(metadata_path, target_support / "source_metadata_measurement_input.json")
    metadata["measurement_recomputation"] = {
        "policy": "validated-final-support-only-v3", "utc": datetime.now(timezone.utc).isoformat(),
        "stage_script": Path(__file__).name, "stage_script_sha256": file_sha256(__file__),
        "upstream_bundle": str(bundle), "upstream_metadata_sha256": file_sha256(metadata_path),
        "upstream_geometry_sha256": file_sha256(bundle_file(bundle, "analysis_geometry.npz")),
        "upstream_hierarchy_sha256": file_sha256(hierarchy_path),
        "upstream_processing_provenance": metadata.get("processing_provenance"),
        "final_centerline_fitting_role": "unchanged upstream fitting report; embedded support is historical",
        "current_measurement_evidence": "support/root_hierarchy.json: roots[].centerline_assessment.measurement_evidence",
        "geometry_ownership_paths_hierarchy_and_top_preserved": True,
        "recomputed_stages": ["measurement_support_intervals", "origin_coverage", "traits", "tables", "angle_figures"],
        "upgrades": upgrades,
    }
    save_overview_plot(output / "overview.png", points, labels == 0, labels,
                       paths["primary"], laterals, coordinate_unit="mm")
    save_angle_front_views(output, points, labels == 0, labels, paths["primary"], laterals,
                          traits, gravity=gravity)
    export_results(output, points, paths["primary"], laterals, labels == 0, labels, traits,
                   normalization, metadata, full_points=points, triangles=faces,
                   full_root_labels=labels, noise_mask=noise,
                   primary_centerline_assessment=primary["centerline_assessment"])
    final = read_bundle_mesh(output)
    for a, b in [(points, final.positions), (faces, final.triangles), (labels, final.root_labels)]:
        np.testing.assert_array_equal(a, b)
    final_rows = json.loads(bundle_file(output, "root_hierarchy.json").read_text(encoding="utf-8"))["roots"]
    final_metadata = json.loads(bundle_file(output, "metadata.json").read_text(encoding="utf-8"))
    np.testing.assert_array_equal(top, final_metadata["primary_top_reference_mm"])
    assert {row["root_id"] for row in rows} == {row["root_id"] for row in final_rows}
    for row in final_rows:
        np.testing.assert_array_equal(paths[row["root_id"]], row["polyline"])
        original_row = next(r for r in rows if r["root_id"] == row["root_id"])
        assert (row["parent_id"], row["root_order"]) == (original_row["parent_id"], original_row["root_order"])
        assert row.get("body_start_index", 0) == original_row.get("body_start_index", 0)
        assert row.get("insertion_index") == original_row.get("insertion_index")
        np.testing.assert_array_equal(row.get("insertion_point"), original_row.get("insertion_point"))
    after = {str(p.relative_to(bundle)): file_sha256(p) for p in bundle.rglob("*") if p.is_file()}
    assert after == original_hashes, "Input bundle changed during measurement migration"
    return {"output": str(output), "roots": len(rows), "source_unchanged": True,
            "geometry_ownership_paths_hierarchy_and_top_preserved": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(remeasure(args.bundle, args.output), indent=2))
