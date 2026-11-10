"""Independent inventory, native-geometry and support audit of repaired bundles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from soyrootbio.measurement_evidence import (
    assess_origin_coverage, build_measurement_evidence, make_evidence_context,
    validate_measurement_evidence,
)
from soyrootbio.surface_patch_audit import audit_discrete_child_patches
from soyrootbio.types import RootPath


METRICS = ("length", "mean_diameter", "volume", "surface_area",
           "base_parent_angle_deg", "tip_angle_parent_deg", "tip_primary_angle_deg",
           "tip_gravity_angle_deg", "tip_start_gravity_angle_deg")


def _audit_evidence(root, row, points, faces, labels, label, context, top, *, terminal_spacing):
    """Rebuild interval evidence from stored native IDs, not its stored masks."""
    assessment = root.get("centerline_assessment", {})
    evidence = assessment.get("measurement_evidence")
    if not evidence:
        assert not row.length_measurement_eligible, (root["root_id"], "measured length without persistent evidence")
        return False
    path = np.asarray(root["polyline"], float)
    start = int(root.get("body_start_index", 0))
    valid, reason = validate_measurement_evidence(evidence, points, labels, label, path,
        triangles=faces, body_start_index=start, context=context)
    assert valid, (root["root_id"], reason)
    assert evidence.get("interval_support_version") == 2, (root["root_id"], "requires interval support v2")
    spacing = float(evidence["spacing"])
    assert np.isfinite(spacing) and spacing > 0
    rebuilt = build_measurement_evidence(points, labels, label, path, triangles=faces,
        selected_vertex_indices=evidence["native_vertex_indices"], spacing=spacing,
        body_start_index=start, identity_resolved=evidence.get("identity_resolved", False),
        attachment_supported=evidence.get("attachment_supported", False),
        tip_supported=evidence.get("biological_tip_verified", False),
        source=evidence.get("source", "final_fit"), context=context)
    for field in ("native_vertex_indices", "support_station_indices", "supported_station_mask",
                  "station_support_counts", "supported_interval_mask"):
        np.testing.assert_array_equal(evidence[field], rebuilt[field], err_msg=f"{root['root_id']}: {field}")
    np.testing.assert_allclose(evidence["support_arc_positions"], rebuilt["support_arc_positions"],
                               rtol=1e-8, atol=1e-10, err_msg=root["root_id"])
    reference = assessment.get("prefit_identity_evidence") or {}
    if reference.get("reference_valid"):
        origin = top if root["root_order"] == 0 else root.get("insertion_point")
        origin = reference.get("origin") if origin is None else origin
        origin_evidence = assess_origin_coverage(path, origin,
            reference_path=reference.get("prior_path"), spacing=spacing,
            attachment_supported=bool(evidence.get("attachment_supported", False))
                and not assessment.get("attachment_context_changed_since_fit", False))
        if row.measurement_evidence_valid == True:
            assert bool(row.origin_coverage_complete) == origin_evidence["origin_coverage_complete"], root["root_id"]
    if evidence.get("biological_tip_verified"):
        from soyrootbio.centerline import _verified_terminal_support
        # Fitting certifies native terminals at the global native spacing
        # d_bar. The evidence spacing can be a smaller root-specific fit
        # spacing and belongs to interval/origin checks, not this certificate.
        assert _verified_terminal_support(points, faces, np.asarray(evidence["native_vertex_indices"], int),
            path[start:], reference, terminal_spacing, context), (root["root_id"], "uncertified anatomical tip")
    if row.length_measurement_eligible:
        assert bool(row.measurement_evidence_valid), root["root_id"]
        mask = np.asarray(rebuilt["supported_interval_mask"], bool)
        if start and evidence.get("attachment_supported") and assessment.get("parent_connector_supported") \
                and not assessment.get("attachment_context_changed_since_fit", False):
            mask[:start] = True
        intervals = np.linalg.norm(np.diff(path, axis=0), axis=1)
        expected = float(intervals[mask & np.isfinite(intervals) & (intervals > 0)].sum())
        assert expected > 0, root["root_id"]
        np.testing.assert_allclose(row.length, expected, rtol=1e-8, atol=1e-8, err_msg=root["root_id"])
    return True


def _native_spacing_in_output_units(metadata):
    spacing = metadata.get("d_bar_mm") if metadata.get("coordinate_unit") == "mm" else None
    if spacing is None:
        spacing = metadata["d_bar_normalized"] * metadata["normalization_scale"]
    assert np.isfinite(spacing) and spacing > 0, "native spacing must be positive in exported coordinate units"
    return float(spacing)


def _audit_patches(roots, mapping, points, faces, labels, mask, metadata):
    """Rerun the native patch audit using exported independent pre-fit anchors."""
    rows = {row["root_id"]: row for row in roots}
    paths = []
    for label in sorted(mapping):
        if label == 0:
            continue
        row = rows[mapping[label]["root_id"]]
        paths.append(RootPath(root_id=row["root_id"], points=np.asarray(row["polyline"], float),
            order=row["root_order"], parent_id=row["parent_id"],
            body_start_index=int(row.get("body_start_index", 0)),
            centerline_assessment=row.get("centerline_assessment", {})))
    assert sorted(mapping) == list(range(len(paths) + 1)), "native root labels must map unambiguously"
    spacing = _native_spacing_in_output_units(metadata)
    audit = audit_discrete_child_patches(points, labels, paths, triangles=faces, d_bar=float(spacing),
        excluded_mask=mask, nonroot_labels=tuple(int(x) for x in np.unique(labels[labels <= -3])))
    stored = metadata["final_surface_patch_audit"]
    signature = lambda report: sorted((p["root_id"], tuple(sorted(p["vertex_indices"])), p["status"])
                                      for p in report["patches"])
    assert signature(audit) == signature(stored), "exported native patch audit differs from current geometry/reference"
    assert audit["status"] == metadata["final_compliance_audit"]["discrete_child_patch_status"]
    assert audit["unresolved_patch_count"] == stored["unresolved_patch_count"]
    identity_signature = lambda report: sorted(
        (row["root_id"], tuple(sorted(row["vertex_indices"])), row["status"])
        for row in report.get("body_anchor_ambiguities", []))
    assert identity_signature(audit) == identity_signature(stored), "native body identity review changed"
    expected_identity_count = len({row["root_id"] for row in audit.get("body_anchor_ambiguities", [])})
    assert metadata["final_compliance_audit"].get("unresolved_native_body_identity_count", 0) == expected_identity_count
    assert stored.get("unresolved_body_anchor_root_count", 0) == expected_identity_count
    assert metadata["final_compliance_audit"].get("native_body_identity_status", "clear") == (
        "unresolved" if expected_identity_count else "clear")
    if audit["unresolved_patch_count"] or expected_identity_count:
        assert metadata["final_compliance_audit"].get("compliant") is not True
    return audit


def audit(bundle: Path, original: Path | None = None) -> dict:
    support = bundle / "support"
    metadata = json.loads((support / "metadata.json").read_text(encoding="utf-8"))
    roots = json.loads((support / "root_hierarchy.json").read_text(encoding="utf-8"))["roots"]
    inventory = pd.read_csv(bundle / "tables/root_inventory.csv")
    measured = pd.read_csv(bundle / "tables/root_traits.csv")
    geometry = np.load(support / "analysis_geometry.npz")
    points, faces, labels = geometry["positions"], geometry["triangles"], geometry["root_labels"]
    mask = np.load(support / "ownership_evidence_masks.npz")["above_base_excluded"]
    assert not np.any(labels[mask] != -1), "collar exclusion changed"
    top = np.asarray(metadata["primary_top_reference_mm"])
    up = -np.asarray(metadata["gravity_vector"])
    assert all(float((np.asarray(r["polyline"])[0] - top) @ up) <= 1e-7
               for r in roots if r["root_order"] > 0), "lateral origin above immutable top"
    assert all(float((np.asarray(r["insertion_point"]) - top) @ up) <= 1e-7
               for r in roots if r["root_order"] > 0 and r.get("insertion_point") is not None)
    assert inventory.root_id.is_unique and measured.root_id.is_unique, "duplicate table identities"
    assert len({r["root_id"] for r in roots}) == len(roots), "duplicate hierarchy identities"
    assert set(inventory.root_id) == {r["root_id"] for r in roots}
    assert set(measured.root_id) == set(inventory.loc[inventory.measurement_eligible.eq(True), "root_id"])
    mapping = {int(row["numeric_label"]): row for row in metadata["root_label_map"] if int(row["numeric_label"]) >= 0}
    label_by_id = {row["root_id"]: key for key, row in mapping.items()}
    edges = np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    # Distinct native indices remain contact even if duplicate coordinates make
    # an edge geometrically zero length. Never filter raw native contact.
    edges = edges[edges[:, 0] != edges[:, 1]]
    high = []
    for label, row in mapping.items():
        if int(row["root_order"]) < 2:
            continue
        count = int(np.count_nonzero(((labels[edges[:, 0]] == label) & (labels[edges[:, 1]] == 0)) |
                                    ((labels[edges[:, 1]] == label) & (labels[edges[:, 0]] == 0))))
        if count:
            high.append({"root_id": row["root_id"], "native_edges": count})
    if high:
        assert metadata["final_compliance_audit"]["higher_order_primary_contact_status"] != "clear"
        assert metadata["final_compliance_audit"].get("compliant") is not True
    else:
        expected_contact_status = "clear" if len(faces) else "unresolved_no_mesh"
        assert metadata["final_compliance_audit"]["higher_order_primary_contact_status"] == expected_contact_status
    stored_contact = metadata["higher_order_primary_contact"].get("final_native_audit", metadata["higher_order_primary_contact"])
    assert sorted((row["root_id"], row["native_edges"]) for row in high) == sorted(
        (row["root_id"], row["contact_edge_count"]) for row in stored_contact["contacts"])
    patch_audit = _audit_patches(roots, mapping, points, faces, labels, mask, metadata)
    context = make_evidence_context(points, faces)
    evidence_counts = {"valid": 0, "unavailable": 0}
    inventory_by_id = inventory.set_index("root_id")
    for review in patch_audit.get("body_anchor_ambiguities", []):
        row = inventory_by_id.loc[review["root_id"]]
        assert "native_body_identity_unresolved" in str(row.qc_flags)
        assert all(row[metric + "_measurement_status"] != "complete" for metric in METRICS), review["root_id"]
    terminal_spacing = _native_spacing_in_output_units(metadata)
    for root in roots:
        valid = _audit_evidence(root, inventory_by_id.loc[root["root_id"]], points, faces, labels,
                                label_by_id[root["root_id"]], context, top, terminal_spacing=terminal_spacing)
        evidence_counts["valid" if valid else "unavailable"] += 1
    counts = {}
    for metric in METRICS:
        column = metric + "_measurement_status"
        statuses = inventory[column]
        assert statuses.isin(["complete", "partial", "unavailable", "excluded"]).all(), column
        eligible = inventory[f"{metric}_measurement_eligible"].eq(True)
        assert (eligible == statuses.isin(["complete", "partial"])).all(), metric
        assert np.isfinite(inventory.loc[eligible, metric]).all(), metric
        assert not np.isfinite(inventory.loc[~eligible, metric]).any(), (metric, "unavailable metric carries a finite value")
        assert inventory[f"{metric}_measurement_reason"].fillna("").astype(str).str.len().gt(0).all(), metric
        if metric in {"length", "mean_diameter", "volume"}:
            assert (inventory.loc[eligible, metric] > 0).all(), metric
        elif metric == "surface_area":
            assert (inventory.loc[eligible, metric] >= 0).all(), metric
        else:
            assert inventory.loc[eligible, metric].between(0, 180).all(), metric
        complete = statuses.eq("complete")
        assert inventory.loc[complete, "measurement_identity_resolved"].eq(True).all(), metric
        if metric != "surface_area":
            assert inventory.loc[eligible, "measurement_evidence_valid"].eq(True).all(), metric
        complete_flag = {"length": "length_estimate_complete", "mean_diameter": "radius_profile_complete",
                         "volume": "volume_estimate_complete", "surface_area": "surface_area_estimate_complete"}.get(metric)
        if complete_flag:
            assert (inventory[complete_flag].eq(True) == complete).all(), metric
        counts[metric] = {state: int((statuses == state).sum()) for state in ["complete", "partial", "unavailable", "excluded"]}
        assert sum(counts[metric].values()) == len(inventory)
        for row in roots:
            if row.get("centerline_assessment", {}).get("identity_qc_passed") is False:
                assert inventory.loc[inventory.root_id == row["root_id"], column].iloc[0] != "complete", (row["root_id"], metric)
    summary = metadata["system_summary"]
    for metric, key in [("length", "root_system_length"), ("volume", "root_system_volume")]:
        contributors = inventory.loc[inventory[f"{metric}_measurement_eligible"].eq(True), metric]
        expected = float(contributors.sum(min_count=1))
        reported = np.nan if summary[key] is None else float(summary[key])
        np.testing.assert_allclose(reported, expected, equal_nan=True)
    architecture = pd.read_csv(bundle / "tables/architecture_summary.csv").set_index("scope")
    for scope, values in architecture.iterrows():
        selected = (inventory if scope == "system" else inventory.loc[inventory.root_order.gt(0)]
                    if scope == "laterals" else inventory.loc[inventory.root_order.eq(int(scope.removeprefix("order_")))])
        active = selected.loc[~selected.get("measurement_excluded", pd.Series(False, index=selected.index)).eq(True)]
        assert int(values.root_inventory_count) == len(active), scope
        for metric in METRICS[:4]:
            for status in ("complete", "partial", "unavailable"):
                assert int(values[f"{metric}_{status}_root_count"]) == int(active[f"{metric}_measurement_status"].eq(status).sum()), (scope, metric, status)
        for metric, field in (("length", "length_sum"), ("volume", "volume_frustum"), ("surface_area", "surface_area_partitioned_sum")):
            values_to_sum = active.loc[active[f"{metric}_measurement_eligible"].eq(True), metric]
            expected = float(values_to_sum.sum(min_count=1)) if len(selected) else 0.
            np.testing.assert_allclose(values[field], expected, equal_nan=True, err_msg=f"{scope}: {field}")
        volume_complete = len(active) > 0 and active.volume_measurement_status.eq("complete").all()
        assert bool(values.volume_frustum_complete) == volume_complete, scope
    result = {"sample": bundle.name, "passed": True, "inventory_count": len(inventory),
              "length_eligible_rows": len(measured), "measurement_status_counts": counts,
              "measurement_evidence": evidence_counts, "native_higher_order_contacts": high,
              "unresolved_patch_count": patch_audit["unresolved_patch_count"],
              "unresolved_body_identity_count": len({row["root_id"] for row in patch_audit.get("body_anchor_ambiguities", [])}),
              "unresolved_body_identity_region_count": len(patch_audit.get("body_anchor_ambiguities", [])),
              "audit_coverage": {"stored_support_hashes": True, "rebuilt_interval_masks_v2": True,
                  "native_mesh_available": bool(len(faces)),
                  "supported_length_in_export_units": True,
                  "terminal_certificate_spacing": "global_native_d_bar_in_output_units",
                  "native_contact_and_patch_checks": "recomputed" if len(faces) else "unavailable_no_native_mesh",
                  "per_metric_status_checks": list(METRICS), "cohort_totals_checked": ["length", "volume", "surface_area"],
                  "anatomical_correctness_certified": False},
              "identity_repair": {key: value for key, value in metadata.get("identity_ownership_repair", {}).items()
                                  if key in {"status", "changed_vertex_count", "accepted_count", "repaired_root_count"}},
              "root_system_length": summary["root_system_length"], "root_system_volume": summary["root_system_volume"],
              "source_tree_sha256": metadata["processing_provenance"]["source_tree_sha256"]}
    if original is not None:
        old = np.load(original / "support/analysis_geometry.npz")
        np.testing.assert_array_equal(points, old["positions"])
        np.testing.assert_array_equal(faces, old["triangles"])
        old_meta = json.loads((original / "support/metadata.json").read_text(encoding="utf-8"))
        np.testing.assert_array_equal(top, old_meta["primary_top_reference_mm"])
        result["native_geometry_unchanged"] = True
        result["changed_ownership_vertices"] = int(np.count_nonzero(labels != old["root_labels"]))
        previous = pd.read_csv(original / "tables/root_traits.csv").set_index("root_id")
        current = inventory.set_index("root_id")
        cases = []
        for rid in previous.index.intersection(current.index):
            old_row, new_row = previous.loc[rid], current.loc[rid]
            if old_row.mean_diameter > max(3., old_row.length):
                cases.append({"root_id": rid, "before": {k: float(old_row[k]) for k in ["length", "mean_diameter", "volume"]},
                              "after": {k: (None if pd.isna(new_row[k]) else float(new_row[k])) for k in ["length", "mean_diameter", "volume"]},
                              "status": {k: new_row[k + "_measurement_status"] for k in ["length", "mean_diameter", "volume"]}})
        result["previous_diameter_outliers"] = cases
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.bundle, args.original)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
