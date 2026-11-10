import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from soyrootbio.export import write_measurement_tables, write_rsml


def test_length_ineligible_identity_keeps_independent_surface_measurement(tmp_path):
    frame = pd.DataFrame([
        {"root_id": "primary", "parent_id": "", "root_order": 0, "length": 10.,
         "measurement_eligible": True, "length_measurement_status": "complete",
         "surface_area": 5., "surface_area_measurement_status": "complete"},
        {"root_id": "body", "parent_id": "primary", "root_order": 1, "length": np.nan,
         "measurement_eligible": False, "surface_area": 3., "surface_area_measurement_status": "partial",
         "surface_area_measurement_eligible": True, "length_measurement_status": "unavailable"},
    ])
    frame.attrs["system_summary"] = {"root_count_total": 1}
    labels = pd.DataFrame([{"numeric_label": 0, "root_id": "primary"}, {"numeric_label": 1, "root_id": "body"}])
    write_measurement_tables(tmp_path, frame, labels)
    inventory = pd.read_csv(tmp_path / "tables/root_inventory.csv").set_index("root_id")
    measured = pd.read_csv(tmp_path / "tables/root_traits.csv")
    assert measured.root_id.tolist() == ["primary"]
    assert inventory.loc["body", "surface_area"] == 3.
    assert inventory.loc["body", "surface_area_measurement_status"] == "partial"
    assert inventory.loc["body", "length_measurement_status"] == "unavailable"


def test_rsml_preserves_metric_quality_separate_from_trace_confidence(tmp_path):
    row = {"root_id": "primary", "length": 1., "confidence": .99, "tracing_confidence": .99,
           "measurement_identity_resolved": False, "measurement_evidence_valid": True,
           "length_measurement_status": "partial", "length_measurement_eligible": True,
           "length_measurement_reason": "disconnected_identity", "length_estimate_complete": False}
    path = write_rsml(tmp_path / "roots.rsml", np.array([[0., 0, 1], [0, 0, 0]]), [],
                      pd.DataFrame([row]), {"coordinate_unit": "mm"})
    properties = ET.parse(path).find(".//root/properties")
    assert properties.find("length_measurement_status").get("value") == "partial"
    assert properties.find("measurement_identity_resolved").get("value") == "false"
    assert properties.find("tracing_confidence").get("value") == "0.99"


def test_rsml_preserves_partial_surface_and_unavailable_angle_values(tmp_path):
    row = {"root_id": "primary", "surface_area": 2.5, "surface_area_measurement_status": "partial",
           "surface_area_measurement_eligible": True, "surface_area_measurement_reason": "native_partial_body",
           "surface_area_estimate_complete": False, "tip_gravity_angle_deg": np.nan,
           "tip_gravity_angle_deg_measurement_status": "unavailable",
           "tip_gravity_angle_deg_measurement_eligible": False,
           "tip_gravity_angle_deg_measurement_reason": "tip_unverified",
           "origin_coverage_complete": False, "origin_coverage_reason": "fitted_body_omits_origin"}
    path = write_rsml(tmp_path / "roots.rsml", np.array([[0., 0, 1], [0, 0, 0]]), [],
                      pd.DataFrame([row]), {"coordinate_unit": "mm"})
    tree = ET.parse(path)
    properties = tree.find(".//root/properties")
    assert float(properties.find("surface_area").get("value")) == 2.5
    assert properties.find("surface_area_estimate_complete").get("value") == "false"
    assert properties.find("tip_gravity_angle_deg").get("available") == "false"
    assert properties.find("origin_coverage_reason").get("value") == "fitted_body_omits_origin"
    definitions = {p.findtext("label"): p.findtext("unit") for p in tree.findall(".//property-definition")}
    assert definitions["surface_area"] == "mm^2"
    assert definitions["tip_gravity_angle_deg"] == "degree"


@pytest.mark.parametrize("missing", [None, pd.NA])
def test_rsml_serializable_missing_angle_remains_unavailable(tmp_path, missing):
    row = {"root_id": "primary", "base_parent_angle_deg": missing}
    path = write_rsml(tmp_path / "roots.rsml", np.array([[0., 0, 1], [0, 0, 0]]), [],
                      pd.DataFrame([row], dtype=object), {"coordinate_unit": "mm"})
    angle = ET.parse(path).find(".//root/properties/base_parent_angle_deg")
    assert angle.attrib == {"available": "false"}
