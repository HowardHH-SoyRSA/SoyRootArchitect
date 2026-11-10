from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soyrootbio.architecture_traits import add_architecture_traits, metric_status_counts
from soyrootbio.measurement_evidence import build_measurement_evidence, make_evidence_context
from soyrootbio.traits import (compute_traits, _radius_profile, _measurement_support, _frustum_measurements,
                               _annotate_measured_length_comparisons)
from soyrootbio.types import Normalization, RootPath


def tube(length=3., radius=.1, rings=41, sides=20, offset=(0., 0., 0.), arc=None, half=False):
    theta = np.linspace(0., np.pi/3 if half else 2*np.pi, sides, endpoint=False)
    z = np.linspace(0., length, rings)
    centers = np.c_[np.zeros(rings), np.zeros(rings), z] if arc is None else np.asarray(arc)
    tangents = np.gradient(centers, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    u = np.cross(tangents, [1., 0, 0]); u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(tangents, u)
    xyz = (centers[:, None]+radius*(np.cos(theta)[None, :, None]*u[:, None]
           +np.sin(theta)[None, :, None]*v[:, None])).reshape(-1, 3)+offset
    faces=[]
    for i in range(rings-1):
        for j in range(sides if not half else sides-1):
            a,b=i*sides+j,i*sides+(j+1)%sides
            faces.extend([[a,b,a+sides], [b,b+sides,a+sides]])
    return xyz, np.asarray(faces), centers+offset


def measure(xyz, faces, path, *, selected=None, identity=True, fitting=None):
    labels=np.zeros(len(xyz), int)
    evidence=build_measurement_evidence(xyz, labels, 0, path, triangles=faces,
        selected_vertex_indices=selected, spacing=.03, identity_resolved=identity,
        attachment_supported=True, tip_supported=True, origin_coverage_complete=True)
    assessment={"status":"fitted", "fit_qc_passed":True, "identity_qc_passed":identity,
                "measurement_evidence":evidence, **(fitting or {})}
    return compute_traits(path, [], xyz, labels==0, labels, Normalization(np.zeros(3),1.,1.),
        full_points=xyz,full_root_labels=labels,triangles=faces,primary_centerline_assessment=assessment),assessment


@pytest.mark.parametrize("length,offset", [(1.5,18.),(.327,6.),(.604,2.2)])
def test_disconnected_remote_owned_body_cannot_become_fragment_radius(length,offset):
    local,faces,path=tube(length=length)
    remote,rf,_=tube(length=length,offset=(offset,0,0))
    xyz=np.vstack([local,remote]); faces=np.vstack([faces,rf+len(local)])
    before=xyz.copy()
    result,_=measure(xyz,faces,path,selected=np.arange(len(local)),identity=False)
    row=result.iloc[0]
    assert row.length == pytest.approx(length)
    assert row.mean_diameter == pytest.approx(.2,rel=.01)
    assert row.volume == pytest.approx(np.pi*.1**2*length,rel=.01)
    assert row.length_measurement_status == row.volume_measurement_status == "partial"
    assert not row.length_estimate_complete
    assert np.isnan(row.tip_gravity_angle_deg)
    assert row.surface_area > 2*np.pi*.1*length  # Both retained native bodies remain inventoried.
    np.testing.assert_array_equal(xyz,before)


def test_one_sided_radius_unavailable_does_not_remove_supported_length_or_surface():
    xyz,faces,path=tube(half=True)
    result,_=measure(xyz,faces,path)
    row=result.iloc[0]
    assert row.length == pytest.approx(3.)
    assert row.length_measurement_status == "complete"
    assert np.isnan(row.mean_diameter) and np.isnan(row.volume)
    assert row.mean_diameter_measurement_status == row.volume_measurement_status == "unavailable"
    assert row.surface_area > 0
    summary=result.attrs["system_summary"]
    assert summary["length_complete_root_count"] == 1
    assert summary["volume_unavailable_root_count"] == 1


def test_changed_parent_context_invalidates_attachment_only():
    xyz,faces,path=tube()
    result,_=measure(xyz,faces,path,fitting={"attachment_context_changed_since_fit":True})
    row=result.iloc[0]
    assert row.measurement_evidence_valid
    assert row.measurement_evidence_reason == "attachment_context_changed_since_fit"
    assert row.length == pytest.approx(3.)
    assert row.mean_diameter == pytest.approx(.2,rel=.01)
    assert row.volume == pytest.approx(3*np.pi*.1**2,rel=.01)
    assert row.length_measurement_status == row.volume_measurement_status == "partial"
    assert not row.length_estimate_complete
    assert np.isnan(row.base_parent_angle_deg)
    assert np.isnan(row.tip_primary_angle_deg)
    assert np.isnan(row.tip_start_gravity_angle_deg)
    assert row.tip_gravity_angle_deg == pytest.approx(180.)
    assert row.tip_gravity_angle_deg_measurement_status == "complete"


def test_native_body_identity_review_keeps_measurements_partial_without_erasing_support():
    xyz, faces, path = tube()
    baseline, assessment = measure(xyz, faces, path)
    labels = np.zeros(len(xyz), int)
    reviewed = compute_traits(path, [], xyz, labels == 0, labels,
        Normalization(np.zeros(3), 1., 1.), full_points=xyz, full_root_labels=labels,
        triangles=faces, primary_centerline_assessment=assessment,
        primary_qc_flags=["native_body_identity_unresolved"])
    row = reviewed.iloc[0]
    for metric in ("length", "mean_diameter", "volume", "surface_area"):
        assert row[metric] == pytest.approx(baseline.iloc[0][metric])
        assert row[f"{metric}_measurement_status"] == "partial"
    assert not row.measurement_identity_resolved
    assert row.tip_gravity_angle_deg_measurement_status == "unavailable"


def test_primary_origin_retreat_only_downgrades_completeness_not_measured_extent():
    xyz,faces,path=tube(offset=(0,0,.1))
    prior=path.copy();prior[0]=[0.,0.,0.]
    result,_=measure(xyz,faces,path,fitting={"prefit_identity_evidence":{
        "reference_valid":True,"prior_path":prior.tolist(),"origin":[0.,0.,0.]}})
    row=result.iloc[0]
    assert row.length == pytest.approx(3.)
    assert row.mean_diameter == pytest.approx(.2,rel=.01)
    assert row.volume == pytest.approx(3*np.pi*.1**2,rel=.01)
    assert not row.origin_coverage_complete
    assert row.origin_axial_loss == pytest.approx(.1)
    assert row.length_measurement_status == row.volume_measurement_status == "partial"
    assert np.isnan(row.tip_start_gravity_angle_deg)
    assert row.tip_gravity_angle_deg == pytest.approx(180.)


@pytest.mark.parametrize("change", ["path","ownership","geometry"])
def test_stale_evidence_never_falls_back_to_all_owned_vertices(change):
    xyz,faces,path=tube()
    _,assessment=measure(xyz,faces,path)
    labels=np.zeros(len(xyz),int)
    if change=="path":path=path+[.01,0,0]
    elif change=="ownership":labels[-1]=-1
    else:xyz=xyz.copy();xyz[-1]+=[.01,0,0]
    result=compute_traits(path,[],xyz,labels==0,np.zeros(len(labels),int),Normalization(np.zeros(3),1.,1.),
        full_points=xyz,full_root_labels=labels,triangles=faces,primary_centerline_assessment=assessment)
    row=result.iloc[0]
    assert not row.measurement_evidence_valid
    assert np.isnan(row.length) and np.isnan(row.mean_diameter) and np.isnan(row.volume)
    assert row.surface_area > 0
    assert row.surface_area_measurement_eligible


def test_same_component_adjacent_coil_keeps_local_radius():
    a=np.linspace(0.,4*np.pi,321)
    centers=np.c_[np.cos(a),np.sin(a),a*.04]
    xyz,faces,path=tube(radius=.035,rings=len(a),arc=centers)
    result,_=measure(xyz,faces,path)
    assert result.iloc[0].mean_diameter == pytest.approx(.07,rel=.035)
    assert result.iloc[0].volume > 0


def test_native_gap_does_not_create_frustum_between_valid_endpoint_sections():
    first,ff,p1=tube(length=1.)
    second,sf,p2=tube(length=1.,offset=(0,0,4.))
    xyz=np.vstack([first,second]);faces=np.vstack([ff,sf+len(first)])
    path=np.vstack([p1,p2])
    result,_=measure(xyz,faces,path,identity=False)
    row=result.iloc[0]
    assert row.length == pytest.approx(2.)
    assert row.volume == pytest.approx(2*np.pi*.1**2,rel=.01)
    assert np.isnan(row.tortuosity)
    assert not row.volume_estimate_complete


def test_metric_inventory_partition_and_contributions_are_independent():
    rows=[]
    for rid,length,volume in [("primary",3.,np.nan),("zero",0.,1.),("missing",np.nan,2.)]:
        rows.append(dict(root_id=rid,root_order=0 if rid=="primary" else 1,length=length,
            volume=volume,mean_diameter=.2,surface_area=1.,length_estimate_complete=rid=="primary",
            volume_estimate_complete=False,length_measurement_status="complete" if rid=="primary" else "unavailable",
            volume_measurement_status="unavailable" if rid=="primary" else "partial",
            mean_diameter_measurement_status="partial",surface_area_measurement_status="partial"))
    frame=pd.DataFrame(rows)
    path=np.array([[0.,0,0],[0,0,3.]])
    laterals=[RootPath(rid,path.copy(),order=1,parent_id="primary") for rid in ["zero","missing"]]
    add_architecture_traits(frame,path,laterals,np.zeros((3,3)),np.arange(3),None,path[0],np.array([0.,0,-1]),"mm")
    summary=frame.attrs["architecture_summary"][0]
    assert summary["length_sum"] == 3
    assert summary["volume_frustum"] == 3
    assert summary["surface_area_partitioned_sum"] == 3
    assert summary["root_inventory_count"] == 3
    for metric in ["length","volume","mean_diameter","surface_area"]:
        counts=metric_status_counts(frame,metric)
        assert counts["complete"]+counts["partial"]+counts["unavailable"] == 3
    assert metric_status_counts(frame,"length")["unavailable"] == 2


def test_measured_overlong_child_only_adds_partial_extent_review_evidence():
    frame=pd.DataFrame([
        dict(root_id="primary",parent_id="",root_order=0,length=10.,length_measurement_status="complete",qc_flags=""),
        dict(root_id="parent",parent_id="primary",root_order=1,length=1.,length_measurement_status="partial",qc_flags="attachment_unresolved"),
        dict(root_id="child",parent_id="parent",root_order=2,length=3.,length_measurement_status="complete",qc_flags="existing_review"),
        dict(root_id="descendant",parent_id="child",root_order=3,length=.5,length_measurement_status="partial",qc_flags=""),
        dict(root_id="missing",parent_id="parent",root_order=2,length=np.nan,length_measurement_status="unavailable",qc_flags=""),
    ])
    preserved=frame[["root_id","parent_id","root_order","length","length_measurement_status"]].copy(deep=True)
    _annotate_measured_length_comparisons(frame)
    child=frame.set_index("root_id").loc["child"]
    assert child.measured_child_parent_length_ratio == 3.
    assert child.measured_child_parent_length_excess == 2.
    assert child.measured_parent_length_measurement_status == "partial"
    assert child.measured_child_parent_length_comparison_basis == "supported_partial_extents"
    assert child.qc_flags == "existing_review;measured_child_longer_than_parent"
    assert not frame.set_index("root_id").loc["descendant"].measured_child_parent_length_review
    assert np.isnan(frame.set_index("root_id").loc["missing"].measured_child_parent_length_ratio)
    pd.testing.assert_frame_equal(frame[preserved.columns],preserved)
    assert len(frame.attrs["measured_child_parent_length_reviews"]) == 1
    # Recomputed values remove this pass's obsolete flag, not earlier QC.
    frame.loc[frame.root_id=="child","length"] = .8
    _annotate_measured_length_comparisons(frame)
    assert frame.set_index("root_id").loc["child"].qc_flags == "existing_review"


@pytest.mark.parametrize("sample,root_id", [
    ("SN14_6-2_20260405","root-o1-058"),
    ("Kaixinlv_3-2_20260525","root-o2-247"),
    ("W82_9cm_water_1-2_20260522","root-o3-004"),
])
def test_optional_real_bundle_disconnected_radius_regression(sample,root_id):
    location=os.environ.get("SOYROOTBIO_AUDIT_BUNDLE_ROOT")
    if not location:
        pytest.skip("set SOYROOTBIO_AUDIT_BUNDLE_ROOT for local native-mesh regression replay")
    directory=Path(location)/sample
    geometry=np.load(directory/"support/analysis_geometry.npz")
    points,labels,faces=geometry["positions"],geometry["root_labels"],geometry["triangles"]
    roots=json.loads((directory/"support/root_hierarchy.json").read_text())["roots"]
    root=next(r for r in roots if r["root_id"]==root_id)
    fitting=root["centerline_assessment"]
    path=np.asarray(root["polyline"],float);start=root["body_start_index"]
    assessment={}
    support,evidence=_measurement_support(points,labels,fitting["numeric_label"],path,fitting,faces,
        make_evidence_context(points,faces),assessment,body_start_index=start)
    # Exact selected-component consumption, not a numerical cap on the result.
    assert len(support)==fitting["assigned_point_count"]-fitting["excluded_fragment_points"]
    assert len(support)<np.count_nonzero(labels==fitting["numeric_label"])
    radius=_radius_profile(support,path[start:],Normalization(np.zeros(3),1.,1.),
                           assessment=assessment,support_evidence=evidence)
    if np.isfinite(radius).any():
        assert np.nanmax(radius)<=np.linalg.norm(np.ptp(support,axis=0))
    area,volume=_frustum_measurements(path[start:],radius,assessment.get("radius_interval_mask",[]))
    assert not evidence["identity_resolved"]
    assert not evidence["biological_tip_verified"]
    # Sparse one-sided fragments are allowed to remain unavailable.
    assert np.isnan(volume) or volume < np.pi*np.linalg.norm(np.ptp(support,axis=0))**2*np.linalg.norm(np.diff(path[start:],axis=0),axis=1).sum()
