"""Replay a complete edited bundle, with frozen and downstream controls.

Example: .venv/Scripts/python.exe scripts/validate_joint_collar.py --source
outputs/w51683_m4-2_20260415_corrected/corrected_bundle --output outputs/joint_collar_new
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from soyrootbio.collar import analyze_joint_collar, _frame
from soyrootbio.centerline import refit_final_centerlines
from soyrootbio.editor.ply import read_labeled_ply
from soyrootbio.export import export_results
from soyrootbio.pipeline import _selected_base_exclusion_mask, _surface_connectivity_edges
from soyrootbio.surface_patches import correct_surface_patches, _segment_radius_profile
from soyrootbio.traits import compute_traits
from soyrootbio.topology import validate_root_tree
from soyrootbio.types import Normalization, RootPath, TopologyReport
from validate_primary_o1_ownership import BatchSession, component_report, digest, save


def differences(before, after):
    b, a = (x.set_index("root_id").sort_index() for x in (before, after))
    assert b.index.equals(a.index) and b.columns.equals(a.columns)
    rows = []
    for rid in b.index:
        for col in b.columns:
            old, new = b.loc[rid,col], a.loc[rid,col]
            if (pd.isna(old) and pd.isna(new)) or old == new:
                continue
            rows.append(dict(root_id=rid, trait=col, before=None if pd.isna(old) else old,
                             after=None if pd.isna(new) else new))
    return rows


def support_report(points, labels, primary, paths, reference_labels, spacing):
    """Fixed baseline-radius station test, measured for every complete root."""
    rows = []
    for label, path in enumerate([primary]+[r.points for r in paths]):
        rid = "primary" if label == 0 else paths[label-1].root_id
        s, r = _segment_radius_profile(points[reference_labels==label],path,spacing)
        start = 0 if label == 0 else paths[label-1].body_start_index
        arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        sample = np.linspace(arc[start],arc[-1],max(2,int(np.ceil((arc[-1]-arc[start])/(4*spacing)))+1))
        stations = _frame(path,sample)[0]
        owned = points[labels==label]
        nearest = cKDTree(owned).query(stations)[0] if len(owned) else np.full(len(stations),np.inf)
        ratio = nearest/(np.interp(sample,s,r)+2*spacing)
        rows.append(dict(root_id=rid, station_count=len(sample),
            unsupported_stations=int(np.sum(ratio>2)),
            maximum_normalized_support_gap=float(np.max(ratio)) if np.isfinite(ratio).all() else None,
            support_point_count=len(owned)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--replay-editor-log",action="store_true",
                        help="Replay source/.soyrootbio-editor and verify its materialised export first")
    args = parser.parse_args()
    source,out = args.source.resolve(),args.output.resolve()
    if out == source or source in out.parents:
        raise ValueError("output must be outside the immutable source bundle")
    out.mkdir(parents=True,exist_ok=False)
    files = [p for p in source.rglob("*") if p.is_file()]
    hashes = {str(p):digest(p) for p in files}
    meta = json.loads((source/"metadata.json").read_text())
    auto = json.loads(Path(meta.get("automatic_metadata_provenance",source/"metadata.json")).read_text())
    if args.replay_editor_log:
        shutil.copytree(source/".soyrootbio-editor",out/"source_session",
                        ignore=shutil.ignore_patterns("materialised"))
    session = BatchSession(source,session_dir=out/"source_session",load_existing_log=args.replay_editor_log)
    if args.replay_editor_log:
        materialised = source/".soyrootbio-editor/materialised"
        edited = read_labeled_ply(materialised/"edited_segmented_root_structure.ply")
        np.testing.assert_array_equal(edited.root_labels,session.mesh.root_labels)
        np.testing.assert_array_equal(edited.positions,session.mesh.positions)
        np.testing.assert_array_equal(edited.triangles,session.mesh.triangles)
        rows = json.loads((materialised/"edited_root_hierarchy.json").read_text())["roots"]
        assert {r["root_id"] for r in rows}==set(session.roots)
        for row in rows:
            root=session.roots[row["root_id"]]
            np.testing.assert_allclose(root.points,row["polyline"],rtol=0,atol=1e-10)
            assert (root.parent_id,root.order)==(row["parent_id"],row["root_order"])
    norm = Normalization(np.array(meta["normalization_minimum"]),meta["normalization_scale"])
    points = norm.transform_points(session.mesh.positions)
    primary = norm.transform_points(session.roots["primary"].points)
    roots = sorted((r for r in session.roots.values() if r.root_id!="primary"),key=lambda r:r.numeric_label)
    numeric = np.array([0]+[r.numeric_label for r in roots])
    before = session.mesh.root_labels.copy()
    for label,original in enumerate(numeric):
        before[session.mesh.root_labels==original] = label
    paths = [RootPath(root_id=r.root_id,parent_id=r.parent_id,order=r.order,
        points=norm.transform_points(r.points),insertion_index=r.insertion_index,
        body_start_index=r.body_start_index,confidence=r.confidence,qc_flags=list(r.qc_flags)) for r in roots]
    hierarchy = [(r.root_id,r.parent_id,r.order) for r in paths]
    spacing,faces = meta["d_bar_normalized"],session.mesh.triangles
    a = auto["point_assignment"]
    excluded = _selected_base_exclusion_mask(points,
        norm.transform_points(np.array(a["base_point_source_coordinates"])[None])[0],
        np.array(a["base_tipward_direction"]),gravity=np.array(a["gravity_direction"]),
        collar_neighborhood_radius=a["base_collar_neighborhood_radius_normalized"],
        tolerance=a["above_base_tolerance_normalized"])
    boundary_control = before.copy()
    boundary_control[excluded] = -1
    started = time.perf_counter()
    after,stage = analyze_joint_collar(points,before,primary,paths,d_bar=spacing,
        triangles=faces,excluded_mask=excluded)
    print(f"Joint stage: {stage['changed_vertex_count']} changed, {len(stage['participating_roots'])} participants, {stage.get('unresolved_vertex_count',0)} unresolved",flush=True)
    save(out/"collar_qc.json",stage)
    save(out/"source_hashes.json",hashes)
    save(out/"numeric_label_mapping.json",[dict(root_id="primary" if i==0 else paths[i-1].root_id,
        source_label=int(old),output_label=i) for i,old in enumerate(numeric)])
    np.save(out/"labels_before.npy",before)
    np.save(out/"labels_after_collar.npy",after)
    region = np.zeros(len(points),bool)
    region[stage["neighborhood_vertex_indices"]] = True
    assert np.array_equal(before[~region & ~excluded],after[~region & ~excluded])
    assert np.all(after[excluded]==-1)
    # Exact stage re-execution and reverse-order replay use untouched evidence.
    remap = np.r_[0,np.arange(len(paths),0,-1)]
    swapped = before.copy()
    swapped[before>=0] = remap[before[before>=0]]
    reverse,reverse_report = analyze_joint_collar(points,swapped,primary,paths[::-1],
        d_bar=spacing,triangles=faces,excluded_mask=excluded)
    reverse[reverse>=0] = remap[reverse[reverse>=0]]
    np.testing.assert_array_equal(reverse,after)
    assert reverse_report["topology_decisions"]==stage["topology_decisions"]
    replay = before.copy()
    for change in stage["changed_assignments"]:
        indices = change["vertex_indices"]
        assert np.all(before[indices]==change["before_label"])
        replay[indices] = change["after_label"]
    np.testing.assert_array_equal(replay,after)
    # Full source-to-output comparison with fixed geometry, then a controlled
    # pair including the actual downstream patch/refit/trait/export stages.
    def traits(labels,line,children,assessment=None):
        return compute_traits(line,children,points,labels==0,labels,norm,
            gravity=session.gravity,full_points=session.mesh.positions,triangles=faces,
            full_root_labels=labels,mesh_metadata=meta.get("source_geometry"),
            primary_confidence=session.roots["primary"].confidence,
            primary_qc_flags=[] if assessment is None else assessment["primary_qc_flags"],
            primary_centerline_assessment=None if assessment is None else assessment["roots"][0],
            tip_vector_window=session.tip_vector_window_mesh_units)
    frozen_before,frozen_after = traits(before,primary,paths),traits(after,primary,paths)
    frozen_before.to_csv(out/"traits_frozen_before.csv",index=False)
    frozen_after.to_csv(out/"traits_frozen_after.csv",index=False)
    frozen_changes = differences(frozen_before,frozen_after)
    support_traits = {"mean_radius","mean_diameter","median_diameter","minimum_diameter",
        "maximum_diameter","surface_area","volume","point_count"}
    assert all(r["trait"] in support_traits for r in frozen_changes)
    save(out/"trait_changes_frozen.json",frozen_changes)
    support_before = support_report(points,before,primary,paths,before,spacing)
    support_boundary = support_report(points,boundary_control,primary,paths,before,spacing)
    support_after = support_report(points,after,primary,paths,before,spacing)
    save(out/"centerline_support_frozen.json",dict(before=support_before,
        exclusion_only=support_boundary,after=support_after))
    compact_roots = {rid:deepcopy(r) for rid,r in session.roots.items()}
    for label,rid in enumerate(["primary"]+[r.root_id for r in paths]):
        compact_roots[rid].numeric_label=label
    edges,_ = _surface_connectivity_edges(points,faces,spacing)
    counts_before = component_report(before,edges,compact_roots)
    counts_boundary = component_report(boundary_control,edges,compact_roots)
    counts_after = component_report(after,edges,compact_roots)
    outputs = {}
    for name,input_labels,barrier in [("control",boundary_control,excluded),("joint",after,excluded|region)]:
        print(f"Full downstream {name}: patches, centerlines, all traits, export",flush=True)
        children = deepcopy(paths)
        labels,patch_report = correct_surface_patches(points,input_labels,primary,children,
            d_bar=spacing,triangles=faces,excluded_mask=barrier)
        if name=="joint":
            np.testing.assert_array_equal(labels[region],after[region])
        line,fit_report = refit_final_centerlines(points,labels,primary.copy(),children,
            d_bar=spacing,triangles=faces)
        assert [(r.root_id,r.parent_id,r.order) for r in children]==hierarchy
        frame = traits(labels,line,children,fit_report)
        assert set(frame.root_id)==set(compact_roots)
        assert np.array_equal(frame.set_index("root_id").loc[["primary"]+[r.root_id for r in children],"point_count"].to_numpy(),
                              np.array([np.sum(labels==i) for i in range(len(children)+1)]))
        metadata = dict(meta,final_centerline_fitting=fit_report,
            selected_lateral_count=len(children),selected_order_counts=dict(Counter(r.order for r in children)),
            validation_source_topology_report=meta.get("topology_report"),
            topology_report=asdict(TopologyReport(warnings=validate_root_tree(children,primary_path=line))),
            point_assignment=dict(assigned_vertex_count=int(np.sum(labels>=0)),
                uncertain_vertex_count=int(np.sum(labels==-2)),unassigned_vertex_count=int(np.sum(labels==-1)),
                surface_patch_correction=patch_report),
            validation_source=str(source),validation_branch=name)
        if name=="joint":
            metadata["joint_root_collar"]=stage
        bundle = out/name
        export_results(bundle,session.mesh.positions,line,children,labels==0,labels,frame,norm,metadata,
            full_points=session.mesh.positions,triangles=faces,full_root_labels=labels)
        loaded = BatchSession(bundle,session_dir=out/(name+"_reload"),load_existing_log=False)
        mesh = read_labeled_ply(bundle/"segmented_root_structure.ply")
        np.testing.assert_array_equal(mesh.positions,session.mesh.positions)
        np.testing.assert_array_equal(mesh.triangles,faces)
        np.testing.assert_array_equal(mesh.root_labels,labels)
        np.testing.assert_array_equal(loaded.mesh.root_labels,labels)
        assert set(loaded.roots)==set(compact_roots)
        for root in children:
            r = loaded.roots[root.root_id]
            assert (r.parent_id,r.order,r.body_start_index)==(root.parent_id,root.order,root.body_start_index)
            np.testing.assert_allclose(r.points,norm.inverse_points(root.points),rtol=0,atol=1e-12)
        reloaded_traits = pd.read_csv(bundle/"root_traits.csv",keep_default_na=False)
        for col in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[col]):
                actual = pd.to_numeric(reloaded_traits[col].replace("",np.nan))
                np.testing.assert_allclose(frame[col].to_numpy(float),actual.to_numpy(float),
                                           rtol=1e-12,atol=1e-12,equal_nan=True)
            else:
                assert frame[col].fillna("").tolist()==reloaded_traits[col].fillna("").tolist(),col
        save(bundle/"system_summary_validation.json",frame.attrs.get("system_summary",{}))
        outputs[name] = dict(labels=labels,traits=frame,fit=fit_report,
            topology_warnings=metadata["topology_report"]["warnings"],
            components=component_report(labels,edges,compact_roots))
    save(out/"trait_changes_downstream.json",differences(outputs["control"]["traits"],outputs["joint"]["traits"]))
    assert hashes == {str(p):digest(p) for p in files}
    states = lambda values: dict(primary=int(np.sum(values==0)),lateral=int(np.sum(values>0)),
        assigned=int(np.sum(values>=0)),uncertain=int(np.sum(values==-2)),unassigned=int(np.sum(values==-1)))
    report = dict(source=str(source),output=str(out/"joint"),elapsed_seconds=time.perf_counter()-started,
        editor_log_replayed_and_matches_materialised=args.replay_editor_log,
        vertices=len(points),triangles=len(faces),source_unchanged=True,
        root_count=len(compact_roots),order_counts=dict(Counter(r.order for r in compact_roots.values())),
        root_counts_orders_parents_preserved=True,reverse_root_order_replay_matches=True,
        metadata_root_counts_checked=True,
        final_topology_validation_warnings={name:row["topology_warnings"] for name,row in outputs.items()},
        qc_assignment_replay_matches=True,exported_mesh_and_reload_match=True,
        all_derived_trait_columns_checked=list(frozen_before.columns),
        source_states=states(before),collar_states=states(after),
        control_final_states=states(outputs["control"]["labels"]),joint_final_states=states(outputs["joint"]["labels"]),
        collar_changed_vertices=stage["changed_vertex_count"],
        exclusion_only_changed_vertices=int(np.sum(before!=boundary_control)),
        joint_ownership_changed_vertices=int(np.sum(boundary_control!=after)),
        participant_count=len(stage["participating_roots"]),unresolved_vertices=stage.get("unresolved_vertex_count",0),
        components_before=counts_before,components_exclusion_only=counts_boundary,components_after_collar=counts_after,
        components_control_final=outputs["control"]["components"],components_joint_final=outputs["joint"]["components"],
        frozen_support_regressions=[a["root_id"] for b,a in zip(support_before,support_after)
            if a["unsupported_stations"]>b["unsupported_stations"]],
        ownership_only_support_regressions=[a["root_id"] for b,a in zip(support_boundary,support_after)
            if a["unsupported_stations"]>b["unsupported_stations"]],
        control_fitting_statuses=dict(Counter(r["status"] for r in outputs["control"]["fit"]["roots"])),
        joint_fitting_statuses=dict(Counter(r["status"] for r in outputs["joint"]["fit"]["roots"])))
    save(out/"validation.json",report)
    save(out/"system_summary_comparison.json",{name:row["traits"].attrs.get("system_summary",{})
                                              for name,row in outputs.items()})
    preview(points,before,after,stage,out)
    print(json.dumps({k:v for k,v in report.items() if not k.startswith("components") and k!="all_derived_trait_columns_checked"},indent=2))


def preview(points,before,after,stage,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ix=np.array(stage["neighborhood_vertex_indices"],int)
    cloud=points[ix]-points[ix].mean(axis=0)
    _,_,basis=np.linalg.svd(cloud,full_matrices=False)
    xy=cloud@basis.T
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout="constrained")
    palette=plt.get_cmap("tab20")
    for ax,labels,title in zip(axes,[before,after],["Edited collar before","Joint collar ownership"]):
        colors=palette((np.maximum(labels[ix],0)%20)/20)
        colors[labels[ix]==0]=[.1,.4,.8,1]
        colors[labels[ix]<0]=[.65,.65,.65,1]
        ax.scatter(xy[:,0],xy[:,1],s=3,c=colors)
        ax.set_aspect("equal")
        ax.set_title(title)
    fig.savefig(out/"collar_comparison.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    main()
