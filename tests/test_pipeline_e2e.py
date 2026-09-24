from pathlib import Path

import json

import numpy as np
import pandas as pd
import pytest

import soyrootbio.pipeline as pipeline_module
from soyrootbio.pipeline import PipelineConfig, run_pipeline
from soyrootbio.primary_guidance import read_primary_guidance
from soyrootbio.synthetic import write_synthetic_dataset


def test_synthetic_pipeline_exports_non_empty_outputs(tmp_path: Path):
    points_path, endpoint_path = write_synthetic_dataset(
        tmp_path / "synthetic_root.csv",
        primary_points=150,
        lateral_points=70,
        lateral_count=3,
        noise=0.002,
        seed=11,
    )
    output_dir = tmp_path / "outputs"
    result = run_pipeline(
        PipelineConfig(
            input_path=points_path,
            output_dir=output_dir,
            endpoint_file=endpoint_path,
            sample_points=1000,
            lateral_max_paths=6,
            max_root_order=2,
        )
    )

    assert result.point_count > 0
    assert result.primary_path.shape[0] > 2
    assert result.lateral_start_count >= 1

    primary = pd.read_csv(output_dir / "primary_skeleton.csv")
    traits = pd.read_csv(output_dir / "root_traits.csv")
    lateral_skeletons = pd.read_csv(output_dir / "lateral_skeletons.csv")

    assert not primary.empty
    assert not traits.empty
    assert float(traits.loc[traits["root_id"] == "primary", "length"].iloc[0]) > 0
    assert int(traits.loc[traits["root_id"] == "primary", "lateral_start_count"].iloc[0]) >= 1
    assert "tip_angle_parent_deg" in traits.columns
    assert "tip_angle_z_deg" in traits.columns
    assert (output_dir / "metadata.json").exists()
    collar = json.loads((output_dir / "collar_qc.json").read_text())
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert collar == metadata["joint_root_collar"]
    assert collar["policy"] == "joint-root-collar-v1"
    assert metadata["higher_order_primary_contact"]["status"] == "unresolved_no_mesh"
    guidance = read_primary_guidance(output_dir / "primary_guidance.json", expected_input=points_path)
    start, end = pipeline_module.read_endpoint_file(endpoint_path)
    np.testing.assert_array_equal(guidance.start, start)
    np.testing.assert_array_equal(guidance.end, end)
    assert guidance.guides.shape == (0, 3)
    assert (output_dir / "overview.png").exists()
    assert not (output_dir / "tip_angles_front_view_600dpi.png").exists()
    assert (output_dir / "tip_gravity_front_view_600dpi.png").exists()
    assert (output_dir / "tip_start_gravity_front_view_600dpi.png").exists()
    assert (output_dir / "tip_primary_front_view_600dpi.png").exists()
    assert (output_dir / "skeleton_original_overlay.ply").exists()
    assert (output_dir / "segmented_root_structure.ply").exists()
    assert (output_dir / "root_system.rsml").exists()
    assert (output_dir / "root_hierarchy.json").exists()
    assert (output_dir / "traits.xlsx").exists()
    assert ["root_id", "parent_id", "root_order", "node_id", "x", "y", "z"] == list(lateral_skeletons.columns[:7])
    assert {"confidence", "qc_flags"}.issubset(lateral_skeletons.columns)
    assert {"tortuosity", "mean_diameter", "tip_gravity_angle_deg", "tip_start_gravity_angle_deg", "tip_primary_angle_deg"}.issubset(traits.columns)
    assert set(traits["length_unit"]) == {"mesh_unit"}
    assert not any("mm" in column.lower() for column in traits.columns)
    length_by_id = traits.set_index("root_id")["length"].to_dict()
    for row in traits.itertuples(index=False):
        if pd.isna(row.parent_id) or not str(row.parent_id):
            continue
        assert float(row.length) <= float(length_by_id[str(row.parent_id)]) + 1e-9
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["primary_guidance_file"] == "primary_guidance.json"
    assert "final_centerline_fitting" in metadata["stage_timings_seconds"]
    fitting = metadata["final_centerline_fitting"]
    assert len(fitting["roots"]) == len(traits)
    assert fitting["roots"][0]["assigned_point_count"] == int(np.sum(result.full_root_labels == 0))
    assert fitting["roots"][0]["length_after"] * result.normalization.scale == pytest.approx(float(traits.iloc[0]["length"]))
    assert fitting["primary_method"] == (
        "prior-guided-area-angular-sections-curvature-spline-v2"
    )
    assert fitting["primary_fit_qc_passed"] is fitting["roots"][0][
        "fit_qc_passed"
    ]
    assert fitting["primary_fit_applied"] is fitting["roots"][0]["fit_applied"]
    assert fitting["roots"][0]["traits_geometry_source"] in {
        "accepted_fit",
        "retained_prior",
        "unmeasurable_support",
    }
    assert "centerline_region" in lateral_skeletons.columns
    assert metadata["lateral_tracing_policy"][
        "child_length_may_not_exceed_parent"
    ] is True
    tracing_policy = metadata["lateral_tracing_policy"]
    assert tracing_policy[
        "main_tracer_step_score_turn_alignment_weight"
    ] == pytest.approx(0.57)
    assert tracing_policy[
        "main_tracer_step_score_local_density_weight"
    ] == pytest.approx(0.20)
    assert tracing_policy[
        "main_tracer_step_score_distance_weight"
    ] == pytest.approx(0.15)
    assert tracing_policy[
        "main_tracer_step_score_radius_continuity_weight"
    ] == pytest.approx(0.08)
    assert tracing_policy["main_tracer_old_direction_weight"] == pytest.approx(
        0.75
    )
    assert tracing_policy["main_tracer_new_direction_weight"] == pytest.approx(
        0.25
    )
    assert metadata["topology_report"]["overlong_children_removed"] >= 0
    assert metadata["topology_report"]["overlong_descendants_removed"] >= 0
    assert metadata["topology_report"]["origins_above_primary_top_removed"] >= 0
    assert metadata["topology_report"][
        "descendants_of_above_top_roots_removed"
    ] >= 0
    assert len(metadata["topology_report"]["primary_top_point_normalized"]) == 3
    assert metadata["final_centerline_fitting"][
        "origins_above_primary_top"
    ] == 0
    origin_constraint = metadata["lateral_origin_constraint"]
    assert origin_constraint["policy"] == (
        "lateral-origin-at-or-below-primary-top-v1"
    )
    assert origin_constraint["primary_top_reference_policy"] == (
        "immutable_preprocessing_selection"
    )
    assert len(origin_constraint["primary_top_point_normalized"]) == 3
    assert len(origin_constraint["gravity_direction"]) == 3
    assert origin_constraint["rejected_start_count"] >= 0
    assert origin_constraint["rejected_refined_path_count"] >= 0
    np.testing.assert_allclose(
        origin_constraint["primary_top_point_normalized"],
        metadata["topology_report"]["primary_top_point_normalized"],
    )
    assert metadata["topology_report"]["primary_top_reference_policy"] == (
        "immutable_preprocessing_selection"
    )
    np.testing.assert_allclose(
        origin_constraint["primary_top_point_normalized"],
        metadata["final_centerline_fitting"]["primary_top_point_normalized"],
    )
    assert metadata["final_centerline_fitting"][
        "primary_top_reference_policy"
    ] == "immutable_preprocessing_selection"
    assert len(
        metadata["final_centerline_fitting"][
            "fitted_primary_top_point_normalized"
        ]
    ) == 3


def test_points_above_selected_base_remain_unassigned_and_are_explained(tmp_path: Path):
    points_path, endpoint_path = write_synthetic_dataset(
        tmp_path / "synthetic_root.csv",
        primary_points=120,
        lateral_points=50,
        lateral_count=2,
        noise=0.001,
        seed=23,
    )
    points = pd.read_csv(points_path)
    endpoints = pd.read_csv(endpoint_path).set_index("name")
    base = endpoints.loc["start", ["x", "y", "z"]].to_numpy(dtype=float)
    above_base = base + np.array(
        [
            [0.0000, 0.0000, 0.0010],
            [0.0005, 0.0000, 0.0020],
            [-0.0005, 0.0005, 0.0030],
            [0.0000, -0.0005, 0.0040],
        ]
    )
    appended_indices = np.arange(len(points), len(points) + len(above_base))
    pd.concat(
        [points, pd.DataFrame(above_base, columns=["x", "y", "z"])],
        ignore_index=True,
    ).to_csv(points_path, index=False)

    output_dir = tmp_path / "outputs-above-base"
    result = run_pipeline(
        PipelineConfig(
            input_path=points_path,
            output_dir=output_dir,
            endpoint_file=endpoint_path,
            lateral_max_paths=4,
            max_root_order=1,
            random_seed=23,
        )
    )

    assert result.full_above_base_mask is not None
    assert result.full_root_labels is not None
    assert np.all(result.full_above_base_mask[appended_indices])
    assert np.all(result.full_root_labels[appended_indices] == -1)
    assert np.all(result.full_root_labels[result.full_above_base_mask] == -1)

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assignment = metadata["point_assignment"]
    assert assignment["full_resolution_above_base_point_count"] == int(
        np.count_nonzero(result.full_above_base_mask)
    )
    assert assignment["unassigned_reason_counts"]["above_selected_base"] == assignment[
        "full_resolution_above_base_point_count"
    ]
    assert sum(assignment["unassigned_reason_counts"].values()) == assignment[
        "unassigned_vertex_count"
    ]
    assert (
        assignment["assigned_vertex_count"]
        + assignment["uncertain_vertex_count"]
        + assignment["unassigned_vertex_count"]
        == assignment["total_vertex_count"]
    )
    cleanup = assignment["surface_patch_correction"]
    assert cleanup["policy"] == "symmetric-mesh-surface-patches-v1"
    assert cleanup["reassigned_patch_count"] >= 0
    assert cleanup["reassigned_vertex_count"] >= 0
    trimming = metadata["branch_facing_transection_trimming"]
    assert trimming["policy"] == "branch-facing-transection-trimming-v1"
    assert trimming["sector_half_angle_degrees"] == 45
    final_cleanup = assignment["final_surface_cleanup"]
    assert final_cleanup["policy"] == "final-local-surface-cleanup-v1"
    assert final_cleanup["changed_vertex_count"] >= 0
    assert metadata["final_surface_cleanup"] == final_cleanup
    assert cleanup["converged"] is True



def test_pipeline_reassigns_points_after_reported_internal_o1_swap(
    tmp_path: Path,
    monkeypatch,
):
    points_path, endpoint_path = write_synthetic_dataset(
        tmp_path / "synthetic-contact-root.csv",
        primary_points=120,
        lateral_points=55,
        lateral_count=3,
        noise=0.001,
        seed=31,
    )
    events: list[str] = []
    real_assign_analysis = pipeline_module._assign_lateral_points
    real_assign_full = pipeline_module._assign_full_root_labels

    def assign_analysis_spy(*args, **kwargs):
        if kwargs.get("return_competing_labels", False):
            events.append("analysis_assignment")
        return real_assign_analysis(*args, **kwargs)

    def assign_full_spy(*args, **kwargs):
        events.append("full_assignment")
        return real_assign_full(*args, **kwargs)

    expected_decision: dict[str, object] = {
        "decision": "swap",
        "reason": "monkeypatched internal O1 contact",
    }

    def report_swap(
        surface_points,
        root_labels,
        lateral_paths,
        *,
        d_bar,
    ):
        events.append("internal_contact_swap")
        assert len(surface_points) == len(root_labels)
        assert d_bar > 0.0
        assert lateral_paths
        changed_id = lateral_paths[0].root_id
        expected_decision["changed_root_ids"] = [changed_id]
        return {changed_id}, [dict(expected_decision)]

    monkeypatch.setattr(
        pipeline_module,
        "_assign_lateral_points",
        assign_analysis_spy,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_assign_full_root_labels",
        assign_full_spy,
    )
    monkeypatch.setattr(
        pipeline_module,
        "uncross_internal_primary_sibling_contacts",
        report_swap,
    )

    output_dir = tmp_path / "contact-swap-output"
    run_pipeline(
        PipelineConfig(
            input_path=points_path,
            output_dir=output_dir,
            endpoint_file=endpoint_path,
            sample_points=800,
            lateral_max_paths=6,
            max_root_order=1,
            random_seed=31,
        )
    )

    assert events == [
        "analysis_assignment",
        "full_assignment",
        "internal_contact_swap",
        "analysis_assignment",
        "full_assignment",
    ]
    metadata = json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["internal_o1_contact_decisions"] == [
        expected_decision
    ]
    assert metadata["internal_o1_contact_changed_root_ids"] == (
        expected_decision["changed_root_ids"]
    )
    assert (
        metadata["lateral_tracing_policy"][
            "internal_o1_contact_uncrossing"
        ]
        is True
    )
