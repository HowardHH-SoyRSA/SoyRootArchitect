from pathlib import Path

import json

import numpy as np
import pandas as pd

from soyrootbio.pipeline import PipelineConfig, run_pipeline
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
