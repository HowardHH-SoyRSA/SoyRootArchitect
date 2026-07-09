from pathlib import Path

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
    assert (output_dir / "tip_angles_front_view_600dpi.png").exists()
    assert (output_dir / "skeleton_original_overlay.ply").exists()
    assert list(lateral_skeletons.columns) == ["root_id", "parent_id", "root_order", "node_id", "x", "y", "z"]
