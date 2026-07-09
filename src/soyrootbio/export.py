from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import resample_polyline
from .io import write_point_cloud
from .types import Normalization, RootPath


SEGMENT_COLORS = {
    "unassigned": np.array([0.55, 0.55, 0.55]),
    "primary": np.array([0.05, 0.23, 0.88]),
    "order_1": np.array([0.88, 0.08, 0.06]),
    "order_2": np.array([0.04, 0.62, 0.22]),
    "order_3": np.array([0.55, 0.20, 0.82]),
    "higher_order": np.array([0.95, 0.65, 0.08]),
}


def order_color(order: int) -> np.ndarray:
    if order <= 0:
        return SEGMENT_COLORS["primary"]
    return SEGMENT_COLORS.get(f"order_{order}", SEGMENT_COLORS["higher_order"])


def export_results(
    output_dir: str | Path,
    original_points: np.ndarray,
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    primary_mask: np.ndarray,
    lateral_labels: np.ndarray,
    traits: pd.DataFrame,
    normalization: Normalization,
    metadata: dict,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_primary_path = normalization.inverse_points(primary_path)
    _write_skeleton_csv(output_dir / "primary_skeleton.csv", "primary", original_primary_path)
    lateral_rows = []
    for path in lateral_paths:
        original = normalization.inverse_points(path.points)
        for node_id, xyz in enumerate(original):
            lateral_rows.append({
                "root_id": path.root_id,
                "parent_id": path.parent_id,
                "root_order": path.order,
                "node_id": node_id,
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
            })
    pd.DataFrame(lateral_rows, columns=["root_id", "parent_id", "root_order", "node_id", "x", "y", "z"]).to_csv(output_dir / "lateral_skeletons.csv", index=False)
    traits.to_csv(output_dir / "root_traits.csv", index=False)

    colors = np.tile(SEGMENT_COLORS["unassigned"], (len(original_points), 1))
    colors[primary_mask] = SEGMENT_COLORS["primary"]
    for label, path in enumerate(lateral_paths, start=1):
        colors[lateral_labels == label] = order_color(path.order)
    write_point_cloud(output_dir / "segmented_points.ply", original_points, colors=colors)
    if np.any(primary_mask):
        write_point_cloud(output_dir / "primary_points.ply", original_points[primary_mask], colors=np.tile(SEGMENT_COLORS["primary"], (int(primary_mask.sum()), 1)))
    if np.any(lateral_labels > 0):
        lateral_colors = colors[lateral_labels > 0]
        write_point_cloud(output_dir / "lateral_points.ply", original_points[lateral_labels > 0], colors=lateral_colors)
    if np.any((~primary_mask) & (lateral_labels == 0)):
        mask = (~primary_mask) & (lateral_labels == 0)
        write_point_cloud(output_dir / "unassigned_points.ply", original_points[mask], colors=np.tile(SEGMENT_COLORS["unassigned"], (int(mask.sum()), 1)))
    _write_skeleton_overlay(output_dir / "skeleton_original_overlay.ply", original_points, primary_path, lateral_paths, normalization, metadata)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(metadata), handle, indent=2, ensure_ascii=False)


def _write_skeleton_csv(path: Path, root_id: str, points: np.ndarray) -> None:
    rows = [{"root_id": root_id, "node_id": i, "x": p[0], "y": p[1], "z": p[2]} for i, p in enumerate(points)]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_skeleton_overlay(
    path: Path,
    original_points: np.ndarray,
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    normalization: Normalization,
    metadata: dict,
) -> None:
    d_bar = float(metadata.get("d_bar_normalized", 0.002))
    skeleton_spacing = max(d_bar * 0.55, 0.00025)
    points = [original_points]
    colors = [np.tile(SEGMENT_COLORS["unassigned"], (len(original_points), 1))]

    primary_original = normalization.inverse_points(resample_polyline(primary_path, skeleton_spacing))
    points.append(primary_original)
    colors.append(np.tile(SEGMENT_COLORS["primary"], (len(primary_original), 1)))
    for lateral in lateral_paths:
        lateral_original = normalization.inverse_points(resample_polyline(lateral.points, skeleton_spacing))
        points.append(lateral_original)
        colors.append(np.tile(order_color(lateral.order), (len(lateral_original), 1)))
    write_point_cloud(path, np.vstack(points), colors=np.vstack(colors))


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
