from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_taproot(
    primary_points: int = 180,
    lateral_points: int = 80,
    lateral_count: int = 3,
    noise: float = 0.004,
    seed: int = 7,
) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    """Create a simple root-only point cloud with one taproot and first-order laterals.

    The geometry is intentionally modest: a mostly vertical primary root and several
    oblique lateral branches. It is for smoke testing pipeline behavior, not for
    reproducing CT surface complexity.
    """
    rng = np.random.default_rng(seed)
    z = np.linspace(0.0, 1.0, primary_points)
    primary = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    primary += rng.normal(scale=noise, size=primary.shape)

    branches = [primary]
    anchors = np.linspace(0.24, 0.78, lateral_count)
    for branch_id, anchor_z in enumerate(anchors):
        length = 0.22 + 0.08 * (branch_id % 2)
        t = np.linspace(0.0, 1.0, lateral_points)
        side = -1.0 if branch_id % 2 else 1.0
        angle = 0.48 + 0.13 * branch_id
        x = side * length * np.sin(angle) * t
        y = 0.04 * np.sin(np.pi * t + branch_id)
        branch_z = anchor_z + length * np.cos(angle) * t
        branch = np.column_stack([x, y, branch_z])
        branch += rng.normal(scale=noise, size=branch.shape)
        branches.append(branch)

    points = np.vstack(branches)
    start = tuple(primary[0].tolist())
    end = tuple(primary[-1].tolist())
    return points, start, end


def write_synthetic_dataset(
    output: str | Path,
    primary_points: int = 180,
    lateral_points: int = 80,
    lateral_count: int = 3,
    noise: float = 0.004,
    seed: int = 7,
) -> tuple[Path, Path]:
    """Write a synthetic XYZ/CSV-compatible point cloud and endpoint CSV file."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    points, start, end = generate_synthetic_taproot(primary_points, lateral_points, lateral_count, noise, seed)
    pd.DataFrame(points, columns=["x", "y", "z"]).to_csv(output, index=False)
    endpoint_path = output.with_name(output.stem + "_endpoints.csv")
    pd.DataFrame(
        [
            {"name": "start", "x": start[0], "y": start[1], "z": start[2]},
            {"name": "end", "x": end[0], "y": end[1], "z": end[2]},
        ]
    ).to_csv(endpoint_path, index=False)
    return output, endpoint_path
