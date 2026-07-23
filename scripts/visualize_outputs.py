from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from soyrootbio.io import load_root_geometry
from soyrootbio.types import RootPath
from soyrootbio.visualize import save_overview_plot


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an overview plot from exported soyrootbio CSV and PLY files.")
    parser.add_argument("--points", required=True, type=Path, help="segmented_points.ply output.")
    parser.add_argument("--primary", required=True, type=Path, help="primary_skeleton.csv output.")
    parser.add_argument("--laterals", required=True, type=Path, help="lateral_skeletons.csv output.")
    parser.add_argument("--output", required=True, type=Path, help="PNG path.")
    args = parser.parse_args()
    cloud = load_root_geometry(args.points)
    points = cloud.points
    primary_csv = pd.read_csv(args.primary)
    primary_path = primary_csv[["x", "y", "z"]].to_numpy(float)
    lateral_csv = pd.read_csv(args.laterals)
    lateral_paths = []
    for root_id, frame in lateral_csv.groupby("root_id"):
        frame = frame.sort_values("node_id")
        lateral_paths.append(RootPath(root_id=root_id, points=frame[["x", "y", "z"]].to_numpy(float)))
    primary_mask = np.zeros(len(points), dtype=bool)
    lateral_labels = np.zeros(len(points), dtype=int)
    save_overview_plot(args.output, points, primary_mask, lateral_labels, primary_path, lateral_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
