from __future__ import annotations

from pathlib import Path
import os

import numpy as np

from .export import order_color
from .types import RootPath


def save_overview_plot(
    path: str | Path,
    points: np.ndarray,
    primary_mask: np.ndarray,
    lateral_labels: np.ndarray,
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    max_points: int = 25000,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".mplconfig"))
    (path.parent / ".mplconfig").mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    idx = rng.choice(len(points), size=max_points, replace=False) if len(points) > max_points else np.arange(len(points))
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    unassigned = idx[(~primary_mask[idx]) & (lateral_labels[idx] == 0)]
    primary = idx[primary_mask[idx]]
    lateral = idx[lateral_labels[idx] > 0]
    if len(unassigned):
        ax.scatter(points[unassigned, 0], points[unassigned, 1], points[unassigned, 2], s=0.4, c="#888888", alpha=0.12)
    if len(primary):
        ax.scatter(points[primary, 0], points[primary, 1], points[primary, 2], s=0.7, c="#1646d8", alpha=0.45)
    if len(lateral):
        lateral_point_colors = np.tile(order_color(1), (len(lateral), 1))
        for label, path_obj in enumerate(lateral_paths, start=1):
            lateral_point_colors[lateral_labels[lateral] == label] = order_color(path_obj.order)
        ax.scatter(points[lateral, 0], points[lateral, 1], points[lateral, 2], s=0.7, c=lateral_point_colors, alpha=0.45)
    ax.plot(primary_path[:, 0], primary_path[:, 1], primary_path[:, 2], c="#001a78", linewidth=2.0)
    for lateral_path in lateral_paths:
        p = lateral_path.points
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=order_color(lateral_path.order), linewidth=1.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect(np.ptp(points, axis=0))
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_tip_angle_front_view(
    path: str | Path,
    points: np.ndarray,
    primary_mask: np.ndarray,
    lateral_labels: np.ndarray,
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    traits,
    max_points: int = 30000,
) -> None:
    """Save a 600 dpi X-Z front view with labels for every traced lateral tip."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".mplconfig"))
    (path.parent / ".mplconfig").mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    idx = rng.choice(len(points), size=max_points, replace=False) if len(points) > max_points else np.arange(len(points))
    fig, ax = plt.subplots(figsize=(8, 10))
    unassigned = idx[(~primary_mask[idx]) & (lateral_labels[idx] == 0)]
    primary = idx[primary_mask[idx]]
    lateral = idx[lateral_labels[idx] > 0]
    if len(unassigned):
        ax.scatter(points[unassigned, 0], points[unassigned, 2], s=0.35, c="#999999", alpha=0.10, linewidths=0)
    if len(primary):
        ax.scatter(points[primary, 0], points[primary, 2], s=0.55, c="#1646d8", alpha=0.35, linewidths=0)
    if len(lateral):
        lateral_point_colors = np.tile(order_color(1), (len(lateral), 1))
        for label, path_obj in enumerate(lateral_paths, start=1):
            lateral_point_colors[lateral_labels[lateral] == label] = order_color(path_obj.order)
        ax.scatter(points[lateral, 0], points[lateral, 2], s=0.55, c=lateral_point_colors, alpha=0.35, linewidths=0)
    ax.plot(primary_path[:, 0], primary_path[:, 2], c="#002090", linewidth=1.8)

    trait_by_id = {row["root_id"]: row for _, row in traits.iterrows()} if hasattr(traits, "iterrows") else {}
    x_span = max(float(np.ptp(points[:, 0])), 1e-6)
    z_span = max(float(np.ptp(points[:, 2])), 1e-6)
    for path_idx, lateral_path in enumerate(lateral_paths):
        p = lateral_path.points
        ax.plot(p[:, 0], p[:, 2], c=order_color(lateral_path.order), linewidth=1.2)
        tip = p[-1]
        row = trait_by_id.get(lateral_path.root_id, {})
        parent_angle = row.get("tip_angle_parent_deg", np.nan) if hasattr(row, "get") else np.nan
        z_angle = row.get("tip_angle_z_deg", np.nan) if hasattr(row, "get") else np.nan
        dx = (0.015 + 0.006 * (path_idx % 3)) * x_span
        dz = (0.010 + 0.006 * (path_idx % 2)) * z_span
        ax.text(tip[0] + dx, tip[2] + dz, f"O{lateral_path.order} parent {parent_angle:.1f} deg", color="#0057d8", fontsize=6)
        ax.text(tip[0] + dx, tip[2] - dz, f"Z {z_angle:.1f} deg", color="#0a8f35", fontsize=6)

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("Front view: lateral tip angles")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)
