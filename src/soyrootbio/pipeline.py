from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .export import export_results
from .geometry import mean_nearest_neighbor_distance, normalize_unit_box
from .io import load_root_geometry
from .lateral import (
    backtrace_to_primary,
    find_lateral_starting_points,
    grow_lateral_candidates,
    reduce_similar_paths,
    select_non_overlapping_paths,
)
from .primary import estimate_primary_path, refine_primary_centerline, tangent_plane_primary_segmentation
from .traits import compute_traits
from .types import Normalization, RootPath
from .visualize import save_overview_plot, save_tip_angle_front_view


LOGGER = logging.getLogger(__name__)
MIN_PIPELINE_POINTS = 20


@dataclass
class PipelineConfig:
    input_path: Path
    output_dir: Path
    start: tuple[float, float, float] | None = None
    end: tuple[float, float, float] | None = None
    endpoint_file: Path | None = None
    auto_endpoints: str | None = None
    sample_points: int = 50000
    graph_k: int = 14
    lateral_max_paths: int | None = None
    max_root_order: int = 1
    unit_scale: float = 1.0
    random_seed: int = 42


@dataclass
class PipelineResult:
    output_dir: Path
    point_count: int
    d_bar: float
    primary_path: np.ndarray
    lateral_paths: list[RootPath]
    primary_mask: np.ndarray
    lateral_labels: np.ndarray
    normalization: Normalization
    lateral_start_count: int


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the soybean root skeletonization and trait workflow.

    Order 1 roots are traced from the primary skeleton. Higher orders are an MVP
    recursive extension: previous-order skeletons become parent paths, and nearby
    unassigned points are clustered and traced as child roots.
    """
    _validate_config(config)
    np.random.seed(config.random_seed)
    LOGGER.info("Loading input geometry: %s", config.input_path)
    cloud = load_root_geometry(config.input_path, sample_points=config.sample_points)
    if len(cloud.points) < MIN_PIPELINE_POINTS:
        raise ValueError(f"Too few points for skeletonization: found {len(cloud.points)}, need at least {MIN_PIPELINE_POINTS}")
    normalized, base_normalization = normalize_unit_box(cloud.points)
    normalization = Normalization(base_normalization.minimum, base_normalization.scale, config.unit_scale)
    d_bar = mean_nearest_neighbor_distance(normalized)
    LOGGER.info("Loaded %d points; d_bar=%g", len(normalized), d_bar)

    start, end = _resolve_endpoints(cloud.points, normalized, normalization, config)
    coarse_primary = estimate_primary_path(normalized, start, end, d_bar=d_bar, graph_k=config.graph_k)
    primary_mask = tangent_plane_primary_segmentation(normalized, coarse_primary.points, d_bar=d_bar)
    refined_primary_points = refine_primary_centerline(normalized, primary_mask, coarse_primary.points, d_bar=d_bar)
    primary = RootPath(root_id="primary", points=refined_primary_points)
    LOGGER.info("Primary segmentation assigned %d/%d points", int(primary_mask.sum()), len(primary_mask))

    selected, lateral_start_count, candidate_count, order_counts = _trace_lateral_orders(
        normalized,
        primary.points,
        primary_mask,
        d_bar,
        max_root_order=config.max_root_order,
        max_paths=config.lateral_max_paths,
    )
    if lateral_start_count == 0:
        LOGGER.warning("No lateral root starting points detected; exporting primary-root-only results.")
    lateral_labels = _assign_lateral_points(normalized, selected, primary_mask, d_bar)
    traits = compute_traits(primary.points, selected, normalized, primary_mask, lateral_labels, normalization, lateral_start_count=lateral_start_count)
    metadata = {
        "source": str(config.input_path),
        "algorithm_reference": "Zhou et al. 2025, Computers and Electronics in Agriculture, DOI 10.1016/j.compag.2025.110890",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "point_count": len(normalized),
        "d_bar_normalized": d_bar,
        "normalization_minimum": normalization.minimum,
        "normalization_scale": normalization.scale,
        "unit_scale": normalization.unit_scale,
        "lateral_start_count": lateral_start_count,
        "candidate_lateral_count": candidate_count,
        "selected_lateral_count": len(selected),
        "selected_order_counts": order_counts,
    }
    export_results(
        config.output_dir,
        cloud.points,
        primary.points,
        selected,
        primary_mask,
        lateral_labels,
        traits,
        normalization,
        metadata,
    )
    save_overview_plot(config.output_dir / "overview.png", normalized, primary_mask, lateral_labels, primary.points, selected)
    save_tip_angle_front_view(config.output_dir / "tip_angles_front_view_600dpi.png", normalized, primary_mask, lateral_labels, primary.points, selected, traits)
    return PipelineResult(
        output_dir=config.output_dir,
        point_count=len(normalized),
        d_bar=d_bar,
        primary_path=primary.points,
        lateral_paths=selected,
        primary_mask=primary_mask,
        lateral_labels=lateral_labels,
        normalization=normalization,
        lateral_start_count=lateral_start_count,
    )


def _trace_lateral_orders(
    points: np.ndarray,
    primary_path: np.ndarray,
    primary_mask: np.ndarray,
    d_bar: float,
    max_root_order: int,
    max_paths: int | None,
) -> tuple[list[RootPath], int, int, dict[int, int]]:
    """Trace lateral roots iteratively from parent skeletons.

    Order 1 uses the broad paper-inspired search. Higher orders use a tighter
    percentile threshold and capped candidate set so dense point clouds remain
    tractable with the current NetworkX/scikit-learn MVP implementation.
    TODO: replace this with a faster graph backend and validated order model.
    """
    selected_all: list[RootPath] = []
    occupied_mask = primary_mask.copy()
    parent_paths: list[tuple[str, np.ndarray]] = [("primary", primary_path)]
    total_starts = 0
    total_candidates = 0
    order_counts: dict[int, int] = {}

    for order in range(1, max(1, int(max_root_order)) + 1):
        if max_paths is not None and len(selected_all) >= max_paths:
            break
        order_candidates: list[RootPath] = []
        order_starts = 0
        for parent_id, parent_path in parent_paths:
            closest_fraction = 0.03 if order == 1 else 0.01
            starts = find_lateral_starting_points(points, occupied_mask, parent_path, closest_fraction=closest_fraction)
            if order > 1 and len(starts) > 3:
                starts = starts[:3]
            order_starts += len(starts)
            max_steps = 80 if order == 1 else 35
            candidates = grow_lateral_candidates(points, starts, parent_path, occupied_mask, d_bar=d_bar, max_steps=max_steps)
            for candidate in candidates:
                candidate.order = order
                candidate.parent_id = parent_id
                candidate.parent_points = parent_path
            order_candidates.extend(candidates)
        total_starts += order_starts
        total_candidates += len(order_candidates)
        if not order_candidates:
            LOGGER.info("No candidate paths for root order %d", order)
            break
        if order > 1 and len(order_candidates) > 160:
            order_candidates = sorted(order_candidates, key=lambda p: (p.score, p.length), reverse=True)[:160]
        reduced = reduce_similar_paths(order_candidates)
        remaining = None if max_paths is None else max(0, max_paths - len(selected_all))
        selected = select_non_overlapping_paths(reduced, points, d_bar=d_bar, max_paths=remaining)
        refined: list[RootPath] = []
        for path in selected:
            path.order = order
            path.root_id = f"order{order}_{len(selected_all) + len(refined) + 1:03d}"
            path.parent_points = path.parent_points if path.parent_points is not None else primary_path
            path.parent_id = path.parent_id or "primary"
            refined.extend(backtrace_to_primary([path], path.parent_points, primary_points=path.parent_points))
        if not refined:
            break
        selected_all.extend(refined)
        order_counts[order] = len(refined)
        labels = _assign_lateral_points(points, selected_all, primary_mask, d_bar)
        occupied_mask = primary_mask | (labels > 0)
        parent_paths = [(path.root_id, path.points) for path in refined]
        LOGGER.info("Selected %d order-%d lateral paths from %d starts and %d candidates", len(refined), order, order_starts, len(order_candidates))
    return selected_all, total_starts, total_candidates, order_counts

def _validate_config(config: PipelineConfig) -> None:
    if config.max_root_order < 1:
        raise ValueError(f"max_root_order must be at least 1; got {config.max_root_order}")
    if config.unit_scale <= 0 or not np.isfinite(config.unit_scale):
        raise ValueError(f"unit_scale must be a positive finite number; got {config.unit_scale}")
    if config.start is not None and config.end is None:
        raise ValueError("Both --start and --end must be provided together.")
    if config.end is not None and config.start is None:
        raise ValueError("Both --start and --end must be provided together.")


def _resolve_endpoints(
    original_points: np.ndarray,
    normalized_points: np.ndarray,
    normalization: Normalization,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if config.endpoint_file is not None:
        start, end = read_endpoint_file(config.endpoint_file)
        return _validate_and_transform_endpoints(start, end, normalization)
    if config.start is not None and config.end is not None:
        return _validate_and_transform_endpoints(np.asarray(config.start, dtype=float), np.asarray(config.end, dtype=float), normalization)
    if config.auto_endpoints == "z":
        start_idx = int(np.argmin(original_points[:, 2]))
        end_idx = int(np.argmax(original_points[:, 2]))
        return normalized_points[start_idx], normalized_points[end_idx]
    if config.auto_endpoints == "pca":
        centered = normalized_points - normalized_points.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
        scores = centered @ axis
        return normalized_points[int(np.argmin(scores))], normalized_points[int(np.argmax(scores))]
    raise ValueError("Provide --start/--end, --endpoint-file, or use --auto-endpoints z|pca.")


def read_endpoint_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read two primary-root endpoints from JSON, CSV, or whitespace text."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Endpoint file does not exist: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return np.asarray(data["start"], dtype=float), np.asarray(data["end"], dtype=float)
        arr = np.asarray(data, dtype=float)
        if arr.shape == (2, 3):
            return arr[0], arr[1]
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else None
    if frame is not None:
        lower_cols = {col.lower(): col for col in frame.columns}
        if {"x", "y", "z"}.issubset(lower_cols):
            xyz = frame[[lower_cols["x"], lower_cols["y"], lower_cols["z"]]].to_numpy(float)
            if len(xyz) >= 2:
                if "name" in lower_cols:
                    names = frame[lower_cols["name"]].astype(str).str.lower().to_numpy()
                    if "start" in names and "end" in names:
                        return xyz[np.where(names == "start")[0][0]], xyz[np.where(names == "end")[0][0]]
                return xyz[0], xyz[1]
    numbers: list[float] = []
    for token in path.read_text(encoding="utf-8", errors="ignore").replace(",", " ").split():
        try:
            numbers.append(float(token))
        except ValueError:
            continue
    if len(numbers) >= 6:
        arr = np.asarray(numbers[:6], dtype=float).reshape(2, 3)
        return arr[0], arr[1]
    raise ValueError(f"Could not parse two endpoint coordinates from {path}")


def _validate_and_transform_endpoints(start: np.ndarray, end: np.ndarray, normalization: Normalization) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError(f"Endpoint coordinates must each contain exactly 3 values; got {start.shape} and {end.shape}")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise ValueError("Endpoint coordinates must be finite numbers.")
    if np.linalg.norm(start - end) <= 1e-12:
        raise ValueError("Primary-root endpoint coordinates are identical; choose two distinct endpoints.")
    return normalization.transform_points(start), normalization.transform_points(end)


def _assign_lateral_points(points: np.ndarray, paths: list[RootPath], primary_mask: np.ndarray, d_bar: float) -> np.ndarray:
    labels = np.zeros(len(points), dtype=int)
    if not paths:
        return labels
    non_primary = np.flatnonzero(~primary_mask)
    if len(non_primary) == 0:
        return labels
    path_nodes = np.vstack([path.points for path in paths])
    node_to_label = np.concatenate([np.full(len(path.points), idx, dtype=int) for idx, path in enumerate(paths, start=1)])
    tree = cKDTree(path_nodes)
    distances, node_idx = tree.query(points[non_primary], k=1, workers=-1)
    radius = max(4.0 * d_bar, 0.006)
    assigned = distances <= radius
    labels[non_primary[assigned]] = node_to_label[node_idx[assigned]]
    return labels


