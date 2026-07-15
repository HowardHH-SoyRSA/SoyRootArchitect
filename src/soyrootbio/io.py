from __future__ import annotations

from pathlib import Path
import logging
import shutil
import tempfile
import time
from typing import Callable

import numpy as np

from .types import PointCloudData


LOGGER = logging.getLogger(__name__)
MIN_POINT_COUNT = 10
POINT_EXTENSIONS = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts", ".txt", ".csv"}
MESH_EXTENSIONS = {".stl", ".obj", ".off", ".gltf", ".glb", ".fbx", ".dae", ".ply"}
ProgressCallback = Callable[[str, float, float | None], None]


def require_open3d():
    """Import Open3D lazily so dependency errors are easy to understand."""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required for loading, sampling, and exporting 3D files.") from exc
    return o3d


def load_root_geometry(
    path: str | Path,
    sample_points: int = 50000,
    random_seed: int | None = 42,
) -> PointCloudData:
    """Load a root-only point cloud or mesh and return points in source units.

    Mesh inputs are sampled uniformly with Open3D. Text inputs use the first
    three numeric columns, allowing simple XYZ files or CSV files with headers.
    """
    return _load_root_geometry(path, sample_points=sample_points, random_seed=random_seed)


def load_root_geometry_with_progress(
    path: str | Path,
    sample_points: int = 50000,
    progress_callback: ProgressCallback | None = None,
    random_seed: int | None = 42,
) -> PointCloudData:
    """Load geometry while reporting progress and a mesh-sampling ETA."""
    return _load_root_geometry(
        path,
        sample_points=sample_points,
        progress_callback=progress_callback,
        random_seed=random_seed,
    )


def _load_root_geometry(
    path: str | Path,
    sample_points: int,
    progress_callback: ProgressCallback | None = None,
    random_seed: int | None = 42,
) -> PointCloudData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if sample_points < MIN_POINT_COUNT:
        raise ValueError(f"sample_points must be at least {MIN_POINT_COUNT}; got {sample_points}")
    suffix = path.suffix.lower()
    if suffix in {".xyz", ".pts", ".txt", ".csv"}:
        _report_progress(progress_callback, "Reading point cloud", 0.10, None)
        cloud = PointCloudData(points=_validate_points(_load_xyz_text(path), path), source_path=path)
        _report_progress(progress_callback, "Point cloud ready", 1.0, 0.0)
        return cloud

    o3d = require_open3d()
    mesh = None
    if suffix == ".ply":
        # PLY can be either a point cloud or a triangle mesh. Check faces first
        # so the GUI's mesh-sample setting is honored for triangle PLY files.
        _report_progress(progress_callback, "Inspecting PLY geometry", 0.05, None)
        candidate_mesh = _read_triangle_mesh(o3d, path)
        if candidate_mesh is not None and len(candidate_mesh.triangles) > 0:
            mesh = candidate_mesh
        else:
            _report_progress(progress_callback, "Reading point cloud", 0.10, None)
            pcd = _read_point_cloud(o3d, path)
            if pcd.has_points():
                cloud = PointCloudData(
                    points=_validate_points(np.asarray(pcd.points, dtype=float), path),
                    source_path=path,
                )
                _report_progress(progress_callback, "Point cloud ready", 1.0, 0.0)
                return cloud

    if mesh is None:
        _report_progress(progress_callback, "Reading mesh", 0.05, None)
        mesh = _read_triangle_mesh(o3d, path)
    if mesh is None or len(mesh.vertices) == 0:
        _report_progress(progress_callback, "Reading point cloud", 0.20, None)
        pcd = _read_point_cloud(o3d, path)
        if not pcd.has_points():
            raise ValueError(f"Could not read point cloud or mesh from {path}")
        cloud = PointCloudData(points=_validate_points(np.asarray(pcd.points, dtype=float), path), source_path=path)
        _report_progress(progress_callback, "Point cloud ready", 1.0, 0.0)
        return cloud

    _report_progress(progress_callback, "Preparing mesh", 0.10, None)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    LOGGER.info("Sampling mesh %s to %d points", path, sample_points)
    eta_seconds: float | None = None
    pilot_count = _pilot_sample_count(sample_points)
    if progress_callback is not None and pilot_count < sample_points:
        _report_progress(progress_callback, "Estimating sampling rate", 0.14, None)
        _seed_open3d(o3d, random_seed)
        started = time.perf_counter()
        mesh.sample_points_uniformly(number_of_points=pilot_count)
        elapsed = time.perf_counter() - started
        if elapsed > 1e-6:
            eta_seconds = elapsed * sample_points / pilot_count
    _report_progress(progress_callback, "Sampling mesh", 0.25, eta_seconds)
    _seed_open3d(o3d, random_seed)
    pcd = mesh.sample_points_uniformly(number_of_points=int(sample_points))
    cloud = PointCloudData(points=_validate_points(np.asarray(pcd.points, dtype=float), path), source_path=path)
    _report_progress(progress_callback, "Point cloud ready", 1.0, 0.0)
    return cloud


def _pilot_sample_count(sample_points: int) -> int:
    if sample_points <= 2000:
        return sample_points
    return min(sample_points, max(1000, min(5000, sample_points // 10)))


def _seed_open3d(o3d, random_seed: int | None) -> None:
    if random_seed is not None:
        o3d.utility.random.seed(int(random_seed))


def _report_progress(
    callback: ProgressCallback | None,
    stage: str,
    fraction: float,
    eta_seconds: float | None,
) -> None:
    if callback is not None:
        callback(stage, float(np.clip(fraction, 0.0, 1.0)), eta_seconds)


def _validate_points(points: np.ndarray, path: Path) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        raise ValueError(f"Empty point cloud loaded from {path}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Point cloud from {path} must have shape (n, 3); got {points.shape}")
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < MIN_POINT_COUNT:
        raise ValueError(f"Too few valid points in {path}: found {len(points)}, need at least {MIN_POINT_COUNT}")
    return points


def _load_xyz_text(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < 3:
                continue
            row = _first_three_numbers(parts)
            if row is not None:
                rows.append(row)
    if not rows:
        raise ValueError(f"No XYZ/CSV coordinate rows found in {path}")
    return np.asarray(rows, dtype=float)


def _first_three_numbers(parts: list[str]) -> list[float] | None:
    numbers: list[float] = []
    for part in parts:
        try:
            numbers.append(float(part))
        except ValueError:
            continue
        if len(numbers) == 3:
            return numbers
    return None


def write_point_cloud(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    o3d = require_open3d()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    if not _write_point_cloud(o3d, path, pcd):
        raise IOError(f"Failed to write point cloud: {path}")


def _read_triangle_mesh(o3d, path: Path):
    try:
        return o3d.io.read_triangle_mesh(str(path))
    except UnicodeDecodeError:
        with tempfile.TemporaryDirectory(prefix="soyrootbio_") as tmpdir:
            temp_path = Path(tmpdir) / f"input{path.suffix.lower()}"
            shutil.copy2(path, temp_path)
            return o3d.io.read_triangle_mesh(str(temp_path))


def _read_point_cloud(o3d, path: Path):
    try:
        return o3d.io.read_point_cloud(str(path))
    except UnicodeDecodeError:
        with tempfile.TemporaryDirectory(prefix="soyrootbio_") as tmpdir:
            temp_path = Path(tmpdir) / f"input{path.suffix.lower()}"
            shutil.copy2(path, temp_path)
            return o3d.io.read_point_cloud(str(temp_path))


def _write_point_cloud(o3d, path: Path, pcd) -> bool:
    try:
        return bool(o3d.io.write_point_cloud(str(path), pcd))
    except (UnicodeDecodeError, UnicodeEncodeError):
        with tempfile.TemporaryDirectory(prefix="soyrootbio_") as tmpdir:
            temp_path = Path(tmpdir) / f"output{path.suffix.lower()}"
            ok = bool(o3d.io.write_point_cloud(str(temp_path), pcd))
            if ok:
                shutil.copy2(temp_path, path)
            return ok
