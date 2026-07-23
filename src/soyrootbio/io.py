from __future__ import annotations

import csv
from pathlib import Path
import logging
import shutil
import tempfile
import time
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .types import PointCloudData
from .runtime import worker_threads


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
    sample_points: int | None = None,
    random_seed: int | None = 42,
    runtime_limit_seconds: float = 1800.0,
    minimum_retained_fraction: float = 0.25,
) -> PointCloudData:
    """Load a root-only point cloud or mesh and return points in source units.

    Mesh vertices and faces are preserved at full resolution for measurement
    and labelled export.  ``sample_points`` is an optional explicit analysis
    cap; when omitted, a short k-NN pilot only reduces the analysis cloud if the
    projected runtime or memory would exceed the configured 30-minute policy.
    CSV inputs prefer named x/y/z columns; other text inputs use the first
    three numeric columns.
    """
    return _load_root_geometry(
        path,
        sample_points=sample_points,
        random_seed=random_seed,
        runtime_limit_seconds=runtime_limit_seconds,
        minimum_retained_fraction=minimum_retained_fraction,
    )


def load_root_geometry_with_progress(
    path: str | Path,
    sample_points: int | None = None,
    progress_callback: ProgressCallback | None = None,
    random_seed: int | None = 42,
    runtime_limit_seconds: float = 1800.0,
    minimum_retained_fraction: float = 0.25,
) -> PointCloudData:
    """Load geometry while reporting progress and a mesh-sampling ETA."""
    return _load_root_geometry(
        path,
        sample_points=sample_points,
        progress_callback=progress_callback,
        random_seed=random_seed,
        runtime_limit_seconds=runtime_limit_seconds,
        minimum_retained_fraction=minimum_retained_fraction,
    )


def _load_root_geometry(
    path: str | Path,
    sample_points: int | None,
    progress_callback: ProgressCallback | None = None,
    random_seed: int | None = 42,
    runtime_limit_seconds: float = 1800.0,
    minimum_retained_fraction: float = 0.25,
) -> PointCloudData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if sample_points is not None and int(sample_points) != 0 and sample_points < MIN_POINT_COUNT:
        raise ValueError(f"sample_points must be at least {MIN_POINT_COUNT}; got {sample_points}")
    if runtime_limit_seconds <= 0:
        raise ValueError("runtime_limit_seconds must be positive")
    if not 0 < minimum_retained_fraction <= 1:
        raise ValueError("minimum_retained_fraction must be in (0, 1]")
    suffix = path.suffix.lower()
    if suffix in {".xyz", ".pts", ".txt", ".csv"}:
        _report_progress(progress_callback, "Reading point cloud", 0.10, None)
        text_points = _validate_points(_load_xyz_text(path), path)
        cloud = PointCloudData(
            points=text_points,
            source_path=path,
            full_points=text_points,
            source_metadata={
                "geometry_kind": "point_cloud",
                "full_point_count": int(len(text_points)),
                "analysis_point_count": int(len(text_points)),
                "analysis_reduced": False,
            },
        )
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
                    full_points=_validate_points(np.asarray(pcd.points, dtype=float), path),
                    source_metadata={
                        "geometry_kind": "point_cloud",
                        "full_point_count": int(len(pcd.points)),
                        "analysis_point_count": int(len(pcd.points)),
                        "analysis_reduced": False,
                    },
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
        point_values = _validate_points(np.asarray(pcd.points, dtype=float), path)
        cloud = PointCloudData(
            points=point_values,
            source_path=path,
            full_points=point_values,
            source_metadata={
                "geometry_kind": "point_cloud",
                "full_point_count": int(len(point_values)),
                "analysis_point_count": int(len(point_values)),
                "analysis_reduced": False,
            },
        )
        _report_progress(progress_callback, "Point cloud ready", 1.0, 0.0)
        return cloud

    _report_progress(progress_callback, "Preparing mesh", 0.10, None)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    full_points, triangles, sanitization = _sanitize_mesh_arrays(
        np.asarray(mesh.vertices, dtype=float),
        np.asarray(mesh.triangles, dtype=np.int64),
        path=path,
    )
    full_points = _validate_points(full_points, path)
    _report_progress(progress_callback, "Auditing full-resolution mesh", 0.18, None)
    audit = _mesh_audit(full_points, triangles)
    explicit_limit = None if sample_points in (None, 0) else int(sample_points)
    if explicit_limit is not None:
        target_count = min(len(full_points), explicit_limit)
        reduction_reason = "explicit_analysis_cap" if target_count < len(full_points) else "none"
        projected_seconds = None
    else:
        target_count, projected_seconds, reduction_reason = _automatic_analysis_target(
            full_points,
            runtime_limit_seconds=float(runtime_limit_seconds),
            minimum_retained_fraction=float(minimum_retained_fraction),
            graph_k=14,
            random_seed=random_seed,
        )
    _report_progress(progress_callback, "Preparing analysis vertices", 0.30, projected_seconds)
    analysis_indices = _analysis_vertex_indices(
        full_points,
        target_count=target_count,
        random_seed=random_seed,
    )
    analysis_points = full_points[analysis_indices]
    metadata = {
        "geometry_kind": "triangle_mesh",
        "full_point_count": int(len(full_points)),
        "triangle_count": int(len(triangles)),
        "analysis_point_count": int(len(analysis_points)),
        "analysis_reduced": bool(len(analysis_points) < len(full_points)),
        "retained_fraction": float(len(analysis_points) / len(full_points)),
        "reduction_reason": reduction_reason,
        "projected_full_analysis_seconds": projected_seconds,
        "runtime_limit_seconds": float(runtime_limit_seconds),
        "minimum_retained_fraction": float(minimum_retained_fraction),
        **sanitization,
        **_source_header_metadata(path),
        **audit,
    }
    LOGGER.info(
        "Loaded mesh %s with %d vertices/%d faces; analysing %d vertices (%s)",
        path,
        len(full_points),
        len(triangles),
        len(analysis_points),
        reduction_reason,
    )
    cloud = PointCloudData(
        points=analysis_points,
        source_path=path,
        full_points=full_points,
        triangles=triangles,
        analysis_indices=analysis_indices,
        source_metadata=metadata,
    )
    _report_progress(progress_callback, "Point cloud ready", 1.0, 0.0)
    return cloud


def _pilot_sample_count(sample_points: int) -> int:
    if sample_points <= 2000:
        return sample_points
    return min(sample_points, max(1000, min(5000, sample_points // 10)))


def _automatic_analysis_target(
    points: np.ndarray,
    *,
    runtime_limit_seconds: float,
    minimum_retained_fraction: float,
    graph_k: int,
    random_seed: int | None,
) -> tuple[int, float, str]:
    """Choose the largest safe analysis cloud after a small compiled k-NN pilot."""

    count = len(points)
    if count <= 25000:
        return count, 0.0, "none"
    pilot_count = min(count, 12000)
    rng = np.random.default_rng(random_seed)
    pilot_indices = rng.choice(count, size=pilot_count, replace=False)
    pilot = points[pilot_indices]
    started = time.perf_counter()
    tree = cKDTree(pilot)
    tree.query(pilot, k=min(graph_k + 1, pilot_count), workers=worker_threads())
    elapsed = max(time.perf_counter() - started, 1e-4)
    scale = (count / pilot_count) * (
        np.log2(max(count, 4)) / np.log2(max(pilot_count, 4))
    )
    # Graph construction is only one part of tracing.  The multiplier covers
    # shortest paths, cross-sections, clustering, assignment, and exports.
    # Calibrated on the supplied 101k--526k soybean meshes.  The compiled k-NN
    # pilot is roughly 1/160 of an order-1--3 end-to-end run after the linear
    # lateral-variant reduction, including topology, figures, and exports.
    projected_seconds = float(elapsed * scale * 160.0)
    estimated_graph_bytes = float(count * max(graph_k, 4) * 32 * 2)
    available_memory = _available_memory_bytes()
    memory_budget = max(512 * 1024**2, 0.35 * available_memory)
    runtime_target = count
    memory_target = count
    reasons: list[str] = []
    if projected_seconds > runtime_limit_seconds:
        runtime_target = max(
            MIN_POINT_COUNT,
            int(count * runtime_limit_seconds / projected_seconds),
        )
        reasons.append("projected_runtime_over_limit")
    if estimated_graph_bytes > memory_budget:
        memory_target = max(
            MIN_POINT_COUNT,
            int(count * memory_budget / estimated_graph_bytes),
        )
        reasons.append("projected_memory_over_limit")
    if not reasons:
        return count, projected_seconds, "none"
    floor = max(MIN_POINT_COUNT, int(np.ceil(count * minimum_retained_fraction)))
    target = min(count, max(floor, min(runtime_target, memory_target)))
    return int(target), projected_seconds, "+".join(reasons)


def _available_memory_bytes() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().available)
    except Exception:
        return float(8 * 1024**3)


def _analysis_vertex_indices(
    points: np.ndarray,
    *,
    target_count: int,
    random_seed: int | None,
) -> np.ndarray:
    count = len(points)
    target_count = max(MIN_POINT_COUNT, min(int(target_count), count))
    if target_count >= count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(random_seed)
    extrema = np.unique(
        np.concatenate(
            [
                np.argmin(points, axis=0),
                np.argmax(points, axis=0),
            ]
        )
    ).astype(np.int64)
    remaining = np.setdiff1d(np.arange(count, dtype=np.int64), extrema, assume_unique=False)
    sample_count = max(0, target_count - len(extrema))
    selected = rng.choice(remaining, size=min(sample_count, len(remaining)), replace=False)
    return np.sort(np.concatenate([extrema, selected]).astype(np.int64))


def _sanitize_mesh_arrays(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Drop non-finite vertices and safely remap surviving triangle indices."""

    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    source = "mesh" if path is None else str(path)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Mesh vertices from {source} must have shape (n, 3); got {vertices.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"Mesh triangles from {source} must have shape (m, 3); got {triangles.shape}")
    if len(triangles) and (
        int(np.min(triangles)) < 0 or int(np.max(triangles)) >= len(vertices)
    ):
        raise ValueError(f"Mesh triangles from {source} contain out-of-range vertex indices")

    finite_vertices = np.all(np.isfinite(vertices), axis=1)
    nonfinite_count = int(np.count_nonzero(~finite_vertices))
    if nonfinite_count == 0:
        return vertices, triangles, {
            "nonfinite_vertex_count": 0,
            "faces_dropped_nonfinite_vertices": 0,
        }

    keep_faces = np.all(finite_vertices[triangles], axis=1) if len(triangles) else np.ones(0, dtype=bool)
    dropped_faces = int(np.count_nonzero(~keep_faces))
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[finite_vertices] = np.arange(int(np.count_nonzero(finite_vertices)), dtype=np.int64)
    clean_vertices = vertices[finite_vertices]
    clean_triangles = old_to_new[triangles[keep_faces]]
    return clean_vertices, clean_triangles, {
        "nonfinite_vertex_count": nonfinite_count,
        "faces_dropped_nonfinite_vertices": dropped_faces,
    }


def _mesh_audit(vertices: np.ndarray, triangles: np.ndarray) -> dict:
    """Compute mesh-integrity and exact triangle-measurement evidence."""

    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        return {
            "surface_area_source_units2": 0.0,
            "signed_volume_source_units3": 0.0,
            "absolute_volume_source_units3": 0.0,
            "volume_reliable": False,
            "volume_reliability_reason": "no_triangle_faces",
            "boundary_edge_count": 0,
            "nonmanifold_edge_count": 0,
            "orientation_inconsistent_edge_count": 0,
            "degenerate_face_count": 0,
            "connected_component_count": int(len(vertices) > 0),
            "largest_component_vertex_fraction": 1.0,
        }
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    double_areas = np.linalg.norm(cross, axis=1)
    surface_area = float(0.5 * np.sum(double_areas))
    degenerate = int(np.count_nonzero(double_areas <= 1e-15))

    directed_edges = np.vstack(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ]
    ).astype(np.int64, copy=False)
    edges = np.sort(directed_edges, axis=1)
    unique_edges, inverse_edges, edge_counts = np.unique(
        edges,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    boundary_edges = int(np.count_nonzero(edge_counts == 1))
    nonmanifold_edges = int(np.count_nonzero(edge_counts > 2))
    directions = np.where(directed_edges[:, 0] <= directed_edges[:, 1], 1, -1)
    direction_sums = np.bincount(
        inverse_edges,
        weights=directions,
        minlength=len(unique_edges),
    )
    orientation_inconsistent_edges = int(
        np.count_nonzero((edge_counts == 2) & (np.abs(direction_sums) > 0.5))
    )
    adjacency = sparse.csr_matrix(
        (
            np.ones(len(unique_edges), dtype=np.uint8),
            (unique_edges[:, 0], unique_edges[:, 1]),
        ),
        shape=(len(vertices), len(vertices)),
    )
    adjacency = adjacency.maximum(adjacency.T)
    component_count, labels = connected_components(adjacency, directed=False)
    component_sizes = np.bincount(labels, minlength=component_count)
    largest_fraction = float(component_sizes.max() / len(vertices)) if len(vertices) else 0.0
    face_components = labels[triangles[:, 0]]
    # In exact arithmetic the scalar-triple-product formula is translation
    # invariant.  Against a distant global origin, however, its large terms
    # catastrophically cancel.  Center every connected component on one of its
    # vertices before accumulating its closed-surface volume.
    component_first_vertices = np.full(component_count, -1, dtype=np.int64)
    for vertex_index, component in enumerate(labels):
        if component_first_vertices[int(component)] < 0:
            component_first_vertices[int(component)] = int(vertex_index)
    component_origins = vertices[component_first_vertices]
    origins = component_origins[face_components]
    centered_v0 = v0 - origins
    centered_v1 = v1 - origins
    centered_v2 = v2 - origins
    signed_face_volumes = (
        np.einsum("ij,ij->i", centered_v0, np.cross(centered_v1, centered_v2)) / 6.0
    )
    component_signed_volumes = np.bincount(
        face_components,
        weights=signed_face_volumes,
        minlength=component_count,
    )
    signed_volume = float(np.sum(component_signed_volumes))
    absolute_volume = float(np.sum(np.abs(component_signed_volumes)))
    reliable = (
        boundary_edges == 0
        and nonmanifold_edges == 0
        and degenerate == 0
        and orientation_inconsistent_edges == 0
    )
    return {
        "surface_area_source_units2": surface_area,
        "signed_volume_source_units3": signed_volume,
        "absolute_volume_source_units3": absolute_volume,
        "volume_reliable": bool(reliable),
        "volume_reliability_reason": (
            "closed_manifold_mesh"
            if reliable
            else (
                f"boundary_edges={boundary_edges};nonmanifold_edges={nonmanifold_edges};"
                f"degenerate_faces={degenerate};orientation_inconsistent_edges={orientation_inconsistent_edges}"
            )
        ),
        "boundary_edge_count": boundary_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
        "degenerate_face_count": degenerate,
        "orientation_inconsistent_edge_count": orientation_inconsistent_edges,
        "connected_component_count": int(component_count),
        "largest_component_vertex_fraction": largest_fraction,
    }


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
    if path.suffix.lower() == ".csv":
        named = _load_named_xyz_csv(path)
        if named is not None:
            return named
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


def _load_named_xyz_csv(path: Path) -> np.ndarray | None:
    """Read named x/y/z columns, returning ``None`` for headerless CSV data."""

    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        header: list[str] | None = None
        for row in reader:
            if not row or not any(value.strip() for value in row):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            header = row
            break
        if header is None:
            return None
        normalized = [value.strip().lower() for value in header]
        if not {"x", "y", "z"}.issubset(normalized):
            return None
        xyz_indices = tuple(normalized.index(axis) for axis in ("x", "y", "z"))
        rows: list[list[float]] = []
        for row in reader:
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                rows.append([float(row[index]) for index in xyz_indices])
            except (IndexError, ValueError):
                continue
    if not rows:
        raise ValueError(f"No XYZ coordinate rows found under named x/y/z columns in {path}")
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


def write_labeled_ply(
    path: str | Path,
    vertices: np.ndarray,
    *,
    triangles: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    root_ids: np.ndarray | None = None,
    root_orders: np.ndarray | None = None,
    assignment_states: np.ndarray | None = None,
) -> None:
    """Write a binary PLY with editable root labels and optional mesh faces."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (n, 3)")
    count = len(vertices)
    triangles = None if triangles is None else np.asarray(triangles, dtype=np.int32)
    if triangles is not None and (triangles.ndim != 2 or triangles.shape[1] != 3):
        raise ValueError("triangles must have shape (m, 3)")
    if colors is None:
        color_bytes = np.full((count, 3), 140, dtype=np.uint8)
    else:
        colors = np.asarray(colors)
        if colors.shape != (count, 3):
            raise ValueError("colors must have shape (n, 3)")
        color_bytes = (
            np.clip(colors, 0.0, 1.0) * 255.0
        ).round().astype(np.uint8) if np.issubdtype(colors.dtype, np.floating) else np.clip(colors, 0, 255).astype(np.uint8)
    root_ids = np.full(count, -1, dtype=np.int32) if root_ids is None else np.asarray(root_ids, dtype=np.int32)
    root_orders = np.full(count, 255, dtype=np.uint8) if root_orders is None else np.asarray(root_orders, dtype=np.uint8)
    assignment_states = np.zeros(count, dtype=np.uint8) if assignment_states is None else np.asarray(assignment_states, dtype=np.uint8)
    for name, values in (
        ("root_ids", root_ids),
        ("root_orders", root_orders),
        ("assignment_states", assignment_states),
    ):
        if values.shape != (count,):
            raise ValueError(f"{name} must contain one value per vertex")
    vertex_dtype = np.dtype(
        [
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("root_id", "<i4"),
            ("root_order", "u1"),
            ("assignment_state", "u1"),
        ]
    )
    vertex_records = np.empty(count, dtype=vertex_dtype)
    vertex_records["x"], vertex_records["y"], vertex_records["z"] = vertices.T
    vertex_records["red"], vertex_records["green"], vertex_records["blue"] = color_bytes.T
    vertex_records["root_id"] = root_ids
    vertex_records["root_order"] = root_orders
    vertex_records["assignment_state"] = assignment_states
    face_count = 0 if triangles is None else len(triangles)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment SoyRootBio root_id -2=uncertain -1=unassigned; root_order 254=uncertain 255=unassigned; assignment_state 0=unassigned 1=assigned 2=uncertain\n"
        f"element vertex {count}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property int root_id\nproperty uchar root_order\nproperty uchar assignment_state\n"
        f"element face {face_count}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertex_records.tofile(handle)
        if triangles is not None and len(triangles):
            face_dtype = np.dtype([("count", "u1"), ("vertices", "<i4", (3,))])
            faces = np.empty(len(triangles), dtype=face_dtype)
            faces["count"] = 3
            faces["vertices"] = triangles
            faces.tofile(handle)


def _source_header_metadata(path: Path) -> dict:
    if path.suffix.lower() != ".stl":
        return {"declared_coordinate_unit": None}
    try:
        with path.open("rb") as handle:
            header = handle.read(80).decode("ascii", errors="ignore").strip("\x00 ")
    except OSError:
        return {"declared_coordinate_unit": None}
    lowered = header.lower()
    unit = "mm" if "unit:millimeter" in lowered or "unit: millimeter" in lowered else None
    return {"stl_header": header, "declared_coordinate_unit": unit}


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
