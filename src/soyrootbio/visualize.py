from __future__ import annotations

from pathlib import Path
import os
import re
import tempfile
import threading

import numpy as np
from scipy.spatial import cKDTree

from .export import order_color
from .geometry import tangent_vectors
from .types import RootPath


_MATPLOTLIB_LOCK = threading.RLock()
_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "soyrootbio-mplconfig"
_ANGLE_VECTOR_SCALE_FRACTION = 0.022
_ANGLE_ARROW_MUTATION_SCALE = 4.5
_ANGLE_LABEL_MAX_FONT_SIZE = 5.4
_ANGLE_LABEL_ENTRY_FRACTION = 0.025
_ANGLE_LABEL_OUTER_FRACTION = 0.120
_ANGLE_LABEL_COLUMN_FRACTION = 0.16
_ANGLE_LABEL_TEXT_GAP_FRACTION = 0.006
_LEGACY_ROOT_NAME = re.compile(r"^order(?P<order>\d+)[_-](?P<number>\d+)$")


def _prepare_matplotlib() -> None:
    _MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))


def save_overview_plot(
    path: str | Path,
    points: np.ndarray,
    primary_mask: np.ndarray,
    lateral_labels: np.ndarray,
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    max_points: int = 25000,
) -> None:
    with _MATPLOTLIB_LOCK:
        _prepare_matplotlib()
        _save_overview_plot(
            path,
            points,
            primary_mask,
            lateral_labels,
            primary_path,
            lateral_paths,
            max_points=max_points,
        )


def _save_overview_plot(
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

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    idx = rng.choice(len(points), size=max_points, replace=False) if len(points) > max_points else np.arange(len(points))
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    unassigned = idx[(~primary_mask[idx]) & (lateral_labels[idx] == 0)]
    uncertain = idx[(~primary_mask[idx]) & (lateral_labels[idx] < 0)]
    primary = idx[primary_mask[idx]]
    lateral = idx[lateral_labels[idx] > 0]
    if len(unassigned):
        ax.scatter(points[unassigned, 0], points[unassigned, 1], points[unassigned, 2], s=0.4, c="#888888", alpha=0.12)
    if len(uncertain):
        ax.scatter(points[uncertain, 0], points[uncertain, 1], points[uncertain, 2], s=0.5, c="#fa7a0a", alpha=0.30)
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


def save_angle_front_views(
    output_dir: str | Path,
    points: np.ndarray,
    primary_mask: np.ndarray,
    lateral_labels: np.ndarray,
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    traits,
    *,
    gravity: np.ndarray = np.array([0.0, 0.0, -1.0]),
    max_points: int = 30000,
) -> None:
    """Write one 600-dpi front view for each requested directional angle."""

    with _MATPLOTLIB_LOCK:
        _prepare_matplotlib()
        _save_angle_front_views(
            output_dir,
            points,
            primary_mask,
            lateral_labels,
            primary_path,
            lateral_paths,
            traits,
            gravity=gravity,
            max_points=max_points,
        )


def _save_angle_front_views(
    output_dir: str | Path,
    points: np.ndarray,
    primary_mask: np.ndarray,
    lateral_labels: np.ndarray,
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    traits,
    *,
    gravity: np.ndarray,
    max_points: int,
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specifications = [
        (
            "tip_gravity_front_view_600dpi.png",
            "tip_gravity_angle_deg",
            "Tip vector vs gravity",
            "tip_gravity",
        ),
        (
            "tip_start_gravity_front_view_600dpi.png",
            "tip_start_gravity_angle_deg",
            "Lateral start-to-tip vector vs gravity",
            "tip_start_gravity",
        ),
        (
            "tip_primary_front_view_600dpi.png",
            "tip_primary_angle_deg",
            "Lateral-start vector vs primary-root tangent",
            "tip_primary",
        ),
    ]
    for filename, trait_column, title, mode in specifications:
        _save_one_angle_front_view(
            output_dir / filename,
            points,
            primary_mask,
            lateral_labels,
            primary_path,
            lateral_paths,
            traits,
            trait_column=trait_column,
            title=title,
            mode=mode,
            gravity=np.asarray(gravity, dtype=float),
            max_points=max_points,
        )
    # This historical alias was a byte-for-byte copy of the tip-gravity figure.
    # Do not emit it, and remove a stale copy when this helper is used directly
    # on an existing output directory.
    (output_dir / "tip_angles_front_view_600dpi.png").unlink(missing_ok=True)


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
    """Compatibility wrapper for the original single gravity-angle figure."""

    with _MATPLOTLIB_LOCK:
        _prepare_matplotlib()
        path = Path(path)
        _save_one_angle_front_view(
            path,
            points,
            primary_mask,
            lateral_labels,
            primary_path,
            lateral_paths,
            traits,
            trait_column="tip_gravity_angle_deg" if "tip_gravity_angle_deg" in traits.columns else "tip_angle_z_deg",
            title="Tip vector vs gravity",
            mode="tip_gravity",
            gravity=np.array([0.0, 0.0, -1.0]),
            max_points=max_points,
        )


def _save_one_angle_front_view(
    path: Path,
    points: np.ndarray,
    primary_mask: np.ndarray,
    lateral_labels: np.ndarray,
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    traits,
    *,
    trait_column: str,
    title: str,
    mode: str,
    gravity: np.ndarray,
    max_points: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    idx = rng.choice(len(points), size=max_points, replace=False) if len(points) > max_points else np.arange(len(points))
    median_x = float(np.median(points[:, 0]))
    side_counts = [
        sum(float(path.points[-1, 0]) < median_x for path in lateral_paths),
        sum(float(path.points[-1, 0]) >= median_x for path in lateral_paths),
    ]
    max_side_count = max(side_counts, default=0)
    # Dense root systems need more vertical room, but keep the raster bounded.
    figure_height = max(10.0, min(18.0, 8.0 + 0.06 * max_side_count))
    fig, ax = plt.subplots(figsize=(12, figure_height))
    unassigned = idx[(~primary_mask[idx]) & (lateral_labels[idx] == 0)]
    uncertain = idx[(~primary_mask[idx]) & (lateral_labels[idx] < 0)]
    primary = idx[primary_mask[idx]]
    lateral = idx[lateral_labels[idx] > 0]
    if len(unassigned):
        ax.scatter(points[unassigned, 0], points[unassigned, 2], s=0.35, c="#999999", alpha=0.10, linewidths=0)
    if len(uncertain):
        ax.scatter(points[uncertain, 0], points[uncertain, 2], s=0.45, c="#fa7a0a", alpha=0.25, linewidths=0)
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
    label_items: dict[int, list[tuple[RootPath, np.ndarray, float]]] = {
        -1: [],
        1: [],
    }
    for lateral_path in lateral_paths:
        p = lateral_path.points
        ax.plot(p[:, 0], p[:, 2], c=order_color(lateral_path.order), linewidth=1.2)
        tip = p[-1]
        row = trait_by_id.get(lateral_path.root_id, {})
        angle = row.get(trait_column, np.nan) if hasattr(row, "get") else np.nan
        outward = -1 if tip[0] < median_x else 1
        _draw_angle_vectors(
            ax,
            lateral_path,
            primary_path,
            trait_row=row,
            mode=mode,
            scale=max(_ANGLE_VECTOR_SCALE_FRACTION * max(x_span, z_span), 1e-4),
        )
        label_items[outward].append((lateral_path, tip, float(angle)))

    x_min, x_max = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
    z_min, z_max = float(np.min(points[:, 2])), float(np.max(points[:, 2]))
    label_z_padding = max(0.025 * z_span, 1e-4)
    pending_labels: list[
        tuple[
            RootPath,
            np.ndarray,
            float,
            tuple[float, float],
            float,
            float,
            float,
            int,
        ]
    ] = []
    for outward, side_items in label_items.items():
        if not side_items:
            continue
        label_x = (
            x_min - _ANGLE_LABEL_COLUMN_FRACTION * x_span
            if outward < 0
            else x_max + _ANGLE_LABEL_COLUMN_FRACTION * x_span
        )
        entry_x = (
            x_min - _ANGLE_LABEL_ENTRY_FRACTION * x_span
            if outward < 0
            else x_max + _ANGLE_LABEL_ENTRY_FRACTION * x_span
        )
        outer_x = (
            x_min - _ANGLE_LABEL_OUTER_FRACTION * x_span
            if outward < 0
            else x_max + _ANGLE_LABEL_OUTER_FRACTION * x_span
        )
        label_line_end_x = (
            label_x - outward * _ANGLE_LABEL_TEXT_GAP_FRACTION * x_span
        )
        for lateral_path, tip, angle, label_z in _ordered_angle_label_layout(
            side_items,
            z_min=z_min,
            z_max=z_max,
            z_padding=label_z_padding,
        ):
            pending_labels.append(
                (
                    lateral_path,
                    tip,
                    angle,
                    (label_x, float(label_z)),
                    entry_x,
                    outer_x,
                    label_line_end_x,
                    outward,
                )
            )

    x_padding = max(0.30 * x_span, 1e-4)
    z_padding = max(0.03 * z_span, 1e-4)

    from matplotlib.lines import Line2D

    legend_spec = _angle_vector_legend_spec(mode)
    ax.legend(
        handles=[
            Line2D([0], [0], color=color, lw=1.2, label=label)
            for color, label in legend_spec
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.025),
        ncol=len(legend_spec),
        fontsize=7,
        framealpha=0.92,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(f"Front view: {title}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(z_min - z_padding, z_max + z_padding)
    fig.tight_layout()
    fig.canvas.draw()
    axes_height_points = (
        float(ax.get_window_extent().height) * 72.0 / float(fig.dpi)
    )
    usable_height_fraction = max(
        (z_span - 2.0 * label_z_padding) / (z_span + 2.0 * z_padding),
        0.01,
    )
    label_font_sizes = {
        outward: _adaptive_angle_label_font_size(
            len(side_items),
            axes_height_points=axes_height_points,
            usable_height_fraction=usable_height_fraction,
        )
        for outward, side_items in label_items.items()
    }
    for (
        lateral_path,
        tip,
        angle,
        label_position,
        entry_x,
        outer_x,
        label_line_end_x,
        outward,
    ) in pending_labels:
        _annotate_angle_tip(
            ax,
            lateral_path,
            tip,
            angle,
            label_position=label_position,
            entry_x=entry_x,
            outer_x=outer_x,
            label_line_end_x=label_line_end_x,
            outward=outward,
            font_size=label_font_sizes[outward],
        )
    fig.savefig(path, dpi=600)
    plt.close(fig)


def _angle_vector_legend_spec(mode: str) -> list[tuple[str, str]]:
    """Return compact legend entries for vectors that are actually displayed."""

    if mode == "tip_start_gravity":
        vector_label = "S–T"
    elif mode == "tip_primary":
        vector_label = "Start"
    else:
        vector_label = "Tip"
    entries = [("#b00020", vector_label)]
    if mode == "tip_primary":
        entries.append(("#14833b", "Primary"))
    return entries


def _display_root_name(root_id: str) -> str:
    """Return the compact order-number identifier used in angle figures."""

    name = str(root_id).removeprefix("root-")
    legacy_match = _LEGACY_ROOT_NAME.fullmatch(name)
    if legacy_match:
        return f"o{legacy_match.group('order')}-{legacy_match.group('number')}"
    return name


def _format_angle_tip_label(root_id: str, angle: float) -> str:
    """Format a short root-and-angle label without the storage ID prefix."""

    angle_text = f"{angle:.1f}°" if np.isfinite(angle) else "—"
    return f"{_display_root_name(root_id)} {angle_text}"


def _adaptive_angle_label_font_size(
    label_count: int,
    *,
    axes_height_points: float,
    usable_height_fraction: float = 0.95,
) -> float:
    """Fit one-line labels into the available vertical side-column height."""

    if label_count <= 1:
        return _ANGLE_LABEL_MAX_FONT_SIZE
    usable_height = max(
        float(axes_height_points) * float(usable_height_fraction),
        1.0,
    )
    label_pitch = usable_height / float(label_count - 1)
    # Keep a visible gap between adjacent glyph boxes. Extremely dense figures
    # may use very small text, but never permit overlap merely to enforce a
    # cosmetic minimum font size.
    return max(1.0, min(_ANGLE_LABEL_MAX_FONT_SIZE, 0.72 * label_pitch))


def _ordered_angle_label_layout(
    side_items: list[tuple[RootPath, np.ndarray, float]],
    *,
    z_min: float,
    z_max: float,
    z_padding: float,
) -> list[tuple[RootPath, np.ndarray, float, float]]:
    """Assign label heights in the same order as projected root-tip heights."""

    ordered = sorted(
        side_items,
        key=lambda item: (float(item[1][2]), item[0].root_id),
    )
    if not ordered:
        return []
    lower = float(z_min + z_padding)
    upper = float(z_max - z_padding)
    if len(ordered) == 1:
        label_z_values = np.asarray(
            [np.clip(float(ordered[0][1][2]), lower, upper)],
            dtype=float,
        )
    else:
        label_z_values = np.linspace(lower, upper, len(ordered))
    return [
        (lateral, tip, angle, float(label_z))
        for (lateral, tip, angle), label_z in zip(
            ordered,
            label_z_values,
            strict=True,
        )
    ]


def _angle_label_route(
    tip: np.ndarray,
    *,
    entry_x: float,
    outer_x: float,
    label_line_end_x: float,
    label_position: tuple[float, float],
) -> np.ndarray:
    """Return a three-segment outward polyline from a tip to its label."""

    return np.asarray(
        [
            [float(tip[0]), float(tip[2])],
            [float(entry_x), float(tip[2])],
            [float(outer_x), float(label_position[1])],
            [float(label_line_end_x), float(label_position[1])],
        ],
        dtype=float,
    )


def _annotate_angle_tip(
    ax,
    lateral: RootPath,
    tip: np.ndarray,
    angle: float,
    *,
    label_position: tuple[float, float],
    entry_x: float,
    outer_x: float,
    label_line_end_x: float,
    outward: int,
    font_size: float,
) -> None:
    """Connect a compact side-column label with a routed polyline indicatrix."""

    route = _angle_label_route(
        tip,
        entry_x=entry_x,
        outer_x=outer_x,
        label_line_end_x=label_line_end_x,
        label_position=label_position,
    )
    ax.plot(
        route[:, 0],
        route[:, 1],
        color="#4774bf",
        linewidth=0.32,
        alpha=0.62,
        solid_capstyle="round",
        solid_joinstyle="round",
        clip_on=False,
        zorder=1.8,
    )
    ax.text(
        label_position[0],
        label_position[1],
        _format_angle_tip_label(lateral.root_id, angle),
        ha="left" if outward >= 0 else "right",
        va="center",
        color="#073f9c",
        fontsize=font_size,
        clip_on=False,
        zorder=3.0,
    )


def _draw_angle_vectors(
    ax,
    lateral: RootPath,
    primary_path: np.ndarray,
    *,
    trait_row,
    mode: str,
    scale: float,
) -> None:
    path = lateral.points
    tip = path[-1]

    def recorded_vector(prefix: str, fallback: np.ndarray) -> np.ndarray:
        if hasattr(trait_row, "get"):
            values = np.asarray(
                [
                    trait_row.get(f"{prefix}_dx", np.nan),
                    trait_row.get(f"{prefix}_dy", np.nan),
                    trait_row.get(f"{prefix}_dz", np.nan),
                ],
                dtype=float,
            )
            if np.all(np.isfinite(values)) and np.linalg.norm(values) > 1e-12:
                return values
        return np.asarray(fallback, dtype=float)

    span = min(4, len(path) - 1)
    tip_start = path[-1 - span] if span > 0 else tip
    measured_vector = recorded_vector("tip_vector", tip - tip_start)
    vector_origin = tip
    if mode == "tip_start_gravity":
        start = path[0]
        measured_vector = recorded_vector("tip_start_vector", tip - start)
    elif mode == "tip_primary":
        vector_origin = path[0]
        start_end = path[span] if span > 0 else vector_origin
        measured_vector = recorded_vector(
            "base_vector",
            start_end - vector_origin,
        )
    measured_unit = measured_vector / max(np.linalg.norm(measured_vector), 1e-12)

    # The gravity views place the measured direction at the root tip.  The
    # start-primary view places both rays at the lateral insertion so the
    # artwork matches the numerical angle definition.
    vector_end = vector_origin + measured_unit * scale
    ax.annotate(
        "",
        xy=(vector_end[0], vector_end[2]),
        xytext=(vector_origin[0], vector_origin[2]),
        arrowprops={
            "arrowstyle": "->",
            "color": "#b00020",
            "lw": 0.70,
            "mutation_scale": _ANGLE_ARROW_MUTATION_SCALE,
        },
    )

    # Gravity is still the numerical reference for the two gravity-angle
    # traits, but it is intentionally omitted from the front-view artwork.
    if mode != "tip_primary":
        return

    primary_tree = cKDTree(primary_path)
    _, primary_index = primary_tree.query(path[0], k=1)
    primary_vector = recorded_vector(
        "primary_vector",
        tangent_vectors(primary_path)[int(primary_index)],
    )
    primary_vector /= max(np.linalg.norm(primary_vector), 1e-12)
    reference_end = vector_origin + primary_vector * scale
    ax.annotate(
        "",
        xy=(reference_end[0], reference_end[2]),
        xytext=(vector_origin[0], vector_origin[2]),
        arrowprops={
            "arrowstyle": "->",
            "color": "#14833b",
            "lw": 0.70,
            "mutation_scale": _ANGLE_ARROW_MUTATION_SCALE,
        },
    )
