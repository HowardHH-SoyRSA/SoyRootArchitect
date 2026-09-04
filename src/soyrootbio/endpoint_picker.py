from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd

from .io import load_root_geometry_with_progress
from .primary_guidance import PrimaryGuidance
from .types import PointCloudData


def _wheel_zoom_factor(button, step: float | int | None = None) -> float | None:
    """Normalize Matplotlib wheel events across Tk/backend variants."""

    numeric_step = 0.0 if step is None else float(step)
    if button == "up" or numeric_step > 0.0:
        return 0.75
    if button == "down" or numeric_step < 0.0:
        return 1.35
    return None


def _maximize_figure_window(figure) -> bool:
    """Maximize a Matplotlib selection window using its active GUI backend."""

    manager = getattr(getattr(figure, "canvas", None), "manager", None)
    window = getattr(manager, "window", None)
    if window is None:
        return False
    for method_name, arguments in (
        ("showMaximized", ()),
        ("state", ("zoomed",)),
        ("wm_state", ("zoomed",)),
        ("Maximize", (True,)),
        ("maximize", ()),
    ):
        method = getattr(window, method_name, None)
        if not callable(method):
            continue
        try:
            method(*arguments)
        except Exception:
            continue
        return True
    return False


def _zoom_3d_axis(axis, factor: float, canvas=None) -> None:
    """Scale all three visible limits about their current centres."""

    for getter, setter in (
        (axis.get_xlim, axis.set_xlim),
        (axis.get_ylim, axis.set_ylim),
        (axis.get_zlim, axis.set_zlim),
    ):
        lower, upper = getter()
        center = (lower + upper) / 2.0
        half = max((upper - lower) * float(factor) / 2.0, 1e-12)
        setter(center - half, center + half)
    if canvas is not None:
        canvas.draw_idle()


def _configure_selection_axis(axis, points: np.ndarray, *, title: str = "Root point cloud") -> None:
    """Apply the shared, sparse-grid styling used by both selection windows."""

    from matplotlib.ticker import MaxNLocator

    axis.set_title(title, fontsize=12, pad=12)
    axis.set_xlabel("x", labelpad=8)
    axis.set_ylabel("y", labelpad=8)
    axis.set_zlabel("z", labelpad=8)
    axis.tick_params(labelsize=8)
    for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    axis.grid(True)
    axis.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1e-9))


def _inverted_drag_rotation(
    start_elevation: float,
    start_azimuth: float,
    delta_x: float,
    delta_y: float,
) -> tuple[float, float]:
    """Return the deliberately reversed left-drag camera orientation."""

    elevation = float(np.clip(float(start_elevation) + 0.30 * float(delta_y), -89.0, 89.0))
    azimuth = float(start_azimuth) - 0.35 * float(delta_x)
    return elevation, azimuth


def select_primary_endpoints_gui(
    points: np.ndarray,
    max_display_points: int = 30000,
    title: str = "Select primary-root endpoints",
    random_seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Open the endpoint picker for an already loaded point cloud.

    This compatibility entry point keeps the coordinate-entry and zoom controls.
    Use :func:`select_primary_endpoints_from_file_gui` for mesh loading with an
    adjustable sample count and progress/ETA display.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("Endpoint selection requires at least two 3D points.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Endpoint selection received non-finite point coordinates.")
    cloud = PointCloudData(points=points, source_path=Path("<memory>"))
    _, start, end, _ = _run_endpoint_picker(
        initial_cloud=cloud,
        input_path=None,
        sample_points=max(len(points), 10),
        max_display_points=max_display_points,
        title=title,
        random_seed=random_seed,
    )
    return start, end


def select_primary_endpoints_from_file_gui(
    input_path: str | Path,
    sample_points: int = 50000,
    max_display_points: int = 30000,
    title: str = "Select primary-root endpoints",
    random_seed: int | None = 42,
) -> tuple[PointCloudData, np.ndarray, np.ndarray, int]:
    """Load a root file in the picker and return the chosen endpoints.

    The window exposes a mesh sample-count input before and after loading. Mesh
    reloads happen on a background worker so users can watch a pilot-sample ETA
    instead of waiting on a frozen selection window.
    """
    cloud, start, end, effective_sample_points = _run_endpoint_picker(
        initial_cloud=None,
        input_path=Path(input_path),
        sample_points=sample_points,
        max_display_points=max_display_points,
        title=title,
        random_seed=random_seed,
    )
    if cloud is None:
        raise RuntimeError("Endpoint picker closed before a root point cloud was loaded.")
    return cloud, start, end, effective_sample_points


def write_endpoint_file(path: str | Path, start: np.ndarray, end: np.ndarray) -> Path:
    """Write two source-coordinate primary endpoints as CSV or JSON."""
    path = Path(path)
    start, end = _validate_endpoint_pair(start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps({"start": start.tolist(), "end": end.tolist()}, indent=2), encoding="utf-8")
    else:
        pd.DataFrame(
            [
                {"name": "start", "x": start[0], "y": start[1], "z": start[2]},
                {"name": "end", "x": end[0], "y": end[1], "z": end[2]},
            ]
        ).to_csv(path, index=False)
    return path


def _run_endpoint_picker(
    initial_cloud: PointCloudData | None,
    input_path: Path | None,
    sample_points: int,
    max_display_points: int,
    title: str,
    random_seed: int | None,
) -> tuple[PointCloudData | None, np.ndarray, np.ndarray, int]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import Button, TextBox
        from mpl_toolkits.mplot3d import proj3d
    except ImportError as exc:
        raise ImportError("matplotlib is required for the interactive endpoint picker.") from exc

    sample_points = _parse_sample_count(str(sample_points))
    figure = plt.figure(figsize=(15, 10))
    _maximize_figure_window(figure)
    axis = figure.add_axes([0.045, 0.095, 0.665, 0.80], projection="3d")
    figure.text(0.045, 0.952, title, fontsize=16, weight="bold")
    status_text = figure.text(0.045, 0.921, "Preparing endpoint picker...", fontsize=11)
    instruction_artist = None
    display_points: np.ndarray | None = None
    selected: list[np.ndarray] = []
    marker_artists: list[object] = []
    label_artists: list[object] = []
    lock = threading.Lock()
    state: dict[str, object] = {
        "cloud": initial_cloud,
        "pending_cloud": None,
        "loading": False,
        "stage": "Point cloud ready" if initial_cloud is not None else "Waiting to load root",
        "fraction": 1.0 if initial_cloud is not None else 0.0,
        "eta_seconds": 0.0 if initial_cloud is not None else None,
        "eta_started": None,
        "error": None,
        "sample_points": sample_points,
        "saved": False,
        "result": None,
    }
    drag_state: dict[str, object] = {
        "active": False,
        "button": None,
        "start_x": 0.0,
        "start_y": 0.0,
        "start_elev": 22.0,
        "start_azim": -60.0,
        "start_xlim": None,
        "start_ylim": None,
        "start_zlim": None,
        "dragged": False,
    }

    control_left = 0.765
    control_width = 0.20
    coordinate_boxes = {}
    coordinate_specs = [
        ("base_x", "Base X", 0.765),
        ("base_y", "Base Y", 0.705),
        ("base_z", "Base Z", 0.645),
        ("tip_x", "Tip X", 0.505),
        ("tip_y", "Tip Y", 0.445),
        ("tip_z", "Tip Z", 0.385),
    ]
    figure.text(control_left, 0.835, "Base endpoint", fontsize=12, weight="bold")
    figure.text(control_left, 0.575, "Tip endpoint", fontsize=12, weight="bold")
    for key, label, y_position in coordinate_specs:
        box = TextBox(
            figure.add_axes([control_left, y_position, control_width, 0.044]),
            label,
            initial="",
            color="#f8fafc",
            hovercolor="#eef2ff",
        )
        box.label.set_fontsize(10)
        box.text_disp.set_fontsize(11)
        coordinate_boxes[key] = box
    use_coordinates_button = Button(figure.add_axes([control_left, 0.315, control_width, 0.052]), "Use typed coordinates")
    sample_box = TextBox(
        figure.add_axes([control_left, 0.245, control_width, 0.048]),
        "Mesh samples",
        initial=str(sample_points),
        color="#f8fafc",
        hovercolor="#eef2ff",
    )
    sample_box.label.set_fontsize(10)
    sample_box.text_disp.set_fontsize(11)
    reload_button = Button(figure.add_axes([control_left, 0.185, control_width, 0.052]), "Load / reload root")
    zoom_in_button = Button(figure.add_axes([control_left, 0.115, 0.092, 0.050]), "Zoom +")
    zoom_out_button = Button(figure.add_axes([control_left + 0.108, 0.115, 0.092, 0.050]), "Zoom -")
    reset_view_button = Button(figure.add_axes([control_left, 0.055, 0.124, 0.050]), "Reset view")
    reset_selection_button = Button(figure.add_axes([control_left + 0.136, 0.055, 0.064, 0.050]), "Clear")
    save_button = Button(figure.add_axes([control_left, 0.005, control_width, 0.042]), "Save endpoints")
    for button in (
        use_coordinates_button,
        reload_button,
        zoom_in_button,
        zoom_out_button,
        reset_view_button,
        reset_selection_button,
        save_button,
    ):
        button.label.set_fontsize(10)

    progress_axis = figure.add_axes([0.05, 0.040, 0.62, 0.028])
    progress_axis.set_xlim(0.0, 1.0)
    progress_axis.set_ylim(0.0, 1.0)
    progress_axis.axis("off")
    progress_axis.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, color="#d1d5db"))
    progress_value = Rectangle((0.0, 0.0), 0.0, 1.0, color="#2563eb")
    progress_axis.add_patch(progress_value)
    progress_text = figure.text(0.05, 0.012, "Waiting to load root", fontsize=10)

    def clear_selection(clear_boxes: bool = False) -> None:
        selected.clear()
        for artist in [*marker_artists, *label_artists]:
            try:
                artist.remove()
            except (AttributeError, ValueError):
                pass
        marker_artists.clear()
        label_artists.clear()
        if clear_boxes:
            for box in coordinate_boxes.values():
                box.set_val("")

    def set_instruction(message: str) -> None:
        status_text.set_text(message)
        if instruction_artist is not None:
            instruction_artist.set_text(message)
        figure.canvas.draw_idle()

    def set_root_limits() -> None:
        if display_points is None or len(display_points) == 0:
            return
        center = display_points.mean(axis=0)
        span = float(max(np.ptp(display_points, axis=0).max(), 1e-9))
        half = span * 0.55
        axis.set_xlim(center[0] - half, center[0] + half)
        axis.set_ylim(center[1] - half, center[1] + half)
        axis.set_zlim(center[2] - half, center[2] + half)

    def reset_view(_event=None) -> None:
        """Restore the initial camera orientation and full-root view."""
        axis.view_init(elev=22.0, azim=-60.0)
        set_root_limits()
        figure.canvas.draw_idle()

    def show_cloud(cloud: PointCloudData) -> None:
        nonlocal display_points, instruction_artist
        display_points = _display_sample(cloud.points, max_display_points)
        axis.cla()
        axis.scatter(
            display_points[:, 0],
            display_points[:, 1],
            display_points[:, 2],
            s=1.2,
            c="#6b7280",
            alpha=0.62,
            depthshade=False,
        )
        _configure_selection_axis(axis, display_points)
        axis.disable_mouse_rotation()
        axis.view_init(elev=22.0, azim=-60.0)
        set_root_limits()
        instruction_artist = axis.text2D(
            0.02,
            0.02,
            "Click: base then tip | Left-drag: rotate | Right-drag: pan | Wheel: zoom",
            transform=axis.transAxes,
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        clear_selection(clear_boxes=True)
        set_instruction("Root loaded. Select the base and tip, or enter their XYZ coordinates.")

    def update_markers() -> None:
        for artist in [*marker_artists, *label_artists]:
            try:
                artist.remove()
            except (AttributeError, ValueError):
                pass
        marker_artists.clear()
        label_artists.clear()
        for index, point in enumerate(selected):
            color = "#16a34a" if index == 0 else "#dc2626"
            label = "Base" if index == 0 else "Tip"
            marker_artists.append(
                axis.scatter([point[0]], [point[1]], [point[2]], s=78, c=color, edgecolors="white", linewidths=0.8)
            )
            label_artists.append(axis.text(point[0], point[1], point[2], f"  {label}", color=color, fontsize=10))
        figure.canvas.draw_idle()

    def populate_coordinate_boxes(points_to_show: list[np.ndarray]) -> None:
        if len(points_to_show) >= 1:
            for key, value in zip(("base_x", "base_y", "base_z"), points_to_show[0]):
                coordinate_boxes[key].set_val(f"{float(value):.8g}")
        if len(points_to_show) >= 2:
            for key, value in zip(("tip_x", "tip_y", "tip_z"), points_to_show[1]):
                coordinate_boxes[key].set_val(f"{float(value):.8g}")

    def read_coordinate_boxes() -> tuple[np.ndarray, np.ndarray] | None:
        raw_values = [box.text.strip() for box in coordinate_boxes.values()]
        if not any(raw_values):
            return None
        if not all(raw_values):
            raise ValueError("Fill all six Base and Tip XYZ coordinate boxes, or clear them and use clicks.")
        values = np.asarray([float(value.replace(",", "")) for value in raw_values], dtype=float)
        return _validate_endpoint_pair(values[:3], values[3:])

    def zoom(factor: float) -> None:
        if display_points is None:
            return
        _zoom_3d_axis(axis, factor, figure.canvas)

    def begin_load(_event=None) -> None:
        if input_path is None:
            set_instruction("This picker already has a point cloud. Mesh sample count applies when loading from a file.")
            return
        try:
            requested_samples = _parse_sample_count(sample_box.text)
        except ValueError as exc:
            set_instruction(str(exc))
            return
        with lock:
            if bool(state["loading"]):
                return
            state.update(
                cloud=None,
                loading=True,
                pending_cloud=None,
                error=None,
                stage="Starting mesh load",
                fraction=0.01,
                eta_seconds=None,
                eta_started=None,
                sample_points=requested_samples,
            )
        clear_selection(clear_boxes=True)
        set_instruction(f"Loading {requested_samples:,} mesh samples in the background...")

        def report(stage: str, fraction: float, eta_seconds: float | None) -> None:
            with lock:
                state["stage"] = stage
                state["fraction"] = fraction
                state["eta_seconds"] = eta_seconds
                state["eta_started"] = time.monotonic() if eta_seconds is not None else None

        def worker() -> None:
            try:
                cloud = load_root_geometry_with_progress(
                    input_path,
                    requested_samples,
                    progress_callback=report,
                    random_seed=random_seed,
                )
            except Exception as exc:  # Captured for display on the GUI thread.
                with lock:
                    state.update(loading=False, error=str(exc), stage="Load failed", fraction=0.0, eta_seconds=None)
                return
            with lock:
                state.update(
                    loading=False,
                    pending_cloud=cloud,
                    stage="Point cloud ready",
                    fraction=1.0,
                    eta_seconds=0.0,
                    eta_started=None,
                )

        threading.Thread(target=worker, name="soyrootbio-mesh-loader", daemon=True).start()

    def apply_coordinates(_event=None) -> None:
        try:
            endpoints = read_coordinate_boxes()
        except ValueError as exc:
            set_instruction(str(exc))
            return
        if endpoints is None:
            set_instruction("Enter all six XYZ coordinate values before using them.")
            return
        selected[:] = [endpoints[0].copy(), endpoints[1].copy()]
        update_markers()
        set_instruction("Using typed Base and Tip coordinates. Choose Save when ready.")

    def select_point_at_event(event) -> None:
        with lock:
            loading = bool(state["loading"])
        if event.inaxes is not axis or event.button != 1 or loading or display_points is None or len(selected) >= 2:
            return
        figure.canvas.draw()
        projected_x, projected_y, _ = proj3d.proj_transform(
            display_points[:, 0],
            display_points[:, 1],
            display_points[:, 2],
            axis.get_proj(),
        )
        screen = axis.transData.transform(np.column_stack([projected_x, projected_y]))
        distance = np.hypot(screen[:, 0] - event.x, screen[:, 1] - event.y)
        selected.append(display_points[int(np.argmin(distance))].copy())
        populate_coordinate_boxes(selected)
        update_markers()
        if len(selected) == 1:
            set_instruction("Base selected. Left-click the primary-root tip, or type the Tip XYZ coordinates.")
        else:
            set_instruction("Base and tip selected. You can refine their XYZ fields, zoom in, then Save.")

    def on_pointer_press(event) -> None:
        """Start a direct manipulation gesture inside the root view."""
        if event.inaxes is not axis or event.button not in (1, 2, 3):
            return
        if event.x is None or event.y is None:
            return
        drag_state.update(
            active=True,
            button=event.button,
            start_x=float(event.x),
            start_y=float(event.y),
            start_elev=float(axis.elev),
            start_azim=float(axis.azim),
            start_xlim=axis.get_xlim(),
            start_ylim=axis.get_ylim(),
            start_zlim=axis.get_zlim(),
            dragged=False,
        )

    def on_pointer_motion(event) -> None:
        if not bool(drag_state["active"]) or event.inaxes is not axis:
            return
        if event.x is None or event.y is None:
            return
        delta_x = float(event.x) - float(drag_state["start_x"])
        delta_y = float(event.y) - float(drag_state["start_y"])
        if abs(delta_x) > 3.0 or abs(delta_y) > 3.0:
            drag_state["dragged"] = True

        if drag_state["button"] == 1:
            elevation, azimuth = _inverted_drag_rotation(
                float(drag_state["start_elev"]),
                float(drag_state["start_azim"]),
                delta_x,
                delta_y,
            )
            axis.view_init(elev=elevation, azim=azimuth)
        else:
            start_xlim = drag_state["start_xlim"]
            start_ylim = drag_state["start_ylim"]
            start_zlim = drag_state["start_zlim"]
            if start_xlim is None or start_ylim is None or start_zlim is None:
                return
            shift = _camera_pan_shift(
                -delta_x,
                -delta_y,
                max(float(axis.bbox.width), 1.0),
                max(float(axis.bbox.height), 1.0),
                float(drag_state["start_azim"]),
                float(drag_state["start_elev"]),
                np.asarray([start_xlim, start_ylim, start_zlim], dtype=float),
            )
            axis.set_xlim(np.asarray(start_xlim, dtype=float) + shift[0])
            axis.set_ylim(np.asarray(start_ylim, dtype=float) + shift[1])
            axis.set_zlim(np.asarray(start_zlim, dtype=float) + shift[2])
        figure.canvas.draw_idle()

    def on_pointer_release(event) -> None:
        if not bool(drag_state["active"]):
            return
        is_click = drag_state["button"] == 1 and not bool(drag_state["dragged"])
        drag_state["active"] = False
        if is_click:
            select_point_at_event(event)

    def on_scroll(event) -> None:
        if event.inaxes is not axis:
            return
        factor = _wheel_zoom_factor(event.button, getattr(event, "step", None))
        if factor is not None:
            zoom(factor)

    def reset_selection(_event=None) -> None:
        clear_selection(clear_boxes=True)
        set_instruction("Selections cleared. Left-click the base and tip, or enter XYZ coordinates.")

    def save(_event=None) -> None:
        with lock:
            cloud = state["cloud"]
            loading = bool(state["loading"])
        if cloud is None or loading:
            set_instruction("Wait for the root point cloud to finish loading before saving endpoints.")
            return
        try:
            endpoints = read_coordinate_boxes()
        except ValueError as exc:
            set_instruction(str(exc))
            return
        if endpoints is None:
            if len(selected) != 2:
                set_instruction("Select two endpoints or provide all six XYZ coordinates before saving.")
                return
            endpoints = _validate_endpoint_pair(selected[0], selected[1])
        with lock:
            state["result"] = endpoints
            state["saved"] = True
        plt.close(figure)

    def refresh_progress() -> None:
        nonlocal instruction_artist
        with lock:
            pending_cloud = state["pending_cloud"]
            if pending_cloud is not None:
                state["cloud"] = pending_cloud
                state["pending_cloud"] = None
            cloud = state["cloud"]
            loading = bool(state["loading"])
            stage = str(state["stage"])
            fraction = float(state["fraction"])
            eta_seconds = state["eta_seconds"]
            eta_started = state["eta_started"]
            error = state["error"]
        if pending_cloud is not None:
            show_cloud(pending_cloud)
        if error:
            progress_text.set_text(f"Load failed: {error}")
            progress_value.set_width(0.0)
        else:
            remaining_eta = eta_seconds
            if loading and isinstance(eta_seconds, (float, int)) and eta_seconds > 0 and eta_started is not None:
                elapsed = time.monotonic() - float(eta_started)
                remaining_eta = max(float(eta_seconds) - elapsed, 0.0)
                fraction = max(fraction, min(0.95, 0.25 + 0.70 * elapsed / max(float(eta_seconds), 1e-6)))
            progress_value.set_width(float(np.clip(fraction, 0.0, 1.0)))
            eta_label = _format_eta(remaining_eta)
            progress_text.set_text(f"{stage}: {fraction * 100:4.0f}% | {eta_label}")
        if cloud is None and not loading and not error:
            status_text.set_text("Set mesh samples and choose Load / reload root.")
        figure.canvas.draw_idle()

    reload_button.on_clicked(begin_load)
    zoom_in_button.on_clicked(lambda _event: zoom(0.75))
    zoom_out_button.on_clicked(lambda _event: zoom(1.35))
    reset_view_button.on_clicked(reset_view)
    use_coordinates_button.on_clicked(apply_coordinates)
    reset_selection_button.on_clicked(reset_selection)
    save_button.on_clicked(save)
    figure.canvas.mpl_connect("button_press_event", on_pointer_press)
    figure.canvas.mpl_connect("motion_notify_event", on_pointer_motion)
    figure.canvas.mpl_connect("button_release_event", on_pointer_release)
    figure.canvas.mpl_connect("scroll_event", on_scroll)
    timer = figure.canvas.new_timer(interval=200)
    timer.add_callback(refresh_progress)
    timer.start()

    if initial_cloud is not None:
        show_cloud(initial_cloud)
    elif input_path is not None:
        begin_load()
    else:
        set_instruction("No root point cloud is available.")
    plt.show(block=True)
    timer.stop()

    with lock:
        cloud = state["cloud"]
        saved = bool(state["saved"])
        result = state["result"]
        effective_sample_points = int(state["sample_points"])
    if not saved or result is None:
        raise RuntimeError("Endpoint selection was closed before two endpoints were saved.")
    start, end = result
    return cloud, start, end, effective_sample_points


def _camera_pan_shift(
    delta_x: float,
    delta_y: float,
    view_width: float,
    view_height: float,
    azimuth_degrees: float,
    elevation_degrees: float,
    limits: np.ndarray,
) -> np.ndarray:
    """Return a world-coordinate shift for a camera pan.

    The calculation uses the displayed camera's horizontal and vertical axes,
    keeping panning intuitive after the user has rotated the root view.
    """
    limits = np.asarray(limits, dtype=float)
    if limits.shape != (3, 2):
        raise ValueError("View limits must contain lower and upper values for X, Y, and Z.")
    azimuth = np.deg2rad(float(azimuth_degrees))
    elevation = np.deg2rad(float(elevation_degrees))
    span = max(float(np.ptp(limits, axis=1).max()), 1e-12)
    horizontal = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    vertical = np.array(
        [
            -np.cos(azimuth) * np.sin(elevation),
            -np.sin(azimuth) * np.sin(elevation),
            np.cos(elevation),
        ]
    )
    return span * (
        float(delta_x) / max(float(view_width), 1.0) * horizontal
        + float(delta_y) / max(float(view_height), 1.0) * vertical
    )


def _display_sample(points: np.ndarray, max_points: int) -> np.ndarray:
    max_points = max(2, int(max_points))
    if len(points) <= max_points:
        return points.copy()
    rng = np.random.default_rng(42)
    indices = np.sort(rng.choice(len(points), size=max_points, replace=False))
    return points[indices]


def _parse_sample_count(text: str) -> int:
    try:
        count = int(float(str(text).strip().replace(",", "")))
    except ValueError as exc:
        raise ValueError("Mesh samples must be a whole number of at least 10.") from exc
    if count < 10:
        raise ValueError("Mesh samples must be at least 10.")
    return count


def _format_eta(seconds: float | int | None) -> str:
    if seconds is None:
        return "ETA calibrating"
    seconds = max(0, int(round(float(seconds))))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"ETA {hours:d}:{minutes:02d}:{seconds:02d}"
    return f"ETA {minutes:02d}:{seconds:02d}"


def _validate_endpoint_pair(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = _validate_endpoint(start, "Base")
    end = _validate_endpoint(end, "Tip")
    if np.linalg.norm(start - end) <= 1e-12:
        raise ValueError("Primary-root endpoints must be distinct.")
    return start, end


def _validate_endpoint(point: np.ndarray, name: str) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} endpoint must contain three finite coordinates.")
    return point


def select_primary_guidance_from_file_gui(
    path: str | Path,
    sample_points: int | None = None,
    max_display_points: int = 30000,
    title: str = "Select primary-root guidance",
    random_seed: int | None = 42,
) -> tuple[PointCloudData, PrimaryGuidance, int]:
    """Pick endpoints, optional primary sections, and a horizontal soil line."""

    effective_samples = 0 if sample_points is None else int(sample_points)
    cloud, start, end, loaded_samples = select_primary_endpoints_from_file_gui(
        path,
        sample_points=max(10, effective_samples) if effective_samples else 50000,
        max_display_points=max_display_points,
        title=title,
        random_seed=random_seed,
    )
    soil_z, guides = select_primary_sections_gui(
        cloud.points,
        start=start,
        end=end,
        max_display_points=max_display_points,
        title=f"{title} — soil line and primary sections",
        random_seed=random_seed,
    )
    return cloud, PrimaryGuidance(start, end, soil_z, guides, True), loaded_samples


def select_soil_guidance_from_file_gui(
    path: str | Path,
    sample_points: int | None = None,
    max_display_points: int = 30000,
    title: str = "Select soil line and primary-root sections",
    random_seed: int | None = 42,
) -> tuple[PointCloudData, PrimaryGuidance, int]:
    """Select a soil line/sections while leaving endpoints to the scorer."""

    effective_samples = 0 if sample_points is None else int(sample_points)
    display_sample_count = max(10, effective_samples) if effective_samples else 50000
    cloud = load_root_geometry_with_progress(
        path,
        sample_points=display_sample_count,
        random_seed=random_seed,
    )
    heights = cloud.points[:, 2]
    start = cloud.points[int(np.argmax(heights))].copy()
    end = cloud.points[int(np.argmin(heights))].copy()
    soil_z, guides = select_primary_sections_gui(
        cloud.points,
        start=start,
        end=end,
        max_display_points=max_display_points,
        title=title,
        random_seed=random_seed,
    )
    if soil_z is None:
        raise ValueError("Soil-guided scoring requires a selected soil line.")
    return cloud, PrimaryGuidance(start, end, soil_z, guides, False), display_sample_count


def select_primary_sections_gui(
    points: np.ndarray,
    *,
    start: np.ndarray,
    end: np.ndarray,
    max_display_points: int = 30000,
    title: str = "Select soil line and primary-root sections",
    random_seed: int | None = 42,
) -> tuple[float | None, np.ndarray]:
    """Select any number of primary guide sections and one soil-line height.

    Shift-clicking chooses the nearest visible cloud point. In ``Guide
    section`` mode every click adds an ordered constraint; in ``Soil line``
    mode the point's Z coordinate defines the horizontal soil surface. The
    camera controls deliberately match the endpoint picker.
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons
        from mpl_toolkits.mplot3d import proj3d
    except ImportError as exc:
        raise ImportError("matplotlib is required for primary-section selection") from exc

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("Primary-section selection requires XYZ points")
    rng = np.random.default_rng(random_seed)
    display_indices = (
        rng.choice(len(points), size=max_display_points, replace=False)
        if len(points) > max_display_points
        else np.arange(len(points))
    )
    display = points[display_indices]
    state: dict[str, object] = {
        "mode": "Guide section",
        "guides": [],
        "soil_z": None,
        "saved": False,
    }
    drag_state: dict[str, object] = {
        "active": False,
        "button": None,
        "start_x": 0.0,
        "start_y": 0.0,
        "start_elev": 22.0,
        "start_azim": -60.0,
        "start_xlim": None,
        "start_ylim": None,
        "start_zlim": None,
        "dragged": False,
        "shift_selection": False,
    }

    figure = plt.figure(figsize=(15, 10))
    _maximize_figure_window(figure)
    figure.canvas.manager.set_window_title(title)
    axis = figure.add_axes([0.045, 0.095, 0.665, 0.80], projection="3d")
    figure.text(0.045, 0.952, title, fontsize=16, weight="bold")
    status_text = figure.text(0.045, 0.921, "Preparing primary guidance...", fontsize=11)

    control_left = 0.765
    control_width = 0.20
    figure.text(control_left, 0.835, "Selection mode", fontsize=12, weight="bold")
    radio = RadioButtons(
        figure.add_axes([control_left, 0.675, control_width, 0.13]),
        ("Guide section", "Soil line"),
    )
    figure.text(
        control_left,
        0.625,
        "Shift-click to select a visible point.\nGuides constrain the primary path.",
        fontsize=9,
        va="top",
        color="#4b5563",
    )
    undo_button = Button(
        figure.add_axes([control_left, 0.515, control_width, 0.052]),
        "Undo guide",
    )
    clear_soil_button = Button(
        figure.add_axes([control_left, 0.452, control_width, 0.052]),
        "Clear soil line",
    )
    zoom_in_button = Button(
        figure.add_axes([control_left, 0.372, 0.092, 0.050]),
        "Zoom +",
    )
    zoom_out_button = Button(
        figure.add_axes([control_left + 0.108, 0.372, 0.092, 0.050]),
        "Zoom -",
    )
    reset_view_button = Button(
        figure.add_axes([control_left, 0.309, control_width, 0.050]),
        "Reset view",
    )
    skip_button = Button(
        figure.add_axes([control_left, 0.135, control_width, 0.052]),
        "No guides / soil",
    )
    save_button = Button(
        figure.add_axes([control_left, 0.065, control_width, 0.055]),
        "Save guidance",
        color="#dbeafe",
        hovercolor="#bfdbfe",
    )
    for label in radio.labels:
        label.set_fontsize(10)
    for button in (
        undo_button,
        clear_soil_button,
        zoom_in_button,
        zoom_out_button,
        reset_view_button,
        skip_button,
        save_button,
    ):
        button.label.set_fontsize(10)

    def set_root_limits() -> None:
        center = display.mean(axis=0)
        span = float(max(np.ptp(display, axis=0).max(), 1e-9))
        half = span * 0.55
        axis.set_xlim(center[0] - half, center[0] + half)
        axis.set_ylim(center[1] - half, center[1] + half)
        axis.set_zlim(center[2] - half, center[2] + half)

    def redraw(*, preserve_view: bool = True) -> None:
        view_limits = None
        view_angles = None
        if preserve_view and axis.has_data():
            view_limits = (axis.get_xlim(), axis.get_ylim(), axis.get_zlim())
            view_angles = (axis.elev, axis.azim)
        axis.clear()
        axis.scatter(
            display[:, 0],
            display[:, 1],
            display[:, 2],
            s=1.2,
            c="#6b7280",
            alpha=0.62,
            depthshade=False,
            linewidths=0,
        )
        axis.scatter(
            [start[0]],
            [start[1]],
            [start[2]],
            s=78,
            c="#16a34a",
            edgecolors="white",
            linewidths=0.8,
            depthshade=False,
            marker="o",
            label="Base",
        )
        axis.scatter(
            [end[0]],
            [end[1]],
            [end[2]],
            s=78,
            c="#dc2626",
            edgecolors="white",
            linewidths=0.8,
            depthshade=False,
            marker="^",
            label="Tip",
        )
        guides = state["guides"]
        if guides:
            guide_array = np.asarray(guides)
            axis.scatter(
                guide_array[:, 0],
                guide_array[:, 1],
                guide_array[:, 2],
                s=70,
                c="#7c3aed",
                edgecolors="white",
                linewidths=0.8,
                depthshade=False,
                marker="D",
                label="Guide section",
            )
            for index, point in enumerate(guide_array, start=1):
                axis.text(
                    point[0],
                    point[1],
                    point[2],
                    f"  G{index}",
                    fontsize=9,
                    color="#6d28d9",
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                    },
                )
        soil_z = state["soil_z"]
        if soil_z is not None:
            x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
            y0, y1 = float(points[:, 1].min()), float(points[:, 1].max())
            axis.plot(
                [x0, x1, x1, x0, x0],
                [y0, y0, y1, y1, y0],
                [soil_z] * 5,
                c="#15803d",
                lw=2.4,
                alpha=0.95,
                label="Soil line",
            )
            axis.text(
                x0,
                y0,
                soil_z,
                f"  soil z={soil_z:.6g}",
                color="#15803d",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
            )
        _configure_selection_axis(axis, points)
        axis.disable_mouse_rotation()
        if view_limits is not None:
            axis.set_xlim(view_limits[0])
            axis.set_ylim(view_limits[1])
            axis.set_zlim(view_limits[2])
            axis.view_init(elev=view_angles[0], azim=view_angles[1])
        else:
            axis.view_init(elev=22.0, azim=-60.0)
            set_root_limits()
        axis.legend(loc="upper right", framealpha=0.92, fontsize=9)
        axis.text2D(
            0.02,
            0.02,
            "Shift-click: select | Left-drag: rotate | Right-drag: pan | Wheel: zoom",
            transform=axis.transAxes,
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
        )
        soil_text = "not selected" if soil_z is None else f"z={float(soil_z):.6g}"
        status_text.set_text(
            f"{state['mode']} mode | {len(guides)} guide section(s) | Soil line {soil_text}"
        )
        figure.canvas.draw_idle()

    def on_mode(label: str) -> None:
        state["mode"] = label
        redraw()

    def select_point_at_event(event) -> None:
        if event.inaxes is not axis or event.x is None or event.y is None:
            return
        figure.canvas.draw()
        projected_x, projected_y, _ = proj3d.proj_transform(
            display[:, 0], display[:, 1], display[:, 2], axis.get_proj()
        )
        screen = axis.transData.transform(np.column_stack([projected_x, projected_y]))
        distance = np.linalg.norm(screen - np.array([event.x, event.y]), axis=1)
        selected = display[int(np.argmin(distance))].copy()
        if state["mode"] == "Soil line":
            state["soil_z"] = float(selected[2])
        else:
            state["guides"].append(selected)
        redraw()

    def on_pointer_press(event) -> None:
        if event.inaxes is not axis or event.button not in (1, 2, 3):
            return
        if event.x is None or event.y is None:
            return
        drag_state.update(
            active=True,
            button=event.button,
            start_x=float(event.x),
            start_y=float(event.y),
            start_elev=float(axis.elev),
            start_azim=float(axis.azim),
            start_xlim=axis.get_xlim(),
            start_ylim=axis.get_ylim(),
            start_zlim=axis.get_zlim(),
            dragged=False,
            shift_selection=(event.button == 1 and "shift" in str(event.key).lower()),
        )

    def on_pointer_motion(event) -> None:
        if not bool(drag_state["active"]) or event.inaxes is not axis:
            return
        if event.x is None or event.y is None:
            return
        delta_x = float(event.x) - float(drag_state["start_x"])
        delta_y = float(event.y) - float(drag_state["start_y"])
        if abs(delta_x) > 3.0 or abs(delta_y) > 3.0:
            drag_state["dragged"] = True

        if drag_state["button"] == 1:
            elevation, azimuth = _inverted_drag_rotation(
                float(drag_state["start_elev"]),
                float(drag_state["start_azim"]),
                delta_x,
                delta_y,
            )
            axis.view_init(elev=elevation, azim=azimuth)
        else:
            start_xlim = drag_state["start_xlim"]
            start_ylim = drag_state["start_ylim"]
            start_zlim = drag_state["start_zlim"]
            if start_xlim is None or start_ylim is None or start_zlim is None:
                return
            shift = _camera_pan_shift(
                -delta_x,
                -delta_y,
                max(float(axis.bbox.width), 1.0),
                max(float(axis.bbox.height), 1.0),
                float(drag_state["start_azim"]),
                float(drag_state["start_elev"]),
                np.asarray([start_xlim, start_ylim, start_zlim], dtype=float),
            )
            axis.set_xlim(np.asarray(start_xlim, dtype=float) + shift[0])
            axis.set_ylim(np.asarray(start_ylim, dtype=float) + shift[1])
            axis.set_zlim(np.asarray(start_zlim, dtype=float) + shift[2])
        figure.canvas.draw_idle()

    def on_pointer_release(event) -> None:
        if not bool(drag_state["active"]):
            return
        select_point = bool(drag_state["shift_selection"]) and not bool(drag_state["dragged"])
        drag_state["active"] = False
        if select_point:
            select_point_at_event(event)

    def on_scroll(event) -> None:
        if event.inaxes is not axis:
            return
        factor = _wheel_zoom_factor(event.button, getattr(event, "step", None))
        if factor is not None:
            _zoom_3d_axis(axis, factor, figure.canvas)

    def reset_view(_event=None) -> None:
        axis.view_init(elev=22.0, azim=-60.0)
        set_root_limits()
        figure.canvas.draw_idle()

    def undo(_event) -> None:
        guides = state["guides"]
        if guides:
            guides.pop()
            redraw()

    def clear_soil(_event) -> None:
        state["soil_z"] = None
        redraw()

    def save(_event) -> None:
        state["saved"] = True
        plt.close(figure)

    def skip(_event) -> None:
        state["guides"] = []
        state["soil_z"] = None
        state["saved"] = True
        plt.close(figure)

    redraw(preserve_view=False)
    radio.on_clicked(on_mode)
    undo_button.on_clicked(undo)
    clear_soil_button.on_clicked(clear_soil)
    zoom_in_button.on_clicked(lambda _event: _zoom_3d_axis(axis, 0.75, figure.canvas))
    zoom_out_button.on_clicked(lambda _event: _zoom_3d_axis(axis, 1.35, figure.canvas))
    reset_view_button.on_clicked(reset_view)
    save_button.on_clicked(save)
    skip_button.on_clicked(skip)
    figure.canvas.mpl_connect("button_press_event", on_pointer_press)
    figure.canvas.mpl_connect("motion_notify_event", on_pointer_motion)
    figure.canvas.mpl_connect("button_release_event", on_pointer_release)
    figure.canvas.mpl_connect("scroll_event", on_scroll)
    plt.show(block=True)
    if not state["saved"]:
        raise RuntimeError("Primary guidance selection was closed before saving")
    guides = np.asarray(state["guides"], dtype=float)
    if not len(guides):
        guides = np.empty((0, 3), dtype=float)
    return state["soil_z"], guides
