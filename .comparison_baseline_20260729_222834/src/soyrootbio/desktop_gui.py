from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import time

import numpy as np

from .batch import write_processing_error_log
from .endpoint_picker import select_primary_endpoints_from_file_gui
from .pipeline import AnalysisCancelled, PipelineConfig, PipelineResult, run_pipeline
from .types import PointCloudData


SUPPORTED_INPUTS = {".ply", ".obj", ".stl", ".xyz", ".csv"}
ENDPOINT_MODES = {"interactive", "coordinates", "auto"}


@dataclass(frozen=True)
class LauncherSettings:
    input_path: Path
    output_dir: Path
    sample_points: int
    display_points: int
    endpoint_mode: str = "interactive"
    start: tuple[float, float, float] | None = None
    end: tuple[float, float, float] | None = None
    auto_endpoints: str | None = None


def validate_launcher_settings(
    input_path: str | Path,
    output_dir: str | Path,
    sample_points: str | int,
    display_points: str | int,
    endpoint_mode: str = "interactive",
    start_coordinates: tuple[object, object, object] | list[object] | None = None,
    end_coordinates: tuple[object, object, object] | list[object] | None = None,
) -> LauncherSettings:
    """Validate desktop form values without requiring a GUI display."""
    source = Path(str(input_path).strip()).expanduser()
    destination = Path(str(output_dir).strip()).expanduser()
    if not source.is_file():
        raise ValueError("Choose an existing root point-cloud or mesh file.")
    if source.suffix.lower() not in SUPPORTED_INPUTS:
        raise ValueError("Input must be a PLY, OBJ, STL, XYZ, or CSV file.")
    if not str(output_dir).strip():
        raise ValueError("Choose an output directory.")
    if destination.exists() and not destination.is_dir():
        raise ValueError("The output location must be a directory, not a file.")
    samples = _parse_positive_integer(sample_points, "Mesh samples", minimum=10)
    displayed = _parse_positive_integer(display_points, "Display points", minimum=2)

    mode = str(endpoint_mode).strip().lower()
    if mode not in ENDPOINT_MODES:
        raise ValueError("Endpoint mode must be interactive, coordinates, or auto.")

    start: tuple[float, float, float] | None = None
    end: tuple[float, float, float] | None = None
    auto_endpoints: str | None = None
    if mode == "coordinates":
        start = _parse_coordinate_triplet(start_coordinates, "Base")
        end = _parse_coordinate_triplet(end_coordinates, "Tip")
        if np.linalg.norm(np.asarray(start) - np.asarray(end)) <= 1e-12:
            raise ValueError("Base and tip endpoint coordinates must be distinct.")
    elif mode == "auto":
        auto_endpoints = "z"

    return LauncherSettings(
        source,
        destination,
        samples,
        displayed,
        endpoint_mode=mode,
        start=start,
        end=end,
        auto_endpoints=auto_endpoints,
    )


def _parse_positive_integer(value: str | int, name: str, minimum: int) -> int:
    try:
        parsed = int(float(str(value).strip().replace(",", "")))
    except ValueError as exc:
        raise ValueError(f"{name} must be a whole number of at least {minimum}.") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return parsed


def _parse_coordinate_triplet(
    values: tuple[object, object, object] | list[object] | None,
    name: str,
) -> tuple[float, float, float]:
    if values is None or len(values) != 3 or any(not str(value).strip() for value in values):
        raise ValueError(f"Enter all three {name} XYZ coordinates.")
    try:
        parsed = tuple(float(str(value).strip().replace(",", "")) for value in values)
    except ValueError as exc:
        raise ValueError(f"{name} XYZ coordinates must be numbers.") from exc
    if not np.all(np.isfinite(parsed)):
        raise ValueError(f"{name} XYZ coordinates must be finite numbers.")
    return parsed


def _project_root() -> Path:
    """Return the repository root for either merged or isolated-candidate use."""
    source_root = Path(__file__).resolve().parents[2]
    if source_root.name.startswith("merge_candidate_"):
        return source_root.parent
    return source_root


def estimate_remaining_seconds(elapsed_seconds: float, completed_fraction: float) -> float | None:
    """Estimate remaining duration from elapsed time and measured progress."""
    if not np.isfinite(elapsed_seconds) or not np.isfinite(completed_fraction):
        return None
    if elapsed_seconds < 1.0 or completed_fraction <= 0.02:
        return None
    if completed_fraction >= 1.0:
        return 0.0
    return max(0.0, float(elapsed_seconds) * (1.0 - float(completed_fraction)) / float(completed_fraction))


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "ETA estimating..."
    rounded = max(0, int(round(seconds)))
    minutes, remaining_seconds = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"ETA {hours:d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"ETA {minutes:02d}:{remaining_seconds:02d}"


class _QueueLogHandler(logging.Handler):
    def __init__(self, messages: queue.Queue[tuple[str, object]]) -> None:
        super().__init__()
        self.messages = messages
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.put(("log", self.format(record)))


class BioInsAlgoDesktopApp:
    def __init__(
        self,
        root: tk.Tk,
        initial_input: str | Path | None = None,
        initial_output: str | Path | None = None,
    ) -> None:
        self.root = root
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.running = False
        self.closed = False
        self.cancel_event = threading.Event()
        self.close_when_stopped = False
        self.analysis_started_at: float | None = None
        self.latest_progress = 0.0
        self.smoothed_eta: float | None = None

        self.input_var = tk.StringVar(value=str(initial_input or ""))
        self.output_var = tk.StringVar(value=str(initial_output or ""))
        self.samples_var = tk.StringVar(value="50000")
        self.display_var = tk.StringVar(value="30000")
        self.endpoint_mode_var = tk.StringVar(value="interactive")
        self.start_coordinate_vars = [tk.StringVar(value="") for _ in range(3)]
        self.end_coordinate_vars = [tk.StringVar(value="") for _ in range(3)]
        self.status_var = tk.StringVar(value="Choose a root file and output directory.")
        self.eta_var = tk.StringVar(value="ETA --:--")

        self._configure_window()
        self._build_layout()
        self._update_endpoint_mode_controls()
        if self.input_var.get() and not self.output_var.get():
            self._set_default_output(Path(self.input_var.get()))
        self.root.after(150, self._process_messages)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _configure_window(self) -> None:
        self.root.title("BioInsAlgo - Soybean Root Analysis")
        self.root.geometry("940x760")
        self.root.minsize(800, 660)
        try:
            self.root.tk.call("tk", "scaling", 1.15)
        except tk.TclError:
            pass
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#4b5563")
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 8))
        style.configure("TButton", padding=(10, 6))
        style.configure("TEntry", padding=5)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=(28, 22, 28, 20))
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(12, weight=1)

        ttk.Label(outer, text="BioInsAlgo", style="Title.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(
            outer,
            text="Desktop interface for the branch's original analysis workflow",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 20))

        ttk.Label(outer, text="Input root file").grid(row=2, column=0, sticky="w", padx=(0, 14), pady=7)
        self.input_entry = ttk.Entry(outer, textvariable=self.input_var)
        self.input_entry.grid(row=2, column=1, sticky="ew", pady=7)
        self.input_button = ttk.Button(outer, text="Browse...", command=self._browse_input)
        self.input_button.grid(row=2, column=2, padx=(10, 0), pady=7)

        ttk.Label(outer, text="Output directory").grid(row=3, column=0, sticky="w", padx=(0, 14), pady=7)
        self.output_entry = ttk.Entry(outer, textvariable=self.output_var)
        self.output_entry.grid(row=3, column=1, sticky="ew", pady=7)
        self.output_button = ttk.Button(outer, text="Browse...", command=self._browse_output)
        self.output_button.grid(row=3, column=2, padx=(10, 0), pady=7)

        ttk.Separator(outer).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(14, 12))

        options = ttk.Frame(outer)
        options.grid(row=5, column=0, columnspan=3, sticky="ew")
        for column in (1, 3):
            options.columnconfigure(column, weight=1)

        ttk.Label(options, text="Mesh samples").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=6)
        self.samples_entry = ttk.Entry(options, textvariable=self.samples_var, width=16)
        self.samples_entry.grid(row=0, column=1, sticky="ew", padx=(0, 28), pady=6)
        ttk.Label(options, text="Display points").grid(row=0, column=2, sticky="w", padx=(0, 10), pady=6)
        self.display_entry = ttk.Entry(options, textvariable=self.display_var, width=16)
        self.display_entry.grid(row=0, column=3, sticky="ew", pady=6)

        endpoint_frame = ttk.LabelFrame(outer, text="Primary-root endpoints", padding=(12, 9))
        endpoint_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 10))
        endpoint_frame.columnconfigure(0, weight=1)
        endpoint_frame.columnconfigure(1, weight=1)
        endpoint_frame.columnconfigure(2, weight=1)

        self.endpoint_mode_buttons = [
            ttk.Radiobutton(
                endpoint_frame,
                text="Select two points in 3D",
                value="interactive",
                variable=self.endpoint_mode_var,
                command=self._update_endpoint_mode_controls,
            ),
            ttk.Radiobutton(
                endpoint_frame,
                text="Enter XYZ coordinates",
                value="coordinates",
                variable=self.endpoint_mode_var,
                command=self._update_endpoint_mode_controls,
            ),
            ttk.Radiobutton(
                endpoint_frame,
                text="Automatic Z-axis extrema",
                value="auto",
                variable=self.endpoint_mode_var,
                command=self._update_endpoint_mode_controls,
            ),
        ]
        for column, button in enumerate(self.endpoint_mode_buttons):
            button.grid(row=0, column=column, sticky="w", padx=(0, 16), pady=(0, 9))

        coordinate_frame = ttk.Frame(endpoint_frame)
        coordinate_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(20, 24))
        for column in (1, 2, 3):
            coordinate_frame.columnconfigure(column, weight=1)
        ttk.Label(coordinate_frame, text="X").grid(row=0, column=1, padx=4)
        ttk.Label(coordinate_frame, text="Y").grid(row=0, column=2, padx=4)
        ttk.Label(coordinate_frame, text="Z").grid(row=0, column=3, padx=4)
        ttk.Label(coordinate_frame, text="Base").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Label(coordinate_frame, text="Tip").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        self.coordinate_entries = []
        for row, variables in ((1, self.start_coordinate_vars), (2, self.end_coordinate_vars)):
            for column, variable in enumerate(variables, start=1):
                entry = ttk.Entry(coordinate_frame, textvariable=variable, width=13)
                entry.grid(row=row, column=column, sticky="ew", padx=4, pady=3)
                self.coordinate_entries.append(entry)

        angle_frame = ttk.LabelFrame(outer, text="Branch-angle measurement", padding=(12, 9))
        angle_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        ttk.Label(angle_frame, text="Tip direction vs downward Z (0, 0, -1)").pack(side="left")

        controls = ttk.Frame(outer)
        controls.grid(row=8, column=0, columnspan=3, sticky="ew")
        self.run_button = ttk.Button(
            controls,
            text="Choose endpoints and run",
            style="Primary.TButton",
            command=self._start,
        )
        self.run_button.pack(side="left")
        self.close_button = ttk.Button(controls, text="Close", command=self._close)
        self.close_button.pack(side="right")

        self.progress = ttk.Progressbar(outer, mode="determinate", maximum=100)
        self.progress.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(18, 7))
        status_line = ttk.Frame(outer)
        status_line.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        status_line.columnconfigure(0, weight=1)
        ttk.Label(status_line, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(status_line, textvariable=self.eta_var, font=("Segoe UI", 10, "bold")).grid(
            row=0,
            column=1,
            sticky="e",
            padx=(18, 0),
        )

        ttk.Label(outer, text="Activity").grid(row=11, column=0, columnspan=3, sticky="w", pady=(8, 5))
        log_frame = ttk.Frame(outer)
        log_frame.grid(row=12, column=0, columnspan=3, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(
            log_frame,
            height=10,
            wrap="word",
            font=("Consolas", 9),
            background="#f8fafc",
            foreground="#1f2937",
            relief="solid",
            borderwidth=1,
            padx=9,
            pady=8,
            state="disabled",
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.form_controls = [
            self.input_entry,
            self.input_button,
            self.output_entry,
            self.output_button,
            self.samples_entry,
            self.display_entry,
            *self.endpoint_mode_buttons,
            *self.coordinate_entries,
        ]

    def _update_endpoint_mode_controls(self) -> None:
        if self.running:
            return
        mode = self.endpoint_mode_var.get()
        coordinate_state = "normal" if mode == "coordinates" else "disabled"
        for entry in self.coordinate_entries:
            entry.configure(state=coordinate_state)
        labels = {
            "interactive": "Choose endpoints and run",
            "coordinates": "Run with XYZ endpoints",
            "auto": "Auto-select endpoints and run",
        }
        self.run_button.configure(text=labels.get(mode, "Run analysis"))

    def _browse_input(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Choose root point cloud or mesh",
            filetypes=[
                ("Supported root files", "*.ply *.obj *.stl *.xyz *.csv"),
                ("PLY files", "*.ply"),
                ("Mesh files", "*.obj *.stl"),
                ("Point tables", "*.xyz *.csv"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.input_var.set(selected)
            if not self.output_var.get().strip():
                self._set_default_output(Path(selected))

    def _set_default_output(self, input_path: Path) -> None:
        self.output_var.set(str(_project_root() / "outputs" / f"{input_path.stem}_first_order"))

    def _browse_output(self) -> None:
        initial = self.output_var.get().strip()
        initial_dir = Path(initial).parent if initial else _project_root() / "outputs"
        selected = filedialog.askdirectory(
            parent=self.root,
            title="Choose output directory",
            initialdir=str(initial_dir),
            mustexist=False,
        )
        if selected:
            self.output_var.set(selected)

    def _start(self) -> None:
        try:
            settings = validate_launcher_settings(
                self.input_var.get(),
                self.output_var.get(),
                self.samples_var.get(),
                self.display_var.get(),
                endpoint_mode=self.endpoint_mode_var.get(),
                start_coordinates=[variable.get() for variable in self.start_coordinate_vars],
                end_coordinates=[variable.get() for variable in self.end_coordinate_vars],
            )
        except ValueError as exc:
            messagebox.showerror("Check settings", str(exc), parent=self.root)
            return
        if settings.output_dir.exists() and any(settings.output_dir.iterdir()):
            proceed = messagebox.askyesno(
                "Output directory is not empty",
                "Existing files with matching names may be replaced. Continue?",
                parent=self.root,
            )
            if not proceed:
                return

        self._set_busy(True)
        self.progress["value"] = 0
        self.eta_var.set("ETA starts after endpoint selection" if settings.endpoint_mode == "interactive" else "ETA estimating...")
        self.status_var.set(
            "Opening the 3D endpoint selector..."
            if settings.endpoint_mode == "interactive"
            else "Starting root analysis..."
        )
        self._append_log(f"Input: {settings.input_path}")
        self._append_log(f"Output: {settings.output_dir}")
        self._append_log(f"Endpoint mode: {settings.endpoint_mode}")
        self._append_log("Angle method: tip direction vs downward Z (0, 0, -1)")
        self.root.update_idletasks()

        cloud: PointCloudData | None = None
        start = settings.start
        end = settings.end
        effective_samples = settings.sample_points
        if settings.endpoint_mode == "interactive":
            self.root.withdraw()
            try:
                cloud, selected_start, selected_end, effective_samples = select_primary_endpoints_from_file_gui(
                    settings.input_path,
                    sample_points=settings.sample_points,
                    max_display_points=settings.display_points,
                    title=f"Select endpoints - {settings.input_path.name}",
                    random_seed=42,
                )
            except Exception as exc:
                self.root.deiconify()
                self.root.lift()
                self._set_busy(False)
                self.eta_var.set("ETA --:--")
                self.status_var.set("Endpoint selection was not completed.")
                messagebox.showerror("Endpoint selection", str(exc), parent=self.root)
                return
            self.root.deiconify()
            self.root.lift()
            start = tuple(float(value) for value in selected_start)
            end = tuple(float(value) for value in selected_end)

        settings.output_dir.mkdir(parents=True, exist_ok=True)
        config = PipelineConfig(
            input_path=settings.input_path,
            output_dir=settings.output_dir,
            start=start,
            end=end,
            auto_endpoints=settings.auto_endpoints,
            sample_points=effective_samples,
        )
        self.cancel_event = threading.Event()
        self.close_when_stopped = False
        self.analysis_started_at = time.monotonic()
        self.latest_progress = 0.0
        self.smoothed_eta = None
        self.eta_var.set("ETA estimating...")
        self.status_var.set("Starting root analysis...")
        thread = threading.Thread(
            target=self._run_worker,
            args=(config, cloud),
            name="soyrootbio-desktop-analysis",
            daemon=True,
        )
        thread.start()

    def _run_worker(self, config: PipelineConfig, cloud: PointCloudData | None) -> None:
        handler = _QueueLogHandler(self.messages)
        logger = logging.getLogger("soyrootbio")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            run_kwargs = {
                "progress_callback": lambda stage, fraction: self.messages.put(("progress", (stage, fraction))),
                "cancel_check": self.cancel_event.is_set,
            }
            if cloud is not None:
                run_kwargs["preloaded_cloud"] = cloud
            result = run_pipeline(config, **run_kwargs)
        except AnalysisCancelled:
            self.messages.put(("cancelled", None))
        except Exception as exc:
            error_log_path = write_processing_error_log(
                output_dir=config.output_dir,
                input_path=config.input_path,
                exception=exc,
                config=config,
            )
            if error_log_path is not None:
                self.messages.put(("log", f"Error log saved to {error_log_path}"))
            self.messages.put(("error", str(exc)))
        else:
            self.messages.put(("done", result))
        finally:
            logger.removeHandler(handler)

    def _process_messages(self) -> None:
        try:
            while True:
                message_type, payload = self.messages.get_nowait()
                if message_type == "log":
                    self._append_log(str(payload))
                elif message_type == "progress":
                    stage, fraction = payload
                    self.latest_progress = float(fraction)
                    self.progress["value"] = self.latest_progress * 100.0
                    self.status_var.set(str(stage))
                elif message_type == "done":
                    self._analysis_done(payload)
                elif message_type == "error":
                    self._analysis_failed(str(payload))
                elif message_type == "cancelled":
                    self._analysis_cancelled()
        except queue.Empty:
            pass
        if self.closed:
            return
        self._refresh_eta()
        self.root.after(150, self._process_messages)

    def _analysis_done(self, result: PipelineResult) -> None:
        if self.close_when_stopped:
            self.closed = True
            self.root.destroy()
            return
        self.progress["value"] = 100
        self.latest_progress = 1.0
        self.eta_var.set("ETA 00:00")
        self.status_var.set(
            f"Complete: {len(result.lateral_paths)} lateral roots from {result.lateral_start_count} starting points."
        )
        self._append_log(f"Complete. Outputs saved to {result.output_dir}")
        self._set_busy(False)
        messagebox.showinfo("BioInsAlgo complete", "Root analysis finished successfully.", parent=self.root)

    def _analysis_failed(self, error: str) -> None:
        if self.close_when_stopped:
            self.closed = True
            self.root.destroy()
            return
        self.progress["value"] = 0
        self.eta_var.set("ETA unavailable")
        self.status_var.set("Analysis failed. Review the activity message below.")
        self._append_log(f"ERROR: {error}")
        self._set_busy(False)
        messagebox.showerror("BioInsAlgo error", error, parent=self.root)

    def _analysis_cancelled(self) -> None:
        self.running = False
        self._append_log("Analysis cancelled by the user.")
        if self.close_when_stopped:
            self.closed = True
            self.root.destroy()
            return
        self.progress["value"] = 0
        self.eta_var.set("ETA --:--")
        self.status_var.set("Analysis cancelled.")
        self._set_busy(False)

    def _refresh_eta(self) -> None:
        if not self.running or self.analysis_started_at is None:
            return
        elapsed = time.monotonic() - self.analysis_started_at
        estimate = estimate_remaining_seconds(elapsed, self.latest_progress)
        if estimate is None:
            self.eta_var.set("ETA estimating...")
            return
        if self.smoothed_eta is None:
            self.smoothed_eta = estimate
        else:
            self.smoothed_eta = 0.82 * self.smoothed_eta + 0.18 * estimate
        self.eta_var.set(format_eta(self.smoothed_eta))

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        self.running = busy
        state = "disabled" if busy else "normal"
        for control in self.form_controls:
            control.configure(state=state)
        self.run_button.configure(state=state)
        self.close_button.configure(state="normal", text="Cancel and close" if busy else "Close")
        if not busy:
            self._update_endpoint_mode_controls()

    def _close(self) -> None:
        if self.running:
            should_cancel = messagebox.askyesno(
                "Cancel analysis?",
                "Stop the current analysis and close BioInsAlgo?",
                parent=self.root,
            )
            if not should_cancel:
                return
            self.close_when_stopped = True
            self.cancel_event.set()
            self.status_var.set("Cancelling analysis...")
            self.eta_var.set("Stopping...")
            self._append_log("Cancellation requested. Waiting for the active operation to stop safely...")
            self.close_button.configure(state="disabled", text="Cancelling...")
            return
        self.closed = True
        self.root.destroy()


def launch_gui(
    initial_input: str | Path | None = None,
    initial_output: str | Path | None = None,
) -> int:
    from .batch_gui import BioInsAlgoBatchApp, create_root

    root = create_root()
    BioInsAlgoBatchApp(root, initial_input=initial_input, initial_output=initial_output)
    root.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the BioInsAlgo desktop application.")
    parser.add_argument("--input", type=Path, help="Optional root file to prefill.")
    parser.add_argument("--output", type=Path, help="Optional output directory to prefill.")
    args = parser.parse_args(argv)
    return launch_gui(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
