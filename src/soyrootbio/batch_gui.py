from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Iterable

import numpy as np

from .batch import (
    BatchEventType,
    BatchJobState,
    BatchScheduler,
    StepTimingHistory,
    canonical_output_directory,
    unique_output_directory,
)
from .endpoint_picker import (
    PrimaryGuidance,
    select_primary_guidance_from_file_gui,
    select_soil_guidance_from_file_gui,
)
from .hardware import allocate_resources, detect_hardware, format_bytes
from .pipeline import PipelineConfig, run_pipeline
from .primary_guidance import read_primary_guidance


SUPPORTED_INPUTS = {".ply", ".stl", ".obj", ".xyz", ".csv"}
PRIMARY_METHODS = {
    "Scored automatic": "scored",
    "Z-axis extrema": "z",
    "Manual soil line + scorer": "soil",
    "Interactive endpoints + sections": "interactive",
    "XYZ endpoints": "coordinates",
}


@dataclass
class SampleEntry:
    item_id: str
    input_path: Path
    output_dir: Path
    job_id: str | None = None
    guidance: PrimaryGuidance | None = None
    correction_file: Path | None = None
    custom_output: bool = False


class BioInsAlgoBatchApp:
    """Batch-first Tk desktop application with per-sample cooperative controls."""

    def __init__(
        self,
        root: tk.Tk,
        initial_input: str | Path | None = None,
        initial_output: str | Path | None = None,
    ) -> None:
        self.root = root
        self.hardware = detect_hardware()
        self.entries: dict[str, SampleEntry] = {}
        self.scheduler: BatchScheduler | None = None
        self.job_to_item: dict[str, str] = {}

        self.output_root_var = tk.StringVar(value=str(initial_output or ""))
        self.primary_method_var = tk.StringVar(value="Scored automatic")
        self.sample_cap_var = tk.StringVar(value="0")
        self.display_points_var = tk.StringVar(value="30000")
        self.max_order_var = tk.StringVar(value="3")
        self.soil_z_var = tk.StringVar(value="")
        self.runtime_limit_var = tk.StringVar(value="30")
        self.minimum_fraction_var = tk.StringVar(value="25")
        self.tip_window_var = tk.StringVar(value="2.0")
        self.concurrency_var = tk.StringVar(value="Auto")
        self.threads_var = tk.StringVar(value="Auto")
        self.start_vars = [tk.StringVar(value="") for _ in range(3)]
        self.end_vars = [tk.StringVar(value="") for _ in range(3)]
        self.guide_text_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Drop STL/PLY files here or choose Add files.")
        self.hardware_var = tk.StringVar(value=self._hardware_text())

        self._configure_window()
        self._build_layout()
        self._register_drop_target()
        if initial_input:
            self.add_files([Path(initial_input)])
        self.root.after(120, self._poll_scheduler)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _configure_window(self) -> None:
        self.root.title("SoyRootBio — Soybean Root Architecture Analysis")
        self.root.geometry("1320x860")
        self.root.minsize(1080, 720)
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure(".", font=("Segoe UI", 9))
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 7))

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.grid(sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(4, weight=1)

        ttk.Label(outer, text="SoyRootBio", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(outer, textvariable=self.hardware_var, foreground="#4b5563").grid(row=1, column=0, sticky="w", pady=(1, 9))

        toolbar = ttk.Frame(outer)
        toolbar.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(toolbar, text="Add files", command=self.choose_files).pack(side="left")
        ttk.Button(toolbar, text="Remove selected", command=self.remove_selected).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Set selected output", command=self.set_selected_output).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Open output folder", command=self.open_output_folder).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Configure selected primary…", command=self.configure_selected_primary).pack(side="left", padx=5)

        # Keep selection import and output-root controls on a second row so
        # they remain visible at the application's minimum window width.
        output_toolbar = ttk.Frame(outer)
        output_toolbar.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(output_toolbar, text="Load endpoints + guides…", command=self.load_selected_guidance).pack(side="left")
        ttk.Label(output_toolbar, text="Output root:").pack(side="left", padx=(18, 4))
        ttk.Entry(output_toolbar, textvariable=self.output_root_var, width=38).pack(side="left", fill="x", expand=True)
        ttk.Button(output_toolbar, text="Browse", command=self.choose_output_root).pack(side="left", padx=(5, 0))

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.grid(row=4, column=0, sticky="nsew")
        queue_frame = ttk.Frame(body, padding=(0, 0, 8, 0))
        settings_frame = ttk.Frame(body, padding=(8, 0, 0, 0))
        body.add(queue_frame, weight=4)
        body.add(settings_frame, weight=2)
        queue_frame.rowconfigure(0, weight=1)
        queue_frame.columnconfigure(0, weight=1)

        columns = ("sample", "output", "primary", "status", "progress", "runtime")
        self.tree = ttk.Treeview(queue_frame, columns=columns, show="headings", selectmode="extended")
        headings = {
            "sample": "Sample",
            "output": "Output directory",
            "primary": "Primary method",
            "status": "Status",
            "progress": "Progress",
            "runtime": "Total run time",
        }
        widths = {"sample": 180, "output": 225, "primary": 180, "status": 95, "progress": 70, "runtime": 110}
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], minwidth=55, stretch=False)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical_scrollbar = ttk.Scrollbar(queue_frame, orient="vertical", command=self.tree.yview)
        vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal_scrollbar = ttk.Scrollbar(queue_frame, orient="horizontal", command=self.tree.xview)
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
        self.tree.configure(
            yscrollcommand=vertical_scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set,
        )
        self.tree.bind("<Double-1>", lambda _event: self.configure_selected_primary())

        self._build_settings(settings_frame)

        controls = ttk.Frame(outer)
        controls.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(controls, text="Start batch", style="Primary.TButton", command=self.start_batch).pack(side="left")
        ttk.Button(controls, text="Pause selected", command=self.pause_selected).pack(side="left", padx=5)
        ttk.Button(controls, text="Resume selected", command=self.resume_selected).pack(side="left", padx=5)
        ttk.Button(controls, text="Cancel selected", command=self.cancel_selected).pack(side="left", padx=5)
        ttk.Button(controls, text="Cancel all", command=self.cancel_all).pack(side="left", padx=5)
        ttk.Label(outer, textvariable=self.status_var, wraplength=1000).grid(
            row=6, column=0, sticky="ew", pady=(6, 0),
        )

    def _build_settings(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(parent, text="Analysis settings", font=("Segoe UI", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 8))
        row += 1
        row = self._setting_row(parent, row, "Primary detection", ttk.Combobox(parent, textvariable=self.primary_method_var, values=list(PRIMARY_METHODS), state="readonly"))
        row = self._setting_row(parent, row, "Analysis vertex cap", ttk.Entry(parent, textvariable=self.sample_cap_var))
        ttk.Label(parent, text="0 = full mesh unless the 30-minute/memory preflight requires limited reduction", foreground="#5f6368", wraplength=330).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6))
        row += 1
        row = self._setting_row(parent, row, "Display points", ttk.Entry(parent, textvariable=self.display_points_var))
        row = self._setting_row(parent, row, "Maximum root order", ttk.Spinbox(parent, from_=1, to=12, textvariable=self.max_order_var))
        row = self._setting_row(parent, row, "Soil-line Z", ttk.Entry(parent, textvariable=self.soil_z_var))
        row = self._setting_row(parent, row, "Runtime limit (min)", ttk.Entry(parent, textvariable=self.runtime_limit_var))
        row = self._setting_row(parent, row, "Minimum retained (%)", ttk.Entry(parent, textvariable=self.minimum_fraction_var))
        row = self._setting_row(
            parent,
            row,
            "Angle vector window (mesh units)",
            ttk.Entry(parent, textvariable=self.tip_window_var),
        )

        ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        row = self._setting_row(parent, row, "Concurrent samples", ttk.Combobox(parent, textvariable=self.concurrency_var, values=["Auto"] + [str(i) for i in range(1, 17)]))
        row = self._setting_row(parent, row, "Threads / sample", ttk.Combobox(parent, textvariable=self.threads_var, values=["Auto"] + [str(i) for i in range(1, self.hardware.logical_cpus + 1)]))

        ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        ttk.Label(parent, text="Manual XYZ base / tip", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        for label, variables in (("Base XYZ", self.start_vars), ("Tip XYZ", self.end_vars)):
            frame = ttk.Frame(parent)
            for variable in variables:
                ttk.Entry(frame, textvariable=variable, width=8).pack(side="left", padx=(0, 3))
            row = self._setting_row(parent, row, label, frame)
        row = self._setting_row(parent, row, "Guide XYZ list", ttk.Entry(parent, textvariable=self.guide_text_var))
        ttk.Label(parent, text="Format: x,y,z; x,y,z. Interactive selection stores per-sample overrides.", foreground="#5f6368", wraplength=330).grid(row=row, column=0, columnspan=2, sticky="w")

    @staticmethod
    def _setting_row(parent: ttk.Frame, row: int, label: str, widget) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        widget.grid(row=row, column=1, sticky="ew", pady=2)
        return row + 1

    def _hardware_text(self) -> str:
        gpu = ", ".join(item.name for item in self.hardware.gpus) or "CPU mode"
        memory = format_bytes(self.hardware.total_memory_bytes)
        if self.hardware.available_memory_bytes is not None:
            memory = f"{format_bytes(self.hardware.available_memory_bytes)} available / {memory}"
        return (
            f"{self.hardware.physical_cpus} physical / {self.hardware.logical_cpus} logical CPUs · "
            f"{memory} RAM · {gpu} (GPU acceleration optional)"
        )

    def _register_drop_target(self) -> None:
        try:
            from tkinterdnd2 import DND_FILES

            self.tree.drop_target_register(DND_FILES)
            self.tree.dnd_bind("<<Drop>>", self._on_drop)
        except (ImportError, AttributeError, tk.TclError):
            self.status_var.set("Drag/drop unavailable; use Add files (install tkinterdnd2 to enable it).")

    def _on_drop(self, event) -> None:
        values = [Path(value) for value in self.root.tk.splitlist(event.data)]
        self.add_files(values)

    def choose_files(self) -> None:
        values = filedialog.askopenfilenames(
            title="Choose reconstructed root meshes",
            filetypes=[("Root geometry", "*.ply *.stl *.obj *.xyz *.csv"), ("All files", "*.*")],
        )
        self.add_files(Path(value) for value in values)

    def add_files(self, paths: Iterable[Path]) -> None:
        if self.scheduler is not None and not self.scheduler.all_done:
            self.status_var.set("Batch active; wait for it to finish or cancel it before adding samples.")
            return
        output_root = self._output_root_default()
        used = {entry.output_dir for entry in self.entries.values()}
        added: list[str] = []
        for path in paths:
            path = Path(path)
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_INPUTS:
                continue
            if any(entry.input_path.resolve() == path.resolve() for entry in self.entries.values()):
                continue
            output = unique_output_directory(path, output_root, used=used)
            used.add(output)
            item = self.tree.insert(
                "",
                "end",
                values=(path.name, str(output), self.primary_method_var.get(), "Queued", "0%", "--"),
            )
            self.entries[item] = SampleEntry(item, path, output)
            added.append(item)
        if added:
            self.tree.selection_set(added)
            self.tree.see(added[-1])
            self.status_var.set(
                f"Added {len(added)} sample(s). Select one and use Load endpoints + guides to reuse a previous selection."
            )

    def _output_root_default(self) -> Path:
        text = self.output_root_var.get().strip()
        if text:
            return Path(text).expanduser()
        if self.entries:
            return next(iter(self.entries.values())).input_path.parent / "SoyRootBio_outputs"
        return Path.cwd() / "SoyRootBio_outputs"

    def choose_output_root(self) -> None:
        value = filedialog.askdirectory(title="Choose batch output root")
        if value:
            self.output_root_var.set(value)
            self._refresh_default_outputs(Path(value))

    def _refresh_default_outputs(
        self,
        output_root: Path,
        *,
        items: Iterable[str] | None = None,
    ) -> None:
        selected = set(self.entries) if items is None else set(items)
        used = {
            entry.output_dir
            for item, entry in self.entries.items()
            if item not in selected or entry.custom_output
        }
        for item, entry in self.entries.items():
            if item not in selected or entry.custom_output:
                continue
            entry.output_dir = unique_output_directory(entry.input_path, output_root, used=used)
            used.add(entry.output_dir)
            self.tree.set(item, "output", str(entry.output_dir))

    def remove_selected(self) -> None:
        if self.scheduler is not None and not self.scheduler.all_done:
            messagebox.showwarning("Batch running", "Cancel running jobs before removing samples.")
            return
        for item in self.tree.selection():
            self.tree.delete(item)
            self.entries.pop(item, None)

    def set_selected_output(self) -> None:
        selected = self.tree.selection()
        if len(selected) != 1:
            messagebox.showinfo("Select one sample", "Select exactly one sample first.")
            return
        value = filedialog.askdirectory(title="Choose this sample's output directory")
        if value:
            entry = self.entries[selected[0]]
            output = canonical_output_directory(value)
            duplicate = next(
                (
                    other.input_path.name
                    for item, other in self.entries.items()
                    if item != selected[0]
                    and canonical_output_directory(other.output_dir) == output
                ),
                None,
            )
            if duplicate is not None:
                messagebox.showerror(
                    "Output already assigned",
                    f"That output directory is already assigned to {duplicate}.",
                )
                return
            entry.output_dir = output
            entry.custom_output = True
            self.tree.set(selected[0], "output", str(output))

    def open_output_folder(self) -> None:
        """Open one selected sample's existing output directory."""

        selected = self.tree.selection()
        if len(selected) != 1:
            messagebox.showinfo(
                "Select one sample",
                "Select exactly one sample whose output folder should open.",
            )
            return
        output = canonical_output_directory(self.entries[selected[0]].output_dir)
        if not output.is_dir():
            messagebox.showinfo(
                "Output not available",
                "This output folder does not exist yet. Start the analysis first.",
            )
            return
        try:
            self._open_directory(output)
        except OSError as exc:
            messagebox.showerror("Cannot open output folder", str(exc))

    @staticmethod
    def _open_directory(path: Path) -> None:
        """Open *path* without passing it through a command shell."""

        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def load_selected_guidance(self) -> None:
        if self.scheduler is not None and not self.scheduler.all_done:
            messagebox.showinfo("Batch active", "Wait for the batch to finish before changing primary guidance.")
            return
        selected = self.tree.selection()
        if len(selected) != 1:
            messagebox.showinfo("Select one sample", "Select exactly one sample to load its endpoints and guides.")
            return
        entry = self.entries[selected[0]]
        value = filedialog.askopenfilename(
            title=f"Load endpoints and guides — {entry.input_path.name}",
            initialdir=str(entry.output_dir if entry.output_dir.is_dir() else self._output_root_default()),
            filetypes=[("Primary guidance", "*.json")],
        )
        if not value:
            return
        try:
            guidance = read_primary_guidance(value, expected_input=entry.input_path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot load primary guidance", str(exc))
            return
        entry.guidance = guidance
        self.tree.set(
            entry.item_id,
            "primary",
            f"Loaded endpoints + {len(guidance.guides)} guides"
            if guidance.use_endpoints else "Loaded soil line + guides",
        )
        self.status_var.set(f"Loaded saved primary guidance for {entry.input_path.name}.")

    def configure_selected_primary(self) -> None:
        if self.scheduler is not None and not self.scheduler.all_done:
            messagebox.showinfo("Batch active", "Wait for the batch to finish before changing primary guidance.")
            return
        selected = self.tree.selection()
        if len(selected) != 1:
            messagebox.showinfo("Select one sample", "Select exactly one sample to configure.")
            return
        entry = self.entries[selected[0]]
        try:
            picker = (
                select_soil_guidance_from_file_gui
                if PRIMARY_METHODS[self.primary_method_var.get()] == "soil"
                else select_primary_guidance_from_file_gui
            )
            _, guidance, _ = picker(
                entry.input_path,
                sample_points=self._sample_cap() or None,
                max_display_points=self._positive_int(self.display_points_var.get(), "Display points", 2),
                title=f"Primary guidance — {entry.input_path.name}",
            )
        except Exception as exc:
            messagebox.showerror("Primary selection", str(exc))
            return
        entry.guidance = guidance
        self.tree.set(
            entry.item_id,
            "primary",
            "Interactive endpoints + sections"
            if guidance.use_endpoints
            else "Interactive soil line + scorer",
        )
        self.status_var.set(
            f"Saved {len(guidance.guides)} guide section(s)"
            + (f" and soil Z {guidance.soil_z:.6g}." if guidance.soil_z is not None else ".")
        )

    def start_batch(self) -> None:
        if not self.entries:
            messagebox.showinfo("No samples", "Add one or more STL/PLY samples first.")
            return
        if self.scheduler is not None and not self.scheduler.all_done:
            messagebox.showinfo("Batch active", "The current batch is already running.")
            return
        selected_items = self._next_batch_items()
        new_scheduler: BatchScheduler | None = None
        try:
            # Refresh the memory snapshot at admission time; available RAM can
            # change substantially while the GUI remains open.
            self.hardware = detect_hardware()
            self.hardware_var.set(self._hardware_text())
            concurrency = self._optional_positive_int(self.concurrency_var.get())
            threads = self._optional_positive_int(self.threads_var.get())
            allocation = allocate_resources(
                self.hardware,
                sample_count=len(selected_items),
                max_concurrent_samples=concurrency,
                threads_per_sample=threads,
            )
            output_root = self._output_root_default()
            self._refresh_default_outputs(output_root, items=selected_items)
            self._validate_output_ownership(selected_items)
            output_root.mkdir(parents=True, exist_ok=True)
            history = StepTimingHistory(output_root / ".soyrootbio_step_timings.json")
            prepared = [
                (item, entry, self._pipeline_config(entry, allocation.threads_per_sample))
                for item, entry in self.entries.items()
                if item in selected_items
            ]
            new_scheduler = BatchScheduler(
                self._run_job,
                max_concurrent_samples=allocation.max_concurrent_samples,
                threads_per_sample=allocation.threads_per_sample,
                timing_history=history,
            )
            new_job_to_item: dict[str, str] = {}
            for item, entry, payload in prepared:
                job = new_scheduler.submit(entry.input_path, entry.output_dir, payload=payload)
                entry.job_id = job.job_id
                new_job_to_item[job.job_id] = item
                self.tree.set(item, "runtime", "--")
                self.tree.set(item, "progress", "0%")
                self.tree.set(item, "status", "Queued")
            if self.scheduler is not None:
                self.scheduler.shutdown(wait=False)
            new_scheduler.start()
            self.scheduler = new_scheduler
            self.job_to_item.clear()
            self.job_to_item.update(new_job_to_item)
            self.status_var.set(
                f"Running up to {allocation.max_concurrent_samples} sample(s) concurrently, "
                f"{allocation.threads_per_sample} thread(s) each."
            )
        except Exception as exc:
            if new_scheduler is not None:
                new_scheduler.shutdown(wait=False, cancel_pending=True)
            for item in selected_items:
                self.entries[item].job_id = None
            messagebox.showerror("Cannot start batch", str(exc))

    def _next_batch_items(self) -> tuple[str, ...]:
        """Run newly queued entries, or explicitly rerun all when none are new."""

        queued = tuple(item for item, entry in self.entries.items() if entry.job_id is None)
        return queued or tuple(self.entries)

    def _validate_output_ownership(self, items: Iterable[str]) -> None:
        owners: dict[Path, str] = {}
        for item in items:
            entry = self.entries[item]
            output = canonical_output_directory(entry.output_dir)
            previous = owners.get(output)
            if previous is not None:
                other = self.entries[previous]
                raise ValueError(
                    f"Output directory is assigned to both {other.input_path.name} "
                    f"and {entry.input_path.name}: {output}"
                )
            owners[output] = item

    def _run_job(self, job, control, progress):
        config: PipelineConfig = job.payload
        config.output_dir = job.output_dir
        config.worker_threads = job.threads_per_sample
        # ``threadpool_limits`` changes process-global native-library state.
        # Per-job contexts overlap in a ThreadPoolExecutor and can restore one
        # another's limits out of order.  The pipeline's ContextVar-based
        # worker setting safely isolates cKDTree parallelism per analysis.
        return run_pipeline(
            config,
            progress_callback=progress,
            cancel_check=control.cancel_check,
            pause_check=lambda: control.paused,
        )

    def _pipeline_config(self, entry: SampleEntry, threads: int) -> PipelineConfig:
        if entry.output_dir.exists() and any(entry.output_dir.iterdir()):
            raise ValueError(
                f"Output directory for {entry.input_path.name} is not empty; choose a fresh directory."
            )
        method = PRIMARY_METHODS[self.primary_method_var.get()]
        start = end = None
        guides: tuple[tuple[float, float, float], ...] = ()
        soil_z = self._optional_float(self.soil_z_var.get(), "Soil-line Z")
        if entry.guidance is not None:
            method = "interactive" if entry.guidance.use_endpoints else "soil"
            if entry.guidance.use_endpoints:
                start = tuple(float(value) for value in entry.guidance.start)
                end = tuple(float(value) for value in entry.guidance.end)
            guides = tuple(tuple(float(value) for value in row) for row in entry.guidance.guides)
            soil_z = entry.guidance.soil_z if entry.guidance.soil_z is not None else soil_z
        elif method == "coordinates":
            start = self._xyz(self.start_vars, "Base")
            end = self._xyz(self.end_vars, "Tip")
            guides = self._guide_text()
        elif method == "interactive":
            raise ValueError(f"Configure primary guidance for {entry.input_path.name} before starting.")
        return PipelineConfig(
            input_path=entry.input_path,
            output_dir=entry.output_dir,
            start=start,
            end=end,
            auto_endpoints=("scored" if method == "soil" else method) if method not in {"interactive", "coordinates"} else None,
            soil_z=soil_z if method in {"soil", "interactive"} else None,
            primary_guides=guides,
            correction_file=entry.correction_file,
            sample_points=self._sample_cap() or None,
            max_root_order=self._positive_int(self.max_order_var.get(), "Maximum root order", 1),
            runtime_limit_minutes=self._positive_float(self.runtime_limit_var.get(), "Runtime limit"),
            minimum_retained_fraction=self._percent_fraction(self.minimum_fraction_var.get()),
            tip_vector_window_mesh_units=self._positive_float(
                self.tip_window_var.get(),
                "Angle vector window",
            ),
            worker_threads=threads,
        )

    def pause_selected(self) -> None:
        self._selected_job_action("pause")

    def resume_selected(self) -> None:
        self._selected_job_action("resume")

    def cancel_selected(self) -> None:
        self._selected_job_action("cancel")

    def _selected_job_action(self, action: str) -> None:
        if self.scheduler is None:
            return
        for item in self.tree.selection():
            job_id = self.entries[item].job_id
            if job_id and job_id in self.job_to_item:
                getattr(self.scheduler, action)(job_id)

    def cancel_all(self) -> None:
        if self.scheduler is not None:
            self.scheduler.cancel_all()

    def _poll_scheduler(self) -> None:
        if self.scheduler is not None:
            for event in self.scheduler.drain_events():
                item = self.job_to_item.get(event.job_id)
                if item is None or not self.tree.exists(item):
                    continue
                snapshot = event.job
                self.tree.set(item, "status", snapshot.state.value.title())
                self.tree.set(item, "progress", f"{snapshot.progress_percent:.0f}%")
                if event.kind == BatchEventType.FAILED:
                    self.tree.set(item, "status", "Failed")
                    self.status_var.set(f"{self.entries[item].input_path.name}: {snapshot.error}")
            jobs = self.scheduler.jobs
            # Stages can run for minutes without a progress event. Refresh
            # the monotonic active duration independently on every GUI poll.
            for job in jobs:
                item = self.job_to_item.get(job.job_id)
                if item is not None and self.tree.exists(item):
                    self.tree.set(
                        item, "runtime",
                        self._format_runtime(job.elapsed_seconds if job.started_at is not None else None),
                    )
            if self.scheduler.all_done and jobs:
                completed = sum(job.state == BatchJobState.COMPLETED for job in jobs)
                failed = sum(job.state == BatchJobState.FAILED for job in jobs)
                if failed:
                    self.status_var.set(
                        f"Batch finished: {completed}/{len(jobs)} completed; "
                        f"{failed} failed. See processing_error.log in each failed output folder."
                    )
                else:
                    self.status_var.set(f"Batch finished: {completed}/{len(jobs)} completed.")
        self.root.after(120, self._poll_scheduler)

    @staticmethod
    def _format_runtime(seconds: float | None) -> str:
        if seconds is None:
            return "--"
        seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    def _sample_cap(self) -> int:
        text = self.sample_cap_var.get().strip()
        if text in {"", "0"}:
            return 0
        return self._positive_int(text, "Analysis vertex cap", 20)

    @staticmethod
    def _positive_int(value: str, name: str, minimum: int) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a whole number") from exc
        if parsed < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        return parsed

    @staticmethod
    def _optional_positive_int(value: str) -> int | None:
        return None if value.strip().lower() in {"", "auto"} else BioInsAlgoBatchApp._positive_int(value, "Resource value", 1)

    @staticmethod
    def _positive_float(value: str, name: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number") from exc
        if not np.isfinite(parsed) or parsed <= 0:
            raise ValueError(f"{name} must be positive")
        return parsed

    @staticmethod
    def _optional_float(value: str, name: str) -> float | None:
        return None if not value.strip() else BioInsAlgoBatchApp._positive_or_finite_float(value, name)

    @staticmethod
    def _positive_or_finite_float(value: str, name: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number") from exc
        if not np.isfinite(parsed):
            raise ValueError(f"{name} must be finite")
        return parsed

    def _percent_fraction(self, value: str) -> float:
        parsed = self._positive_float(value, "Minimum retained percentage") / 100.0
        if parsed > 1:
            raise ValueError("Minimum retained percentage cannot exceed 100")
        return parsed

    @staticmethod
    def _xyz(variables: list[tk.StringVar], name: str) -> tuple[float, float, float]:
        values = [BioInsAlgoBatchApp._positive_or_finite_float(variable.get(), f"{name} coordinate") for variable in variables]
        return tuple(values)

    def _guide_text(self) -> tuple[tuple[float, float, float], ...]:
        text = self.guide_text_var.get().strip()
        if not text:
            return ()
        guides = []
        for group in text.split(";"):
            values = [part.strip() for part in group.split(",")]
            if len(values) != 3:
                raise ValueError("Each guide must contain x,y,z")
            guides.append(tuple(self._positive_or_finite_float(value, "Guide coordinate") for value in values))
        return tuple(guides)

    def close(self) -> None:
        if self.scheduler is not None and not self.scheduler.all_done:
            if not messagebox.askyesno("Exit", "Cancel all active analyses and close?"):
                return
            self.scheduler.shutdown(wait=False, cancel_pending=True)
        self.root.destroy()


def create_root() -> tk.Tk:
    """Create a TkDND root when available, otherwise a standard Tk root."""

    try:
        from tkinterdnd2 import TkinterDnD

        return TkinterDnD.Tk()
    except ImportError:
        return tk.Tk()


__all__ = ["BioInsAlgoBatchApp", "PRIMARY_METHODS", "SampleEntry", "create_root"]
