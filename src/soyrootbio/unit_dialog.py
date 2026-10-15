"""Main-thread calibration preflight shared by single and batch desktop flows."""
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Callable, Iterable

from .units import UnitResolutionError, resolve_unit_calibration, source_unit_metadata, validate_scale


class UnitResolutionCancelled(RuntimeError):
    pass


class UnitScaleDialog:
    def __init__(self, parent, path: Path, *, allow_remaining: bool = True):
        self.result = None
        self.window = tk.Toplevel(parent)
        self.window.title("Set mesh units")
        self.window.transient(parent)
        self.scale_var = tk.StringVar(self.window, value="")
        self.apply_var = tk.BooleanVar(self.window, value=False)
        self.error_var = tk.StringVar(self.window, value="")
        body = ttk.Frame(self.window, padding=18)
        body.grid(sticky="nsew")
        ttk.Label(body, text=f"{path.name}\nNo supported unambiguous unit declaration was found.",
                  wraplength=430).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(body, text="Millimeters per mesh unit").grid(row=1, column=0, sticky="w", pady=(14, 0))
        self.entry = ttk.Entry(body, textvariable=self.scale_var, width=18)
        self.entry.grid(row=1, column=1, sticky="ew", pady=(14, 0))
        ttk.Label(body, text="Examples: 1 for mm; 10 for cm; 1000 for m; 25.4 for inches.",
                  wraplength=430).grid(row=2, column=0, columnspan=2, sticky="w", pady=6)
        self.checkbox = None
        if allow_remaining:
            self.checkbox = ttk.Checkbutton(body, text="Apply to remaining samples with unknown units in this batch",
                                            variable=self.apply_var)
            self.checkbox.grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Label(body, textvariable=self.error_var, foreground="#b91c1c", wraplength=430).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=6)
        buttons = ttk.Frame(body)
        buttons.grid(row=5, column=0, columnspan=2, sticky="e")
        ttk.Button(buttons, text="Cancel", command=self.cancel).pack(side="left", padx=5)
        ttk.Button(buttons, text="Use scale", command=self.accept).pack(side="left")
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)
        self.window.bind("<Return>", lambda _event: self.accept())
        self.window.bind("<Escape>", lambda _event: self.cancel())
        self.entry.focus_set()

    def accept(self):
        try:
            scale = validate_scale(self.scale_var.get())
        except UnitResolutionError as exc:
            self.error_var.set(str(exc))
            return
        self.result = (scale, bool(self.apply_var.get()))
        self.window.destroy()

    def cancel(self):
        self.result = None
        self.window.destroy()

    def show(self):
        self.window.grab_set()
        self.window.wait_window()
        return self.result


def resolve_batch_units(paths: Iterable[Path], prompt: Callable, *, existing: dict | None = None) -> dict:
    """Resolve the entire launch atomically; a batch rule never overrides headers."""
    resolved = {}
    rule = None
    existing = existing or {}
    paths = tuple(Path(path) for path in paths)
    for index, path in enumerate(paths):
        info = source_unit_metadata(path)
        previous = existing.get(path)
        if previous and previous.get("source") == "stl_header":
            previous = None  # Current declarations, never cached header evidence, establish scale.
        if info.get("declared_mm_per_mesh_unit") is not None:
            value = previous.get("mm_per_mesh_unit") if previous else None
            calibration = resolve_unit_calibration(path, value, metadata=info)
        elif previous:
            calibration = resolve_unit_calibration(path, previous["mm_per_mesh_unit"],
                                                  source=previous["source"], metadata=info)
        elif rule is not None:
            calibration = resolve_unit_calibration(path, rule, source="batch_rule", metadata=info)
        else:
            answer = prompt(path, allow_remaining=index < len(paths) - 1)
            if answer is None:
                raise UnitResolutionCancelled("Unit selection cancelled; no samples were started.")
            value, apply = answer
            calibration = resolve_unit_calibration(path, value, source="user", metadata=info)
            if apply:
                rule = calibration.mm_per_mesh_unit
        resolved[path] = calibration.to_dict()
    return resolved
