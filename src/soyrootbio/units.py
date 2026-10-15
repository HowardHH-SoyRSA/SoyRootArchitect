"""Explicit source-coordinate calibration; STL declarations are evidence only.

STL has no mandatory unit field. Only explicit Unit/Units declarations in a
verified binary header are accepted. Arbitrary prose and ASCII STL are unknown.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import re
import struct

import numpy as np


_ALIASES = {
    "mm": ("mm", 1.0), "millimeter": ("mm", 1.0), "millimeters": ("mm", 1.0),
    "millimetre": ("mm", 1.0), "millimetres": ("mm", 1.0),
    "cm": ("cm", 10.0), "centimeter": ("cm", 10.0), "centimeters": ("cm", 10.0),
    "centimetre": ("cm", 10.0), "centimetres": ("cm", 10.0),
    "m": ("m", 1000.0), "meter": ("m", 1000.0), "meters": ("m", 1000.0),
    "metre": ("m", 1000.0), "metres": ("m", 1000.0),
    "um": ("um", .001), "micrometer": ("um", .001), "micrometers": ("um", .001),
    "micrometre": ("um", .001), "micrometres": ("um", .001),
    "nm": ("nm", .000001), "nanometer": ("nm", .000001), "nanometers": ("nm", .000001),
    "nanometre": ("nm", .000001), "nanometres": ("nm", .000001),
    "in": ("inch", 25.4), "inch": ("inch", 25.4), "inches": ("inch", 25.4),
}
_DECLARATION = re.compile(r"\bunits?\s*[:=]\s*([^\]\[\x00;,()]+)", re.IGNORECASE)


class UnitResolutionError(ValueError):
    """A source needs deliberate calibration before it can be processed."""


def validate_scale(value) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise UnitResolutionError("Millimeters per mesh unit must be a positive finite number.")
    try:
        scale = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise UnitResolutionError("Millimeters per mesh unit must be a positive finite number.") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise UnitResolutionError("Millimeters per mesh unit must be a positive finite number.")
    return scale


@dataclass(frozen=True)
class UnitCalibration:
    mm_per_mesh_unit: float
    source: str
    declared_unit: str | None = None
    raw_declaration: str | None = None
    schema_version: int = 1
    source_coordinate_unit: str = "mesh_unit"
    output_coordinate_unit: str = "mm"

    def __post_init__(self):
        object.__setattr__(self, "mm_per_mesh_unit", validate_scale(self.mm_per_mesh_unit))
        if self.source not in {"stl_header", "user", "batch_rule", "cli"}:
            raise UnitResolutionError("Unknown calibration provenance.")
        if (self.schema_version != 1 or self.source_coordinate_unit != "mesh_unit"
                or self.output_coordinate_unit != "mm"):
            raise UnitResolutionError("Unsupported unit calibration contract.")

    def to_dict(self) -> dict:
        return asdict(self)


def source_unit_metadata(path: str | Path) -> dict:
    path = Path(path)
    result = {"declared_coordinate_unit": None, "declared_mm_per_mesh_unit": None,
              "unit_declaration_status": "missing", "raw_unit_declaration": None}
    if path.suffix.lower() != ".stl":
        return result
    with path.open("rb") as handle:
        prefix = handle.read(84)
    if len(prefix) < 84:
        return result
    count = struct.unpack("<I", prefix[80:84])[0]
    if path.stat().st_size != 84 + 50 * count:
        return {**result, "unit_declaration_status": "not_binary_stl"}
    header = prefix[:80].decode("utf-8", errors="replace").strip("\x00 ")
    result["stl_header"] = header
    tokens = [value.strip() for value in _DECLARATION.findall(header)]
    if not tokens:
        return result
    result["raw_unit_declaration"] = "; ".join(tokens)
    resolved = [_ALIASES.get(value.lower().replace("μ", "u").replace("µ", "u")) for value in tokens]
    if any(item is None for item in resolved):
        return {**result, "unit_declaration_status": "unsupported"}
    if len({item[1] for item in resolved}) != 1:
        return {**result, "unit_declaration_status": "conflicting"}
    return {**result, "declared_coordinate_unit": resolved[0][0],
            "declared_mm_per_mesh_unit": resolved[0][1], "unit_declaration_status": "known"}


def resolve_unit_calibration(path: str | Path, mm_per_mesh_unit=None, *,
                             source: str = "cli", metadata: dict | None = None) -> UnitCalibration:
    info = source_unit_metadata(path) if metadata is None else metadata
    declared = info.get("declared_mm_per_mesh_unit")
    if declared is not None:
        declared = validate_scale(declared)
        if mm_per_mesh_unit is not None and not math.isclose(
                validate_scale(mm_per_mesh_unit), declared, rel_tol=1e-12, abs_tol=0.):
            raise UnitResolutionError(f"Supplied scale conflicts with {Path(path).name}'s declared unit {info.get('declared_coordinate_unit')}.")
        return UnitCalibration(declared, "stl_header", info.get("declared_coordinate_unit"),
                               info.get("raw_unit_declaration"))
    if mm_per_mesh_unit is None:
        raise UnitResolutionError(
            f"{Path(path).name} has no unambiguous supported unit declaration. "
            "Specify millimeters per mesh unit (--mm-per-mesh-unit for the CLI).")
    if source == "stl_header":
        raise UnitResolutionError("A header calibration requires a current supported unit declaration.")
    return UnitCalibration(validate_scale(mm_per_mesh_unit), source,
                           info.get("declared_coordinate_unit"), info.get("raw_unit_declaration"))
