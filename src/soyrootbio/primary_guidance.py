from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


PRIMARY_GUIDANCE_FILENAME = "primary_guidance.json"
PRIMARY_GUIDANCE_SCHEMA = "soyrootbio.primary-guidance"


@dataclass(frozen=True)
class PrimaryGuidance:
    """Manual biological constraints in the original mesh coordinates."""

    start: np.ndarray
    end: np.ndarray
    soil_z: float | None
    guides: np.ndarray
    use_endpoints: bool = True


def _coordinates(value, name: str, *, multiple: bool = False) -> np.ndarray:
    try:
        array = np.asarray(value)
        if multiple and array.shape == (0,):
            array = np.empty((0, 3), dtype=float)
        valid_shape = (
            array.ndim == 2 and array.shape[1] == 3
            if multiple else array.shape == (3,)
        )
        if (
            not valid_shape or array.dtype.kind not in "fiu"
            or any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat)
        ):
            raise ValueError
        array = array.astype(float, copy=True)
        if not np.all(np.isfinite(array)):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numeric XYZ coordinates.") from exc
    return array


def _validated_guidance(guidance: PrimaryGuidance) -> PrimaryGuidance:
    start = _coordinates(guidance.start, "Collar")
    end = _coordinates(guidance.end, "Tip")
    guides = _coordinates(guidance.guides, "Guides", multiple=True)
    if not isinstance(guidance.use_endpoints, bool):
        raise ValueError("use_endpoints must be true or false.")
    if guidance.use_endpoints and np.linalg.norm(end - start) <= 1e-12:
        raise ValueError("Collar and tip must be distinct.")
    soil_z = guidance.soil_z
    if soil_z is not None:
        if isinstance(soil_z, (bool, str)):
            raise ValueError("Soil-line Z must be a finite number.")
        try:
            soil_z = float(soil_z)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Soil-line Z must be a finite number.") from exc
        if not np.isfinite(soil_z):
            raise ValueError("Soil-line Z must be a finite number.")
    return PrimaryGuidance(start, end, soil_z, guides, guidance.use_endpoints)


def write_primary_guidance(
    path: str | Path,
    guidance: PrimaryGuidance,
    *,
    input_path: str | Path,
    overwrite: bool = True,
) -> Path:
    """Save endpoints and ordered guides together without external references."""

    selected = _validated_guidance(guidance)
    payload = {
        "schema": PRIMARY_GUIDANCE_SCHEMA,
        "schema_version": 1,
        "input_path": str(input_path),
        "coordinate_space": "source",
        "coordinate_unit": "mesh_unit",
        "start": selected.start.tolist(),
        "end": selected.end.tolist(),
        "guides": selected.guides.tolist(),
        "soil_z": selected.soil_z,
        "use_endpoints": selected.use_endpoints,
    }
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w" if overwrite else "x", encoding="utf-8") as handle:
        handle.write(serialized)
    return destination


def read_primary_guidance(
    path: str | Path,
    *,
    expected_input: str | Path | None = None,
) -> PrimaryGuidance:
    """Read a reusable selection, rejecting wrong units and sample mismatches.

    The sample filename is checked instead of its directory so result bundles
    and their source meshes can be moved together to another computer.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("schema") != PRIMARY_GUIDANCE_SCHEMA:
        raise ValueError("Choose an exported primary_guidance.json selection file.")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported primary guidance schema version.")
    if payload.get("coordinate_space") != "source" or payload.get("coordinate_unit") != "mesh_unit":
        raise ValueError("Primary guidance must use original source mesh coordinates.")
    if expected_input is not None:
        source = payload.get("input_path")
        if not isinstance(source, str) or not source:
            raise ValueError("The selection file is missing its source sample name.")
        source_name = Path(source.replace("\\", "/")).name
        expected_name = Path(str(expected_input).replace("\\", "/")).name
        if source_name.casefold() != expected_name.casefold():
            raise ValueError(
                f"This selection belongs to {source_name}, not {expected_name}."
            )
    return _validated_guidance(
        PrimaryGuidance(
            start=payload.get("start"),
            end=payload.get("end"),
            soil_z=payload.get("soil_z"),
            guides=payload.get("guides"),
            use_endpoints=payload.get("use_endpoints", True),
        )
    )
