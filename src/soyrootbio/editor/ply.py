from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


_SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass(slots=True)
class LabeledMesh:
    path: Path
    positions: np.ndarray
    triangles: np.ndarray
    colors: np.ndarray
    root_labels: np.ndarray
    root_orders: np.ndarray
    assignment_states: np.ndarray
    vertex_count: int
    face_count: int


def read_labeled_ply(path: str | Path) -> LabeledMesh:
    """Read SoyRootBio's binary labelled PLY without discarding scalar fields.

    Open3D's legacy mesh reader intentionally focuses on standard geometry
    attributes.  The editor needs the custom root labels too, so this compact
    reader handles the deterministic PLY layout emitted by ``write_labeled_ply``
    and other binary little-endian PLY files with scalar vertex properties and
    triangular faces.
    """

    path = Path(path)
    header, data_offset = _read_header(path)
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("The editor currently requires a binary little-endian PLY mesh.")

    vertex_count = _element_count(header, "vertex")
    face_count = _element_count(header, "face")
    if vertex_count <= 0:
        raise ValueError("PLY must contain at least one vertex.")
    vertex_properties = _vertex_properties(header)
    required = {"x", "y", "z"}
    missing = sorted(required - {name for name, _ in vertex_properties})
    if missing:
        raise ValueError("PLY is missing required vertex properties: " + ", ".join(missing))

    vertex_dtype = np.dtype([(name, _SCALAR_DTYPES[type_name]) for name, type_name in vertex_properties])
    expected_vertex_end = data_offset + vertex_dtype.itemsize * vertex_count
    if expected_vertex_end > path.stat().st_size:
        raise ValueError("PLY vertex data is truncated.")
    vertices = np.memmap(
        path,
        dtype=vertex_dtype,
        mode="r",
        offset=data_offset,
        shape=(vertex_count,),
    )
    positions = np.column_stack(
        [vertices["x"], vertices["y"], vertices["z"]]
    ).astype(np.float64, copy=False)

    colors = np.full((vertex_count, 3), 140, dtype=np.uint8)
    if all(name in vertices.dtype.names for name in ("red", "green", "blue")):
        colors = np.column_stack(
            [vertices["red"], vertices["green"], vertices["blue"]]
        ).astype(np.uint8, copy=False)

    root_labels = _property_or_default(vertices, "root_id", -1, np.int32)
    root_orders = _property_or_default(vertices, "root_order", 255, np.uint8)
    assignment_states = _property_or_default(vertices, "assignment_state", 0, np.uint8)

    triangles = np.empty((0, 3), dtype=np.int32)
    if face_count:
        face_dtype = np.dtype([("count", "u1"), ("vertices", "<i4", (3,))])
        expected_face_end = expected_vertex_end + face_dtype.itemsize * face_count
        if expected_face_end > path.stat().st_size:
            raise ValueError("PLY face data is truncated or does not contain triangular faces.")
        faces = np.memmap(
            path,
            dtype=face_dtype,
            mode="r",
            offset=expected_vertex_end,
            shape=(face_count,),
        )
        if np.any(faces["count"] != 3):
            raise ValueError("The editor requires triangular PLY faces.")
        triangles = np.asarray(faces["vertices"], dtype=np.int32).copy()
        if len(triangles) and (
            int(np.min(triangles)) < 0
            or int(np.max(triangles)) >= vertex_count
        ):
            raise ValueError("PLY contains a face index outside the vertex array.")

    return LabeledMesh(
        path=path,
        positions=np.asarray(positions, dtype=np.float64),
        triangles=triangles,
        colors=np.asarray(colors, dtype=np.uint8),
        root_labels=np.asarray(root_labels, dtype=np.int32).copy(),
        root_orders=np.asarray(root_orders, dtype=np.uint8).copy(),
        assignment_states=np.asarray(assignment_states, dtype=np.uint8).copy(),
        vertex_count=vertex_count,
        face_count=face_count,
    )


def _read_header(path: Path) -> tuple[str, int]:
    marker = b"end_header\n"
    marker_crlf = b"end_header\r\n"
    collected = bytearray()
    with path.open("rb") as handle:
        while len(collected) < 1024 * 1024:
            chunk = handle.read(4096)
            if not chunk:
                break
            collected.extend(chunk)
            for candidate in (marker_crlf, marker):
                index = collected.find(candidate)
                if index >= 0:
                    end = index + len(candidate)
                    return bytes(collected[:end]).decode("ascii"), end
    raise ValueError("PLY header terminator is missing.")


def _element_count(header: str, name: str) -> int:
    match = re.search(rf"^element\s+{re.escape(name)}\s+(\d+)\s*$", header, re.MULTILINE)
    return int(match.group(1)) if match else 0


def _vertex_properties(header: str) -> list[tuple[str, str]]:
    properties: list[tuple[str, str]] = []
    in_vertices = False
    for raw_line in header.splitlines():
        line = raw_line.strip()
        if line.startswith("element "):
            in_vertices = line.startswith("element vertex ")
            continue
        if not in_vertices or not line.startswith("property "):
            continue
        parts = line.split()
        if len(parts) != 3 or parts[1] == "list":
            raise ValueError(f"Unsupported PLY vertex property declaration: {line}")
        type_name, property_name = parts[1], parts[2]
        if type_name not in _SCALAR_DTYPES:
            raise ValueError(f"Unsupported PLY scalar type: {type_name}")
        properties.append((property_name, type_name))
    return properties


def _property_or_default(
    vertices: np.ndarray,
    name: str,
    default: int,
    dtype,
) -> np.ndarray:
    if name not in (vertices.dtype.names or ()):
        return np.full(len(vertices), default, dtype=dtype)
    return np.asarray(vertices[name], dtype=dtype)
