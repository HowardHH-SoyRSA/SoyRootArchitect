"""Versioned output paths; legacy bundles remain readable."""
from pathlib import Path
import json
import math
import numpy as np


def bundle_file(root, relative):
    root, relative = Path(root), Path(relative)
    name = relative.name
    aliases = {"edited_root_hierarchy.json": "root_hierarchy.json", "edited_root_traits.csv": "root_traits.csv",
               "edited_root_label_map.csv": "root_label_map.csv", "edited_root_system.rsml": "root_system.rsml"}
    if name in aliases:
        alias = bundle_file(root, aliases[name])
        if alias.is_file() and (root / "support").is_dir():
            return alias
    if name in {"segmented_root_structure.ply", "edited_segmented_root_structure.ply"}:
        candidate = root / "presentation_root_structure.ply"
        if candidate.is_file() and (root / "support" / "analysis_geometry.npz").is_file():
            return candidate
    directory = "tables" if relative.suffix.lower() in {".csv", ".xlsx", ".xls"} else "support"
    candidate = root / directory / name
    if candidate.is_file():
        return candidate
    return root / relative


def save_analysis_geometry(root, points, faces, colors, labels, orders, states):
    support = Path(root) / "support"
    support.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(support / "analysis_geometry.npz", positions=points,
        triangles=np.empty((0, 3), np.int32) if faces is None else faces,
        colors=np.asarray(np.round(np.asarray(colors)*255), dtype=np.uint8),
        root_labels=labels, root_orders=orders, assignment_states=states)


def read_bundle_mesh(root):
    from .editor.ply import LabeledMesh, read_labeled_ply
    path = bundle_file(root, "analysis_geometry.npz")
    if not path.is_file():
        return read_labeled_ply(bundle_file(root, "segmented_root_structure.ply"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in
                  ("positions", "triangles", "colors", "root_labels", "root_orders", "assignment_states")}
    n = len(arrays["positions"])
    if any(len(arrays[key]) != n for key in ("colors", "root_labels", "root_orders", "assignment_states")):
        raise ValueError("Bundle analysis arrays differ in vertex count")
    faces = arrays["triangles"]
    if len(faces) and (faces.min() < 0 or faces.max() >= n):
        raise ValueError("Bundle face indices are invalid")
    return LabeledMesh(path=bundle_file(root, "segmented_root_structure.ply"),
                       vertex_count=n, face_count=len(faces), **arrays)


def finalize_bundle(root):
    """Organize newly generated artifacts and rewrite exact relative references."""
    root = Path(root).resolve()
    (root / "tables").mkdir(exist_ok=True)
    (root / "support").mkdir(exist_ok=True)
    moved = {}
    for path in list(root.iterdir()):
        if not path.is_file() or path.suffix.lower() in {".png", ".ply"}:
            continue
        directory = "tables" if path.suffix.lower() in {".csv", ".xlsx", ".xls"} else "support"
        destination = root / directory / path.name
        path.replace(destination)
        moved[path.name] = destination.relative_to(root).as_posix()
    # Auxiliary blob paths remain relative to the manifest in support/.
    if (root / "blobs").is_dir():
        import shutil
        shutil.copytree(root / "blobs", root / "support" / "blobs", dirs_exist_ok=True)
        for path in (root / "blobs").iterdir():
            if path.is_file():
                path.unlink()
        (root / "blobs").rmdir()
    def rewrite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: rewrite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        return moved.get(value, value) if isinstance(value, str) else value
    for path in (root / "support").glob("*.json"):
        # Historical evidence is immutable provenance, not current paths.
        if path.name.startswith("source_metadata_"):
            continue
        value = rewrite(json.loads(path.read_text(encoding="utf-8")))
        if path.name == "metadata.json":
            value["export_layout"] = {"version": 2, "tables": "tables", "support": "support",
                "mesh": "presentation_root_structure.ply", "analysis_geometry": "support/analysis_geometry.npz"}
            value["outputs"] = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
