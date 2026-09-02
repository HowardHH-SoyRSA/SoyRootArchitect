from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import threading
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..export import order_color, write_rsml
from ..geometry import child_length_exceeds_parent, path_length
from ..hardware import HardwareInfo, detect_hardware
from ..io import write_labeled_ply
from ..traits import compute_traits
from ..types import Normalization, RootPath
from .ply import LabeledMesh, read_labeled_ply


PRIMARY_ID = "primary"
LOG_SCHEMA = "soyrootbio.editor-log/v1"
OPERATION_SCHEMA = "soyrootbio.editor-operation/v1"


class EditorValidationError(ValueError):
    """An invalid interactive edit that must not alter materialised state."""


class EditorRevisionConflict(EditorValidationError):
    """A point-patch request that refers to an obsolete label revision."""


@dataclass
class RootNode:
    root_id: str
    parent_id: str | None
    order: int
    points: np.ndarray
    numeric_label: int
    traits: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    qc_flags: list[str] = field(default_factory=list)
    insertion_point: np.ndarray | None = None
    insertion_index: int | None = None
    order_overridden: bool = False
    source_root_ids: list[str] = field(default_factory=list)

    def clone(self) -> "RootNode":
        return RootNode(
            root_id=self.root_id,
            parent_id=self.parent_id,
            order=int(self.order),
            points=np.asarray(self.points, dtype=float).copy(),
            numeric_label=int(self.numeric_label),
            traits=dict(self.traits),
            confidence=float(self.confidence),
            qc_flags=list(self.qc_flags),
            insertion_point=(
                None
                if self.insertion_point is None
                else np.asarray(self.insertion_point, dtype=float).copy()
            ),
            insertion_index=self.insertion_index,
            order_overridden=bool(self.order_overridden),
            source_root_ids=list(self.source_root_ids),
        )


@dataclass
class Operation:
    operation_id: str
    sequence: int
    timestamp: str
    type: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OPERATION_SCHEMA,
            "operation_id": self.operation_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "type": self.type,
            "arguments": _json_safe(self.arguments),
        }


@dataclass
class _UndoRecord:
    roots_before: dict[str, RootNode]
    next_numeric_label_before: int
    changed_indices: np.ndarray
    changed_labels_before: np.ndarray
    changed_assignment_before: np.ndarray
    created_blobs: tuple[Path, ...] = ()


@dataclass
class _HistoryFrame:
    operation: Operation
    undo: _UndoRecord


class EditorSession:
    """Materialised editable view of an immutable SoyRootBio output bundle."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        session_dir: str | Path | None = None,
        load_existing_log: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self._require_bundle()
        self.mesh: LabeledMesh = read_labeled_ply(
            self.output_dir / "segmented_root_structure.ply"
        )
        self.mesh_minimum = np.min(self.mesh.positions, axis=0)
        self.mesh_maximum = np.max(self.mesh.positions, axis=0)
        self.render_origin = 0.5 * (self.mesh_minimum + self.mesh_maximum)
        self._baseline_labels = self.mesh.root_labels.copy()
        self._baseline_assignment_states = self.mesh.assignment_states.copy()
        self._baseline_roots = self._load_roots()
        self.roots = self._clone_roots(self._baseline_roots)
        (
            self.tip_vector_window_mesh_units,
            self.gravity,
        ) = self._load_trait_configuration()
        self._next_numeric_label = max(
            [root.numeric_label for root in self.roots.values()] + [0]
        ) + 1
        self._validate_state()
        self._baseline_next_numeric_label = self._next_numeric_label
        self.baseline_fingerprint = self._baseline_fingerprint()
        self.session_dir = (
            Path(session_dir).resolve()
            if session_dir is not None
            else self.output_dir / ".soyrootbio-editor"
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.blob_dir = self.session_dir / "blobs"
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.session_dir / "operations.jsonl"
        self.manifest_path = self.session_dir / "session.json"
        self.hardware: HardwareInfo = detect_hardware()
        self._history: list[_HistoryFrame] = []
        self._redo: list[Operation] = []
        self._sequence = 0
        self._mesh_tree: cKDTree | None = None
        self._mesh_edges: tuple[np.ndarray, np.ndarray] | None = None
        self._point_patch_revision = -1
        self._point_patch_summaries: list[dict[str, Any]] = []
        self._point_patch_indices: dict[str, np.ndarray] = {}
        self._lock = threading.RLock()
        self._pending_index_chunks: list[np.ndarray] = []
        self._pending_old_label_chunks: list[np.ndarray] = []
        self._pending_old_assignment_chunks: list[np.ndarray] = []
        self._pending_created_blobs: list[Path] = []
        self.label_revision = 0
        self._write_session_manifest()
        if load_existing_log and self.log_path.exists():
            self._replay_log()

    # ------------------------------------------------------------------
    # Public state and persistence
    # ------------------------------------------------------------------
    def public_state(self) -> dict[str, Any]:
        with self._lock:
            children = self._children_map()
            point_patches = self._point_patch_public()
            roots = [
                self._root_public(root, children.get(root.root_id, []))
                for root in sorted(
                    self.roots.values(),
                    key=lambda item: (item.order, item.root_id),
                )
            ]
            return {
                "schema": "soyrootbio.editor-state/v1",
                "baseline_fingerprint": self.baseline_fingerprint,
                "source_output_dir": str(self.output_dir),
                "session_dir": str(self.session_dir),
                "mesh": {
                    "vertex_count": self.mesh.vertex_count,
                    "face_count": self.mesh.face_count,
                    "root_label_revision": self.label_revision,
                    "url": "/api/mesh",
                    "labels_url": "/api/mesh-labels",
                    "bounds": {
                        "minimum": self.mesh_minimum.tolist(),
                        "maximum": self.mesh_maximum.tolist(),
                    },
                    "render_origin": self.render_origin.tolist(),
                },
                "roots": roots,
                "root_count": len(roots),
                "point_patches": point_patches,
                "point_patch_count": len(point_patches),
                "can_undo": bool(self._history),
                "can_redo": bool(self._redo),
                "operation_count": len(self._history),
                "supported_operations": [
                    "create_root",
                    "split_root",
                    "merge_roots",
                    "assign_points",
                    "reconnect_root",
                    "reparent_root",
                    "delete_root",
                    "redraw_root",
                    "correct_root_order",
                ],
                "hardware": self._hardware_public(),
            }

    def apply_operation(
        self,
        operation_type: str,
        arguments: dict[str, Any],
        *,
        operation_id: str | None = None,
        timestamp: str | None = None,
        sequence: int | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            sequence_before = self._sequence
            label_revision_before = self.label_revision
            redo_before = list(self._redo)
            operation = Operation(
                operation_id=operation_id or str(uuid4()),
                sequence=int(sequence if sequence is not None else self._sequence + 1),
                timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
                type=str(operation_type),
                arguments=dict(arguments),
            )
            undo = self._execute(operation)
            self._history.append(_HistoryFrame(operation=operation, undo=undo))
            self._redo.clear()
            self._sequence = max(self._sequence, operation.sequence)
            if persist:
                try:
                    self._append_log(
                        {"event": "apply", "operation": operation.to_dict()}
                    )
                except Exception:
                    self._history.pop()
                    self._restore(undo)
                    self._redo = redo_before
                    self._sequence = sequence_before
                    self.label_revision = label_revision_before
                    self._remove_created_blobs(undo.created_blobs)
                    raise
            return {
                "operation": operation.to_dict(),
                "state": self.public_state(),
            }

    def labels_snapshot(self) -> tuple[bytes, int]:
        """Return a consistent binary label snapshot for the renderer."""

        with self._lock:
            return (
                self.mesh.root_labels.astype("<i4", copy=False).tobytes(order="C"),
                self.label_revision,
            )

    def point_patch_indices_snapshot(
        self,
        patch_id: str,
        *,
        expected_revision: int,
    ) -> tuple[bytes, int, int]:
        """Return one connected uncertain/unassigned patch as uint32 indices."""

        with self._lock:
            if int(expected_revision) != self.label_revision:
                raise EditorRevisionConflict(
                    "Point patches changed after this selection; choose the patch again."
                )
            self._refresh_point_patch_cache()
            indices = self._point_patch_indices.get(str(patch_id))
            if indices is None:
                raise EditorValidationError(f"Unknown point patch: {patch_id}")
            canonical = np.asarray(indices, dtype="<u4")
            return canonical.tobytes(order="C"), len(canonical), self.label_revision

    def operation_log_text(self) -> str:
        with self._lock:
            return (
                self.log_path.read_text(encoding="utf-8")
                if self.log_path.exists()
                else ""
            )

    def undo(self, *, persist: bool = True) -> dict[str, Any]:
        with self._lock:
            if not self._history:
                raise EditorValidationError("There is no operation to undo.")
            roots_before = self._clone_roots(self.roots)
            next_label_before = self._next_numeric_label
            sequence_before = self._sequence
            label_revision_before = self.label_revision
            frame = self._history.pop()
            current_labels = self.mesh.root_labels[frame.undo.changed_indices].copy()
            current_assignments = self.mesh.assignment_states[
                frame.undo.changed_indices
            ].copy()
            self._restore(frame.undo)
            self._redo.append(frame.operation)
            self._sequence += 1
            if persist:
                try:
                    self._append_log(
                        {
                            "event": "undo",
                            "sequence": self._sequence,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "target_operation_id": frame.operation.operation_id,
                        }
                    )
                except Exception:
                    self.roots = roots_before
                    self._next_numeric_label = next_label_before
                    if len(frame.undo.changed_indices):
                        self.mesh.root_labels[
                            frame.undo.changed_indices
                        ] = current_labels
                        self.mesh.assignment_states[
                            frame.undo.changed_indices
                        ] = current_assignments
                    self._redo.pop()
                    self._history.append(frame)
                    self._sequence = sequence_before
                    self.label_revision = label_revision_before
                    raise
            return {
                "undone_operation_id": frame.operation.operation_id,
                "state": self.public_state(),
            }

    def redo(self, *, persist: bool = True) -> dict[str, Any]:
        with self._lock:
            if not self._redo:
                raise EditorValidationError("There is no operation to redo.")
            sequence_before = self._sequence
            label_revision_before = self.label_revision
            operation = self._redo.pop()
            undo = self._execute(operation)
            self._history.append(_HistoryFrame(operation=operation, undo=undo))
            self._sequence += 1
            if persist:
                try:
                    self._append_log(
                        {
                            "event": "redo",
                            "sequence": self._sequence,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "target_operation_id": operation.operation_id,
                        }
                    )
                except Exception:
                    self._history.pop()
                    self._restore(undo)
                    self._redo.append(operation)
                    self._sequence = sequence_before
                    self.label_revision = label_revision_before
                    self._remove_created_blobs(undo.created_blobs)
                    raise
            return {
                "redone_operation_id": operation.operation_id,
                "state": self.public_state(),
            }

    def export_materialised(self, target_dir: str | Path | None = None) -> Path:
        """Write edited artefacts without altering the automatic bundle."""

        with self._lock:
            target = (
                Path(target_dir).resolve()
                if target_dir is not None
                else self.session_dir / "materialised"
            )
            target.mkdir(parents=True, exist_ok=True)
            hierarchy = {
                "schema": "soyrootbio.edited-root-hierarchy/v1",
                "baseline_fingerprint": self.baseline_fingerprint,
                "coordinate_space": "source_coordinates",
                "coordinate_unit": "mesh_unit",
                "roots": [
                    {
                        "root_id": root.root_id,
                        "parent_id": root.parent_id,
                        "root_order": root.order,
                        "numeric_label": root.numeric_label,
                        "valid": True,
                        "confidence": root.confidence,
                        "qc_flags": root.qc_flags,
                        "insertion_index": root.insertion_index,
                        "insertion_point": (
                            None
                            if root.insertion_point is None
                            else root.insertion_point.tolist()
                        ),
                        "polyline": root.points.tolist(),
                        "order_overridden": root.order_overridden,
                        "source_root_ids": root.source_root_ids,
                    }
                    for root in sorted(
                        self.roots.values(),
                        key=lambda item: (item.order, item.root_id),
                    )
                ],
            }
            (target / "edited_root_hierarchy.json").write_text(
                json.dumps(hierarchy, indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            trait_frame = pd.DataFrame(
                [root.traits for root in self.roots.values()]
            ).sort_values(["root_order", "root_id"])
            trait_frame.to_csv(target / "edited_root_traits.csv", index=False)
            label_rows = [
                {
                    "numeric_label": -2,
                    "root_id": "uncertain",
                    "parent_id": "",
                    "root_order": "",
                    "color_rgb": "250,122,13",
                },
                {
                    "numeric_label": -1,
                    "root_id": "unassigned",
                    "parent_id": "",
                    "root_order": "",
                    "color_rgb": "140,140,140",
                },
            ]
            for root in sorted(
                self.roots.values(),
                key=lambda item: item.numeric_label,
            ):
                rgb = np.round(order_color(root.order) * 255).astype(np.uint8)
                label_rows.append(
                    {
                        "numeric_label": root.numeric_label,
                        "root_id": root.root_id,
                        "parent_id": root.parent_id or "",
                        "root_order": root.order,
                        "color_rgb": ",".join(str(int(value)) for value in rgb),
                    }
                )
            pd.DataFrame(label_rows).to_csv(
                target / "edited_root_label_map.csv",
                index=False,
            )

            orders = np.full(self.mesh.vertex_count, 255, dtype=np.uint8)
            assignment_states = np.zeros(self.mesh.vertex_count, dtype=np.uint8)
            colors = np.full((self.mesh.vertex_count, 3), 140, dtype=np.uint8)
            orders[self.mesh.root_labels == -2] = 254
            assignment_states[self.mesh.root_labels == -2] = 2
            colors[self.mesh.root_labels == -2] = np.array([250, 122, 13], dtype=np.uint8)
            for root in self.roots.values():
                mask = self.mesh.root_labels == root.numeric_label
                orders[mask] = np.uint8(min(max(root.order, 0), 253))
                assignment_states[mask] = 1
                colors[mask] = np.round(order_color(root.order) * 255).astype(np.uint8)
            write_labeled_ply(
                target / "edited_segmented_root_structure.ply",
                self.mesh.positions,
                triangles=self.mesh.triangles,
                colors=colors,
                root_ids=self.mesh.root_labels,
                root_orders=orders,
                assignment_states=assignment_states,
            )

            primary = self.roots[PRIMARY_ID]
            laterals = self._root_paths()
            write_rsml(
                target / "edited_root_system.rsml",
                primary.points,
                laterals,
                trait_frame,
                {"source": str(self.output_dir), "editor": True},
            )
            if self.log_path.exists():
                shutil.copy2(self.log_path, target / "operations.jsonl")
            else:
                (target / "operations.jsonl").write_text("", encoding="utf-8")
            export_blob_dir = target / "blobs"
            shutil.copytree(self.blob_dir, export_blob_dir, dirs_exist_ok=True)
            export_manifest = {
                "schema": "soyrootbio.editor-export/v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "baseline_fingerprint": self.baseline_fingerprint,
                "source_output_dir": str(self.output_dir),
                "operation_blob_directory": "blobs",
                "operation_blob_count": sum(
                    1 for path in export_blob_dir.iterdir() if path.is_file()
                ),
                "active_operation_ids": [
                    frame.operation.operation_id for frame in self._history
                ],
            }
            (target / "manifest.json").write_text(
                json.dumps(export_manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return target

    # ------------------------------------------------------------------
    # Operation execution
    # ------------------------------------------------------------------
    def _execute(self, operation: Operation) -> _UndoRecord:
        roots_before = self._clone_roots(self.roots)
        next_label_before = self._next_numeric_label
        self._pending_index_chunks = []
        self._pending_old_label_chunks = []
        self._pending_old_assignment_chunks = []
        self._pending_created_blobs = []
        try:
            dispatcher = {
                "create_root": self._create_root,
                "split_root": self._split_root,
                "merge_roots": self._merge_roots,
                "assign_points": self._assign_points,
                "reconnect_root": self._reconnect_root,
                "reparent_root": self._reparent_root,
                "delete_root": self._delete_root,
                "redraw_root": self._redraw_root,
                "correct_root_order": self._correct_root_order,
            }
            action = dispatcher.get(operation.type)
            if action is None:
                raise EditorValidationError(f"Unsupported editor operation: {operation.type}")
            action(operation)
            self._refresh_attachments()
            self._validate_state()
            self._recompute_traits()
        except Exception:
            self.roots = roots_before
            self._next_numeric_label = next_label_before
            self._restore_pending_labels()
            self._remove_created_blobs(tuple(self._pending_created_blobs))
            raise

        if self._pending_index_chunks:
            indices = np.concatenate(self._pending_index_chunks).astype(np.int64, copy=False)
            labels = np.concatenate(self._pending_old_label_chunks).astype(np.int32, copy=False)
            assignments = np.concatenate(self._pending_old_assignment_chunks).astype(np.uint8, copy=False)
            # The same index can be touched twice in a compound operation.
            # Preserve only its earliest value so undo restores the exact state.
            unique, first = np.unique(indices, return_index=True)
            order = np.argsort(unique)
            indices = unique[order]
            labels = labels[first[order]]
            assignments = assignments[first[order]]
            self.label_revision += 1
        else:
            indices = np.empty(0, dtype=np.int64)
            labels = np.empty(0, dtype=np.int32)
            assignments = np.empty(0, dtype=np.uint8)
        return _UndoRecord(
            roots_before=roots_before,
            next_numeric_label_before=next_label_before,
            changed_indices=indices,
            changed_labels_before=labels,
            changed_assignment_before=assignments,
            created_blobs=tuple(self._pending_created_blobs),
        )

    def _create_root(self, operation: Operation) -> None:
        args = operation.arguments
        parent = self._root(str(args.get("parent_id", "")))
        if parent.order >= 253:
            raise EditorValidationError(
                "A child cannot be created below a root at maximum order 253."
            )
        drawn_points = np.asarray(args.get("points"), dtype=float)
        if (
            drawn_points.ndim != 2
            or drawn_points.shape[1] != 3
            or len(drawn_points) < 2
            or not np.all(np.isfinite(drawn_points))
        ):
            raise EditorValidationError(
                "A new root needs at least two finite XYZ path points."
            )
        if not np.any(np.linalg.norm(np.diff(drawn_points, axis=0), axis=1) > 1e-12):
            raise EditorValidationError("A new root path must have positive length.")

        new_id = str(args.get("new_root_id") or f"root-manual-{uuid4().hex[:12]}")
        if new_id in self.roots:
            raise EditorValidationError(f"Root ID already exists: {new_id}")
        operation.arguments["new_root_id"] = new_id

        parent_tree = cKDTree(parent.points)
        _, insertion_index = parent_tree.query(drawn_points[0], k=1)
        insertion_index = int(insertion_index)
        insertion_point = np.asarray(
            parent.points[insertion_index],
            dtype=float,
        ).copy()
        recorded_attachment = args.get("resolved_attachment")
        if recorded_attachment is not None:
            if not isinstance(recorded_attachment, dict):
                raise EditorValidationError(
                    "Recorded new-root attachment must be an object."
                )
            recorded_parent = str(recorded_attachment.get("parent_id", ""))
            try:
                recorded_index = int(
                    recorded_attachment.get("insertion_index", -1)
                )
            except (TypeError, ValueError) as exc:
                raise EditorValidationError(
                    "Recorded new-root insertion index must be an integer."
                ) from exc
            recorded_position = _point(
                recorded_attachment.get("position"),
                "Recorded new-root attachment",
            )
            if (
                recorded_parent != parent.root_id
                or recorded_index != insertion_index
                or not np.allclose(
                    recorded_position,
                    insertion_point,
                    rtol=1e-7,
                    atol=1e-7,
                )
            ):
                raise EditorValidationError(
                    "Recorded new-root attachment no longer matches its parent."
                )
        if np.allclose(
            drawn_points[0],
            insertion_point,
            rtol=1e-7,
            atol=1e-7,
        ):
            root_points = drawn_points.copy()
            root_points[0] = insertion_point
        else:
            root_points = np.vstack([insertion_point, drawn_points])

        indices: np.ndarray
        if "indices_blob" in args:
            indices = self._load_index_blob(str(args["indices_blob"]))
            expected_count = args.get("resolved_point_count")
            if expected_count is not None and int(expected_count) != len(indices):
                raise EditorValidationError(
                    "Created-root index blob does not match its recorded point count."
                )
            expected_digest = args.get("indices_sha256")
            if (
                expected_digest is not None
                and str(expected_digest) != self._indices_sha256(indices)
            ):
                raise EditorValidationError(
                    "Created-root index blob failed its integrity check."
                )
        elif "indices" in args:
            indices = self._coerce_vertex_indices(args["indices"])
            self._validate_vertex_indices(indices)
        else:
            radius = float(args.get("claim_radius", 0.0))
            if not math.isfinite(radius) or radius <= 0:
                raise EditorValidationError(
                    "The new-root point-claim radius must be positive."
                )
            indices = self._unassigned_indices_near_polyline(root_points, radius)

        self._validate_vertex_indices(indices)
        indices = np.unique(indices).astype(np.int64, copy=False)
        if not len(indices):
            raise EditorValidationError(
                "No unassigned mesh points fall within the new root path radius."
            )
        if np.any(self.mesh.root_labels[indices] != -1) or np.any(
            self.mesh.assignment_states[indices] != 0
        ):
            raise EditorValidationError(
                "A new root can claim only currently unassigned mesh points."
            )
        if "indices_blob" not in args:
            blob = self._store_index_blob(indices, operation.operation_id)
            operation.arguments.pop("indices", None)
            operation.arguments["indices_blob"] = blob
            operation.arguments["resolved_point_count"] = int(len(indices))
            operation.arguments["indices_sha256"] = self._indices_sha256(indices)

        new_root = RootNode(
            root_id=new_id,
            parent_id=parent.root_id,
            order=parent.order + 1,
            points=np.asarray(root_points, dtype=float),
            numeric_label=self._allocate_numeric_label(),
            traits={"root_id": new_id},
            confidence=0.0,
            qc_flags=["manual_created_from_unassigned"],
            insertion_point=insertion_point,
            insertion_index=insertion_index,
            order_overridden=False,
            source_root_ids=[],
        )
        self.roots[new_id] = new_root
        self._set_labels(indices, new_root.numeric_label)
        operation.arguments["resolved_attachment"] = {
            "parent_id": parent.root_id,
            "insertion_index": insertion_index,
            "position": insertion_point.tolist(),
        }

    def _split_root(self, operation: Operation) -> None:
        args = operation.arguments
        root = self._root(str(args.get("root_id", "")))
        index = self._polyline_index(root, args)
        if index <= 0 or index >= len(root.points) - 1:
            raise EditorValidationError("Split point must be inside the root polyline.")
        new_id = str(args.get("new_root_id") or f"root-manual-{uuid4().hex[:12]}")
        if new_id in self.roots:
            raise EditorValidationError(f"Root ID already exists: {new_id}")
        # Generated IDs are part of the operation, not incidental runtime state.
        # Persisting the resolved value makes log replay bit-for-bit deterministic
        # and lets later operations safely refer to the new root.
        operation.arguments["new_root_id"] = new_id

        original_points = root.points.copy()
        proximal = original_points[: index + 1]
        distal = original_points[index:]
        root.points = proximal
        new_parent = root.root_id
        new_order = root.order + 1
        new_root = RootNode(
            root_id=new_id,
            parent_id=new_parent,
            order=new_order,
            points=distal,
            numeric_label=self._allocate_numeric_label(),
            traits=dict(root.traits),
            confidence=0.0,
            qc_flags=list(dict.fromkeys(root.qc_flags + ["manual_split"])),
            insertion_point=distal[0].copy(),
            insertion_index=None,
            order_overridden=False,
            source_root_ids=[root.root_id],
        )
        self.roots[new_id] = new_root

        assigned = np.flatnonzero(self.mesh.root_labels == root.numeric_label)
        if len(assigned):
            points = self.mesh.positions[assigned]
            proximal_tree = cKDTree(proximal)
            distal_tree = cKDTree(distal)
            proximal_distance, _ = proximal_tree.query(points, k=1)
            distal_distance, _ = distal_tree.query(points, k=1)
            move = assigned[distal_distance < proximal_distance]
            self._set_labels(move, new_root.numeric_label)

        for child in self.roots.values():
            if child.root_id in {root.root_id, new_id} or child.parent_id != root.root_id:
                continue
            start = child.points[0]
            d_proximal = float(cKDTree(proximal).query(start, k=1)[0])
            d_distal = float(cKDTree(distal).query(start, k=1)[0])
            if d_distal < d_proximal:
                child.parent_id = new_id

    def _merge_roots(self, operation: Operation) -> None:
        args = operation.arguments
        keep = self._root(str(args.get("root_id", "")))
        remove = self._root(str(args.get("other_root_id", "")))
        if keep.root_id == remove.root_id:
            raise EditorValidationError("Choose two different roots to merge.")
        if keep.root_id == PRIMARY_ID or remove.root_id == PRIMARY_ID:
            raise EditorValidationError("The primary root cannot be merged; redraw it instead.")

        keep_parent_before = keep.parent_id
        remove_parent_before = remove.parent_id
        keep_is_direct_parent = remove_parent_before == keep.root_id
        remove_is_direct_parent = keep_parent_before == remove.root_id
        if (
            self._is_ancestor(keep.root_id, remove.root_id)
            and not keep_is_direct_parent
        ) or (
            self._is_ancestor(remove.root_id, keep.root_id)
            and not remove_is_direct_parent
        ):
            raise EditorValidationError(
                "Roots separated by intermediate descendants cannot be merged directly; "
                "reconnect or reparent them first."
            )

        combinations = [
            (
                float(np.linalg.norm(keep.points[-1] - remove.points[0])),
                keep.points,
                remove.points,
                keep,
                remove,
            ),
            (
                float(np.linalg.norm(remove.points[-1] - keep.points[0])),
                remove.points,
                keep.points,
                remove,
                keep,
            ),
        ]
        join_gap, first, second, base_source, distal_source = min(
            combinations,
            key=lambda item: item[0],
        )
        segment_lengths = np.concatenate(
            [
                np.linalg.norm(np.diff(keep.points, axis=0), axis=1),
                np.linalg.norm(np.diff(remove.points, axis=0), axis=1),
            ]
        )
        positive_steps = segment_lengths[segment_lengths > 1e-12]
        typical_step = (
            float(np.median(positive_steps))
            if len(positive_steps)
            else 0.0
        )
        merge_gap_limit = max(
            typical_step * 8.0,
            0.02 * (path_length(keep.points) + path_length(remove.points)),
            1e-6,
        )
        if join_gap > merge_gap_limit:
            raise EditorValidationError(
                "The selected roots do not form a directed tip-to-start join "
                f"(gap {join_gap:.6g}, allowed {merge_gap_limit:.6g}). "
                "Reconnect or redraw the paths before merging."
            )
        operation.arguments["resolved_join"] = {
            "proximal_root_id": base_source.root_id,
            "distal_root_id": distal_source.root_id,
            "gap": join_gap,
            "gap_limit": merge_gap_limit,
        }
        if np.linalg.norm(first[-1] - second[0]) <= 1e-9:
            merged = np.vstack([first, second[1:]])
        else:
            merged = np.vstack([first, second])
        keep.points = np.asarray(merged, dtype=float)
        if keep_is_direct_parent:
            keep.parent_id = keep_parent_before
        elif remove_is_direct_parent:
            keep.parent_id = remove_parent_before
        else:
            keep.parent_id = base_source.parent_id
        keep.order = min(keep.order, remove.order)
        keep.confidence = 0.0
        keep.qc_flags = list(
            dict.fromkeys(keep.qc_flags + remove.qc_flags + ["manual_merge"])
        )
        keep.source_root_ids = list(
            dict.fromkeys(keep.source_root_ids + remove.source_root_ids + [remove.root_id])
        )

        move = np.flatnonzero(self.mesh.root_labels == remove.numeric_label)
        self._set_labels(move, keep.numeric_label)
        for child in self.roots.values():
            if child.root_id != keep.root_id and child.parent_id == remove.root_id:
                child.parent_id = keep.root_id
        del self.roots[remove.root_id]
        if keep.parent_id in {keep.root_id, remove.root_id}:
            raise EditorValidationError("Merge would create a self-parent relationship.")
        self._recalculate_descendant_orders(keep.root_id)

    def _assign_points(self, operation: Operation) -> None:
        args = operation.arguments
        root = self._root(str(args.get("root_id", "")))
        indices: np.ndarray
        if "indices_blob" in args:
            indices = self._load_index_blob(str(args["indices_blob"]))
            expected_count = args.get("resolved_point_count")
            if expected_count is not None and int(expected_count) != len(indices):
                raise EditorValidationError(
                    "Operation index blob does not match its recorded point count."
                )
            expected_digest = args.get("indices_sha256")
            if (
                expected_digest is not None
                and str(expected_digest) != self._indices_sha256(indices)
            ):
                raise EditorValidationError(
                    "Operation index blob failed its integrity check."
                )
        elif "indices" in args:
            indices = self._coerce_vertex_indices(args["indices"])
            self._validate_vertex_indices(indices)
            blob = self._store_index_blob(indices, operation.operation_id)
            operation.arguments.pop("indices", None)
            operation.arguments["indices_blob"] = blob
            operation.arguments["resolved_point_count"] = int(len(indices))
            operation.arguments["indices_sha256"] = self._indices_sha256(indices)
        elif "positions" in args:
            positions = np.asarray(args.get("positions"), dtype=float)
            if (
                positions.ndim != 2
                or positions.shape[1] != 3
                or not 1 <= len(positions) <= 20_000
                or not np.all(np.isfinite(positions))
            ):
                raise EditorValidationError(
                    "An assignment stroke needs 1 to 20,000 finite XYZ points."
                )
            radius = float(args.get("radius", 0.0))
            if not math.isfinite(radius) or radius <= 0:
                raise EditorValidationError("Assignment brush radius must be positive.")
            indices = self._indices_near_polyline(positions, radius)
            indices = np.unique(indices).astype(np.int64, copy=False)
            blob = self._store_index_blob(indices, operation.operation_id)
            operation.arguments.pop("positions", None)
            operation.arguments["stroke_point_count"] = int(len(positions))
            operation.arguments["indices_blob"] = blob
            operation.arguments["resolved_point_count"] = int(len(indices))
            operation.arguments["indices_sha256"] = self._indices_sha256(indices)
        else:
            position = _point(args.get("position"), "Assignment position")
            radius = float(args.get("radius", 0.0))
            if not math.isfinite(radius) or radius <= 0:
                raise EditorValidationError("Assignment brush radius must be positive.")
            if self._mesh_tree is None:
                self._mesh_tree = cKDTree(self.mesh.positions)
            indices = np.asarray(
                self._mesh_tree.query_ball_point(position, radius),
                dtype=np.int64,
            )
            indices = np.unique(indices).astype(np.int64, copy=False)
            blob = self._store_index_blob(indices, operation.operation_id)
            operation.arguments.pop("indices", None)
            operation.arguments["indices_blob"] = blob
            operation.arguments["resolved_point_count"] = int(len(indices))
            operation.arguments["indices_sha256"] = self._indices_sha256(indices)
        self._validate_vertex_indices(indices)
        self._set_labels(indices, root.numeric_label)

    def _reconnect_root(self, operation: Operation) -> None:
        source = self._root(str(operation.arguments.get("root_id", "")))
        target = self._root(str(operation.arguments.get("target_root_id", "")))
        self._validate_new_parent(source.root_id, target.root_id)
        nearest = self._nearest_polyline_point(
            target.points,
            operation.arguments.get("position", source.points[0]),
        )
        source.points = source.points.copy()
        source.points[0] = nearest
        source.parent_id = target.root_id
        source.insertion_point = nearest.copy()
        source.insertion_index = int(
            cKDTree(target.points).query(nearest, k=1)[1]
        )
        source.confidence = 0.0
        source.qc_flags = list(
            dict.fromkeys(source.qc_flags + ["manual_reconnect"])
        )
        source.order_overridden = False
        source.order = target.order + 1
        self._recalculate_descendant_orders(source.root_id)

    def _reparent_root(self, operation: Operation) -> None:
        source = self._root(str(operation.arguments.get("root_id", "")))
        target = self._root(str(operation.arguments.get("new_parent_id", "")))
        self._validate_new_parent(source.root_id, target.root_id)
        source.parent_id = target.root_id
        source.insertion_point = self._nearest_polyline_point(
            target.points,
            operation.arguments.get("position", source.points[0]),
        )
        source.insertion_index = int(
            cKDTree(target.points).query(source.insertion_point, k=1)[1]
        )
        source.points = source.points.copy()
        source.points[0] = source.insertion_point
        source.confidence = 0.0
        source.qc_flags = list(
            dict.fromkeys(source.qc_flags + ["manual_reparent"])
        )
        source.order_overridden = False
        source.order = target.order + 1
        self._recalculate_descendant_orders(source.root_id)

    def _delete_root(self, operation: Operation) -> None:
        root = self._root(str(operation.arguments.get("root_id", "")))
        if root.root_id == PRIMARY_ID:
            raise EditorValidationError("The primary root cannot be deleted.")
        parent_id = root.parent_id
        for child in self.roots.values():
            if child.parent_id == root.root_id:
                child.parent_id = parent_id
                child.order = max(1, root.order)
                child.qc_flags = list(
                    dict.fromkeys(child.qc_flags + ["manual_parent_deleted"])
                )
        assigned = np.flatnonzero(self.mesh.root_labels == root.numeric_label)
        self._set_labels(assigned, -1)
        del self.roots[root.root_id]
        if parent_id:
            self._recalculate_descendant_orders(parent_id)

    def _redraw_root(self, operation: Operation) -> None:
        root = self._root(str(operation.arguments.get("root_id", "")))
        points = np.asarray(operation.arguments.get("points"), dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or len(points) < 2
            or not np.all(np.isfinite(points))
        ):
            raise EditorValidationError("A redrawn root needs at least two finite XYZ points.")
        root.points = points
        root.insertion_point = points[0].copy() if root.parent_id else None
        root.confidence = 0.0
        root.qc_flags = list(
            dict.fromkeys(root.qc_flags + ["manual_redraw"])
        )

    def _correct_root_order(self, operation: Operation) -> None:
        root = self._root(str(operation.arguments.get("root_id", "")))
        order = int(operation.arguments.get("root_order", -1))
        if root.root_id == PRIMARY_ID and order != 0:
            raise EditorValidationError("The primary root must remain order 0.")
        if root.root_id != PRIMARY_ID and not 1 <= order <= 253:
            raise EditorValidationError("Root order must be between 1 and 253.")
        root.order = order
        root.order_overridden = True
        root.qc_flags = list(
            dict.fromkeys(root.qc_flags + ["manual_order_correction"])
        )
        self._recalculate_descendant_orders(root.root_id, preserve_root=True)

    # ------------------------------------------------------------------
    # Loading, validation, traits, and helpers
    # ------------------------------------------------------------------
    def _load_trait_configuration(self) -> tuple[float, np.ndarray]:
        metadata_path = self.output_dir / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
        config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
        try:
            tip_window = float(config.get("tip_vector_window_mesh_units", 2.0))
        except (TypeError, ValueError):
            tip_window = 2.0
        if not math.isfinite(tip_window) or tip_window <= 0:
            tip_window = 2.0
        gravity_value = config.get("gravity", metadata.get("gravity_vector", [0.0, 0.0, -1.0]))
        gravity = np.asarray(gravity_value, dtype=float)
        if (
            gravity.shape != (3,)
            or not np.all(np.isfinite(gravity))
            or np.linalg.norm(gravity) <= 1e-12
        ):
            gravity = np.array([0.0, 0.0, -1.0], dtype=float)
        else:
            gravity = gravity / np.linalg.norm(gravity)
        return tip_window, gravity

    def _load_roots(self) -> dict[str, RootNode]:
        hierarchy = json.loads(
            (self.output_dir / "root_hierarchy.json").read_text(encoding="utf-8")
        )
        traits_frame = pd.read_csv(self.output_dir / "root_traits.csv")
        trait_lookup = {
            str(row["root_id"]): _row_dict(row)
            for _, row in traits_frame.iterrows()
        }
        label_frame = pd.read_csv(self.output_dir / "csv" / "root_label_map.csv")
        label_lookup = {
            str(row["root_id"]): int(row["numeric_label"])
            for _, row in label_frame.iterrows()
            if int(row["numeric_label"]) >= 0
        }
        roots: dict[str, RootNode] = {}
        for row in hierarchy.get("roots", []):
            root_id = str(row["root_id"])
            if root_id not in label_lookup:
                raise ValueError(f"root_label_map.csv is missing {root_id}")
            points = np.asarray(row["polyline"], dtype=float)
            roots[root_id] = RootNode(
                root_id=root_id,
                parent_id=(
                    None
                    if row.get("parent_id") in (None, "")
                    else str(row["parent_id"])
                ),
                order=int(row.get("root_order", 0)),
                points=points,
                numeric_label=label_lookup[root_id],
                traits=trait_lookup.get(root_id, {"root_id": root_id}),
                confidence=float(row.get("confidence", 0.0)),
                qc_flags=list(row.get("qc_flags", [])),
                insertion_point=(
                    None
                    if row.get("insertion_point") is None
                    else np.asarray(row["insertion_point"], dtype=float)
                ),
                insertion_index=row.get("insertion_index"),
                source_root_ids=[root_id],
            )
        if PRIMARY_ID not in roots:
            raise ValueError("root_hierarchy.json does not contain the primary root.")
        return roots

    def _recompute_traits(self) -> None:
        primary = self.roots[PRIMARY_ID]
        laterals = sorted(
            (root for root in self.roots.values() if root.root_id != PRIMARY_ID),
            key=lambda item: item.numeric_label,
        )
        compute_index = {primary.numeric_label: 0}
        for index, root in enumerate(laterals, start=1):
            compute_index[root.numeric_label] = index
        max_label = max(compute_index) if compute_index else 0
        lookup = np.full(max_label + 1, -1, dtype=np.int32)
        for numeric_label, index in compute_index.items():
            lookup[numeric_label] = index
        labels = self.mesh.root_labels
        mapped = np.full(len(labels), -1, dtype=np.int32)
        uncertain = labels == -2
        mapped[uncertain] = -2
        nonnegative = labels >= 0
        in_range = nonnegative & (labels <= max_label)
        mapped[in_range] = lookup[labels[in_range]]

        root_paths: list[RootPath] = []
        for root in laterals:
            parent = self.roots.get(root.parent_id or PRIMARY_ID, primary)
            root_paths.append(
                RootPath(
                    root_id=root.root_id,
                    parent_id=root.parent_id or PRIMARY_ID,
                    order=root.order,
                    confidence=root.confidence,
                    qc_flags=list(root.qc_flags),
                    points=root.points,
                    parent_points=parent.points,
                    insertion_point=root.insertion_point,
                    insertion_index=root.insertion_index,
                )
            )
        identity = Normalization(minimum=np.zeros(3), scale=1.0)
        traits = compute_traits(
            primary.points,
            root_paths,
            self.mesh.positions,
            mapped == 0,
            mapped,
            identity,
            full_points=self.mesh.positions,
            triangles=self.mesh.triangles,
            full_root_labels=mapped,
            primary_confidence=primary.confidence,
            primary_qc_flags=primary.qc_flags,
            tip_vector_window=self.tip_vector_window_mesh_units,
            gravity=self.gravity,
        )
        for _, row in traits.iterrows():
            root_id = str(row["root_id"])
            if root_id in self.roots:
                record = _row_dict(row)
                record["parent_id"] = self.roots[root_id].parent_id
                record["root_order"] = self.roots[root_id].order
                self.roots[root_id].traits = record

    def _validate_state(self) -> None:
        if PRIMARY_ID not in self.roots:
            raise EditorValidationError("The root system must contain a primary root.")
        primary = self.roots[PRIMARY_ID]
        if primary.parent_id is not None:
            raise EditorValidationError("The primary root cannot have a parent.")
        if primary.order != 0:
            raise EditorValidationError("The primary root must have root order 0.")
        labels: set[int] = set()
        for root in self.roots.values():
            if root.numeric_label < 0 or root.numeric_label in labels:
                raise EditorValidationError("Every root must have a unique non-negative numeric label.")
            labels.add(root.numeric_label)
            if root.parent_id is not None and root.parent_id not in self.roots:
                raise EditorValidationError(
                    f"{root.root_id} references missing parent {root.parent_id}."
                )
            if root.root_id != PRIMARY_ID and root.parent_id is None:
                raise EditorValidationError(
                    f"{root.root_id} is disconnected from the primary root."
                )
            if root.root_id != PRIMARY_ID and not 1 <= root.order <= 253:
                raise EditorValidationError(
                    f"{root.root_id} has invalid root order {root.order}."
                )
            if (
                root.points.ndim != 2
                or root.points.shape[1] != 3
                or len(root.points) < 2
                or not np.all(np.isfinite(root.points))
            ):
                raise EditorValidationError(f"{root.root_id} has an invalid polyline.")
            if root.root_id != PRIMARY_ID:
                parent = self.roots[root.parent_id]
                child_length = path_length(root.points)
                parent_length = path_length(parent.points)
                if child_length_exceeds_parent(child_length, parent_length):
                    raise EditorValidationError(
                        f"{root.root_id} has centreline length {child_length:.9g}, "
                        f"which exceeds parent {parent.root_id} length "
                        f"{parent_length:.9g}."
                    )
                if (
                    root.insertion_index is None
                    or not 0 <= int(root.insertion_index) < len(parent.points)
                ):
                    raise EditorValidationError(
                        f"{root.root_id} has an invalid parent insertion index."
                    )
                expected_insertion = parent.points[int(root.insertion_index)]
                if root.insertion_point is None or not np.allclose(
                    root.insertion_point,
                    expected_insertion,
                    rtol=1e-7,
                    atol=1e-7,
                ):
                    raise EditorValidationError(
                        f"{root.root_id} has a stale parent insertion point."
                    )
                if not np.allclose(
                    root.points[0],
                    expected_insertion,
                    rtol=1e-7,
                    atol=1e-7,
                ):
                    raise EditorValidationError(
                        f"{root.root_id} is geometrically detached from its parent."
                    )
        for root_id in self.roots:
            seen: set[str] = set()
            cursor: str | None = root_id
            while cursor is not None:
                if cursor in seen:
                    raise EditorValidationError("The edited hierarchy contains a cycle.")
                seen.add(cursor)
                cursor = self.roots[cursor].parent_id
        valid_labels = np.array(sorted(labels), dtype=np.int32)
        assigned = self.mesh.root_labels >= 0
        if np.any(~np.isin(self.mesh.root_labels[assigned], valid_labels)):
            raise EditorValidationError("Mesh vertices reference a deleted or unknown root.")

    def _validate_new_parent(self, root_id: str, parent_id: str) -> None:
        if root_id == PRIMARY_ID:
            raise EditorValidationError("The primary root cannot be reparented.")
        if root_id == parent_id:
            raise EditorValidationError("A root cannot be its own parent.")
        cursor: str | None = parent_id
        while cursor is not None:
            if cursor == root_id:
                raise EditorValidationError("This parent choice would create a cycle.")
            cursor = self.roots[cursor].parent_id

    def _is_ancestor(self, ancestor_id: str, descendant_id: str) -> bool:
        cursor = self.roots[descendant_id].parent_id
        while cursor is not None:
            if cursor == ancestor_id:
                return True
            cursor = self.roots[cursor].parent_id
        return False

    def _root_paths(self) -> list[RootPath]:
        primary = self.roots[PRIMARY_ID]
        output: list[RootPath] = []
        for root in sorted(
            (item for item in self.roots.values() if item.root_id != PRIMARY_ID),
            key=lambda item: (item.order, item.root_id),
        ):
            parent = self.roots.get(root.parent_id or PRIMARY_ID, primary)
            output.append(
                RootPath(
                    root_id=root.root_id,
                    parent_id=root.parent_id or PRIMARY_ID,
                    order=root.order,
                    confidence=root.confidence,
                    qc_flags=list(root.qc_flags),
                    points=root.points,
                    parent_points=parent.points,
                    insertion_point=root.insertion_point,
                    insertion_index=root.insertion_index,
                )
            )
        return output

    def _root(self, root_id: str) -> RootNode:
        try:
            return self.roots[root_id]
        except KeyError as exc:
            raise EditorValidationError(f"Unknown root ID: {root_id}") from exc

    def _polyline_index(self, root: RootNode, arguments: dict[str, Any]) -> int:
        if "node_index" in arguments:
            return int(arguments["node_index"])
        position = _point(arguments.get("position"), "Split position")
        return int(cKDTree(root.points).query(position, k=1)[1])

    @staticmethod
    def _nearest_polyline_point(points: np.ndarray, position: Any) -> np.ndarray:
        position_array = _point(position, "Attachment position")
        index = int(cKDTree(points).query(position_array, k=1)[1])
        return np.asarray(points[index], dtype=float).copy()

    def _allocate_numeric_label(self) -> int:
        value = self._next_numeric_label
        self._next_numeric_label += 1
        return value

    def _unassigned_indices_near_polyline(
        self,
        points: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """Return unassigned vertices inside an exact radius tube around a path."""

        unassigned = np.flatnonzero(self.mesh.root_labels == -1)
        return self._indices_near_polyline(
            points,
            radius,
            candidate_indices=unassigned,
        )

    def _indices_near_polyline(
        self,
        points: np.ndarray,
        radius: float,
        *,
        candidate_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return candidate vertices inside an exact radius tube around a path."""

        points = np.asarray(points, dtype=float)
        if candidate_indices is None:
            candidate_indices = np.arange(self.mesh.vertex_count, dtype=np.int64)
        else:
            candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        if not len(candidate_indices):
            return np.empty(0, dtype=np.int64)
        if len(points) == 1:
            if self._mesh_tree is None:
                self._mesh_tree = cKDTree(self.mesh.positions)
            nearby = np.asarray(
                self._mesh_tree.query_ball_point(points[0], radius),
                dtype=np.int64,
            )
            if len(candidate_indices) == self.mesh.vertex_count:
                return nearby
            return np.intersect1d(
                nearby,
                candidate_indices,
                assume_unique=False,
            ).astype(np.int64, copy=False)

        segment_vectors = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        sample_parts: list[np.ndarray] = []
        sample_count = 0
        for segment_index, length in enumerate(segment_lengths):
            subdivisions = max(1, int(math.ceil(float(length) / radius)))
            sample_count += subdivisions + (1 if segment_index == 0 else 0)
            if sample_count > 250_000:
                raise EditorValidationError(
                    "The claim radius is too small for this path; increase it "
                    "before creating the root."
                )
            fractions = np.linspace(
                0.0,
                1.0,
                subdivisions + 1,
                dtype=float,
            )
            if segment_index:
                fractions = fractions[1:]
            sample_parts.append(
                points[segment_index]
                + fractions[:, None] * segment_vectors[segment_index]
            )
        samples = np.vstack(sample_parts)

        # Samples are no farther than radius apart. Inflating this preliminary
        # query by sqrt(1 + 0.5^2) cannot omit a vertex within radius of a
        # segment; the exact segment-distance filter below removes extras.
        sample_tree = cKDTree(samples)
        approximate_distance, _ = sample_tree.query(
            self.mesh.positions[candidate_indices],
            k=1,
        )
        candidates = candidate_indices[
            approximate_distance <= radius * math.sqrt(1.25)
        ]
        if not len(candidates):
            return np.empty(0, dtype=np.int64)

        candidate_points = self.mesh.positions[candidates]
        minimum_squared = np.full(len(candidates), np.inf, dtype=float)
        for start, vector in zip(points[:-1], segment_vectors):
            squared_length = float(np.dot(vector, vector))
            if squared_length <= 1e-24:
                projected = np.broadcast_to(start, candidate_points.shape)
            else:
                fraction = np.clip(
                    ((candidate_points - start) @ vector) / squared_length,
                    0.0,
                    1.0,
                )
                projected = start + fraction[:, None] * vector
            squared_distance = np.sum(
                (candidate_points - projected) ** 2,
                axis=1,
            )
            minimum_squared = np.minimum(minimum_squared, squared_distance)
        return candidates[
            minimum_squared <= radius * radius * (1.0 + 1e-12)
        ].astype(np.int64, copy=False)

    def _set_labels(self, indices: np.ndarray, numeric_label: int) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        if not len(indices):
            return
        self._validate_vertex_indices(indices)
        self._pending_index_chunks.append(indices.copy())
        self._pending_old_label_chunks.append(self.mesh.root_labels[indices].copy())
        self._pending_old_assignment_chunks.append(
            self.mesh.assignment_states[indices].copy()
        )
        self.mesh.root_labels[indices] = int(numeric_label)
        self.mesh.assignment_states[indices] = 1 if numeric_label >= 0 else 0

    def _restore_pending_labels(self) -> None:
        for indices, labels, states in reversed(
            list(
                zip(
                    self._pending_index_chunks,
                    self._pending_old_label_chunks,
                    self._pending_old_assignment_chunks,
                )
            )
        ):
            self.mesh.root_labels[indices] = labels
            self.mesh.assignment_states[indices] = states

    def _restore(self, undo: _UndoRecord) -> None:
        self.roots = self._clone_roots(undo.roots_before)
        self._next_numeric_label = undo.next_numeric_label_before
        if len(undo.changed_indices):
            self.mesh.root_labels[undo.changed_indices] = undo.changed_labels_before
            self.mesh.assignment_states[
                undo.changed_indices
            ] = undo.changed_assignment_before
            self.label_revision += 1

    def _validate_vertex_indices(self, indices: np.ndarray) -> None:
        if indices.ndim != 1:
            raise EditorValidationError("Vertex indices must be a one-dimensional list.")
        if len(indices) and (
            int(np.min(indices)) < 0
            or int(np.max(indices)) >= self.mesh.vertex_count
        ):
            raise EditorValidationError("Point assignment contains an out-of-range vertex index.")

    @staticmethod
    def _coerce_vertex_indices(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            if value.ndim != 1 or value.dtype.kind not in {"i", "u"}:
                raise EditorValidationError(
                    "Vertex indices must be a one-dimensional integer list."
                )
            return np.asarray(value, dtype=np.int64)
        if not isinstance(value, (list, tuple)):
            raise EditorValidationError("Vertex indices must be an integer list.")
        if any(
            isinstance(item, bool) or not isinstance(item, (int, np.integer))
            for item in value
        ):
            raise EditorValidationError("Every vertex index must be an integer.")
        return np.asarray(value, dtype=np.int64)

    def _point_patch_public(self) -> list[dict[str, Any]]:
        self._refresh_point_patch_cache()
        return [
            {
                **summary,
                "bounds": {
                    "minimum": list(summary["bounds"]["minimum"]),
                    "maximum": list(summary["bounds"]["maximum"]),
                },
                "centroid": list(summary["centroid"]),
            }
            for summary in self._point_patch_summaries
        ]

    def _refresh_point_patch_cache(self) -> None:
        if self._point_patch_revision == self.label_revision:
            return
        if self._mesh_edges is None:
            triangles = np.asarray(self.mesh.triangles, dtype=np.int64)
            if len(triangles):
                starts = np.concatenate(
                    [triangles[:, 0], triangles[:, 1], triangles[:, 2]]
                )
                ends = np.concatenate(
                    [triangles[:, 1], triangles[:, 2], triangles[:, 0]]
                )
                keep = starts != ends
                self._mesh_edges = (
                    starts[keep].astype(np.int32, copy=False),
                    ends[keep].astype(np.int32, copy=False),
                )
            else:
                empty = np.empty(0, dtype=np.int32)
                self._mesh_edges = (empty, empty)

        edge_starts, edge_ends = self._mesh_edges
        summaries: list[dict[str, Any]] = []
        indices_by_id: dict[str, np.ndarray] = {}
        labels = self.mesh.root_labels
        for numeric_label, kind in ((-2, "uncertain"), (-1, "unassigned")):
            eligible = np.flatnonzero(labels == numeric_label).astype(
                np.int64,
                copy=False,
            )
            if not len(eligible):
                continue
            local_index = np.full(self.mesh.vertex_count, -1, dtype=np.int32)
            local_index[eligible] = np.arange(len(eligible), dtype=np.int32)
            valid_edges = (
                (labels[edge_starts] == numeric_label)
                & (labels[edge_ends] == numeric_label)
            )
            rows = local_index[edge_starts[valid_edges]]
            columns = local_index[edge_ends[valid_edges]]
            if len(rows):
                graph_rows = np.concatenate([rows, columns])
                graph_columns = np.concatenate([columns, rows])
                graph = coo_matrix(
                    (
                        np.ones(len(graph_rows), dtype=np.uint8),
                        (graph_rows, graph_columns),
                    ),
                    shape=(len(eligible), len(eligible)),
                ).tocsr()
                component_count, component_labels = connected_components(
                    graph,
                    directed=False,
                    return_labels=True,
                )
            else:
                component_count = len(eligible)
                component_labels = np.arange(len(eligible), dtype=np.int32)

            counts = np.bincount(
                component_labels,
                minlength=component_count,
            ).astype(np.int64, copy=False)
            positions = self.mesh.positions[eligible]
            coordinate_sums = np.column_stack(
                [
                    np.bincount(
                        component_labels,
                        weights=positions[:, axis],
                        minlength=component_count,
                    )
                    for axis in range(3)
                ]
            )
            centroids = coordinate_sums / counts[:, None]
            minimum = np.full((component_count, 3), np.inf, dtype=float)
            maximum = np.full((component_count, 3), -np.inf, dtype=float)
            for axis in range(3):
                np.minimum.at(minimum[:, axis], component_labels, positions[:, axis])
                np.maximum.at(maximum[:, axis], component_labels, positions[:, axis])
            anchors = np.full(
                component_count,
                np.iinfo(np.int64).max,
                dtype=np.int64,
            )
            np.minimum.at(anchors, component_labels, eligible)

            membership_order = np.argsort(component_labels, kind="stable")
            offsets = np.concatenate(
                [np.array([0], dtype=np.int64), np.cumsum(counts)]
            )
            for component in range(component_count):
                members = eligible[
                    membership_order[offsets[component] : offsets[component + 1]]
                ].astype(np.uint32, copy=False)
                patch_id = f"{kind}-{int(anchors[component])}"
                membership_digest = (
                    "sha256:"
                    + sha256(
                        np.asarray(members, dtype="<u4").tobytes(order="C")
                    ).hexdigest()
                )
                indices_by_id[patch_id] = members
                summaries.append(
                    {
                        "patch_id": patch_id,
                        "kind": kind,
                        "numeric_label": numeric_label,
                        "point_count": int(counts[component]),
                        "anchor_vertex_index": int(anchors[component]),
                        "centroid": centroids[component].tolist(),
                        "bounds": {
                            "minimum": minimum[component].tolist(),
                            "maximum": maximum[component].tolist(),
                        },
                        "membership_sha256": membership_digest,
                        "revision": self.label_revision,
                        "indices_url": (
                            f"/api/point-patches/{patch_id}/indices"
                            f"?revision={self.label_revision}"
                        ),
                    }
                )

        kind_order = {"uncertain": 0, "unassigned": 1}
        summaries.sort(
            key=lambda patch: (
                kind_order[str(patch["kind"])],
                -int(patch["point_count"]),
                int(patch["anchor_vertex_index"]),
            )
        )
        self._point_patch_summaries = summaries
        self._point_patch_indices = indices_by_id
        self._point_patch_revision = self.label_revision

    def _children_map(self) -> dict[str, list[str]]:
        children: dict[str, list[str]] = {}
        for root in self.roots.values():
            if root.parent_id is not None:
                children.setdefault(root.parent_id, []).append(root.root_id)
        for values in children.values():
            values.sort()
        return children

    def _refresh_attachments(self) -> None:
        """Snap every non-primary start to a concrete node on its current parent."""

        def depth(root: RootNode) -> int:
            value = 0
            cursor = root.parent_id
            seen = {root.root_id}
            while cursor is not None:
                if cursor in seen or cursor not in self.roots:
                    raise EditorValidationError(
                        "Cannot refresh attachments for an invalid hierarchy."
                    )
                seen.add(cursor)
                value += 1
                cursor = self.roots[cursor].parent_id
            return value

        for root in sorted(self.roots.values(), key=lambda item: (depth(item), item.root_id)):
            if root.root_id == PRIMARY_ID:
                root.parent_id = None
                root.insertion_point = None
                root.insertion_index = None
                continue
            parent = self.roots[root.parent_id]
            insertion_index = int(cKDTree(parent.points).query(root.points[0], k=1)[1])
            insertion_point = np.asarray(
                parent.points[insertion_index],
                dtype=float,
            ).copy()
            root.points = root.points.copy()
            root.points[0] = insertion_point
            root.insertion_point = insertion_point
            root.insertion_index = insertion_index

    def _recalculate_descendant_orders(
        self,
        root_id: str,
        *,
        preserve_root: bool = False,
    ) -> None:
        children = self._children_map()
        root = self.roots[root_id]
        if not preserve_root and root.parent_id is not None and not root.order_overridden:
            root.order = self.roots[root.parent_id].order + 1
        queue = [root_id]
        visited: set[str] = set()
        while queue:
            parent_id = queue.pop(0)
            if parent_id in visited:
                raise EditorValidationError(
                    "Order propagation encountered a cycle in the edited hierarchy."
                )
            visited.add(parent_id)
            parent = self.roots[parent_id]
            for child_id in children.get(parent_id, []):
                child = self.roots[child_id]
                if not child.order_overridden:
                    child.order = parent.order + 1
                queue.append(child_id)

    def _store_index_blob(self, indices: np.ndarray, operation_id: str) -> str:
        path = self.blob_dir / f"{operation_id}.npy"
        temporary = self.blob_dir / f".{operation_id}.{uuid4().hex}.tmp.npy"
        try:
            np.save(
                temporary,
                np.asarray(indices, dtype=np.int64),
                allow_pickle=False,
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self._pending_created_blobs.append(path)
        return str(path.relative_to(self.session_dir)).replace("\\", "/")

    def _load_index_blob(self, relative_path: str) -> np.ndarray:
        path = (self.session_dir / relative_path).resolve()
        if self.session_dir not in path.parents:
            raise EditorValidationError("Operation blob escapes the editor session directory.")
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)

    @staticmethod
    def _indices_sha256(indices: np.ndarray) -> str:
        canonical = np.asarray(indices, dtype="<i8")
        return f"sha256:{sha256(canonical.tobytes(order='C')).hexdigest()}"

    @staticmethod
    def _remove_created_blobs(paths: Iterable[Path]) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _root_public(self, root: RootNode, children: list[str]) -> dict[str, Any]:
        trait = root.traits
        return _json_safe(
            {
                "root_id": root.root_id,
                "numeric_label": root.numeric_label,
                "parent_id": root.parent_id,
                "children_ids": children,
                "root_order": root.order,
                "order_overridden": root.order_overridden,
                "polyline": root.points.tolist(),
                "insertion_point": (
                    root.points[0].tolist()
                    if root.insertion_point is None
                    else root.insertion_point.tolist()
                ),
                "tip_point": root.points[-1].tolist(),
                "length": trait.get("length"),
                "chord_length": trait.get("chord_length"),
                "tip_gravity_angle_deg": trait.get("tip_gravity_angle_deg"),
                "tip_start_gravity_angle_deg": trait.get(
                    "tip_start_gravity_angle_deg"
                ),
                "tip_primary_angle_deg": trait.get("tip_primary_angle_deg"),
                "mean_diameter": trait.get("mean_diameter"),
                "surface_area": trait.get("surface_area"),
                "volume": trait.get("volume"),
                "tortuosity": trait.get("tortuosity"),
                "point_count": trait.get("point_count"),
                "confidence": root.confidence,
                "qc_flags": root.qc_flags,
                "units": {
                    "length": trait.get("length_unit", "mesh_unit"),
                    "area": trait.get("area_unit", "mesh_unit^2"),
                    "volume": trait.get("volume_unit", "mesh_unit^3"),
                    "angle": "degree",
                },
            }
        )

    def _hardware_public(self) -> dict[str, Any]:
        gpus = [
            {
                "index": gpu.index,
                "name": gpu.name,
                "memory_total_bytes": gpu.memory_total_bytes,
                "driver_version": gpu.driver_version,
                "backend": gpu.backend,
                "is_discrete": True,
            }
            for gpu in self.hardware.gpus
        ]
        return {
            "logical_cpus": self.hardware.logical_cpus,
            "physical_cpus": self.hardware.physical_cpus,
            "total_memory_bytes": self.hardware.total_memory_bytes,
            "available_memory_bytes": self.hardware.available_memory_bytes,
            "gpus": gpus,
            "discrete_gpu_present": bool(gpus),
            "acceleration": {
                "rendering": "WebGL2 high-performance GPU",
                "spatial_picking": "BVH worker",
                "geometry_processing": "multicore CPU",
                "full_resolution_policy": "retain unless browser/device allocation fails",
            },
        }

    def _require_bundle(self) -> None:
        required = [
            "segmented_root_structure.ply",
            "root_hierarchy.json",
            "root_traits.csv",
            "csv/root_label_map.csv",
        ]
        missing = [name for name in required if not (self.output_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "Not a complete SoyRootBio output directory; missing "
                + ", ".join(missing)
            )

    def _baseline_fingerprint(self) -> str:
        digest = sha256()
        relatives = [
            "segmented_root_structure.ply",
            "root_hierarchy.json",
            "root_traits.csv",
            "csv/root_label_map.csv",
        ]
        if (self.output_dir / "metadata.json").is_file():
            relatives.append("metadata.json")
        for relative in relatives:
            path = self.output_dir / relative
            digest.update(relative.encode("utf-8"))
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    def _write_session_manifest(self) -> None:
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing.get("baseline_fingerprint") != self.baseline_fingerprint:
                raise EditorValidationError(
                    "Editor session belongs to a different automatic result."
                )
            return
        payload = {
            "schema": LOG_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_output_dir": str(self.output_dir),
            "baseline_fingerprint": self.baseline_fingerprint,
            "automatic_files_are_immutable": True,
            "trait_configuration": {
                "tip_vector_window_mesh_units": self.tip_vector_window_mesh_units,
                "gravity": self.gravity.tolist(),
            },
        }
        _atomic_write_text(
            self.manifest_path,
            json.dumps(payload, indent=2, ensure_ascii=False),
        )

    def _append_log(self, event: dict[str, Any]) -> None:
        envelope = {
            "schema": LOG_SCHEMA,
            "baseline_fingerprint": self.baseline_fingerprint,
            **event,
        }
        encoded = (
            json.dumps(
                _json_safe(envelope),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        original_size = self.log_path.stat().st_size if self.log_path.exists() else 0
        try:
            with self.log_path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                if self.log_path.exists():
                    with self.log_path.open("r+b") as handle:
                        handle.truncate(original_size)
                        handle.flush()
                        os.fsync(handle.fileno())
            except OSError:
                pass
            raise

    def _replay_log(self) -> None:
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if event.get("baseline_fingerprint") != self.baseline_fingerprint:
                    raise EditorValidationError(
                        "Operation log belongs to a different automatic result."
                    )
                event_type = event.get("event")
                if event_type == "apply":
                    payload = event["operation"]
                    self.apply_operation(
                        payload["type"],
                        payload.get("arguments", {}),
                        operation_id=payload["operation_id"],
                        timestamp=payload.get("timestamp"),
                        sequence=int(payload.get("sequence", self._sequence + 1)),
                        persist=False,
                    )
                elif event_type == "undo":
                    target_id = str(event.get("target_operation_id", ""))
                    if (
                        not self._history
                        or self._history[-1].operation.operation_id != target_id
                    ):
                        raise EditorValidationError(
                            "Undo event does not match the active history head."
                        )
                    self.undo(persist=False)
                    self._sequence = max(self._sequence, int(event.get("sequence", 0)))
                elif event_type == "redo":
                    target_id = str(event.get("target_operation_id", ""))
                    if (
                        not self._redo
                        or self._redo[-1].operation_id != target_id
                    ):
                        raise EditorValidationError(
                            "Redo event does not match the redo history head."
                        )
                    self.redo(persist=False)
                    self._sequence = max(self._sequence, int(event.get("sequence", 0)))
                else:
                    raise EditorValidationError(f"Unknown log event: {event_type}")
            except Exception as exc:
                raise EditorValidationError(
                    f"Invalid operation log at line {line_number}: {exc}"
                ) from exc

    @staticmethod
    def _clone_roots(roots: dict[str, RootNode]) -> dict[str, RootNode]:
        return {root_id: root.clone() for root_id, root in roots.items()}


def _point(value: Any, label: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise EditorValidationError(f"{label} must contain three finite coordinates.")
    return point


def _row_dict(row: pd.Series) -> dict[str, Any]:
    return {
        str(key): _json_scalar(value)
        for key, value in row.to_dict().items()
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return _json_scalar(value)


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
