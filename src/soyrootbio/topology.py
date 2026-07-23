from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geometry import tangent_vectors, vector_angle_degrees
from .types import Normalization, RootPath, TopologyReport


PRIMARY_ID = "primary"


def _reject_nonfinite_json(token: str):
    raise ValueError(f"Hierarchy correction contains non-finite JSON constant: {token}")


def repair_root_hierarchy(
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    *,
    d_bar: float,
) -> tuple[list[RootPath], TopologyReport]:
    """Orient, attach, validate, and deterministically label a root tree.

    Candidate tracing is allowed to be imperfect, but downstream traits are not
    allowed to consume a cyclic or dangling hierarchy.  Parents are selected
    from already established lower-order paths using attachment distance and
    tangent continuity, then orders are recomputed recursively from the primary
    root (order 0).
    """

    report = TopologyReport()
    primary_path = np.asarray(primary_path, dtype=float)
    tolerance = max(6.0 * float(d_bar), 0.008)
    available: dict[str, RootPath] = {
        PRIMARY_ID: RootPath(
            root_id=PRIMARY_ID,
            points=primary_path,
            order=0,
            parent_id="",
            confidence=1.0,
        )
    }
    provisional = sorted(
        [path for path in lateral_paths if len(path.points) >= 2],
        key=lambda item: (max(1, int(item.order)), str(item.parent_id), str(item.root_id)),
    )

    repaired: list[RootPath] = []
    old_to_current: dict[str, str] = {PRIMARY_ID: PRIMARY_ID}
    reassigned_roots: set[str] = set()
    for path in provisional:
        old_id = str(path.root_id)
        if path.raw_start_point is None:
            path.raw_start_point = np.asarray(path.points[0], dtype=float).copy()
        candidate_parents = [
            parent
            for parent in available.values()
            if parent.root_id == PRIMARY_ID or int(parent.order) < max(1, int(path.order))
        ]
        if not candidate_parents:
            candidate_parents = [available[PRIMARY_ID]]
        preferred_parent = old_to_current.get(str(path.parent_id), str(path.parent_id))
        parent, endpoint, parent_index, gap, branch_separation = _best_attachment(
            path,
            candidate_parents,
            preferred_parent=preferred_parent,
            tolerance=tolerance,
        )
        attachment_evidence = (
            np.asarray(path.raw_start_point, dtype=float).copy()
            if endpoint == 0 and path.raw_start_point is not None
            else np.asarray(path.points[-1], dtype=float).copy()
        )
        if endpoint == 1:
            path.points = path.points[::-1].copy()
            if path.node_indices is not None:
                path.node_indices = path.node_indices[::-1].copy()
            report.roots_reoriented += 1
        path.raw_start_point = attachment_evidence
        if parent.root_id != preferred_parent:
            reassigned_roots.add(old_id)
        path.parent_id = parent.root_id
        path.parent_points = parent.points
        path.insertion_index = int(parent_index)
        path.insertion_point = parent.points[int(parent_index)].copy()
        path.points[0] = path.insertion_point
        attachment_score = float(np.exp(-gap / max(tolerance, 1e-12)))
        length_score = float(1.0 - np.exp(-path.length / max(10.0 * d_bar, 1e-12)))
        support_score = float(min(1.0, len(path.covered_indices) / 30.0)) if path.covered_indices else 0.5
        trace_score = 0.5 if not np.isfinite(path.score) or path.score <= 0 else 1.0 - np.exp(-path.score / 100.0)
        path.confidence = float(
            np.clip(
                0.42 * attachment_score
                + 0.23 * branch_separation
                + 0.18 * length_score
                + 0.10 * support_score
                + 0.07 * trace_score,
                0.0,
                1.0,
            )
        )
        path.score_components.update(
            {
                "attachment": attachment_score,
                # Kept for compatibility with the v1 QC table.  For a branch
                # junction this is deliberately a separation/plausibility
                # score, not a reward for being collinear with its parent.
                "junction_tangent_continuity": branch_separation,
                "junction_branch_separation": branch_separation,
                "length_support": length_score,
                "point_support": support_score,
            }
        )
        _append_qc(path, gap=gap, tolerance=tolerance, d_bar=d_bar)
        if path.confidence < 0.55:
            report.low_confidence_roots += 1
        repaired.append(path)
        available[old_id] = path
        old_to_current[old_id] = old_id

    graph = _build_graph(repaired)
    for cycle in list(nx.simple_cycles(graph)):
        if not cycle:
            continue
        weakest_id = min(cycle, key=lambda root_id: _path_by_id(repaired, root_id).confidence)
        weakest = _path_by_id(repaired, weakest_id)
        weakest.parent_id = PRIMARY_ID
        weakest.parent_points = primary_path
        _, insertion_index = cKDTree(primary_path).query(weakest.points[0], k=1)
        weakest.insertion_index = int(insertion_index)
        weakest.insertion_point = primary_path[int(insertion_index)].copy()
        weakest.points[0] = weakest.insertion_point
        if "cycle_repaired" not in weakest.qc_flags:
            weakest.qc_flags.append("cycle_repaired")
        report.cycles_removed += 1
    reassigned_roots.update(
        _promote_base_attached_children(
            primary_path,
            repaired,
            d_bar=d_bar,
            tolerance=tolerance,
        )
    )
    report.parents_reassigned = len(reassigned_roots)
    _assign_recursive_orders(repaired)
    _assign_stable_ids(repaired)
    _refresh_parent_references(primary_path, repaired)
    errors = validate_root_tree(repaired)
    report.warnings.extend(errors)
    report.disconnected_roots = sum("missing parent" in error for error in errors)
    return repaired, report


def validate_root_tree(paths: Iterable[RootPath]) -> list[str]:
    """Return invariant violations; an empty result proves a rooted tree."""

    paths = list(paths)
    by_id = {path.root_id: path for path in paths}
    errors: list[str] = []
    if len(by_id) != len(paths):
        errors.append("root IDs are not unique")
    graph = nx.DiGraph()
    graph.add_node(PRIMARY_ID)
    for path in paths:
        if path.parent_id != PRIMARY_ID and path.parent_id not in by_id:
            errors.append(f"{path.root_id}: missing parent {path.parent_id}")
            continue
        graph.add_edge(path.parent_id, path.root_id)
        expected_order = 1 if path.parent_id == PRIMARY_ID else int(by_id[path.parent_id].order) + 1
        if int(path.order) != expected_order:
            errors.append(
                f"{path.root_id}: order {path.order} does not equal parent order + 1 ({expected_order})"
            )
        if path.insertion_point is None or path.insertion_index is None:
            errors.append(f"{path.root_id}: insertion location is missing")
    if not nx.is_directed_acyclic_graph(graph):
        errors.append("root hierarchy contains a cycle")
    unreachable = set(graph.nodes) - set(nx.descendants(graph, PRIMARY_ID)) - {PRIMARY_ID}
    for root_id in sorted(unreachable):
        errors.append(f"{root_id}: root is not connected to primary")
    return errors


def hierarchy_frame(
    paths: Iterable[RootPath],
    *,
    normalization: Normalization | None = None,
    primary_path: np.ndarray | None = None,
    primary_confidence: float = 1.0,
    primary_qc_flags: Iterable[str] = (),
) -> pd.DataFrame:
    columns = [
        "root_id",
        "parent_id",
        "root_order",
        "insertion_index",
        "insertion_x",
        "insertion_y",
        "insertion_z",
        "coordinate_unit",
        "confidence",
        "qc_flags",
    ]
    rows = []
    if primary_path is not None and len(primary_path):
        insertion = np.asarray(primary_path, dtype=float)[0]
        if normalization is not None:
            insertion = normalization.inverse_points(insertion[None, :])[0]
        rows.append(
            {
                "root_id": PRIMARY_ID,
                "parent_id": "",
                "root_order": 0,
                "insertion_index": 0,
                "insertion_x": float(insertion[0]),
                "insertion_y": float(insertion[1]),
                "insertion_z": float(insertion[2]),
                "coordinate_unit": "mesh_unit" if normalization is not None else "analysis_normalized",
                "confidence": float(primary_confidence),
                "qc_flags": ";".join(primary_qc_flags),
            }
        )
    for path in paths:
        insertion = np.asarray(path.insertion_point if path.insertion_point is not None else path.points[0])
        if normalization is not None:
            insertion = normalization.inverse_points(insertion[None, :])[0]
        rows.append(
            {
                "root_id": path.root_id,
                "parent_id": path.parent_id,
                "root_order": int(path.order),
                "insertion_index": path.insertion_index,
                "insertion_x": float(insertion[0]),
                "insertion_y": float(insertion[1]),
                "insertion_z": float(insertion[2]),
                "coordinate_unit": "mesh_unit" if normalization is not None else "analysis_normalized",
                "confidence": float(path.confidence),
                "qc_flags": ";".join(path.qc_flags),
            }
        )
    return pd.DataFrame.from_records(rows, columns=columns)


def write_editable_hierarchy(
    path: str | Path,
    primary_path: np.ndarray,
    lateral_paths: Iterable[RootPath],
    *,
    primary_confidence: float = 1.0,
    primary_qc_flags: Iterable[str] = (),
) -> Path:
    """Write the complete topology contract as editable JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    roots = [
        {
            "root_id": PRIMARY_ID,
            "parent_id": None,
            "root_order": 0,
            "valid": True,
            "editable": False,
            "confidence": float(primary_confidence),
            "qc_flags": list(primary_qc_flags),
            "polyline": np.asarray(primary_path, dtype=float).tolist(),
            "geometry_fingerprint": _polyline_fingerprint(primary_path),
        }
    ]
    for root in lateral_paths:
        roots.append(
            {
                "root_id": root.root_id,
                "parent_id": root.parent_id,
                "root_order": int(root.order),
                "valid": True,
                "editable": True,
                "confidence": float(root.confidence),
                "qc_flags": list(root.qc_flags),
                "insertion_index": root.insertion_index,
                "insertion_point": None if root.insertion_point is None else np.asarray(root.insertion_point).tolist(),
                "polyline": np.asarray(root.points, dtype=float).tolist(),
                "geometry_fingerprint": _polyline_fingerprint(root.points),
            }
        )
    payload = {
        "schema": "soyrootbio.root-hierarchy/v1",
        "coordinate_space": "source_coordinates",
        "coordinate_unit": "mesh_unit",
        "instructions": "Coordinates are in source_coordinates. The primary row is immutable; override it with GUI endpoints/soil/guides. For lateral rows, edit parent_id, valid, or polyline and pass this file back as a correction file.",
        "roots": roots,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path


def apply_hierarchy_corrections(
    primary_path: np.ndarray,
    lateral_paths: list[RootPath],
    correction_file: str | Path,
    *,
    normalization: Normalization | None = None,
) -> list[RootPath]:
    """Apply validated parent/order/polyline edits from an exported hierarchy."""

    payload = json.loads(
        Path(correction_file).read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json,
    )
    if not isinstance(payload, dict) or payload.get("schema") != "soyrootbio.root-hierarchy/v1":
        raise ValueError("Unsupported hierarchy correction schema; expected soyrootbio.root-hierarchy/v1")
    rows = payload.get("roots", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Hierarchy correction roots must be a list of objects")
    row_ids = [str(row.get("root_id", "")) for row in rows]
    if any(not root_id for root_id in row_ids):
        raise ValueError("Every hierarchy correction row must have a root_id")
    duplicate_ids = sorted({root_id for root_id in row_ids if row_ids.count(root_id) > 1})
    if duplicate_ids:
        raise ValueError("Hierarchy correction contains duplicate root IDs: " + ", ".join(duplicate_ids))
    coordinate_space = str(payload.get("coordinate_space", "source_coordinates"))
    if coordinate_space not in {"source_coordinates", "analysis_normalized"}:
        raise ValueError(f"Unsupported hierarchy coordinate_space: {coordinate_space}")
    primary_rows = [row for row in rows if str(row.get("root_id")) == PRIMARY_ID]
    for primary_row in primary_rows:
        if primary_row.get("valid", True) is False:
            raise ValueError("The primary root cannot be removed in a hierarchy correction; use a primary override.")
        if "polyline" in primary_row:
            edited_primary = np.asarray(primary_row["polyline"], dtype=float)
            if coordinate_space == "source_coordinates":
                if normalization is None:
                    # A full exported correction may include an unchanged
                    # primary.  Without a transform it cannot be verified.
                    edited_primary = None
                else:
                    edited_primary = normalization.transform_points(edited_primary)
            if edited_primary is not None and (
                edited_primary.shape != np.asarray(primary_path).shape
                or not np.allclose(edited_primary, primary_path, rtol=1e-7, atol=1e-9)
            ):
                raise ValueError("Primary polyline edits are not supported here; use endpoints, soil line, or guide sections.")
    corrections = {str(row.get("root_id")): row for row in rows if row.get("root_id") != PRIMARY_ID}
    current_ids = {path.root_id for path in lateral_paths}
    unknown_ids = sorted(set(corrections) - current_ids)
    if unknown_ids:
        raise ValueError("Hierarchy correction contains unknown or stale root IDs: " + ", ".join(unknown_ids))
    kept: list[RootPath] = []
    for path in lateral_paths:
        correction = corrections.get(path.root_id)
        if correction is None:
            kept.append(path)
            continue
        expected_fingerprint = correction.get("geometry_fingerprint")
        if expected_fingerprint:
            current_polyline = np.asarray(path.points, dtype=float)
            if coordinate_space == "source_coordinates" and normalization is not None:
                current_polyline = normalization.inverse_points(current_polyline)
            if _polyline_fingerprint(current_polyline) != str(expected_fingerprint):
                raise ValueError(f"Hierarchy correction is stale for {path.root_id}: geometry fingerprint changed")
        if correction.get("valid", True) is False:
            continue
        geometry_changed = False
        if "parent_id" in correction:
            corrected_parent = str(correction["parent_id"])
            geometry_changed = corrected_parent != path.parent_id
            path.parent_id = corrected_parent
        if "polyline" in correction:
            edited = np.asarray(correction["polyline"], dtype=float)
            if edited.ndim != 2 or edited.shape[1] != 3 or len(edited) < 2 or not np.all(np.isfinite(edited)):
                raise ValueError(f"Invalid corrected polyline for {path.root_id}")
            if coordinate_space == "source_coordinates":
                if normalization is None:
                    raise ValueError(
                        "A Normalization is required to import source-coordinate hierarchy polylines."
                    )
                edited = normalization.transform_points(edited)
            current_points = np.asarray(path.points, dtype=float)
            geometry_changed = geometry_changed or (
                edited.shape != current_points.shape
                or not np.allclose(
                    edited,
                    current_points,
                    rtol=1e-7,
                    atol=1e-9,
                )
            )
            path.points = edited
        if geometry_changed:
            # Automatic attachment confidence is no longer valid after a
            # human changes the parent or geometry.  Keep this explicit and
            # conservative until the edited result is reviewed.
            path.confidence = 0.0
            for flag in ("manual_correction", "attachment_confidence_invalidated", "low_confidence"):
                if flag not in path.qc_flags:
                    path.qc_flags.append(flag)
            for component in (
                "attachment",
                "junction_tangent_continuity",
                "junction_branch_separation",
            ):
                path.score_components.pop(component, None)
            path.score_components["manual_correction"] = 1.0
        kept.append(path)
    by_id = {path.root_id: path for path in kept}
    for path in kept:
        if path.parent_id != PRIMARY_ID and path.parent_id not in by_id:
            raise ValueError(f"Correction gives {path.root_id} a missing parent: {path.parent_id}")
    graph = _build_graph(kept)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Corrected hierarchy contains a cycle.")
    _assign_recursive_orders(kept)
    # IDs are immutable provenance keys.  Renumbering here would allow a
    # deleted ID to identify different geometry and would break correction
    # audit trails.  root_order is the authoritative post-edit order.
    _refresh_parent_references(np.asarray(primary_path, dtype=float), kept)
    errors = validate_root_tree(kept)
    if errors:
        raise ValueError("Invalid corrected hierarchy: " + "; ".join(errors))
    return kept


def _polyline_fingerprint(points: np.ndarray) -> str:
    rounded = np.round(np.asarray(points, dtype=np.float64), decimals=8)
    return hashlib.sha256(rounded.tobytes(order="C")).hexdigest()[:20]


def _best_attachment(
    path: RootPath,
    parents: list[RootPath],
    *,
    preferred_parent: str,
    tolerance: float,
) -> tuple[RootPath, int, int, float, float]:
    best = None
    for parent in parents:
        parent_tree = cKDTree(parent.points)
        parent_tangents = tangent_vectors(parent.points)
        for endpoint in (0, 1):
            endpoint_index = 0 if endpoint == 0 else -1
            gap, parent_index = parent_tree.query(path.points[endpoint_index], k=1)
            if endpoint == 0:
                child_vector = path.points[min(2, len(path.points) - 1)] - path.points[0]
            else:
                child_vector = path.points[max(0, len(path.points) - 3)] - path.points[-1]
            angle = vector_angle_degrees(child_vector, parent_tangents[int(parent_index)])
            branch_angle = min(float(angle), 180.0 - float(angle))
            # A true lateral should diverge from its parent.  sqrt(sin(theta))
            # remains permissive for acute soybean laterals while strongly
            # rejecting a parallel continuation that happens to be nearby.
            branch_separation = max(float(np.sqrt(max(0.0, np.sin(np.radians(branch_angle))))), 0.05)
            preferred_bonus = 0.45 if parent.root_id == preferred_parent else 0.0
            cost = float(gap / max(tolerance, 1e-12) + 0.25 * (1.0 - branch_separation) - preferred_bonus)
            item = (cost, parent, endpoint, int(parent_index), float(gap), float(branch_separation))
            if best is None or item[0] < best[0]:
                best = item
    assert best is not None
    return best[1], best[2], best[3], best[4], best[5]


def _promote_base_attached_children(
    primary_path: np.ndarray,
    paths: list[RootPath],
    *,
    d_bar: float,
    tolerance: float,
) -> set[str]:
    """Promote ambiguous node-zero children to the oldest plausible ancestor.

    Later tracing passes necessarily propose the lateral traced in the previous
    pass as parent.  At a shared crown junction that proposal is ambiguous: a
    genuine sibling can be just as close to the lateral's first node as to the
    primary.  Promotion is deliberately conservative in two independent ways:
    the insertion must lie inside a short basal arc of the proposed parent, and
    the unsnapped seed must lie inside the proposed ancestor's attachment
    envelope.  A junction that is clearly distal therefore retains its higher
    order even when its child points back toward the primary.
    """

    primary = RootPath(
        root_id=PRIMARY_ID,
        points=np.asarray(primary_path, dtype=float),
        order=0,
        parent_id="",
        confidence=1.0,
    )
    by_id = {path.root_id: path for path in paths}
    by_id[PRIMARY_ID] = primary
    basal_guard = max(14.0 * float(d_bar), 1.5 * float(tolerance))
    ancestor_envelope = max(14.0 * float(d_bar), 1.5 * float(tolerance))
    promoted_ids: set[str] = set()

    for path in paths:
        promotion_count = 0
        first_parent_arc: float | None = None
        final_ancestor_gap: float | None = None
        visited: set[str] = set()
        while path.parent_id != PRIMARY_ID and path.parent_id not in visited:
            visited.add(path.parent_id)
            parent = by_id.get(path.parent_id)
            if parent is None or len(parent.points) < 2:
                break
            parent_segments = np.linalg.norm(np.diff(parent.points, axis=0), axis=1)
            parent_arc = np.concatenate([[0.0], np.cumsum(parent_segments)])
            _, parent_index = cKDTree(parent.points).query(path.points[0], k=1)
            attachment_arc = float(parent_arc[int(parent_index)])
            if first_parent_arc is None:
                first_parent_arc = attachment_arc
            if attachment_arc > basal_guard:
                break

            ancestor = by_id.get(parent.parent_id)
            if ancestor is None or len(ancestor.points) < 2:
                break
            evidence = (
                np.asarray(path.raw_start_point, dtype=float)
                if path.raw_start_point is not None
                else np.asarray(path.points[0], dtype=float)
            )
            ancestor_gap, ancestor_index = cKDTree(ancestor.points).query(evidence, k=1)
            if float(ancestor_gap) > ancestor_envelope:
                break

            path.parent_id = ancestor.root_id
            path.parent_points = ancestor.points
            path.insertion_index = int(ancestor_index)
            path.insertion_point = ancestor.points[int(ancestor_index)].copy()
            path.points[0] = path.insertion_point
            promotion_count += 1
            final_ancestor_gap = float(ancestor_gap)

        if promotion_count:
            promoted_ids.add(path.root_id)
            path.score_components["shared_origin_promoted"] = 1.0
            path.score_components["shared_origin_promotion_count"] = float(promotion_count)
            path.score_components["shared_origin_parent_basal_arc"] = float(first_parent_arc or 0.0)
            path.score_components["shared_origin_ancestor_gap"] = float(final_ancestor_gap or 0.0)
            if "shared_origin_promoted" not in path.qc_flags:
                path.qc_flags.append("shared_origin_promoted")
    return promoted_ids


def _append_qc(path: RootPath, *, gap: float, tolerance: float, d_bar: float) -> None:
    if gap > tolerance and "attachment_gap" not in path.qc_flags:
        path.qc_flags.append("attachment_gap")
    if path.length < max(8.0 * d_bar, 0.008) and "short_trace" not in path.qc_flags:
        path.qc_flags.append("short_trace")
    if path.confidence < 0.55 and "low_confidence" not in path.qc_flags:
        path.qc_flags.append("low_confidence")


def _build_graph(paths: Iterable[RootPath]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(PRIMARY_ID)
    for path in paths:
        graph.add_edge(path.parent_id, path.root_id)
    return graph


def _path_by_id(paths: Iterable[RootPath], root_id: str) -> RootPath:
    for path in paths:
        if path.root_id == root_id:
            return path
    raise KeyError(root_id)


def _assign_recursive_orders(paths: list[RootPath]) -> None:
    children: dict[str, list[RootPath]] = defaultdict(list)
    for path in paths:
        children[path.parent_id].append(path)
    queue: deque[tuple[str, int]] = deque([(PRIMARY_ID, 0)])
    visited = {PRIMARY_ID}
    while queue:
        parent_id, parent_order = queue.popleft()
        for child in children.get(parent_id, []):
            child.order = parent_order + 1
            if child.root_id not in visited:
                visited.add(child.root_id)
                queue.append((child.root_id, child.order))


def _assign_stable_ids(paths: list[RootPath]) -> None:
    children: dict[str, list[RootPath]] = defaultdict(list)
    for path in paths:
        children[path.parent_id].append(path)
    counters: dict[int, int] = defaultdict(int)
    mapping: dict[str, str] = {PRIMARY_ID: PRIMARY_ID}
    queue = deque([PRIMARY_ID])
    while queue:
        parent_id = queue.popleft()
        siblings = sorted(
            children.get(parent_id, []),
            key=lambda root: (
                root.insertion_index if root.insertion_index is not None else 10**12,
                -root.length,
                root.root_id,
            ),
        )
        for root in siblings:
            old_id = root.root_id
            counters[int(root.order)] += 1
            new_id = f"root-o{int(root.order)}-{counters[int(root.order)]:03d}"
            mapping[old_id] = new_id
            root.root_id = new_id
            queue.append(old_id)
    for root in paths:
        root.parent_id = mapping.get(root.parent_id, root.parent_id)


def _refresh_parent_references(primary_path: np.ndarray, paths: list[RootPath]) -> None:
    by_id = {path.root_id: path for path in paths}
    for path in paths:
        parent_points = primary_path if path.parent_id == PRIMARY_ID else by_id[path.parent_id].points
        path.parent_points = parent_points
        _, parent_index = cKDTree(parent_points).query(path.points[0], k=1)
        path.insertion_index = int(parent_index)
        path.insertion_point = parent_points[int(parent_index)].copy()
        path.points[0] = path.insertion_point
