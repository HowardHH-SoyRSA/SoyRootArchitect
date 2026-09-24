from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Normalization:
    minimum: np.ndarray
    scale: float

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        return (points - self.minimum) / self.scale

    def inverse_points(self, points: np.ndarray) -> np.ndarray:
        return points * self.scale + self.minimum

    def transform_vector(self, vector: np.ndarray) -> np.ndarray:
        return vector / self.scale

    def inverse_length(self, length: float) -> float:
        """Restore a normalized length to the source mesh coordinate unit."""

        return float(length * self.scale)


@dataclass
class PointCloudData:
    """Geometry used for analysis plus an optional full-resolution mesh.

    ``points`` is the analysis cloud.  When adaptive point reduction is used,
    ``full_points`` and ``triangles`` preserve the source mesh for labelled
    exports and final surface measurements.  ``analysis_indices`` maps analysis
    points back to vertices in ``full_points`` when they are a vertex subset.
    """

    points: np.ndarray
    source_path: Path
    colors: np.ndarray | None = None
    full_points: np.ndarray | None = None
    triangles: np.ndarray | None = None
    analysis_indices: np.ndarray | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def export_points(self) -> np.ndarray:
        return self.full_points if self.full_points is not None else self.points

    @property
    def was_reduced(self) -> bool:
        return self.full_points is not None and len(self.points) < len(self.full_points)


@dataclass
class RootPath:
    root_id: str
    points: np.ndarray
    node_indices: np.ndarray | None = None
    covered_indices: set[int] = field(default_factory=set)
    score: float = 0.0
    start_index: int | None = None
    order: int = 1
    parent_id: str = "primary"
    parent_points: np.ndarray | None = None
    insertion_point: np.ndarray | None = None
    insertion_index: int | None = None
    confidence: float = 0.0
    qc_flags: list[str] = field(default_factory=list)
    mean_radius: float | None = None
    score_components: dict[str, float] = field(default_factory=dict)
    # Unsnapped surface seed used to infer biological attachment.  ``points[0]``
    # is deliberately snapped to a parent centreline during tracing, so it
    # cannot distinguish siblings that share a crown junction from a true
    # child attached farther along a lateral parent.
    raw_start_point: np.ndarray | None = None
    # Density evidence is narrower than geometric coverage: occupied points and
    # points inside the parent tube can be traversed at a junction, but must not
    # improve candidate ranking or overlap selection.
    novel_support_indices: set[int] | None = None
    # Final-support fitting keeps the internal attachment separate from the
    # exposed body. Zero means there is no supported internal connector.
    body_start_index: int = 0
    centerline_assessment: dict[str, Any] = field(default_factory=dict)
    # Alternate continuations emitted at one supported fork share this key and
    # the same basal prefix.  They remain separate paths so later consensus and
    # topology passes can preserve both arm geometries while deciding which
    # arm owns the continuation identity.
    fork_hypothesis_group: str | None = None
    fork_hypothesis_index: int = 0
    fork_common_prefix_nodes: int = 0
    deferred_fork_hypotheses: list[dict[str, Any]] = field(
        default_factory=list
    )

    @property
    def length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())


@dataclass
class PrimaryCandidate:
    """A ranked collar-to-taproot candidate returned by automatic detection."""

    rank: int
    start_index: int
    end_index: int
    start: np.ndarray
    end: np.ndarray
    path: np.ndarray
    score: float
    confidence: float
    components: dict[str, float] = field(default_factory=dict)
    qc_flags: list[str] = field(default_factory=list)


@dataclass
class TopologyReport:
    """Machine-readable evidence produced while repairing the root tree."""

    roots_reoriented: int = 0
    parents_reassigned: int = 0
    attachment_constraint_decisions: list[dict[str, Any]] = field(default_factory=list)
    cycles_removed: int = 0
    fork_arms_reconciled: int = 0
    fork_arm_details: list[dict[str, Any]] = field(default_factory=list)
    fork_resurvey_iterations: int = 0
    fork_resurvey_cycle_states: int = 0
    fork_resurvey_decisions: list[dict[str, Any]] = field(default_factory=list)
    unresolved_long_arm_details: list[dict[str, Any]] = field(default_factory=list)
    overlong_children_removed: int = 0
    overlong_descendants_removed: int = 0
    overlong_child_details: list[dict[str, Any]] = field(default_factory=list)
    origins_above_primary_top_removed: int = 0
    descendants_of_above_top_roots_removed: int = 0
    above_primary_top_details: list[dict[str, Any]] = field(default_factory=list)
    primary_top_point_normalized: list[float] = field(default_factory=list)
    primary_top_reference_policy: str = ""
    gravity_direction: list[float] = field(default_factory=list)
    disconnected_roots: int = 0
    low_confidence_roots: int = 0
    warnings: list[str] = field(default_factory=list)



