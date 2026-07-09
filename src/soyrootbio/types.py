from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Normalization:
    minimum: np.ndarray
    scale: float
    unit_scale: float = 1.0

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        return (points - self.minimum) / self.scale

    def inverse_points(self, points: np.ndarray) -> np.ndarray:
        return points * self.scale + self.minimum

    def transform_vector(self, vector: np.ndarray) -> np.ndarray:
        return vector / self.scale

    def inverse_length(self, length: float) -> float:
        return float(length * self.scale * self.unit_scale)


@dataclass
class PointCloudData:
    points: np.ndarray
    source_path: Path
    colors: np.ndarray | None = None


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

    @property
    def length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())



