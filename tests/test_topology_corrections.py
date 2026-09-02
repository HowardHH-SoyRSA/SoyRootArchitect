from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from soyrootbio.topology import apply_hierarchy_corrections
from soyrootbio.types import RootPath


def _hierarchy() -> tuple[np.ndarray, list[RootPath]]:
    primary = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.8],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
        ]
    )
    first = RootPath(
        root_id="root-o1-007",
        points=np.array(
            [
                [0.0, 0.0, 0.8],
                [0.2, 0.0, 0.75],
                [0.4, 0.0, 0.7],
            ]
        ),
        order=1,
        parent_id="primary",
        confidence=0.91,
        qc_flags=["automatic_trace"],
        score_components={
            "attachment": 0.9,
            "junction_tangent_continuity": 0.8,
            "junction_branch_separation": 0.7,
            "point_support": 0.85,
        },
    )
    second = RootPath(
        root_id="root-o2-004",
        points=np.array(
            [
                [0.2, 0.0, 0.75],
                [0.2, 0.2, 0.7],
                [0.2, 0.4, 0.65],
            ]
        ),
        order=2,
        parent_id=first.root_id,
        confidence=0.82,
    )
    sibling = RootPath(
        root_id="root-o1-011",
        points=np.array(
            [
                [0.0, 0.0, 0.5],
                [-0.2, 0.1, 0.45],
                [-0.4, 0.2, 0.4],
            ]
        ),
        order=1,
        parent_id="primary",
        confidence=0.78,
    )
    return primary, [first, second, sibling]


def _write_payload(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"schema": "soyrootbio.root-hierarchy/v2", "roots": []}, "Unsupported hierarchy correction schema"),
        ({"roots": []}, "Unsupported hierarchy correction schema"),
        (
            {"schema": "soyrootbio.root-hierarchy/v1", "roots": {}},
            "roots must be a list of objects",
        ),
        (
            {"schema": "soyrootbio.root-hierarchy/v1", "roots": [{"valid": False}]},
            "must have a root_id",
        ),
        (
            {
                "schema": "soyrootbio.root-hierarchy/v1",
                "roots": [{"root_id": "root-o1-007"}, {"root_id": "root-o1-007"}],
            },
            "duplicate root IDs",
        ),
        (
            {
                "schema": "soyrootbio.root-hierarchy/v1",
                "roots": [{"root_id": "root-o9-deleted", "valid": False}],
            },
            "unknown or stale root IDs",
        ),
        (
            {
                "schema": "soyrootbio.root-hierarchy/v1",
                "roots": [{"root_id": "root-o1-007", "geometry_fingerprint": "stale"}],
            },
            "geometry fingerprint changed",
        ),
    ],
)
def test_correction_import_rejects_invalid_schema_ids_and_stale_geometry(
    tmp_path: Path,
    payload: dict,
    message: str,
) -> None:
    primary, roots = _hierarchy()
    correction = _write_payload(tmp_path / "invalid.json", payload)

    with pytest.raises(ValueError, match=message):
        apply_hierarchy_corrections(primary, copy.deepcopy(roots), correction)


def test_polyline_vertex_count_edit_is_accepted_and_invalidates_automatic_confidence(
    tmp_path: Path,
) -> None:
    primary, roots = _hierarchy()
    edited_points = np.array(
        [
            [0.0, 0.0, 0.8],
            [0.1, 0.03, 0.78],
            [0.25, 0.04, 0.74],
            [0.45, 0.05, 0.68],
        ]
    )
    correction = _write_payload(
        tmp_path / "shape-change.json",
        {
            "schema": "soyrootbio.root-hierarchy/v1",
            "coordinate_space": "analysis_normalized",
            "roots": [{"root_id": "root-o1-007", "polyline": edited_points.tolist()}],
        },
    )

    corrected = apply_hierarchy_corrections(primary, copy.deepcopy(roots), correction)
    edited = next(root for root in corrected if root.root_id == "root-o1-007")

    assert edited.points.shape == (4, 3)
    np.testing.assert_allclose(edited.points, edited_points)
    assert edited.confidence == 0.0
    assert {
        "automatic_trace",
        "manual_correction",
        "attachment_confidence_invalidated",
        "low_confidence",
    }.issubset(edited.qc_flags)
    assert edited.score_components["manual_correction"] == 1.0
    assert "attachment" not in edited.score_components
    assert "junction_tangent_continuity" not in edited.score_components
    assert "junction_branch_separation" not in edited.score_components


def test_correction_rejects_child_longer_than_its_parent(tmp_path: Path) -> None:
    primary, roots = _hierarchy()
    correction = _write_payload(
        tmp_path / "overlong-child.json",
        {
            "schema": "soyrootbio.root-hierarchy/v1",
            "coordinate_space": "analysis_normalized",
            "roots": [
                {
                    "root_id": "root-o2-004",
                    "polyline": [
                        [0.2, 0.0, 0.75],
                        [0.2, 0.8, 0.75],
                    ],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="exceeds parent"):
        apply_hierarchy_corrections(
            primary,
            copy.deepcopy(roots),
            correction,
        )


def test_deleting_a_leaf_preserves_every_surviving_provenance_id(tmp_path: Path) -> None:
    primary, roots = _hierarchy()
    removed_id = "root-o1-011"
    correction = _write_payload(
        tmp_path / "delete.json",
        {
            "schema": "soyrootbio.root-hierarchy/v1",
            "roots": [{"root_id": removed_id, "valid": False}],
        },
    )

    corrected = apply_hierarchy_corrections(primary, copy.deepcopy(roots), correction)

    assert {root.root_id for root in corrected} == {root.root_id for root in roots} - {removed_id}
    child = next(root for root in corrected if root.root_id == "root-o2-004")
    assert child.parent_id == "root-o1-007"
    assert child.order == 2


def test_correction_import_rejects_nonfinite_json_constants(tmp_path: Path) -> None:
    primary, roots = _hierarchy()
    correction = tmp_path / "nonfinite.json"
    correction.write_text(
        '{"schema":"soyrootbio.root-hierarchy/v1","roots":['
        '{"root_id":"root-o1-007","polyline":[[NaN,0,0],[1,0,0]]}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite JSON constant: NaN"):
        apply_hierarchy_corrections(primary, copy.deepcopy(roots), correction)
