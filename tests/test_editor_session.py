from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from soyrootbio.editor.ply import read_labeled_ply
from soyrootbio.editor.server import create_editor_app
from soyrootbio.editor.session import EditorSession, EditorValidationError
from soyrootbio.hardware import GPUInfo, HardwareInfo
from soyrootbio.io import write_labeled_ply


SOURCE_FILES = (
    "segmented_root_structure.ply",
    "root_hierarchy.json",
    "root_traits.csv",
    "csv/root_label_map.csv",
)


@pytest.fixture(autouse=True)
def fixed_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep editor tests deterministic and independent of workstation hardware."""

    hardware = HardwareInfo(
        logical_cpus=8,
        physical_cpus=4,
        total_memory_bytes=16 * 1024**3,
        available_memory_bytes=12 * 1024**3,
        gpus=(
            GPUInfo(
                index=0,
                name="Test discrete GPU",
                memory_total_bytes=8 * 1024**3,
                driver_version="test",
            ),
        ),
    )
    monkeypatch.setattr(
        "soyrootbio.editor.session.detect_hardware",
        lambda: hardware,
    )


@pytest.fixture
def editor_bundle(tmp_path: Path) -> Path:
    output_dir = tmp_path / "automatic-output"
    (output_dir / "csv").mkdir(parents=True)

    root_points = {
        "primary": np.array(
            [
                [0.0, 0.0, 10.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        "root-a": np.array(
            [
                [0.0, 0.0, 3.0],
                [0.5, 0.0, 3.0],
                [1.0, 0.0, 3.0],
                [1.5, 0.0, 2.8],
            ]
        ),
        "root-b": np.array(
            [
                [0.0, 0.0, 1.0],
                [-0.5, 0.0, 1.0],
                [-1.0, 0.0, 0.8],
                [-4.0, 0.0, 0.5],
            ]
        ),
        "root-c": np.array(
            [
                [0.5, 0.0, 3.0],
                [0.5, 0.25, 3.0],
                [0.5, 0.5, 2.9],
                [0.5, 0.75, 2.8],
            ]
        ),
    }
    vertices = np.vstack(
        [
            root_points["primary"],
            root_points["root-a"],
            root_points["root-b"],
            root_points["root-c"],
            np.array([[2.0, 2.0, 2.0], [-2.0, -2.0, -2.0]]),
        ]
    )
    root_labels = np.array(
        [0] * 5 + [1] * 4 + [2] * 4 + [3] * 4 + [-1, -2],
        dtype=np.int32,
    )
    root_orders = np.array(
        [0] * 5 + [1] * 4 + [1] * 4 + [2] * 4 + [255, 254],
        dtype=np.uint8,
    )
    assignment_states = np.array(
        [1] * 17 + [0, 2],
        dtype=np.uint8,
    )
    colors = np.arange(len(vertices) * 3, dtype=np.uint8).reshape(-1, 3)
    triangles = np.array(
        [
            [0, 1, 2],
            [2, 3, 4],
            [5, 6, 7],
            [6, 7, 8],
            [9, 10, 11],
            [10, 11, 12],
            [13, 14, 15],
            [14, 15, 16],
        ],
        dtype=np.int32,
    )
    write_labeled_ply(
        output_dir / "segmented_root_structure.ply",
        vertices,
        triangles=triangles,
        colors=colors,
        root_ids=root_labels,
        root_orders=root_orders,
        assignment_states=assignment_states,
    )

    hierarchy_rows = [
        {
            "root_id": "primary",
            "parent_id": None,
            "root_order": 0,
            "polyline": root_points["primary"].tolist(),
            "insertion_point": None,
            "insertion_index": None,
            "confidence": 0.99,
            "qc_flags": [],
        },
        {
            "root_id": "root-a",
            "parent_id": "primary",
            "root_order": 1,
            "polyline": root_points["root-a"].tolist(),
            "insertion_point": root_points["root-a"][0].tolist(),
            "insertion_index": 1,
            "confidence": 0.91,
            "qc_flags": [],
        },
        {
            "root_id": "root-b",
            "parent_id": "primary",
            "root_order": 1,
            "polyline": root_points["root-b"].tolist(),
            "insertion_point": root_points["root-b"][0].tolist(),
            "insertion_index": 3,
            "confidence": 0.88,
            "qc_flags": [],
        },
        {
            "root_id": "root-c",
            "parent_id": "root-a",
            "root_order": 2,
            "polyline": root_points["root-c"].tolist(),
            "insertion_point": root_points["root-c"][0].tolist(),
            "insertion_index": 1,
            "confidence": 0.82,
            "qc_flags": [],
        },
    ]
    (output_dir / "root_hierarchy.json").write_text(
        json.dumps(
            {
                "schema": "soyrootbio.root-hierarchy/v1",
                "coordinate_space": "source_coordinates",
                "roots": hierarchy_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metric_values = {
        "primary": (10.0, 10.0, None, 0.20, 3.0, 0.12, 1.0),
        "root-a": (1.54, 1.51, 74.0, 0.10, 1.1, 0.03, 1.02),
        "root-b": (4.05, 4.03, 62.0, 0.09, 1.0, 0.025, 1.01),
        "root-c": (0.79, 0.78, 68.0, 0.07, 0.8, 0.015, 1.01),
    }
    parents = {
        "primary": "",
        "root-a": "primary",
        "root-b": "primary",
        "root-c": "root-a",
    }
    orders = {"primary": 0, "root-a": 1, "root-b": 1, "root-c": 2}
    trait_rows = []
    for root_id, values in metric_values.items():
        length, chord, angle, diameter, area, volume, tortuosity = values
        trait_rows.append(
            {
                "root_id": root_id,
                "parent_id": parents[root_id],
                "root_order": orders[root_id],
                "length": length,
                "chord_length": chord,
                "tip_gravity_angle_deg": angle,
                "tip_start_gravity_angle_deg": angle,
                "tip_primary_angle_deg": angle,
                "mean_diameter": diameter,
                "surface_area": area,
                "volume": volume,
                "tortuosity": tortuosity,
                "point_count": int(np.count_nonzero(root_labels == orders[root_id]))
                if root_id == "primary"
                else 4,
                "confidence": next(
                    row["confidence"]
                    for row in hierarchy_rows
                    if row["root_id"] == root_id
                ),
                "qc_flags": "",
                "length_unit": "mesh_unit",
                "area_unit": "mesh_unit^2",
                "volume_unit": "mesh_unit^3",
            }
        )
    pd.DataFrame(trait_rows).to_csv(output_dir / "root_traits.csv", index=False)

    pd.DataFrame(
        [
            {
                "numeric_label": -2,
                "root_id": "uncertain",
                "parent_id": "",
                "root_order": "",
            },
            {
                "numeric_label": -1,
                "root_id": "unassigned",
                "parent_id": "",
                "root_order": "",
            },
            {
                "numeric_label": 0,
                "root_id": "primary",
                "parent_id": "",
                "root_order": 0,
            },
            {
                "numeric_label": 1,
                "root_id": "root-a",
                "parent_id": "primary",
                "root_order": 1,
            },
            {
                "numeric_label": 2,
                "root_id": "root-b",
                "parent_id": "primary",
                "root_order": 1,
            },
            {
                "numeric_label": 3,
                "root_id": "root-c",
                "parent_id": "root-a",
                "root_order": 2,
            },
        ]
    ).to_csv(output_dir / "csv" / "root_label_map.csv", index=False)
    return output_dir


def _file_digests(output_dir: Path) -> dict[str, str]:
    return {
        relative: sha256((output_dir / relative).read_bytes()).hexdigest()
        for relative in SOURCE_FILES
    }


def _expected_baseline_fingerprint(output_dir: Path) -> str:
    digest = sha256()
    for relative in SOURCE_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update((output_dir / relative).read_bytes())
    return f"sha256:{digest.hexdigest()}"


def _materialised_snapshot(session: EditorSession) -> tuple[str, bytes, bytes, int]:
    """Capture all editable materialised data, excluding history counters."""

    roots = json.dumps(
        session.public_state()["roots"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (
        roots,
        session.mesh.root_labels.tobytes(),
        session.mesh.assignment_states.tobytes(),
        session._next_numeric_label,
    )


def _new_session(bundle: Path, name: str = "session") -> EditorSession:
    return EditorSession(
        bundle,
        session_dir=bundle.parent / name,
        load_existing_log=False,
    )


def test_binary_ply_reader_preserves_geometry_faces_and_editor_scalars(
    editor_bundle: Path,
) -> None:
    mesh = read_labeled_ply(editor_bundle / "segmented_root_structure.ply")

    assert mesh.vertex_count == 19
    assert mesh.face_count == 8
    assert mesh.positions.dtype == np.float64
    assert mesh.triangles.dtype == np.int32
    assert mesh.colors.dtype == np.uint8
    np.testing.assert_array_equal(mesh.triangles[0], [0, 1, 2])
    np.testing.assert_allclose(mesh.positions[8], [1.5, 0.0, 2.8])
    np.testing.assert_array_equal(mesh.colors[4], [12, 13, 14])
    np.testing.assert_array_equal(
        mesh.root_labels,
        [0] * 5 + [1] * 4 + [2] * 4 + [3] * 4 + [-1, -2],
    )
    np.testing.assert_array_equal(mesh.root_orders[-2:], [255, 254])
    np.testing.assert_array_equal(mesh.assignment_states[-2:], [0, 2])


def test_editor_ply_reader_keeps_double_precision_coordinates(
    tmp_path: Path,
) -> None:
    vertices = np.array(
        [
            [1_000_000_000.1234567, -2_000_000_000.7654321, 0.000000123],
            [1_000_000_000.1235567, -2_000_000_000.7653322, 0.000000987],
        ],
        dtype=np.float64,
    )
    path = tmp_path / "precise.ply"
    write_labeled_ply(path, vertices, root_ids=np.array([0, 0]))

    loaded = read_labeled_ply(path)
    assert loaded.positions.dtype == np.float64
    np.testing.assert_array_equal(loaded.positions, vertices)


def test_public_state_exposes_full_metrics_relationships_and_gpu_policy(
    editor_bundle: Path,
) -> None:
    session = _new_session(editor_bundle)
    state = session.public_state()
    roots = {root["root_id"]: root for root in state["roots"]}

    assert state["root_count"] == 4
    assert state["point_patch_count"] == 2
    assert [
        (patch["patch_id"], patch["kind"], patch["point_count"])
        for patch in state["point_patches"]
    ] == [
        ("uncertain-18", "uncertain", 1),
        ("unassigned-17", "unassigned", 1),
    ]
    assert state["mesh"]["vertex_count"] == 19
    assert state["mesh"]["face_count"] == 8
    assert state["supported_operations"] == [
        "create_root",
        "split_root",
        "merge_roots",
        "assign_points",
        "reconnect_root",
        "reparent_root",
        "delete_root",
        "redraw_root",
        "correct_root_order",
    ]
    assert state["hardware"]["discrete_gpu_present"] is True
    assert state["hardware"]["gpus"][0]["name"] == "Test discrete GPU"
    assert (
        state["hardware"]["acceleration"]["full_resolution_policy"]
        == "retain unless browser/device allocation fails"
    )
    assert roots["primary"]["children_ids"] == ["root-a", "root-b"]
    assert roots["root-a"]["children_ids"] == ["root-c"]
    assert roots["root-c"]["parent_id"] == "root-a"
    assert roots["root-a"]["length"] == pytest.approx(1.54)
    assert roots["root-a"]["chord_length"] == pytest.approx(1.51)
    assert roots["root-a"]["tip_gravity_angle_deg"] == pytest.approx(74.0)
    assert roots["root-a"]["mean_diameter"] == pytest.approx(0.10)
    assert roots["root-a"]["surface_area"] == pytest.approx(1.1)
    assert roots["root-a"]["volume"] == pytest.approx(0.03)
    assert roots["root-a"]["tortuosity"] == pytest.approx(1.02)
    assert roots["root-a"]["insertion_point"] == [0.0, 0.0, 3.0]
    assert roots["root-a"]["tip_point"] == [1.5, 0.0, 2.8]


def test_point_patches_follow_same_label_triangle_connectivity_and_history(
    editor_bundle: Path,
) -> None:
    session = _new_session(editor_bundle)

    result = session.apply_operation(
        "delete_root",
        {"root_id": "root-a"},
        operation_id="patch-delete-root-a",
    )
    patches = {
        patch["patch_id"]: patch for patch in result["state"]["point_patches"]
    }

    connected = patches["unassigned-5"]
    assert connected["point_count"] == 4
    assert connected["anchor_vertex_index"] == 5
    assert connected["revision"] == session.label_revision
    assert connected["indices_url"].endswith(
        f"?revision={session.label_revision}"
    )
    np.testing.assert_allclose(connected["centroid"], [0.75, 0.0, 2.95])
    assert connected["bounds"] == {
        "minimum": [0.0, 0.0, 2.8],
        "maximum": [1.5, 0.0, 3.0],
    }
    payload, count, revision = session.point_patch_indices_snapshot(
        "unassigned-5",
        expected_revision=session.label_revision,
    )
    assert count == 4
    assert revision == session.label_revision
    np.testing.assert_array_equal(
        np.frombuffer(payload, dtype="<u4"),
        [5, 6, 7, 8],
    )
    assert patches["unassigned-17"]["point_count"] == 1
    assert patches["uncertain-18"]["point_count"] == 1

    undone = session.undo()
    assert {
        patch["patch_id"] for patch in undone["state"]["point_patches"]
    } == {"unassigned-17", "uncertain-18"}
    redone = session.redo()
    assert "unassigned-5" in {
        patch["patch_id"] for patch in redone["state"]["point_patches"]
    }


def test_local_editor_api_serves_full_mesh_labels_and_history(
    editor_bundle: Path,
) -> None:
    app = create_editor_app(
        editor_bundle,
        session_dir=editor_bundle.parent / "api-session",
    )
    client = app.test_client()

    assert client.get("/", headers={"Host": "rebound.example"}).status_code == 403
    assert client.get("/api/state").status_code == 403
    bootstrap = client.get("/")
    assert bootstrap.status_code == 200
    assert "soyrootbio_editor_token=" in bootstrap.headers["Set-Cookie"]

    state_response = client.get("/api/state")
    assert state_response.status_code == 200
    assert state_response.json["mesh"]["vertex_count"] == 19

    mesh_response = client.get(
        "/api/mesh",
        headers={"Range": "bytes=0-31"},
    )
    assert mesh_response.status_code == 206
    assert len(mesh_response.data) == 32

    labels_response = client.get("/api/mesh-labels")
    assert labels_response.status_code == 200
    assert labels_response.headers["X-Vertex-Count"] == "19"
    np.testing.assert_array_equal(
        np.frombuffer(labels_response.data, dtype="<i4"),
        [0] * 5 + [1] * 4 + [2] * 4 + [3] * 4 + [-1, -2],
    )

    unassigned_patch = next(
        patch
        for patch in state_response.json["point_patches"]
        if patch["kind"] == "unassigned"
    )
    patch_response = client.get(unassigned_patch["indices_url"])
    assert patch_response.status_code == 200
    assert patch_response.headers["X-Point-Count"] == "1"
    assert patch_response.headers["X-Label-Revision"] == "0"
    np.testing.assert_array_equal(
        np.frombuffer(patch_response.data, dtype="<u4"),
        [17],
    )
    assert (
        client.get(
            f"/api/point-patches/{unassigned_patch['patch_id']}/indices"
        ).status_code
        == 400
    )
    stale_response = client.get(
        f"/api/point-patches/{unassigned_patch['patch_id']}/indices?revision=99"
    )
    assert stale_response.status_code == 409
    assert stale_response.json["kind"] == "revision_conflict"

    edit_response = client.post(
        "/api/operations",
        json={
            "type": "assign_points",
            "arguments": {"root_id": "root-a", "indices": [17]},
        },
    )
    assert edit_response.status_code == 200
    assert edit_response.json["state"]["can_undo"] is True
    edited_labels = np.frombuffer(
        client.get("/api/mesh-labels").data,
        dtype="<i4",
    )
    assert edited_labels[17] == 1

    assert client.post("/api/undo").status_code == 200
    assert client.post("/api/redo").status_code == 200
    invalid = client.post(
        "/api/operations",
        json={"type": "delete_root", "arguments": {"root_id": "primary"}},
    )
    assert invalid.status_code == 400
    assert invalid.json["kind"] == "validation"
    cross_origin = client.post(
        "/api/undo",
        headers={"Origin": "https://malicious.example"},
    )
    assert cross_origin.status_code == 403
    outside_export = client.post(
        "/api/export",
        json={"target_dir": str(editor_bundle.parent / "outside-session")},
    )
    assert outside_export.status_code == 400


def test_assignment_drag_is_one_resolved_undoable_operation(
    editor_bundle: Path,
) -> None:
    session = _new_session(editor_bundle)
    labels_before = session.mesh.root_labels.copy()

    result = session.apply_operation(
        "assign_points",
        {
            "root_id": "root-b",
            "positions": session.mesh.positions[5:9].tolist(),
            "radius": 0.01,
        },
        operation_id="continuous-brush-stroke",
    )

    arguments = result["operation"]["arguments"]
    assert "positions" not in arguments
    assert arguments["stroke_point_count"] == 4
    assert arguments["resolved_point_count"] == 6
    assert arguments["indices_blob"].endswith(".npy")
    np.testing.assert_array_equal(
        np.load(
            session.session_dir / arguments["indices_blob"],
            allow_pickle=False,
        ),
        [1, 5, 6, 7, 8, 13],
    )
    np.testing.assert_array_equal(session.mesh.root_labels[5:9], [2, 2, 2, 2])
    assert result["state"]["operation_count"] == 1

    session.undo()
    np.testing.assert_array_equal(session.mesh.root_labels, labels_before)


@pytest.mark.parametrize(
    ("operation_type", "arguments"),
    [
        (
            "create_root",
            {
                "parent_id": "primary",
                "points": [
                    [0.2, 0.2, 3.0],
                    [1.0, 1.0, 2.5],
                    [2.0, 2.0, 2.0],
                ],
                "indices": [17],
                "new_root_id": "root-manual-path",
            },
        ),
        (
            "split_root",
            {
                "root_id": "root-a",
                "node_index": 2,
                "new_root_id": "root-a-distal",
            },
        ),
        (
            "merge_roots",
            {"root_id": "root-a", "other_root_id": "root-b"},
        ),
        (
            "assign_points",
            {"root_id": "root-b", "indices": [17, 18]},
        ),
        (
            "reconnect_root",
            {
                "root_id": "root-c",
                "target_root_id": "root-b",
                "position": [-0.5, 0.0, 1.0],
            },
        ),
        (
            "reparent_root",
            {
                "root_id": "root-c",
                "new_parent_id": "root-b",
                "position": [-1.0, 0.0, 0.8],
                "snap_to_parent": False,
            },
        ),
        (
            "delete_root",
            {"root_id": "root-a"},
        ),
        (
            "redraw_root",
            {
                "root_id": "root-b",
                "points": [
                    [0.0, 0.0, 1.0],
                    [-0.4, 0.2, 0.9],
                    [-0.9, 0.4, 0.6],
                ],
            },
        ),
        (
            "correct_root_order",
            {"root_id": "root-c", "root_order": 3},
        ),
    ],
    ids=[
        "create-root",
        "split",
        "merge",
        "assign-points",
        "reconnect",
        "reparent",
        "delete",
        "redraw",
        "root-order-correction",
    ],
)
def test_all_editor_operations_have_exact_undo_and_redo(
    editor_bundle: Path,
    operation_type: str,
    arguments: dict,
) -> None:
    session = _new_session(editor_bundle)
    before = _materialised_snapshot(session)

    result = session.apply_operation(
        operation_type,
        arguments,
        operation_id=f"test-{operation_type}",
    )
    after = _materialised_snapshot(session)

    assert after != before
    assert result["operation"]["type"] == operation_type
    assert result["state"]["can_undo"] is True
    assert result["state"]["can_redo"] is False
    assert result["state"]["operation_count"] == 1

    if operation_type == "create_root":
        root = session.roots["root-manual-path"]
        assert root.numeric_label == 4
        assert root.parent_id == "primary"
        assert root.order == 1
        assert root.insertion_index == 1
        np.testing.assert_allclose(root.insertion_point, [0.0, 0.0, 3.0])
        np.testing.assert_allclose(root.points[0], root.insertion_point)
        assert "manual_created_from_unassigned" in root.qc_flags
        np.testing.assert_array_equal(session.mesh.root_labels[[17, 18]], [4, -2])
        np.testing.assert_array_equal(
            session.mesh.assignment_states[[17, 18]],
            [1, 2],
        )
        assert "indices" not in result["operation"]["arguments"]
        assert result["operation"]["arguments"]["resolved_point_count"] == 1
        assert result["operation"]["arguments"]["indices_blob"].endswith(".npy")
    elif operation_type == "split_root":
        assert "root-a-distal" in session.roots
        assert session.roots["root-a-distal"].numeric_label == 4
        assert session.roots["root-a-distal"].parent_id == "root-a"
        assert session.roots["root-a-distal"].order == 2
        assert len(session.roots["root-a"].points) == 3
        assert len(session.roots["root-a-distal"].points) == 2
    elif operation_type == "merge_roots":
        assert "root-b" not in session.roots
        assert not np.any(session.mesh.root_labels == 2)
    elif operation_type == "assign_points":
        np.testing.assert_array_equal(session.mesh.root_labels[[17, 18]], [2, 2])
        np.testing.assert_array_equal(
            session.mesh.assignment_states[[17, 18]],
            [1, 1],
        )
        assert "indices" not in result["operation"]["arguments"]
        assert result["operation"]["arguments"]["resolved_point_count"] == 2
        assert result["operation"]["arguments"]["indices_blob"].endswith(".npy")
    elif operation_type == "reconnect_root":
        root = session.roots["root-c"]
        assert root.parent_id == "root-b"
        np.testing.assert_allclose(root.points[0], [-0.5, 0.0, 1.0])
        assert "manual_reconnect" in root.qc_flags
    elif operation_type == "reparent_root":
        root = session.roots["root-c"]
        assert root.parent_id == "root-b"
        np.testing.assert_allclose(root.insertion_point, [-1.0, 0.0, 0.8])
        np.testing.assert_allclose(root.points[0], [-1.0, 0.0, 0.8])
        assert root.insertion_index == 2
    elif operation_type == "delete_root":
        assert "root-a" not in session.roots
        assert session.roots["root-c"].parent_id == "primary"
        assert not np.any(session.mesh.root_labels == 1)
    elif operation_type == "redraw_root":
        assert len(session.roots["root-b"].points) == 3
        assert "manual_redraw" in session.roots["root-b"].qc_flags
    elif operation_type == "correct_root_order":
        assert session.roots["root-c"].order == 3
        assert session.roots["root-c"].order_overridden is True

    for root in session.roots.values():
        if root.parent_id is None:
            continue
        parent = session.roots[root.parent_id]
        assert root.insertion_index is not None
        np.testing.assert_allclose(
            root.insertion_point,
            parent.points[root.insertion_index],
        )
        np.testing.assert_allclose(root.points[0], root.insertion_point)

    undone = session.undo()
    assert undone["state"]["can_redo"] is True
    assert _materialised_snapshot(session) == before

    redone = session.redo()
    assert redone["state"]["can_undo"] is True
    assert _materialised_snapshot(session) == after


def test_create_root_claims_unassigned_points_near_the_drawn_path(
    editor_bundle: Path,
) -> None:
    session = _new_session(editor_bundle, "create-from-path")
    assigned_before = session.mesh.root_labels[:17].copy()

    result = session.apply_operation(
        "create_root",
        {
            "parent_id": "root-a",
            "points": [
                [1.5, 0.0, 2.8],
                [1.8, 0.2, 2.6],
            ],
            # Deliberately broad enough to cover assigned and uncertain points.
            # Creation must still claim only label -1.
            "claim_radius": 10.0,
            "new_root_id": "root-from-grey-points",
        },
        operation_id="create-from-path",
    )

    created = session.roots["root-from-grey-points"]
    assert created.parent_id == "root-a"
    assert created.order == 2
    assert result["operation"]["arguments"]["resolved_point_count"] == 1
    np.testing.assert_array_equal(session.mesh.root_labels[:17], assigned_before)
    np.testing.assert_array_equal(session.mesh.root_labels[[17, 18]], [4, -2])
    np.testing.assert_array_equal(
        session.mesh.assignment_states[[17, 18]],
        [1, 2],
    )


def test_created_root_replays_with_stable_id_attachment_and_labels(
    editor_bundle: Path,
) -> None:
    session_dir = editor_bundle.parent / "create-replay-session"
    session = EditorSession(
        editor_bundle,
        session_dir=session_dir,
        load_existing_log=True,
    )
    result = session.apply_operation(
        "create_root",
        {
            "parent_id": "primary",
            "points": [[0.2, 0.2, 3.0], [2.0, 2.0, 2.0]],
            "indices": [17],
        },
        operation_id="create-replay",
        timestamp="2026-07-26T00:00:00+00:00",
    )
    expected = _materialised_snapshot(session)
    arguments = result["operation"]["arguments"]
    created_id = arguments["new_root_id"]

    assert "indices" not in arguments
    assert arguments["resolved_attachment"]["parent_id"] == "primary"
    assert arguments["indices_sha256"].startswith("sha256:")

    replayed = EditorSession(
        editor_bundle,
        session_dir=session_dir,
        load_existing_log=True,
    )
    assert _materialised_snapshot(replayed) == expected
    assert created_id in replayed.roots
    assert (
        replayed.mesh.root_labels[17]
        == replayed.roots[created_id].numeric_label
    )


def test_created_root_round_trips_through_all_edited_exports(
    editor_bundle: Path,
) -> None:
    source_before = _file_digests(editor_bundle)
    session = _new_session(editor_bundle, "create-export-session")
    session.apply_operation(
        "create_root",
        {
            "parent_id": "primary",
            "points": [[0.2, 0.2, 3.0], [2.0, 2.0, 2.0]],
            "indices": [17],
            "new_root_id": "root-manual-path",
        },
        operation_id="create-export",
    )

    export_dir = session.export_materialised(
        session.session_dir / "materialised-create"
    )
    exported_mesh = read_labeled_ply(
        export_dir / "edited_segmented_root_structure.ply"
    )
    np.testing.assert_array_equal(exported_mesh.root_labels[[17, 18]], [4, -2])
    np.testing.assert_array_equal(exported_mesh.root_orders[[17, 18]], [1, 254])
    np.testing.assert_array_equal(
        exported_mesh.assignment_states[[17, 18]],
        [1, 2],
    )
    np.testing.assert_array_equal(exported_mesh.colors[17], [255, 0, 255])

    label_map = pd.read_csv(export_dir / "edited_root_label_map.csv")
    created_row = label_map.set_index("root_id").loc["root-manual-path"]
    assert int(created_row["numeric_label"]) == 4
    assert created_row["parent_id"] == "primary"
    assert int(created_row["root_order"]) == 1
    assert created_row["color_rgb"] == "255,0,255"

    traits = pd.read_csv(export_dir / "edited_root_traits.csv")
    assert "root-manual-path" in set(traits["root_id"])
    rsml = ET.parse(export_dir / "edited_root_system.rsml")
    assert (
        rsml.find(
            "./scene/plant/root[@id='primary']/root[@id='root-manual-path']"
        )
        is not None
    )
    assert _file_digests(editor_bundle) == source_before


def test_operation_log_replay_restores_active_state_and_exact_labels(
    editor_bundle: Path,
) -> None:
    session_dir = editor_bundle.parent / "persistent-session"
    session = EditorSession(
        editor_bundle,
        session_dir=session_dir,
        load_existing_log=True,
    )
    session.apply_operation(
        "assign_points",
        {"root_id": "root-b", "indices": [0, 17, 18]},
        operation_id="assign-for-replay",
        timestamp="2026-07-25T00:00:00+00:00",
    )
    session.apply_operation(
        "redraw_root",
        {
            "root_id": "root-b",
            "points": [[0.0, 0.0, 1.0], [-0.8, 0.2, 0.7], [-1.4, 0.3, 0.2]],
        },
        operation_id="redraw-for-replay",
        timestamp="2026-07-25T00:00:01+00:00",
    )
    session.undo()
    session.redo()
    expected = _materialised_snapshot(session)

    events = [
        json.loads(line)
        for line in (session_dir / "operations.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [event["event"] for event in events] == [
        "apply",
        "apply",
        "undo",
        "redo",
    ]
    assert "indices" not in events[0]["operation"]["arguments"]
    assert events[0]["operation"]["arguments"]["indices_sha256"].startswith(
        "sha256:"
    )
    blob = session_dir / events[0]["operation"]["arguments"]["indices_blob"]
    np.testing.assert_array_equal(np.load(blob, allow_pickle=False), [0, 17, 18])

    replayed = EditorSession(
        editor_bundle,
        session_dir=session_dir,
        load_existing_log=True,
    )
    assert _materialised_snapshot(replayed) == expected
    assert replayed.public_state()["operation_count"] == 2
    assert replayed.public_state()["can_redo"] is False


def test_operation_log_rejects_a_tampered_point_index_blob(
    editor_bundle: Path,
) -> None:
    session_dir = editor_bundle.parent / "tampered-blob-session"
    session = EditorSession(
        editor_bundle,
        session_dir=session_dir,
        load_existing_log=True,
    )
    result = session.apply_operation(
        "assign_points",
        {"root_id": "root-b", "indices": [17, 18]},
        operation_id="tamper-check",
    )
    blob_path = session_dir / result["operation"]["arguments"]["indices_blob"]
    np.save(blob_path, np.array([0, 1], dtype=np.int64), allow_pickle=False)

    with pytest.raises(EditorValidationError, match="integrity check"):
        EditorSession(
            editor_bundle,
            session_dir=session_dir,
            load_existing_log=True,
        )


def test_log_write_failure_rolls_back_apply_and_removes_new_blob(
    editor_bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _new_session(editor_bundle, "apply-log-failure")
    before = _materialised_snapshot(session)

    def fail_log(_event: dict) -> None:
        raise OSError("simulated durable log failure")

    monkeypatch.setattr(session, "_append_log", fail_log)
    with pytest.raises(OSError, match="durable log failure"):
        session.apply_operation(
            "assign_points",
            {"root_id": "root-a", "indices": [17, 18]},
            operation_id="unpersisted-assignment",
        )

    assert _materialised_snapshot(session) == before
    assert session.public_state()["operation_count"] == 0
    assert session.public_state()["can_undo"] is False
    assert not (session.blob_dir / "unpersisted-assignment.npy").exists()


@pytest.mark.parametrize("history_action", ["undo", "redo"])
def test_log_write_failure_keeps_history_action_atomic(
    editor_bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
    history_action: str,
) -> None:
    session = _new_session(editor_bundle, f"{history_action}-log-failure")
    session.apply_operation(
        "assign_points",
        {"root_id": "root-a", "indices": [17]},
        operation_id=f"{history_action}-atomic-operation",
    )
    if history_action == "redo":
        session.undo()
    before = _materialised_snapshot(session)
    state_before = session.public_state()

    def fail_log(_event: dict) -> None:
        raise OSError("simulated durable log failure")

    monkeypatch.setattr(session, "_append_log", fail_log)
    with pytest.raises(OSError, match="durable log failure"):
        getattr(session, history_action)()

    assert _materialised_snapshot(session) == before
    state_after = session.public_state()
    assert state_after["operation_count"] == state_before["operation_count"]
    assert state_after["can_undo"] == state_before["can_undo"]
    assert state_after["can_redo"] == state_before["can_redo"]


@pytest.mark.parametrize(
    ("keep_id", "remove_id"),
    [
        ("root-a", "root-c"),
        ("root-c", "root-a"),
    ],
    ids=["keep-parent", "keep-child"],
)
def test_merge_direct_parent_and_child_remains_acyclic_and_replayable(
    editor_bundle: Path,
    keep_id: str,
    remove_id: str,
) -> None:
    session_dir = editor_bundle.parent / f"merge-{keep_id}"
    session = EditorSession(
        editor_bundle,
        session_dir=session_dir,
        load_existing_log=True,
    )
    session.apply_operation(
        "merge_roots",
        {"root_id": keep_id, "other_root_id": remove_id},
        operation_id=f"merge-{keep_id}-{remove_id}",
    )

    assert remove_id not in session.roots
    assert session.roots[keep_id].parent_id == "primary"
    assert session.roots[keep_id].parent_id != keep_id
    session._validate_state()
    expected = _materialised_snapshot(session)

    replayed = EditorSession(
        editor_bundle,
        session_dir=session_dir,
        load_existing_log=True,
    )
    assert _materialised_snapshot(replayed) == expected


def test_merge_rejects_roots_separated_by_an_intermediate_descendant(
    editor_bundle: Path,
) -> None:
    session = _new_session(editor_bundle, "deep-merge")
    session.apply_operation(
        "split_root",
        {
            "root_id": "root-c",
            "node_index": 2,
            "new_root_id": "root-c-distal",
        },
        operation_id="create-depth-three",
    )
    before = _materialised_snapshot(session)

    with pytest.raises(EditorValidationError, match="intermediate descendants"):
        session.apply_operation(
            "merge_roots",
            {"root_id": "root-a", "other_root_id": "root-c-distal"},
            operation_id="invalid-deep-merge",
        )

    assert _materialised_snapshot(session) == before
    assert session.public_state()["operation_count"] == 1


def test_source_bundle_is_immutable_and_session_is_bound_to_fingerprint(
    editor_bundle: Path,
) -> None:
    before = _file_digests(editor_bundle)
    session_dir = editor_bundle.parent / "immutable-session"
    session = EditorSession(
        editor_bundle,
        session_dir=session_dir,
        load_existing_log=False,
    )

    assert session.baseline_fingerprint == _expected_baseline_fingerprint(editor_bundle)
    manifest = json.loads(
        (session_dir / "session.json").read_text(encoding="utf-8")
    )
    assert manifest["baseline_fingerprint"] == session.baseline_fingerprint
    assert manifest["automatic_files_are_immutable"] is True

    session.apply_operation(
        "assign_points",
        {"root_id": "root-a", "indices": [17]},
        operation_id="immutability-check",
    )
    session.export_materialised(editor_bundle.parent / "edited-export")
    assert _file_digests(editor_bundle) == before

    same_source = EditorSession(
        editor_bundle,
        session_dir=editor_bundle.parent / "second-session",
        load_existing_log=False,
    )
    assert same_source.baseline_fingerprint == session.baseline_fingerprint


def test_existing_session_rejects_a_changed_automatic_source(
    editor_bundle: Path,
) -> None:
    session_dir = editor_bundle.parent / "fingerprint-session"
    EditorSession(
        editor_bundle,
        session_dir=session_dir,
        load_existing_log=False,
    )
    traits_path = editor_bundle / "root_traits.csv"
    traits_path.write_text(
        traits_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EditorValidationError,
        match="different automatic result",
    ):
        EditorSession(
            editor_bundle,
            session_dir=session_dir,
            load_existing_log=False,
        )


def test_editor_reuses_original_angle_window_and_gravity_configuration(
    editor_bundle: Path,
) -> None:
    (editor_bundle / "metadata.json").write_text(
        json.dumps(
            {
                "config": {
                    "tip_vector_window_mesh_units": 4.25,
                    "gravity": [0.0, -2.0, 0.0],
                }
            }
        ),
        encoding="utf-8",
    )
    session = _new_session(editor_bundle, "trait-configuration")

    assert session.tip_vector_window_mesh_units == pytest.approx(4.25)
    np.testing.assert_allclose(session.gravity, [0.0, -1.0, 0.0])
    manifest = json.loads(session.manifest_path.read_text(encoding="utf-8"))
    assert manifest["trait_configuration"] == {
        "tip_vector_window_mesh_units": 4.25,
        "gravity": [0.0, -1.0, 0.0],
    }


def test_materialised_export_is_complete_and_round_trips_labels_and_hierarchy(
    editor_bundle: Path,
    tmp_path: Path,
) -> None:
    source_before = _file_digests(editor_bundle)
    session = _new_session(editor_bundle)
    session.apply_operation(
        "assign_points",
        {"root_id": "root-a", "indices": [17, 18]},
        operation_id="export-assignment",
    )
    export_dir = session.export_materialised(tmp_path / "materialised")

    expected_names = {
        "edited_root_hierarchy.json",
        "edited_root_label_map.csv",
        "edited_root_traits.csv",
        "edited_segmented_root_structure.ply",
        "edited_root_system.rsml",
        "operations.jsonl",
        "blobs",
        "manifest.json",
    }
    assert {path.name for path in export_dir.iterdir()} == expected_names
    exported_events = [
        json.loads(line)
        for line in (export_dir / "operations.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    exported_blob = exported_events[0]["operation"]["arguments"]["indices_blob"]
    assert (export_dir / exported_blob).is_file()

    exported_mesh = read_labeled_ply(
        export_dir / "edited_segmented_root_structure.ply"
    )
    np.testing.assert_array_equal(
        exported_mesh.root_labels,
        session.mesh.root_labels,
    )
    np.testing.assert_array_equal(
        exported_mesh.assignment_states[[17, 18]],
        [1, 1],
    )
    np.testing.assert_array_equal(exported_mesh.triangles, session.mesh.triangles)

    hierarchy = json.loads(
        (export_dir / "edited_root_hierarchy.json").read_text(encoding="utf-8")
    )
    assert hierarchy["baseline_fingerprint"] == session.baseline_fingerprint
    assert {root["root_id"] for root in hierarchy["roots"]} == set(session.roots)
    traits = pd.read_csv(export_dir / "edited_root_traits.csv")
    assert set(traits["root_id"]) == set(session.roots)
    label_map = pd.read_csv(export_dir / "edited_root_label_map.csv")
    assert set(label_map["root_id"]) == {
        "uncertain",
        "unassigned",
        *session.roots,
    }
    assert dict(
        zip(label_map["root_id"], label_map["numeric_label"], strict=True)
    )["root-a"] == session.roots["root-a"].numeric_label

    rsml = ET.parse(export_dir / "edited_root_system.rsml")
    assert rsml.findtext("./metadata/unit") == "mesh_unit"
    rsml_roots = {
        element.attrib["id"]
        for element in rsml.findall("./scene/plant//root")
    }
    assert rsml_roots == set(session.roots)
    root_c = rsml.find("./scene/plant/root[@id='primary']//root[@id='root-c']")
    assert root_c is not None

    manifest = json.loads(
        (export_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["baseline_fingerprint"] == session.baseline_fingerprint
    assert manifest["active_operation_ids"] == ["export-assignment"]
    assert manifest["operation_blob_directory"] == "blobs"
    assert manifest["operation_blob_count"] == 1
    assert _file_digests(editor_bundle) == source_before


def test_editor_rejects_overlong_child_atomically(editor_bundle: Path) -> None:
    session = _new_session(editor_bundle, "overlong-child")
    before = _materialised_snapshot(session)

    with pytest.raises(EditorValidationError, match="exceeds parent"):
        session.apply_operation(
            "redraw_root",
            {
                "root_id": "root-c",
                "points": [
                    [0.5, 0.0, 3.0],
                    [0.5, 3.0, 3.0],
                ],
            },
            operation_id="invalid-overlong-child",
        )

    assert _materialised_snapshot(session) == before
    assert session.public_state()["operation_count"] == 0
    assert not session.log_path.exists()


@pytest.mark.parametrize(
    ("operation_type", "arguments", "message"),
    [
        (
            "reparent_root",
            {"root_id": "root-a", "new_parent_id": "root-c"},
            "cycle",
        ),
        (
            "assign_points",
            {"root_id": "root-a", "indices": [19]},
            "out-of-range",
        ),
        (
            "assign_points",
            {"root_id": "root-a", "indices": [-1]},
            "out-of-range",
        ),
        (
            "assign_points",
            {"root_id": "root-a", "indices": [1.5]},
            "integer",
        ),
        (
            "assign_points",
            {"root_id": "root-a", "indices": [True]},
            "integer",
        ),
        (
            "create_root",
            {
                "parent_id": "primary",
                "points": [[0.0, 0.0, 3.0], [1.0, 0.0, 3.0]],
                "indices": [5],
            },
            "unassigned",
        ),
        (
            "create_root",
            {
                "parent_id": "primary",
                "points": [[100.0, 100.0, 100.0], [100.1, 100.0, 100.0]],
                "claim_radius": 0.01,
            },
            "No unassigned",
        ),
        (
            "create_root",
            {
                "parent_id": "primary",
                "points": [[0.0, 0.0, 1.0], [-2.0, -2.0, -2.0]],
                "indices": [18],
            },
            "unassigned",
        ),
    ],
    ids=[
        "cycle",
        "index-too-large",
        "negative-index",
        "fractional-index",
        "boolean-index",
        "create-cannot-steal-assigned",
        "create-cannot-claim-uncertain",
        "create-needs-nearby-unassigned",
    ],
)
def test_invalid_cycles_and_vertex_indices_are_atomic_and_not_logged(
    editor_bundle: Path,
    operation_type: str,
    arguments: dict,
    message: str,
) -> None:
    session = _new_session(editor_bundle)
    before = _materialised_snapshot(session)

    with pytest.raises(EditorValidationError, match=message):
        session.apply_operation(
            operation_type,
            arguments,
            operation_id="invalid-operation",
        )

    assert _materialised_snapshot(session) == before
    assert session.public_state()["operation_count"] == 0
    assert session.public_state()["can_undo"] is False
    assert not session.log_path.exists()
