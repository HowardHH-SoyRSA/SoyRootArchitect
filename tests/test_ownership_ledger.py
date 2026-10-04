import json

import numpy as np
import pytest

from soyrootbio.pipeline import (
    _unmapped_native_attachment_evidence,
    _write_ownership_evidence_ledger,
)
from soyrootbio.ownership_ledger import (
    ProvisionalEvidenceLedger,
    ProvisionalTraceEvidenceLedger,
    reconcile_provisional_attachment_interfaces,
    reconcile_released_primary_contact_vertices,
)
from soyrootbio.types import RootPath


def _cube():
    points = np.array([
        [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
        [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    roots = [
        RootPath("o1", points[[2, 5]], order=1, parent_id="primary", mean_radius=.3),
        RootPath("o2", points[[1, 6, 7]], order=2, parent_id="o1", mean_radius=.3),
    ]
    return points, faces, roots


def test_ledger_keeps_exclusion_and_commits_frozen_regions_atomically():
    labels = np.array([-1, 0, -2, -2, 1])
    ledger = ProvisionalEvidenceLedger(
        labels, excluded=np.array([True, False, False, False, False]),
        exposed_body_anchors=np.array([False, True, False, False, True]),
    )
    snapshot = ledger.snapshot()
    with pytest.raises(ValueError, match="read-only"):
        snapshot.labels[2] = 1
    ledger.record_interface([2, 3], owner_candidates=[0, 1],
                            reason="frozen_competition", eligible_for_reconsideration=True)
    committed = ledger.commit([(np.array([2]), 1, "supported"),
                               (np.array([3]), 0, "supported")], expected_generation=0)
    np.testing.assert_array_equal(committed, [-1, 0, 1, 0, 1])
    assert ledger.generation == 1
    assert len(ledger.history) == 2
    assert ledger.summary()["excluded_vertex_count"] == 1
    with pytest.raises(ValueError, match="stale"):
        ledger.commit([(np.array([2]), 1, "stale")], expected_generation=0)


def test_trace_ledger_keeps_checkpoint_candidates_and_path_versions():
    points = np.column_stack([np.arange(5, dtype=float), np.zeros((5, 2))])
    ledger = ProvisionalTraceEvidenceLedger(
        points, np.array([False, False, False, False, True]))
    root = RootPath("child", points[[1, 2, 3]].copy(), order=1,
                    covered_indices={1, 2, 3})
    labels = np.array([0, 1, -2, 1, -1])
    first = ledger.capture(labels, [root], stage="before_tip_continuation",
                           root_order=1, d_bar=.5,
                           competitor_labels={2: (0, 1)})
    assert first["generation"] == 0
    assert first["changed_path_domains"] == ["child"]
    assert first["exposed_body_anchor_count"] == 1
    assert first["owner_roots"][1]["label"] == 1
    assert first["owner_roots"][1]["root_id"] == "child"
    frozen_version = first["path_versions_sha256"]["child"]
    assert ledger.interfaces[0]["owner_candidates"] == [0, 1]
    assert not ledger.interfaces[0]["eligible_for_reconsideration"]
    second = ledger.capture(labels, [root], stage="after_order_assignment_and_attachment",
                            root_order=1, d_bar=.5,
                            released_interfaces={2: (1,)})
    assert second["generation"] == 1
    assert second["changed_path_domains"] == []
    assert second["reused_body_index_count"] == 1
    assert second["path_versions_sha256"]["child"] == frozen_version
    assert ledger.interfaces[-1]["eligible_for_reconsideration"]
    root.points[-1, 1] += .5
    third = ledger.capture(labels, [root], stage="changed_path", root_order=1,
                           d_bar=.5)
    assert third["changed_path_domains"] == ["child"]
    assert third["reused_body_index_count"] == 0
    assert third["path_versions_sha256"]["child"] != frozen_version
    assert first["path_versions_sha256"]["child"] == frozen_version
    removed = ledger.capture(np.array([0, -2, -2, -2, -1]), [],
                             stage="removed_path", root_order=1, d_bar=.5)
    assert removed["changed_path_domains"] == ["child"]
    with pytest.raises(ValueError, match="excluded"):
        ledger.capture(np.array([0, 1, -2, 1, 1]), [root],
                       stage="invalid", root_order=1, d_bar=.5)


def test_trace_checkpoint_rejects_out_of_family_candidate_atomically():
    points = np.column_stack([np.arange(3, dtype=float), np.zeros((3, 2))])
    root = RootPath("child", points.copy(), order=1, parent_id="primary")
    ledger = ProvisionalTraceEvidenceLedger(points, np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="frozen parent/order family"):
        ledger.capture(
            np.array([1, -2, 1]), [root],
            stage="after_order_assignment_and_attachment", root_order=1,
            d_bar=.5, released_interfaces={1: (0,)},
            released_interface_origins={1: 1},
        )
    assert ledger.generation == 0
    assert not ledger.checkpoints and not ledger.interfaces
    assert not ledger.label_snapshots and not ledger.native_snapshots


def test_contact_released_region_uses_native_family_and_exposed_geometry():
    points, faces, roots = _cube()
    before = np.array([0, 2, 1, 1, 1, 1, 2, 2])
    after = before.copy()
    after[1] = -2
    result, ledger, report = reconcile_released_primary_contact_vertices(
        points, before, after, points[[0, 4]], roots,
        triangles=faces, d_bar=.5,
    )
    assert result[1] == 1
    np.testing.assert_array_equal(result[[6, 7]], [2, 2])
    assert report["reassigned_vertex_count"] == 1
    assert report["raw_higher_order_primary_contacts_after"] == 0
    assert ledger.history[0]["prior_record"]["owner_candidates"] == [1, 2]
    assert ledger.history[0]["evidence_generation"] == 0


def test_provisional_attachment_uses_full_native_mesh_and_nonidentity_mapping(tmp_path):
    points, faces, roots = _cube()
    roots.append(RootPath("o3", points[[6, 7]], order=3, parent_id="o2"))
    before = np.array([0, 2, 1, 1, 1, 1, 2, 3])
    restricted = before.copy()
    restricted[1] = -2
    # The sampled analysis array omits a source vertex. Its indices are still
    # exact source-vertex identities, not nearest-coordinate guesses.
    mapping = np.array([0, 1, 2, 3, 4, 5, 6])
    sampled = before[mapping].copy()
    (
        result, interfaces, reasons, origins, commits, report, native_evidence,
    ) = reconcile_provisional_attachment_interfaces(
        points[mapping], sampled, mapping, points, before, restricted,
        points[[0, 4]], roots, triangles=faces, d_bar=.5,
        analysis_excluded_mask=np.zeros(len(mapping), dtype=bool),
        mesh_excluded_mask=np.zeros(len(points), dtype=bool), generation=1,
    )
    assert result[1] == 1
    np.testing.assert_array_equal(result[[2, 6]], [1, 2])
    assert not interfaces and not reasons and not origins
    assert len(commits) == 1
    assert commits[0]["source_owner"] == 2
    assert commits[0]["owner"] == 1
    assert commits[0]["generation"] == 1
    assert report["mapped_supported_commit_count"] == 1
    assert report["candidate_scope"] == "frozen_lateral_parent_and_same_parent_order_peers"
    assert report["native_released_vertex_records"][0]["mesh_vertex"] == 1
    assert report["native_released_vertex_records"][0]["source_owner"] == 2
    assert report["native_released_vertex_records"][0]["final_owner"] == 1
    assert "before_labels" in report["native_frozen_snapshot_sha256"]

    trace = ProvisionalTraceEvidenceLedger(
        points[mapping], np.zeros(len(mapping), dtype=bool))
    trace.capture(sampled, roots, stage="before_tip_continuation",
                  root_order=2, d_bar=.5, primary_path=points[[0, 4]])
    checkpoint = trace.capture(
        result, roots, stage="after_order_assignment_and_attachment",
        root_order=2, d_bar=.5, supported_commits=commits,
        native_evidence=native_evidence,
        native_reconciliation={"policy": report["policy"],
                               "mapped_supported_commit_count": 1},
        primary_path=points[[0, 4]],
    )
    assert checkpoint["generation"] == 1
    assert checkpoint["supported_native_commit_count"] == 1
    assert trace.payload()["commits"][0]["mesh_vertex"] == 1
    assert checkpoint["native_archive_sha256"]["released_vertices"]
    _write_ownership_evidence_ledger(
        tmp_path, trace,
        [ProvisionalEvidenceLedger(before, np.zeros(len(before), dtype=bool),
                                   np.zeros(len(before), dtype=bool))],
        [{"released_vertex_count": 0, "reassigned_vertex_count": 0,
          "unresolved_region_count": 0}],
        np.zeros(len(before), dtype=bool),
    )
    with np.load(tmp_path / "ownership_evidence_masks.npz") as archive:
        np.testing.assert_array_equal(
            archive["trace_native_checkpoint_1_analysis_to_mesh"], mapping)
        np.testing.assert_array_equal(
            archive["trace_native_checkpoint_1_released_vertices"], [1])
        np.testing.assert_array_equal(
            archive["trace_native_checkpoint_1_source_owners"], [2])
        np.testing.assert_array_equal(
            archive["trace_native_checkpoint_1_final_owners"], [1])
        np.testing.assert_array_equal(
            archive["trace_native_checkpoint_1_before_labels"], before)
        np.testing.assert_array_equal(
            archive["trace_native_checkpoint_1_restricted_labels"], restricted)
        expected_committed = restricted.copy()
        expected_committed[1] = 1
        np.testing.assert_array_equal(
            archive["trace_native_checkpoint_1_committed_labels"],
            expected_committed)
        anchor_mask = archive["trace_native_checkpoint_1_exposed_body_anchors"]
        assert np.all(archive["trace_native_checkpoint_1_restricted_labels"][anchor_mask] >= 0)


def test_provisional_open_native_interface_remains_uncertain_with_family_record():
    points, faces, roots = _cube()
    before = np.array([0, 2, 1, 1, 1, 1, 2, 2])
    restricted = before.copy()
    restricted[1] = -2
    mapping = np.arange(len(points))
    result, interfaces, reasons, origins, commits, report, native_evidence = (
        reconcile_provisional_attachment_interfaces(
            points, before, mapping, points, before, restricted,
            points[[0, 4]], roots, triangles=faces[1:], d_bar=.3,
            analysis_excluded_mask=np.zeros(len(points), dtype=bool),
            mesh_excluded_mask=np.zeros(len(points), dtype=bool), generation=1,
        )
    )
    assert result[1] == -2
    assert interfaces[1] == ()
    assert reasons[1] == "unresolved_open_or_nonmanifold_region"
    assert origins[1] == 2
    assert not commits
    assert report["mapped_unresolved_interface_count"] == 1
    np.testing.assert_array_equal(native_evidence["released_vertices"], [1])
    trace = ProvisionalTraceEvidenceLedger(points, np.zeros(len(points), dtype=bool))
    trace.capture(before, roots, stage="before_tip_continuation",
                  root_order=2, d_bar=.3)
    trace.capture(result, roots, stage="after_order_assignment_and_attachment",
                  root_order=2, d_bar=.3,
                  released_interfaces=interfaces,
                  released_interface_reasons=reasons,
                  released_interface_origins=origins)
    record = trace.payload()["interfaces"][-1]
    assert record["generation"] == 1
    assert record["source_order"] == 2
    assert record["source_parent_id"] == "o1"
    assert record["candidate_scope"] == "frozen_parent_order_native_family"


def test_provisional_rejects_primary_reclaim_and_invalid_mapping():
    points, faces, _ = _cube()
    root = RootPath("o1", points[[1, 6, 7]], order=1, parent_id="primary")
    before = np.array([0, 1, 0, 0, 0, 0, 1, 1])
    restricted = before.copy()
    restricted[1] = -2
    args = (points, before, np.arange(len(points)), points,
            before, restricted, points[[0, 4]], [root])
    options = dict(triangles=faces, d_bar=.5,
                   analysis_excluded_mask=np.zeros(len(points), dtype=bool),
                   mesh_excluded_mask=np.zeros(len(points), dtype=bool), generation=1)
    result, interfaces, reasons, _, commits, _, _ = (
        reconcile_provisional_attachment_interfaces(*args, **options))
    assert result[1] == -2
    assert interfaces[1] == ()
    assert reasons[1] == "unresolved_no_supported_candidate"
    assert not commits
    with pytest.raises(ValueError, match="distinct"):
        reconcile_provisional_attachment_interfaces(
            points, before, np.zeros(len(points), dtype=int), points,
            before, restricted, points[[0, 4]], [root], **options)
    wrong_mapping = np.arange(len(points))
    wrong_mapping[[1, 2]] = wrong_mapping[[2, 1]]
    with pytest.raises(ValueError, match="exact ordered coordinates"):
        reconcile_provisional_attachment_interfaces(
            points, before, wrong_mapping, points,
            before, restricted, points[[0, 4]], [root], **options)


def test_open_native_surface_keeps_released_region_unresolved():
    points, faces, roots = _cube()
    before = np.array([0, 2, 1, 1, 1, 1, 2, 2])
    after = before.copy()
    after[1] = -2
    # Removing one face creates an open vertex fan at the interface.
    result, ledger, report = reconcile_released_primary_contact_vertices(
        points, before, after, points[[0, 4]], roots,
        triangles=faces[1:], d_bar=.3,
    )
    np.testing.assert_array_equal(result, after)
    assert report["unresolved_region_count"] == 1
    assert ledger.records[1]["reason"] == "unresolved_open_or_nonmanifold_region"


def test_disconnected_nearby_island_cannot_authenticate_released_seam():
    points, faces, _ = _cube()
    distal = points + np.array([4., 0., 0.])
    points = np.vstack([points, distal])
    faces = np.vstack([faces, faces + 8])
    roots = [
        RootPath("o1", points[[2, 13]], order=1, parent_id="primary",
                 mean_radius=.3),
        RootPath("o2", np.array([[20., 0., 0.], [21., 0., 0.]]),
                 order=2, parent_id="o1", mean_radius=.3),
    ]
    before = np.array([0, 2, 1, 1, 1, 1, 2, 2, *([1] * 8)])
    after = before.copy()
    after[1] = -2
    result, ledger, report = reconcile_released_primary_contact_vertices(
        points, before, after, points[[0, 4]], roots,
        triangles=faces, d_bar=.5,
    )
    np.testing.assert_array_equal(result, after)
    assert report["unresolved_region_count"] == 1
    island_evidence = next(row for row in report["regions"][0]["candidate_evidence"]
                           if row["label"] == 1)
    assert island_evidence["native_boundary_vertex_count"] >= 2
    assert island_evidence["connected_exposed_boundary_vertex_count"] == 0
    assert not island_evidence["supported"]
    assert not ledger.records[1]["eligible_for_reconsideration"]


def test_export_preserves_each_trace_and_contact_pass_anchor_mask(tmp_path):
    points = np.column_stack([np.arange(3, dtype=float), np.zeros((3, 2))])
    trace = ProvisionalTraceEvidenceLedger(points, np.zeros(3, dtype=bool))
    root = RootPath("child", points.copy(), order=1, covered_indices={0, 1, 2})
    labels = np.ones(3, dtype=int)
    trace.capture(labels, [root], stage="first", root_order=1, d_bar=0.5)
    second_labels = np.array([1, -2, 1])
    trace.capture(second_labels, [root], stage="second", root_order=1, d_bar=0.5)
    ledgers = [
        ProvisionalEvidenceLedger(labels, np.zeros(3, dtype=bool),
                                  np.array([True, False, False]), generation=1),
        ProvisionalEvidenceLedger(labels, np.zeros(3, dtype=bool),
                                  np.array([False, True, False]), generation=1),
    ]
    reports = [{"released_vertex_count": 0, "reassigned_vertex_count": 0,
                "unresolved_region_count": 0}] * 2
    _write_ownership_evidence_ledger(
        tmp_path, trace, ledgers, reports, np.zeros(3, dtype=bool))
    with np.load(tmp_path / "ownership_evidence_masks.npz") as masks:
        np.testing.assert_array_equal(masks["trace_labels_checkpoint_0"], labels)
        np.testing.assert_array_equal(masks["trace_labels_checkpoint_1"], second_labels)
        assert "trace_anchor_owner_checkpoint_0" in masks
        assert "trace_anchor_owner_checkpoint_1" in masks
        np.testing.assert_array_equal(
            masks["anchor_owner_pass_0_generation_1"], [1, -1, -1])
        np.testing.assert_array_equal(
            masks["anchor_owner_pass_1_generation_1"], [-1, 1, -1])


def test_unmapped_native_release_keeps_analysis_and_archives_unresolved_evidence(tmp_path):
    analysis_points = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    analysis_labels = np.array([0, 1, -2])
    before = np.array([0, 2, 1, 2, -1])
    restricted = before.copy()
    restricted[[1, 3]] = -2
    excluded = np.array([False, False, False, False, True])
    unchanged, report, native = _unmapped_native_attachment_evidence(
        analysis_labels, before, restricted, excluded, generation=1,
    )
    np.testing.assert_array_equal(unchanged, analysis_labels)
    np.testing.assert_array_equal(analysis_labels, [0, 1, -2])
    np.testing.assert_array_equal(native["analysis_to_mesh"], [-1, -1, -1])
    np.testing.assert_array_equal(native["committed_labels"], restricted)
    np.testing.assert_array_equal(native["released_vertices"], [1, 3])
    assert report["mapped_supported_commit_count"] == 0
    assert [row["reason"] for row in report["native_released_vertex_records"]] == [
        "unresolved_no_analysis_to_native_mesh_mapping",
    ] * 2
    assert all(row["evidence_generation"] == 1
               for row in report["native_released_vertex_records"])

    trace = ProvisionalTraceEvidenceLedger(
        analysis_points, np.zeros(len(analysis_points), dtype=bool))
    root = RootPath("o1", analysis_points[[1, 2]], order=1,
                    parent_id="primary", covered_indices={1})
    trace.capture(analysis_labels, [root], stage="before_tip_continuation",
                  root_order=1, d_bar=.5)
    checkpoint = trace.capture(
        unchanged, [root], stage="after_order_assignment_and_attachment",
        root_order=1, d_bar=.5, native_evidence=native,
        native_reconciliation=report,
    )
    assert checkpoint["native_archive_sha256"]["before_labels"] == (
        report["native_frozen_snapshot_sha256"]["before_labels"])
    _write_ownership_evidence_ledger(
        tmp_path, trace,
        [ProvisionalEvidenceLedger(before, excluded, np.zeros(len(before), dtype=bool))],
        [{"released_vertex_count": 0, "reassigned_vertex_count": 0,
          "unresolved_region_count": 0}], excluded,
    )
    with np.load(tmp_path / "ownership_evidence_masks.npz") as archive:
        prefix = "trace_native_checkpoint_1_"
        for name, expected in native.items():
            np.testing.assert_array_equal(archive[prefix + name], expected)
    saved = json.loads((tmp_path / "ownership_evidence_ledger.json").read_text(encoding="utf-8"))
    saved_report = saved["trace"]["checkpoints"][1]["native_reconciliation"]
    assert saved_report["mapping_policy"] == report["mapping_policy"]
    assert saved_report["native_released_vertex_records"] == report["native_released_vertex_records"]
