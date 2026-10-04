"""Nodule review on editor-owned copies, sharing the normal undo/log machinery."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import numpy as np

from ..nodules import NoduleResult, NODULE_RGB, export_nodules, geometry_digest, quantify_nodules


class NoduleReviewMixin:
    def _load_nodules(self):
        path = self.output_dir / "nodules.json"
        self.nodule_objects = []
        self._nodule_detection_info = {}
        self._nodule_collar_excluded = np.zeros(self.mesh.vertex_count, dtype=bool)
        self._nodule_review_active = False
        if not path.is_file():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self._nodule_detection_info = {k: v for k, v in data.items() if k != "objects"}
        masks_path = self.output_dir / "ownership_evidence_masks.npz"
        if masks_path.is_file():
            with np.load(masks_path, allow_pickle=False) as masks:
                self._nodule_collar_excluded = np.asarray(masks["above_base_excluded"], dtype=bool).copy()
            if self._nodule_collar_excluded.shape != (self.mesh.vertex_count,):
                raise ValueError("Nodule collar mask differs from native vertices")
        digest = data.get("evidence", {}).get("geometry_sha256")
        if digest and digest != geometry_digest(self.mesh.positions):
            raise ValueError("Nodule geometry fingerprint differs from the labelled mesh")
        occupied = np.zeros(self.mesh.vertex_count, bool)
        labels, identifiers = set(), set()
        for obj in data.get("objects", []):
            v = np.asarray(obj["vertex_indices"], dtype=int)
            if not len(v) or v.min() < 0 or v.max() >= self.mesh.vertex_count or np.any(occupied[v]):
                raise ValueError("Nodule membership is invalid or overlapping")
            fingerprint = hashlib.sha256(np.sort(v).astype('<i8').tobytes()).hexdigest()
            if obj.get("geometry_fingerprint") != fingerprint or len(np.unique(v)) != len(v):
                raise ValueError("Nodule membership fingerprint differs from its vertices")
            if obj["numeric_label"] > -3 or obj["numeric_label"] in labels or obj["nodule_id"] in identifiers:
                raise ValueError("Nodules must have unique nonroot labels and identifiers")
            labels.add(obj["numeric_label"]); identifiers.add(obj["nodule_id"])
            occupied[v] = True
            if obj["status"] == "accepted" and not np.all(self.mesh.root_labels[v] == obj["numeric_label"]):
                raise ValueError("Accepted nodule does not match exported vertex labels")
            if obj["status"] == "accepted" and np.any(self._nodule_collar_excluded[v]):
                raise ValueError("An accepted nodule cannot include above-collar vertices")
            obj["restoration_labels"] = self.mesh.root_labels[v].tolist()
            self.nodule_objects.append(obj)

    def _review_nodule(self, operation):
        obj = next((o for o in self.nodule_objects if o["nodule_id"] == operation.arguments.get("nodule_id")), None)
        status = operation.arguments.get("status")
        if obj is None or status not in {"accepted", "rejected", "candidate"}:
            raise ValueError("Choose a detected nodule and a valid review decision")
        v = np.asarray(obj["vertex_indices"], int)
        self._nodule_review_active = True
        try:
            if status == "accepted":
                if np.any(self._nodule_collar_excluded[v]):
                    raise ValueError("Nodule review cannot assign above-collar vertices")
                # A label is removed only if every owned vertex belongs to this
                # independently reviewed object. Length is never a criterion.
                labels = self.mesh.root_labels
                removable = set()
                for root in self.roots.values():
                    owned = np.flatnonzero(labels == root.numeric_label)
                    if len(owned) and np.all(np.isin(owned, v)):
                        if root.root_id == "primary":
                            raise ValueError("A nodule cannot replace the primary root")
                        removable.add(root.root_id)
                for root in self.roots.values():
                    if root.parent_id in removable and root.root_id not in removable:
                        raise ValueError("This boundary contains a parent of a supported branch. Resolve its attachment before accepting the nodule.")
                if obj["status"] != "accepted":
                    obj["restoration_labels"] = labels[v].tolist()
                    obj["restoration_roots"] = {k: self.roots[k].clone() for k in removable}
                for root_id in removable:
                    del self.roots[root_id]
                affected = set(labels[v])
                for root in self.roots.values():
                    if root.numeric_label in affected:
                        root.centerline_assessment.update(status="nodule_obscured", fit_qc_passed=False,
                                                          traits_geometry_source="retained_prior_nodule_review")
                        root.qc_flags = list(dict.fromkeys([*root.qc_flags, "centerline_nodule_obscured"]))
                        obj["requires_reanalysis"] = True
                self._set_labels(v, int(obj["numeric_label"]))
            elif obj["status"] == "accepted":
                old = np.asarray(obj.get("restoration_labels", np.full(len(v), -1)), int)
                # Automatic detection precedes tracing. A rejected automatic
                # nodule needs reanalysis to discover its root topology.
                needs_rerun = bool(np.any(old <= -3))
                old[old <= -3] = -1
                for label in np.unique(old):
                    self._set_labels(v[old == label], int(label))
                self.roots.update({k: r.clone() for k, r in obj.get("restoration_roots", {}).items()})
                obj["requires_reanalysis"] = needs_rerun or obj.get("requires_reanalysis", False)
            obj["status"] = status
            obj["review_status"] = "manual"
            # Status-only changes must invalidate patch membership/public state.
            self._point_patch_revision = -1
        finally:
            self._nodule_review_active = False

    def _nodule_result(self):
        objects = [{k: deepcopy(v) for k, v in obj.items() if k not in {"restoration_labels", "restoration_roots"}}
                   for obj in self.nodule_objects]
        labels = np.where(self.mesh.root_labels <= -3, self.mesh.root_labels, -1).astype(np.int32)
        result = NoduleResult(labels, objects, self._nodule_detection_info.get("status", "complete"),
                              deepcopy(self._nodule_detection_info.get("evidence", {})))
        roots = {r.numeric_label: {"root_id": r.root_id, "order": r.order, "points": r.points, "length": r.traits.get("length")} for r in self.roots.values()}
        metadata_path = self.output_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        ref = metadata.get("final_compliance_audit", {}).get("primary_top_reference_normalized")
        top = self._baseline_roots["primary"].points[0]
        if ref is not None:
            top = np.asarray(ref) * metadata["normalization_scale"] + np.asarray(metadata["normalization_minimum"])
        quantify_nodules(result, self.mesh.positions, self.mesh.triangles, self.mesh.root_labels, roots, top, self.gravity)
        result.evidence["geometry_sha256"] = geometry_digest(self.mesh.positions)
        return result

    def _export_nodule_review(self, target):
        if not self._nodule_detection_info:
            return
        result = self._nodule_result()
        export_nodules(target, result)
        review = {"schema": "soyrootbio.nodule-review/v1", "geometry_sha256": geometry_digest(self.mesh.positions),
                  "decisions": [{"geometry_fingerprint": o["geometry_fingerprint"], "status": o["status"]}
                                for o in self.nodule_objects if o.get("review_status") == "manual"]}
        (target / "nodule_review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")

    def _nodule_patches(self):
        rows, indices = [], {}
        measured = {o["nodule_id"]: o for o in self._nodule_result().objects} if self.nodule_objects else {}
        for obj in self.nodule_objects:
            v = np.asarray(obj["vertex_indices"], dtype=np.uint32)
            xyz = self.mesh.positions[v]
            key = obj["nodule_id"]
            indices[key] = v
            rows.append({"patch_id": key, "kind": "nodule" if obj["status"] == "accepted" else "nodule_candidate",
                         "numeric_label": obj["numeric_label"], "point_count": len(v), "anchor_vertex_index": int(v.min()),
                         "centroid": xyz.mean(0).tolist(), "bounds": {"minimum": xyz.min(0).tolist(), "maximum": xyz.max(0).tolist()},
                         "membership_sha256": obj["geometry_fingerprint"], "revision": self.label_revision,
                         "indices_url": f"/api/point-patches/{key}/indices?revision={self.label_revision}",
                         "nodule": {k: value for k, value in measured[key].items() if k != "vertex_indices"}})
        return rows, indices
