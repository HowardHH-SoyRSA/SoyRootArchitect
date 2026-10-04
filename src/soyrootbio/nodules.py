"""Conservative, mesh-native nodule morphology and separate object measurements.

No physical unit or biological identity is inferred. Ray chords propose local
thickness; only native edges join surface patches. Open/nonmanifold patches,
weak neck evidence and competing tubular geometry remain review candidates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, minimum_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .mesh_geometry import MeshGeometryContext

NODULE_RGB = (255, 244, 179)  # #FFF4B3, deliberately distinct from order-4 gold.
NODULE_COLOR = np.asarray(NODULE_RGB, dtype=float) / 255.0
NODULE_SCHEMA = "soyrootbio.nodules/v1"


@dataclass
class NoduleResult:
    vertex_labels: np.ndarray
    objects: list[dict] = field(default_factory=list)
    status: str = "complete"
    evidence: dict = field(default_factory=dict)

    @property
    def mask(self):
        return self.vertex_labels <= -3

    def public(self):
        return {"schema": NODULE_SCHEMA, "enabled": True, "status": self.status,
                "classification": "nodule_like_morphology", "color_rgb": list(NODULE_RGB),
                "coordinate_unit": "mesh_unit", "method": "native-ray-chord-compactness-neck-v1",
                "accepted_count": sum(o["status"] == "accepted" for o in self.objects),
                "candidate_count": sum(o["status"] == "candidate" for o in self.objects),
                "objects": self.objects, "evidence": self.evidence}


def _graph(n, edges):
    return coo_matrix((np.ones(2 * len(edges)),
                       (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(n, n)).tocsr()


def surface_thickness(points, triangles, context, cooperate=None):
    """Area-weighted inward, opposing-face ray chords; never a solid volume."""
    from .io import require_open3d
    o3d = require_open3d()
    p, f = np.asarray(points), np.asarray(triangles)
    graph = _graph(len(p), context.edges[context.edge_incidence == 2])
    _, components = connected_components(graph, directed=False)
    face_components = components[f[:, 0]]
    xyz = p[f]
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(twice_area[:, None], 1e-30)
    # Use a translated origin for numerical stability. A globally reversed
    # surface is supported; locally inconsistent winding stays unresolved.
    signed = np.einsum("ij,ij->i", xyz[:, 0] - p.mean(0), cross).sum()
    if signed < 0:
        normals = -normals
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.core.Tensor(p.astype(np.float32)),
                        o3d.core.Tensor(f.astype(np.uint32)))
    spacing = float(np.median(context.edge_lengths))
    eps = max(spacing * 1e-3, np.ptp(p, axis=0).max() * 1e-7)
    chord = np.zeros(len(f))
    good = np.zeros(len(f), dtype=bool)
    for start in range(0, len(f), 50000):
        if cooperate:
            cooperate()
        end = min(start + 50000, len(f))
        rays = np.c_[xyz[start:end].mean(1) - eps * normals[start:end], -normals[start:end]]
        hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)), nthreads=2)
        dist = hit["t_hit"].numpy().astype(float) + eps
        hit_normals = hit["primitive_normals"].numpy() * (-1 if signed < 0 else 1)
        hit_faces = hit["primitive_ids"].numpy().astype(np.int64)
        same_component = np.zeros(end - start, dtype=bool)
        has_hit = hit_faces < len(f)
        same_component[has_hit] = face_components[hit_faces[has_hit]] == face_components[start:end][has_hit]
        valid = (np.isfinite(dist) & (dist > 2 * eps)
                 & same_component
                 & (np.einsum("ij,ij->i", hit_normals, normals[start:end]) < -.25))
        chord[start:end] = np.where(valid, dist, 0)
        good[start:end] = valid
    weighted = np.bincount(f.ravel(), weights=np.repeat(chord * twice_area, 3), minlength=len(p))
    areas = np.bincount(f.ravel(), weights=np.repeat(twice_area, 3), minlength=len(p))
    coverage = np.bincount(f.ravel(), weights=np.repeat(good * twice_area, 3), minlength=len(p))
    diameter = weighted / np.maximum(areas, 1e-30)
    coverage /= np.maximum(areas, 1e-30)
    degree = np.asarray(graph.sum(1)).ravel() + 1
    for _ in range(3):
        diameter = (graph @ diameter + diameter) / degree
    return diameter, coverage, normals, graph


def _primary_envelope(points, path, diameter, spacing):
    """Protect a conservative observed host tube, not the inflated outer wall."""
    if path is None or len(path) < 2:
        return np.zeros(len(points), bool)
    path = np.asarray(path)
    distance, station = cKDTree(path).query(points, workers=1)
    radii = np.full(len(path), np.nan)
    order = np.argsort(station, kind="stable")
    cuts = np.r_[0, np.flatnonzero(np.diff(station[order])) + 1, len(order)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        ids = order[a:b]
        near = ids[distance[ids] < np.maximum(diameter[ids] * .9, 3 * spacing)]
        if len(near) >= 5:
            radii[station[ids[0]]] = np.quantile(distance[near], .25)
    known = np.flatnonzero(np.isfinite(radii))
    if not len(known):
        return np.zeros(len(points), bool)
    radii = np.interp(np.arange(len(path)), known, radii[known])
    radii = median_filter(minimum_filter1d(radii, size=7, mode="nearest"), size=5, mode="nearest")
    return distance <= np.maximum(1.05 * radii[station], spacing)


def detect_nodules(points, triangles, *, primary_path=None, excluded_mask=None,
                   mesh_context=None, cooperate: Callable | None = None) -> NoduleResult:
    p = np.asarray(points, dtype=float)
    result = NoduleResult(np.full(len(p), -1, dtype=np.int32))
    excluded = np.zeros(len(p), bool) if excluded_mask is None else np.asarray(excluded_mask, bool)
    if excluded.shape != (len(p),):
        raise ValueError("nodule exclusion mask must match native vertices")
    if triangles is None or not len(triangles):
        result.status = "unresolved_no_native_mesh"
        result.evidence["reason"] = "Native connectivity and neck boundaries cannot be established."
        return result
    ctx = mesh_context or MeshGeometryContext.build(p, triangles)
    ctx.validate(p, triangles)
    f = ctx.triangles
    diameter, coverage, normals, graph = surface_thickness(p, f, ctx, cooperate)
    spacing = float(np.median(ctx.edge_lengths))
    protected = _primary_envelope(p, primary_path, diameter, spacing)
    usable = ~excluded & ~protected & (coverage > .55)
    positive = diameter[usable & (diameter > 3 * spacing)]
    result.evidence.update({"native_vertex_count": len(p), "native_face_count": len(f),
                           "ray_coverage_fraction": float(np.mean(coverage > .55)),
                           "protected_primary_vertex_count": int(protected.sum()),
                           "threshold_policy": "relative_to_native_spacing_and_local_ray_chords"})
    if len(positive) < 30:
        return result
    levels = np.geomspace(max(3 * spacing, np.quantile(positive, .3)), np.quantile(positive, .997), 24)
    proposals = []
    bad_vertices = np.zeros(len(p), bool)
    bad_vertices[ctx.edges[ctx.edge_incidence != 2].ravel()] = True
    for level in levels:
        if cooperate:
            cooperate()
        ids = np.flatnonzero(usable & (diameter >= level))
        if len(ids) < 30:
            continue
        _, cc = connected_components(graph[ids][:, ids], directed=False)
        groups = np.argsort(cc, kind="stable")
        cuts = np.r_[0, np.flatnonzero(np.diff(cc[groups])) + 1, len(groups)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < 30:
                continue
            v = ids[groups[a:b]]
            q = p[v]
            eig = np.linalg.eigvalsh(np.cov(q.T))
            compactness = float(eig[0] / max(eig[-1], 1e-30))
            if compactness < .10:
                continue
            member = np.zeros(len(p), bool)
            member[v] = True
            face_ids = np.flatnonzero(np.all(member[f], axis=1))
            if len(face_ids) < 20:
                continue
            weights = ctx.face_areas[face_ids]
            ns = normals[face_ids]
            neig = np.linalg.eigvalsh((ns * weights[:, None]).T @ ns / weights.sum())
            normal_spread = float(neig[0])
            if normal_spread < .07:
                continue
            neighbors = np.unique(graph[v].indices)
            rim = neighbors[~member[neighbors] & ~excluded[neighbors]]
            if len(rim) < 5:
                continue
            peak = float(np.quantile(diameter[v], .9))
            neck = float(np.median(diameter[rim]))
            ratio = peak / max(neck, spacing)
            surface_coverage = float(weights.sum() / (np.pi * peak**2))
            if surface_coverage < .25:
                continue
            if ratio < 1.18 or np.ptp(q, axis=0).max() > 3.5 * peak:
                continue
            qc = []
            if np.any(bad_vertices[v]):
                qc.append("open_or_nonmanifold_surface")
            if ratio < 1.35:
                qc.append("weak_neck_contrast")
            if compactness < .16 or normal_spread < .10:
                qc.append("competing_tubular_geometry")
            if surface_coverage < .45:
                qc.append("incomplete_bulge_surface")
            # A candidate on the collar boundary cannot be automatically cut.
            if np.any(excluded[neighbors]):
                qc.append("collar_boundary_candidate")
            score = float(compactness * normal_spread * ratio * np.sqrt(weights.sum()))
            proposals.append((score, v, {"compactness": compactness, "normal_spread": normal_spread,
                                       "neck_diameter_ratio": ratio, "ray_diameter": peak,
                                       "surface_coverage": surface_coverage,
                                       "threshold": float(level)}, qc))
    # Frozen, deterministic overlap arbitration. Do not assemble a blob out of
    # disconnected labels or merge neighboring objects by Euclidean distance.
    claimed = np.zeros(len(p), bool)
    accepted = []
    for score, v, evidence, qc in sorted(proposals, key=lambda x: (-x[0], int(x[1].min()))):
        if np.any(claimed[v]):
            continue
        # A prior axis can run on a bulb wall and protect a small island of
        # that same bulb. Reclaim only enclosed native components whose whole
        # surface still has the independently measured bulb thickness. A tube
        # extending outside the component cannot pass this test.
        member = np.zeros(len(p), bool); member[v] = True
        remainder = np.flatnonzero(~member)
        _, complement = connected_components(graph[remainder][:, remainder], directed=False)
        order = np.argsort(complement, kind="stable")
        cuts = np.r_[0, np.flatnonzero(np.diff(complement[order])) + 1, len(order)]
        holes = []
        # A protected island can contain the outermost vertex of the bulb.
        # Permit one native edge spacing around its observed bounding box.
        lower, upper = p[v].min(0) - spacing, p[v].max(0) + spacing
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a > max(4, len(v) // 4):
                continue
            patch = remainder[order[a:b]]
            if (not np.any(excluded[patch] | claimed[patch] | bad_vertices[patch])
                    and np.all(diameter[patch] >= evidence["threshold"])
                    and np.all(p[patch] >= lower) and np.all(p[patch] <= upper)):
                neighbors = np.unique(graph[patch].indices)
                if np.any(member[neighbors]):
                    holes.extend(patch.tolist())
        if holes:
            v = np.sort(np.r_[v, holes])
            evidence["enclosed_bulge_vertices_recovered"] = len(holes)
        claimed[v] = True
        accepted.append((v, evidence, qc))
    accepted.sort(key=lambda x: tuple(np.mean(p[x[0]], axis=0)))
    for index, (v, evidence, qc) in enumerate(accepted):
        label = -3 - index
        row = {"nodule_id": f"nodule-{index + 1:03d}", "numeric_label": label,
               "status": "candidate" if qc else "accepted", "review_status": "automatic",
               "vertex_indices": sorted(map(int, v)), "vertex_count": len(v),
               "geometry_fingerprint": hashlib.sha256(np.sort(v).astype('<i8').tobytes()).hexdigest(),
               "evidence": evidence, "qc_flags": qc, "supporting_root_id": None}
        result.objects.append(row)
        if not qc:
            result.vertex_labels[v] = label
    return result


def _patch_volume(points, faces, member):
    """Volume of an observed face patch, with explicit planar-neck cap estimates."""
    local = np.asarray(faces)[np.all(member[faces], axis=1)]
    if len(local) < 4:
        return None, "unavailable_insufficient_surface"
    directed = np.concatenate([local[:, [0, 1]], local[:, [1, 2]], local[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        return None, "unavailable_nonmanifold_surface"
    boundary = directed[counts[inverse] == 1]
    origin = points[np.unique(local)].mean(0)
    xyz = points[local] - origin
    volume6 = np.einsum('ij,ij->i', xyz[:, 0], np.cross(xyz[:, 1], xyz[:, 2])).sum()
    if not len(boundary):
        return abs(float(volume6)) / 6, "native_closed_surface"
    unique, degree = np.unique(boundary, return_counts=True)
    if np.any(degree != 2):
        return None, "unavailable_irregular_attachment_boundary"
    _, components = connected_components(_graph(len(points), boundary)[unique][:, unique], directed=False)
    if components.max() > 1:
        return None, "unavailable_multiple_attachment_boundaries"
    for component in np.unique(components):
        vertices = unique[components == component]
        rim = points[vertices]
        ev = np.linalg.eigvalsh(np.cov(rim.T))
        if ev[0] / max(ev[-1], 1e-30) > .08:
            return None, "unavailable_nonplanar_attachment_boundary"
        centroid = rim.mean(0) - origin
        edges = boundary[np.isin(boundary[:, 0], vertices)]
        # Reverse directed boundary edges to close the existing winding.
        a, b = points[edges[:, 1]] - origin, points[edges[:, 0]] - origin
        volume6 += np.einsum('ij,ij->i', a, np.cross(b, centroid)).sum()
    return abs(float(volume6)) / 6, "estimated_planar_neck_cap_on_observed_faces"


def quantify_nodules(result, points, triangles, root_labels, roots, primary_top, gravity=(0, 0, -1)):
    """Measure observed patches; attachment caps are explicit volume estimates."""
    p, f = np.asarray(points, float), np.asarray(triangles if triangles is not None else [], int).reshape(-1, 3)
    gravity = np.asarray(gravity, float)
    gravity = gravity / np.linalg.norm(gravity)
    ctx = MeshGeometryContext.build(p, f)
    graph = _graph(len(p), ctx.edges)
    result.evidence["measured_root_lengths"] = {
        r["root_id"]: float(r["length"]) for r in roots.values()
        if r.get("length") is not None and np.isfinite(r["length"]) and r["length"] > 0
    }
    for obj in result.objects:
        v = np.asarray(obj["vertex_indices"], int)
        q = p[v]
        center = np.average(q, axis=0, weights=np.maximum(ctx.vertex_area_weights[v], 1e-30))
        _, axes = np.linalg.eigh(np.cov(q.T))
        dimensions = np.sort(np.ptp((q - center) @ axes, axis=0))[::-1]
        member = np.zeros(len(p), bool); member[v] = True
        fraction = np.mean(member[f], axis=1) if len(f) else np.empty(0)
        surface = float(np.dot(ctx.face_areas, fraction))
        neighbors = np.unique(graph[v].indices)
        adjacent = neighbors[~member[neighbors] & (np.asarray(root_labels)[neighbors] >= 0)]
        owners, counts = np.unique(np.asarray(root_labels)[adjacent], return_counts=True)
        supporting = None
        attachment = None
        arc = None
        arc_fraction = None
        if len(owners):
            best = int(np.argmax(counts))
            if counts[best] / counts.sum() >= .7 and int(owners[best]) in roots:
                supporting = roots[int(owners[best])]
                boundary = p[adjacent[np.asarray(root_labels)[adjacent] == owners[best]]].mean(0)
                path = np.asarray(supporting["points"])
                if len(path) >= 2:
                    delta = np.diff(path, axis=0)
                    t = np.clip(np.einsum('ij,ij->i', boundary - path[:-1], delta) / np.maximum(np.sum(delta**2, axis=1), 1e-30), 0, 1)
                    projection = path[:-1] + t[:, None] * delta
                    k = int(np.argmin(np.linalg.norm(projection - boundary, axis=1)))
                    attachment = projection[k].tolist()
                    arc = float(np.linalg.norm(delta[:k], axis=1).sum() + t[k] * np.linalg.norm(delta[k]))
                    arc_fraction = arc / max(float(np.linalg.norm(delta, axis=1).sum()), 1e-30)
        # Native observed faces are not capped silently. No hull or sphere
        # replacement can turn an open patch into a measured solid volume.
        volume, volume_method = _patch_volume(p, f, member)
        if "open_or_nonmanifold_surface" in obj["qc_flags"]:
            volume, volume_method = None, "unavailable_open_or_nonmanifold_surface"
        obj.update({"centroid": center.tolist(), "depth_below_primary_top": float((center - primary_top) @ gravity),
                    "principal_dimensions": dimensions.tolist(), "surface_area": surface,
                    "surface_area_method": "partitioned_native_mesh_triangles",
                    "volume": volume, "volume_method": volume_method,
                    "equivalent_sphere_diameter": (6 * volume / np.pi)**(1/3) if volume is not None else None,
                    "length_unit": "mesh_unit", "area_unit": "mesh_unit^2", "volume_unit": "mesh_unit^3",
                    "supporting_root_id": supporting["root_id"] if supporting else None,
                    "supporting_root_order": supporting["order"] if supporting else None,
                    "attachment_point": attachment, "distance_along_supporting_root": arc,
                    "fraction_along_supporting_root": arc_fraction})
        if supporting is None and "supporting_root_unresolved" not in obj["qc_flags"]:
            obj["qc_flags"].append("supporting_root_unresolved")
    return result


def export_nodules(directory, result):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "nodules.json").write_text(json.dumps(result.public(), indent=2, allow_nan=False), encoding="utf-8")
    rows = []
    for o in result.objects:
        row = {k: v for k, v in o.items() if k not in {"vertex_indices", "evidence", "centroid", "principal_dimensions", "attachment_point", "qc_flags"}}
        row.update(dict(zip(("centroid_x", "centroid_y", "centroid_z"), o.get("centroid", [None]*3))))
        row.update(dict(zip(("major_size", "middle_size", "minor_size"), o.get("principal_dimensions", [None]*3))))
        row["qc_flags"] = ";".join(o["qc_flags"])
        rows.append(row)
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["nodule_id", "status", "surface_area", "volume", "depth_below_primary_top"])
    frame.to_csv(directory / "nodule_traits.csv", index=False)
    accepted = frame[frame.status == "accepted"]
    measured_length = sum(result.evidence.get("measured_root_lengths", {}).values())
    summary = pd.DataFrame([{"accepted_nodule_count": len(accepted), "unresolved_candidate_count": int((frame.status == "candidate").sum()),
                             "rejected_candidate_count": int((frame.status == "rejected").sum()),
                             "nodules_per_measured_root_length": len(accepted) / measured_length if measured_length else None,
                             "measured_root_length": measured_length,
                             "total_observed_nodule_surface_area": float(accepted.surface_area.sum()) if len(accepted) else 0,
                             "volume_available_count": int(accepted.volume.notna().sum()) if len(accepted) else 0}])
    summary.to_csv(directory / "nodule_summary.csv", index=False)
    depth = pd.DataFrame(columns=["depth_start", "depth_end", "nodule_count", "surface_area"])
    if len(accepted):
        bins = np.linspace(float(accepted.depth_below_primary_top.min()), float(accepted.depth_below_primary_top.max()) + 1e-9, 11)
        ids = np.minimum(np.searchsorted(bins, accepted.depth_below_primary_top, side="right") - 1, 9)
        depth = pd.DataFrame([{"depth_start": bins[i], "depth_end": bins[i+1], "nodule_count": int(np.sum(ids == i)),
                               "surface_area": float(accepted.loc[ids == i, "surface_area"].sum())} for i in range(10)])
    depth.to_csv(directory / "nodule_depth_distribution.csv", index=False)
    by_root = []
    if len(accepted):
        for root_id, group in accepted.groupby("supporting_root_id", dropna=False):
            length = result.evidence.get("measured_root_lengths", {}).get(root_id)
            by_root.append({"supporting_root_id": root_id, "supporting_root_order": group.supporting_root_order.iloc[0],
                            "nodule_count": len(group), "surface_area": float(group.surface_area.sum()),
                            "nodules_per_measured_root_length": len(group) / length if length else None})
    pd.DataFrame(by_root, columns=["supporting_root_id", "supporting_root_order", "nodule_count", "surface_area", "nodules_per_measured_root_length"]).to_csv(directory / "nodules_by_root.csv", index=False)
    return frame, summary, depth


def geometry_digest(points):
    return hashlib.sha256(np.ascontiguousarray(points, dtype='<f8').tobytes()).hexdigest()


def apply_nodule_review(result, path, points, excluded):
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    if review.get("geometry_sha256") != geometry_digest(points):
        raise ValueError("Nodule review geometry does not match this input")
    by_id = {o["geometry_fingerprint"]: o for o in result.objects}
    for decision in review.get("decisions", []):
        obj = by_id.get(decision.get("geometry_fingerprint"))
        if obj is None:
            raise ValueError("Reviewed nodule boundary differs from this detection run")
        action = decision.get("status")
        if action not in {"accepted", "rejected", "candidate"}:
            raise ValueError("Invalid nodule review decision")
        v = np.asarray(obj["vertex_indices"], int)
        if action == "accepted" and np.any(np.asarray(excluded)[v]):
            raise ValueError("Nodule review cannot assign above-collar vertices")
        obj["status"] = action
        obj["review_status"] = "manual"
        result.vertex_labels[v] = obj["numeric_label"] if action == "accepted" else -1
