"""Reproducible synthetic diagnostic benchmark, independent of tracing accuracy.

Run: python scripts/benchmark_point_evidence.py --output validation_runs/point_evidence
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from soyrootbio.point_evidence import assess_point_only_evidence
from soyrootbio.runtime import worker_thread_limit
from soyrootbio.types import RootPath


def tube(rings=60, sectors=32):
    theta = np.arange(sectors) * 2 * np.pi / sectors
    z = np.linspace(0, 2, rings)
    points = np.column_stack([np.tile(0.1 * np.cos(theta), rings),
                              np.tile(0.1 * np.sin(theta), rings),
                              np.repeat(z, sectors)])
    faces = []
    for row in range(rings - 1):
        for col in range(sectors):
            a, b = row * sectors + col, row * sectors + (col + 1) % sectors
            faces.extend([[a, b, a + sectors], [b, b + sectors, a + sectors]])
    return points, np.asarray(faces, int), (points[:, 2] >= 1).astype(int)


def cases():
    points, faces, labels = tube()
    yield "connected_tube_seam", points, labels, faces, False, "surface_points", True
    for name, gap in [("separated_tubes", 0.4), ("small_unmeshed_gap", 0.015)]:
        p = points.copy()
        p[labels == 1, 2] += gap
        f = faces[np.all(labels[faces] == labels[faces[:, :1]], axis=1)]
        yield name, p, labels, f, False, "surface_points", False
    x, y = np.meshgrid(np.linspace(0, 1, 24), np.linspace(0, 1, 24))
    sheet = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    p = np.vstack([sheet, sheet + [0, 0, 0.04]])
    yield "near_parallel_sheets", p, np.repeat([0, 1], len(sheet)), None, False, "surface_points", False
    keep = (labels == 0) | ((np.arange(len(points)) // 32) % 2 == 0)
    yield "unequal_density", points[keep], labels[keep], None, False, "surface_points", True
    keep = (points[:, 2] < 0.85) | (points[:, 2] > 1.15)
    yield "missing_sectors_at_contact", points[keep], labels[keep], None, False, "surface_points", True
    yield "collar_excluded_child", points, labels, faces, labels == 1, "surface_points", True
    rng = np.random.default_rng(42)
    p = rng.uniform([-0.1, -0.1, 0], [0.1, 0.1, 2], size=(5000, 3))
    p = p[np.linalg.norm(p[:, :2], axis=1) <= 0.1]
    yield "occupied_cylinder", p, (p[:, 2] >= 1).astype(int), None, False, "occupied_volume", True
    yield "sparse_points", points[[0, -1]], np.array([0, 1]), None, False, "surface_points", False


def benchmark_case(name, points, labels, faces, excluded, mode, physical_contact):
    primary = np.array([[0, 0, 0], [0, 0, 1]], float)
    root = RootPath("order2-probe", np.array([[0, 0, 1], [0, 0, 2]], float), order=2, parent_id="unobserved-order1")
    started = time.perf_counter()
    report = assess_point_only_evidence(points, labels, primary, [root], input_mode=mode,
                                       excluded_mask=None if excluded is False else excluded)
    seconds = time.perf_counter() - started
    native_contact = None
    if faces is not None:
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        native_contact = bool(np.any(labels[edges[:, 0]] != labels[edges[:, 1]]))
    observed = bool(report["contacts"])
    tangent_observed = any(row["tangent_compatible_edge_count"] > 0 for row in report["contacts"])
    assert report["changed_vertex_count"] == 0 and not report["mesh_generated"]
    assert report["native_contact_compliance"] == "unresolved_no_mesh"
    return {
        "case": name, "point_count": len(points), "input_mode": mode,
        "geometry_sha256": hashlib.sha256(points.astype("<f8").tobytes() + labels.astype("<i8").tobytes()).hexdigest(),
        "physical_contact_in_constructed_fixture": physical_contact,
        "native_reference_contact": native_contact,
        "contact_excluded": excluded is not False,
        "proximity_contact_observed": observed, "tangent_contact_observed": tangent_observed,
        "seconds": seconds, "report": report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with worker_thread_limit(4):
        rows = [benchmark_case(*case) for case in cases()]
        scaling = []
        for rings in (20, 100, 400):
            points, faces, labels = tube(rings=rings)
            scaling.append(benchmark_case(f"scaling_{len(points)}", points, labels, faces, False, "surface_points", True))
    data = {
        "benchmark": "synthetic-point-evidence-v1", "seed": 42, "worker_threads": 4,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(),
        "evidence_source_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "src" / "soyrootbio" / "point_evidence.py").read_bytes()).hexdigest(),
        "scope": "Assigned synthetic fixtures isolate point evidence. This is not segmentation accuracy, real-soybean validation, or a guarantee of physical contact.",
        "cases": rows, "scaling": scaling,
    }
    (args.output / "benchmark.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    table = ["# Point-only evidence benchmark", "", data["scope"], "",
             "| Fixture | Points | Native reference contact | Proximity observed | Tangent observed | Seconds |",
             "| --- | ---: | --- | --- | --- | ---: |"]
    for row in rows + scaling:
        table.append(f"| {row['case']} | {row['point_count']} | {row['native_reference_contact']} | {row['proximity_contact_observed']} | {row['tangent_contact_observed']} | {row['seconds']:.4f} |")
    table += ["", "All cases retain unresolved native contact/patch status, make zero ownership changes, and generate no mesh.", "",
              "A small physical gap can yield both proximity and tangent evidence. Missing contact samples can hide real contact. These cases explain why diagnostic neighborhoods cannot certify the repository's native-mesh rules.", "",
              "The JSON records full reports, geometry hashes, fixed seeds and runtime versions. Synthetic timings depend on the machine and sampling density; no real-data accuracy claim follows.", ""]
    (args.output / "README.md").write_text("\n".join(table), encoding="utf-8")
    print("\n".join(table))


if __name__ == "__main__":
    main()
