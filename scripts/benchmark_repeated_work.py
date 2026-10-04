"""Compare repeated work against an explicit frozen source tree.

Prepare a bounded real Baxi candidate workload once, then run this script in
separate processes with --source-root pointing to the baseline and new source.
JSON fingerprints must match before timings are interpreted. This is a helper
benchmark, not a six-sample end-to-end performance or scientific certification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "src")
    parser.add_argument("--workdir", type=Path, default=ROOT / "performance_reuse_20261003")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--pipeline-bundle", type=Path)
    parser.add_argument("--refresh-fingerprints", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if not args.prepare and args.output is None:
        parser.error("--output is required for a benchmark or fingerprint refresh")
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    from scipy.sparse import coo_matrix
    from soyrootbio import lateral, pipeline, mesh_geometry
    from soyrootbio.runtime import worker_thread_limit
    from threadpoolctl import threadpool_limits

    args.workdir.mkdir(parents=True, exist_ok=True)
    dataset = args.workdir / "baxi_candidates.pkl"

    def fingerprint(value):
        if isinstance(value, np.ndarray):
            return {"shape": value.shape, "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
        return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

    def scientific_files(directory):
        # Compare exported scientific contents, including reloaded NPZ/XLSX
        # data, while omitting only the documented timing/path metadata.
        from openpyxl import load_workbook
        result = {}
        for file in sorted(directory.rglob("*")):
            if not file.is_file() or file.suffix not in {".json", ".npz", ".csv", ".ply", ".rsml", ".xlsx"}:
                continue
            name = file.relative_to(directory).as_posix()
            if file.suffix == ".json":
                value = json.loads(file.read_text(encoding="utf-8"))
                if file.name == "metadata.json":
                    value.pop("stage_timings_seconds", None)
                    value["config"]["output_dir"] = "<replay-output>"
                    value.get("source_geometry", {}).pop("projected_full_analysis_seconds", None)
                result[name] = fingerprint(value)
            elif file.suffix == ".rsml":
                import xml.etree.ElementTree as ET
                tree = ET.parse(file)
                tree.getroot().find("metadata/last-modified").text = "<replay-timestamp>"
                result[name] = hashlib.sha256(ET.tostring(tree.getroot())).hexdigest()
            elif file.suffix == ".npz":
                with np.load(file) as archive:
                    result[name] = {key: fingerprint(archive[key]) for key in archive.files}
            elif file.suffix == ".xlsx":
                book = load_workbook(file, read_only=True, data_only=False)
                result[name] = fingerprint({sheet.title: list(sheet.values) for sheet in book})
                book.close()
            else:
                result[name] = hashlib.sha256(file.read_bytes()).hexdigest()
        return result

    if args.refresh_fingerprints:
        archived = json.loads(args.output.read_text(encoding="utf-8"))
        archived["pipeline"]["scientific_files"] = scientific_files(args.workdir / args.output.stem)
        archived["pipeline"]["ignored_nondeterministic_fields"] = [
            "metadata.json:stage_timings_seconds", "metadata.json:config.output_dir",
            "metadata.json:source_geometry.projected_full_analysis_seconds",
            "root_system.rsml:metadata/last-modified",
        ]
        config = json.loads((args.workdir / args.output.stem / "metadata.json").read_text(encoding="utf-8"))["config"]
        archived["workers"] = config.get("worker_threads")
        archived["threadpool_limit"] = 1
        args.output.write_text(json.dumps(archived, indent=2), encoding="utf-8")
        print(json.dumps({"refreshed": str(args.output)}), flush=True)
        return

    with worker_thread_limit(1), threadpool_limits(limits=1):
        if args.prepare:
            capture = ROOT / "performance_audit_20261002" / "baxi_first_order_capture.pkl"
            with capture.open("rb") as stream:
                data = pickle.load(stream)
            ids = np.linspace(0, len(data["starts"]) - 1, 8, dtype=int)
            data["starts"] = [deepcopy(data["starts"][i]) for i in ids]
            started = time.perf_counter()
            candidates = lateral.grow_lateral_candidates(**data)
            with dataset.open("wb") as stream:
                pickle.dump({"points": data["points"], "candidates": candidates,
                             "d_bar": data["d_bar"], "primary": data["primary_path"],
                             "capture_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
                             "start_indices": ids.tolist()}, stream)
            print(json.dumps({"prepared": str(dataset), "candidates": len(candidates),
                              "seconds": time.perf_counter() - started}), flush=True)
            return

        source_hashes = {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                         for file in (args.source_root / "soyrootbio").glob("*.py")}
        result = {"source_root": str(args.source_root.resolve()), "source_hashes": source_hashes,
                  "workers": 1, "python": sys.version, "numpy": np.__version__}
        if args.pipeline_bundle:
            config = json.loads((args.pipeline_bundle / "metadata.json").read_text(encoding="utf-8"))["config"]
            config["input_path"] = Path(config["input_path"])
            config["output_dir"] = args.workdir / args.output.stem
            for name in ("endpoint_file", "guide_file", "correction_file"):
                if config.get(name):
                    config[name] = Path(config[name])
            result["workers"] = config.get("worker_threads")
            result["threadpool_limit"] = 1
            started = time.perf_counter()
            outcome = pipeline.run_pipeline(pipeline.PipelineConfig(**config))
            result["pipeline"] = {"seconds": time.perf_counter() - started,
                                   "labels": fingerprint(outcome.full_root_labels),
                                   "scientific_files": scientific_files(config["output_dir"])}
        else:
            with dataset.open("rb") as stream:
                data = pickle.load(stream)
            result["capture_sha256"] = data["capture_sha256"]
            result["start_indices"] = data["start_indices"]
            result["candidate_count"] = len(data["candidates"])
            selection_times = []
            signatures = []
            for _ in range(args.repeat):
                candidates = deepcopy(data["candidates"])
                started = time.perf_counter()
                selected = lateral.select_non_overlapping_paths(
                    candidates, data["points"], data["d_bar"], rename_selected=False,
                    initial_used=set(range(100)),
                )
                selection_times.append(time.perf_counter() - started)
                signatures.append(fingerprint([
                    {"id": path.root_id, "points": fingerprint(path.points),
                     "covered": sorted(path.covered_indices),
                     "novel": sorted(path.novel_support_indices or ()),
                     "score": path.score, "components": path.score_components}
                    for path in selected
                ]))
            assert len(set(signatures)) == 1
            result["selection"] = {"seconds": selection_times, "median_seconds": statistics.median(selection_times),
                                   "selected_count": len(selected), "fingerprint": signatures[0]}
            paths = [(0, data["primary"])] + [(i, path.points[max(0, path.body_start_index):])
                                               for i, path in enumerate(selected, 1)]
            query_points = data["points"][np.linspace(0, len(data["points"]) - 1, 10000, dtype=int)]
            query_cache = pipeline._ExposedSegmentIndexCache()
            query_times, query_signatures = [], []
            for _ in range(args.repeat):
                started = time.perf_counter()
                evidence = pipeline._nearest_exposed_segments(
                    query_points, paths, d_bar=data["d_bar"], radius=max(5 * data["d_bar"], .008),
                    margin=max(.75 * data["d_bar"], .001), segment_index_cache=query_cache,
                )
                query_times.append(time.perf_counter() - started)
                query_signatures.append([fingerprint(array) for array in evidence])
            assert all(value == query_signatures[0] for value in query_signatures)
            result["assignment"] = {"seconds": query_times, "fingerprints": query_signatures[0],
                                     "query_points": len(query_points)}

            # Isolate the old fitter's repeated full-cloud label/edge scans.
            # The new branch uses the exact compact graph consumed by fitting.
            random = np.random.default_rng(701)
            vertices_per_root, root_count = 400, 300
            points = random.normal(size=(vertices_per_root * root_count, 3))
            labels = np.repeat(np.arange(root_count), vertices_per_root)
            edges = np.column_stack([np.arange(len(points) - 1), np.arange(1, len(points))])
            edges.setflags(write=False)
            extraction_times, extraction_signatures = [], []
            for _ in range(args.repeat):
                started = time.perf_counter()
                signatures = []
                if hasattr(mesh_geometry, "OwnershipGeometryGeneration"):
                    generation = mesh_geometry.OwnershipGeometryGeneration(labels, points)
                    for label in range(root_count):
                        graph = generation.local_graph(label, edges)
                        signatures.append([fingerprint(generation.vertices(label)), fingerprint(graph.data),
                                           fingerprint(graph.indices), fingerprint(graph.indptr)])
                else:
                    same = labels[edges[:, 0]] == labels[edges[:, 1]]
                    owned_edges = edges[same]
                    for label in range(root_count):
                        ids = np.flatnonzero(labels == label)
                        lookup = np.full(len(points), -1, dtype=int)
                        lookup[ids] = np.arange(len(ids))
                        local = lookup[owned_edges[labels[owned_edges[:, 0]] == label]]
                        cloud = points[ids]
                        lengths = np.maximum(np.linalg.norm(cloud[local[:, 0]] - cloud[local[:, 1]], axis=1), 1e-15)
                        graph = coo_matrix((np.r_[lengths, lengths],
                                            (np.r_[local[:, 0], local[:, 1]], np.r_[local[:, 1], local[:, 0]])),
                                           shape=(len(ids), len(ids))).tocsr()
                        signatures.append([fingerprint(ids), fingerprint(graph.data),
                                           fingerprint(graph.indices), fingerprint(graph.indptr)])
                extraction_times.append(time.perf_counter() - started)
                extraction_signatures.append(fingerprint(signatures))
            assert len(set(extraction_signatures)) == 1
            result["grouped_graphs"] = {"seconds": extraction_times, "median_seconds": statistics.median(extraction_times),
                                        "fingerprint": extraction_signatures[0], "points": len(points), "roots": root_count}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        displayed = {key: value for key, value in result.items() if key not in {"source_hashes"}}
        if "pipeline" in displayed:
            displayed["pipeline"] = {"seconds": result["pipeline"]["seconds"],
                                     "labels": result["pipeline"]["labels"],
                                     "scientific_file_count": len(result["pipeline"]["scientific_files"])}
        print(json.dumps(displayed, indent=2), flush=True)


if __name__ == "__main__":
    main()
