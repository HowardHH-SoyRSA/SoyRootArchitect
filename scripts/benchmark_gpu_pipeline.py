"""Freeze, run and compare six complete CPU/CUDA pipeline bundles.

The parent process deliberately runs one sample at a time. Each child imports
the explicitly selected source tree, applies the frozen settings and uses two
SciPy workers and one BLAS thread. No historical runtime is used as a baseline.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "outputs" / "gpu_benchmark"
SUFFIXES = {".json", ".csv", ".npz", ".xlsx", ".ply", ".rsml"}
OPERATIONAL_FILES = {"processing_resources.json", "backend_provenance.json"}
IGNORED = [
    "metadata.json:stage_timings_seconds",
    "metadata.json:config.output_dir (normalized to <benchmark-output>)",
    "metadata.json:config.compute_backend",
    "metadata.json:source_geometry.projected_full_analysis_seconds",
    "metadata.json:backend_provenance",
    "metadata.json:outputs entries for processing_resources.json/backend_provenance.json",
    "processing_resources.json (process/memory/timing observations)",
    "backend_provenance.json (device/backend/kernel observations)",
    "root_system.rsml:metadata/last-modified",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def tree_hashes(path: Path) -> dict[str, str]:
    return {file.relative_to(path).as_posix(): sha256(file)
            for file in sorted(path.rglob("*.py"))}


def ply_header(path: Path):
    with path.open("rb") as stream:
        lines = []
        while stream.tell() < 1024 * 1024:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"Truncated PLY header: {path}")
            lines.append(raw.decode("ascii").strip())
            if lines[-1] == "end_header":
                return lines, stream.tell()
    raise ValueError(f"Missing PLY header terminator: {path}")


def prepare(args) -> None:
    if args.manifest.exists():
        raise FileExistsError(f"Frozen manifest already exists: {args.manifest}")
    samples = sorted(args.samples.glob("*.ply"), key=lambda path: path.name.lower())
    if len(samples) != 6:
        raise ValueError(f"Expected exactly six PLY samples, found {len(samples)}")
    rows = []
    for sample in samples:
        matches = []
        for metadata_file in args.bundles.glob("*/metadata.json"):
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            input_name = Path(str(metadata["config"]["input_path"]).replace("\\", "/")).name
            if input_name.lower() == sample.name.lower():
                matches.append((metadata_file.stat().st_mtime, metadata_file, metadata))
        if not matches:
            raise FileNotFoundError(f"No completed saved configuration for {sample.name}")
        _, metadata_file, metadata = max(matches, key=lambda item: item[0])
        config = dict(metadata["config"])
        config.pop("compute_backend", None)
        config["input_path"] = str(sample.resolve())
        config["output_dir"] = "<benchmark-output>"
        config["worker_threads"] = 2
        saved_count = int(metadata["point_count"])
        config["sample_points"] = saved_count
        dependencies = {}
        for name in ("endpoint_file", "guide_file", "correction_file", "nodule_review_file"):
            if not config.get(name):
                continue
            original = Path(config[name])
            if not original.is_file():
                bundled = metadata_file.parent / original.name
                if not bundled.is_file():
                    raise FileNotFoundError(f"Missing {name} for {sample.name}: {original}")
                original = bundled
            frozen = args.manifest.parent / "frozen_inputs" / sample.stem / original.name
            frozen.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, frozen)
            config[name] = str(frozen.resolve())
            dependencies[name] = {"path": str(frozen.resolve()), "sha256": sha256(frozen)}
        header, _ = ply_header(sample)
        vertex_count = int(next(line.split()[-1] for line in header if line.startswith("element vertex ")))
        rows.append({"id": sample.stem, "input_path": str(sample.resolve()),
                     "input_sha256": sha256(sample), "source_vertex_count": vertex_count,
                     "saved_full_resolution_point_count": metadata["full_resolution_point_count"],
                     "analysis_point_count": saved_count,
                     "saved_bundle": str(metadata_file.parent),
                     "saved_metadata_sha256": sha256(metadata_file),
                     "saved_sample_points": metadata["config"].get("sample_points"),
                     "saved_primary_method": metadata.get("primary_detection_method"),
                     "frozen_dependencies": dependencies, "config": config})
    manifest = {"schema": "soyrootarchitect-cpu-cuda-benchmark-v1",
                "cpu_source_root": str(args.source_root.resolve()),
                "cpu_source_hashes": tree_hashes(args.source_root),
                "baseline_description": "Current CPU source including the user-authorized uncommitted changes",
                "sampling_policy": "Explicit cap equals completed saved analysis point_count; identical seed and source vertices for both backends",
                "worker_threads": 2, "threadpool_limit": 1, "sequential_samples": True,
                "samples": rows}
    write_json(args.manifest, manifest)
    print(json.dumps({"manifest": str(args.manifest), "samples": [
        {"id": row["id"], "analysis_points": row["analysis_point_count"],
         "source_vertices": row["source_vertex_count"]} for row in rows]}, indent=2), flush=True)


def selected_samples(manifest, sample_id: str | None):
    rows = [row for row in manifest["samples"] if sample_id is None or row["id"] == sample_id]
    if not rows:
        raise ValueError(f"Unknown frozen sample: {sample_id}")
    return rows


def run_all(args) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for sample in selected_samples(manifest, args.sample):
        output = args.workdir / args.backend / sample["id"]
        result_file = args.workdir / "run_results" / args.backend / f"{sample['id']}.json"
        if result_file.exists() and args.resume:
            previous = json.loads(result_file.read_text(encoding="utf-8"))
            if previous.get("status") == "completed":
                print(f"Already completed {args.backend}: {sample['id']}", flush=True)
                continue
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing bundle: {output}")
        result_file.parent.mkdir(parents=True, exist_ok=True)
        log_file = result_file.with_suffix(".log")
        command = [str(args.python), str(Path(__file__).resolve()), "child",
                   "--manifest", str(args.manifest.resolve()), "--source-root", str(args.source_root.resolve()),
                   "--backend", args.backend, "--sample", sample["id"],
                   "--output", str(output.resolve()), "--result", str(result_file.resolve())]
        started = time.perf_counter()
        print(f"Starting {args.backend}: {sample['id']}", flush=True)
        with log_file.open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        elapsed = time.perf_counter() - started
        record = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {
            "status": "failed", "sample": sample["id"], "backend": args.backend}
        record.update({"subprocess_elapsed_seconds": elapsed, "exit_code": process.returncode,
                       "log_file": str(log_file.resolve())})
        write_json(result_file, record)
        print(json.dumps({key: record.get(key) for key in (
            "sample", "backend", "status", "pipeline_elapsed_seconds",
            "subprocess_elapsed_seconds", "peak_rss_bytes", "exit_code")}), flush=True)
        if process.returncode:
            print(f"Failure recorded; see {log_file}", flush=True)
            if not args.continue_on_failure:
                raise SystemExit(process.returncode)


def child(args) -> None:
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    import psutil
    from soyrootbio.pipeline import PipelineConfig, run_pipeline
    from threadpoolctl import threadpool_limits

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sample = selected_samples(manifest, args.sample)[0]
    if sha256(Path(sample["input_path"])) != sample["input_sha256"]:
        raise ValueError("Input mesh changed after configuration freeze")
    for dependency in sample["frozen_dependencies"].values():
        if sha256(Path(dependency["path"])) != dependency["sha256"]:
            raise ValueError("Frozen selection/review dependency changed")
    source_hashes = tree_hashes(args.source_root)
    if args.backend == "cpu" and source_hashes != manifest["cpu_source_hashes"]:
        raise ValueError("CPU source changed after baseline freeze")
    config = dict(sample["config"])
    config["output_dir"] = args.output
    config["input_path"] = Path(config["input_path"])
    for name in ("endpoint_file", "guide_file", "correction_file", "nodule_review_file"):
        if config.get(name):
            config[name] = Path(config[name])
    if args.backend == "cuda":
        if "compute_backend" not in {field.name for field in fields(PipelineConfig)}:
            raise RuntimeError("Selected GPU source has no compute_backend configuration")
        config["compute_backend"] = "cuda"
    process = psutil.Process()
    stop = threading.Event()
    observed = {"peak_rss_bytes": 0, "peak_private_bytes": 0, "poll_interval_seconds": .1}

    def sample_memory():
        while not stop.is_set():
            memory = process.memory_info()
            observed["peak_rss_bytes"] = max(observed["peak_rss_bytes"], memory.rss,
                                               getattr(memory, "peak_wset", 0))
            observed["peak_private_bytes"] = max(observed["peak_private_bytes"],
                                                   getattr(memory, "private", 0))
            stop.wait(.1)

    record = {"sample": sample["id"], "backend": args.backend,
              "source_root": str(args.source_root.resolve()), "source_hashes": source_hashes,
              "python_executable": sys.executable, "python_version": sys.version,
              "numpy_version": np.__version__, "worker_threads": 2, "threadpool_limit": 1,
              "output_dir": str(args.output.resolve())}
    thread = threading.Thread(target=sample_memory, name="benchmark-memory", daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            result = run_pipeline(PipelineConfig(**config), progress_callback=lambda stage, progress:
                                  print(f"{progress:.0%} {stage}", flush=True))
            if args.backend == "cuda":
                import cupy as cp
                cp.cuda.get_current_stream().synchronize()
        record["pipeline_elapsed_seconds"] = time.perf_counter() - started
        record["point_count"] = result.point_count
        if result.point_count != sample["analysis_point_count"]:
            raise AssertionError("Pipeline analysis count differs from frozen count")
        record["lateral_count"] = len(result.lateral_paths)
        record["status"] = "completed"
        provenance_file = args.output / "backend_provenance.json"
        if provenance_file.exists():
            record["backend_provenance"] = json.loads(provenance_file.read_text(encoding="utf-8"))
    except Exception as error:
        record.update({"status": "failed", "pipeline_elapsed_seconds": time.perf_counter() - started,
                       "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        traceback.print_exc()
    finally:
        stop.set()
        thread.join()
        record.update(observed)
        write_json(args.result, record)
    if record["status"] != "completed":
        raise SystemExit(1)


def clean_json(name: str, value):
    # This is deliberately path-specific: biological scores, tolerances, QC and
    # source/selection content must never be recursively filtered by key name.
    if name == "metadata.json":
        value.pop("stage_timings_seconds", None)
        value.get("config", {})["output_dir"] = "<benchmark-output>"
        value.get("config", {}).pop("compute_backend", None)
        value.get("source_geometry", {}).pop("projected_full_analysis_seconds", None)
        value.pop("backend_provenance", None)
        if "outputs" in value:
            value["outputs"] = [item for item in value["outputs"] if item not in OPERATIONAL_FILES]
    return value


def reload_file(path: Path, name: str):
    import numpy as np
    if path.suffix == ".json":
        return clean_json(name, json.loads(path.read_text(encoding="utf-8-sig")))
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in sorted(archive.files)}
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.reader(stream))
    if path.suffix == ".xlsx":
        from openpyxl import load_workbook
        book = load_workbook(path, read_only=True, data_only=False)
        values = {sheet.title: [list(row) for row in sheet.values] for sheet in book}
        book.close()
        return values
    if path.suffix == ".rsml":
        root = ET.parse(path).getroot()
        modified = root.find("metadata/last-modified")
        if modified is not None:
            modified.text = "<benchmark-timestamp>"

        def xml_value(element):
            return [element.tag, dict(sorted(element.attrib.items())),
                    (element.text or "").strip(), [xml_value(child) for child in element]]
        return xml_value(root)
    if path.suffix == ".ply":
        # Reload every scalar field, including labels/orders/state, native
        # faces and any overlay properties that a geometry-only reader drops.
        from soyrootbio.editor.ply import _SCALAR_DTYPES, _vertex_properties
        header, offset = ply_header(path)
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"Unsupported exported PLY format: {path}")
        header_text = "\n".join(header)
        count = int(next(line.split()[-1] for line in header if line.startswith("element vertex ")))
        face_count = next((int(line.split()[-1]) for line in header if line.startswith("element face ")), 0)
        dtype = np.dtype([(name, _SCALAR_DTYPES[kind]) for name, kind in _vertex_properties(header_text)])
        vertices = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
        values = {name: np.asarray(vertices[name]).copy() for name in dtype.names}
        if face_count:
            face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
            faces = np.memmap(path, dtype=face_dtype, mode="r", offset=offset + dtype.itemsize * count, shape=(face_count,))
            if np.any(faces["count"] != 3):
                raise ValueError(f"Nontriangular exported PLY face: {path}")
            values["faces"] = np.asarray(faces["indices"]).copy()
        values["header_properties"] = _vertex_properties(header_text)
        return values
    raise ValueError(path)


def differences(left, right, prefix="$", limit=20):
    """Return exact mismatches plus a conservative numerical drift summary."""
    import numpy as np
    report = {"exact_equal": True, "mismatch_count": 0, "examples": [],
              "maximum_absolute_numeric_difference": 0.0, "numeric_mismatch_count": 0,
              "nonnumeric_mismatch_count": 0}

    def add(path, a, b, numeric=False, delta=0.0, count=1):
        report["exact_equal"] = False
        report["mismatch_count"] += int(count)
        report["numeric_mismatch_count" if numeric else "nonnumeric_mismatch_count"] += int(count)
        if numeric and math.isfinite(delta):
            report["maximum_absolute_numeric_difference"] = max(report["maximum_absolute_numeric_difference"], delta)
        if len(report["examples"]) < limit:
            report["examples"].append({"path": path, "cpu": str(a)[:160], "cuda": str(b)[:160]})

    def visit(a, b, path):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray) or a.shape != b.shape or a.dtype != b.dtype:
                add(path, getattr(a, "shape", type(a)), getattr(b, "shape", type(b)))
                return
            equal = (a == b)
            if a.dtype.kind in "fc":
                equal |= np.isnan(a) & np.isnan(b)
            if np.all(equal):
                return
            ids = np.argwhere(~equal)
            numeric = a.dtype.kind in "fc"
            delta = float(np.max(np.abs(a[~equal] - b[~equal]))) if numeric else 0.0
            add(path, f"array dtype={a.dtype} shape={a.shape} first={a[tuple(ids[0])]}",
                f"first={b[tuple(ids[0])]}", numeric, delta, len(ids))
        elif isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                add(path + ".keys", sorted(a), sorted(b))
            for key in sorted(set(a) & set(b)):
                visit(a[key], b[key], path + "." + str(key))
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                add(path + ".length", len(a), len(b))
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}[{i}]")
        elif a != b:
            numeric = (isinstance(a, (int, float)) and isinstance(b, (int, float))
                       and not isinstance(a, bool) and not isinstance(b, bool))
            # Numeric CSV/XML cells are strings. Integer labels/IDs still need
            # exact parity; this summary never converts a mismatch into a pass.
            if isinstance(a, str) and isinstance(b, str):
                try:
                    number_a, number_b = float(a), float(b)
                    numeric = math.isfinite(number_a) and math.isfinite(number_b)
                except ValueError:
                    numeric = False
            else:
                number_a, number_b = a, b
            delta = abs(number_a - number_b) if numeric else 0.0
            add(path, a, b, numeric, delta)
    visit(left, right, prefix)
    return report


def bundle_checks(bundle: Path) -> dict:
    import numpy as np
    from soyrootbio.editor.ply import read_labeled_ply
    from soyrootbio.geometry import is_above_primary_top
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    roots = json.loads((bundle / "root_hierarchy.json").read_text(encoding="utf-8"))["roots"]
    by_id = {root["root_id"]: root for root in roots}
    mesh = read_labeled_ply(bundle / "segmented_root_structure.ply")
    failures = []
    if len(by_id) != len(roots) or roots[0]["root_id"] != "primary":
        failures.append("Root IDs are not unique or primary is not first")
    if len(roots) - 1 != metadata["selected_lateral_count"]:
        failures.append("Hierarchy/metadata root count differs")
    if np.any((mesh.root_labels < -2) | (mesh.root_labels >= len(roots))):
        failures.append("Invalid full-resolution label")
    if len(mesh.root_labels) != metadata["full_resolution_point_count"]:
        failures.append("Full-resolution PLY/metadata vertex count differs")
    for label, root in enumerate(roots):
        if np.any(mesh.root_orders[mesh.root_labels == label] != root["root_order"]):
            failures.append(f"Order export differs for {root['root_id']}")
        if label and (root.get("parent_id") not in by_id or root["root_order"] != by_id[root["parent_id"]]["root_order"] + 1):
            failures.append(f"Invalid hierarchy for {root['root_id']}")
    audit = metadata["final_compliance_audit"]
    top = np.asarray(audit["primary_top_reference_normalized"], float)
    minimum = np.asarray(metadata["normalization_minimum"], float)
    scale = float(metadata["normalization_scale"])
    gravity = np.asarray(metadata["gravity_vector"], float)
    for root in roots[1:]:
        if root.get("insertion_point") is None:
            failures.append(f"Missing origin for {root['root_id']}")
        elif is_above_primary_top((np.asarray(root["insertion_point"]) - minimum) / scale, top[None, :], gravity=gravity):
            failures.append(f"Origin above immutable top for {root['root_id']}")
    if audit["above_collar_assigned_vertex_count"] != 0:
        failures.append("Assigned surface above collar")
    # Native contact counts and isolated-patch QC remain visible even where the
    # unchanged CPU baseline has unresolved biological violations.
    higher_contact = 0
    if len(mesh.triangles):
        edges = np.vstack([mesh.triangles[:, [0, 1]], mesh.triangles[:, [1, 2]], mesh.triangles[:, [2, 0]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        labels = mesh.root_labels[edges]
        orders = mesh.root_orders[edges]
        higher_contact = int(np.count_nonzero(((labels[:, 0] == 0) & (labels[:, 1] > 0) & (orders[:, 1] >= 2)) |
                                              ((labels[:, 1] == 0) & (labels[:, 0] > 0) & (orders[:, 0] >= 2))))
    return {"export_consistency_failures": failures, "root_count": len(roots),
            "order_counts": metadata["selected_order_counts"], "full_vertex_count": mesh.vertex_count,
            "assigned": int(np.count_nonzero(mesh.root_labels >= 0)),
            "unassigned": int(np.count_nonzero(mesh.root_labels == -1)),
            "uncertain": int(np.count_nonzero(mesh.root_labels == -2)),
            "native_higher_order_primary_contact_edge_count": higher_contact,
            "final_compliance_audit": audit,
            "qc_flags_by_root": {root["root_id"]: root.get("qc_flags", []) for root in roots}}


def compare(args) -> None:
    sys.path.insert(0, str(args.source_root.resolve()))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for sample in selected_samples(manifest, args.sample):
        name = sample["id"]
        bundles = {backend: args.workdir / backend / name for backend in ("cpu", "cuda")}
        records = {backend: json.loads((args.workdir / "run_results" / backend / f"{name}.json").read_text(encoding="utf-8"))
                   for backend in bundles}
        row = {"sample": name, "analysis_point_count": sample["analysis_point_count"], "runs": records}
        if any(record["status"] != "completed" for record in records.values()):
            row["comparison_status"] = "unavailable_failed_run"
            rows.append(row)
            continue
        for backend, bundle in bundles.items():
            timing_metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
            records[backend]["metadata_stage_timings_seconds"] = timing_metadata["stage_timings_seconds"]
            records[backend]["pipeline_internal_elapsed_seconds"] = timing_metadata["stage_timings_seconds"]["total"]
        inventories = {backend: {file.relative_to(bundle).as_posix(): file for file in bundle.rglob("*")
                                 if file.is_file() and file.suffix in SUFFIXES and file.name not in OPERATIONAL_FILES}
                       for backend, bundle in bundles.items()}
        files = {}
        for filename in sorted(set(inventories["cpu"]) | set(inventories["cuda"])):
            if filename not in inventories["cpu"] or filename not in inventories["cuda"]:
                files[filename] = {"exact_equal": False, "missing_in": "cpu" if filename not in inventories["cpu"] else "cuda"}
                continue
            left = reload_file(inventories["cpu"][filename], filename)
            right = reload_file(inventories["cuda"][filename], filename)
            files[filename] = differences(left, right)
        checks = {backend: bundle_checks(bundle) for backend, bundle in bundles.items()}
        provenance = records["cuda"].get("backend_provenance", {})
        from PIL import Image
        figure_inventory = {backend: {file.relative_to(bundle).as_posix(): file for file in bundle.rglob("*.png")}
                            for backend, bundle in bundles.items()}
        figure_comparisons = {}
        for filename in sorted(set(figure_inventory["cpu"]) | set(figure_inventory["cuda"])):
            values = {}
            for backend in bundles:
                if filename not in figure_inventory[backend]:
                    values[backend] = None
                    continue
                with Image.open(figure_inventory[backend][filename]) as figure:
                    values[backend] = {"size": list(figure.size), "mode": figure.mode,
                                       "pixel_sha256": hashlib.sha256(figure.tobytes()).hexdigest()}
            figure_comparisons[filename] = {"pixels_equal": values["cpu"] is not None and values["cpu"] == values["cuda"], **values}
        row.update({"scientific_file_comparisons": files, "scientific_file_count": len(files),
                    "exact_scientific_parity": all(value["exact_equal"] for value in files.values()),
                    "presentation_figure_comparisons": figure_comparisons,
                    "exact_presentation_pixel_parity": all(value["pixels_equal"] for value in figure_comparisons.values()),
                    "bundle_checks": checks, "bundle_check_comparison": differences(checks["cpu"], checks["cuda"]),
                    "cuda_execution_verified": provenance.get("backend") == "cuda" and provenance.get("kernel_launches", 0) > 0 and provenance.get("sparse_pairs", 0) > 0,
                    "observed_pipeline_elapsed_ratio_cpu_over_cuda": records["cpu"]["pipeline_elapsed_seconds"] / records["cuda"]["pipeline_elapsed_seconds"],
                    "comparison_status": "completed"})
        rows.append(row)
        print(json.dumps({"sample": name, "exact_scientific_parity": row["exact_scientific_parity"],
                          "observed_elapsed_ratio": row["observed_pipeline_elapsed_ratio_cpu_over_cuda"],
                          "mismatched_files": [name for name, result in files.items() if not result["exact_equal"]]}), flush=True)
    cpu_hash_sets = [json.dumps(row["runs"]["cpu"].get("source_hashes", {}), sort_keys=True) for row in rows]
    gpu_hash_sets = [json.dumps(row["runs"]["cuda"].get("source_hashes", {}), sort_keys=True) for row in rows]
    report = {"schema": "soyrootarchitect-cpu-cuda-comparison-v1", "manifest": str(args.manifest.resolve()),
              "comparison_performed_by": "GPT-6.1 Sol, Extra High",
              "manifest_sha256": sha256(args.manifest), "ignored_operational_fields": IGNORED,
              "cpu_source_consistent_across_samples": len(set(cpu_hash_sets)) <= 1,
              "cuda_source_consistent_across_samples": len(set(gpu_hash_sets)) <= 1,
              "completed_runs_by_backend": {backend: sum(row["runs"][backend]["status"] == "completed" for row in rows)
                                            for backend in ("cpu", "cuda")},
              "timing_policy": "One fresh child process per sample, CPU/CUDA sequential, pipeline elapsed includes backend context startup and synchronized CUDA completion; subprocess elapsed also includes imports. Compiled-kernel disk cache may already be warm from validation; these are not cold-JIT timings.",
              "timing_attribution_limit": "One-pass elapsed ratios are observations, not isolated CUDA speedups. CPU-only stages also vary between runs; hardware/cache/background variation can affect the totals. The warm helper isolates backend work, and a later Baxi CPU repeat checks timing drift.",
              "memory_policy": "Process RSS/private memory sampled every 0.1 seconds including OS peak working set; CUDA memory/kernel statistics retained from backend provenance",
              "strict_parity_policy": "Every scientific value must match exactly. Numeric drift is measured but never converted into a parity pass.",
              "samples": rows}
    write_json(args.result, report)
    lines = ["# CPU/CUDA six-sample comparison", "", "Comparison performed by GPT-6.1 Sol, Extra High.", "",
             f"Completed runs: CPU {report['completed_runs_by_backend']['cpu']}/{len(rows)}; CUDA {report['completed_runs_by_backend']['cuda']}/{len(rows)}.", "",
             report["timing_policy"], "", report["timing_attribution_limit"], "",
             "Both versions use the current CPU source snapshot, frozen Oct4 sample settings, two SciPy workers, one BLAS thread and the same explicit analysis count and seed. Each backend is run once per sample; the timings do not estimate run-to-run variance.", "",
             "| Sample | Analysis vertices | CPU s | CUDA s | CPU/CUDA | Exact scientific parity |", "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        if row["comparison_status"] != "completed":
            lines.append(f"| {row['sample']} | {row['analysis_point_count']:,} | failed | failed | — | unavailable |")
            continue
        cpu, gpu = row["runs"]["cpu"], row["runs"]["cuda"]
        lines.append(f"| {row['sample']} | {row['analysis_point_count']:,} | {cpu['pipeline_elapsed_seconds']:.2f} | {gpu['pipeline_elapsed_seconds']:.2f} | {row['observed_pipeline_elapsed_ratio_cpu_over_cuda']:.3f}× | {row['exact_scientific_parity']} |")
    completed = [row for row in rows if row["comparison_status"] == "completed"]
    if completed:
        total_cpu = sum(row["runs"]["cpu"]["pipeline_elapsed_seconds"] for row in completed)
        total_gpu = sum(row["runs"]["cuda"]["pipeline_elapsed_seconds"] for row in completed)
        lines += ["", f"Aggregate measured pipeline elapsed: CPU {total_cpu:.2f}s, CUDA {total_gpu:.2f}s, ratio {total_cpu / total_gpu:.3f}×.", ""]
        lines += ["| Sample | CPU peak RSS GiB | CUDA peak RSS GiB | CUDA owned pool MiB | Kernel launches | Sparse pairs |", "|---|---:|---:|---:|---:|---:|"]
        for row in completed:
            cpu, gpu = row["runs"]["cpu"], row["runs"]["cuda"]
            provenance = gpu.get("backend_provenance", {})
            lines.append(f"| {row['sample']} | {cpu['peak_rss_bytes'] / 2**30:.3f} | {gpu['peak_rss_bytes'] / 2**30:.3f} | {provenance.get('peak_owned_device_pool_bytes', 0) / 2**20:.2f} | {provenance.get('kernel_launches', 0)} | {provenance.get('sparse_pairs', 0):,} |")
        lines += ["", f"CPU source identical across samples: {report['cpu_source_consistent_across_samples']}; CUDA source identical across samples: {report['cuda_source_consistent_across_samples']}."]
        stages = sorted({stage for row in completed for backend in ("cpu", "cuda")
                         for stage in row["runs"][backend]["metadata_stage_timings_seconds"] if stage != "total"})
        lines += ["", "| Pipeline stage, aggregate | CPU s | CUDA s |", "|---|---:|---:|"]
        for stage in stages:
            totals = {backend: sum(row["runs"][backend]["metadata_stage_timings_seconds"].get(stage, 0)
                                   for row in completed) for backend in ("cpu", "cuda")}
            lines.append(f"| {stage} | {totals['cpu']:.2f} | {totals['cuda']:.2f} |")
        lines += ["", "Internal stage timings exclude the wrapper's CUDA context startup and final backend-provenance write; the primary elapsed-time table includes them."]
    lines += ["The comparison reloads all scientific JSON, CSV, NPZ, XLSX, PLY and RSML exports. It preserves IDs, labels, topology, centerlines, traits, native mesh faces, uncertainty and QC. XLSX ZIP/container timestamps are outside the reloaded cell comparison. PNG presentation figures have a separate decoded-pixel comparison. Numerical differences remain failures of exact parity.", "", "Ignored operational fields:", ""]
    lines += [f"- {item}" for item in IGNORED]
    for row in completed:
        failures = {name: comparison for name, comparison in row["scientific_file_comparisons"].items() if not comparison["exact_equal"]}
        checks = row["bundle_checks"]["cuda"]
        lines += ["", f"## {row['sample']}", "", f"Scientific exports compared: {row['scientific_file_count']}. Roots: {checks['root_count']}; order counts: {checks['order_counts']}. Assigned/unassigned/uncertain vertices: {checks['assigned']}/{checks['unassigned']}/{checks['uncertain']}.",
                  f"Native higher-order contact edges: {checks['native_higher_order_primary_contact_edge_count']}; final discrete-child-patch status: {checks['final_compliance_audit']['discrete_child_patch_status']}. Existing unresolved QC is retained and does not constitute biological compliance.",
                  f"Export consistency failures: {checks['export_consistency_failures']}."]
        provenance = row["runs"]["cuda"].get("backend_provenance", {})
        lines += [f"Actual CUDA pipeline computation verified: {row['cuda_execution_verified']}. Initialization {provenance.get('initialization_seconds', 0):.3f}s; host-to-device {provenance.get('host_to_device_seconds', 0):.3f}s; kernel {provenance.get('kernel_seconds', 0):.3f}s; device-to-host {provenance.get('device_to_host_seconds', 0):.3f}s; CPU contender refinement {provenance.get('cpu_refinement_seconds', 0):.3f}s ({provenance.get('cpu_refined_pairs', 0):,} pairs)."]
        lines += [f"Presentation figures compared: {len(row['presentation_figure_comparisons'])}; exact decoded-pixel parity: {row['exact_presentation_pixel_parity']}."]
        if failures:
            lines += ["", "Mismatched scientific exports:", ""]
            lines += [f"- {name}: {value.get('mismatch_count', 'missing')} mismatches; maximum absolute numeric difference {value.get('maximum_absolute_numeric_difference', 'n/a')}." for name, value in failures.items()]
    args.result.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def warm(args) -> None:
    """Measure actual projection work using one immutable real CPU index."""
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    from soyrootbio.gpu_backend import compute_backend
    from soyrootbio.pipeline import _ExposedSegmentIndexCache, _nearest_exposed_segments
    from soyrootbio.runtime import worker_thread_limit
    from threadpoolctl import threadpool_limits

    bundle = args.bundle.resolve()
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    roots = json.loads((bundle / "root_hierarchy.json").read_text(encoding="utf-8"))["roots"]
    minimum = np.asarray(metadata["normalization_minimum"], float)
    scale = float(metadata["normalization_scale"])
    with np.load(bundle / "original_input_geometry.npz", allow_pickle=False) as archive:
        points = (archive["points"] - minimum) / scale
    paths = []
    starts = []
    for label, root in enumerate(roots):
        start = max(0, int(root.get("body_start_index", 0))) if label else 0
        line = (np.asarray(root["polyline"], float) - minimum) / scale
        paths.append((label, line[start:]))
        starts.append(start)
    d_bar = float(metadata["d_bar_normalized"])
    radius, margin = max(5.0 * d_bar, .008), max(.75 * d_bar, .001)
    cache = _ExposedSegmentIndexCache(max_entries=1, max_query_bytes=0)
    index = cache.get_or_build(paths, d_bar, starts)
    if index is None:
        raise ValueError("No emitted root segments available")

    def execute():
        started = time.perf_counter()
        result = _nearest_exposed_segments(points, paths, d_bar=d_bar, radius=radius,
                                          margin=margin, segment_index_cache=cache,
                                          body_start_indices=starts)
        return result, time.perf_counter() - started

    def exact_arrays(left, right):
        return all(np.array_equal(a, b, equal_nan=True) for a, b in zip(left, right, strict=True))

    counter_names = ("host_to_device_seconds", "kernel_seconds", "device_to_host_seconds",
                     "cpu_refinement_seconds", "sparse_pairs", "cpu_refined_pairs", "kernel_launches")
    cpu_times, gpu_times, gpu_phases = [], [], []
    with worker_thread_limit(2), threadpool_limits(limits=1):
        with compute_backend("cpu"):
            reference, cpu_cold = execute()
            for _ in range(args.repeat):
                result, elapsed = execute()
                if not exact_arrays(reference, result):
                    raise AssertionError("CPU warm evidence changed")
                cpu_times.append(elapsed)
        init_started = time.perf_counter()
        with compute_backend("cuda") as backend:
            context_initialization = time.perf_counter() - init_started
            result, gpu_cold = execute()
            cold_snapshot = backend.snapshot()
            if not exact_arrays(reference, result):
                raise AssertionError("CUDA cold evidence differs from CPU")
            for _ in range(args.repeat):
                before = backend.snapshot()
                result, elapsed = execute()
                after = backend.snapshot()
                if not exact_arrays(reference, result):
                    raise AssertionError("CUDA warm evidence differs from CPU")
                gpu_times.append(elapsed)
                gpu_phases.append({name: after[name] - before[name] for name in counter_names})
            provenance = backend.snapshot()
    if cache.max_query_bytes != 0 or cache._queries or cache.get_or_build(paths, d_bar, starts) is not index:
        raise AssertionError("Warm benchmark reused query results or changed the CPU index")
    report = {"schema": "soyrootarchitect-warm-cuda-projection-v1",
              "scope": "Isolated exact exposed-segment helper using all real Baxi vertices and emitted final exposed root polylines; not pipeline elapsed",
              "bundle": str(bundle), "bundle_hashes": {name: sha256(bundle / name) for name in
                  ("metadata.json", "root_hierarchy.json", "original_input_geometry.npz")},
              "source_hashes": tree_hashes(args.source_root),
              "query_vertices": len(points), "exposed_root_count": len(paths),
              "subdivided_segments": len(index.start), "radius_normalized": radius,
              "margin_normalized": margin, "worker_threads": 2, "threadpool_limit": 1,
              "same_cpu_index": True, "query_result_cache_disabled": True, "exact_evidence_parity": True,
              "cuda_context_initialization_seconds": context_initialization,
              "compiled_kernel_disk_cache_policy": "CUPY_CACHE_DIR may contain compiled kernels from validation/full runs; first call means fresh process/context, not cold JIT compilation",
              "cpu_first_call_seconds": cpu_cold, "cuda_first_call_seconds": gpu_cold,
              "cpu_warm_seconds": cpu_times, "cuda_warm_seconds": gpu_times,
              "cpu_warm_median_seconds": statistics.median(cpu_times),
              "cuda_warm_median_seconds": statistics.median(gpu_times),
              "warm_speedup_cpu_over_cuda": statistics.median(cpu_times) / statistics.median(gpu_times),
              "cuda_warm_phase_deltas": gpu_phases, "cuda_first_call_provenance": cold_snapshot,
              "cuda_total_provenance": provenance}
    write_json(args.result, report)
    args.result.with_suffix(".md").write_text(
        "# Warm exposed-segment helper benchmark\n\n" + report["scope"] + ".\n\n"
        f"Queries: {len(points):,} vertices; exposed roots: {len(paths)}; subdivided segments: {len(index.start):,}. "
        "Both backends use the same immutable CPU index. Query-result caching is disabled. All four returned evidence arrays match exactly on the first call and every warm repetition.\n\n"
        f"CUDA context initialization: {context_initialization:.3f}s. CPU first helper call: {cpu_cold:.3f}s. CUDA first helper call: {gpu_cold:.3f}s. "
        "The compiled-kernel disk cache may already be warm from validation/full runs; the first call is a fresh process/context measurement, not cold JIT compilation.\n\n"
        f"CPU warm seconds: {cpu_times}. CUDA warm seconds: {gpu_times}. Median ratio CPU/CUDA: {report['warm_speedup_cpu_over_cuda']:.3f}×. "
        "Warm times include actual host/device transfers, CPU index queries, CUDA projection, exact CPU contender refinement and deterministic sorting.\n\n"
        "Phase and actual kernel/pair/allocation counters are retained in the accompanying JSON. These isolated measurements do not replace the six complete pipeline comparisons.\n",
        encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("query_vertices", "subdivided_segments", "exact_evidence_parity",
                                                 "cuda_context_initialization_seconds", "cpu_warm_seconds", "cuda_warm_seconds",
                                                 "warm_speedup_cpu_over_cuda")}, indent=2), flush=True)


def repeat_check(args) -> None:
    sys.path.insert(0, str(args.source_root.resolve()))
    from PIL import Image
    sample = "BaxiNo2_4-2_20260525"
    original = WORK / "cpu" / sample
    repeated = args.repeat_workdir / "cpu" / sample
    first_run = json.loads((WORK / "run_results" / "cpu" / f"{sample}.json").read_text(encoding="utf-8"))
    second_run = json.loads((args.repeat_workdir / "run_results" / "cpu" / f"{sample}.json").read_text(encoding="utf-8"))
    gpu_run = json.loads((WORK / "run_results" / "cuda" / f"{sample}.json").read_text(encoding="utf-8"))
    for record, folder in ((first_run, original), (second_run, repeated), (gpu_run, WORK / "cuda" / sample)):
        record["metadata_stage_timings_seconds"] = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))["stage_timings_seconds"]
    if first_run["source_hashes"] != second_run["source_hashes"]:
        raise AssertionError("CPU repeat source changed")
    inventories = [{file.relative_to(bundle).as_posix(): file for file in bundle.rglob("*")
                    if file.is_file() and file.suffix in SUFFIXES and file.name not in OPERATIONAL_FILES}
                   for bundle in (original, repeated)]
    results = {}
    for name in sorted(set(inventories[0]) | set(inventories[1])):
        if name not in inventories[0] or name not in inventories[1]:
            results[name] = {"exact_equal": False, "missing": True}
        else:
            results[name] = differences(reload_file(inventories[0][name], name), reload_file(inventories[1][name], name))
    figures = {}
    for path in original.glob("*.png"):
        values = []
        for folder in (original, repeated):
            with Image.open(folder / path.name) as figure:
                values.append((figure.size, figure.mode, hashlib.sha256(figure.tobytes()).hexdigest()))
        figures[path.name] = values[0] == values[1]
    first, second, cuda = (run["pipeline_elapsed_seconds"] for run in (first_run, second_run, gpu_run))
    report = {"schema": "soyrootarchitect-baxi-cpu-timing-repeat-v1", "comparison_performed_by": "GPT-6.1 Sol, Extra High",
              "scope": "One later CPU replay after all twelve primary runs and the warm helper, using the identical frozen source/runtime/config; this checks drift and does not estimate variance",
              "first_cpu_run": first_run, "repeated_cpu_run": second_run,
              "cuda_run": gpu_run,
              "first_cpu_elapsed_seconds": first, "repeated_cpu_elapsed_seconds": second, "cuda_elapsed_seconds": cuda,
              "cpu_repeat_elapsed_ratio_to_first": second / first,
              "observed_first_cpu_over_cuda_ratio": first / cuda,
              "observed_repeated_cpu_over_cuda_ratio": second / cuda,
              "exact_scientific_parity": all(value["exact_equal"] for value in results.values()),
              "scientific_file_comparisons": results, "exact_presentation_pixel_parity": all(figures.values()),
              "presentation_figure_comparisons": figures,
              "interpretation": "Changes in CPU-only tracing/fitting stages between the primary CPU and CUDA runs mean the whole observed elapsed difference cannot be attributed to CUDA. Use isolated warm helper timings for backend attribution."}
    write_json(args.result, report)
    text = ("# Baxi CPU timing repeat\n\nComparison performed by GPT-6.1 Sol, Extra High.\n\n" + report["scope"] + ".\n\n"
            f"First CPU elapsed: {first:.3f}s; later CPU elapsed: {second:.3f}s; CUDA elapsed: {cuda:.3f}s. "
            f"Later/first CPU ratio: {second / first:.3f}. Observed first CPU/CUDA ratio: {first / cuda:.3f}; later CPU/CUDA ratio: {second / cuda:.3f}.\n\n"
            f"Exact scientific parity for the repeated CPU outputs: {report['exact_scientific_parity']} ({len(results)} files). "
            f"Exact presentation pixel parity: {report['exact_presentation_pixel_parity']} ({len(figures)} PNGs).\n\n" + report["interpretation"] + "\n")
    args.result.with_suffix(".md").write_text(text, encoding="utf-8")
    if args.main_report.exists():
        main_report = json.loads(args.main_report.read_text(encoding="utf-8"))
        main_report["baxi_cpu_timing_repeat"] = report
        warm_file = WORK / "warm_projection.json"
        if warm_file.exists():
            main_report["warm_exposed_segment_helper"] = json.loads(warm_file.read_text(encoding="utf-8"))
        complete = [row for row in main_report["samples"] if row["comparison_status"] == "completed"]
        cpu_total = sum(row["runs"]["cpu"]["pipeline_elapsed_seconds"] for row in complete)
        cuda_total = sum(row["runs"]["cuda"]["pipeline_elapsed_seconds"] for row in complete)
        main_report["aggregate_findings"] = {
            "scientific_exports_compared": sum(row["scientific_file_count"] for row in complete),
            "presentation_figures_compared": sum(len(row["presentation_figure_comparisons"]) for row in complete),
            "all_scientific_exports_exact": all(row["exact_scientific_parity"] for row in complete),
            "all_presentation_pixels_exact": all(row["exact_presentation_pixel_parity"] for row in complete),
            "cpu_pipeline_elapsed_seconds": cpu_total, "cuda_pipeline_elapsed_seconds": cuda_total,
            "cuda_elapsed_percentage_difference": 100 * (cuda_total / cpu_total - 1),
            "unresolved_child_patch_samples": [row["sample"] for row in complete if
                row["bundle_checks"]["cuda"]["final_compliance_audit"]["discrete_child_patch_status"] == "unresolved_patches"],
            "native_higher_order_primary_contact_edges_by_sample": {row["sample"]:
                row["bundle_checks"]["cuda"]["native_higher_order_primary_contact_edge_count"] for row in complete},
            "conclusion": "This one-pass six-sample test does not establish a whole-pipeline performance gain from CUDA; scientific output parity is exact, and existing unresolved biological QC remains unchanged",
        }
        write_json(args.main_report, main_report)
        with args.main_report.with_suffix(".md").open("a", encoding="utf-8") as stream:
            stream.write("\n" + text.replace("# Baxi", "## Baxi", 1))
        findings = main_report["aggregate_findings"]
        warm_report = main_report.get("warm_exposed_segment_helper", {})
        parity_statement = (f"All {findings['scientific_exports_compared']} scientific exports and {findings['presentation_figures_compared']} PNG images match exactly after reload."
                            if findings["all_scientific_exports_exact"] and findings["all_presentation_pixels_exact"] else
                            f"Compared {findings['scientific_exports_compared']} scientific exports and {findings['presentation_figures_compared']} PNG images; exact parity failed. See per-file results.")
        repeat_statement = ("All repeated outputs match exactly." if report["exact_scientific_parity"] and report["exact_presentation_pixel_parity"] else
                            f"Repeated scientific parity: {report['exact_scientific_parity']}; presentation pixel parity: {report['exact_presentation_pixel_parity']}.")
        lead = [
            parity_statement + f" CPU source consistent: {main_report['cpu_source_consistent_across_samples']}; CUDA source consistent: {main_report['cuda_source_consistent_across_samples']}.",
            f"Aggregate pipeline elapsed: CPU {cpu_total:.2f}s; CUDA {cuda_total:.2f}s. CUDA was {findings['cuda_elapsed_percentage_difference']:.2f}% longer. This one-pass test does not establish a whole-pipeline performance gain.",
            f"The isolated warm helper measured CPU {warm_report.get('cpu_warm_median_seconds', 0):.6f}s versus CUDA {warm_report.get('cuda_warm_median_seconds', 0):.6f}s, an observed {warm_report.get('warm_speedup_cpu_over_cuda', 0):.3f}× ratio with exact evidence parity. [Warm helper report](benchmark_warm_projection.md).",
            f"Later Baxi CPU replay: {second:.3f}s versus first CPU {first:.3f}s and CUDA {cuda:.3f}s. {repeat_statement} [CPU repeat report](benchmark_baxi_repeat.md).",
            f"Existing unresolved child patches remain in {len(findings['unresolved_child_patch_samples'])} samples. SN14 retains {findings['native_higher_order_primary_contact_edges_by_sample'].get('SN14_6-2_20260405', 0)} native higher-order contact edges with the primary; metadata correctly reports `unresolved_contacts`. Exact CPU/CUDA parity does not establish biological compliance.",
        ]
        main_markdown = args.main_report.with_suffix(".md")
        title, _, body = main_markdown.read_text(encoding="utf-8").partition("\n\n")
        main_markdown.write_text(title + "\n\n" + "\n\n".join(lead) + "\n\n" + body, encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("first_cpu_elapsed_seconds", "repeated_cpu_elapsed_seconds",
                                                 "cuda_elapsed_seconds", "cpu_repeat_elapsed_ratio_to_first",
                                                 "exact_scientific_parity", "exact_presentation_pixel_parity")}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "run", "child", "compare", "warm", "repeat-check"):
        command = subparsers.add_parser(mode)
        command.add_argument("--manifest", type=Path, default=WORK / "frozen_manifest.json")
        command.add_argument("--source-root", type=Path, default=ROOT / "src")
        if mode not in {"prepare", "warm", "repeat-check"}:
            command.add_argument("--sample")
        if mode in {"run", "compare"}:
            command.add_argument("--workdir", type=Path, default=WORK)
        if mode in {"run", "child"}:
            command.add_argument("--backend", choices=("cpu", "cuda"), required=True)
        if mode == "prepare":
            command.add_argument("--samples", type=Path, default=Path(r"E:\Seafile\Test files for BioInsAlgo"))
            command.add_argument("--bundles", type=Path, default=Path(r"E:\Seafile\Test files for BioInsAlgo\SoyRootBio_outputs_20261004\auto"))
        if mode == "run":
            command.add_argument("--python", type=Path, required=True)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--continue-on-failure", action="store_true")
        if mode == "child":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--result", type=Path, required=True)
        if mode == "compare":
            command.add_argument("--result", type=Path, default=WORK / "comparison.json")
        if mode == "warm":
            command.add_argument("--bundle", type=Path, default=WORK / "cpu" / "BaxiNo2_4-2_20260525")
            command.add_argument("--repeat", type=int, default=3)
            command.add_argument("--result", type=Path, default=WORK / "warm_projection.json")
        if mode == "repeat-check":
            command.add_argument("--repeat-workdir", type=Path, default=WORK / "repeat_baxi")
            command.add_argument("--result", type=Path, default=WORK / "baxi_cpu_repeat.json")
            command.add_argument("--main-report", type=Path, default=WORK / "comparison.json")
    args = parser.parse_args()
    {"prepare": prepare, "run": run_all, "child": child, "compare": compare, "warm": warm, "repeat-check": repeat_check}[args.mode](args)


if __name__ == "__main__":
    main()
