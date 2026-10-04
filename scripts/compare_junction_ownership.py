"""Reproduce six full PLY runs with frozen settings and an isolated source snapshot.

The manual bundles supply ownership evaluation only, never fitted centerlines.
Each invocation writes new outputs and preserves the source bundles.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time


SAMPLES = ["SN14_6-2_20260405", "Kaixinlv_3-2_20260525", "BaxiNo2_4-2_20260525",
           "w5168-3_m4-2_20260415", "W82_9cm_water_1-2_20260522", "W82_MS4-2_20260617"]


def worker(args):
    from soyrootbio import pipeline

    source = args.source / args.sample
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8-sig"))
    config = dict(metadata["config"])
    if config.get("correction_file"):
        raise ValueError("Comparison must not use an edited centerline correction file")
    config["input_path"] = Path(config["input_path"])
    config["output_dir"] = args.output / args.sample
    # Pin the actual previous analysis count, including previously automatic runs.
    config["sample_points"] = metadata["point_count"]
    for key in ("endpoint_file", "guide_file", "correction_file", "nodule_review_file"):
        if config.get(key):
            config[key] = Path(config[key])
    config = {k: v for k, v in config.items() if k in {f.name for f in fields(pipeline.PipelineConfig)}}
    out = config["output_dir"]
    (args.output / (args.sample + "_config.json")).write_text(json.dumps(config, default=str, indent=2))
    hook = "trim_parent_junctions" if hasattr(pipeline, "trim_parent_junctions") else "trim_primary_junctions"
    original = getattr(pipeline, hook)

    def capture(*pos, **kw):
        out.mkdir(parents=True, exist_ok=True)
        snapshot = (pos, {k: v for k, v in kw.items() if k != "mesh_context"})
        with (out / "before_junction_trim.pkl").open("wb") as stream:
            pickle.dump(snapshot, stream, protocol=5)
        return original(*pos, **kw)

    setattr(pipeline, hook, capture)
    if hasattr(pipeline, 'reconcile_parent_owned_tubes'):
        tube_original = pipeline.reconcile_parent_owned_tubes
        def capture_tube(*pos, **kw):
            with (out / 'before_child_tubes.pkl').open('wb') as stream:
                pickle.dump((pos, {k: v for k, v in kw.items() if k != 'mesh_context'}), stream, protocol=5)
            return tube_original(*pos, **kw)
        pipeline.reconcile_parent_owned_tubes = capture_tube
    started = time.perf_counter()
    result = pipeline.run_pipeline(pipeline.PipelineConfig(**config))
    print(json.dumps({"sample": args.sample, "seconds": time.perf_counter() - started,
                      "roots": len(result.lateral_paths) + 1}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory of six reference bundles")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--sample", choices=SAMPLES)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    snapshot = args.output / "source_snapshot"
    shutil.copytree(repo / "src", snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    hashes = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in snapshot.rglob("*.py")}
    (args.output / "source_hashes.json").write_text(json.dumps(hashes, indent=2))
    reference_files = [p for p in args.source.rglob('*') if p.is_file()]
    references = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in reference_files}
    (args.output / "reference_hashes.json").write_text(json.dumps(references, indent=2))
    env = dict(os.environ, PYTHONPATH=str(snapshot), MPLBACKEND="Agg", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")

    def run(sample):
        print("START " + sample, flush=True)
        with (args.output / (sample + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                "--source", str(args.source), "--output", str(args.output), "--sample", sample],
                env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"FINISH {sample} exit={process.returncode}", flush=True)
        return {"sample": sample, "returncode": process.returncode}

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run, [args.sample] if args.sample else SAMPLES))
    (args.output / "run_status.json").write_text(json.dumps(results, indent=2))
    unchanged = all(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == references[str(p)]
                    for p in reference_files)
    (args.output / "reference_unchanged.json").write_text(json.dumps({"unchanged": unchanged}))
    if not unchanged:
        raise RuntimeError("Reference files changed during the comparison")
    if any(r["returncode"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
