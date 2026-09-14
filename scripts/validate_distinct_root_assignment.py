"""Run a pinned-source whole-pipeline replay and retain raw competition evidence.

Usage: python scripts/validate_distinct_root_assignment.py SOURCE_TREE OUTPUT SAMPLE...
SAMPLE is a saved bundle containing the input/configuration metadata. Inputs are
read only; sampling is pinned to the saved analysis count, with two workers.
"""
from __future__ import annotations

import json
import inspect
import hashlib
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np


def main():
    source_tree, output = map(Path, sys.argv[1:3])
    sys.path.insert(0, str(source_tree.resolve()))
    import soyrootbio.pipeline as pipeline

    original = pipeline._assign_full_root_labels
    for sample in map(Path, sys.argv[3:]):
        metadata = json.loads((sample / "metadata.json").read_text())
        destination = output / sample.name
        destination.mkdir(parents=True, exist_ok=False)
        manifest = {
            "source_tree": str(source_tree.resolve()),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "source_hashes": {str(p.relative_to(source_tree)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((source_tree / "soyrootbio").glob("*.py"))},
        }
        (destination / "replay_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        original_lateral = pipeline._assign_lateral_points
        captured = False
        def capture_state(*args, **kwargs):
            nonlocal captured
            frame = inspect.currentframe().f_back
            if not captured and frame.f_code.co_name == "_run_pipeline_impl":
                state = dict(frame.f_locals)
                for key in ("checkpoint", "cooperate", "progress_callback", "cancel_check", "pause_check"):
                    state.pop(key, None)
                with (destination / "pre_assignment.pickle").open("wb") as stream:
                    pickle.dump(state, stream)
                captured = True
            return original_lateral(*args, **kwargs)
        pipeline._assign_lateral_points = capture_state
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            labels, pairs = result
            np.save(destination / "raw_assignment.npy", labels)
            np.save(destination / "competition.npy", np.asarray(
                [(i, *pair) for i, pair in sorted(pairs.items())], dtype=np.int64).reshape(-1, 3))
            return result
        pipeline._assign_full_root_labels = capture
        config = metadata["config"].copy()
        config.update(output_dir=destination, sample_points=metadata["point_count"], worker_threads=2)
        for key in ("input_path", "endpoint_file", "guide_file", "correction_file"):
            if config.get(key):
                config[key] = Path(config[key])
        print("START", sample.name, flush=True)
        pipeline.run_pipeline(pipeline.PipelineConfig(**config))
        pipeline._assign_lateral_points = original_lateral
        print("DONE", sample.name, flush=True)


def replay():
    """Replay the entire downstream pipeline from our own trusted snapshot.

    Snapshots must be created by this script; pickle is not an import format
    for third-party bundles. The executable tail comes from the selected code
    tree, not the snapshot. Every baseline/new replay receives fresh objects.
    """
    source_tree, output = map(Path, sys.argv[2:4])
    sys.path.insert(0, str(source_tree.resolve()))
    import soyrootbio.pipeline as pipeline
    from soyrootbio.runtime import worker_thread_limit

    source = inspect.getsource(pipeline._run_pipeline_impl)
    tail = source[source.index("    segmented_primary_mask ="):]
    original = pipeline._assign_full_root_labels
    for sample in map(Path, sys.argv[4:]):
        with (sample / "pre_assignment.pickle").open("rb") as stream:
            state = pickle.load(stream)
        destination = output / sample.name
        destination.mkdir(parents=True, exist_ok=False)
        state["config"].output_dir = destination
        if state.get("manual_guidance") is not None:
            pipeline.write_primary_guidance(
                destination / pipeline.PRIMARY_GUIDANCE_FILENAME,
                state["manual_guidance"], input_path=state["config"].input_path,
            )
        state["timings"] = dict(state["timings"])
        last = time.perf_counter()
        elapsed_before = sum(state["timings"].values())
        state["pipeline_started"] = last - elapsed_before
        def checkpoint(name, *_):
            nonlocal last
            now = time.perf_counter()
            state["timings"][name] = now - last
            last = now
        state.update(checkpoint=checkpoint, cooperate=lambda: None,
                     progress_callback=None, cancel_check=None, pause_check=None)
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            labels, pairs = result
            np.save(destination / "raw_assignment.npy", labels)
            np.save(destination / "competition.npy", np.asarray(
                [(i, *pair) for i, pair in sorted(pairs.items())], dtype=np.int64).reshape(-1, 3))
            return result
        pipeline._assign_full_root_labels = capture
        namespace = dict(vars(pipeline))
        exec("def downstream(" + ",".join(state) + "):\n" + tail, namespace)
        print("REPLAY", sample.name, flush=True)
        with worker_thread_limit(2):
            namespace["downstream"](**state)
        print("DONE", sample.name, flush=True)


if __name__ == "__main__":
    replay() if sys.argv[1] == "--replay" else main()
