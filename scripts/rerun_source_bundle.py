"""Run a saved bundle's input/config into a separate output directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from soyrootbio.pipeline import PipelineConfig, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    output = args.output.resolve()
    if output == bundle or bundle in output.parents:
        raise ValueError("output must be separate from the source bundle")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite an existing output: {output}")
    source = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    config = dict(source["config"])
    config["input_path"] = Path(config["input_path"])
    config["output_dir"] = output
    for key in ("endpoint_file", "guide_file", "correction_file"):
        if config.get(key):
            config[key] = Path(config[key])
    if not config["input_path"].is_file():
        raise FileNotFoundError(config["input_path"])
    result = run_pipeline(
        PipelineConfig(**config),
        progress_callback=lambda stage, progress: print(
            f"{progress:.0%} {stage}", flush=True),
    )
    print(f"Completed {result.output_dir} with {len(result.lateral_paths)} laterals", flush=True)


if __name__ == "__main__":
    main()
