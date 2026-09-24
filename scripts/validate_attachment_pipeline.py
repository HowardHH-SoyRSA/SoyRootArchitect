"""Run one current-code audit sample into an isolated, writable output folder."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soyrootbio.pipeline import PipelineConfig, run_pipeline


SAMPLES = Path(r"E:\Seafile\Test files for BioInsAlgo\SoyRootBio_outputs_20260922")
OUTPUTS = Path(__file__).resolve().parents[1] / "outputs" / "attachment_validation_20260923"


def main(sample: str, output_name: str | None = None) -> None:
    source = SAMPLES / sample
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    config = dict(metadata["config"])
    config["output_dir"] = OUTPUTS / (output_name or sample)
    for field in ("input_path", "endpoint_file", "guide_file", "correction_file"):
        if config.get(field):
            config[field] = Path(config[field])
    result = run_pipeline(
        PipelineConfig(**config),
        progress_callback=lambda stage, fraction: print(
            f"{fraction:.0%} {stage}", flush=True),
    )
    exported_metadata = json.loads(
        (result.output_dir / "metadata.json").read_text(encoding="utf-8"))
    attachment = exported_metadata["attachment_constraint"]
    summary = {
        "sample": sample,
        "selected_roots": len(result.lateral_paths),
        "per_order": [
            {"order": row["root_order"],
             "statuses": {status: sum(j["status"] == status for j in row["assessment"]["junctions"])
                          for status in sorted({j["status"] for j in row["assessment"]["junctions"]})},
             "proximal_interface_analysis_vertices": row.get(
                 "proximal_interface_analysis_vertex_count",
                 row.get("reopened_analysis_vertex_count", 0)),
             "deferred_rejected_footprint_starts": row.get(
                 "deferred_rejected_footprint_start_count", 0)}
            for row in attachment["per_order"]],
        "final_restricted_vertices": attachment["final_contact_restriction"]["changed_vertex_count"],
    }
    output = result.output_dir / "attachment_validation_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
