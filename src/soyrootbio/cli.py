from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline
from .synthetic import write_synthetic_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soyrootbio", description="Bio-inspired 3D soybean root skeletonization and phenotyping.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console logging level.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the full segmentation, skeletonization, and trait pipeline.")
    run.add_argument("--input", required=True, type=Path, help="Root-only point cloud or mesh exported from VG Studio.")
    run.add_argument("--output", required=True, type=Path, help="Output directory.")
    run.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Primary-root endpoint in original coordinates.")
    run.add_argument("--end", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Primary-root endpoint in original coordinates.")
    run.add_argument("--endpoint-file", type=Path, help="CSV/JSON/TXT file containing start and end endpoint coordinates.")
    run.add_argument("--auto-endpoints", choices=["z", "pca"], help="Endpoint fallback for smoke tests when manual endpoints are unavailable.")
    run.add_argument("--sample-points", type=int, default=50000, help="Uniform sample count when input is a mesh.")
    run.add_argument("--graph-k", type=int, default=14, help="Neighbor count for the local Dijkstra graph.")
    run.add_argument("--max-laterals", type=int, help="Optional cap on selected lateral roots.")
    run.add_argument("--max-root-order", type=int, default=1, help="Trace laterals recursively up to this root order.")
    run.add_argument("--unit-scale", type=float, default=1.0, help="Multiplier from input coordinate units to reported trait units.")

    synth = subparsers.add_parser("generate-synthetic", help="Write a synthetic taproot point cloud and endpoint file.")
    synth.add_argument("--output", required=True, type=Path, help="Output CSV/XYZ-style point cloud path.")
    synth.add_argument("--primary-points", type=int, default=180)
    synth.add_argument("--lateral-points", type=int, default=80)
    synth.add_argument("--lateral-count", type=int, default=3)
    synth.add_argument("--noise", type=float, default=0.004)
    synth.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    if args.command == "generate-synthetic":
        point_path, endpoint_path = write_synthetic_dataset(
            args.output,
            primary_points=args.primary_points,
            lateral_points=args.lateral_points,
            lateral_count=args.lateral_count,
            noise=args.noise,
            seed=args.seed,
        )
        print(f"Synthetic point cloud: {point_path}")
        print(f"Endpoint file: {endpoint_path}")
        return 0
    if args.command == "run":
        config = PipelineConfig(
            input_path=args.input,
            output_dir=args.output,
            start=tuple(args.start) if args.start else None,
            end=tuple(args.end) if args.end else None,
            endpoint_file=args.endpoint_file,
            auto_endpoints=args.auto_endpoints,
            sample_points=args.sample_points,
            graph_k=args.graph_k,
            lateral_max_paths=args.max_laterals,
            max_root_order=args.max_root_order,
            unit_scale=args.unit_scale,
        )
        result = run_pipeline(config)
        print(f"Processed {result.point_count} points")
        print(f"Mean nearest-neighbor distance: {result.d_bar:.6g}")
        print(f"Detected lateral starts: {result.lateral_start_count}")
        print(f"Selected lateral roots: {len(result.lateral_paths)}")
        print(f"Outputs: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


