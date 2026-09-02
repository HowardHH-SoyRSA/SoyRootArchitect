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

    gui = subparsers.add_parser("gui", help="Open the desktop application.")
    gui.add_argument("--input", type=Path, help="Optional root file to prefill in the desktop application.")
    gui.add_argument("--output", type=Path, help="Optional output directory to prefill in the desktop application.")

    editor = subparsers.add_parser(
        "editor",
        help="Open an existing SoyRootBio output in the interactive 3D graph editor.",
    )
    editor.add_argument(
        "--output",
        required=True,
        type=Path,
        help="SoyRootBio output directory containing the labelled PLY and root hierarchy.",
    )
    editor.add_argument(
        "--session-dir",
        type=Path,
        help="Optional directory for the append-only operation log and materialised exports.",
    )
    editor.add_argument("--host", default="127.0.0.1")
    editor.add_argument("--port", type=int, default=8765)
    editor.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the local editor server without opening a browser.",
    )

    run = subparsers.add_parser("run", help="Run the full segmentation, skeletonization, and trait pipeline.")
    run.add_argument("--input", required=True, type=Path, help="Root-only point cloud or mesh exported from VG Studio.")
    run.add_argument("--output", required=True, type=Path, help="Output directory.")
    run.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Primary-root endpoint in original coordinates.")
    run.add_argument("--end", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Primary-root endpoint in original coordinates.")
    run.add_argument("--endpoint-file", type=Path, help="CSV/JSON/TXT file containing start and end endpoint coordinates.")
    run.add_argument("--auto-endpoints", choices=["scored", "z", "pca"], default="scored", help="Automatic primary detector; scored is the measurement default.")
    run.add_argument("--soil-z", type=float, help="Manual horizontal soil-line Z used by the scored collar detector.")
    run.add_argument("--guide-file", type=Path, help="CSV/JSON/TXT XYZ points that the primary centerline must cross.")
    run.add_argument("--correction-file", type=Path, help="Edited root_hierarchy.json from an earlier run.")
    run.add_argument("--sample-points", type=int, default=0, help="Optional analysis vertex cap; 0 keeps the full mesh unless the 30-minute/memory preflight requires reduction.")
    run.add_argument("--graph-k", type=int, default=14, help="Neighbor count for the local Dijkstra graph.")
    run.add_argument("--max-laterals", type=int, help="Optional cap on selected lateral roots.")
    run.add_argument("--max-root-order", type=int, default=3, help="Trace laterals recursively up to this root order.")
    run.add_argument("--runtime-limit-minutes", type=float, default=30.0, help="Projected runtime threshold before limited analysis reduction is allowed.")
    run.add_argument("--minimum-retained-fraction", type=float, default=0.25, help="Smallest automatic fraction of mesh vertices to retain (0-1).")
    run.add_argument(
        "--tip-window-mesh-units",
        type=float,
        default=2.0,
        help="Arc-length window, in source mesh units, used for local tip, lateral-start, and primary-reference vectors.",
    )

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
    if args.command == "gui":
        from .desktop_gui import launch_gui

        return launch_gui(args.input, args.output)
    if args.command == "editor":
        from .editor.server import launch_editor

        launch_editor(
            args.output,
            session_dir=args.session_dir,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )
        return 0
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
            soil_z=args.soil_z,
            guide_file=args.guide_file,
            correction_file=args.correction_file,
            sample_points=args.sample_points or None,
            graph_k=args.graph_k,
            lateral_max_paths=args.max_laterals,
            max_root_order=args.max_root_order,
            runtime_limit_minutes=args.runtime_limit_minutes,
            minimum_retained_fraction=args.minimum_retained_fraction,
            tip_vector_window_mesh_units=args.tip_window_mesh_units,
        )
        result = run_pipeline(config)
        print(f"Processed {result.point_count} points")
        print(f"Normalized mean nearest-neighbor distance: {result.d_bar:.6g}")
        print(f"Detected lateral starts: {result.lateral_start_count}")
        print(f"Selected lateral roots: {len(result.lateral_paths)}")
        if result.primary_candidates:
            print(f"Primary candidate confidence: {result.primary_candidates[0].confidence:.3f}")
        print(f"Outputs: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


