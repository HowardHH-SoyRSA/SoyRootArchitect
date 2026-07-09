# BioInsAlgo

BioInsAlgo is a Python MVP implementation of a bio-inspired workflow for 3D soybean root skeletonization and phenotyping from root-only point clouds or meshes. It was built to reproduce the main computational structure of the paper "3D skeletonization and phenotyping for soybean root system architecture using a bio-inspired algorithm" as a reliable first working package, not yet as a fully validated author-equivalent reproduction.

## What It Does

- Loads root-only `PLY`, `OBJ`, `STL`, `XYZ`, or `CSV` inputs where feasible.
- Converts mesh inputs to uniformly sampled point clouds with Open3D.
- Normalizes point clouds to a unit bounding box and computes mean nearest-neighbor distance `d_bar`.
- Extracts the primary root from two endpoints using a local nearest-neighbor graph and Dijkstra shortest path.
- Segments primary root points using tangent-plane neighborhoods and HDBSCAN.
- Detects lateral root starting points from non-primary points near parent-root boundaries using a percentile threshold and HDBSCAN.
- Traces lateral skeleton paths recursively up to a requested root order.
- Exports skeletons, segmented point clouds, root traits, angle-labelled figures, and skeleton overlays.

## Repository Layout

```text
BioInsAlgo/
  src/soyrootbio/                 Python package
  tests/                          Unit and synthetic end-to-end tests
  scripts/                        Helper visualization scripts
  data/synthetic/                 Small generated synthetic point cloud and endpoints
  data/real/20260525_stl/         VG Studio STL testing files
  pyproject.toml                  Installable package metadata
  requirements.txt                Dependency list
  .gitattributes                  Git LFS rules for large 3D data
```

## Installation

```powershell
cd BioInsAlgo
python -m pip install -r requirements.txt
python -m pip install -e .
```

For tests:

```powershell
python -m pytest tests
```

## Generate Synthetic Test Data

```powershell
soyrootbio generate-synthetic --output data/synthetic/synthetic_taproot.csv --lateral-count 3 --seed 21
```

This writes:

- `data/synthetic/synthetic_taproot.csv`
- `data/synthetic/synthetic_taproot_endpoints.csv`

## Run on Synthetic Data

```powershell
soyrootbio run `
  --input data/synthetic/synthetic_taproot.csv `
  --endpoint-file data/synthetic/synthetic_taproot_endpoints.csv `
  --output outputs/synthetic_run `
  --max-root-order 2 `
  --max-laterals 20
```

## Run on a VG Studio STL/PLY/OBJ/XYZ/CSV File

With an endpoint file:

```powershell
soyrootbio run `
  --input "path/to/root_only_export.stl" `
  --endpoint-file "path/to/endpoints.csv" `
  --output outputs/my_root_run `
  --sample-points 50000 `
  --max-root-order 3 `
  --max-laterals 80
```

With direct endpoints:

```powershell
soyrootbio run `
  --input "path/to/root_only_export.ply" `
  --output outputs/my_root_run `
  --start x1 y1 z1 `
  --end x2 y2 z2 `
  --sample-points 50000 `
  --max-root-order 3
```

For smoke tests only, when endpoints are not yet manually picked:

```powershell
soyrootbio run `
  --input data/real/20260525_stl/巴西2号4-2.stl `
  --output outputs/baxi2_4_2_smoke `
  --auto-endpoints z `
  --sample-points 5000
```

## Main Outputs

Each run writes to the requested output directory:

- `metadata.json`: run configuration, normalization, `d_bar`, point counts, and root-order counts.
- `primary_skeleton.csv`: primary root skeleton nodes.
- `lateral_skeletons.csv`: non-primary skeleton nodes with `root_id`, `parent_id`, and `root_order`.
- `root_traits.csv`: root length, parent/root order, base angle, tip angle to parent, tip angle to primary root, tip angle to Z-axis, point counts, and radius estimates.
- `segmented_points.ply`: point cloud colored by root class/order.
- `skeleton_original_overlay.ply`: gray original point cloud plus colored skeleton overlay.
- `overview.png`: 3D overview.
- `tip_angles_front_view_600dpi.png`: 600 dpi front view with angle labels.

## Color Code

- Primary root: blue
- Order 1 laterals: red
- Order 2 laterals: green
- Order 3 laterals: purple
- Higher-order laterals: gold
- Unassigned/original background: gray

## Data

The included real test data are root-only STL exports from `20260525 CT扫描结果`. The files are intended for testing and method development. They are tracked with Git LFS because the complete STL set is about 583 MB.

If cloning this repository, install Git LFS first and then run:

```powershell
git lfs install
git lfs pull
```

## Current Limitations

- This is an MVP engineering reproduction, not a verified bit-for-bit reproduction of the paper author's implementation.
- Automatic endpoint modes are for smoke tests; manually selected primary endpoints are recommended for measurement-quality runs.
- Recursive order tracing is heuristic and needs validation against annotated roots.
- Dense high-sample runs can be slow because graph and clustering steps are currently CPU-bound Python/scikit-learn workflows.
- Inputs should be root-only. Soil, pot, stem, or strong segmentation noise should be removed before analysis.

## Development Checks

The current prepared repository was validated with:

```powershell
python -m pytest tests
```

Expected result: all tests pass.
