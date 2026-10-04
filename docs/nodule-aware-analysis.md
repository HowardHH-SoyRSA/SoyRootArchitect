# Nodule-aware analysis

The analysis window has a **Nodule-aware analysis** checkbox, initially **off**. Turn it on before starting a new analysis. Accepted nodule-like bulges are displayed in pale yellow-white **#FFF4B3 (255, 244, 179)**, distinct from every existing root-order and assignment color. With the checkbox off, the original analysis path runs without calling the nodule detector. Loaded review decisions are ignored while the GUI checkbox is off.

## Detection and segmentation

This is a conservative morphology detector, not a biological confirmation of nodulation. It examines the original mesh before lateral tracing, independently of existing root IDs. Inward opposing-face ray chords propose local thickness; native mesh edges, compactness, normal spread, local neck contrast and surface coverage define connected bulge objects. The initial primary tube and frozen collar exclusion are protected. Rays cannot borrow thickness from a disconnected mesh component. Deterministic overlap arbitration prevents double assignment.

Accepted objects are excluded from root assignment and all downstream cleanup. They have their own stable object IDs and negative numeric labels starting at -3. They never become roots, root orders or hierarchy nodes. Existing unassigned (-1) and uncertain (-2) meanings are unchanged. Source coordinates and triangles are preserved.

Open/nonmanifold surfaces, weak necks and competing tubular shapes stay **candidates** for review and retain normal root processing until accepted. Native mesh connectivity is required for automatic acceptance; point-only input reports an unresolved nodule analysis. Neighboring roots are not joined across missing surface. If the same accepted nodule separates a previously traced root's support, its previous path is retained for topology and marked `nodule_obscured`; affected length, diameter and volume measurements are unavailable. The hidden root is not reconstructed or silently measured through the nodule. Existing contact/patch violations remain explicit QC.

## Separate measurements

The exported bundle includes:

| File | Contents |
| --- | --- |
| `nodules.json` | Object membership, classification, review status, detection evidence and measurements |
| `nodule_traits.csv` | One row per accepted, candidate or rejected object, with an explicit status |
| `nodule_summary.csv` | Accepted count, review counts, observed surface area and density per available measured root length |
| `nodule_depth_distribution.csv` | Accepted count and area in ten depth intervals |
| `nodules_by_root.csv` | Accepted count, area and density by supporting root; unresolved associations remain blank |
| `nodule_points.ply` | Accepted nodule points in pale yellow-white |
| `traits.xlsx` | Separate Nodules, Nodule summary and Nodule depth sheets |

Size includes three principal dimensions, observed mesh surface area, and volume/equivalent diameter where defensible. Surface area is partitioned by vertex thirds at class boundaries, so the root and nodule shares never overlap. Volume from a planar attachment cap is explicitly marked as an estimate; irregular, open or nonmanifold boundaries have no volume. No cap geometry is added to the exported mesh. Units remain source mesh units, squared units and cubed units.

Position includes the XYZ centroid, depth along gravity below the immutable primary top, and—only with a dominant native-adjacent supporting root—an attachment projection and distance/fraction along that root. The projection is an estimate from the stored path. Whole-system root totals exclude nodules; original source-mesh totals remain separately named. Density denominators use only available measured root lengths and can be incomplete where root geometry is unresolved.

## Review in the 3D editor

Accepted nodules and review candidates appear in separate lists with independent nodule visibility. Selecting an object shows its size, location, supporting root and QC. **Accept as nodule**, **Keep as root**, and **Mark unresolved** are undoable and recorded in the editor log. Whole-root reclassification only removes a root record when all of its owned surface belongs to the reviewed object; it refuses to remove a parent with a supported descendant outside that object. Collar exclusions remain protected.

Manual acceptance can make a retained root's earlier fit stale; its affected measurements are withheld pending reanalysis. Rejecting an automatically accepted nodule also requires reanalysis because it was excluded before tracing. Export the edits, select the original sample in the analysis window, choose **Load nodule review…**, load `nodule_review.json`, and run to a fresh output directory with the same input and primary guidance. The review file checks source geometry and detected boundary fingerprints; changed geometry or boundaries require a new review. Automatic source bundles are never modified by editor review.

Equivalent command-line usage:

```powershell
.venv\Scripts\python.exe -m soyrootbio.cli run --input sample.stl --output new-output --nodule-aware
.venv\Scripts\python.exe -m soyrootbio.cli run --input sample.stl --output reviewed-output --nodule-aware --nodule-review-file nodule_review.json
```

Omit `--nodule-aware` for the original algorithms. Detection thresholds are currently internal and scale relative to native mesh spacing and observed local thickness. Validate candidate classifications for the scan/reconstruction protocol before treating these morphology-based counts as biological nodule counts.
