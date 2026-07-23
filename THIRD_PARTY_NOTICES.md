# Research and third-party notices

This file records the source lineage, research references, and license decisions used while developing SoyRootBio/BioInsAlgo 0.2. It is not a substitute for the license text of any named project.

## What is incorporated

The repository is an evolution of HowardHH-SoyRSA's [BioInsAlgo `agent/refine-primary-centerline-log` branch](https://github.com/HowardHH-SoyRSA/BioInsAlgo/tree/agent/refine-primary-centerline-log), whose repository license is MIT. That lineage is compatible with this repository's [MIT License](LICENSE).

No source code from the reference-only projects below is claimed to be copied, linked, vendored, or redistributed in this tree. Runtime Python packages are installed separately by the user through normal package management and remain under their own licenses.

## Reference matrix

| Resource | Relevance to this project | License/use decision |
|---|---|---|
| [Zhou et al. (2025), *3D skeletonization and phenotyping for soybean root system architecture using a bio-inspired algorithm*](https://doi.org/10.1016/j.compag.2025.110890) | Motivated the original BioInsAlgo primary/lateral graph workflow and soybean trait focus. | Article reported under CC BY-NC 4.0; no author software was identified or incorporated. Ideas are cited, and this implementation is not presented as the authors' code. |
| [BioInsAlgo baseline](https://github.com/HowardHH-SoyRSA/BioInsAlgo/tree/agent/refine-primary-centerline-log) | Direct code lineage for package structure, graph-based primary/lateral processing, CLI, and original desktop prototype. | MIT; reused and substantially extended under the same MIT terms. |
| [Plant 3D (P3D)](https://github.com/iziamtso/P3D) and its [Bioinformatics paper](https://doi.org/10.1093/bioinformatics/btaa220) | Reference for GUI-centred 3D point-cloud phenotyping, graph skeletons, and semantic labels. | Repository is MIT. It was evaluated as a reference; no P3D C++/model code is included. |
| [4DRoot](https://github.com/TIDOP-USAL/4DRoot) and [Herrero-Huerta et al. (2022)](https://doi.org/10.3389/fpls.2022.986856) | Reference for STL input, cylindrical/root quantitative structure models, hierarchy, per-order traits, and Excel-style reporting. | No explicit software license was found in the inspected repository, so its MATLAB code is reference-only and was not copied. The Frontiers article is CC BY. |
| [TopoRoot](https://github.com/danzeng8/TopoRoot) and [Zeng et al. (2021)](https://doi.org/10.1186/s13007-021-00829-z) | Reference for hierarchy-first CT phenotyping, topological repair, skeletonization, parent-child traits, QC, and batch processing. | No explicit repository software license was found during review; concepts/publication only, no copied code. The article is open access under CC BY 4.0. |
| [TopoRoot+](https://doi.org/10.1186/s13007-024-01240-0) | Reference for automated soil-line detection, whorl/level traits, and an editable hierarchy interface. | Article is CC BY 4.0. Associated distribution was treated as reference-only because no permissive software license was established for reuse. |
| [DIRT/3D](https://doi.org/10.1093/insilicoplants/diab039) | Reference for 3D point-cloud root phenotyping, level-set/slice analysis, trait validation, and containerized high-throughput workflows. | Paper and concepts cited. GPL-covered reconstruction/analysis code was not incorporated into this MIT project. |
| [Rootine v2](https://doi.org/10.1186/s13007-021-00735-4) | Reference for tubular/vesselness-based segmentation of roots in X-ray CT volumes and diameter-aware processing. | Fiji/ImageJ macro and publication are reference-only; no Rootine code is included. |
| [archiDART](https://archidart.github.io/), [source](https://github.com/archidart/archidart), and [Delory et al.](https://doi.org/10.1007/s11104-015-2673-4) | Reference for per-root architecture traits, topology analysis, RSML interoperability, and time-series analysis. | GNU GPL v2.0; not copied or linked. SoyRootBio's RSML writer is an independent implementation of an interchange format. |
| [DynamicRoots](https://doi.org/10.1371/journal.pone.0127657) | Reference for reconstructing growing root hierarchies and using temporal information to resolve topology. | Publication/algorithm reference only; no code or data incorporated. |
| [GiA Roots paper](https://doi.org/10.1186/1471-2229-12-116), [Topp Roots Lab tools](https://github.com/Topp-Roots-Lab), and [Gia3D](https://github.com/Topp-Roots-Lab/Gia3D) | Reference for high-throughput RSA trait extraction and 3D reconstruction/skeleton workflows. | Reference/validation context only; no code or trained assets incorporated. Individual repository licenses must be checked before any future reuse. |
| [RooTrak](https://doi.org/10.1104/pp.111.186221) | Reference for automated 3D recovery of roots from soil in X-ray micro-CT using visual tracking. | Publication/algorithm reference only; this project expects already reconstructed root-only geometry and does not incorporate RooTrak. |
| [RooTh](https://github.com/rootth/rootth) and the [RooTrak/RooTh protocol](https://doi.org/10.1002/cppb.20049) | Reference for CT-root segmentation and a practical reconstruction/quantification workflow. | GPL-covered software was not incorporated into this MIT project. |
| [VRoot paper](https://doi.org/10.1016/j.plaphe.2025.100013) and [implementation](https://github.com/dhelmrich/VRoot) | Reference for immersive manual 3D RSA tracing, graph editing, and RSML authoring. | Reference-only; no Unreal Engine, VR, or VRoot source is included. Check the repository and Unreal/third-party licenses before reuse. |
| [RootForce, Gerth et al. (2021)](https://doi.org/10.34133/2021/8747930) | Reference for semi-automated vesselness-based 3D root segmentation and Reeb-graph/skeleton trait extraction from CT imagery. | Publication/algorithm reference only; RootForce code is not included. The article is CC BY 4.0. |
| [open_iA](https://github.com/3dct/open_iA), [JOSS article](https://doi.org/10.21105/joss.01185), and [Zenodo release](https://doi.org/10.5281/zenodo.2591999) | Reference for extensible CT visualization, interactive inspection, filtering, and optional GPU/OpenCL processing. | GPL-licensed C++ application; no code is incorporated or linked. |
| [Lin et al. (2025), *3D Plant Root Skeleton Detection and Extraction*](https://arxiv.org/abs/2508.08094v1) | Reference for multi-view primary/lateral detection, triangulation, and integrated 3D skeleton extraction. | Preprint/reference only; no associated code was incorporated. Its few-image reconstruction setting differs from the micro-CT mesh input used here. |
| [HowardHH-SoyRSA/3D_model_Gm](https://github.com/HowardHH-SoyRSA/3D_model_Gm) | Reference for soybean 3D model handling and expected research workflow. | No explicit software license was established during review; no code copied. |
| [STL-PLY-batch-converter](https://github.com/HowardHH-SoyRSA/STL-PLY-batch-converter) | Optional reference for format conversion and batch UX. | Not incorporated. Verify its repository license before any future source reuse. SoyRootBio uses Open3D directly for supported input/output. |
| [Lucas et al. (2019), *Roots compact the surrounding soil depending on the structures they encounter*](https://doi.org/10.1038/s41598-019-52665-w) | Biological/CT context supplied for review, not a software or skeletonization source. | Scientific reference only; no code or content incorporated. |

## Additional license boundary

Some related tools advertise that they are “open source” or make a repository publicly visible without including a detectable license. Public visibility alone does not grant permission to copy, modify, or redistribute source code. Those projects are therefore listed only as scientific/design references unless a compatible license is established later.

Likewise, a Creative Commons license on an article normally governs the article, not a separately distributed software repository. GPL software can be run externally for comparison or validation, but its source was not combined with this MIT-licensed implementation.

## Runtime dependencies

The application declares Open3D, NumPy, SciPy, HDBSCAN, scikit-learn, NetworkX, pandas, Matplotlib, openpyxl, psutil, and tkinterdnd2 as install-time dependencies. They are not vendored in this repository. Consult each installed distribution's metadata and license when redistributing a packaged application.

This notice lists research and design references; it is not a claim that every linked project, paper, or remote branch is required at runtime. The authoritative dependency list for this source tree is the `[project.dependencies]` table in `pyproject.toml`.
