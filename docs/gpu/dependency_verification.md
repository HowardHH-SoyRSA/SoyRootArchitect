# GPU dependency verification

Verified on 2026-10-04 by GPT-6.1 Sol, Extra High. Reviewed both available turns of [Install GPU Python libraries](codex://threads/01a0fd65-6c4f-75d1-a799-72a50174a8cc), including the completed installation turn, and repeated real computations. This updates the installation status in section 5 of `E:\SoyRSA Build\performance_audit_20261002\astra_assessment.md`.

The requested GPU libraries **are successfully installed and usable**: CuPy/PyTorch/Numba on native Windows and all five libraries in Ubuntu WSL2. The existing CPU application's environment has none of these GPU modules installed.

| Check | Native Windows | Ubuntu-24.04 WSL2 |
|---|---|---|
| CuPy | 14.2.0; GPU sum = 45 | 14.2.0; GPU sum = 45 |
| PyTorch | 2.13.0+cu130; CUDA GPU sum = 45 | 2.14.1+cu132; CUDA GPU sum = 45 |
| Numba / numba-cuda | 0.68.0 / 0.30.4; CUDA add-one kernel passed | 0.64.0 / 0.30.4; CUDA add-one kernel passed |
| cuML | Absent in native environment | 26.08.00; GPU KMeans found two clusters |
| cuGraph | Absent in native environment | 26.08.00; GPU PageRank returned three vertices |
| NumPy | 2.4.6 | 2.4.6 |
| pip check | No broken requirements found | No broken requirements found |

Native Windows passed both CuPy-first and PyTorch-first import orders; this WSL rerun used PyTorch-first. The device is NVIDIA GeForce RTX 5090, compute capability 12.0, driver 616.56, 32,607 MiB reported by `nvidia-smi`. These are smoke checks of actual device calculations, not a SoyRootArchitect pipeline benchmark.

## Runtime locations

- Native Windows: `C:\Users\57340\Documents\Codex\2026-10-03\cupy-pytorch-numba-cuml-and-cugraph\work\gpu-windows\Scripts\python.exe` (Python 3.12.12).
- WSL2: `/home/gpu/gpu-stack/bin/python`, distribution `Ubuntu-24.04`, user `gpu`.
- Existing CPU application: `E:\SoyRSA Build\.venv\Scripts\python.exe` (Python 3.12.14).
- Original smoke verifier: `C:\Users\57340\Documents\Codex\2026-10-03\cupy-pytorch-numba-cuml-and-cugraph\outputs\verify_gpu.py`.

The original native GPU stack environment lacks application dependencies including Open3D, SciPy, scikit-learn, hdbscan, pandas, matplotlib, openpyxl, psutil, threadpoolctl, Flask and tkinterdnd2. GPU package readiness does not imply that original stack environment can already run the desktop application. Preserve the CPU runtime and create a separate application runtime.

## Proposed sparse distance backend readiness

A fresh native-Windows CuPy RawKernel projected **4,096 sparse point/segment pairs in float64**, including 16 zero-length segments. With `--fmad=false` and no fast math, the output matched its NumPy reference exactly in this smoke fixture (maximum absolute error 0.0). This verifies the section-5 prototype's CUDA arithmetic capability, not whole-bundle segmentation parity.

The CPU runtime's **NumPy 2.5.1 and SciPy 1.18.0 are compatible with CuPy 14.2.0** for the checked operations: a 10,000-value float64 GPU sum and a fresh RawKernel square passed exactly. That test loaded CPU NumPy first and temporarily exposed the existing GPU wheels plus their DLL/header search locations. No packages were installed into the CPU environment.

For a cloned CPU application runtime, the small initial package set to validate is `cupy-cuda13x==14.2.0`, `cuda-pathfinder==1.8.3`, `nvidia-cuda-runtime==13.4.92` and `nvidia-cuda-nvrtc==13.4.92`. CuPy requires `numpy>=2.0,<2.6`; the existing NumPy satisfies it. `cuda-toolkit[cudart,nvrtc]==13.4.2` selects those CUDA runtime/compiler wheels. A normal install into the isolated interpreter should allow CUDA Pathfinder to discover the unified Windows wheel layout (`nvidia/cu13/bin/x86_64` for DLLs and `nvidia/cu13/include` for headers). Set `CUPY_CACHE_DIR` to a GPU-owned writable location. A bare addition of a foreign wheel directory to `sys.path` failed header/DLL discovery during the compatibility probe.

PyTorch, Numba, RAPIDS and NVVM are unnecessary for this selected CuPy RawKernel boundary. The original combined Windows stack keeps NumPy below 2.5 for its Numba dependency and pins `nvidia-nvvm==13.0.88` for PyTorch compatibility; those constraints need not be imported into a CuPy-only application runtime. Installing CuPy's `[ctk]` extra additionally brings BLAS/FFT/RNG/solver/sparse libraries that the custom sparse distance kernel does not inherently need.

All installed environments were left unchanged by this dependency review. The initial sandbox WSL service denial was resolved using approved read-only elevated verification. Compiler caches were written only to temporary directories. Machine-readable smoke outputs and package evidence are in [dependency_verification.json](dependency_verification.json).

## Isolated application runtime inspection

After the implementation agent created `.venv-gpu`, a read-only inspection confirmed its sole SoyRootArchitect editable `.pth` points to `C:\Users\57340\.codex\worktrees\0be1\SoyRSA Build\src`; its `direct_url.json` also names that checkout. The only other `.pth` is setuptools' `distutils-precedence.pth`. The original CPU environment's editable `.pth` still points to `E:\SoyRSA Build\src`. The isolated interpreter has `include-system-site-packages = false`, Python 3.12.14, and metadata for CuPy 14.2.0, CUDA Pathfinder 1.8.3, CUDA Toolkit 13.4.2, CUDA runtime 13.4.92 and NVRTC 13.4.92. Package consistency and backend CUDA tests for this new runtime are reported by the implementation agent; this additional inspection did not launch numerical work during the benchmark.
