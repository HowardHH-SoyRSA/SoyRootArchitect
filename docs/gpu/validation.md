# GPU build validation

Validated on this machine on 2026-10-04. The dependency review and benchmark
comparison were performed by GPT-6.1 Sol with Extra High reasoning.

- Full regression suite, including opt-in real CUDA tests: **570 passed in
  58.47 seconds**. Command: `.venv-gpu\Scripts\python.exe -m pytest -q tests
  --basetemp .pytest-runs/gpu-final`, with `SOYROOTBIO_TEST_CUDA=1` and the
  checkout's `src` and `.cupy-cache` paths selected.
- `.venv-gpu\Scripts\python.exe -m pip check`: no broken requirements.
- [Complete sample comparison](benchmark_comparison.md): six CPU and six CUDA
  runs completed; 180 scientific exports and 24 decoded PNGs matched exactly.
  Each CUDA run recorded actual kernel execution. Total: 6,984 launches and
  263,492,169 sparse point/segment pairs.
- The [warm projection check](benchmark_warm_projection.md) and
  [additional CPU replay](benchmark_baxi_repeat.md) retained exact outputs.
  Whole-pipeline timing showed no demonstrated CUDA gain; the CPU replay
  exposed substantial timing drift.

The desktop smoke check ran the GPU launcher's CUDA startup check, constructed
the real Tk batch window, and verified its `GPU / CUDA` title, required-CUDA
hardware text, CUDA pipeline configuration and separate `outputs/gpu_gui`
default. The temporary window was closed after verification. This passed
under the user's Windows profile; the restricted sandbox could not initialize
Tcl/Tk. No launcher change was needed. The result is recorded in
[desktop_verification.json](desktop_verification.json).

The desktop shortcut was created and read back through Windows Shell under
the user's profile:

- Shortcut: `C:\Users\57340\OneDrive\桌面\SoyRootArchitect GPU.lnk`.
- Target: `C:\Users\57340\.codex\worktrees\0be1\SoyRSA Build\.venv-gpu\Scripts\pythonw.exe`.
- Argument: `"C:\Users\57340\.codex\worktrees\0be1\SoyRSA Build\scripts\launch_gpu.py"`.
- Working directory: `C:\Users\57340\.codex\worktrees\0be1\SoyRSA Build`.

The target exists. The original CPU launcher and environment were preserved.
All 48 Python source files in `E:\SoyRSA Build\src` still matched the frozen
baseline hashes when checked after the CUDA runs began. The isolated runtime's
editable source mapping resolves only to this GPU checkout; the original CPU
runtime still resolves to `E:\SoyRSA Build\src`.

The baseline deliberately includes the user's existing uncommitted CPU
centerline/topology changes. [Source provenance](cpu_source_provenance.json)
and [the baseline patch](cpu_baseline.patch) capture those changes separately
from CUDA implementation. The worktree is on `GPU-version`, tracks
`origin/GPU-version`, and has a worktree-only push destination of
`HEAD:refs/heads/GPU-version`. `AGENTS.md` retains this publication rule.

Exact output parity does not establish biological compliance. Unresolved
child patches remain in all six samples, and SN14 retains 244 native
higher-order/primary contact edges, reported as `unresolved_contacts`.
