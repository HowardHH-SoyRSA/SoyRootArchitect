"""Desktop entry point for this checkout's required-CUDA build."""
from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["PYTHONPATH"] = str(ROOT / "src")
os.environ["SOYROOTBIO_BACKEND"] = "cuda"
os.environ["CUPY_CACHE_DIR"] = str(ROOT / ".cupy-cache")


def main():
    from multiprocessing import freeze_support
    freeze_support()
    from soyrootbio.gpu_backend import compute_backend
    try:
        with compute_backend("cuda"):
            pass
    except Exception as exc:
        import tkinter as tk
        from tkinter import messagebox
        window = tk.Tk()
        window.withdraw()
        messagebox.showerror("SoyRootArchitect GPU cannot start", str(exc), parent=window)
        window.destroy()
        return 1
    from soyrootbio.desktop_gui import launch_gui
    return launch_gui(initial_output=ROOT / "outputs" / "gpu_gui")


if __name__ == "__main__":
    raise SystemExit(main())
