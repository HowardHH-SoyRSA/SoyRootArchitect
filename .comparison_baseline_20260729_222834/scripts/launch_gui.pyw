"""Launch the candidate GUI directly from its isolated source snapshot."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_DIRECTORY / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from soyrootbio.desktop_gui import launch_gui  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(launch_gui())
