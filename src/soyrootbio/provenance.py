"""Content-addressed provenance for reproducible biological measurements."""
from __future__ import annotations

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def processing_provenance(source: str | Path | None = None, config: dict | None = None) -> dict:
    """Hash actual installed source, including uncommitted and untracked modules.

    A Git revision is supplementary: it cannot identify a dirty working tree.
    No environment variables, credentials, or repository remotes are recorded.
    """
    package = Path(__file__).resolve().parent
    files = {str(p.relative_to(package)).replace("\\", "/"): file_sha256(p)
             for p in sorted(package.rglob("*.py"))}
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    versions = {}
    for name in ("numpy", "scipy", "pandas", "open3d", "openpyxl", "hdbscan", "scikit-learn", "networkx"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unavailable"
    result = {
        "schema": "soyrootbio.processing-provenance/v1",
        "source_tree_sha256": sha256(canonical).hexdigest(),
        "source_files_sha256": files,
        "python_version": platform.python_version(),
        "dependency_versions": versions,
        "measurement_contract": "native-supported-intervals-v2",
        "interval_support_version": 2,
        "terminal_direction_support_version": 1,
        "origin_coverage_policy": "immutable-origin-axial-coverage-v1",
    }
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package,
                                  capture_output=True, text=True, timeout=5, check=False)
        if revision.returncode == 0:
            result["git_commit"] = revision.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    if source is not None:
        path = Path(source)
        result["input_path"] = str(path)
        if path.is_file():
            result["input_sha256"] = file_sha256(path)
            result["input_size_bytes"] = path.stat().st_size
        else:
            result["input_hash_status"] = "source_unavailable"
    if config is not None:
        payload = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
        result["config_sha256"] = sha256(payload.encode()).hexdigest()
    return result
