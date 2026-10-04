"""Record the frozen CPU source delta and installed package versions."""
from pathlib import Path
import difflib
import importlib.metadata
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/gpu_benchmark/baseline/src"
DEST = ROOT / "docs/gpu"


def main():
    base_commit = "e149ba1fd95275ddfa3520b2fe772d220983b826"
    chunks, changed = [], []
    for path in sorted(BASE.rglob("*.py")):
        relative = "src/" + path.relative_to(BASE).as_posix()
        original = subprocess.run(["git", "show", f"{base_commit}:{relative}"], cwd=ROOT,
                                  capture_output=True, check=False)
        before = original.stdout.decode("utf-8").splitlines(keepends=True) if original.returncode == 0 else []
        after = path.read_text(encoding="utf-8").splitlines(keepends=True)
        diff = list(difflib.unified_diff(before, after,
                    fromfile="a/" + relative if original.returncode == 0 else "/dev/null",
                    tofile="b/" + relative))
        if diff:
            changed.append(relative)
            chunks.extend(diff)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "cpu_baseline.patch").write_text("".join(chunks), encoding="utf-8")
    packages = sorted(f"{dist.metadata['Name']}=={dist.version}"
                      for dist in importlib.metadata.distributions()
                      if dist.metadata['Name'].lower() != "soybean-root-bio")
    (DEST / "validated_environment.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    manifest = json.loads((ROOT / "outputs/gpu_benchmark/frozen_manifest.json").read_text(encoding="utf-8"))
    provenance = {"cpu_base_commit": base_commit, "cpu_baseline_patch": "cpu_baseline.patch",
                  "cpu_changed_python_files": changed,
                  "cpu_source_hashes": manifest["cpu_source_hashes"],
                  "benchmark_manifest": "benchmark_manifest.json"}
    (DEST / "cpu_source_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline_files": changed, "package_count": len(packages)}))


if __name__ == "__main__":
    main()
