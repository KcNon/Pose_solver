#!/usr/bin/env python3
"""Check whether an Isaac Sim release can be launched by the current user."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except PermissionError:
        return False


def safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except PermissionError:
        return False


def first_inaccessible_component(path: Path) -> str | None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_dir() and not os.access(current, os.X_OK):
            return str(current)
        if current.exists() and not os.access(current, os.R_OK):
            return str(current)
    return None


def command_result(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as error:
        return {"returncode": None, "error": repr(error)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-sim-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    release = args.isaac_sim_dir.absolute()
    launcher = release / "python.sh"
    kit_link = release / "kit"
    kit_target = Path(os.path.realpath(kit_link))
    python_binary = kit_target / "python/bin/python3"
    inaccessible = first_inaccessible_component(python_binary)
    gpu = command_result(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    )
    checks = {
        "release_exists": safe_is_dir(release),
        "launcher_exists": safe_is_file(launcher),
        "launcher_executable": os.access(launcher, os.X_OK),
        "kit_is_symlink": kit_link.is_symlink(),
        "kit_target": str(kit_target),
        "kit_target_accessible": inaccessible is None,
        "first_inaccessible_component": inaccessible,
        "python_binary_exists": safe_is_file(python_binary),
        "python_binary_executable": os.access(python_binary, os.X_OK),
        "gpu_query": gpu,
    }
    blocking_reasons = []
    if not checks["launcher_executable"]:
        blocking_reasons.append("Isaac python.sh is missing or not executable")
    if inaccessible is not None:
        blocking_reasons.append(f"Current user cannot traverse/read Isaac's resolved Kit dependency: {inaccessible}")
    if not checks["python_binary_executable"]:
        blocking_reasons.append("Resolved Kit Python binary is not executable by the current user")
    if gpu.get("returncode") != 0:
        blocking_reasons.append("nvidia-smi cannot communicate with an NVIDIA driver in this execution environment")
    report = {
        "schema_version": 1,
        "status": "ready" if not blocking_reasons else "blocked",
        "uid": os.getuid(),
        "gid": os.getgid(),
        "checks": checks,
        "blocking_reasons": blocking_reasons,
        "recommended_action": (
            "Run Isaac as the build owner, or grant the execution user traverse/read access to the resolved Packman "
            "cache. Do not grant global write access. Ensure the NVIDIA device and driver are visible."
            if blocking_reasons
            else "Runtime preflight passed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "ready" else 2)


if __name__ == "__main__":
    main()
