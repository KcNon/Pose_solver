"""Fail-closed execution policy for memory-intensive command-line tools."""
from __future__ import annotations

import os


MEMORY_GUARD_ACTIVE = "POSE_SOLVER_MEMORY_GUARD_ACTIVE"


def require_memory_guard(tool: str) -> None:
    """Reject a heavy standalone CLI that is not inside the memory guard."""

    if os.environ.get(MEMORY_GUARD_ACTIVE) == "1":
        return
    raise SystemExit(
        f"{tool} is a memory-intensive internal command and cannot run "
        "unguarded. Use `python -m pose_solver run ...` or invoke it through "
        "`tools/diagnostics/run_with_memory_guard.py -- <command>`."
    )
