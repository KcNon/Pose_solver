#!/usr/bin/env python
"""Merge accepted per-part static-pose diagnostics into one trajectory."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.pose_config import validate_pose_config
from common.pose_validation import validate_trajectory
from common.static_pose_refinement import merge_static_pose_refinements
from common.trajectory_io import write_trajectory_files


def parse_refinement(value: str) -> tuple[str, Path]:
    part, separator, raw_path = value.partition("=")
    if not separator or not part or not raw_path:
        raise argparse.ArgumentTypeError("expected PART=TRAJECTORY.json")
    return part, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--refined", required=True, action="append", type=parse_refinement)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--propagate", action="store_true")
    args = parser.parse_args()

    cfg = validate_pose_config(load_json(args.config), check_paths=True)
    refined = {}
    for part, path in args.refined:
        if part in refined:
            raise ValueError(f"duplicate refinement for {part}")
        refined[part] = load_json(path)
    trajectory, report = merge_static_pose_refinements(
        cfg,
        load_json(args.base),
        refined,
        frame=args.frame,
        propagate=args.propagate,
    )
    validation, failures = validate_trajectory(
        cfg, trajectory, enforce_assembly=False
    )
    report["trajectory_validation"] = validation
    report["validation_passed"] = not failures
    report["failures"] = failures
    write_json(args.report, report)
    if failures:
        raise RuntimeError("; ".join(failures))
    write_trajectory_files(trajectory, args.output)
    print(f"trajectory -> {args.output}")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
