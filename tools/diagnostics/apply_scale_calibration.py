#!/usr/bin/env python3
"""Create a derived trajectory with accepted uniform scales and frozen poses."""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.trajectory_io import (
    refresh_trajectory_derived_fields,
    write_trajectory_files,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preferred_factors(report: dict[str, Any]) -> dict[str, float]:
    factors: dict[str, float] = {}
    for name, relation in report["relations"].items():
        part = str(relation["moving_part"])
        factor = relation["joint_acceptance"].get(
            "preferred_scale_factor"
        )
        if factor is None:
            raise ValueError(f"{name}: no jointly accepted scale candidate")
        factor = float(factor)
        if part in factors and not np.isclose(factors[part], factor):
            raise ValueError(
                f"conflicting accepted scale factors for {part}: "
                f"{factors[part]} and {factor}"
            )
        factors[part] = factor
    return factors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--isaac-report", required=True, type=Path)
    parser.add_argument("--output-trajectory", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    input_path = args.trajectory.resolve()
    isaac_path = args.isaac_report.resolve()
    output_path = args.output_trajectory.resolve()
    report_path = (
        args.report.resolve()
        if args.report
        else output_path.parents[1]
        / "diagnostics"
        / "scale_calibration_application.json"
    )
    baseline = load_json(input_path)
    isaac_report = load_json(isaac_path)
    trajectory = copy.deepcopy(baseline)
    factors = preferred_factors(isaac_report)
    changes = {}
    for part, factor in factors.items():
        old_scale = float(baseline["scales"][part])
        new_scale = old_scale * factor
        trajectory["scales"][part] = new_scale
        changes[part] = {
            "scale_factor": factor,
            "old_absolute_scale": old_scale,
            "new_absolute_scale": new_scale,
        }
    trajectory["scale_calibration"] = {
        "method": "joint_visual_geometry_isaac_frozen_pose_scale",
        "source_trajectory": str(input_path),
        "isaac_report": str(isaac_path),
        "pose_changed": False,
        "changes": changes,
    }
    refresh_trajectory_derived_fields(
        trajectory, recompute_similarity=True
    )

    maximum_pose_error = 0.0
    for key, frame in baseline["frames"].items():
        for part in baseline["parts"]:
            before = np.asarray(
                frame["parts"][part]["T_world_from_part"],
                dtype=np.float64,
            )
            after = np.asarray(
                trajectory["frames"][key]["parts"][part][
                    "T_world_from_part"
                ],
                dtype=np.float64,
            )
            maximum_pose_error = max(
                maximum_pose_error,
                float(np.max(np.abs(before - after))),
            )
    if maximum_pose_error != 0.0:
        raise AssertionError(
            f"scale calibration changed pose by {maximum_pose_error}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_trajectory_files(trajectory, output_path)
    report = {
        "schema_version": 1,
        "method": "apply_jointly_accepted_uniform_scale_only",
        "source_trajectory": str(input_path),
        "source_trajectory_sha256": sha256_file(input_path),
        "isaac_report": str(isaac_path),
        "isaac_report_sha256": sha256_file(isaac_path),
        "output_trajectory": str(output_path),
        "output_trajectory_sha256": sha256_file(output_path),
        "changes": changes,
        "maximum_abs_pose_matrix_change": maximum_pose_error,
        "source_trajectory_overwritten": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print(f"Scale-calibrated trajectory: {output_path}", flush=True)
    print(f"Application report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
