#!/usr/bin/env python
"""Project interpolated assembly poses back onto configured physical axes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.pose_config import validate_pose_config
from common.pose_validation import validate_trajectory
from common.render_loss_refinement import world_pose_delta_vector
from common.trajectory_constraints import (
    pairwise_alignment_metrics,
    project_coaxial_pose,
)
from common.trajectory_io import (
    refresh_trajectory_derived_fields,
    write_trajectory_files,
)


def canonical_axis_origin(
    settings: dict[str, Any],
    role: str,
    part: str,
    trajectory: dict[str, Any],
) -> np.ndarray:
    metric_key = f"{role}_axis_origin_part_m"
    if metric_key in settings:
        return np.asarray(settings[metric_key], dtype=np.float64)
    raw_key = f"{role}_axis_origin_raw"
    if raw_key not in settings:
        return np.zeros(3, dtype=np.float64)
    return float(trajectory["scales"][part]) * (
        np.asarray(settings[raw_key], dtype=np.float64)
        - np.asarray(trajectory["raw_mesh_origins"][part], dtype=np.float64)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output-trajectory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--parts", nargs="+", default=None)
    args = parser.parse_args()

    cfg = validate_pose_config(load_json(args.config), check_paths=True)
    trajectory = load_json(args.trajectory)
    part_settings = cfg.get("render_loss_refinement", {}).get("parts", {})
    requested = set(args.parts or part_settings)
    report: dict[str, Any] = {
        "method": "physical_axis_projection_after_temporal_interpolation",
        "input": str(args.trajectory.resolve()),
        "parts": {},
    }
    for part, settings in part_settings.items():
        coaxial = dict(settings.get("coaxial_constraint", {}))
        if part not in requested or not coaxial.get("enabled", False):
            continue
        reference_part = str(coaxial["reference_part"])
        reference_origin = canonical_axis_origin(
            coaxial, "reference", reference_part, trajectory
        )
        moving_origin = canonical_axis_origin(
            coaxial, "moving", part, trajectory
        )
        maximum_translation = float(
            coaxial.get("maximum_final_projection_translation_m", 0.02)
        )
        maximum_rotation = float(
            coaxial.get("maximum_final_projection_rotation_deg", 6.0)
        )
        rows = {}
        for start, end in coaxial.get("ranges", []):
            for frame in range(int(start), int(end) + 1):
                key = f"{frame:06d}"
                if key not in trajectory["frames"]:
                    continue
                records = trajectory["frames"][key]["parts"]
                reference_world = np.asarray(
                    records[reference_part]["T_world_from_part"],
                    dtype=np.float64,
                )
                moving_world = np.asarray(
                    records[part]["T_world_from_part"], dtype=np.float64
                )
                relative = np.linalg.inv(reference_world) @ moving_world
                before = pairwise_alignment_metrics(
                    relative,
                    reference_axis=coaxial["reference_axis_part"],
                    moving_axis=coaxial["moving_axis_part"],
                    allow_axis_flip=bool(
                        coaxial.get("allow_axis_flip", False)
                    ),
                    reference_axis_origin_m=reference_origin,
                    moving_axis_origin_m=moving_origin,
                )
                projected = project_coaxial_pose(
                    relative,
                    reference_axis=coaxial["reference_axis_part"],
                    moving_axis=coaxial["moving_axis_part"],
                    reference_axis_origin_m=reference_origin,
                    moving_axis_origin_m=moving_origin,
                    allow_axis_flip=bool(
                        coaxial.get("allow_axis_flip", False)
                    ),
                    target_axis_offset_m=float(
                        coaxial.get("target_axis_offset_m", 0.0)
                    ),
                )
                selected_world = reference_world @ projected
                delta = world_pose_delta_vector(moving_world, selected_world)
                translation = float(np.linalg.norm(delta[:3]))
                rotation = float(np.degrees(np.linalg.norm(delta[3:])))
                if (
                    translation > maximum_translation + 1e-9
                    or rotation > maximum_rotation + 1e-9
                ):
                    raise RuntimeError(
                        f"{part} {key}: final coaxial projection exceeds "
                        f"bounds ({1000.0 * translation:.2f}mm, "
                        f"{rotation:.2f}deg)"
                    )
                records[part]["T_world_from_part"] = selected_world.tolist()
                records[part]["source"] = (
                    str(records[part].get("source", "pose"))
                    + "+final_coaxial_projection"
                )
                after = pairwise_alignment_metrics(
                    projected,
                    reference_axis=coaxial["reference_axis_part"],
                    moving_axis=coaxial["moving_axis_part"],
                    allow_axis_flip=bool(
                        coaxial.get("allow_axis_flip", False)
                    ),
                    reference_axis_origin_m=reference_origin,
                    moving_axis_origin_m=moving_origin,
                )
                rows[key] = {
                    "before": before,
                    "after": after,
                    "translation_delta_m": translation,
                    "rotation_delta_deg": rotation,
                }
        report["parts"][part] = {
            "reference_part": reference_part,
            "frames": rows,
            "summary": {
                "projected_frames": len(rows),
                "maximum_translation_delta_m": max(
                    (row["translation_delta_m"] for row in rows.values()),
                    default=0.0,
                ),
                "maximum_rotation_delta_deg": max(
                    (row["rotation_delta_deg"] for row in rows.values()),
                    default=0.0,
                ),
                "maximum_before_axis_angle_deg": max(
                    (row["before"]["axis_angle_deg"] for row in rows.values()),
                    default=0.0,
                ),
                "maximum_before_axis_offset_m": max(
                    (row["before"]["axis_offset_m"] for row in rows.values()),
                    default=0.0,
                ),
                "maximum_after_axis_angle_deg": max(
                    (row["after"]["axis_angle_deg"] for row in rows.values()),
                    default=0.0,
                ),
                "maximum_after_axis_offset_m": max(
                    (row["after"]["axis_offset_m"] for row in rows.values()),
                    default=0.0,
                ),
            },
        }

    refresh_trajectory_derived_fields(trajectory)
    validation, failures = validate_trajectory(
        cfg, trajectory, enforce_assembly=False
    )
    report["trajectory_validation"] = validation
    report["validation_passed"] = not failures
    if failures:
        write_json(args.report, report)
        raise RuntimeError("; ".join(failures))
    trajectory.setdefault("refinements", []).append(
        {
            "method": report["method"],
            "input": str(args.trajectory.resolve()),
            "report": str(args.report.resolve()),
        }
    )
    write_trajectory_files(trajectory, args.output_trajectory)
    write_json(args.report, report)
    print(f"trajectory -> {args.output_trajectory}", flush=True)
    print(f"report -> {args.report}", flush=True)


if __name__ == "__main__":
    main()
