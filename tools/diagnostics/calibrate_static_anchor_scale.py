#!/usr/bin/env python
"""Calibrate one part's uniform scale at one already-refined static anchor.

This diagnostic deliberately freezes the rigid pose.  It is intended to run
after a clean multi-view static-anchor pose has been accepted, so scale is not
allowed to compensate for a bad rotation or translation.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.pose_config import validate_pose_config
from common.pose_transforms import similarity_from_rigid
from common.silhouette_scale_calibration import calibrate_part_scale
from common.trajectory_io import refresh_trajectory_derived_fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output-trajectory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--part", required=True)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument(
        "--scale-factors",
        nargs="+",
        type=float,
        default=[0.94, 0.96, 0.98, 0.99, 1.0, 1.01, 1.02, 1.04, 1.06],
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    args = parser.parse_args()

    cfg = validate_pose_config(load_json(args.config.resolve()), check_paths=True)
    trajectory = load_json(args.trajectory.resolve())
    part = str(args.part)
    timestamp = f"{int(args.frame):06d}"
    if part not in trajectory["parts"]:
        raise ValueError(f"unknown trajectory part: {part}")
    if timestamp not in trajectory["frames"]:
        raise ValueError(f"trajectory does not contain frame {timestamp}")

    base_scale = float(trajectory["scales"][part])
    raw_origin = np.asarray(
        trajectory["raw_mesh_origins"][part], dtype=np.float64
    )
    pose = np.asarray(
        trajectory["frames"][timestamp]["parts"][part]["T_world_from_part"],
        dtype=np.float64,
    )
    anchor = similarity_from_rigid(pose, base_scale, raw_origin)
    mesh = trimesh.load(
        Path(cfg["mesh_dir"]) / f"{part}.glb", force="mesh"
    )

    settings = copy.deepcopy(
        cfg.get("automation", {}).get("silhouette_scale_calibration", {})
    )
    render = cfg.get("render_loss_refinement", {})
    settings.update({
        "enabled": True,
        "scale_factors": sorted(set(float(value) for value in args.scale_factors)),
        "maximum_anchor_frames": 1,
        "anchor_frames": [int(args.frame)],
        "first_static_interval_only": False,
        "exact_mesh_render": True,
        "resolution": [int(args.width), int(args.height)],
        "optimize_views": list(render.get("optimize_views", cfg["views"])),
        "holdout_views": list(render.get("holdout_views", [])),
        "minimum_optimize_views": 3,
        "minimum_full_mask_pixels": 100,
        "min_mask_pixels": 20,
        "use_cloud_supported_view_gate": False,
        "occlusion_aware": False,
        "minimum_improvement": 0.0,
        "maximum_holdout_degradation": 0.02,
        "maximum_area_log_degradation": 0.01,
        "visual_loss_tie_tolerance": 0.01,
        "configured_scale_prior": False,
    })
    weights = dict(settings.get("weights", {}))
    weights["depth"] = 0.0
    settings["weights"] = weights

    selected_scale, _anchors, report = calibrate_part_scale(
        cfg=cfg,
        part=part,
        mesh=mesh,
        raw_origin=raw_origin,
        base_scale=base_scale,
        anchors={int(args.frame): anchor},
        settings=settings,
        seed=13 + int(args.frame),
    )
    output = copy.deepcopy(trajectory)
    output["scales"][part] = float(selected_scale)
    refresh_trajectory_derived_fields(output)
    report.update({
        "scope": "single_static_anchor_frozen_pose_scale",
        "config": str(args.config.resolve()),
        "trajectory_input": str(args.trajectory.resolve()),
        "trajectory_output": str(args.output_trajectory.resolve()),
        "part": part,
        "frame": int(args.frame),
    })
    write_json(args.output_trajectory.resolve(), output)
    write_json(args.report.resolve(), report)
    print(
        f"{part} {timestamp}: scale {base_scale:.9f} -> "
        f"{selected_scale:.9f} accepted={report['accepted']} "
        f"IoU={report['candidates'][report['selected_index']]['optimize_iou']:.4f}",
        flush=True,
    )
    print(f"trajectory -> {args.output_trajectory.resolve()}", flush=True)
    print(f"report -> {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    from common.resource_safety import require_memory_guard

    require_memory_guard("tools/diagnostics/calibrate_static_anchor_scale.py")
    main()
