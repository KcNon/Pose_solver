#!/usr/bin/env python3
"""Compare full-nozzle and rigid-core pose refinement on bounded keyframes."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.resource_safety import require_memory_guard
from common.trajectory_constraints import interpolate_pose
from common.trajectory_io import (
    refresh_trajectory_derived_fields,
    write_trajectory_files,
)


MAX_FRAMES = 8
PART = "nozzle"


def _variant_config(
    base: dict[str, Any],
    *,
    masks_dir: Path,
    mesh_dir: Path,
    evaluate_only: bool,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["masks_dir"] = str(masks_dir.resolve())
    cfg["mesh_dir"] = str(mesh_dir.resolve())
    views = [str(view) for view in cfg["views"]]
    holdout_count = 2 if len(views) >= 6 else 1
    optimize_views = views if evaluate_only else views[:-holdout_count]
    holdout_views = [] if evaluate_only else views[-holdout_count:]
    cfg["render_loss_refinement"] = {
        "enabled": True,
        "resolution": [160, 90],
        "surface_points": 12000,
        "optimize_views": optimize_views,
        "holdout_views": holdout_views,
        "minimum_optimize_views": min(3, len(optimize_views)),
        "minimum_holdout_views": 0 if evaluate_only else 1,
        "mask_primary": True,
        "use_cloud_supported_view_gate": False,
        "use_depth_loss": False,
        "disable_depth_for_mask_only_fallback": True,
        "occlusion_aware": False,
        "known_part_occlusion_aware": True,
        "known_occluder_labels": sorted(
            {int(value) for value in cfg["part_ids"].values()} | {3}
        ),
        "minimum_full_mask_pixels": 20,
        "presence_minimum_full_mask_pixels": 20,
        "minimum_mask_area_ratio": 0.03,
        "maximum_mask_area_ratio": 8.0,
        "translation_steps_m": [] if evaluate_only else [0.012, 0.006, 0.003],
        "rotation_steps_deg": [] if evaluate_only else [6.0, 3.0, 1.5],
        "maximum_translation_delta_m": 0.04,
        "maximum_rotation_delta_deg": 15.0,
        "minimum_improvement": 0.0 if evaluate_only else 0.001,
        "maximum_holdout_degradation": 0.03,
        "minimum_refined_iou": 0.0,
        "minimum_holdout_iou": 0.0,
        "minimum_per_view_iou": 0.0,
        "prior_weight": 0.005,
        "temporal_weight": 0.0,
        "trim_worst_views": 0,
        "exact_triangle_refinement": False,
        "global_reacquire_enabled": not evaluate_only,
        "global_reacquire_iou_threshold": 0.30,
        "global_reacquire_translation_radii_m": [0.12, 0.08, 0.04, 0.02],
        "global_reacquire_rotation_angles_deg": [45.0, 30.0, 15.0],
        "global_reacquire_alternating_passes": 1,
        "global_reacquire_minimum_improvement": 0.01,
        "weights": {
            "iou": 1.0,
            "contour": 0.25,
            "target_coverage": 0.20,
            "depth": 0.0,
        },
        "parts": {
            PART: {
                "enabled": True,
                "protect_tracking_anchors": False,
            }
        },
    }
    cfg["states"] = {
        name: {
            **dict(value),
            "validation": {
                **dict(value.get("validation", {})),
                "max_translation_step_m": 1.0e6,
                "max_rotation_step_deg": 360.0,
                "fail_on_violation": False,
            },
        }
        for name, value in cfg["states"].items()
    }
    return cfg


def _bridge_seed(
    trajectory: dict[str, Any],
    bridges: list[tuple[int, int, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = copy.deepcopy(trajectory)
    report = []
    for frame, left, right in bridges:
        if not left < frame < right:
            raise ValueError(
                f"bridge {frame}:{left}:{right} must satisfy left < frame < right"
            )
        keys = [f"{value:06d}" for value in (frame, left, right)]
        missing = [key for key in keys if key not in result["frames"]]
        if missing:
            raise ValueError(f"bridge is missing trajectory frames {missing}")
        before = np.asarray(
            result["frames"][keys[0]]["parts"][PART]["T_world_from_part"],
            dtype=np.float64,
        )
        start = np.asarray(
            result["frames"][keys[1]]["parts"][PART]["T_world_from_part"],
            dtype=np.float64,
        )
        end = np.asarray(
            result["frames"][keys[2]]["parts"][PART]["T_world_from_part"],
            dtype=np.float64,
        )
        amount = float((frame - left) / (right - left))
        seeded = interpolate_pose(start, end, amount)
        record = result["frames"][keys[0]]["parts"][PART]
        record["T_world_from_part"] = seeded.tolist()
        record["source"] = (
            str(record.get("source", "pose")) + "+temporal_bridge_seed"
        )
        delta = np.linalg.inv(before) @ seeded
        report.append({
            "frame": frame,
            "left_frame": left,
            "right_frame": right,
            "interpolation": amount,
            "translation_delta_m": float(np.linalg.norm(delta[:3, 3])),
            "rotation_delta_deg": float(np.degrees(
                Rotation.from_matrix(delta[:3, :3]).magnitude()
            )),
        })
    refresh_trajectory_derived_fields(result)
    return result, report


def _run_refinement(
    config: Path,
    trajectory: Path,
    output_trajectory: Path,
    report: Path,
    frames: list[int],
) -> None:
    command = [
        sys.executable,
        "-u",
        str(ROOT / "tools" / "stages" / "pose" / "refine_pose_render_loss.py"),
        "--config",
        str(config),
        "--trajectory",
        str(trajectory),
        "--output-trajectory",
        str(output_trajectory),
        "--report",
        str(report),
        "--parts",
        PART,
        "--frames",
        *[str(frame) for frame in frames],
    ]
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _metric_rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return dict(report.get("parts", {}).get(PART, {}).get("frames", {}))


def _mean_iou(report: dict[str, Any], key: str = "baseline_optimize") -> float | None:
    values = [
        float(row[key]["mean_iou"])
        for row in _metric_rows(report).values()
        if row.get(key, {}).get("mean_iou") is not None
    ]
    return float(np.mean(values)) if values else None


def _accepted_frames(report: dict[str, Any]) -> int:
    return sum(bool(row.get("accepted", False)) for row in _metric_rows(report).values())


def _pose_delta(
    first: dict[str, Any], second: dict[str, Any], frame: int
) -> dict[str, float]:
    key = f"{frame:06d}"
    a = np.asarray(
        first["frames"][key]["parts"][PART]["T_world_from_part"],
        dtype=np.float64,
    )
    b = np.asarray(
        second["frames"][key]["parts"][PART]["T_world_from_part"],
        dtype=np.float64,
    )
    delta = np.linalg.inv(a) @ b
    return {
        "translation_m": float(np.linalg.norm(delta[:3, 3])),
        "rotation_deg": float(np.degrees(
            Rotation.from_matrix(delta[:3, :3]).magnitude()
        )),
    }


def _parse_bridge(raw: str) -> tuple[int, int, int]:
    fields = raw.split(":")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("bridge must be FRAME:LEFT:RIGHT")
    try:
        return tuple(int(value) for value in fields)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bridge values must be integers") from exc


def main() -> None:
    require_memory_guard("tools/diagnostics/run_rigid_core_nozzle_validation.py")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--rigid-masks-dir", required=True, type=Path)
    parser.add_argument("--rigid-mesh-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--frames", required=True, nargs="+", type=int)
    parser.add_argument("--bridge", action="append", default=[], type=_parse_bridge)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    frames = sorted(set(int(value) for value in args.frames))
    if not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError(f"frames must contain between 1 and {MAX_FRAMES} values")
    bridge_frames = [item[0] for item in args.bridge]
    if len(bridge_frames) != len(set(bridge_frames)):
        raise ValueError("each bridged frame may appear only once")
    if sorted(set(bridge_frames).difference(frames)):
        raise ValueError("every bridged frame must also be requested in --frames")

    output_root = args.output_root.resolve()
    summary_path = output_root / "summary.json"
    if summary_path.exists() and not args.force:
        print(f"[resume] {summary_path}")
        return
    runtime = output_root / "runtime"
    trajectories = output_root / "trajectories"
    reports = output_root / "reports"
    for path in (runtime, trajectories, reports):
        path.mkdir(parents=True, exist_ok=True)

    base_config = load_json(args.base_config.resolve())
    original = load_json(args.trajectory.resolve())
    if PART not in original.get("parts", []):
        raise ValueError(f"trajectory does not contain {PART!r}")
    seeded, bridge_report = _bridge_seed(original, list(args.bridge))
    seed_path = trajectories / "seeded.json"
    write_trajectory_files(seeded, seed_path)

    full_masks = Path(base_config["masks_dir"])
    full_meshes = Path(base_config["mesh_dir"])
    rigid_masks = args.rigid_masks_dir.resolve()
    rigid_meshes = args.rigid_mesh_dir.resolve()
    for path in (full_masks, full_meshes, rigid_masks, rigid_meshes):
        if not path.exists():
            raise FileNotFoundError(path)

    configs = {
        "full_optimize": _variant_config(
            base_config, masks_dir=full_masks, mesh_dir=full_meshes,
            evaluate_only=False,
        ),
        "rigid_optimize": _variant_config(
            base_config, masks_dir=rigid_masks, mesh_dir=rigid_meshes,
            evaluate_only=False,
        ),
        "rigid_evaluate": _variant_config(
            base_config, masks_dir=rigid_masks, mesh_dir=rigid_meshes,
            evaluate_only=True,
        ),
    }
    config_paths = {}
    for name, config in configs.items():
        path = runtime / f"{name}.json"
        write_json(path, config)
        config_paths[name] = path

    candidates = {"seeded": seed_path}
    optimization_reports = {}
    for name in ("full", "rigid"):
        output = trajectories / f"{name}_optimized.json"
        report = reports / f"{name}_optimization.json"
        _run_refinement(
            config_paths[f"{name}_optimize"], seed_path, output, report, frames
        )
        candidates[name] = output
        optimization_reports[name] = load_json(report)

    common_reports = {}
    for name, trajectory_path in candidates.items():
        output = trajectories / f"{name}_common_evaluated.json"
        report = reports / f"{name}_common_rigid_evaluation.json"
        _run_refinement(
            config_paths["rigid_evaluate"], trajectory_path, output, report, frames
        )
        common_reports[name] = load_json(report)

    full_trajectory = load_json(candidates["full"])
    rigid_trajectory = load_json(candidates["rigid"])
    summary = {
        "schema_version": 1,
        "method": "full_vs_rigid_core_common_objective_keyframe_validation",
        "frames": frames,
        "bridge_seeds": bridge_report,
        "inputs": {
            "base_config": str(args.base_config.resolve()),
            "trajectory": str(args.trajectory.resolve()),
            "rigid_masks_dir": str(rigid_masks),
            "rigid_mesh_dir": str(rigid_meshes),
        },
        "optimization_objectives": {
            name: {
                "baseline_mean_iou": _mean_iou(report, "baseline_optimize"),
                # ``refined_optimize`` describes the optimizer's proposal even
                # when an independent holdout view rejects that proposal.
                "proposed_mean_iou": _mean_iou(report, "refined_optimize"),
                "accepted_frames": _accepted_frames(report),
            }
            for name, report in optimization_reports.items()
        },
        "common_rigid_core_mean_iou": {
            name: _mean_iou(report) for name, report in common_reports.items()
        },
        "per_frame": {
            f"{frame:06d}": {
                "seeded_common_rigid_iou": _metric_rows(
                    common_reports["seeded"]
                ).get(f"{frame:06d}", {}).get("baseline_optimize", {}).get("mean_iou"),
                "full_candidate_common_rigid_iou": _metric_rows(
                    common_reports["full"]
                ).get(f"{frame:06d}", {}).get("baseline_optimize", {}).get("mean_iou"),
                "rigid_candidate_common_rigid_iou": _metric_rows(
                    common_reports["rigid"]
                ).get(f"{frame:06d}", {}).get("baseline_optimize", {}).get("mean_iou"),
                "full_pose_delta_from_seed": _pose_delta(
                    seeded, full_trajectory, frame
                ),
                "rigid_pose_delta_from_seed": _pose_delta(
                    seeded, rigid_trajectory, frame
                ),
            }
            for frame in frames
        },
    }
    write_json(summary_path, summary)
    print(f"summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
