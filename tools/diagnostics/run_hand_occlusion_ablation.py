#!/usr/bin/env python3
"""Run a bounded rigid-only versus rigid-plus-hand render-loss A/B test."""
from __future__ import annotations

import argparse
import hashlib
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


MAX_FRAMES = 40
MAX_PARTS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refinement_settings(
    cfg: dict[str, Any],
    *,
    parts: list[str],
    known_labels: list[int],
    evaluate_only: bool,
) -> dict[str, Any]:
    views = [str(view) for view in cfg["views"]]
    if evaluate_only:
        optimize_views, holdout_views = views, []
    else:
        holdout_count = 2 if len(views) >= 6 else 1
        optimize_views = views[:-holdout_count]
        holdout_views = views[-holdout_count:]
    return {
        "enabled": True,
        "resolution": [160, 90],
        "surface_points": 15000,
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
        "known_occluder_labels": known_labels,
        "minimum_full_mask_pixels": 20,
        "presence_minimum_full_mask_pixels": 20,
        "maximum_mask_area_ratio": 5.0,
        "translation_steps_m": [] if evaluate_only else [0.008, 0.004, 0.002],
        "rotation_steps_deg": [] if evaluate_only else [4.0, 2.0, 1.0],
        "maximum_translation_delta_m": 0.02,
        "maximum_rotation_delta_deg": 8.0,
        "minimum_improvement": 0.0 if evaluate_only else 0.001,
        "maximum_holdout_degradation": 0.03,
        "minimum_refined_iou": 0.0,
        "minimum_holdout_iou": 0.0,
        "minimum_per_view_iou": 0.0,
        "prior_weight": 0.01,
        "temporal_weight": 0.0,
        "trim_worst_views": 0,
        "exact_triangle_refinement": False,
        "weights": {
            "iou": 1.0,
            "contour": 0.25,
            "target_coverage": 0.20,
            "depth": 0.0,
        },
        "parts": {
            part: {
                "enabled": True,
                "protect_tracking_anchors": False,
            }
            for part in parts
        },
    }


def write_variant_config(
    base: dict[str, Any],
    path: Path,
    *,
    parts: list[str],
    known_labels: list[int],
    evaluate_only: bool,
) -> None:
    cfg = dict(base)
    cfg["render_loss_refinement"] = refinement_settings(
        cfg,
        parts=parts,
        known_labels=known_labels,
        evaluate_only=evaluate_only,
    )
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
    write_json(path, cfg)


def run_refinement(
    config: Path,
    trajectory: Path,
    output_trajectory: Path,
    report: Path,
    *,
    parts: list[str],
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
        *parts,
        "--frames",
        *[str(frame) for frame in frames],
    ]
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def aggregate_eval(
    report: dict[str, Any],
    mask_analysis: dict[str, Any],
) -> dict[str, Any]:
    accumulators: dict[str, dict[str, list[float]]] = {}
    for part, part_report in report.get("parts", {}).items():
        for timestamp, row in part_report.get("frames", {}).items():
            metric = row.get("baseline_optimize")
            if not metric:
                continue
            mask_row = mask_analysis["frames"].get(timestamp, {})
            if int(mask_row.get("hand_views", 0)) == 0:
                category = "clean"
            elif float(mask_row.get("maximum_boundary_touch_ratio", 0.0)) > 0:
                category = "hand_touching"
            else:
                category = "hand_noncontact"
            for key in ("all", category):
                bucket = accumulators.setdefault(
                    f"{part}:{key}",
                    {"iou": [], "loss": [], "coverage": []},
                )
                bucket["iou"].append(float(metric["mean_iou"]))
                bucket["loss"].append(float(metric["loss"]))
                bucket["coverage"].append(
                    float(metric["mean_target_coverage"])
                )
    result = {}
    for key, values in accumulators.items():
        result[key] = {
            "frames": len(values["iou"]),
            "mean_iou": float(np.mean(values["iou"])),
            "mean_loss": float(np.mean(values["loss"])),
            "mean_target_coverage": float(np.mean(values["coverage"])),
        }
    return result


def pose_differences(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    parts: list[str],
    frames: list[int],
) -> dict[str, Any]:
    result = {}
    for part in parts:
        translations, rotations = [], []
        for frame in frames:
            timestamp = f"{frame:06d}"
            a = np.asarray(
                first["frames"][timestamp]["parts"][part]["T_world_from_part"],
                dtype=np.float64,
            )
            b = np.asarray(
                second["frames"][timestamp]["parts"][part]["T_world_from_part"],
                dtype=np.float64,
            )
            translations.append(float(np.linalg.norm(a[:3, 3] - b[:3, 3])))
            rotations.append(float(np.degrees(
                Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).magnitude()
            )))
        result[part] = {
            "frames": len(translations),
            "translation_difference_median_m": float(np.median(translations)),
            "translation_difference_max_m": float(np.max(translations)),
            "rotation_difference_median_deg": float(np.median(rotations)),
            "rotation_difference_max_deg": float(np.max(rotations)),
        }
    return result


def main() -> None:
    require_memory_guard("tools/diagnostics/run_hand_occlusion_ablation.py")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--mask-analysis", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--parts", nargs="+", required=True)
    parser.add_argument("--frames", nargs="+", required=True, type=int)
    args = parser.parse_args()

    parts = [str(part) for part in args.parts]
    frames = sorted(set(int(frame) for frame in args.frames))
    if not 1 <= len(parts) <= MAX_PARTS:
        raise ValueError(f"parts must contain at most {MAX_PARTS} entries")
    if not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError(f"frames must contain at most {MAX_FRAMES} entries")

    base_config = load_json(args.base_config.resolve())
    trajectory_path = args.trajectory.resolve()
    trajectory = load_json(trajectory_path)
    unknown_parts = sorted(set(parts).difference(trajectory["parts"]))
    missing_frames = [
        frame for frame in frames
        if f"{frame:06d}" not in trajectory["frames"]
    ]
    if unknown_parts or missing_frames:
        raise ValueError(
            f"unknown parts={unknown_parts}, missing frames={missing_frames}"
        )

    output_root = args.output_root.resolve()
    runtime = output_root / "runtime"
    reports = output_root / "reports"
    trajectories = output_root / "trajectories"
    runtime.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    trajectories.mkdir(parents=True, exist_ok=True)

    rigid_labels = sorted(int(value) for value in base_config["part_ids"].values())
    mask_analysis = load_json(args.mask_analysis.resolve())
    hand_label = int(mask_analysis["occluder_label"])
    variants = {
        "rigid_only": rigid_labels,
        "rigid_plus_hand": sorted(set(rigid_labels + [hand_label])),
    }
    optimized_paths = {}
    optimization_reports = {}
    evaluation_reports = {}
    for name, labels in variants.items():
        config_path = runtime / f"{name}.json"
        write_variant_config(
            base_config,
            config_path,
            parts=parts,
            known_labels=labels,
            evaluate_only=False,
        )
        output_trajectory = trajectories / f"{name}.json"
        report_path = reports / f"{name}_optimization.json"
        run_refinement(
            config_path,
            trajectory_path,
            output_trajectory,
            report_path,
            parts=parts,
            frames=frames,
        )
        optimized_paths[name] = output_trajectory
        optimization_reports[name] = load_json(report_path)

        eval_config = runtime / f"{name}_common_eval.json"
        write_variant_config(
            base_config,
            eval_config,
            parts=parts,
            known_labels=variants["rigid_plus_hand"],
            evaluate_only=True,
        )
        eval_report_path = reports / f"{name}_common_eval.json"
        run_refinement(
            eval_config,
            output_trajectory,
            trajectories / f"{name}_evaluated.json",
            eval_report_path,
            parts=parts,
            frames=frames,
        )
        evaluation_reports[name] = load_json(eval_report_path)

    metrics = {
        name: aggregate_eval(report, mask_analysis)
        for name, report in evaluation_reports.items()
    }
    deltas = {}
    for key in sorted(set(metrics["rigid_only"]) & set(metrics["rigid_plus_hand"])):
        baseline = metrics["rigid_only"][key]
        hand = metrics["rigid_plus_hand"][key]
        deltas[key] = {
            "mean_iou_delta": float(hand["mean_iou"] - baseline["mean_iou"]),
            "mean_loss_improvement": float(
                baseline["mean_loss"] - hand["mean_loss"]
            ),
            "mean_target_coverage_delta": float(
                hand["mean_target_coverage"]
                - baseline["mean_target_coverage"]
            ),
        }

    summary = {
        "schema_version": 1,
        "method": "same_initial_pose_label_selective_hand_occlusion_ablation",
        "base_config": str(args.base_config.resolve()),
        "initial_trajectory": str(trajectory_path),
        "initial_trajectory_sha256": sha256_file(trajectory_path),
        "mask_analysis": str(args.mask_analysis.resolve()),
        "parts": parts,
        "frames": frames,
        "variants": variants,
        "common_evaluator_labels": variants["rigid_plus_hand"],
        "common_metrics": metrics,
        "hand_aware_minus_baseline": deltas,
        "pose_differences": pose_differences(
            load_json(optimized_paths["rigid_only"]),
            load_json(optimized_paths["rigid_plus_hand"]),
            parts=parts,
            frames=frames,
        ),
        "optimization": {
            name: {
                part: report["parts"].get(part, {}).get("summary", {})
                for part in parts
            }
            for name, report in optimization_reports.items()
        },
    }
    write_json(output_root / "ablation_summary.json", summary)
    print(f"summary -> {output_root / 'ablation_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
