#!/usr/bin/env python
"""Refine moving-part SE(3) poses with multi-view mesh render losses."""
from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.normalized_recon import load_recon
from common.pose_config import validate_pose_config
from common.pose_refinement import sample_canonical
from common.pose_validation import validate_trajectory
from common.pose_visualization import camera_from_recon
from common.render_loss_refinement import (
    MultiViewRenderObjective,
    RenderObservation,
    refine_pose_coordinate_search,
    world_pose_delta_vector,
)
from common.symmetry import symmetry_spec_from_state
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


def configured_ranges(
    state: dict[str, Any],
    part_config: dict[str, Any],
) -> list[tuple[int, int]]:
    values = part_config.get("ranges", state.get("dynamic_ranges", []))
    return [(int(item[0]), int(item[1])) for item in values]


def load_observations(
    cfg: dict[str, Any],
    timestamp: str,
    part: str,
    refinement: dict[str, Any],
) -> list[RenderObservation]:
    width, height = [
        int(value) for value in refinement.get("resolution", [160, 90])
    ]
    part_id = int(cfg["part_ids"][part])
    minimum_pixels = int(refinement.get("min_mask_pixels", 30))
    recon = load_recon(cfg, timestamp, backend=cfg["recon_backend"])
    observations = []
    for view_index, view in enumerate(cfg["views"]):
        labels = np.asarray(
            Image.open(Path(cfg["masks_dir"]) / timestamp / f"{view}.png")
        )
        target = cv2.resize(
            (labels == part_id).astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if int(target.sum()) < minimum_pixels:
            continue
        intrinsics, extrinsics = camera_from_recon(
            recon, view_index, (height, width)
        )
        observed_depth = cv2.resize(
            recon["depth"][view_index].astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        observations.append(
            RenderObservation(
                view=view,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                target_mask=target,
                observed_depth=observed_depth,
            )
        )
    return observations


def compact_metric(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "loss",
            "mean_iou",
            "mean_contour_chamfer_px",
            "mean_target_coverage",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output-trajectory", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--parts", nargs="+", default=None)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="debugging limit across all parts; omitted runs every configured frame",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    trajectory_path = Path(args.trajectory).resolve()
    output_path = Path(args.output_trajectory).resolve()
    report_path = (
        Path(args.report).resolve()
        if args.report
        else output_path.parents[1] / "diagnostics/render_loss_refinement.json"
    )
    cfg = validate_pose_config(load_json(config_path), check_paths=True)
    refinement = dict(cfg.get("render_loss_refinement", {}))
    if not refinement.get("enabled", False):
        raise ValueError("render_loss_refinement.enabled must be true")
    baseline = load_json(trajectory_path)
    trajectory = copy.deepcopy(baseline)
    requested_parts = list(args.parts or refinement.get("parts", {}).keys())
    unknown = sorted(set(requested_parts).difference(trajectory["parts"]))
    if unknown:
        raise ValueError(f"unknown refinement parts: {unknown}")

    optimize_views = [
        str(value)
        for value in refinement.get("optimize_views", cfg["views"])
    ]
    holdout_views = [
        str(value)
        for value in refinement.get("holdout_views", [])
    ]
    overlap = sorted(set(optimize_views).intersection(holdout_views))
    if overlap:
        raise ValueError(f"optimize_views and holdout_views overlap: {overlap}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "multi_view_sampled_mesh_render_loss_coordinate_search",
        "config": str(config_path),
        "trajectory_input": str(trajectory_path),
        "trajectory_input_sha256": sha256_file(trajectory_path),
        "trajectory_output": str(output_path),
        "resolution": refinement.get("resolution", [160, 90]),
        "optimize_views": optimize_views,
        "holdout_views": holdout_views,
        "parts": {},
    }
    processed_frames = 0
    for part_index, part in enumerate(requested_parts):
        part_config = dict(refinement.get("parts", {}).get(part, {}))
        if not part_config.get("enabled", True):
            continue
        state = cfg["states"][part]
        ranges = configured_ranges(state, part_config)
        if not ranges:
            continue
        raw_mesh = trimesh.load(
            Path(cfg["mesh_dir"]) / f"{part}.glb", force="mesh"
        )
        points = sample_canonical(
            raw_mesh,
            float(baseline["scales"][part]),
            np.asarray(baseline["raw_mesh_origins"][part], dtype=np.float64),
            count=int(refinement.get("surface_points", 30000)),
            seed=int(refinement.get("seed", 1701)) + part_index,
        )
        symmetry = symmetry_spec_from_state(state)
        symmetry_axis = (
            symmetry.axis
            if symmetry.equivalence == "continuous_axial"
            else None
        )
        part_report: dict[str, Any] = {
            "ranges": [list(item) for item in ranges],
            "surface_points": int(len(points)),
            "frames": {},
        }
        accepted_count = 0
        temporal_delta = np.zeros(6, dtype=np.float64)
        for range_start, range_end in ranges:
            temporal_delta[:] = 0.0
            for frame in range(range_start, range_end + 1):
                if (
                    args.max_frames is not None
                    and processed_frames >= args.max_frames
                ):
                    break
                timestamp = f"{frame:06d}"
                if timestamp not in baseline["frames"]:
                    continue
                record = baseline["frames"][timestamp]["parts"][part]
                if int(record.get("observing_views", 0)) <= 0:
                    part_report["frames"][timestamp] = {
                        "accepted": False,
                        "skip_reason": "unobservable",
                    }
                    continue
                observations = load_observations(
                    cfg, timestamp, part, refinement
                )
                available = {item.view for item in observations}
                effective_optimize = [
                    view for view in optimize_views if view in available
                ]
                if len(effective_optimize) < int(
                    refinement.get("minimum_optimize_views", 3)
                ):
                    part_report["frames"][timestamp] = {
                        "accepted": False,
                        "skip_reason": "insufficient_views",
                        "available_views": sorted(available),
                    }
                    continue
                objective = MultiViewRenderObjective(
                    points, observations, refinement
                )
                initial = np.asarray(
                    record["T_world_from_part"], dtype=np.float64
                )
                selected, frame_report = refine_pose_coordinate_search(
                    objective,
                    initial,
                    optimize_views=optimize_views,
                    holdout_views=holdout_views,
                    translation_steps_m=[
                        float(value)
                        for value in part_config.get(
                            "translation_steps_m",
                            refinement.get(
                                "translation_steps_m",
                                [0.006, 0.003, 0.0015],
                            ),
                        )
                    ],
                    rotation_steps_deg=[
                        float(value)
                        for value in part_config.get(
                            "rotation_steps_deg",
                            refinement.get(
                                "rotation_steps_deg", [2.0, 1.0, 0.5]
                            ),
                        )
                    ],
                    symmetry_axis_part=symmetry_axis,
                    optimize_rotation=bool(
                        part_config.get("optimize_rotation", True)
                    ),
                    maximum_translation_delta_m=float(
                        part_config.get(
                            "maximum_translation_delta_m",
                            refinement.get(
                                "maximum_translation_delta_m", 0.015
                            ),
                        )
                    ),
                    maximum_rotation_delta_deg=float(
                        part_config.get(
                            "maximum_rotation_delta_deg",
                            refinement.get(
                                "maximum_rotation_delta_deg", 4.0
                            ),
                        )
                    ),
                    minimum_improvement=float(
                        refinement.get("minimum_improvement", 0.002)
                    ),
                    maximum_holdout_degradation=float(
                        refinement.get(
                            "maximum_holdout_degradation", 0.015
                        )
                    ),
                    prior_weight=float(refinement.get("prior_weight", 0.03)),
                    temporal_delta_reference=temporal_delta,
                    temporal_weight=float(
                        refinement.get("temporal_weight", 0.02)
                    ),
                )
                accepted = bool(frame_report["accepted"])
                if accepted:
                    target_record = trajectory["frames"][timestamp]["parts"][
                        part
                    ]
                    target_record["T_world_from_part"] = selected.tolist()
                    target_record["source"] = (
                        str(target_record.get("source", "pose"))
                        + "+render_loss"
                    )
                    temporal_delta = world_pose_delta_vector(initial, selected)
                    accepted_count += 1
                else:
                    temporal_delta *= float(
                        refinement.get("rejected_temporal_decay", 0.5)
                    )
                part_report["frames"][timestamp] = {
                    "accepted": accepted,
                    "available_views": sorted(available),
                    "evaluations": frame_report["evaluations"],
                    "translation_delta_m": frame_report[
                        "translation_delta_m"
                    ],
                    "translation_delta_norm_m": frame_report[
                        "translation_delta_norm_m"
                    ],
                    "rotation_delta_deg": frame_report[
                        "rotation_delta_deg"
                    ],
                    "optimize_loss_improvement": frame_report[
                        "optimize_loss_improvement"
                    ],
                    "holdout_loss_degradation": frame_report[
                        "holdout_loss_degradation"
                    ],
                    "baseline_optimize": compact_metric(
                        frame_report["baseline_optimize"]
                    ),
                    "refined_optimize": compact_metric(
                        frame_report["refined_optimize"]
                    ),
                    "baseline_holdout": compact_metric(
                        frame_report["baseline_holdout"]
                    ),
                    "refined_holdout": compact_metric(
                        frame_report["refined_holdout"]
                    ),
                }
                processed_frames += 1
                print(
                    f"{part} {timestamp}: accepted={accepted} "
                    f"opt Δ={frame_report['optimize_loss_improvement']:+.4f} "
                    f"holdout Δ={frame_report['holdout_loss_degradation']:+.4f} "
                    f"t={1000.0 * frame_report['translation_delta_norm_m']:.1f}mm "
                    f"r={frame_report['rotation_delta_deg']:.2f}deg",
                    flush=True,
                )
            if (
                args.max_frames is not None
                and processed_frames >= args.max_frames
            ):
                break
        frame_rows = list(part_report["frames"].values())
        evaluated = [
            row for row in frame_rows if "baseline_optimize" in row
        ]
        accepted_rows = [row for row in evaluated if row["accepted"]]

        def selected_mean_iou(split: str) -> tuple[float | None, float | None]:
            rows = [
                row
                for row in evaluated
                if row[f"baseline_{split}"].get("mean_iou") is not None
                and row[f"refined_{split}"].get("mean_iou") is not None
            ]
            if not rows:
                return None, None
            baseline_mean = float(
                np.mean(
                    [row[f"baseline_{split}"]["mean_iou"] for row in rows]
                )
            )
            selected_mean = float(
                np.mean(
                    [
                        (
                            row[f"refined_{split}"]["mean_iou"]
                            if row["accepted"]
                            else row[f"baseline_{split}"]["mean_iou"]
                        )
                        for row in rows
                    ]
                )
            )
            return baseline_mean, selected_mean

        baseline_optimize_iou, selected_optimize_iou = selected_mean_iou(
            "optimize"
        )
        baseline_holdout_iou, selected_holdout_iou = selected_mean_iou(
            "holdout"
        )
        part_report["summary"] = {
            "evaluated_frames": len(evaluated),
            "accepted_frames": accepted_count,
            "acceptance_rate": (
                float(accepted_count / len(evaluated)) if evaluated else None
            ),
            "baseline_optimize_mean_iou": baseline_optimize_iou,
            "selected_optimize_mean_iou": selected_optimize_iou,
            "baseline_holdout_mean_iou": baseline_holdout_iou,
            "selected_holdout_mean_iou": selected_holdout_iou,
            "accepted_translation_delta_median_m": (
                float(
                    np.median(
                        [
                            row["translation_delta_norm_m"]
                            for row in accepted_rows
                        ]
                    )
                )
                if accepted_rows
                else None
            ),
            "accepted_translation_delta_max_m": (
                float(
                    max(
                        row["translation_delta_norm_m"]
                        for row in accepted_rows
                    )
                )
                if accepted_rows
                else None
            ),
            "accepted_rotation_delta_median_deg": (
                float(
                    np.median(
                        [row["rotation_delta_deg"] for row in accepted_rows]
                    )
                )
                if accepted_rows
                else None
            ),
            "accepted_rotation_delta_max_deg": (
                float(
                    max(row["rotation_delta_deg"] for row in accepted_rows)
                )
                if accepted_rows
                else None
            ),
        }
        report["parts"][part] = part_report
        if (
            args.max_frames is not None
            and processed_frames >= args.max_frames
        ):
            break

    refresh_trajectory_derived_fields(trajectory)
    trajectory_validation, validation_failures = validate_trajectory(
        cfg, trajectory, enforce_assembly=False
    )
    report["trajectory_validation"] = trajectory_validation
    if validation_failures:
        report["summary"] = {
            "processed_frames": processed_frames,
            "accepted_frames": int(
                sum(
                    item["summary"]["accepted_frames"]
                    for item in report["parts"].values()
                )
            ),
            "validation_passed": False,
        }
        write_json(report_path, report)
        raise RuntimeError("; ".join(validation_failures))
    trajectory.setdefault("refinements", []).append(
        {
            "method": report["method"],
            "input": str(trajectory_path),
            "report": str(report_path),
        }
    )
    write_trajectory_files(trajectory, output_path)
    report["trajectory_output_sha256"] = sha256_file(output_path)
    report["summary"] = {
        "processed_frames": processed_frames,
        "accepted_frames": int(
            sum(
                item["summary"]["accepted_frames"]
                for item in report["parts"].values()
            )
        ),
        "validation_passed": True,
    }
    write_json(report_path, report)
    print(f"trajectory -> {output_path}", flush=True)
    print(f"report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
