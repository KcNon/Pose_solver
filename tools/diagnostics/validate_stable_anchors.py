#!/usr/bin/env python
"""Select, fit, rank, and visualize stable multi-view pose anchors.

This is intentionally an anchor-only diagnostic.  It shortlists frames from a
stable interval, invokes the normal calibration stack (cloud registration,
fixed scale, silhouette scale, and appearance orientation), compares the
surviving poses with the same occlusion-aware mask/depth render objective, and
exports review sheets.  It never solves a temporal trajectory or runs Isaac.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.geom import project_points
from common.io_utils import load_json, write_json
from common.mask_io import frame_path
from common.mesh_render import SceneRenderer
from common.normalized_recon import load_recon
from common.pose_config import validate_pose_config
from common.pose_refinement import sample_canonical
from common.pose_transforms import rigid_from_similarity
from common.pose_visualization import (
    camera_from_recon,
    part_color,
    solid_mesh,
    tile_image_panels,
)
from common.render_loss_refinement import MultiViewRenderObjective
from common.stable_anchor import (
    centered_stable_window,
    rank_stable_frames,
    select_separated_candidates,
)
from common.support_plane import estimate_local_table_plane
from tools.stages.pose.refine_pose_render_loss import load_observations


def _state_rows(report: dict[str, Any], part: str) -> dict[int, dict[str, Any]]:
    return {
        int(timestamp): dict(row)
        for timestamp, row in report["parts"][part]["states"].items()
    }


def _cloud_rows(report: dict[str, Any], part: str) -> dict[int, dict[str, Any]]:
    return {
        int(timestamp): dict(frame[part])
        for timestamp, frame in report.get("frames", {}).items()
        if part in frame
    }


def _parse_intervals(values: list[list[str]]) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for part, raw_start, raw_end in values:
        start, end = int(raw_start), int(raw_end)
        if start > end:
            raise ValueError(f"invalid interval {part}:{start}:{end}")
        if part in result:
            raise ValueError(f"duplicate interval for {part}")
        result[part] = (start, end)
    return result


def _parse_scale_priors(values: list[list[str]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for part, raw_scale in values:
        scale = float(raw_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"invalid scale prior {part}:{raw_scale}")
        if part in result:
            raise ValueError(f"duplicate scale prior for {part}")
        result[part] = scale
    return result


def _runtime_config(
    source: dict[str, Any],
    *,
    intervals: dict[str, tuple[int, int]],
    candidates: dict[str, list[int]],
    windows: dict[str, dict[str, list[int]]],
    reference_part: str,
    output_root: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(source)
    cfg["output_root"] = str(output_root)
    cfg["parts"] = list(intervals)
    cfg["part_ids"] = {
        part: int(source["part_ids"][part]) for part in cfg["parts"]
    }
    cfg["reference_part"] = reference_part
    cfg["automation"]["allow_moving_reference"] = True
    cfg["automation"]["enabled"] = False
    cfg["automation"]["select_stable_interval_anchor_winner"] = False
    if cfg["automation"].get("align_parts_to_table") is not None:
        cfg["automation"]["align_parts_to_table"] = [
            part
            for part in cfg["automation"]["align_parts_to_table"]
            if part in cfg["parts"]
        ]
    if cfg["automation"].get("table_support_ranges") is not None:
        cfg["automation"]["table_support_ranges"] = {
            part: ranges
            for part, ranges in cfg["automation"][
                "table_support_ranges"
            ].items()
            if part in cfg["parts"]
        }
    # Collision/trajectory constraints are intentionally outside this
    # diagnostic and may reference retired runtime configs.
    cfg["trajectory_constraints"] = {"enabled": False}
    scale = cfg.get("automation", {}).get("silhouette_scale_calibration", {})
    if scale.get("enabled", False):
        scale["maximum_anchor_frames"] = max(
            len(values) for values in candidates.values()
        )
    cfg["states"] = {
        part: copy.deepcopy(source["states"][part]) for part in cfg["parts"]
    }
    for part in cfg["parts"]:
        state = cfg["states"][part]
        state["anchor_frames"] = list(candidates[part])
        # Reused calibrations validate that tracking anchors are present in the
        # calibration. Keep this diagnostic's tracking anchors synchronized
        # with its candidate set instead of inheriting stale production ones.
        state["tracking_anchor_frames"] = list(candidates[part])
        state["anchor_windows"] = dict(windows[part])
        appearance = state.get("appearance", {})
        if appearance.get("enabled", False):
            # Every candidate in this diagnostic must stand on its own. A
            # static carry hypothesis or a temporal chain can otherwise make
            # a later candidate inherit an earlier pose and conceal a bad
            # single-frame multi-view initialization.
            appearance["allow_static_consensus_hypotheses"] = False
            appearance["transition_weight"] = 0.0
            appearance["hard_rotation_rate"] = False
            appearance["anchor_evidence_frames"] = {
                str(frame): [frame] for frame in candidates[part]
            }
    return cfg


def _run_calibration(
    cfg: dict[str, Any],
    *,
    config_path: Path,
    force: bool,
) -> dict[str, Any]:
    write_json(config_path, cfg)
    calibration_path = Path(cfg["output_root"]) / "pose" / "calibration.json"
    command = [
        sys.executable,
        "-u",
        "tools/stages/pose/solve_multiview_pose.py",
        "--config",
        str(config_path),
        "--calibrate-only",
    ]
    if calibration_path.exists() and not force:
        command.append("--reuse-calibration")
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return load_json(calibration_path)


def _compact_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metric.get(key)
        for key in (
            "loss",
            "mean_iou",
            "mean_contour_chamfer_px",
            "mean_target_coverage",
            "views",
        )
    }


def _evaluate_candidates(
    cfg: dict[str, Any],
    calibration: dict[str, Any],
    candidates: dict[str, list[int]],
    rankings: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, int | None]]:
    refinement = dict(cfg.get("render_loss_refinement", {}))
    optimize_views = [
        str(value) for value in refinement.get("optimize_views", cfg["views"])
    ]
    holdout_views = [str(value) for value in refinement.get("holdout_views", [])]
    result: dict[str, Any] = {}
    winners: dict[str, int | None] = {}
    for part_index, part in enumerate(candidates):
        mesh = trimesh.load(
            Path(cfg["mesh_dir"]) / f"{part}.glb", force="mesh"
        )
        scale = float(calibration["scales"][part])
        origin = np.asarray(
            calibration["raw_mesh_origins"][part], dtype=np.float64
        )
        points = sample_canonical(
            mesh,
            scale,
            origin,
            count=int(refinement.get("surface_points", 30000)),
            seed=int(refinement.get("seed", 1701)) + part_index,
        )
        pre_by_frame = {int(row["frame"]): row for row in rankings[part]}
        candidate_rows = []
        finite_pre = [
            float(pre_by_frame[frame]["shortlist_score"])
            for frame in candidates[part]
        ]
        pre_min, pre_max = min(finite_pre), max(finite_pre)
        for frame in candidates[part]:
            info = calibration["anchors"][part].get(str(frame), {})
            row: dict[str, Any] = {
                "frame": frame,
                "shortlist": pre_by_frame[frame],
                "reliable": bool(info.get("reliable", False)),
                "fit_rmse_m": info.get("fit_rmse_m"),
                "fixed_scale": info.get("fixed_scale", scale),
                "appearance_selection": info.get("appearance_selection"),
            }
            transform = info.get("S_world_from_raw")
            if not row["reliable"] or transform is None:
                row.update(
                    {
                        "selection_loss": None,
                        "rejection_reason": info.get(
                            "rejection_reason", "calibration_rejected"
                        ),
                    }
                )
                candidate_rows.append(row)
                continue
            pose = rigid_from_similarity(np.asarray(transform), origin)
            observations = load_observations(
                cfg, f"{frame:06d}", part, refinement
            )
            objective = MultiViewRenderObjective(points, observations, refinement)
            optimize = objective.evaluate(pose, optimize_views)
            holdout = objective.evaluate(pose, holdout_views)
            pre_value = float(pre_by_frame[frame]["shortlist_score"])
            pre_normalized = (
                (pre_value - pre_min) / (pre_max - pre_min)
                if pre_max > pre_min
                else 1.0
            )
            fit_penalty = min(
                float(info.get("fit_rmse_m", 1.0)) / 0.05, 2.0
            )
            holdout_loss = (
                float(holdout["loss"])
                if holdout.get("views") and math.isfinite(float(holdout["loss"]))
                else float(optimize["loss"])
            )
            selection_loss = (
                0.55 * float(optimize["loss"])
                + 0.20 * holdout_loss
                + 0.15 * fit_penalty
                + 0.10 * (1.0 - pre_normalized)
            )
            appearance = info.get("appearance_selection", {}) or {}
            if not appearance.get("appearance_accepted", True):
                selection_loss += 0.5
            all_view_rows = optimize.get("views", []) + holdout.get("views", [])
            observed_views = {str(item["view"]) for item in all_view_rows}
            missing_views = [
                str(view) for view in cfg["views"] if str(view) not in observed_views
            ]
            catastrophic_views = [
                item["view"]
                for item in all_view_rows
                if float(item.get("iou", 0.0)) < 0.15
                or float(item.get("target_coverage", 0.0)) < 0.20
            ]
            catastrophic_views = sorted(set(catastrophic_views))
            minimum_valid_views = int(
                cfg.get("anchor_validation", {}).get(
                    "minimum_valid_views", max(3, len(cfg["views"]) - 2)
                )
            )
            anchor_quality_passed = bool(
                len(observed_views) >= minimum_valid_views
                and not catastrophic_views
            )
            row.update(
                {
                    "T_world_from_part": pose.tolist(),
                    "S_world_from_raw_mesh": transform,
                    "optimize": _compact_metric(optimize),
                    "holdout": _compact_metric(holdout),
                    "per_view": {
                        item["view"]: item
                        for item in all_view_rows
                    },
                    "raw_selection_loss": float(selection_loss),
                    "selection_loss": (
                        float(selection_loss) if anchor_quality_passed else None
                    ),
                    "anchor_quality_passed": anchor_quality_passed,
                    "catastrophic_views": catastrophic_views,
                    "missing_render_views": missing_views,
                    "observed_render_views": len(observed_views),
                    "minimum_valid_views": minimum_valid_views,
                    "selection_loss_terms": {
                        "optimize_weight": 0.55,
                        "holdout_weight": 0.20,
                        "registration_weight": 0.15,
                        "shortlist_weight": 0.10,
                        "normalized_shortlist_score": pre_normalized,
                        "registration_penalty": fit_penalty,
                    },
                }
            )
            if not anchor_quality_passed:
                row["rejection_reason"] = "catastrophic_multiview_render_mismatch"
            candidate_rows.append(row)
        ordered = sorted(
            candidate_rows,
            key=lambda row: (
                row["selection_loss"] is None,
                float(
                    row["selection_loss"]
                    if row["selection_loss"] is not None
                    else row.get("raw_selection_loss", float("inf"))
                ),
            ),
        )
        reliable = [row for row in ordered if row["selection_loss"] is not None]
        for rank, row in enumerate(ordered, 1):
            row["final_rank"] = rank
        best_candidate = int(ordered[0]["frame"]) if ordered else None
        winners[part] = int(reliable[0]["frame"]) if reliable else None
        result[part] = {
            "scale": scale,
            "winner": winners[part],
            "best_candidate": best_candidate,
            "accepted": bool(reliable),
            "candidates": ordered,
        }
    return result, winners


def _draw_axes(
    image: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    length: float,
) -> None:
    points = np.vstack(
        [
            pose[:3, 3],
            pose[:3, 3] + pose[:3, 0] * length,
            pose[:3, 3] + pose[:3, 1] * length,
            pose[:3, 3] + pose[:3, 2] * length,
        ]
    )
    uv, depth = project_points(points, intrinsics, extrinsics)
    if not np.all(np.isfinite(uv)) or np.any(depth <= 0):
        return
    origin = tuple(np.rint(uv[0]).astype(int))
    for index, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0)), 1):
        endpoint = tuple(np.rint(uv[index]).astype(int))
        cv2.arrowedLine(image, origin, endpoint, color, 3, cv2.LINE_AA, tipLength=0.18)


def _draw_mask_and_bbox(
    image: np.ndarray,
    labels: np.ndarray,
    part_id: int,
    color: tuple[int, int, int],
) -> None:
    mask = labels == int(part_id)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
    ys, xs = np.nonzero(mask)
    if len(xs):
        cv2.rectangle(
            image,
            (int(xs.min()), int(ys.min())),
            (int(xs.max()), int(ys.max())),
            (0, 255, 255),
            2,
        )


def _render_review_sheet(
    cfg: dict[str, Any],
    calibration: dict[str, Any],
    evaluations: dict[str, Any],
    *,
    focus_part: str,
    frame: int,
    output_path: Path,
    width: int,
    height: int,
) -> None:
    timestamp = f"{frame:06d}"
    parts = [focus_part]
    if focus_part != cfg["reference_part"]:
        parts.insert(0, cfg["reference_part"])
    raw_meshes = {
        part: trimesh.load(
            Path(cfg["mesh_dir"]) / f"{part}.glb", force="mesh"
        )
        for part in parts
    }
    styled = {
        part: solid_mesh(raw_meshes[part], part_color(cfg["parts"].index(part)))
        for part in parts
    }
    transforms = {}
    poses = {}
    for part in parts:
        info = calibration["anchors"][part].get(str(frame), {})
        if not info.get("reliable", False):
            continue
        transform = np.asarray(info["S_world_from_raw"], dtype=np.float64)
        transforms[part] = transform
        poses[part] = rigid_from_similarity(
            transform,
            np.asarray(calibration["raw_mesh_origins"][part], dtype=np.float64),
        )
    if focus_part not in transforms:
        raise RuntimeError(f"missing reliable {focus_part} transform at {frame}")
    recon = load_recon(cfg, timestamp, backend=cfg["recon_backend"])
    metrics = next(
        row
        for row in evaluations[focus_part]["candidates"]
        if int(row["frame"]) == frame
    )
    panels = []
    with SceneRenderer(width, height, cache_mesh_resources=True) as renderer:
        for view_index, view in enumerate(cfg["views"]):
            source_path = Path(
                frame_path(
                    cfg["frames_dir"],
                    cfg.get("frames_layout", "normalized"),
                    timestamp,
                    view,
                )
            )
            source = cv2.imread(str(source_path))
            source = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
            labels = np.asarray(
                Image.open(Path(cfg["masks_dir"]) / timestamp / f"{view}.png")
            )
            labels = cv2.resize(
                labels.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            intrinsics, extrinsics = camera_from_recon(
                recon, view_index, (height, width)
            )
            rgb, depth = renderer.render(
                [(styled[part], transforms[part]) for part in parts if part in transforms],
                intrinsics,
                extrinsics,
            )
            rendered = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            visible = depth > 0
            source[visible] = np.clip(
                0.48 * source[visible] + 0.52 * rendered[visible], 0, 255
            ).astype(np.uint8)
            for part in parts:
                if part not in transforms:
                    continue
                _draw_mask_and_bbox(
                    source,
                    labels,
                    int(cfg["part_ids"][part]),
                    tuple(reversed(part_color(cfg["parts"].index(part)))),
                )
                extent = np.asarray(raw_meshes[part].extents, dtype=float)
                axis_length = max(0.025, float(np.max(extent)) * float(calibration["scales"][part]) * 0.45)
                _draw_axes(
                    source, poses[part], intrinsics, extrinsics, axis_length
                )
            per_view = metrics.get("per_view", {}).get(view, {})
            label = (
                f"{focus_part} {timestamp} {view}  "
                f"IoU {per_view.get('iou', 0.0):.3f}  "
                f"depth {per_view.get('depth_loss') if per_view.get('depth_loss') is not None else 'n/a'}"
            )
            cv2.putText(source, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(source, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            panels.append(source)
    sheet = tile_image_panels(panels, columns=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-report", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--interval",
        nargs=3,
        metavar=("PART", "START", "END"),
        action="append",
        required=True,
    )
    parser.add_argument("--reference-part", required=True)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--minimum-separation", type=int, default=12)
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--review-width", type=int, default=640)
    parser.add_argument("--review-height", type=int, default=360)
    parser.add_argument("--force-calibration", action="store_true")
    parser.add_argument(
        "--scale-prior",
        nargs=2,
        metavar=("PART", "SCALE"),
        action="append",
        default=[],
        help="reuse a scale fixed by an earlier stable anchor validation",
    )
    args = parser.parse_args()

    source_path = Path(args.config).resolve()
    source_raw = load_json(source_path)
    scale_priors = _parse_scale_priors(args.scale_prior)
    for part, scale in scale_priors.items():
        if part not in source_raw.get("states", {}):
            raise ValueError(f"unknown part in scale prior: {part}")
        source_raw["states"][part]["scale_prior"] = scale
    if scale_priors:
        # An explicit prior comes from an earlier clear, unassembled anchor.
        # Re-estimating scale after assembly would shrink a partly occluded
        # mesh to the visible fragment and defeat the purpose of the prior.
        source_raw.setdefault("automation", {}).setdefault(
            "silhouette_scale_calibration", {}
        )["enabled"] = False
    source_raw["trajectory_constraints"] = {"enabled": False}
    source = validate_pose_config(source_raw, check_paths=True)
    intervals = _parse_intervals(args.interval)
    if args.reference_part not in intervals:
        raise ValueError("reference part must have a stable interval")
    unknown = sorted(set(intervals).difference(source["parts"]))
    if unknown:
        raise ValueError(f"unknown parts: {unknown}")
    output = Path(args.output_root).resolve()
    state_path = Path(
        args.state_report
        or Path(source["output_root"]) / "diagnostics" / "part_states.json"
    ).resolve()
    cloud_path = Path(source["point_cloud_root"]) / "quality_cloud_summary.json"
    states_report = load_json(state_path)
    clouds_report = load_json(cloud_path)

    rankings: dict[str, list[dict[str, Any]]] = {}
    candidates: dict[str, list[int]] = {}
    windows: dict[str, dict[str, list[int]]] = {}
    state_by_part: dict[str, dict[int, dict[str, Any]]] = {}
    for part, (start, end) in intervals.items():
        states = _state_rows(states_report, part)
        clouds = _cloud_rows(clouds_report, part)
        state_by_part[part] = states
        ranking = rank_stable_frames(
            states,
            clouds,
            start=start,
            end=end,
            maximum_views=len(source["views"]),
        )
        selected = select_separated_candidates(
            ranking,
            count=args.candidate_count,
            minimum_separation=args.minimum_separation,
        )
        rankings[part] = ranking
        candidates[part] = selected
        windows[part] = {
            str(frame): centered_stable_window(
                states,
                frame,
                start=start,
                end=end,
                size=args.window_size,
            )
            for frame in selected
        }
        print(f"shortlist {part}: {selected}", flush=True)

    reference = args.reference_part
    preliminary_table = estimate_local_table_plane(
        source, candidates[reference][0]
    )
    if not preliminary_table.get("accepted", False):
        raise RuntimeError(
            "stable reference interval has no reliable table plane"
        )
    source["support_plane"] = preliminary_table
    source["automation"]["align_parts_to_table"] = list(intervals)
    source["automation"]["maximum_table_alignment_m"] = 0.05
    target_candidates = copy.deepcopy(candidates)
    target_windows = copy.deepcopy(windows)

    # First determine each physical part's one fixed scale using only that
    # part's own first stable interval. This prevents a later single-frame
    # reference fit from contaminating global scale consensus.
    individual_calibrations: dict[str, dict[str, Any]] = {}
    runtime_paths: dict[str, str] = {}
    for part in intervals:
        pass_root = output / "passes" / part
        single_runtime = _runtime_config(
            source,
            intervals={part: intervals[part]},
            candidates={part: target_candidates[part]},
            windows={part: target_windows[part]},
            reference_part=part,
            output_root=pass_root,
        )
        single_path = pass_root / "runtime" / "anchor_pose_config.json"
        individual_calibrations[part] = _run_calibration(
            single_runtime,
            config_path=single_path,
            force=args.force_calibration,
        )
        runtime_paths[f"scale_{part}"] = str(single_path)

    # At every later-part candidate, independently refit the reference part
    # in the same frame with the already fixed scales. The relative pose is
    # therefore simultaneous and never borrowed from a temporal tracker.
    reference_extra = sorted(
        set().union(
            *(
                set(values)
                for part, values in target_candidates.items()
                if part != reference
            )
        )
    )
    pair_candidates = copy.deepcopy(target_candidates)
    pair_candidates[reference] = reference_extra
    pair_windows = copy.deepcopy(target_windows)
    pair_windows[reference] = {
        str(frame): centered_stable_window(
            state_by_part[reference],
            frame,
            start=int(source["frames"]["start"]),
            end=int(source["frames"]["end"]),
            size=args.window_size,
        )
        for frame in reference_extra
    }

    runtime = _runtime_config(
        source,
        intervals=intervals,
        candidates=pair_candidates,
        windows=pair_windows,
        reference_part=reference,
        output_root=output / "passes" / "simultaneous",
    )
    runtime["automation"]["silhouette_scale_calibration"]["enabled"] = False
    for part in intervals:
        runtime["states"][part]["scale_prior"] = float(
            individual_calibrations[part]["scales"][part]
        )
    runtime_path = output / "runtime" / "anchor_pose_config.json"
    simultaneous_path = (
        output / "passes" / "simultaneous" / "runtime" / "anchor_pose_config.json"
    )
    simultaneous = _run_calibration(
        runtime,
        config_path=simultaneous_path,
        force=args.force_calibration,
    )
    runtime_paths["simultaneous"] = str(simultaneous_path)

    # Merge the global reference anchors from its scale pass with all
    # simultaneous later-part anchors. The merged file is the only calibration
    # consumed by ranking and visualization below.
    calibration = copy.deepcopy(simultaneous)
    calibration["scales"] = {
        part: float(individual_calibrations[part]["scales"][part])
        for part in intervals
    }
    calibration["anchors"][reference].update(
        individual_calibrations[reference]["anchors"][reference]
    )
    calibration.setdefault("table_alignment", {}).setdefault(
        reference, {}
    ).update(
        individual_calibrations[reference]
        .get("table_alignment", {})
        .get(reference, {})
    )
    calibration["scale_passes"] = {
        part: {
            "calibration": str(
                output / "passes" / part / "pose" / "calibration.json"
            ),
            "scale": calibration["scales"][part],
        }
        for part in intervals
    }
    calibration_path = output / "pose" / "calibration.json"
    write_json(calibration_path, calibration)
    pair_runtime_for_review = copy.deepcopy(runtime)
    pair_runtime_for_review["output_root"] = str(output)
    write_json(runtime_path, pair_runtime_for_review)
    write_json(
        output / "diagnostics" / "shortlist.json",
        {
            "schema_version": 1,
            "source_config": str(source_path),
            "state_report": str(state_path),
            "cloud_report": str(cloud_path),
            "intervals": {part: list(value) for part, value in intervals.items()},
            "candidates_for_scale_fit": target_candidates,
            "candidates_for_simultaneous_fit": pair_candidates,
            "windows_for_scale_fit": target_windows,
            "windows_for_simultaneous_fit": pair_windows,
            "rankings": rankings,
            "runtime_configs": runtime_paths,
        },
    )
    evaluations, winners = _evaluate_candidates(
        pair_runtime_for_review,
        calibration,
        target_candidates,
        rankings,
    )
    main_frame = winners[reference]
    if main_frame is None:
        raise RuntimeError("reference part has no anchor passing the eight-view gate")
    table = estimate_local_table_plane(pair_runtime_for_review, main_frame)
    support_validation = {}
    alignment_normal = np.asarray(
        preliminary_table["normal_world"], dtype=np.float64
    )
    alignment_point = np.asarray(
        preliminary_table["point_world"], dtype=np.float64
    )
    for part, frame in winners.items():
        if frame is None:
            continue
        alignment = (
            calibration.get("table_alignment", {})
            .get(part, {})
            .get(str(frame), {})
        )
        transform = np.asarray(
            calibration["anchors"][part][str(frame)]["S_world_from_raw"],
            dtype=np.float64,
        )
        mesh = trimesh.load(
            Path(pair_runtime_for_review["mesh_dir"]) / f"{part}.glb",
            force="mesh",
        )
        vertices = (
            np.asarray(mesh.vertices, dtype=np.float64)
            @ transform[:3, :3].T
            + transform[:3, 3]
        )
        bottom_gap = float(np.quantile(
            (vertices - alignment_point) @ alignment_normal, 0.005
        ))
        support_validation[part] = {
            "frame": int(frame),
            "accepted": bool(
                alignment.get("accepted", False)
                and abs(bottom_gap) <= 0.005
            ),
            "bottom_gap_after_m": bottom_gap,
            "maximum_bottom_gap_after_m": 0.005,
            "alignment": alignment,
        }
    relative = {}
    for part, frame in winners.items():
        if part == reference or frame is None:
            continue
        main_info = calibration["anchors"][reference].get(str(frame), {})
        part_info = calibration["anchors"][part].get(str(frame), {})
        if main_info.get("reliable", False) and part_info.get("reliable", False):
            main_pose = rigid_from_similarity(
                np.asarray(main_info["S_world_from_raw"]),
                np.asarray(calibration["raw_mesh_origins"][reference]),
            )
            part_pose = rigid_from_similarity(
                np.asarray(part_info["S_world_from_raw"]),
                np.asarray(calibration["raw_mesh_origins"][part]),
            )
            relative[part] = {
                "frame": frame,
                "T_reference_from_part": (np.linalg.inv(main_pose) @ part_pose).tolist(),
                "source": "independent_same_frame_multiview_anchor_fits",
            }

    report = {
        "schema_version": 1,
        "method": "stable_shortlist_then_multiview_cloud_scale_render_depth_rank",
        "scope": "anchor_only_no_trajectory_no_isaac",
        "runtime_config": str(runtime_path),
        "calibration": str(calibration_path),
        "intervals": {part: list(value) for part, value in intervals.items()},
        "reference_part": reference,
        "winners": winners,
        "evaluations": evaluations,
        "table_plane": table,
        "anchor_alignment_plane": preliminary_table,
        "support_validation": support_validation,
        "relative_poses": relative,
    }
    report_path = output / "diagnostics" / "anchor_validation.json"
    write_json(report_path, report)

    for part, part_report in evaluations.items():
        review_dir = output / "review" / part
        review_dir.mkdir(parents=True, exist_ok=True)
        for stale in review_dir.glob("rank_*.jpg"):
            stale.unlink()
        (review_dir / "best_failed.jpg").unlink(missing_ok=True)
        for row in part_report["candidates"]:
            if "T_world_from_part" not in row:
                continue
            frame = int(row["frame"])
            path = review_dir / f"rank_{int(row['final_rank']):02d}_{frame:06d}.jpg"
            _render_review_sheet(
                runtime,
                calibration,
                evaluations,
                focus_part=part,
                frame=frame,
                output_path=path,
                width=args.review_width,
                height=args.review_height,
            )
            if frame == winners[part]:
                _render_review_sheet(
                    runtime,
                    calibration,
                    evaluations,
                    focus_part=part,
                    frame=frame,
                    output_path=review_dir / "winner.jpg",
                    width=args.review_width,
                    height=args.review_height,
                )
            if (
                winners[part] is None
                and frame == part_report.get("best_candidate")
            ):
                _render_review_sheet(
                    pair_runtime_for_review,
                    calibration,
                    evaluations,
                    focus_part=part,
                    frame=frame,
                    output_path=review_dir / "best_failed.jpg",
                    width=args.review_width,
                    height=args.review_height,
                )
        winner_path = review_dir / "winner.jpg"
        if winners[part] is None:
            winner_path.unlink(missing_ok=True)
    print(f"winners: {json.dumps(winners, sort_keys=True)}", flush=True)
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
