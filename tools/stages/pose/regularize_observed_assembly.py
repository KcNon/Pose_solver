#!/usr/bin/env python
"""Regularize a visually observed coaxial insertion without inventing screw turns.

The stable assembled interval first supplies one common relative pose from
multiple frames and cameras.  During the preceding insertion, rotation and
radial offset are taken from that validated anchor while only axial position
is estimated from masks.  A small dynamic program selects the smooth,
physically directed 1-DoF path through the per-frame render-loss grid.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.exact_render_refinement import ExactMultiViewRenderObjective
from common.io_utils import load_json, write_json
from common.mesh_render import SceneRenderer
from common.pose_config import validate_pose_config
from common.pose_refinement import sample_canonical
from common.pose_tracking import bridge_pose_ranges
from common.pose_validation import validate_trajectory
from common.render_loss_refinement import MultiViewRenderObjective
from common.trajectory_constraints import (
    _axis_origin_coordinates,
    _set_axis_origin_coordinates,
    axis_vector,
    pairwise_alignment_metrics,
    project_coaxial_pose,
    solve_monotonic_axial_path,
)
from common.trajectory_io import (
    refresh_trajectory_derived_fields,
    write_trajectory_files,
)
from tools.stages.pose.refine_pose_render_loss import load_observations


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_axis_origin(
    relation: dict[str, Any],
    role: str,
    part: str,
    trajectory: dict[str, Any],
) -> np.ndarray:
    metric_key = f"{role}_axis_origin_part_m"
    if metric_key in relation:
        return np.asarray(relation[metric_key], dtype=np.float64)
    raw_key = f"{role}_axis_origin_raw"
    if raw_key not in relation:
        return np.zeros(3, dtype=np.float64)
    return float(trajectory["scales"][part]) * (
        np.asarray(relation[raw_key], dtype=np.float64)
        - np.asarray(trajectory["raw_mesh_origins"][part], dtype=np.float64)
    )


def metric_score(
    metric: dict[str, Any],
    primary_view: str | None,
    primary_weight: float,
) -> float:
    value = float(metric["loss"])
    if primary_view and primary_weight > 0.0:
        row = next(
            (item for item in metric.get("views", []) if item["view"] == primary_view),
            None,
        )
        if row is not None:
            value += float(primary_weight) * float(row["loss"])
    return value


def compact_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metric.get(key)
        for key in (
            "loss",
            "worst_view_loss",
            "mean_iou",
            "worst_view_iou",
            "mean_contour_chamfer_px",
            "mean_target_coverage",
        )
    }


def relative_pose(
    trajectory: dict[str, Any],
    frame: int,
    reference_part: str,
    moving_part: str,
) -> tuple[np.ndarray, np.ndarray]:
    records = trajectory["frames"][f"{frame:06d}"]["parts"]
    reference_world = np.asarray(
        records[reference_part]["T_world_from_part"], dtype=np.float64
    )
    moving_world = np.asarray(
        records[moving_part]["T_world_from_part"], dtype=np.float64
    )
    return np.linalg.inv(reference_world) @ moving_world, reference_world


def candidate_relative_pose(
    anchor_relative: np.ndarray,
    axial_m: float,
    fixed_axis: np.ndarray,
    reference_origin: np.ndarray,
    moving_origin: np.ndarray,
) -> np.ndarray:
    return _set_axis_origin_coordinates(
        anchor_relative,
        fixed_axis,
        reference_origin,
        moving_origin,
        float(axial_m),
        np.zeros(3, dtype=np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output-trajectory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    cfg = validate_pose_config(load_json(args.config), check_paths=True)
    settings = dict(cfg.get("observed_assembly_regularization", {}))
    if not settings.get("enabled", False):
        raise ValueError("observed_assembly_regularization.enabled must be true")
    relations = list(settings.get("relations", []))
    if not relations:
        raise ValueError("observed assembly regularization has no relations")
    baseline = load_json(args.trajectory)
    trajectory = copy.deepcopy(baseline)
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "stable_multiframe_anchor_plus_mask_axial_dp",
        "config": str(args.config.resolve()),
        "trajectory_input": str(args.trajectory.resolve()),
        "trajectory_input_sha256": sha256_file(args.trajectory),
        "relations": {},
    }

    for relation_index, relation_value in enumerate(relations):
        relation = dict(relation_value)
        if relation.get("type", "coaxial_insert") != "coaxial_insert":
            raise ValueError("only coaxial_insert relations are supported")
        name = str(relation.get("name", f"relation_{relation_index}"))
        reference_part = str(relation["reference_part"])
        moving_part = str(relation["moving_part"])
        if reference_part not in trajectory["parts"]:
            raise ValueError(f"{name}: unknown reference part {reference_part}")
        if moving_part not in trajectory["parts"]:
            raise ValueError(f"{name}: unknown moving part {moving_part}")
        start, seat = [int(value) for value in relation["frame_range"]]
        terminal_anchor = int(relation["terminal_anchor_frame"])
        evidence_frames = [
            int(value) for value in relation.get("stable_evidence_frames", [])
        ]
        if not evidence_frames:
            evidence_frames = [terminal_anchor]
        required_frames = set(range(start, seat + 1)) | {
            terminal_anchor,
            *evidence_frames,
        }
        missing = [
            frame
            for frame in sorted(required_frames)
            if f"{frame:06d}" not in trajectory["frames"]
        ]
        if missing:
            raise ValueError(f"{name}: missing trajectory frames {missing[:8]}")

        reference_axis = relation["reference_axis_part"]
        moving_axis = relation["moving_axis_part"]
        fixed_axis = axis_vector(reference_axis)
        reference_origin = canonical_axis_origin(
            relation, "reference", reference_part, baseline
        )
        moving_origin = canonical_axis_origin(
            relation, "moving", moving_part, baseline
        )
        seed_relative, _ = relative_pose(
            baseline, terminal_anchor, reference_part, moving_part
        )
        projected_seed = project_coaxial_pose(
            seed_relative,
            reference_axis=reference_axis,
            moving_axis=moving_axis,
            reference_axis_origin_m=reference_origin,
            moving_axis_origin_m=moving_origin,
            allow_axis_flip=bool(relation.get("allow_axis_flip", False)),
            target_axis_offset_m=float(relation.get("target_axis_offset_m", 0.0)),
        )

        render_cfg = dict(cfg.get("render_loss_refinement", {}))
        render_cfg.update(
            dict(
                cfg.get("render_loss_refinement", {})
                .get("parts", {})
                .get(moving_part, {})
            )
        )
        render_cfg.update(dict(relation.get("render", {})))
        render_cfg["resolution"] = relation.get(
            "resolution", render_cfg.get("resolution", [160, 90])
        )
        render_cfg["use_depth_loss"] = False
        render_cfg["mask_primary"] = True
        primary_view = relation.get("primary_view")
        primary_weight = float(relation.get("primary_view_weight", 0.20))
        raw_mesh = trimesh.load(
            Path(cfg["mesh_dir"]) / f"{moving_part}.glb", force="mesh"
        )
        points = sample_canonical(
            raw_mesh,
            float(baseline["scales"][moving_part]),
            np.asarray(
                baseline["raw_mesh_origins"][moving_part], dtype=np.float64
            ),
            count=int(relation.get("surface_points", 20000)),
            seed=int(relation.get("seed", 2901)) + relation_index,
        )

        observations: dict[int, list[Any]] = {}
        point_objectives: dict[int, MultiViewRenderObjective] = {}
        for frame in sorted(required_frames):
            items = load_observations(
                cfg, f"{frame:06d}", moving_part, render_cfg
            )
            observations[frame] = items
            if items:
                point_objectives[frame] = MultiViewRenderObjective(
                    points, items, render_cfg
                )

        def point_score(candidate: np.ndarray, frames: list[int]) -> tuple[float, list[dict[str, Any]]]:
            values = []
            rows = []
            for frame in frames:
                objective = point_objectives.get(frame)
                if objective is None:
                    continue
                _, reference_world = relative_pose(
                    baseline, frame, reference_part, moving_part
                )
                metric = objective.evaluate(reference_world @ candidate)
                values.append(metric_score(metric, primary_view, primary_weight))
                rows.append({"frame": frame, **compact_metric(metric)})
            return (
                float(np.mean(values)) if values else float("inf"),
                rows,
            )

        anchor_candidates: list[dict[str, Any]] = []
        twist_offsets = [
            float(value)
            for value in relation.get(
                "anchor_twist_offsets_deg", [-30, -20, -10, 0, 10, 20, 30]
            )
        ]
        axial_offsets = [
            float(value)
            for value in relation.get(
                "anchor_axial_offsets_m",
                [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03],
            )
        ]
        for twist_deg in twist_offsets:
            for axial_delta in axial_offsets:
                candidate = project_coaxial_pose(
                    projected_seed,
                    reference_axis=reference_axis,
                    moving_axis=moving_axis,
                    reference_axis_origin_m=reference_origin,
                    moving_axis_origin_m=moving_origin,
                    allow_axis_flip=bool(relation.get("allow_axis_flip", False)),
                    target_axis_offset_m=float(
                        relation.get("target_axis_offset_m", 0.0)
                    ),
                    twist_rad=np.deg2rad(twist_deg),
                    axial_delta_m=axial_delta,
                )
                score, rows = point_score(candidate, evidence_frames)
                anchor_candidates.append({
                    "twist_offset_deg": twist_deg,
                    "axial_offset_m": axial_delta,
                    "point_score": score,
                    "point_frames": rows,
                    "relative_pose": candidate,
                })
        coarse_best = min(anchor_candidates, key=lambda item: item["point_score"])
        local_twist_step = float(relation.get("anchor_local_twist_step_deg", 2.0))
        local_twist_radius = float(
            relation.get("anchor_local_twist_radius_deg", 10.0)
        )
        local_axial_step = float(relation.get("anchor_local_axial_step_m", 0.001))
        local_axial_radius = float(
            relation.get("anchor_local_axial_radius_m", 0.006)
        )
        local_twists = np.arange(
            float(coarse_best["twist_offset_deg"]) - local_twist_radius,
            float(coarse_best["twist_offset_deg"]) + local_twist_radius + 0.5 * local_twist_step,
            local_twist_step,
        )
        local_axials = np.arange(
            float(coarse_best["axial_offset_m"]) - local_axial_radius,
            float(coarse_best["axial_offset_m"]) + local_axial_radius + 0.5 * local_axial_step,
            local_axial_step,
        )
        for twist_deg in local_twists:
            for axial_delta in local_axials:
                candidate = project_coaxial_pose(
                    projected_seed,
                    reference_axis=reference_axis,
                    moving_axis=moving_axis,
                    reference_axis_origin_m=reference_origin,
                    moving_axis_origin_m=moving_origin,
                    allow_axis_flip=bool(relation.get("allow_axis_flip", False)),
                    target_axis_offset_m=float(
                        relation.get("target_axis_offset_m", 0.0)
                    ),
                    twist_rad=np.deg2rad(float(twist_deg)),
                    axial_delta_m=float(axial_delta),
                )
                score, rows = point_score(candidate, evidence_frames)
                anchor_candidates.append({
                    "twist_offset_deg": float(twist_deg),
                    "axial_offset_m": float(axial_delta),
                    "point_score": score,
                    "point_frames": rows,
                    "relative_pose": candidate,
                })

        width, height = [int(value) for value in render_cfg["resolution"]]
        exact_top_k = int(relation.get("anchor_exact_top_k", 12))
        exact_renderer = SceneRenderer(
            width, height, cache_mesh_resources=True
        )
        exact_objectives = {
            frame: ExactMultiViewRenderObjective(
                raw_mesh,
                float(baseline["scales"][moving_part]),
                np.asarray(
                    baseline["raw_mesh_origins"][moving_part], dtype=np.float64
                ),
                observations[frame],
                render_cfg,
                exact_renderer,
            )
            for frame in evidence_frames
            if observations.get(frame)
        }

        def exact_score(candidate: np.ndarray, frames: list[int]) -> tuple[float, list[dict[str, Any]]]:
            values = []
            rows = []
            for frame in frames:
                objective = exact_objectives.get(frame)
                if objective is None:
                    continue
                _, reference_world = relative_pose(
                    baseline, frame, reference_part, moving_part
                )
                metric = objective.evaluate(reference_world @ candidate)
                values.append(metric_score(metric, primary_view, primary_weight))
                rows.append({"frame": frame, **compact_metric(metric)})
            return (
                float(np.mean(values)) if values else float("inf"),
                rows,
            )

        ranked = sorted(anchor_candidates, key=lambda item: item["point_score"])
        exact_candidates = ranked[: max(exact_top_k, 1)]
        if not any(
            abs(float(item["twist_offset_deg"])) < 1e-9
            and abs(float(item["axial_offset_m"])) < 1e-9
            for item in exact_candidates
        ):
            seed_item = min(
                anchor_candidates,
                key=lambda item: abs(float(item["twist_offset_deg"]))
                + 1000.0 * abs(float(item["axial_offset_m"])),
            )
            exact_candidates.append(seed_item)
        for item in exact_candidates:
            score, rows = exact_score(item["relative_pose"], evidence_frames)
            item["exact_score"] = score
            item["exact_frames"] = rows
        anchor_best = min(exact_candidates, key=lambda item: item["exact_score"])
        anchor_relative = np.asarray(
            anchor_best["relative_pose"], dtype=np.float64
        )
        anchor_axial, _ = _axis_origin_coordinates(
            anchor_relative, fixed_axis, reference_origin, moving_origin
        )

        dynamic_frames = list(range(start, seat + 1))
        start_relative, _ = relative_pose(
            baseline, start, reference_part, moving_part
        )
        start_projected = project_coaxial_pose(
            start_relative,
            reference_axis=reference_axis,
            moving_axis=moving_axis,
            reference_axis_origin_m=reference_origin,
            moving_axis_origin_m=moving_origin,
            allow_axis_flip=bool(relation.get("allow_axis_flip", False)),
            target_axis_offset_m=float(relation.get("target_axis_offset_m", 0.0)),
        )
        start_axial, _ = _axis_origin_coordinates(
            start_projected, fixed_axis, reference_origin, moving_origin
        )
        direction = float(
            relation.get(
                "insertion_direction",
                1.0 if anchor_axial >= start_axial else -1.0,
            )
        )
        grid_step = float(relation.get("axial_grid_step_m", 0.002))
        grid_margin = float(relation.get("axial_grid_margin_m", 0.015))
        low = min(start_axial, anchor_axial) - grid_margin
        high = max(start_axial, anchor_axial) + grid_margin
        axial_grid = np.arange(low, high + 0.5 * grid_step, grid_step)
        axial_grid = np.unique(np.sort(np.append(axial_grid, anchor_axial)))
        terminal_index = int(np.argmin(np.abs(axial_grid - anchor_axial)))
        axial_grid[terminal_index] = anchor_axial
        unary = np.empty((len(dynamic_frames), len(axial_grid)), dtype=np.float64)
        candidate_metrics: list[list[dict[str, Any]]] = []
        for frame_index, frame in enumerate(dynamic_frames):
            objective = point_objectives.get(frame)
            frame_metrics = []
            _, reference_world = relative_pose(
                baseline, frame, reference_part, moving_part
            )
            for candidate_index, axial in enumerate(axial_grid):
                candidate = candidate_relative_pose(
                    anchor_relative,
                    float(axial),
                    fixed_axis,
                    reference_origin,
                    moving_origin,
                )
                if objective is None:
                    metric = {"loss": 0.0, "mean_iou": None, "views": []}
                    unary[frame_index, candidate_index] = 0.0
                else:
                    metric = objective.evaluate(reference_world @ candidate)
                    unary[frame_index, candidate_index] = metric_score(
                        metric, primary_view, primary_weight
                    )
                frame_metrics.append(metric)
            candidate_metrics.append(frame_metrics)
        path_indices, path_cost = solve_monotonic_axial_path(
            unary,
            axial_grid,
            direction=direction,
            maximum_step_m=float(relation.get("maximum_axial_step_m", 0.008)),
            maximum_backtrack_m=float(
                relation.get("maximum_axial_backtrack_m", 0.001)
            ),
            temporal_weight=float(relation.get("axial_temporal_weight", 0.08)),
            terminal_index=terminal_index,
        )
        selected_axial = axial_grid[path_indices]
        dynamic_rows = []
        for index, frame in enumerate(dynamic_frames):
            frame_id = f"{frame:06d}"
            records = trajectory["frames"][frame_id]["parts"]
            reference_world = np.asarray(
                records[reference_part]["T_world_from_part"], dtype=np.float64
            )
            selected_relative = candidate_relative_pose(
                anchor_relative,
                float(selected_axial[index]),
                fixed_axis,
                reference_origin,
                moving_origin,
            )
            record = records[moving_part]
            record["T_world_from_part"] = (
                reference_world @ selected_relative
            ).tolist()
            record["source"] = (
                str(record.get("source", "pose"))
                + "+observed_coaxial_insert_dp"
            )
            chosen_metric = candidate_metrics[index][int(path_indices[index])]
            mean_iou = chosen_metric.get("mean_iou")
            confidence = (
                None
                if mean_iou is None
                else float(np.clip((float(mean_iou) - 0.15) / 0.55, 0.0, 1.0))
            )
            record["pose_source"] = "mask_render+coaxial_insert_dp"
            record["pose_confidence"] = confidence
            record["observability"] = (
                "observed_constrained"
                if len(observations.get(frame, [])) >= 4
                and mean_iou is not None
                and float(mean_iou) >= 0.30
                else "occluded_constrained"
            )
            dynamic_rows.append({
                "frame": frame,
                "axial_m": float(selected_axial[index]),
                "candidate_index": int(path_indices[index]),
                "available_views": len(observations.get(frame, [])),
                "point_metric": compact_metric(chosen_metric),
                "pose_confidence": confidence,
            })

        follow_start, follow_end = [
            int(value)
            for value in relation.get(
                "static_follow_range", [seat + 1, terminal_anchor]
            )
        ]
        followed_frames = []
        for frame in range(follow_start, follow_end + 1):
            frame_id = f"{frame:06d}"
            if frame_id not in trajectory["frames"]:
                continue
            records = trajectory["frames"][frame_id]["parts"]
            reference_world = np.asarray(
                records[reference_part]["T_world_from_part"], dtype=np.float64
            )
            record = records[moving_part]
            record["T_world_from_part"] = (
                reference_world @ anchor_relative
            ).tolist()
            record["source"] = (
                str(record.get("source", "pose"))
                + "+validated_assembly_anchor_follow"
            )
            record["pose_source"] = "validated_multiframe_assembly_anchor"
            record["pose_confidence"] = 1.0
            record["observability"] = "observed_static_anchor"
            followed_frames.append(frame)

        entry_bridge_report = None
        entry_bridge = relation.get("entry_bridge_range")
        if entry_bridge is not None:
            bridge_start, bridge_end = [int(value) for value in entry_bridge]
            pose_values = {
                int(frame_id): np.asarray(
                    row["parts"][moving_part]["T_world_from_part"],
                    dtype=np.float64,
                )
                for frame_id, row in trajectory["frames"].items()
            }
            entry_bridge_report = bridge_pose_ranges(
                pose_values, [[bridge_start, bridge_end]]
            )[0]
            for frame in range(bridge_start, bridge_end + 1):
                record = trajectory["frames"][f"{frame:06d}"]["parts"][
                    moving_part
                ]
                record["T_world_from_part"] = pose_values[frame].tolist()
                record["source"] = (
                    str(record.get("source", "pose"))
                    + "+observed_assembly_entry_bridge"
                )
                record["pose_source"] = "se3_bridge_between_visual_branches"
                record["observability"] = "occluded_bridged"

        diagnostic_frames = sorted({
            start,
            seat,
            terminal_anchor,
            *evidence_frames,
            *[
                int(value)
                for value in relation.get(
                    "diagnostic_frames", [290, 300, 320, 340, 350, 355]
                )
            ],
        })
        exact_diagnostics = {}
        for frame in diagnostic_frames:
            if frame not in required_frames:
                items = load_observations(
                    cfg, f"{frame:06d}", moving_part, render_cfg
                )
                if not items:
                    continue
                exact_objective = ExactMultiViewRenderObjective(
                    raw_mesh,
                    float(baseline["scales"][moving_part]),
                    np.asarray(
                        baseline["raw_mesh_origins"][moving_part], dtype=np.float64
                    ),
                    items,
                    render_cfg,
                    exact_renderer,
                )
            else:
                if not observations.get(frame):
                    continue
                exact_objective = exact_objectives.get(frame)
                if exact_objective is None:
                    exact_objective = ExactMultiViewRenderObjective(
                        raw_mesh,
                        float(baseline["scales"][moving_part]),
                        np.asarray(
                            baseline["raw_mesh_origins"][moving_part],
                            dtype=np.float64,
                        ),
                        observations[frame],
                        render_cfg,
                        exact_renderer,
                    )
            base_relative, reference_world = relative_pose(
                baseline, frame, reference_part, moving_part
            )
            selected_relative, selected_reference_world = relative_pose(
                trajectory, frame, reference_part, moving_part
            )
            baseline_metric = exact_objective.evaluate(
                reference_world @ base_relative
            )
            selected_metric = exact_objective.evaluate(
                selected_reference_world @ selected_relative
            )
            exact_diagnostics[f"{frame:06d}"] = {
                "baseline": compact_metric(baseline_metric),
                "selected": compact_metric(selected_metric),
                "loss_improvement": float(
                    baseline_metric["loss"] - selected_metric["loss"]
                ),
                "iou_improvement": float(
                    selected_metric["mean_iou"] - baseline_metric["mean_iou"]
                ),
            }
        exact_renderer.close()
        alignment = pairwise_alignment_metrics(
            anchor_relative,
            reference_axis=reference_axis,
            moving_axis=moving_axis,
            allow_axis_flip=bool(relation.get("allow_axis_flip", False)),
            reference_axis_origin_m=reference_origin,
            moving_axis_origin_m=moving_origin,
        )
        report["relations"][name] = {
            "type": "coaxial_insert",
            "reference_part": reference_part,
            "moving_part": moving_part,
            "frame_range": [start, seat],
            "terminal_anchor_frame": terminal_anchor,
            "stable_evidence_frames": evidence_frames,
            "primary_view": primary_view,
            "anchor_search": {
                "candidate_count": len(anchor_candidates),
                "exact_candidate_count": len(exact_candidates),
                "selected_twist_offset_deg": float(
                    anchor_best["twist_offset_deg"]
                ),
                "selected_axial_offset_m": float(
                    anchor_best["axial_offset_m"]
                ),
                "selected_point_score": float(anchor_best["point_score"]),
                "selected_exact_score": float(anchor_best["exact_score"]),
                "selected_exact_frames": anchor_best["exact_frames"],
                "alignment": alignment,
                "anchor_axial_m": float(anchor_axial),
            },
            "axial_dp": {
                "grid_min_m": float(axial_grid[0]),
                "grid_max_m": float(axial_grid[-1]),
                "grid_count": int(len(axial_grid)),
                "direction": 1.0 if direction >= 0.0 else -1.0,
                "start_axial_m": float(start_axial),
                "terminal_axial_m": float(anchor_axial),
                "path_cost": path_cost,
                "frames": dynamic_rows,
            },
            "static_follow_range": [follow_start, follow_end],
            "static_follow_frame_count": len(followed_frames),
            "entry_bridge": entry_bridge_report,
            "exact_diagnostics": exact_diagnostics,
        }

    refresh_trajectory_derived_fields(trajectory)
    validation, failures = validate_trajectory(
        cfg, trajectory, enforce_assembly=False
    )
    report["trajectory_validation"] = validation
    report["validation_passed"] = not failures
    if failures:
        report["failures"] = failures
        write_json(args.report, report)
        raise RuntimeError("; ".join(failures))
    trajectory.setdefault("refinements", []).append({
        "method": report["method"],
        "input": str(args.trajectory.resolve()),
        "report": str(args.report.resolve()),
    })
    write_trajectory_files(trajectory, args.output_trajectory)
    report["trajectory_output"] = str(args.output_trajectory.resolve())
    report["trajectory_output_sha256"] = sha256_file(args.output_trajectory)
    write_json(args.report, report)
    print(f"trajectory -> {args.output_trajectory}", flush=True)
    print(f"report -> {args.report}", flush=True)


if __name__ == "__main__":
    main()
