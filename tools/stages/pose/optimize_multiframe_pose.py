#!/usr/bin/env python
"""Jointly regularize pose windows with evidence from multiple frames/views.

Static windows select one visually supported absolute or relative pose.
Dynamic windows solve a render-loss/SE(3) candidate lattice jointly.  Relation
windows can further reduce motion to a semantic mechanism such as coaxial
insertion. Bridge windows explicitly mark intervals that have no trustworthy
per-frame observation and interpolate only between their independently solved
neighbours. The old observed-assembly config key is accepted as a compatibility
alias, but no dataset or part name is embedded in the implementation.
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
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.exact_render_refinement import ExactMultiViewRenderObjective
from common.io_utils import load_json, write_json
from common.mesh_render import SceneRenderer
from common.multiframe_pose import (
    boundary_candidate_mask,
    candidate_pose,
    evenly_spaced_frames,
    internal_continuity_gate,
    multiframe_settings,
    parallel_transport_orientations,
    pose_distance,
    solve_pose_candidate_path,
    static_candidate_scores,
    world_pose,
)
from common.pose_config import validate_pose_config
from common.pose_refinement import sample_canonical
from common.pose_tracking import bridge_pose_ranges
from common.pose_validation import validate_trajectory
from common.render_loss_refinement import (
    MultiViewRenderObjective,
    refine_pose_coordinate_search,
)
from common.trajectory_constraints import (
    _axis_origin_coordinates,
    _set_axis_origin_coordinates,
    axis_vector,
    interpolate_pose,
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


def _bridge_window(
    baseline: dict[str, Any],
    trajectory: dict[str, Any],
    window: dict[str, Any],
) -> dict[str, Any]:
    """Bridge a declared unobservable interval without claiming observations."""

    part = str(window["part"])
    reference_part = window.get("reference_part")
    if reference_part is not None:
        reference_part = str(reference_part)
    start, end = [int(value) for value in window["frame_range"]]
    left, right = start - 1, end + 1
    for frame in (left, right):
        if f"{frame:06d}" not in baseline["frames"]:
            raise ValueError(
                f"bridge window {start}..{end} requires endpoint frame {frame}"
            )

    apply_bridge = bool(window.get("apply", True))
    poses: dict[int, np.ndarray] = {}
    for frame in range(left, right + 1):
        frame_id = f"{frame:06d}"
        records = baseline["frames"][frame_id]["parts"]
        moving_world = np.asarray(
            records[part]["T_world_from_part"], dtype=np.float64
        )
        if reference_part is None:
            poses[frame] = moving_world
        else:
            reference_world = np.asarray(
                records[reference_part]["T_world_from_part"], dtype=np.float64
            )
            poses[frame] = np.linalg.inv(reference_world) @ moving_world
    bridge_report = bridge_pose_ranges(poses, [[start, end]])[0]

    for frame in range(start, end + 1):
        frame_id = f"{frame:06d}"
        records = trajectory["frames"][frame_id]["parts"]
        record = records[part]
        if apply_bridge:
            bridged = poses[frame]
            if reference_part is not None:
                reference_world = np.asarray(
                    records[reference_part]["T_world_from_part"],
                    dtype=np.float64,
                )
                bridged = reference_world @ bridged
            record["T_world_from_part"] = bridged.tolist()
            record["source"] = (
                str(record.get("source", "pose"))
                + "+declared_occlusion_bridge"
            )
            record["pose_source"] = "se3_bridge_between_observed_endpoints"
        else:
            record["pose_source"] = "unverified_pose_in_declared_occlusion"
        record["pose_confidence"] = None
        record["observability"] = (
            "occluded_bridged" if apply_bridge else "occluded_unverified"
        )
    return {
        "mode": "bridge_window",
        "part": part,
        "reference_part": reference_part,
        "frame_range": [start, end],
        "endpoint_frames": [left, right],
        "method": (
            "reference_relative_se3_bridge"
            if reference_part is not None
            else "world_se3_bridge"
        ),
        "interpolated_frame_count": end - start + 1,
        "applied": apply_bridge,
        "transform_preserved": not apply_bridge,
        "bridge": bridge_report,
    }


def _coaxial_snap_window(
    cfg: dict[str, Any],
    baseline: dict[str, Any],
    trajectory: dict[str, Any],
    window: dict[str, Any],
) -> dict[str, Any]:
    """Deterministically seat a connector without an expensive render search."""

    reference_part = str(window["reference_part"])
    moving_part = str(window["moving_part"])
    start, seat = map(int, window["frame_range"])
    terminal = int(window["terminal_anchor_frame"])
    reference_axis = axis_vector(window["reference_axis_part"])
    moving_axis = axis_vector(window["moving_axis_part"])
    reference_origin = canonical_axis_origin(
        window, "reference", reference_part, baseline
    )
    moving_origin = canonical_axis_origin(
        window, "moving", moving_part, baseline
    )
    terminal_relative, _ = relative_pose(
        baseline, terminal, reference_part, moving_part
    )
    terminal_axial, _ = _axis_origin_coordinates(
        terminal_relative, reference_axis, reference_origin, moving_origin
    )
    if "target_axial_offset_m" in window:
        axial_delta = float(window["target_axial_offset_m"]) - float(
            terminal_axial
        )
    else:
        axial_delta = float(window.get("terminal_axial_delta_m", 0.0))
    twist_offsets = [
        float(value)
        for value in window.get("twist_search_offsets_deg", [0.0])
    ]
    if not twist_offsets:
        twist_offsets = [0.0]

    def projected(twist_deg: float) -> np.ndarray:
        return project_coaxial_pose(
            terminal_relative,
            reference_axis=reference_axis,
            moving_axis=moving_axis,
            reference_axis_origin_m=reference_origin,
            moving_axis_origin_m=moving_origin,
            allow_axis_flip=bool(window.get("allow_axis_flip", False)),
            target_axis_offset_m=float(window.get("target_axis_offset_m", 0.0)),
            twist_rad=float(np.deg2rad(twist_deg)),
            axial_delta_m=axial_delta,
        )

    twist_rows = []
    target_relative = projected(twist_offsets[0])
    if len(twist_offsets) > 1:
        evidence_frames = [
            int(value)
            for value in window.get("twist_evidence_frames", [terminal])
        ]
        _, _, observations, objectives = _window_objectives(
            cfg, baseline, window, moving_part, evidence_frames
        )
        primary_view = window.get("primary_view")
        primary_weight = float(window.get("primary_view_weight", 0.0))
        candidates = []
        for twist_deg in twist_offsets:
            candidate = projected(twist_deg)
            scores = []
            frame_rows = []
            for frame in evidence_frames:
                objective = objectives.get(frame)
                if objective is None:
                    continue
                _, reference_world = relative_pose(
                    baseline, frame, reference_part, moving_part
                )
                metric = objective.evaluate(reference_world @ candidate)
                score = metric_score(metric, primary_view, primary_weight)
                scores.append(score)
                frame_rows.append({"frame": frame, **compact_metric(metric)})
            mean_score = float(np.mean(scores)) if scores else float("inf")
            row = {
                "twist_offset_deg": twist_deg,
                "score": mean_score,
                "frames": frame_rows,
                "candidate": candidate,
            }
            candidates.append(row)
        selected = min(candidates, key=lambda row: row["score"])
        if not np.isfinite(float(selected["score"])):
            raise ValueError("coaxial twist search has no renderable evidence")
        target_relative = np.asarray(selected["candidate"], dtype=np.float64)
        twist_rows = [
            {key: value for key, value in row.items() if key != "candidate"}
            for row in candidates
        ]
        selected_twist = float(selected["twist_offset_deg"])
    else:
        evidence_frames = []
        selected_twist = float(twist_offsets[0])

    def metrics(value: np.ndarray) -> dict[str, float | None]:
        result = pairwise_alignment_metrics(
            value,
            reference_axis=reference_axis,
            moving_axis=moving_axis,
            allow_axis_flip=bool(window.get("allow_axis_flip", False)),
            reference_axis_origin_m=reference_origin,
            moving_axis_origin_m=moving_origin,
        )
        axial, _ = _axis_origin_coordinates(
            value, reference_axis, reference_origin, moving_origin
        )
        result["axial_offset_m"] = float(axial)
        return result

    apply_snap = bool(window.get("apply", True))
    dynamic_rows = []
    denominator = max(seat - start, 1)
    for frame in range(start, seat + 1):
        current, _ = relative_pose(
            baseline, frame, reference_part, moving_part
        )
        fraction = float((frame - start) / denominator)
        # Smoothly distribute only the terminal connector correction.  The
        # observed free-motion centre and tilt remain authoritative at entry.
        amount = fraction * fraction * (3.0 - 2.0 * fraction)
        selected = interpolate_pose(current, target_relative, amount)
        if apply_snap:
            records = trajectory["frames"][f"{frame:06d}"]["parts"]
            reference_world = np.asarray(
                records[reference_part]["T_world_from_part"], dtype=np.float64
            )
            record = records[moving_part]
            record["T_world_from_part"] = (
                reference_world @ selected
            ).tolist()
            record["source"] = str(record.get("source", "pose")) + "+coaxial_snap"
            record["pose_source"] = "mechanism_coaxial_snap"
        dynamic_rows.append({
            "frame": frame,
            "correction_fraction": amount,
            "before": metrics(current),
            "after": metrics(selected),
        })

    follow_start, follow_end = map(
        int, window.get("static_follow_range", [terminal, terminal])
    )
    followed = []
    if apply_snap:
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
                reference_world @ target_relative
            ).tolist()
            record["source"] = str(record.get("source", "pose")) + "+coaxial_seated"
            record["pose_source"] = "mechanism_coaxial_seated"
            followed.append(frame)
    return {
        "mode": "coaxial_snap_window",
        "reference_part": reference_part,
        "moving_part": moving_part,
        "frame_range": [start, seat],
        "terminal_anchor_frame": terminal,
        "target_axial_offset_m": window.get("target_axial_offset_m"),
        "applied_axial_delta_m": axial_delta,
        "twist_evidence_frames": evidence_frames,
        "twist_candidates": twist_rows,
        "selected_twist_offset_deg": selected_twist,
        "static_follow_range": [follow_start, follow_end],
        "terminal_before": metrics(terminal_relative),
        "terminal_after": metrics(target_relative),
        "dynamic_frames": dynamic_rows,
        "followed_frame_count": len(followed),
        "applied": apply_snap,
    }


def _orientation_transport_window(
    baseline: dict[str, Any],
    trajectory: dict[str, Any],
    window: dict[str, Any],
) -> dict[str, Any]:
    """Remove accumulated connector-axis twist while preserving tilt and centres."""

    part = str(window["part"])
    reference_part = (
        str(window["reference_part"])
        if window.get("reference_part") is not None
        else None
    )
    start, end = map(int, window["frame_range"])
    seed = int(window.get("seed_frame", end))
    axis = axis_vector(window["moving_axis_part"])
    poses: dict[int, np.ndarray] = {}
    for frame in range(start, end + 1):
        records = baseline["frames"][f"{frame:06d}"]["parts"]
        moving = np.asarray(records[part]["T_world_from_part"], dtype=np.float64)
        if reference_part is None:
            poses[frame] = moving
        else:
            reference = np.asarray(
                records[reference_part]["T_world_from_part"], dtype=np.float64
            )
            poses[frame] = np.linalg.inv(reference) @ moving
    lock_full_orientation = bool(window.get("lock_full_orientation", False))
    if lock_full_orientation:
        lock_start = int(window.get("lock_start_frame", start))
        if lock_start < start or lock_start > end:
            raise ValueError("lock_start_frame must lie inside frame_range")
        seed_rotation = poses[seed][:3, :3]
        start_rotation = poses[start][:3, :3]
        transported = {}
        denominator = max(lock_start - start, 1)
        for frame in range(start, end + 1):
            if frame >= lock_start:
                transported[frame] = seed_rotation.copy()
                continue
            fraction = float((frame - start) / denominator)
            amount = fraction * fraction * (3.0 - 2.0 * fraction)
            initial = np.eye(4, dtype=np.float64)
            # Use one trusted pickup orientation as the left endpoint.  Using
            # each noisy per-frame ICP rotation here would only attenuate the
            # jitter; it would not produce a continuous orientation path.
            initial[:3, :3] = start_rotation
            target = np.eye(4, dtype=np.float64)
            target[:3, :3] = seed_rotation
            transported[frame] = interpolate_pose(
                initial, target, amount
            )[:3, :3]
        method = "trusted_endpoint_full_orientation_lock_with_pickup_blend"
    else:
        lock_start = None
        transported = parallel_transport_orientations(
            {frame: pose[:3, :3] for frame, pose in poses.items()},
            axis,
            seed_frame=seed,
        )
        method = "minimum_rotation_parallel_transport_from_trusted_seed"
    apply_transport = bool(window.get("apply", True))
    rows = []
    for frame in range(start, end + 1):
        original = poses[frame]
        corrected = original.copy()
        corrected[:3, :3] = transported[frame]
        axis_error = float(np.degrees(np.arccos(np.clip(
            np.dot(original[:3, :3] @ axis, corrected[:3, :3] @ axis),
            -1.0,
            1.0,
        ))))
        adjustment = Rotation.from_matrix(
            corrected[:3, :3] @ original[:3, :3].T
        )
        if apply_transport:
            records = trajectory["frames"][f"{frame:06d}"]["parts"]
            selected_world = corrected
            if reference_part is not None:
                reference = np.asarray(
                    records[reference_part]["T_world_from_part"], dtype=np.float64
                )
                selected_world = reference @ corrected
            record = records[part]
            record["T_world_from_part"] = selected_world.tolist()
            record["source"] = (
                str(record.get("source", "pose")) + "+axis_parallel_transport"
            )
            record["pose_source"] = "connector_axis_parallel_transport"
        rows.append({
            "frame": frame,
            "rotation_adjustment_deg": float(np.degrees(adjustment.magnitude())),
            "connector_axis_direction_error_deg": axis_error,
            "center_preserved": bool(np.array_equal(
                original[:3, 3], corrected[:3, 3]
            )),
        })
    return {
        "mode": "orientation_transport_window",
        "part": part,
        "reference_part": reference_part,
        "frame_range": [start, end],
        "seed_frame": seed,
        "method": method,
        "lock_full_orientation": lock_full_orientation,
        "lock_start_frame": lock_start,
        "maximum_rotation_adjustment_deg": max(
            row["rotation_adjustment_deg"] for row in rows
        ),
        "maximum_connector_axis_direction_error_deg": max(
            row["connector_axis_direction_error_deg"] for row in rows
        ),
        "translations_changed": False,
        "frames": rows,
        "applied": apply_transport,
    }


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


def _window_render_config(
    cfg: dict[str, Any],
    window: dict[str, Any],
    part: str,
) -> dict[str, Any]:
    render_cfg = dict(cfg.get("render_loss_refinement", {}))
    render_cfg.update(dict(
        cfg.get("render_loss_refinement", {})
        .get("parts", {})
        .get(part, {})
    ))
    render_cfg.update(dict(window.get("render", {})))
    render_cfg["resolution"] = window.get(
        "resolution", render_cfg.get("resolution", [160, 90])
    )
    render_cfg["use_depth_loss"] = bool(
        window.get("use_depth_loss", render_cfg.get("use_depth_loss", False))
    )
    render_cfg["mask_primary"] = True
    return render_cfg


def _observed_frames(
    trajectory: dict[str, Any],
    part: str,
    start: int,
    end: int,
    minimum_views: int,
) -> list[int]:
    frames = []
    for frame in range(start, end + 1):
        frame_id = f"{frame:06d}"
        if frame_id not in trajectory["frames"]:
            continue
        record = trajectory["frames"][frame_id]["parts"][part]
        if (
            record.get("pose_valid", True) is not False
            and int(record.get("observing_views", 0)) >= minimum_views
        ):
            frames.append(frame)
    return frames


def _window_objectives(
    cfg: dict[str, Any],
    trajectory: dict[str, Any],
    window: dict[str, Any],
    part: str,
    frames: list[int],
) -> tuple[
    dict[str, Any],
    trimesh.Trimesh,
    dict[int, list[Any]],
    dict[int, MultiViewRenderObjective],
]:
    render_cfg = _window_render_config(cfg, window, part)
    raw_mesh = trimesh.load(
        Path(cfg["mesh_dir"]) / f"{part}.glb", force="mesh"
    )
    points = sample_canonical(
        raw_mesh,
        float(trajectory["scales"][part]),
        np.asarray(trajectory["raw_mesh_origins"][part], dtype=np.float64),
        count=int(window.get("surface_points", 16000)),
        seed=int(window.get("seed", 1709)),
    )
    observations = {}
    objectives = {}
    for frame in frames:
        items = load_observations(
            cfg, f"{frame:06d}", part, render_cfg
        )
        observations[frame] = items
        if items:
            objectives[frame] = MultiViewRenderObjective(
                points, items, render_cfg
            )
    return render_cfg, raw_mesh, observations, objectives


class AggregateStaticObjective:
    """Evaluate one reference-relative pose across multiple timestamps."""

    def __init__(
        self,
        trajectory: dict[str, Any],
        *,
        part: str,
        reference_part: str | None,
        objectives: dict[int, Any],
        optimize_frames: list[int],
        holdout_frames: list[int],
    ) -> None:
        self.trajectory = trajectory
        self.part = part
        self.reference_part = reference_part
        self.objectives = objectives
        self.optimize_frames = list(optimize_frames)
        self.holdout_frames = list(holdout_frames)
        # ``refine_pose_coordinate_search`` routes evidence by view name.
        # These synthetic names instead route disjoint timestamp groups;
        # every per-frame objective still evaluates all of its real cameras.
        self.by_view = {"optimize_frames": True}
        if self.holdout_frames:
            self.by_view["holdout_frames"] = True

    def evaluate(
        self,
        pose: np.ndarray,
        views: list[str] | None = None,
    ) -> dict[str, Any]:
        requested = set(views or self.by_view)
        frames = []
        if "optimize_frames" in requested:
            frames.extend(self.optimize_frames)
        if "holdout_frames" in requested:
            frames.extend(self.holdout_frames)
        rows = []
        view_rows = []
        for frame in frames:
            metric = self.objectives[frame].evaluate(world_pose(
                self.trajectory,
                frame=frame,
                pose=np.asarray(pose, dtype=np.float64),
                reference_part=self.reference_part,
            ))
            rows.append(metric)
            view_rows.extend([
                {**dict(item), "frame": int(frame)}
                for item in metric.get("views", [])
            ])
        if not rows:
            return {
                "loss": float("inf"),
                "worst_view_loss": float("inf"),
                "mean_iou": 0.0,
                "worst_view_iou": 0.0,
                "mean_contour_chamfer_px": float("inf"),
                "mean_target_coverage": 0.0,
                "views": [],
                "frames": [],
            }
        return {
            "loss": float(np.mean([row["loss"] for row in rows])),
            "worst_view_loss": float(max(
                row.get("worst_view_loss", row["loss"]) for row in rows
            )),
            "mean_iou": float(np.mean([
                row.get("mean_iou", 0.0) for row in rows
            ])),
            "worst_view_iou": float(min(
                row.get("worst_view_iou", row.get("mean_iou", 0.0))
                for row in rows
            )),
            "mean_contour_chamfer_px": float(np.mean([
                row.get("mean_contour_chamfer_px", 0.0) for row in rows
            ])),
            "mean_target_coverage": float(np.mean([
                row.get("mean_target_coverage", 0.0) for row in rows
            ])),
            "views": view_rows,
            "frames": frames,
        }


def _refine_constant_static_pose(
    cfg: dict[str, Any],
    baseline: dict[str, Any],
    trajectory: dict[str, Any],
    window: dict[str, Any],
    *,
    part: str,
    reference_part: str | None,
    start: int,
    end: int,
    evidence: list[int],
) -> dict[str, Any]:
    render_cfg, raw_mesh, observations, _ = _window_objectives(
        cfg, baseline, window, part, evidence
    )
    usable = [frame for frame in evidence if observations.get(frame)]
    if not usable:
        return {
            "mode": "static_window",
            "part": part,
            "reference_part": reference_part,
            "frame_range": [start, end],
            "applied": False,
            "reason": "no_renderable_evidence",
        }
    configured_holdout = [
        int(value) for value in window.get("constant_holdout_frames", [])
        if int(value) in usable
    ]
    holdout = (
        configured_holdout
        if configured_holdout
        else [usable[-1]] if len(usable) >= 3 else []
    )
    holdout_set = set(holdout)
    optimize = [frame for frame in usable if frame not in holdout_set]
    if not optimize:
        optimize = usable
        holdout = []
    width, height = [int(value) for value in render_cfg["resolution"]]
    renderer = SceneRenderer(width, height, cache_mesh_resources=True)
    try:
        exact = {
            frame: ExactMultiViewRenderObjective(
                raw_mesh,
                float(baseline["scales"][part]),
                np.asarray(
                    baseline["raw_mesh_origins"][part], dtype=np.float64
                ),
                observations[frame],
                render_cfg,
                renderer,
            )
            for frame in usable
        }
        objective = AggregateStaticObjective(
            baseline,
            part=part,
            reference_part=reference_part,
            objectives=exact,
            optimize_frames=optimize,
            holdout_frames=holdout,
        )
        initial = candidate_pose(
            baseline,
            frame=usable[0],
            part=part,
            reference_part=reference_part,
        )
        selected, refinement = refine_pose_coordinate_search(
            objective,
            initial,
            optimize_views=["optimize_frames"],
            holdout_views=["holdout_frames"] if holdout else [],
            translation_steps_m=[
                float(value) for value in window.get(
                    "constant_translation_steps_m",
                    [0.015, 0.008, 0.004, 0.002],
                )
            ],
            rotation_steps_deg=[
                float(value) for value in window.get(
                    "constant_rotation_steps_deg", [2.0, 1.0]
                )
            ],
            symmetry_axis_part=None,
            optimize_rotation=bool(
                window.get("constant_optimize_rotation", False)
            ),
            maximum_translation_delta_m=float(
                window.get("constant_maximum_translation_delta_m", 0.03)
            ),
            maximum_rotation_delta_deg=float(
                window.get("constant_maximum_rotation_delta_deg", 5.0)
            ),
            minimum_improvement=float(
                window.get("constant_minimum_improvement", 0.005)
            ),
            maximum_holdout_degradation=float(
                window.get("constant_maximum_holdout_degradation", 0.02)
            ),
            minimum_refined_iou=float(
                window.get("constant_minimum_refined_iou", 0.02)
            ),
            prior_weight=float(window.get("constant_prior_weight", 0.002)),
            temporal_weight=0.0,
        )
    finally:
        renderer.close()

    applied_frames = []
    if refinement["accepted"]:
        for frame in range(start, end + 1):
            frame_id = f"{frame:06d}"
            if frame_id not in trajectory["frames"]:
                continue
            record = trajectory["frames"][frame_id]["parts"][part]
            record["T_world_from_part"] = world_pose(
                trajectory,
                frame=frame,
                pose=selected,
                reference_part=reference_part,
            ).tolist()
            record["source"] = (
                str(record.get("source", "pose"))
                + "+constant_static_exact_refinement"
            )
            record["pose_source"] = "constant_static_exact_refinement"
            applied_frames.append(frame)
    return {
        "mode": "static_window",
        "part": part,
        "reference_part": reference_part,
        "frame_range": [start, end],
        "evidence_frames": usable,
        "optimize_frames": optimize,
        "holdout_frames": holdout,
        "constant_pose_refinement": refinement,
        "applied": bool(refinement["accepted"]),
        "reason": None if refinement["accepted"] else "local_visual_gate_failed",
        "applied_frame_count": len(applied_frames),
    }


def _static_window(
    cfg: dict[str, Any],
    baseline: dict[str, Any],
    trajectory: dict[str, Any],
    window: dict[str, Any],
) -> dict[str, Any]:
    name = str(window["name"])
    part = str(window["part"])
    reference_part = (
        str(window["reference_part"])
        if window.get("reference_part") is not None
        else None
    )
    start, end = map(int, window["frame_range"])
    minimum_views = int(window.get("minimum_observing_views", 2))
    observed = _observed_frames(
        baseline, part, start, end, minimum_views
    )
    evidence = [int(value) for value in window.get("evidence_frames", [])]
    if not evidence:
        evidence = evenly_spaced_frames(
            observed, int(window.get("maximum_evidence_frames", 5))
        )
    candidates = [
        int(value) for value in window.get("candidate_frames", [])
    ]
    if not candidates:
        candidates = evenly_spaced_frames(
            observed, int(window.get("maximum_candidate_frames", 9))
        )
    if not evidence or not candidates:
        return {
            "mode": "static_window",
            "part": part,
            "frame_range": [start, end],
            "applied": False,
            "reason": "no_reliable_observed_frames",
        }
    apply_start, apply_end = map(
        int, window.get("apply_range", [start, end])
    )
    interval_values = [
        candidate_pose(
            baseline,
            frame=frame,
            part=part,
            reference_part=reference_part,
        )
        for frame in range(apply_start, apply_end + 1)
        if f"{frame:06d}" in baseline["frames"]
    ]
    if interval_values and all(
        np.allclose(interval_values[0], value, atol=1e-10, rtol=1e-10)
        for value in interval_values[1:]
    ):
        if window.get("refine_constant_pose", False):
            return _refine_constant_static_pose(
                cfg,
                baseline,
                trajectory,
                window,
                part=part,
                reference_part=reference_part,
                start=apply_start,
                end=apply_end,
                evidence=evidence,
            )
        return {
            "mode": "static_window",
            "part": part,
            "reference_part": reference_part,
            "frame_range": [start, end],
            "evidence_frames": evidence,
            "candidate_frames": candidates,
            "applied": False,
            "reason": "already_constant_pose",
            "applied_frame_count": 0,
        }
    render_cfg, raw_mesh, observations, objectives = _window_objectives(
        cfg, baseline, window, part, sorted(set(evidence))
    )
    evidence = [frame for frame in evidence if frame in objectives]
    if not evidence:
        return {
            "mode": "static_window",
            "part": part,
            "frame_range": [start, end],
            "applied": False,
            "reason": "no_renderable_evidence",
        }
    candidate_values = [
        candidate_pose(
            baseline,
            frame=frame,
            part=part,
            reference_part=reference_part,
        )
        for frame in candidates
    ]
    primary_view = window.get("primary_view")
    primary_weight = float(window.get("primary_view_weight", 0.0))

    def coarse_evaluate(frame: int, value: np.ndarray) -> float:
        metric = objectives[frame].evaluate(world_pose(
            baseline,
            frame=frame,
            pose=value,
            reference_part=reference_part,
        ))
        return metric_score(metric, primary_view, primary_weight)

    coarse_scores = static_candidate_scores(
        candidate_values, evidence, coarse_evaluate
    )
    exact_top_k = max(1, int(window.get("exact_top_k", 5)))
    ranked = np.argsort(coarse_scores)[:exact_top_k]
    width, height = [int(value) for value in render_cfg["resolution"]]
    renderer = SceneRenderer(width, height, cache_mesh_resources=True)
    exact_objectives = {
        frame: ExactMultiViewRenderObjective(
            raw_mesh,
            float(baseline["scales"][part]),
            np.asarray(
                baseline["raw_mesh_origins"][part], dtype=np.float64
            ),
            observations[frame],
            render_cfg,
            renderer,
        )
        for frame in evidence
    }
    exact_scores: dict[int, float] = {}
    for candidate_index in ranked:
        values = []
        for frame in evidence:
            metric = exact_objectives[frame].evaluate(world_pose(
                baseline,
                frame=frame,
                pose=candidate_values[int(candidate_index)],
                reference_part=reference_part,
            ))
            values.append(metric_score(metric, primary_view, primary_weight))
        exact_scores[int(candidate_index)] = float(np.mean(values))
    selected_index = min(exact_scores, key=exact_scores.get)

    baseline_exact = []
    for frame in evidence:
        original = candidate_pose(
            baseline,
            frame=frame,
            part=part,
            reference_part=reference_part,
        )
        metric = exact_objectives[frame].evaluate(world_pose(
            baseline,
            frame=frame,
            pose=original,
            reference_part=reference_part,
        ))
        baseline_exact.append(metric_score(metric, primary_view, primary_weight))
    renderer.close()
    baseline_score = float(np.mean(baseline_exact))
    selected_score = float(exact_scores[selected_index])
    maximum_degradation = float(
        window.get("maximum_visual_loss_degradation", 0.08)
    )
    accepted = selected_score <= baseline_score + maximum_degradation
    applied_frames = []
    if accepted:
        selected = candidate_values[selected_index]
        for frame in range(apply_start, apply_end + 1):
            frame_id = f"{frame:06d}"
            if frame_id not in trajectory["frames"]:
                continue
            record = trajectory["frames"][frame_id]["parts"][part]
            record["T_world_from_part"] = world_pose(
                trajectory,
                frame=frame,
                pose=selected,
                reference_part=reference_part,
            ).tolist()
            record["source"] = (
                str(record.get("source", "pose"))
                + "+multiframe_static_visual_anchor"
            )
            record["pose_source"] = "multiframe_static_visual_anchor"
            applied_frames.append(frame)
    return {
        "mode": "static_window",
        "part": part,
        "reference_part": reference_part,
        "frame_range": [start, end],
        "evidence_frames": evidence,
        "candidate_frames": candidates,
        "selected_candidate_frame": candidates[selected_index],
        "baseline_exact_score": baseline_score,
        "selected_exact_score": selected_score,
        "maximum_visual_loss_degradation": maximum_degradation,
        "applied": accepted,
        "applied_frame_count": len(applied_frames),
    }


def _dynamic_window(
    cfg: dict[str, Any],
    baseline: dict[str, Any],
    trajectory: dict[str, Any],
    window: dict[str, Any],
) -> dict[str, Any]:
    part = str(window["part"])
    reference_part = (
        str(window["reference_part"])
        if window.get("reference_part") is not None
        else None
    )
    start, end = map(int, window["frame_range"])
    minimum_views = int(window.get("minimum_observing_views", 2))
    frames = [
        frame for frame in range(start, end + 1)
        if f"{frame:06d}" in baseline["frames"]
    ]
    if len(frames) < 2:
        return {
            "mode": "dynamic_window",
            "part": part,
            "frame_range": [start, end],
            "applied": False,
            "reason": "fewer_than_two_trajectory_frames",
        }
    render_cfg, _mesh, observations, objectives = _window_objectives(
        cfg, baseline, window, part, frames
    )
    if not objectives:
        return {
            "mode": "dynamic_window",
            "part": part,
            "frame_range": [start, end],
            "applied": False,
            "reason": "no_renderable_evidence",
        }
    radius = int(window.get("candidate_radius_frames", 2))
    primary_view = window.get("primary_view")
    primary_weight = float(window.get("primary_view_weight", 0.0))
    candidate_values: list[list[np.ndarray]] = []
    candidate_sources: list[list[int]] = []
    unary_costs: list[np.ndarray] = []
    baseline_indices: list[int] = []
    for frame in frames:
        source_frames = [
            candidate for candidate in frames
            if abs(candidate - frame) <= radius
        ]
        if frame not in source_frames:
            source_frames.append(frame)
            source_frames.sort()
        values = [
            candidate_pose(
                baseline,
                frame=source,
                part=part,
                reference_part=reference_part,
            )
            for source in source_frames
        ]
        objective = objectives.get(frame)
        costs = []
        for value in values:
            if objective is None:
                costs.append(0.0)
            else:
                metric = objective.evaluate(world_pose(
                    baseline,
                    frame=frame,
                    pose=value,
                    reference_part=reference_part,
                ))
                costs.append(metric_score(
                    metric, primary_view, primary_weight
                ))
        candidate_sources.append(source_frames)
        candidate_values.append(values)
        unary_costs.append(np.asarray(costs, dtype=np.float64))
        baseline_indices.append(source_frames.index(frame))
    visual_costs = [row.copy() for row in unary_costs]
    boundary_reports: dict[str, Any] = {}
    translation_boundary_tolerance = float(
        window.get("maximum_boundary_translation_degradation_m", 0.002)
    )
    rotation_boundary_tolerance = float(
        window.get("maximum_boundary_rotation_degradation_deg", 0.5)
    )
    for label, boundary_frame, row_index in (
        ("start", start - 1, 0),
        ("end", end + 1, -1),
    ):
        boundary_id = f"{boundary_frame:06d}"
        if boundary_id not in baseline["frames"]:
            continue
        boundary = candidate_pose(
            baseline,
            frame=boundary_frame,
            part=part,
            reference_part=reference_part,
        )
        mask, boundary_report = boundary_candidate_mask(
            candidate_values[row_index],
            boundary_pose=boundary,
            baseline_pose=(
                candidate_values[row_index][baseline_indices[row_index]]
            ),
            maximum_translation_degradation_m=(
                translation_boundary_tolerance
            ),
            maximum_rotation_degradation_deg=rotation_boundary_tolerance,
        )
        unary_costs[row_index][~mask] = np.inf
        boundary_report["frame"] = boundary_frame
        boundary_report["candidate_source_frames"] = (
            candidate_sources[row_index]
        )
        boundary_reports[label] = boundary_report
    maximum_translation = float(
        window.get("maximum_translation_step_m", 0.04)
    )
    maximum_rotation = float(
        window.get("maximum_rotation_step_deg", 20.0)
    )
    try:
        selected_indices, path_cost = solve_pose_candidate_path(
            unary_costs,
            candidate_values,
            maximum_translation_step_m=maximum_translation,
            maximum_rotation_step_deg=maximum_rotation,
            temporal_weight=float(window.get("temporal_weight", 0.20)),
            baseline_indices=baseline_indices,
            baseline_weight=float(window.get("baseline_weight", 0.03)),
        )
    except RuntimeError as error:
        return {
            "mode": "dynamic_window",
            "part": part,
            "frame_range": [start, end],
            "applied": False,
            "reason": str(error),
        }
    baseline_visual = float(np.mean([
        row[index] for row, index in zip(visual_costs, baseline_indices)
    ]))
    selected_visual = float(np.mean([
        row[index] for row, index in zip(visual_costs, selected_indices)
    ]))
    maximum_degradation = float(
        window.get("maximum_visual_loss_degradation", 0.03)
    )
    baseline_steps = [
        pose_distance(
            candidate_values[index - 1][baseline_indices[index - 1]],
            candidate_values[index][baseline_indices[index]],
        )
        for index in range(1, len(frames))
    ]
    selected_steps = [
        pose_distance(
            candidate_values[index - 1][selected_indices[index - 1]],
            candidate_values[index][selected_indices[index]],
        )
        for index in range(1, len(frames))
    ]
    visual_gate_passed = bool(
        selected_visual <= baseline_visual + maximum_degradation
    )
    continuity_gate_passed, continuity_gate = internal_continuity_gate(
        baseline_steps,
        selected_steps,
        maximum_translation_degradation_m=float(
            window.get("maximum_internal_translation_degradation_m", 0.0)
        ),
        maximum_rotation_degradation_deg=float(
            window.get("maximum_internal_rotation_degradation_deg", 0.0)
        ),
    )
    accepted = bool(visual_gate_passed and continuity_gate_passed)
    selected_source_frames = []
    selected_frame_rows = []
    if accepted:
        for frame, index, values, sources in zip(
            frames, selected_indices, candidate_values, candidate_sources
        ):
            record = trajectory["frames"][f"{frame:06d}"]["parts"][part]
            record["T_world_from_part"] = world_pose(
                trajectory,
                frame=frame,
                pose=values[index],
                reference_part=reference_part,
            ).tolist()
            record["source"] = (
                str(record.get("source", "pose"))
                + "+multiframe_dynamic_dp"
            )
            record["pose_source"] = "multiframe_render_temporal_dp"
            selected_source_frames.append(sources[index])
            objective = objectives.get(frame)
            if objective is None:
                metric = None
                mean_iou = None
            else:
                metric = objective.evaluate(world_pose(
                    trajectory,
                    frame=frame,
                    pose=values[index],
                    reference_part=reference_part,
                ))
                mean_iou = metric.get("mean_iou")
            available_views = len(observations.get(frame, []))
            observed = bool(
                available_views
                >= int(window.get("minimum_observed_views", minimum_views))
                and mean_iou is not None
                and float(mean_iou)
                >= float(window.get("minimum_observed_iou", 0.30))
            )
            confidence = (
                None
                if mean_iou is None
                else float(np.clip((float(mean_iou) - 0.15) / 0.55, 0.0, 1.0))
            )
            record["pose_confidence"] = confidence
            record["observability"] = (
                "observed_temporal" if observed else "temporally_inferred"
            )
            selected_frame_rows.append({
                "frame": frame,
                "selected_source_frame": sources[index],
                "available_views": available_views,
                "pose_confidence": confidence,
                "observability": record["observability"],
                "metric": None if metric is None else compact_metric(metric),
            })
    return {
        "mode": "dynamic_window",
        "part": part,
        "reference_part": reference_part,
        "frame_range": [start, end],
        "renderable_frame_count": len(objectives),
        "candidate_radius_frames": radius,
        "baseline_visual_score": baseline_visual,
        "selected_visual_score": selected_visual,
        "maximum_visual_loss_degradation": maximum_degradation,
        "acceptance_gates": {
            "visual": {
                "passed": visual_gate_passed,
                "baseline_score": baseline_visual,
                "selected_score": selected_visual,
                "maximum_score": baseline_visual + maximum_degradation,
            },
            "internal_continuity": continuity_gate,
        },
        "path_cost": path_cost,
        "baseline_max_translation_step_m": max(
            (value[0] for value in baseline_steps), default=0.0
        ),
        "selected_max_translation_step_m": max(
            (value[0] for value in selected_steps), default=0.0
        ),
        "baseline_max_rotation_step_deg": max(
            (value[1] for value in baseline_steps), default=0.0
        ),
        "selected_max_rotation_step_deg": max(
            (value[1] for value in selected_steps), default=0.0
        ),
        "selected_source_frames": selected_source_frames,
        "selected_frames": selected_frame_rows,
        "boundary_gates": boundary_reports,
        "applied": accepted,
        "reason": (
            None
            if accepted
            else "visual_gate_failed"
            if not visual_gate_passed
            else "internal_continuity_gate_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output-trajectory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    cfg = validate_pose_config(load_json(args.config), check_paths=True)
    settings = multiframe_settings(cfg)
    if not settings.get("enabled", False):
        raise ValueError("multiframe_optimization.enabled must be true")
    windows = list(settings.get("windows", []))
    relations = [
        {
            **dict(window),
            "type": window.get("relation_type", "coaxial_insert"),
        }
        for window in windows
        if window.get("mode") == "relation_window"
    ]
    baseline = load_json(args.trajectory)
    trajectory = copy.deepcopy(baseline)
    report: dict[str, Any] = {
        "schema_version": 2,
        "method": "multiframe_render_temporal_optimization",
        "config": str(args.config.resolve()),
        "trajectory_input": str(args.trajectory.resolve()),
        "trajectory_input_sha256": sha256_file(args.trajectory),
        "legacy_source": settings.get("legacy_source"),
        "windows": {},
        "relations": {},
    }

    for window in windows:
        mode = str(window.get("mode"))
        name = str(window["name"])
        window_baseline = copy.deepcopy(trajectory)
        if mode == "static_window":
            report["windows"][name] = _static_window(
                cfg, window_baseline, trajectory, dict(window)
            )
        elif mode == "dynamic_window":
            report["windows"][name] = _dynamic_window(
                cfg, window_baseline, trajectory, dict(window)
            )
        elif mode == "bridge_window":
            report["windows"][name] = _bridge_window(
                window_baseline, trajectory, dict(window)
            )
        elif mode == "coaxial_snap_window":
            report["windows"][name] = _coaxial_snap_window(
                cfg, window_baseline, trajectory, dict(window)
            )
        elif mode == "orientation_transport_window":
            report["windows"][name] = _orientation_transport_window(
                window_baseline, trajectory, dict(window)
            )

    # Relation windows consume the output of preceding generic windows.  The
    # order in the config is therefore a genuine factor-graph schedule, not a
    # set of independent rewrites against stale poses.
    baseline = copy.deepcopy(trajectory)
    for relation_index, relation_value in enumerate(relations):
        baseline = copy.deepcopy(trajectory)
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
        preserve_observed_twist = bool(
            relation.get("preserve_observed_twist", True)
        )
        frame_templates = {}
        for frame in dynamic_frames:
            frame_relative, _ = relative_pose(
                baseline, frame, reference_part, moving_part
            )
            source_template = (
                frame_relative if preserve_observed_twist else anchor_relative
            )
            frame_templates[frame] = project_coaxial_pose(
                source_template,
                reference_axis=reference_axis,
                moving_axis=moving_axis,
                reference_axis_origin_m=reference_origin,
                moving_axis_origin_m=moving_origin,
                allow_axis_flip=bool(relation.get("allow_axis_flip", False)),
                target_axis_offset_m=float(
                    relation.get("target_axis_offset_m", 0.0)
                ),
            )
        terminal_twist_bridge_frames = max(
            0, int(relation.get("terminal_twist_bridge_frames", 6))
        )
        if preserve_observed_twist and terminal_twist_bridge_frames > 0:
            twist_bridge_start = max(start, seat - terminal_twist_bridge_frames)
            bridge_seed = frame_templates[twist_bridge_start]
            denominator = max(seat - twist_bridge_start, 1)
            for frame in range(twist_bridge_start, seat + 1):
                amount = (frame - twist_bridge_start) / denominator
                frame_templates[frame] = project_coaxial_pose(
                    interpolate_pose(bridge_seed, anchor_relative, amount),
                    reference_axis=reference_axis,
                    moving_axis=moving_axis,
                    reference_axis_origin_m=reference_origin,
                    moving_axis_origin_m=moving_origin,
                    allow_axis_flip=bool(
                        relation.get("allow_axis_flip", False)
                    ),
                    target_axis_offset_m=float(
                        relation.get("target_axis_offset_m", 0.0)
                    ),
                )
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
                    frame_templates[frame],
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
                # Keeping every candidate's per-view arrays alive makes the
                # relation DP scale with frames * candidates * views.  Only
                # the compact scalar summary is needed after unary scoring.
                frame_metrics.append(compact_metric(metric))
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
                frame_templates[frame],
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
        visual_gate_settings = dict(
            relation.get("visual_acceptance_gate", {})
        )
        visual_gate_rows = [
            value for value in exact_diagnostics.values()
            if value.get("baseline", {}).get("loss") is not None
            and value.get("selected", {}).get("loss") is not None
        ]
        loss_degradations = [
            float(value["selected"]["loss"])
            - float(value["baseline"]["loss"])
            for value in visual_gate_rows
        ]
        selected_ious = [
            float(value["selected"]["mean_iou"])
            for value in visual_gate_rows
            if value["selected"].get("mean_iou") is not None
        ]
        visual_gate_enabled = bool(
            visual_gate_settings.get("enabled", False)
        )
        visual_gate_passed = bool(visual_gate_rows)
        if visual_gate_enabled:
            visual_gate_passed = bool(
                len(visual_gate_rows)
                >= int(visual_gate_settings.get("minimum_frames", 1))
                and (
                    not loss_degradations
                    or float(np.mean(loss_degradations))
                    <= float(
                        visual_gate_settings.get(
                            "maximum_mean_loss_degradation", 0.08
                        )
                    )
                )
                and (
                    not loss_degradations
                    or max(loss_degradations)
                    <= float(
                        visual_gate_settings.get(
                            "maximum_frame_loss_degradation", 0.20
                        )
                    )
                )
                and (
                    not selected_ious
                    or float(np.mean(selected_ious))
                    >= float(
                        visual_gate_settings.get(
                            "minimum_selected_mean_iou", 0.25
                        )
                    )
                )
            )
        visual_gate = {
            "enabled": visual_gate_enabled,
            "evaluated_frames": len(visual_gate_rows),
            "mean_loss_degradation": (
                float(np.mean(loss_degradations))
                if loss_degradations
                else None
            ),
            "maximum_frame_loss_degradation": (
                max(loss_degradations) if loss_degradations else None
            ),
            "selected_mean_iou": (
                float(np.mean(selected_ious)) if selected_ious else None
            ),
            "passed": visual_gate_passed,
        }
        apply_enabled = bool(relation.get("apply_enabled", True))
        relation_applied = bool(
            apply_enabled
            and (
                visual_gate_passed
                or visual_gate_settings.get("apply_on_failure", False)
            )
        )
        if not relation_applied:
            trajectory = copy.deepcopy(baseline)
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
                "preserve_observed_twist": preserve_observed_twist,
                "terminal_twist_bridge_frames": terminal_twist_bridge_frames,
                "path_cost": path_cost,
                "frames": dynamic_rows,
            },
            "static_follow_range": [follow_start, follow_end],
            "static_follow_frame_count": len(followed_frames),
            "entry_bridge": entry_bridge_report,
            "exact_diagnostics": exact_diagnostics,
            "visual_acceptance_gate": visual_gate,
            "apply_enabled": apply_enabled,
            "applied": relation_applied,
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
