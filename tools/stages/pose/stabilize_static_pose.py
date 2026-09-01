#!/usr/bin/env python
"""Replace high-confidence static intervals with an auditable pose consensus."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.pose_validation import validate_trajectory
from common.simulation_assets import robust_average_pose
from common.trajectory_io import (
    refresh_trajectory_derived_fields,
    write_trajectory_files,
)


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = Rotation.from_matrix(a).inv() * Rotation.from_matrix(b)
    return float(np.degrees(delta.magnitude()))


def interpolate_pose(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate a rigid pose without blending rotation-matrix entries."""

    alpha = float(np.clip(alpha, 0.0, 1.0))
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = (
        (1.0 - alpha) * left[:3, 3] + alpha * right[:3, 3]
    )
    left_rotation = Rotation.from_matrix(left[:3, :3])
    relative = left_rotation.inv() * Rotation.from_matrix(right[:3, :3])
    result[:3, :3] = (
        left_rotation
        * Rotation.from_rotvec(alpha * relative.as_rotvec())
    ).as_matrix()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output-trajectory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_json(args.config)
    trajectory = load_json(args.trajectory)
    settings = cfg.get("static_pose_consensus", {})
    report = {
        "method": "stable_interval_anchor_or_medoid_consensus",
        "input": str(args.trajectory.resolve()),
        "ranges": {},
    }
    for part, ranges in settings.get("parts", {}).items():
        if part not in trajectory["parts"]:
            raise ValueError(f"unknown part in static consensus: {part}")
        part_rows = []
        protected = {
            int(value)
            for value in cfg.get("states", {}).get(part, {}).get(
                "tracking_anchor_frames", []
            )
        }
        for start, end in ranges:
            start, end = int(start), int(end)
            frame_ids = [
                f"{frame:06d}"
                for frame in range(start, end + 1)
                if f"{frame:06d}" in trajectory["frames"]
                and int(
                    trajectory["frames"][f"{frame:06d}"]["parts"][part].get(
                        "observing_views", 0
                    )
                ) > 0
                and trajectory["frames"][f"{frame:06d}"]["parts"][part].get(
                    "pose_valid", True
                ) is not False
            ]
            if not frame_ids:
                # A part can become deliberately unobservable after assembly.
                # In that case the tracker has already carried/interpolated a
                # pose from the last observation.  There is no measured sample
                # from which to compute a new absolute pose, but a range already
                # classified as static must still be constant.  Select the
                # inferred pose nearest the robust consensus and freeze it.
                inferred_frame_ids = [
                    f"{frame:06d}"
                    for frame in range(start, end + 1)
                    if f"{frame:06d}" in trajectory["frames"]
                ]
                range_records = [
                    trajectory["frames"][frame_id]["parts"][part]
                    for frame_id in inferred_frame_ids
                ]
                if not inferred_frame_ids:
                    part_rows.append(
                        {
                            "frame_range": [start, end],
                            "selection": "empty_trajectory_interval",
                            "selected_frame": None,
                            "candidate_statistics": {
                                "observed_pose_count": 0,
                                "trajectory_frame_count": 0,
                            },
                        }
                    )
                    continue
                inferred_poses = [
                    np.asarray(record["T_world_from_part"], dtype=np.float64)
                    for record in range_records
                ]
                inferred_average, inferred_stats = robust_average_pose(
                    inferred_poses,
                    inferred_frame_ids,
                    max_translation_residual_m=float(
                        settings.get("maximum_consensus_translation_m", 0.04)
                    ),
                    max_rotation_residual_deg=float(
                        settings.get("maximum_consensus_rotation_deg", 25.0)
                    ),
                )
                trans_scale = max(
                    float(settings.get("maximum_consensus_translation_m", 0.04)),
                    1e-6,
                )
                rot_scale = max(
                    float(settings.get("maximum_consensus_rotation_deg", 25.0)),
                    1e-6,
                )
                inferred_scores = [
                    float(np.linalg.norm(
                        pose[:3, 3] - inferred_average[:3, 3]
                    ))
                    / trans_scale
                    + rotation_error_deg(
                        pose[:3, :3], inferred_average[:3, :3]
                    )
                    / rot_scale
                    for pose in inferred_poses
                ]
                selected_index = int(np.argmin(inferred_scores))
                selected_id = inferred_frame_ids[selected_index]
                selected = inferred_poses[selected_index]
                for frame_id in inferred_frame_ids:
                    record = trajectory["frames"][frame_id]["parts"][part]
                    record["T_world_from_part"] = selected.tolist()
                    record["source"] = (
                        str(record.get("source", "pose"))
                        + "+static_consensus_unobservable"
                    )
                part_rows.append(
                    {
                        "frame_range": [start, end],
                        "selection": (
                            "inferred_pose_nearest_robust_consensus"
                        ),
                        "selected_frame": selected_id,
                        "candidate_statistics": {
                            "observed_pose_count": 0,
                            "trajectory_frame_count": len(range_records),
                            "states": sorted(
                                {
                                    str(record.get("state", "unknown"))
                                    for record in range_records
                                }
                            ),
                            "sources": sorted(
                                {
                                    str(record.get("source", "unknown"))
                                    for record in range_records
                                }
                            ),
                            "inferred_pose_consensus": inferred_stats,
                        },
                    }
                )
                continue
            poses = [
                np.asarray(
                    trajectory["frames"][frame_id]["parts"][part][
                        "T_world_from_part"
                    ],
                    dtype=np.float64,
                )
                for frame_id in frame_ids
            ]
            anchor_ids = [
                frame_id for frame_id in frame_ids if int(frame_id) in protected
            ]
            average, stats = robust_average_pose(
                poses,
                frame_ids,
                max_translation_residual_m=float(
                    settings.get("maximum_consensus_translation_m", 0.04)
                ),
                max_rotation_residual_deg=float(
                    settings.get("maximum_consensus_rotation_deg", 25.0)
                ),
            )
            range_key = f"{start}:{end}"
            validated_frame = (
                settings.get("validated_selection_frames", {})
                .get(part, {})
                .get(range_key)
            )
            if validated_frame is not None:
                selected_id = f"{int(validated_frame):06d}"
                if selected_id not in frame_ids:
                    raise ValueError(
                        "validated static selection must be an observed, "
                        f"pose-valid frame in {part} {range_key}: "
                        f"{selected_id}"
                    )
                selected = np.asarray(
                    trajectory["frames"][selected_id]["parts"][part][
                        "T_world_from_part"
                    ],
                    dtype=np.float64,
                )
                selection = "validated_multiview_visual_selection"
            elif anchor_ids:
                selected_id = anchor_ids[0]
                selected = np.asarray(
                    trajectory["frames"][selected_id]["parts"][part][
                        "T_world_from_part"
                    ],
                    dtype=np.float64,
                )
                selection = "protected_stable_anchor"
            else:
                trans_scale = max(
                    float(settings.get("maximum_consensus_translation_m", 0.04)),
                    1e-6,
                )
                rot_scale = max(
                    float(settings.get("maximum_consensus_rotation_deg", 25.0)),
                    1e-6,
                )
                scores = [
                    float(np.linalg.norm(pose[:3, 3] - average[:3, 3]))
                    / trans_scale
                    + rotation_error_deg(pose[:3, :3], average[:3, :3])
                    / rot_scale
                    for pose in poses
                ]
                selected_index = int(np.argmin(scores))
                selected_id = frame_ids[selected_index]
                selected = poses[selected_index]
                selection = "observed_pose_nearest_robust_consensus"
            entry_rotation_fallback = None
            previous_id = f"{start - 1:06d}"
            maximum_entry_rotation = settings.get(
                "maximum_entry_rotation_deg"
            )
            if (
                maximum_entry_rotation is not None
                and previous_id in trajectory["frames"]
            ):
                previous_pose = np.asarray(
                    trajectory["frames"][previous_id]["parts"][part][
                        "T_world_from_part"
                    ],
                    dtype=np.float64,
                )
                entry_rotation_deg = rotation_error_deg(
                    previous_pose[:3, :3], selected[:3, :3]
                )
                if entry_rotation_deg > float(maximum_entry_rotation):
                    selected = selected.copy()
                    selected[:3, :3] = previous_pose[:3, :3]
                    entry_rotation_fallback = {
                        "from_frame": previous_id,
                        "rejected_entry_rotation_deg": entry_rotation_deg,
                        "maximum_entry_rotation_deg": float(
                            maximum_entry_rotation
                        ),
                        "fallback": "previous_dynamic_orientation",
                    }
            for frame in range(start, end + 1):
                frame_id = f"{frame:06d}"
                if frame_id not in trajectory["frames"]:
                    continue
                record = trajectory["frames"][frame_id]["parts"][part]
                record["T_world_from_part"] = selected.tolist()
                record["source"] = (
                    str(record.get("source", "pose")) + "+static_consensus"
                )
            part_rows.append(
                {
                    "frame_range": [start, end],
                    "selection": selection,
                    "selected_frame": selected_id,
                    "candidate_statistics": stats,
                    "entry_rotation_fallback": entry_rotation_fallback,
                }
            )
        # Consensus is applied after per-frame refinement.  Freezing the
        # preceding static interval can therefore expose a discontinuity that
        # was not present while refinement was running.  Bridge only that
        # boundary over a few observed dynamic frames in SE(3).
        dynamic_entry_bridges = []
        maximum_dynamic_entry_rotation = settings.get(
            "maximum_dynamic_entry_rotation_deg"
        )
        maximum_dynamic_entry_translation = settings.get(
            "maximum_dynamic_entry_translation_m"
        )
        bridge_frames = int(settings.get("dynamic_entry_bridge_frames", 3))
        if (
            maximum_dynamic_entry_rotation is not None
            or maximum_dynamic_entry_translation is not None
        ) and bridge_frames < 2:
            raise ValueError("dynamic_entry_bridge_frames must be at least 2")
        if (
            maximum_dynamic_entry_rotation is not None
            or maximum_dynamic_entry_translation is not None
        ):
            static_frames = {
                frame
                for range_start, range_end in ranges
                for frame in range(int(range_start), int(range_end) + 1)
            }
            for _range_start, range_end in ranges:
                previous_frame = int(range_end)
                entry_frame = previous_frame + 1
                target_frame = entry_frame + bridge_frames - 1
                if any(
                    frame in static_frames
                    for frame in range(entry_frame, target_frame + 1)
                ):
                    continue
                previous_id = f"{previous_frame:06d}"
                entry_id = f"{entry_frame:06d}"
                target_id = f"{target_frame:06d}"
                if any(
                    frame_id not in trajectory["frames"]
                    for frame_id in (previous_id, entry_id, target_id)
                ):
                    continue
                previous_pose = np.asarray(
                    trajectory["frames"][previous_id]["parts"][part][
                        "T_world_from_part"
                    ],
                    dtype=np.float64,
                )
                entry_pose = np.asarray(
                    trajectory["frames"][entry_id]["parts"][part][
                        "T_world_from_part"
                    ],
                    dtype=np.float64,
                )
                entry_rotation = rotation_error_deg(
                    previous_pose[:3, :3], entry_pose[:3, :3]
                )
                entry_translation = float(np.linalg.norm(
                    entry_pose[:3, 3] - previous_pose[:3, 3]
                ))
                rotation_exceeded = (
                    maximum_dynamic_entry_rotation is not None
                    and entry_rotation
                    > float(maximum_dynamic_entry_rotation)
                )
                translation_exceeded = (
                    maximum_dynamic_entry_translation is not None
                    and entry_translation
                    > float(maximum_dynamic_entry_translation)
                )
                if not rotation_exceeded and not translation_exceeded:
                    continue
                target_pose = np.asarray(
                    trajectory["frames"][target_id]["parts"][part][
                        "T_world_from_part"
                    ],
                    dtype=np.float64,
                )
                for offset, frame in enumerate(
                    range(entry_frame, target_frame + 1), start=1
                ):
                    frame_id = f"{frame:06d}"
                    record = trajectory["frames"][frame_id]["parts"][part]
                    record["T_world_from_part"] = interpolate_pose(
                        previous_pose,
                        target_pose,
                        offset / bridge_frames,
                    ).tolist()
                    record["source"] = (
                        str(record.get("source", "pose"))
                        + "+dynamic_entry_bridge"
                    )
                dynamic_entry_bridges.append({
                    "from_frame": previous_id,
                    "entry_frame": entry_id,
                    "target_frame": target_id,
                    "rejected_entry_rotation_deg": entry_rotation,
                    "maximum_dynamic_entry_rotation_deg": (
                        float(maximum_dynamic_entry_rotation)
                        if maximum_dynamic_entry_rotation is not None
                        else None
                    ),
                    "rejected_entry_translation_m": entry_translation,
                    "maximum_dynamic_entry_translation_m": (
                        float(maximum_dynamic_entry_translation)
                        if maximum_dynamic_entry_translation is not None
                        else None
                    ),
                    "bridge_frame_count": bridge_frames,
                })
        report["ranges"][part] = part_rows
        report.setdefault("dynamic_entry_bridges", {})[part] = (
            dynamic_entry_bridges
        )

    # A fully inserted/closed part can disappear from every camera while the
    # assembled object is still being moved.  Freezing that hidden part in the
    # world makes it visibly detach from the reference body.  Preserve one
    # robust, observed relative transform at the end of assembly and carry it
    # rigidly with the independently tracked reference for the configured
    # unobservable range.
    report["rigid_follow"] = {}
    for part, rules in settings.get("rigid_follow", {}).items():
        if part not in trajectory["parts"]:
            raise ValueError(f"unknown rigid-follow part: {part}")
        part_reports = []
        for rule in rules:
            reference_part = str(rule["reference_part"])
            if reference_part not in trajectory["parts"]:
                raise ValueError(
                    f"{part}: unknown rigid-follow reference {reference_part}"
                )
            start, end = [int(value) for value in rule["frame_range"]]
            anchor_frames = [
                int(value) for value in rule.get("relative_anchor_frames", [])
            ]
            if not anchor_frames:
                anchor_frames = [start - 1]
            relative_poses = []
            relative_labels = []
            for frame in anchor_frames:
                frame_id = f"{frame:06d}"
                if frame_id not in trajectory["frames"]:
                    continue
                records = trajectory["frames"][frame_id]["parts"]
                reference_world = np.asarray(
                    records[reference_part]["T_world_from_part"],
                    dtype=np.float64,
                )
                part_world = np.asarray(
                    records[part]["T_world_from_part"], dtype=np.float64
                )
                relative_poses.append(np.linalg.inv(reference_world) @ part_world)
                relative_labels.append(frame_id)
            if not relative_poses:
                raise ValueError(
                    f"{part}: no rigid-follow relative anchor is available"
                )
            relative_pose, relative_stats = robust_average_pose(
                relative_poses,
                relative_labels,
                max_translation_residual_m=float(
                    rule.get("maximum_relative_translation_residual_m", 0.03)
                ),
                max_rotation_residual_deg=float(
                    rule.get("maximum_relative_rotation_residual_deg", 20.0)
                ),
            )
            applied_frames = []
            for frame in range(start, end + 1):
                frame_id = f"{frame:06d}"
                if frame_id not in trajectory["frames"]:
                    continue
                records = trajectory["frames"][frame_id]["parts"]
                reference_world = np.asarray(
                    records[reference_part]["T_world_from_part"],
                    dtype=np.float64,
                )
                records[part]["T_world_from_part"] = (
                    reference_world @ relative_pose
                ).tolist()
                records[part]["source"] = (
                    str(records[part].get("source", "pose"))
                    + "+rigid_follow_reference"
                )
                records[part]["pose_source"] = "assembled_rigid_follow"
                records[part]["pose_valid"] = True
                records[part]["state"] = "assembled"
                observation_state = str(
                    records[part].get("observation_state", "")
                )
                records[part]["observability"] = (
                    "occluded_attached"
                    if observation_state in {
                        "unobserved", "occluded", "visibility_rejected"
                    }
                    or not records[part].get("visible_views", [])
                    else "observed_attached"
                )
                applied_frames.append(frame)
            part_reports.append({
                "frame_range": [start, end],
                "reference_part": reference_part,
                "relative_anchor_frames": relative_labels,
                "relative_pose_consensus": relative_stats,
                "T_reference_from_part": relative_pose.tolist(),
                "applied_frame_count": len(applied_frames),
            })
        report["rigid_follow"][part] = part_reports

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
