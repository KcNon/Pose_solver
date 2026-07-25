#!/usr/bin/env python
"""Refine one fast-motion part on a high-frame-rate synchronized segment."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.pose_tracking import track_mask_bbox_translation
from common.pose_transforms import similarity_from_rigid
from common.pose_validation import validate_trajectory
from common.symmetry import resolve_symmetric_pose, symmetry_spec_from_state


def observing_views(config: dict, part: str, frame: int) -> int:
    part_id = int(config["part_ids"][part])
    root = Path(config["masks_dir"]) / f"{frame:06d}"
    count = 0
    for view in config["views"]:
        labels = np.asarray(Image.open(root / f"{view}.png"))
        count += int(np.count_nonzero(labels == part_id) >= 1000)
    return count


def rotation_step_deg(previous: np.ndarray, current: np.ndarray) -> float:
    relative = previous[:3, :3].T @ current[:3, :3]
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def update_record(
    record: dict,
    pose: np.ndarray,
    *,
    body_pose: np.ndarray,
    scale: float,
    origin: np.ndarray,
    source: str,
    observed: int,
) -> None:
    body_from_part = np.linalg.inv(body_pose) @ pose
    record.update({
        "state": "moving",
        "source": source,
        "observing_views": int(observed),
        "T_world_from_part": pose.tolist(),
        "T_body_from_part": body_from_part.tolist(),
        "S_world_from_raw_mesh": similarity_from_rigid(
            pose, scale, origin
        ).tolist(),
        "translation_body_m": body_from_part[:3, 3].tolist(),
        "quaternion_body_xyzw": Rotation.from_matrix(
            body_from_part[:3, :3]
        ).as_quat().tolist(),
        "detected_state": "moving",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-trajectory", required=True)
    parser.add_argument("--high-config", required=True)
    parser.add_argument("--part", required=True)
    parser.add_argument("--base-start-anchor", required=True, type=int)
    parser.add_argument("--base-end-anchor", required=True, type=int)
    parser.add_argument("--high-start", required=True, type=int)
    parser.add_argument("--high-end", required=True, type=int)
    parser.add_argument("--base-fps", required=True, type=float)
    parser.add_argument("--high-fps", required=True, type=float)
    parser.add_argument(
        "--high-timeline-start-seconds", required=True, type=float
    )
    parser.add_argument("--replace-start", required=True, type=int)
    parser.add_argument("--replace-end", required=True, type=int)
    parser.add_argument("--segment-output", required=True)
    parser.add_argument("--merged-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()

    base = load_json(Path(args.base_trajectory))
    config = load_json(Path(args.high_config))
    part = args.part
    if part not in config["parts"]:
        raise ValueError(f"{part!r} is absent from high-frame-rate config")

    scale = float(base["scales"][part])
    origin = np.asarray(base["raw_mesh_origins"][part], dtype=np.float64)
    mesh = trimesh.load(
        Path(config["mesh_dir"]) / f"{part}.glb", force="mesh"
    )
    start_pose = np.asarray(
        base["frames"][f"{args.base_start_anchor:06d}"]["parts"][part][
            "T_world_from_part"
        ],
        dtype=np.float64,
    )
    end_pose = np.asarray(
        base["frames"][f"{args.base_end_anchor:06d}"]["parts"][part][
            "T_world_from_part"
        ],
        dtype=np.float64,
    )
    symmetry = symmetry_spec_from_state(config["states"][part])
    if symmetry.equivalence != "none":
        end_pose = resolve_symmetric_pose(
            end_pose,
            start_pose,
            symmetry,
            include_observation_ambiguities=False,
        ).pose

    poses, tracking_report = track_mask_bbox_translation(
        part,
        mesh,
        scale,
        origin,
        args.high_start,
        args.high_end,
        start_pose,
        end_pose,
        Path(config["masks_dir"]),
        int(config["part_ids"][part]),
        config["views"],
        config,
    )
    for frame in range(0, args.high_start):
        poses[frame] = start_pose.copy()
    maximum_high_frame = int(config["frames"]["end"])
    for frame in range(args.high_end + 1, maximum_high_frame + 1):
        poses[frame] = end_pose.copy()

    segment = {
        "config": str(Path(args.high_config).resolve()),
        "provenance": {
            "source_trajectory": str(Path(args.base_trajectory).resolve()),
            "method": "highfps_multiview_mask_bbox_tracking",
            "base_fps": float(args.base_fps),
            "high_fps": float(args.high_fps),
            "timeline_start_seconds": float(args.high_timeline_start_seconds),
        },
        "parts": [part],
        "reference_part": part,
        "scales": {part: scale},
        "raw_mesh_origins": {part: origin.tolist()},
        "frames": {},
    }
    for frame in range(maximum_high_frame + 1):
        pose = poses[frame]
        segment["frames"][f"{frame:06d}"] = {
            "parts": {
                part: {
                    "state": (
                        "moving"
                        if args.high_start <= frame <= args.high_end
                        else "static"
                    ),
                    "source": "highfps_multiview_mask_bbox_tracking",
                    "observing_views": observing_views(
                        config, part, frame
                    ),
                    "T_world_from_part": pose.tolist(),
                    "T_body_from_part": np.eye(4).tolist(),
                    "S_world_from_raw_mesh": similarity_from_rigid(
                        pose, scale, origin
                    ).tolist(),
                }
            }
        }
    write_json(Path(args.segment_output), segment)

    merged = deepcopy(base)
    body = str(base["reference_part"])
    replacements = {}
    for base_frame in range(args.replace_start, args.replace_end + 1):
        seconds = base_frame / args.base_fps
        high_frame = int(round(
            (seconds - args.high_timeline_start_seconds) * args.high_fps
        ))
        high_frame = min(max(high_frame, args.high_start), args.high_end)
        pose = poses[high_frame]
        frame_record = merged["frames"][f"{base_frame:06d}"]["parts"]
        body_pose = np.asarray(
            frame_record[body]["T_world_from_part"], dtype=np.float64
        )
        update_record(
            frame_record[part],
            pose,
            body_pose=body_pose,
            scale=scale,
            origin=origin,
            source="highfps_multiview_mask_bbox_tracking",
            observed=observing_views(config, part, high_frame),
        )
        replacements[f"{base_frame:06d}"] = high_frame

    previous = None
    for key in sorted(merged["frames"]):
        record = merged["frames"][key]["parts"][part]
        pose = np.asarray(record["T_world_from_part"], dtype=np.float64)
        if previous is None:
            translation_step = 0.0
            rotation_step = 0.0
        else:
            translation_step = float(
                np.linalg.norm(pose[:3, 3] - previous[:3, 3])
            )
            rotation_step = rotation_step_deg(previous, pose)
        record["translation_step_m"] = translation_step
        record["rotation_step_deg"] = rotation_step
        previous = pose
    merged.setdefault("provenance", {})["highfps_refinement"] = {
        "part": part,
        "segment_trajectory": str(Path(args.segment_output).resolve()),
        "high_config": str(Path(args.high_config).resolve()),
        "replacements": replacements,
    }
    measured = [
        value for key, value in tracking_report.items()
        if key != "calibration"
    ]
    base_config_path = Path(base["config"])
    if not base_config_path.is_absolute():
        base_config_path = ROOT / base_config_path
    base_config = load_json(base_config_path)
    validation, validation_failures = validate_trajectory(
        base_config, merged
    )
    write_json(Path(args.merged_output), merged)

    report = {
        "part": part,
        "high_frame_range": [args.high_start, args.high_end],
        "base_replace_range": [args.replace_start, args.replace_end],
        "replacements": replacements,
        "tracking": tracking_report,
        "validation": validation,
        "summary": {
            "frames": len(measured),
            "measured_frames": sum(
                value["status"] == "measured" for value in measured
            ),
            "median_edge_error_px": float(np.median([
                value["median_edge_error_px"] for value in measured
                if value["median_edge_error_px"] is not None
            ])),
            "max_translation_step_m": float(max(
                value["translation_step_m"] for value in measured
            )),
        },
    }
    write_json(Path(args.report_output), report)
    print(report["summary"], flush=True)
    if validation_failures:
        raise SystemExit(
            "high-FPS merged trajectory failed validation: "
            + "; ".join(validation_failures)
        )


if __name__ == "__main__":
    main()
