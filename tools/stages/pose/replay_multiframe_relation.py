#!/usr/bin/env python3
"""Replay an accepted relation candidate from a saved multi-frame report."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.multiframe_pose import multiframe_settings
from common.pose_validation import validate_trajectory
from common.trajectory_constraints import (
    axis_vector,
    interpolate_pose,
    project_coaxial_pose,
)
from common.trajectory_io import (
    refresh_trajectory_derived_fields,
    write_trajectory_files,
)
from tools.stages.pose.optimize_multiframe_pose import (
    canonical_axis_origin,
    candidate_relative_pose,
    relative_pose,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--multiframe-report", required=True, type=Path)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--output-trajectory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_json(args.config)
    trajectory = load_json(args.trajectory)
    source_report = load_json(args.multiframe_report)
    trajectory_input_sha256 = sha256_file(args.trajectory)
    reported_output_sha256 = source_report.get("trajectory_output_sha256")
    if reported_output_sha256 != trajectory_input_sha256:
        raise ValueError(
            "trajectory does not match the saved optimizer report output: "
            f"report={reported_output_sha256!r}, input={trajectory_input_sha256!r}"
        )
    relation_name = str(args.relation)
    source_relation = source_report.get("relations", {}).get(relation_name)
    if source_relation is None:
        raise ValueError(f"report has no relation {relation_name!r}")
    visual_gate = dict(source_relation.get("visual_acceptance_gate", {}))
    if visual_gate.get("enabled", False) and not visual_gate.get("passed", False):
        raise ValueError("cannot replay a relation that failed its visual gate")
    if source_relation.get("applied", False):
        raise ValueError(
            "source report already applied this relation; replay requires the "
            "reported unmodified trajectory"
        )
    relation = next(
        (
            dict(value)
            for value in multiframe_settings(cfg).get("windows", [])
            if value.get("mode") == "relation_window"
            and str(value.get("name")) == relation_name
        ),
        None,
    )
    if relation is None:
        raise ValueError(f"config has no relation {relation_name!r}")

    result = copy.deepcopy(trajectory)
    reference_part = str(relation["reference_part"])
    moving_part = str(relation["moving_part"])
    start, seat = map(int, relation["frame_range"])
    terminal_anchor = int(relation["terminal_anchor_frame"])
    reference_axis = relation["reference_axis_part"]
    moving_axis = relation["moving_axis_part"]
    fixed_axis = axis_vector(reference_axis)
    reference_origin = canonical_axis_origin(
        relation, "reference", reference_part, trajectory
    )
    moving_origin = canonical_axis_origin(
        relation, "moving", moving_part, trajectory
    )
    seed_relative, _ = relative_pose(
        trajectory, terminal_anchor, reference_part, moving_part
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
    anchor_search = source_relation["anchor_search"]
    anchor_relative = project_coaxial_pose(
        projected_seed,
        reference_axis=reference_axis,
        moving_axis=moving_axis,
        reference_axis_origin_m=reference_origin,
        moving_axis_origin_m=moving_origin,
        allow_axis_flip=bool(relation.get("allow_axis_flip", False)),
        target_axis_offset_m=float(relation.get("target_axis_offset_m", 0.0)),
        twist_rad=np.deg2rad(
            float(anchor_search["selected_twist_offset_deg"])
        ),
        axial_delta_m=float(anchor_search["selected_axial_offset_m"]),
    )

    preserve_twist = bool(relation.get("preserve_observed_twist", True))
    frame_templates = {}
    for frame in range(start, seat + 1):
        frame_relative, _ = relative_pose(
            trajectory, frame, reference_part, moving_part
        )
        frame_templates[frame] = project_coaxial_pose(
            frame_relative if preserve_twist else anchor_relative,
            reference_axis=reference_axis,
            moving_axis=moving_axis,
            reference_axis_origin_m=reference_origin,
            moving_axis_origin_m=moving_origin,
            allow_axis_flip=bool(relation.get("allow_axis_flip", False)),
            target_axis_offset_m=float(
                relation.get("target_axis_offset_m", 0.0)
            ),
        )
    twist_bridge_frames = max(
        0, int(relation.get("terminal_twist_bridge_frames", 6))
    )
    if preserve_twist and twist_bridge_frames:
        bridge_start = max(start, seat - twist_bridge_frames)
        bridge_seed = frame_templates[bridge_start]
        denominator = max(seat - bridge_start, 1)
        for frame in range(bridge_start, seat + 1):
            frame_templates[frame] = project_coaxial_pose(
                interpolate_pose(
                    bridge_seed,
                    anchor_relative,
                    (frame - bridge_start) / denominator,
                ),
                reference_axis=reference_axis,
                moving_axis=moving_axis,
                reference_axis_origin_m=reference_origin,
                moving_axis_origin_m=moving_origin,
                allow_axis_flip=bool(relation.get("allow_axis_flip", False)),
                target_axis_offset_m=float(
                    relation.get("target_axis_offset_m", 0.0)
                ),
            )

    axial_rows = {
        int(row["frame"]): row
        for row in source_relation["axial_dp"]["frames"]
    }
    for frame in range(start, seat + 1):
        row = axial_rows[frame]
        selected_relative = candidate_relative_pose(
            frame_templates[frame],
            float(row["axial_m"]),
            fixed_axis,
            reference_origin,
            moving_origin,
        )
        records = result["frames"][f"{frame:06d}"]["parts"]
        reference_world = np.asarray(
            records[reference_part]["T_world_from_part"], dtype=np.float64
        )
        record = records[moving_part]
        record["T_world_from_part"] = (
            reference_world @ selected_relative
        ).tolist()
        record["source"] = str(record.get("source", "pose")) + "+replayed_relation"
        record["pose_source"] = "mask_render+coaxial_insert_dp_replay"
        record["pose_confidence"] = row.get("pose_confidence")
        metric = row.get("point_metric", {})
        record["observability"] = (
            "observed_constrained"
            if int(row.get("available_views", 0)) >= 4
            and metric.get("mean_iou") is not None
            and float(metric["mean_iou"]) >= 0.30
            else "occluded_constrained"
        )

    follow_start, follow_end = map(
        int, relation.get("static_follow_range", [seat + 1, terminal_anchor])
    )
    for frame in range(follow_start, follow_end + 1):
        records = result["frames"][f"{frame:06d}"]["parts"]
        reference_world = np.asarray(
            records[reference_part]["T_world_from_part"], dtype=np.float64
        )
        record = records[moving_part]
        record["T_world_from_part"] = (
            reference_world @ anchor_relative
        ).tolist()
        record["source"] = str(record.get("source", "pose")) + "+replayed_anchor"
        record["pose_source"] = "validated_multiframe_assembly_anchor_replay"
        record["pose_confidence"] = 1.0
        record["observability"] = "observed_static_anchor"

    refresh_trajectory_derived_fields(result)
    validation, failures = validate_trajectory(
        cfg, result, enforce_assembly=False
    )
    if failures:
        raise RuntimeError("; ".join(failures))
    replay_report = {
        "schema_version": 1,
        "method": "deterministic_multiframe_relation_replay",
        "config": str(args.config.resolve()),
        "trajectory_input": str(args.trajectory.resolve()),
        "trajectory_input_sha256": trajectory_input_sha256,
        "multiframe_report": str(args.multiframe_report.resolve()),
        "multiframe_report_sha256": sha256_file(args.multiframe_report),
        "relation": relation_name,
        "visual_acceptance_gate": visual_gate,
        "frame_range": [start, seat],
        "static_follow_range": [follow_start, follow_end],
        "trajectory_validation": validation,
    }
    result.setdefault("refinements", []).append(replay_report)
    write_trajectory_files(result, args.output_trajectory)
    replay_report["trajectory_output"] = str(args.output_trajectory.resolve())
    replay_report["trajectory_output_sha256"] = sha256_file(
        args.output_trajectory
    )
    write_json(args.report, replay_report)
    print(f"trajectory -> {args.output_trajectory}")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
