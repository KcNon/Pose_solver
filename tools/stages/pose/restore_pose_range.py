#!/usr/bin/env python3
"""Restore one part's pose over a frame range from a compatible trajectory."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.trajectory_io import (
    refresh_trajectory_derived_fields,
    write_trajectory_files,
)
from tools.stages.pose.optimize_multiframe_pose import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--part", required=True)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame", required=True, type=int)
    parser.add_argument("--observability", default="occluded_unverified")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    if args.start_frame > args.end_frame:
        raise ValueError("start-frame must not exceed end-frame")
    base = load_json(args.base)
    source = load_json(args.source)
    if base.get("parts") != source.get("parts"):
        raise ValueError("trajectory parts do not match")
    if args.part not in base["parts"]:
        raise ValueError(f"unknown part {args.part!r}")
    result = copy.deepcopy(base)
    restored = []
    for frame in range(args.start_frame, args.end_frame + 1):
        frame_id = f"{frame:06d}"
        if frame_id not in result["frames"] or frame_id not in source["frames"]:
            raise ValueError(f"missing frame {frame_id}")
        source_record = source["frames"][frame_id]["parts"][args.part]
        record = result["frames"][frame_id]["parts"][args.part]
        record["T_world_from_part"] = copy.deepcopy(
            source_record["T_world_from_part"]
        )
        record["source"] = (
            str(source_record.get("source", "pose")) + "+range_restored"
        )
        record["pose_source"] = "restored_unverified_source_pose"
        record["pose_confidence"] = None
        record["observability"] = str(args.observability)
        restored.append(frame)
    refresh_trajectory_derived_fields(result)
    report = {
        "schema_version": 1,
        "method": "compatible_trajectory_pose_range_restore",
        "base": str(args.base.resolve()),
        "base_sha256": sha256_file(args.base),
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "part": args.part,
        "frame_range": [args.start_frame, args.end_frame],
        "restored_frame_count": len(restored),
        "observability": args.observability,
    }
    result.setdefault("refinements", []).append(report)
    write_trajectory_files(result, args.output)
    report["output"] = str(args.output.resolve())
    report["output_sha256"] = sha256_file(args.output)
    write_json(args.report, report)
    print(f"trajectory -> {args.output}")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
