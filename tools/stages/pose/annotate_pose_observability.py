#!/usr/bin/env python3
"""Mark an unobserved pose interval without changing any transforms."""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.pose_validation import validate_trajectory
from common.trajectory_io import write_trajectory_files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotate_range(
    trajectory: dict,
    *,
    part: str,
    start: int,
    end: int,
    observability: str = "occluded_unverified",
    pose_source: str = "unobserved_pose_preserved",
) -> tuple[dict, dict]:
    if start > end:
        raise ValueError("start must not exceed end")
    result = copy.deepcopy(trajectory)
    before = []
    for frame in range(start, end + 1):
        frame_id = f"{frame:06d}"
        try:
            record = result["frames"][frame_id]["parts"][part]
        except KeyError as exc:
            raise ValueError(f"missing {part} at frame {frame_id}") from exc
        transform = np.asarray(record["T_world_from_part"], dtype=np.float64)
        before.append(transform.copy())
        record["observability"] = observability
        record["pose_source"] = pose_source
        record["pose_confidence"] = None
        record["source"] = (
            str(record.get("source", "pose")) + "+observability_annotation"
        )

    maximum_transform_change = max(
        (
            float(
                np.max(
                    np.abs(
                        np.asarray(
                            result["frames"][f"{frame:06d}"]["parts"][part][
                                "T_world_from_part"
                            ],
                            dtype=np.float64,
                        )
                        - before[index]
                    )
                )
            )
            for index, frame in enumerate(range(start, end + 1))
        ),
        default=0.0,
    )
    report = {
        "part": part,
        "frame_range": [start, end],
        "frame_count": end - start + 1,
        "observability": observability,
        "pose_source": pose_source,
        "maximum_transform_change": maximum_transform_change,
    }
    return result, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--part", required=True)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--observability", default="occluded_unverified")
    parser.add_argument("--pose-source", default="unobserved_pose_preserved")
    parser.add_argument("--output-trajectory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_json(args.config)
    input_trajectory = load_json(args.trajectory)
    result, annotation = annotate_range(
        input_trajectory,
        part=args.part,
        start=args.start,
        end=args.end,
        observability=args.observability,
        pose_source=args.pose_source,
    )
    validation, failures = validate_trajectory(
        cfg, result, enforce_assembly=False
    )
    if failures:
        raise RuntimeError("; ".join(failures))
    report = {
        "schema_version": 1,
        "method": "pose_observability_annotation",
        "config": str(args.config.resolve()),
        "trajectory_input": str(args.trajectory.resolve()),
        "trajectory_input_sha256": sha256_file(args.trajectory),
        "annotation": annotation,
        "trajectory_validation": validation,
    }
    result.setdefault("refinements", []).append({
        "method": report["method"],
        "input": report["trajectory_input"],
        "report": str(args.report.resolve()),
        "annotation": annotation,
    })
    write_trajectory_files(result, args.output_trajectory)
    report["trajectory_output"] = str(args.output_trajectory.resolve())
    report["trajectory_output_sha256"] = sha256_file(args.output_trajectory)
    write_json(args.report, report)
    print(f"trajectory -> {args.output_trajectory}", flush=True)
    print(f"report -> {args.report}", flush=True)


if __name__ == "__main__":
    main()
