#!/usr/bin/env python3
"""Generate bounded pose candidates with a retimed endpoint rotation.

The cloud tracker originally distributes its endpoint orientation correction
linearly over a dynamic window.  This diagnostic keeps every saved part center
unchanged and changes only that known rotation schedule, allowing render-based
comparison of whether the endpoint orientation or its timing is responsible
for a visible failure.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.resource_safety import require_memory_guard
from common.trajectory_io import refresh_trajectory_derived_fields, write_trajectory_files


MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_WINDOW_FRAMES = 128
MAX_EDIT_FRAMES = 256


def rotation_fraction(
    frame: int, start: int, end: int, terminal_fraction: float
) -> float:
    if frame <= start:
        return 0.0
    if frame >= end:
        return float(terminal_fraction)
    return float(terminal_fraction) * (frame - start) / (end - start)


def retime_rotation(
    candidate: dict,
    solver: dict,
    registrations: dict,
    *,
    part: str,
    window_start: int,
    window_end: int,
    schedule_start: int,
    schedule_end: int,
    terminal_fraction: float,
    follow_end: int | None = None,
) -> tuple[dict, dict]:
    if not window_start < window_end:
        raise ValueError("window end must be greater than start")
    if window_end - window_start + 1 > MAX_WINDOW_FRAMES:
        raise ValueError(f"window exceeds {MAX_WINDOW_FRAMES} frames")
    if not window_start <= schedule_start < schedule_end <= window_end:
        raise ValueError("schedule must lie inside the tracking window")
    if not 0.0 <= terminal_fraction <= 1.0:
        raise ValueError("terminal fraction must lie in [0, 1]")
    effective_follow_end = window_end if follow_end is None else int(follow_end)
    if effective_follow_end < window_end:
        raise ValueError("follow end must not precede the tracking window end")
    if effective_follow_end - window_start + 1 > MAX_EDIT_FRAMES:
        raise ValueError(f"edited range exceeds {MAX_EDIT_FRAMES} frames")
    part_rows = registrations.get(part)
    if not isinstance(part_rows, dict):
        raise ValueError(f"registrations have no part {part!r}")

    start_key = f"{window_start:06d}"
    end_key = f"{window_end:06d}"
    raw = np.asarray(
        solver["frames"][start_key]["parts"][part]["T_world_from_part"],
        dtype=np.float64,
    )
    for frame in range(window_start + 1, window_end + 1):
        key = f"{frame:06d}_to_{frame - 1:06d}"
        if key not in part_rows:
            raise ValueError(f"missing registration {part}/{key}")
        pair = np.asarray(part_rows[key]["T_target_from_source"], dtype=np.float64)
        raw = np.linalg.inv(pair) @ raw
    target = np.asarray(
        solver["frames"][end_key]["parts"][part]["T_world_from_part"],
        dtype=np.float64,
    )
    correction = target[:3, :3] @ raw[:3, :3].T
    correction_rotvec = Rotation.from_matrix(correction).as_rotvec()
    correction_deg = float(np.degrees(np.linalg.norm(correction_rotvec)))

    result = copy.deepcopy(candidate)
    rows = []
    denominator = window_end - window_start
    for frame in range(window_start, effective_follow_end + 1):
        key = f"{frame:06d}"
        record = result["frames"][key]["parts"][part]
        pose = np.asarray(record["T_world_from_part"], dtype=np.float64)
        old_fraction = min(1.0, (frame - window_start) / denominator)
        new_fraction = (
            float(terminal_fraction)
            if frame > window_end
            else rotation_fraction(
                frame, schedule_start, schedule_end, terminal_fraction
            )
        )
        old_rotation = Rotation.from_rotvec(
            correction_rotvec * old_fraction
        ).as_matrix()
        new_rotation = Rotation.from_rotvec(
            correction_rotvec * new_fraction
        ).as_matrix()
        adjusted = pose.copy()
        adjusted[:3, :3] = new_rotation @ old_rotation.T @ pose[:3, :3]
        record["T_world_from_part"] = adjusted.tolist()
        record["source"] = str(record.get("source", "pose")) + "+rotation_retimed"
        rows.append({
            "frame": frame,
            "old_fraction": float(old_fraction),
            "new_fraction": float(new_fraction),
            "rotation_adjustment_deg": float(
                correction_deg * (new_fraction - old_fraction)
            ),
            "center_preserved": bool(np.array_equal(
                adjusted[:3, 3], pose[:3, 3]
            )),
        })
    refresh_trajectory_derived_fields(result)
    report = {
        "schema_version": 1,
        "method": "retime_known_cloud_tracking_endpoint_rotation",
        "part": part,
        "window": [window_start, window_end],
        "schedule": [schedule_start, schedule_end],
        "terminal_fraction": float(terminal_fraction),
        "edited_frame_range": [window_start, effective_follow_end],
        "endpoint_rotation_deg": correction_deg,
        "translation_changed": False,
        "frames": rows,
    }
    result.setdefault("refinements", []).append(report)
    return result, report


def main() -> None:
    require_memory_guard(
        "tools/diagnostics/retime_tracking_endpoint_rotation.py"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--solver-trajectory", type=Path, required=True)
    parser.add_argument("--registrations", type=Path, required=True)
    parser.add_argument("--part", required=True)
    parser.add_argument("--window-start", type=int, required=True)
    parser.add_argument("--window-end", type=int, required=True)
    parser.add_argument("--schedule-start", type=int, required=True)
    parser.add_argument("--schedule-end", type=int, required=True)
    parser.add_argument("--terminal-fraction", type=float, default=1.0)
    parser.add_argument("--follow-end", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.trajectory, args.solver_trajectory, args.registrations):
        if path.stat().st_size > MAX_INPUT_BYTES:
            raise SystemExit(f"refusing oversized JSON input: {path}")
    result, report = retime_rotation(
        load_json(args.trajectory),
        load_json(args.solver_trajectory),
        load_json(args.registrations),
        part=args.part,
        window_start=args.window_start,
        window_end=args.window_end,
        schedule_start=args.schedule_start,
        schedule_end=args.schedule_end,
        terminal_fraction=args.terminal_fraction,
        follow_end=args.follow_end,
    )
    write_trajectory_files(result, args.output)
    report["output"] = str(args.output.resolve())
    write_json(args.report, report)
    print(
        f"{args.part}: endpoint rotation {report['endpoint_rotation_deg']:.2f} deg, "
        f"schedule {args.schedule_start}..{args.schedule_end}, "
        f"terminal fraction {args.terminal_fraction:.2f}"
    )
    print(f"trajectory -> {args.output}")


if __name__ == "__main__":
    main()
