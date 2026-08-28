#!/usr/bin/env python3
"""Audit how endpoint correction changes one bounded cloud-tracking window.

This diagnostic reads only JSON pose/registration artifacts.  It reconstructs
the raw pairwise-ICP trajectory using the same composition rule as
``track_cloud_registration`` and compares it with the endpoint-corrected,
smoothed trajectory saved by the solver.  It never loads depth, point clouds,
images, or meshes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.resource_safety import require_memory_guard


MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_WINDOW_FRAMES = 128


def transform_metrics(transform: np.ndarray) -> dict[str, float]:
    return {
        "translation_m": float(np.linalg.norm(transform[:3, 3])),
        "rotation_deg": float(
            np.degrees(Rotation.from_matrix(transform[:3, :3]).magnitude())
        ),
    }


def pose_step(previous: np.ndarray, current: np.ndarray) -> dict[str, float]:
    return transform_metrics(np.linalg.inv(previous) @ current)


def interpolate_delta(delta: np.ndarray, fraction: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(
        Rotation.from_matrix(delta[:3, :3]).as_rotvec() * fraction
    ).as_matrix()
    result[:3, 3] = delta[:3, 3] * fraction
    return result


def _pose(trajectory: dict, frame: int, part: str) -> np.ndarray:
    key = f"{frame:06d}"
    try:
        value = trajectory["frames"][key]["parts"][part]["T_world_from_part"]
    except KeyError as exc:
        raise ValueError(f"trajectory has no {part} pose at frame {frame}") from exc
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"invalid {part} pose at frame {frame}")
    return pose


def analyze_window(
    trajectory: dict,
    registrations: dict,
    *,
    part: str,
    start: int,
    end: int,
) -> dict:
    if end <= start:
        raise ValueError("end must be greater than start")
    if end - start + 1 > MAX_WINDOW_FRAMES:
        raise ValueError(
            f"window has {end - start + 1} frames; maximum is {MAX_WINDOW_FRAMES}"
        )
    part_registrations = registrations.get(part)
    if not isinstance(part_registrations, dict):
        raise ValueError(f"registrations have no part {part!r}")

    solved = {frame: _pose(trajectory, frame, part) for frame in range(start, end + 1)}
    raw = {start: solved[start].copy()}
    rows = []
    for frame in range(start + 1, end + 1):
        key = f"{frame:06d}_to_{frame - 1:06d}"
        if key not in part_registrations:
            raise ValueError(f"missing consecutive registration {part}/{key}")
        registration = part_registrations[key]
        pair = np.asarray(registration["T_target_from_source"], dtype=np.float64)
        if pair.shape != (4, 4) or not np.isfinite(pair).all():
            raise ValueError(f"invalid transform in {part}/{key}")
        raw[frame] = np.linalg.inv(pair) @ raw[frame - 1]
        quality = registration.get("quality", {})
        rows.append(
            {
                "frame": frame,
                "window_fraction": float((frame - start) / (end - start)),
                "pair_reported": {
                    name: quality.get(name)
                    for name in (
                        "translation_m",
                        "rotation_deg",
                        "fitness_8mm",
                        "median_nn_m",
                        "rejected",
                    )
                },
                "raw_pairwise_path_step": pose_step(raw[frame - 1], raw[frame]),
                "raw_pairwise_path_from_start": pose_step(raw[start], raw[frame]),
                "saved_solver_path_step": pose_step(solved[frame - 1], solved[frame]),
                "saved_solver_path_from_start": pose_step(
                    solved[start], solved[frame]
                ),
            }
        )

    predicted_end = raw[end]
    forced_end = solved[end]
    endpoint_delta = forced_end @ np.linalg.inv(predicted_end)
    endpoint_center_delta = forced_end[:3, 3] - predicted_end[:3, 3]
    endpoint_rotation_delta = (
        forced_end[:3, :3] @ predicted_end[:3, :3].T
    )
    corrected = {}
    origin_safe_corrected = {}
    for frame in range(start, end + 1):
        fraction = (frame - start) / (end - start)
        corrected[frame] = interpolate_delta(endpoint_delta, fraction) @ raw[frame]
        rotation_fraction = Rotation.from_rotvec(
            Rotation.from_matrix(endpoint_rotation_delta).as_rotvec() * fraction
        ).as_matrix()
        origin_safe = np.eye(4, dtype=np.float64)
        origin_safe[:3, :3] = rotation_fraction @ raw[frame][:3, :3]
        origin_safe[:3, 3] = (
            raw[frame][:3, 3] + fraction * endpoint_center_delta
        )
        origin_safe_corrected[frame] = origin_safe
    corrected[start] = raw[start].copy()
    corrected[end] = forced_end.copy()
    origin_safe_corrected[start] = raw[start].copy()
    origin_safe_corrected[end] = forced_end.copy()
    for row in rows:
        frame = row["frame"]
        row["endpoint_corrected_unsmoothed_step"] = pose_step(
            corrected[frame - 1], corrected[frame]
        )
        row["origin_safe_corrected_unsmoothed_step"] = pose_step(
            origin_safe_corrected[frame - 1], origin_safe_corrected[frame]
        )
        row["saved_vs_corrected_unsmoothed"] = pose_step(
            corrected[frame], solved[frame]
        )

    def maximum(section: str, metric: str) -> dict:
        row = max(rows, key=lambda item: item[section][metric])
        return {"frame": row["frame"], metric: row[section][metric]}

    return {
        "part": part,
        "frame_range": [start, end],
        "frame_count": end - start + 1,
        "interpretation": {
            "raw_pairwise_path": "pairwise ICP composed from the saved start pose",
            "endpoint_delta": "forced_end @ inv(raw_predicted_end), distributed over the window before smoothing",
            "saved_solver_path": "trajectory after endpoint correction and two smoothing passes",
            "origin_safe_correction": "same endpoint rotation, but translation is corrected at the part origin instead of rotating it around the world origin",
        },
        "raw_predicted_end_to_forced_end_pose_mismatch": pose_step(
            predicted_end, forced_end
        ),
        "applied_left_endpoint_delta": transform_metrics(endpoint_delta),
        "raw_start_to_predicted_end": pose_step(raw[start], predicted_end),
        "saved_start_to_forced_end": pose_step(solved[start], forced_end),
        "maxima": {
            "raw_translation_step": maximum("raw_pairwise_path_step", "translation_m"),
            "raw_rotation_step": maximum("raw_pairwise_path_step", "rotation_deg"),
            "corrected_unsmoothed_translation_step": maximum(
                "endpoint_corrected_unsmoothed_step", "translation_m"
            ),
            "corrected_unsmoothed_rotation_step": maximum(
                "endpoint_corrected_unsmoothed_step", "rotation_deg"
            ),
            "origin_safe_translation_step": maximum(
                "origin_safe_corrected_unsmoothed_step", "translation_m"
            ),
            "origin_safe_rotation_step": maximum(
                "origin_safe_corrected_unsmoothed_step", "rotation_deg"
            ),
            "saved_translation_step": maximum("saved_solver_path_step", "translation_m"),
            "saved_rotation_step": maximum("saved_solver_path_step", "rotation_deg"),
        },
        "frames": rows,
    }


def main() -> None:
    require_memory_guard("tools/diagnostics/analyze_cloud_tracking_window.py")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--registrations", type=Path, required=True)
    parser.add_argument("--part", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.trajectory, args.registrations):
        size = path.stat().st_size
        if size > MAX_INPUT_BYTES:
            raise SystemExit(
                f"refusing {path}: {size} bytes exceeds {MAX_INPUT_BYTES} byte limit"
            )
    report = analyze_window(
        load_json(args.trajectory),
        load_json(args.registrations),
        part=args.part,
        start=args.start,
        end=args.end,
    )
    write_json(args.output, report)
    mismatch = report["raw_predicted_end_to_forced_end_pose_mismatch"]
    applied = report["applied_left_endpoint_delta"]
    print(
        f"{args.part} {args.start}..{args.end}: endpoint pose mismatch "
        f"{mismatch['translation_m'] * 1000.0:.1f} mm, "
        f"{mismatch['rotation_deg']:.2f} deg; applied left delta "
        f"{applied['translation_m'] * 1000.0:.1f} mm"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
