"""Trajectory normalization and serialization shared by every pose stage."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from common.io_utils import write_json
from common.pose_transforms import similarity_from_rigid


def world_pose_record(
    pose: np.ndarray,
    *,
    state: str,
    source: str,
    observing_views: int,
) -> dict:
    """Create the non-derived portion of one trajectory part record."""
    return {
        "state": state,
        "source": source,
        "observing_views": int(observing_views),
        "T_world_from_part": np.asarray(pose, dtype=np.float64).tolist(),
    }


def refresh_trajectory_derived_fields(
    trajectory: dict,
    *,
    recompute_similarity: bool = True,
) -> dict:
    """Refresh every field derived from ``T_world_from_part`` in place.

    Refinement stages are allowed to edit world poses only.  This function is
    the single authority for body-relative transforms, quaternions, render
    similarities, and per-frame motion diagnostics.
    """
    parts = list(trajectory["parts"])
    reference = str(trajectory["reference_part"])
    scales = trajectory.get("scales", {})
    origins = trajectory.get("raw_mesh_origins", {})
    previous: dict[str, np.ndarray | None] = {part: None for part in parts}
    for key in sorted(trajectory["frames"], key=int):
        frame = trajectory["frames"][key]
        body_pose = np.asarray(
            frame["parts"][reference]["T_world_from_part"], dtype=np.float64
        )
        body_inverse = np.linalg.inv(body_pose)
        for part in parts:
            record = frame["parts"][part]
            pose = np.asarray(record["T_world_from_part"], dtype=np.float64)
            relative = body_inverse @ pose
            record["T_body_from_part"] = relative.tolist()
            record["translation_body_m"] = relative[:3, 3].tolist()
            record["quaternion_body_xyzw"] = Rotation.from_matrix(
                relative[:3, :3]
            ).as_quat().tolist()
            if recompute_similarity and part in scales and part in origins:
                record["S_world_from_raw_mesh"] = similarity_from_rigid(
                    pose,
                    float(scales[part]),
                    np.asarray(origins[part], dtype=np.float64),
                ).tolist()
            prior = previous[part]
            if prior is None:
                record["translation_step_m"] = 0.0
                record["rotation_step_deg"] = 0.0
            else:
                delta = np.linalg.inv(prior) @ pose
                record["translation_step_m"] = float(
                    np.linalg.norm(delta[:3, 3])
                )
                record["rotation_step_deg"] = float(
                    np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude())
                )
            previous[part] = pose
    return trajectory


def write_trajectory_csv(trajectory: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "frame", "part", "state", "source", "observing_views",
            "tx", "ty", "tz", "qx", "qy", "qz", "qw",
            "translation_step_m", "rotation_step_deg",
        ])
        for key, frame in trajectory["frames"].items():
            for part in trajectory["parts"]:
                record = frame["parts"][part]
                writer.writerow([
                    int(key), part, record["state"], record["source"],
                    record["observing_views"], *record["translation_body_m"],
                    *record["quaternion_body_xyzw"], record["translation_step_m"],
                    record["rotation_step_deg"],
                ])


def write_trajectory_files(
    trajectory: dict,
    json_path: str | Path,
    csv_path: str | Path | None = None,
) -> None:
    """Write the canonical JSON and CSV pair."""
    json_output = Path(json_path)
    write_json(json_output, trajectory)
    write_trajectory_csv(
        trajectory,
        csv_path or json_output.with_suffix(".csv"),
    )
