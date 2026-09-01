"""Pure helpers for projecting solved assembly poses onto physical states."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np

from common.io_utils import write_json
from common.simulation_assets import canonical_from_raw_matrix


def minimum_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the minimum-angle rotation that maps ``source`` to ``target``."""
    first = np.asarray(source, dtype=np.float64)
    second = np.asarray(target, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cross = np.cross(first, second)
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    sine = float(np.linalg.norm(cross))
    if sine <= 1e-12:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        trial = (
            np.asarray([1.0, 0.0, 0.0])
            if abs(first[0]) < 0.9
            else np.asarray([0.0, 1.0, 0.0])
        )
        axis = np.cross(first, trial)
        axis /= np.linalg.norm(axis)
        return axis_angle_matrix(axis, math.pi)
    return axis_angle_matrix(cross / sine, math.atan2(sine, cosine))


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    direction = np.asarray(axis, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    x, y, z = direction
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one = 1.0 - cosine
    return np.asarray(
        [
            [cosine + x * x * one, x * y * one - z * sine, x * z * one + y * sine],
            [y * x * one + z * sine, cosine + y * y * one, y * z * one - x * sine],
            [z * x * one - y * sine, z * y * one + x * sine, cosine + z * z * one],
        ],
        dtype=np.float64,
    )


def rotation_vector(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    if math.pi - angle <= 1e-6:
        diagonal = np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        axis[1] = math.copysign(axis[1], rotation[0, 1] + rotation[1, 0])
        axis[2] = math.copysign(axis[2], rotation[0, 2] + rotation[2, 0])
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return axis * angle
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * math.sin(angle))
    return axis * angle


def fractional_transform(transform: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate an identity-to-correction transform for phase ramping."""
    value = float(np.clip(fraction, 0.0, 1.0))
    matrix = np.asarray(transform, dtype=np.float64)
    vector = rotation_vector(matrix[:3, :3])
    angle = float(np.linalg.norm(vector))
    result = np.eye(4, dtype=np.float64)
    if angle > 1e-12:
        result[:3, :3] = axis_angle_matrix(vector / angle, value * angle)
    result[:3, 3] = value * matrix[:3, 3]
    return result


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> list[float]:
    rotation = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = np.asarray(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            values = np.asarray([0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[2, 1] - rotation[1, 2]) / scale])
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            values = np.asarray([(rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale])
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            values = np.asarray([(rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale, (rotation[1, 0] - rotation[0, 1]) / scale])
    values /= np.linalg.norm(values)
    return values.tolist()


def rotation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first) @ np.asarray(second).T
    return math.degrees(float(np.linalg.norm(rotation_vector(relative))))


def project_pose_without_axial_yaw(
    visual_pose: np.ndarray,
    physical_pose: np.ndarray,
    tube_direction_part: np.ndarray,
) -> np.ndarray:
    """Use physical translation/tilt while retaining vision's axial yaw."""
    visual = np.asarray(visual_pose, dtype=np.float64)
    physical = np.asarray(physical_pose, dtype=np.float64)
    direction = np.asarray(tube_direction_part, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    visual_axis = visual[:3, :3] @ direction
    physical_axis = physical[:3, :3] @ direction
    result = physical.copy()
    result[:3, :3] = minimum_rotation(visual_axis, physical_axis) @ visual[:3, :3]
    return result


def write_physics_refined_trajectory(
    trajectory: dict[str, Any],
    *,
    moving_part: str,
    reference_part: str,
    visual_body_pose: np.ndarray,
    refined_body_pose: np.ndarray,
    apply_frame_range: list[int],
    report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write a consistent trajectory with a phase-ramped physical correction."""
    result = copy.deepcopy(trajectory)
    start, end = (int(value) for value in apply_frame_range)
    correction = np.asarray(refined_body_pose) @ np.linalg.inv(
        np.asarray(visual_body_pose)
    )
    canonical = canonical_from_raw_matrix(
        float(result["scales"][moving_part]),
        result["raw_mesh_origins"][moving_part],
    )
    changed: list[int] = []
    for key in sorted(result["frames"], key=int):
        frame_id = int(key)
        if frame_id < start:
            continue
        fraction = 1.0 if end <= start or frame_id >= end else (frame_id - start) / (end - start)
        frame = result["frames"][key]
        moving = frame["parts"][moving_part]
        old_body = np.asarray(moving["T_body_from_part"], dtype=np.float64)
        new_body = fractional_transform(correction, fraction) @ old_body
        reference_world = np.asarray(
            frame["parts"][reference_part]["T_world_from_part"],
            dtype=np.float64,
        )
        new_world = reference_world @ new_body
        moving["T_body_from_part"] = new_body.tolist()
        moving["T_world_from_part"] = new_world.tolist()
        moving["S_world_from_raw_mesh"] = (new_world @ canonical).tolist()
        moving["translation_body_m"] = new_body[:3, 3].tolist()
        moving["quaternion_body_xyzw"] = matrix_to_quaternion_xyzw(new_body[:3, :3])
        moving["source"] = str(moving.get("source", "pose")) + "+physics_projection"
        moving["physics_projection_fraction"] = float(fraction)
        changed.append(frame_id)

    previous: np.ndarray | None = None
    for key in sorted(result["frames"], key=int):
        moving = result["frames"][key]["parts"][moving_part]
        current = np.asarray(moving["T_body_from_part"], dtype=np.float64)
        if previous is not None:
            moving["translation_step_m"] = float(
                np.linalg.norm(current[:3, 3] - previous[:3, 3])
            )
            moving["rotation_step_deg"] = rotation_error_degrees(
                current[:3, :3], previous[:3, :3]
            )
        previous = current

    result["physics_pose_refinement"] = {
        "method": "isaac_short_horizon_physical_projection",
        "moving_part": moving_part,
        "reference_part": reference_part,
        "report": str(report_path),
        "apply_frame_range": [start, end],
        "changed_frame_count": len(changed),
        "correction_T_body": correction.tolist(),
        "original_trajectory_unchanged": True,
    }
    write_json(output_path, result)
    return {
        "path": str(output_path),
        "changed_frame_count": len(changed),
        "first_changed_frame": min(changed) if changed else None,
        "last_changed_frame": max(changed) if changed else None,
        "correction_T_body": correction.tolist(),
    }

