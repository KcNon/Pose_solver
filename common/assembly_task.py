"""Assembly-oriented trajectory contract independent of a physics engine.

The visual pose trajectory is the product.  This module separates its
phase/terminal quality from connector manufacturing readiness, so a PhysX
contact result cannot silently rewrite an observed pose.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


PHASE_SEMANTICS = {
    "inactive",
    "external_kinematic_constraint",
    "connector_constraint",
    "terminal_hold",
}


def _unit(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain three finite values")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError(f"{label} cannot be zero")
    return vector / norm


def _rotation_angle_deg(matrix: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(matrix[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_assembly_task_config(
    task: dict[str, Any],
    *,
    parts: list[str],
    frame_start: int,
    frame_end: int,
) -> dict[str, Any]:
    if not isinstance(task, dict):
        raise ValueError("assembly_task must be an object")
    reference = str(task.get("reference_part", ""))
    moving = str(task.get("moving_part", ""))
    if reference not in parts or moving not in parts or reference == moving:
        raise ValueError("assembly_task has invalid reference/moving parts")
    phases = task.get("phases")
    if not isinstance(phases, dict) or not phases:
        raise ValueError("assembly_task.phases must be a non-empty object")
    previous_end = frame_start - 1
    for name, spec in phases.items():
        if not name or not isinstance(spec, dict):
            raise ValueError("assembly phase names must map to objects")
        values = spec.get("frame_range")
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"assembly_task.phases.{name}.frame_range is invalid")
        start, end = map(int, values)
        if start != previous_end + 1 or end < start or end > frame_end:
            raise ValueError(
                "assembly phase ranges must be ordered, contiguous, and inside "
                f"{frame_start}..{frame_end}"
            )
        semantic = str(spec.get("physics_semantic", ""))
        if semantic not in PHASE_SEMANTICS:
            raise ValueError(f"unsupported physics semantic {semantic!r}")
        previous_end = end
    if previous_end != frame_end:
        raise ValueError("assembly phase ranges must cover the complete trajectory")
    terminal = task.get("terminal_frame_range")
    if not isinstance(terminal, list) or len(terminal) != 2:
        raise ValueError("assembly_task.terminal_frame_range must be [start, end]")
    terminal_start, terminal_end = map(int, terminal)
    if terminal_start < frame_start or terminal_end > frame_end or terminal_start > terminal_end:
        raise ValueError("invalid assembly_task.terminal_frame_range")
    _unit(task["reference_axis_part"], "assembly_task.reference_axis_part")
    return task


def _relative_pose(
    trajectory: dict[str, Any], frame: int, reference: str, moving: str
) -> np.ndarray:
    records = trajectory["frames"][f"{frame:06d}"]["parts"]
    reference_world = np.asarray(
        records[reference]["T_world_from_part"], dtype=np.float64
    )
    moving_world = np.asarray(
        records[moving]["T_world_from_part"], dtype=np.float64
    )
    return np.linalg.inv(reference_world) @ moving_world


def _visual_evidence(task: dict[str, Any], trajectory_path: Path) -> dict[str, Any]:
    value = task.get("terminal_visual_evidence")
    if not isinstance(value, dict) or not value.get("report"):
        return {"passed": False, "failures": ["terminal_visual_evidence_missing"]}
    path = Path(value["report"]).expanduser().resolve()
    if not path.is_file():
        return {
            "passed": False,
            "report": str(path),
            "failures": ["terminal_visual_evidence_report_missing"],
        }
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    best_offset = abs(float(summary.get("best", {}).get("offset_m", np.inf)))
    diagnoses = set(map(str, value.get("accepted_diagnoses", ())))
    diagnosis = str(summary.get("diagnosis", ""))
    expected_sha = payload.get("inputs", {}).get("trajectory_sha256")
    current_sha = _sha256(trajectory_path)
    failures = []
    if expected_sha != current_sha:
        failures.append("terminal_visual_evidence_trajectory_mismatch")
    if diagnosis not in diagnoses:
        failures.append("terminal_visual_evidence_diagnosis_rejected")
    if best_offset > float(value.get("maximum_abs_best_offset_m", 0.005)) + 1e-12:
        failures.append("terminal_visual_evidence_offset_exceeds_limit")
    return {
        "passed": not failures,
        "report": str(path),
        "trajectory_sha256": current_sha,
        "diagnosis": diagnosis,
        "best_offset_m": best_offset,
        "maximum_abs_best_offset_m": float(
            value.get("maximum_abs_best_offset_m", 0.005)
        ),
        "failures": failures,
    }


def evaluate_assembly_task(
    task: dict[str, Any],
    trajectory: dict[str, Any],
    *,
    trajectory_path: Path,
    connector_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frames = sorted(map(int, trajectory["frames"]))
    validate_assembly_task_config(
        task,
        parts=list(map(str, trajectory["parts"])),
        frame_start=frames[0],
        frame_end=frames[-1],
    )
    reference = str(task["reference_part"])
    moving = str(task["moving_part"])
    axis = _unit(task["reference_axis_part"], "reference_axis_part")
    terminal_start, terminal_end = map(int, task["terminal_frame_range"])
    target_frame = int(task.get("terminal_target_frame", terminal_end))
    if not terminal_start <= target_frame <= terminal_end:
        raise ValueError("terminal_target_frame must lie in terminal_frame_range")
    target = _relative_pose(trajectory, target_frame, reference, moving)

    rows: dict[int, dict[str, Any]] = {}
    previous = None
    for frame in frames:
        relative = _relative_pose(trajectory, frame, reference, moving)
        translation_delta = relative[:3, 3] - target[:3, 3]
        axial = float(np.dot(translation_delta, axis))
        radial = float(np.linalg.norm(translation_delta - axial * axis))
        delta_rotation = target[:3, :3].T @ relative[:3, :3]
        row = {
            "frame": frame,
            "target_translation_error_m": float(np.linalg.norm(translation_delta)),
            "target_axial_error_m": axial,
            "target_radial_error_m": radial,
            "target_rotation_error_deg": _rotation_angle_deg(delta_rotation),
            "step_translation_m": None,
            "step_rotation_deg": None,
        }
        if previous is not None:
            step = np.linalg.inv(previous) @ relative
            row["step_translation_m"] = float(np.linalg.norm(step[:3, 3]))
            row["step_rotation_deg"] = _rotation_angle_deg(step)
        rows[frame] = row
        previous = relative

    phase_reports = {}
    for name, spec in task["phases"].items():
        start, end = map(int, spec["frame_range"])
        selected = [rows[frame] for frame in range(start, end + 1)]
        internal_step_rows = [
            row
            for row in selected
            if row["frame"] > start and row["step_translation_m"] is not None
        ]
        entry_row = rows[start]
        phase_reports[name] = {
            "frame_range": [start, end],
            "physics_semantic": str(spec["physics_semantic"]),
            "requires_external_constraint": str(spec["physics_semantic"])
            == "external_kinematic_constraint",
            "maximum_target_translation_error_m": max(
                row["target_translation_error_m"] for row in selected
            ),
            "maximum_target_rotation_error_deg": max(
                row["target_rotation_error_deg"] for row in selected
            ),
            "maximum_step_translation_m": max(
                (row["step_translation_m"] for row in internal_step_rows),
                default=0.0,
            ),
            "maximum_step_rotation_deg": max(
                (row["step_rotation_deg"] for row in internal_step_rows),
                default=0.0,
            ),
            "entry_step_translation_m": entry_row["step_translation_m"],
            "entry_step_rotation_deg": entry_row["step_rotation_deg"],
        }

    terminal_rows = [rows[frame] for frame in range(terminal_start, terminal_end + 1)]
    thresholds = task.get("terminal_thresholds", {})
    max_translation = max(row["target_translation_error_m"] for row in terminal_rows)
    max_rotation = max(row["target_rotation_error_deg"] for row in terminal_rows)
    terminal_failures = []
    if max_translation > float(thresholds.get("maximum_translation_residual_m", 0.005)):
        terminal_failures.append("terminal_translation_stability_failed")
    if max_rotation > float(thresholds.get("maximum_rotation_residual_deg", 2.0)):
        terminal_failures.append("terminal_rotation_stability_failed")
    visual = _visual_evidence(task, trajectory_path)
    connector_ready = bool(
        connector_report is not None and connector_report.get("simulation_ready")
    )
    pose_failures = terminal_failures + list(visual["failures"])
    physics_blockers = []
    if not connector_ready:
        physics_blockers.append("connector_not_simulation_ready")
    if any(
        report["requires_external_constraint"] for report in phase_reports.values()
    ) and not bool(task.get("external_constraint_model_available", False)):
        physics_blockers.append("external_hand_or_gripper_model_missing")
    omitted = list(map(str, task.get("omitted_geometry", ())))
    return {
        "schema_version": 1,
        "method": "assembly_phase_and_terminal_pose_contract",
        "reference_part": reference,
        "moving_part": moving,
        "trajectory": str(trajectory_path.resolve()),
        "terminal_target": {
            "frame": target_frame,
            "frame_range": [terminal_start, terminal_end],
            "T_reference_from_moving": target.tolist(),
            "maximum_translation_residual_m": max_translation,
            "maximum_rotation_residual_deg": max_rotation,
            "failures": terminal_failures,
        },
        "terminal_visual_evidence": visual,
        "phases": phase_reports,
        "omitted_geometry": omitted,
        "pose_product_ready": not pose_failures,
        "pose_failures": sorted(set(pose_failures)),
        "physics_replay_ready": not physics_blockers,
        "physics_blockers": sorted(set(physics_blockers)),
        "connector_readiness": connector_report,
        "interpretation": {
            "pose": "Terminal visual pose and trajectory phases are evaluated without changing trajectory values.",
            "physics": "Grasped transport requires an external hand/gripper constraint; connector readiness is a separate fail-closed gate.",
            "omitted_geometry": "Omitted geometry changes dynamics and collision, not the observed rigid nozzle pose.",
        },
    }
