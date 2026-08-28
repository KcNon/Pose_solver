"""Shared contracts and solvers for multi-frame pose regularization.

The module deliberately contains no dataset or part names.  A window states
whether a part is static in the world, moves through a sequence of visual
candidates, or follows a semantic relation to another part.  Rendering stays
in the stage adapter; the numerical path selection and configuration contract
live here so they can be tested without a GPU.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterable

import numpy as np
from scipy.spatial.transform import Rotation


WINDOW_MODES = {
    "static_window",
    "dynamic_window",
    "bridge_window",
    "relation_window",
}


def multiframe_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical settings, mapping the former assembly key.

    ``observed_assembly_regularization`` remains readable so existing results
    are reproducible.  New configs must use ``multiframe_optimization`` and
    explicit window modes.
    """

    configured = config.get("multiframe_optimization")
    if configured is not None:
        result = deepcopy(dict(configured))
        result.setdefault("windows", [])
        return result

    legacy = config.get("observed_assembly_regularization", {})
    if not legacy:
        return {"enabled": False, "windows": []}
    windows = []
    for relation in legacy.get("relations", []):
        window = deepcopy(dict(relation))
        window["mode"] = "relation_window"
        window["relation_type"] = window.pop("type", "coaxial_insert")
        windows.append(window)
    return {
        "enabled": bool(legacy.get("enabled", False)),
        "windows": windows,
        "legacy_source": "observed_assembly_regularization",
    }


def validate_multiframe_settings(
    config: dict[str, Any],
    *,
    parts: Iterable[str],
    frame_start: int,
    frame_end: int,
) -> dict[str, Any]:
    """Validate and return the normalized multi-frame settings."""

    settings = multiframe_settings(config)
    if not settings.get("enabled", False):
        return settings
    windows = settings.get("windows")
    if not isinstance(windows, list):
        raise ValueError("multiframe_optimization.windows must be a list")
    automatic = bool(
        settings.get("auto_static_windows", False)
        or settings.get("auto_dynamic_windows", False)
    )
    if not windows and not automatic:
        raise ValueError(
            "multiframe_optimization requires at least one window"
        )
    part_names = set(map(str, parts))
    names: set[str] = set()
    for index, value in enumerate(windows):
        if not isinstance(value, dict):
            raise ValueError(f"multi-frame window {index} must be an object")
        name = str(value.get("name", ""))
        if not name or name in names:
            raise ValueError(
                "multi-frame window names must be non-empty and unique"
            )
        names.add(name)
        mode = str(value.get("mode", ""))
        if mode not in WINDOW_MODES:
            raise ValueError(f"{name}: unsupported multi-frame mode {mode!r}")
        if mode == "relation_window":
            moving_part = str(value.get("moving_part", ""))
            reference_part = str(value.get("reference_part", ""))
            relation_type = str(value.get("relation_type", "coaxial_insert"))
            if relation_type != "coaxial_insert":
                raise ValueError(
                    f"{name}: unsupported relation_type {relation_type!r}"
                )
            if moving_part not in part_names or reference_part not in part_names:
                raise ValueError(f"{name}: relation contains unknown parts")
            if moving_part == reference_part:
                raise ValueError(f"{name}: moving and reference parts must differ")
            required = {
                "reference_axis_part",
                "moving_axis_part",
                "terminal_anchor_frame",
            }
            missing = required.difference(value)
            if missing:
                raise ValueError(f"{name}: missing keys {sorted(missing)}")
        else:
            part = str(value.get("part", ""))
            if part not in part_names:
                raise ValueError(f"{name}: unknown part {part!r}")
            reference_part = value.get("reference_part")
            if reference_part is not None:
                reference_part = str(reference_part)
                if reference_part not in part_names or reference_part == part:
                    raise ValueError(f"{name}: invalid reference_part")
        start, end = _frame_range(value, name)
        if start < int(frame_start) or end > int(frame_end):
            raise ValueError(
                f"{name}: frame_range {start}..{end} lies outside "
                f"{frame_start}..{frame_end}"
            )
        for key in ("evidence_frames", "candidate_frames"):
            for frame in map(int, value.get(key, [])):
                if frame < frame_start or frame > frame_end:
                    raise ValueError(f"{name}: {key} contains frame {frame}")
        if mode == "static_window":
            apply_start, apply_end = map(
                int, value.get("apply_range", [start, end])
            )
            if apply_start < frame_start or apply_end > frame_end:
                raise ValueError(f"{name}: apply_range lies outside the sequence")
            if apply_start > apply_end:
                raise ValueError(f"{name}: invalid apply_range")
        if mode == "dynamic_window":
            if int(value.get("candidate_radius_frames", 2)) < 0:
                raise ValueError(f"{name}: candidate_radius_frames must be >= 0")
            if float(value.get("maximum_translation_step_m", 0.04)) <= 0.0:
                raise ValueError(
                    f"{name}: maximum_translation_step_m must be positive"
                )
            if float(value.get("maximum_rotation_step_deg", 20.0)) <= 0.0:
                raise ValueError(
                    f"{name}: maximum_rotation_step_deg must be positive"
                )
            for key in (
                "maximum_internal_translation_degradation_m",
                "maximum_internal_rotation_degradation_deg",
            ):
                if float(value.get(key, 0.0)) < 0.0:
                    raise ValueError(f"{name}: {key} must be non-negative")
    return settings


def _frame_range(window: dict[str, Any], name: str) -> tuple[int, int]:
    values = window.get("frame_range")
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{name}: frame_range must be [start, end]")
    start, end = map(int, values)
    if start > end:
        raise ValueError(f"{name}: invalid frame_range {start}..{end}")
    return start, end


def evenly_spaced_frames(frames: Iterable[int], maximum: int) -> list[int]:
    """Deterministically retain at most ``maximum`` ordered frame IDs."""

    values = sorted(set(map(int, frames)))
    maximum = int(maximum)
    if maximum <= 0 or len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum)
    return [values[int(round(index))] for index in indices]


def pose_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return translation metres and geodesic rotation degrees."""

    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    translation = float(np.linalg.norm(left[:3, 3] - right[:3, 3]))
    rotation = Rotation.from_matrix(left[:3, :3]).inv() * Rotation.from_matrix(
        right[:3, :3]
    )
    return translation, float(np.degrees(rotation.magnitude()))


def boundary_candidate_mask(
    candidates: list[np.ndarray],
    *,
    boundary_pose: np.ndarray,
    baseline_pose: np.ndarray,
    maximum_translation_degradation_m: float = 0.002,
    maximum_rotation_degradation_deg: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Gate window endpoints against the unchanged neighbouring pose.

    A dynamic-programming path can look perfectly smooth *inside* a window
    while selecting the same source pose at every row and creating a large
    jump at the first or last frame.  The original endpoint is always a valid
    candidate, so allow alternatives only when their boundary transition is
    no worse than that baseline transition (up to small numerical tolerances).
    """

    if not candidates:
        raise ValueError("boundary candidate gate needs candidates")
    translation_tolerance = float(maximum_translation_degradation_m)
    rotation_tolerance = float(maximum_rotation_degradation_deg)
    if translation_tolerance < 0.0 or rotation_tolerance < 0.0:
        raise ValueError("boundary degradation tolerances must be non-negative")
    baseline_translation, baseline_rotation = pose_distance(
        boundary_pose, baseline_pose
    )
    allowed_translation = baseline_translation + translation_tolerance
    allowed_rotation = baseline_rotation + rotation_tolerance
    rows = []
    accepted = []
    for candidate in candidates:
        translation, rotation = pose_distance(boundary_pose, candidate)
        passed = bool(
            translation <= allowed_translation + 1e-12
            and rotation <= allowed_rotation + 1e-9
        )
        accepted.append(passed)
        rows.append({
            "translation_step_m": translation,
            "rotation_step_deg": rotation,
            "accepted": passed,
        })
    return np.asarray(accepted, dtype=bool), {
        "baseline_translation_step_m": baseline_translation,
        "baseline_rotation_step_deg": baseline_rotation,
        "maximum_translation_step_m": allowed_translation,
        "maximum_rotation_step_deg": allowed_rotation,
        "candidate_steps": rows,
    }


def internal_continuity_gate(
    baseline_steps: Iterable[tuple[float, float]],
    selected_steps: Iterable[tuple[float, float]],
    *,
    maximum_translation_degradation_m: float = 0.0,
    maximum_rotation_degradation_deg: float = 0.0,
) -> tuple[bool, dict[str, Any]]:
    """Reject a visually attractive path that worsens internal pose jumps.

    Endpoint gates protect only transitions across a window boundary. A path
    can still introduce a large discontinuity inside the window, so compare
    its maximum internal steps with the unchanged baseline and allow only
    explicitly configured tolerances.
    """

    translation_tolerance = float(maximum_translation_degradation_m)
    rotation_tolerance = float(maximum_rotation_degradation_deg)
    if translation_tolerance < 0.0 or rotation_tolerance < 0.0:
        raise ValueError("internal continuity tolerances must be non-negative")
    baseline = list(baseline_steps)
    selected = list(selected_steps)
    baseline_translation = max(
        (float(value[0]) for value in baseline), default=0.0
    )
    baseline_rotation = max(
        (float(value[1]) for value in baseline), default=0.0
    )
    selected_translation = max(
        (float(value[0]) for value in selected), default=0.0
    )
    selected_rotation = max(
        (float(value[1]) for value in selected), default=0.0
    )
    allowed_translation = baseline_translation + translation_tolerance
    allowed_rotation = baseline_rotation + rotation_tolerance
    passed = bool(
        selected_translation <= allowed_translation + 1e-12
        and selected_rotation <= allowed_rotation + 1e-9
    )
    return passed, {
        "passed": passed,
        "baseline_max_translation_step_m": baseline_translation,
        "selected_max_translation_step_m": selected_translation,
        "maximum_translation_step_m": allowed_translation,
        "baseline_max_rotation_step_deg": baseline_rotation,
        "selected_max_rotation_step_deg": selected_rotation,
        "maximum_rotation_step_deg": allowed_rotation,
    }


def solve_pose_candidate_path(
    unary_costs: list[np.ndarray],
    candidate_poses: list[list[np.ndarray]],
    *,
    maximum_translation_step_m: float,
    maximum_rotation_step_deg: float,
    temporal_weight: float,
    baseline_indices: list[int] | None = None,
    baseline_weight: float = 0.0,
) -> tuple[list[int], float]:
    """Solve a variable-cardinality SE(3) candidate lattice by dynamic programming.

    Impossible temporal jumps are removed.  If all incoming jumps to a state
    are impossible, that state remains unreachable instead of silently
    violating the configured rate limit.
    """

    if not unary_costs or len(unary_costs) != len(candidate_poses):
        raise ValueError("unary_costs and candidate_poses must align and be non-empty")
    costs = [np.asarray(row, dtype=np.float64) for row in unary_costs]
    for index, (row, poses) in enumerate(zip(costs, candidate_poses)):
        if row.ndim != 1 or len(row) != len(poses) or not poses:
            raise ValueError(f"invalid candidate lattice row {index}")
    maximum_translation = float(maximum_translation_step_m)
    maximum_rotation = float(maximum_rotation_step_deg)
    temporal_weight = float(temporal_weight)
    if maximum_translation <= 0.0 or maximum_rotation <= 0.0:
        raise ValueError("maximum pose steps must be positive")

    previous = costs[0].copy()
    if baseline_indices is not None and baseline_weight > 0.0:
        baseline = int(baseline_indices[0])
        for index, pose in enumerate(candidate_poses[0]):
            translation, rotation = pose_distance(
                candidate_poses[0][baseline], pose
            )
            previous[index] += float(baseline_weight) * (
                translation / maximum_translation + rotation / maximum_rotation
            )
    backpointers: list[np.ndarray] = []
    for frame_index in range(1, len(costs)):
        current = np.full(len(costs[frame_index]), np.inf, dtype=np.float64)
        back = np.full(len(costs[frame_index]), -1, dtype=np.int64)
        for current_index, current_pose in enumerate(
            candidate_poses[frame_index]
        ):
            best_cost = np.inf
            best_previous = -1
            for previous_index, previous_pose in enumerate(
                candidate_poses[frame_index - 1]
            ):
                if not np.isfinite(previous[previous_index]):
                    continue
                translation, rotation = pose_distance(
                    previous_pose, current_pose
                )
                if (
                    translation > maximum_translation + 1e-12
                    or rotation > maximum_rotation + 1e-9
                ):
                    continue
                transition = temporal_weight * (
                    translation / maximum_translation
                    + rotation / maximum_rotation
                )
                value = previous[previous_index] + transition
                if value < best_cost:
                    best_cost = value
                    best_previous = previous_index
            if best_previous >= 0:
                value = best_cost + costs[frame_index][current_index]
                if baseline_indices is not None and baseline_weight > 0.0:
                    baseline = int(baseline_indices[frame_index])
                    translation, rotation = pose_distance(
                        candidate_poses[frame_index][baseline], current_pose
                    )
                    value += float(baseline_weight) * (
                        translation / maximum_translation
                        + rotation / maximum_rotation
                    )
                current[current_index] = value
                back[current_index] = best_previous
        if not np.isfinite(current).any():
            raise RuntimeError(
                f"multi-frame pose lattice is disconnected at row {frame_index}"
            )
        backpointers.append(back)
        previous = current

    selected = [int(np.argmin(previous))]
    total = float(previous[selected[-1]])
    for back in reversed(backpointers):
        selected.append(int(back[selected[-1]]))
    selected.reverse()
    return selected, total


def candidate_pose(
    trajectory: dict[str, Any],
    *,
    frame: int,
    part: str,
    reference_part: str | None,
) -> np.ndarray:
    """Read an absolute pose or a pose relative to another tracked part."""

    records = trajectory["frames"][f"{int(frame):06d}"]["parts"]
    pose = np.asarray(records[part]["T_world_from_part"], dtype=np.float64)
    if reference_part is None:
        return pose
    reference = np.asarray(
        records[reference_part]["T_world_from_part"], dtype=np.float64
    )
    return np.linalg.inv(reference) @ pose


def world_pose(
    trajectory: dict[str, Any],
    *,
    frame: int,
    pose: np.ndarray,
    reference_part: str | None,
) -> np.ndarray:
    """Place an absolute or reference-relative candidate into world space."""

    value = np.asarray(pose, dtype=np.float64)
    if reference_part is None:
        return value
    record = trajectory["frames"][f"{int(frame):06d}"]["parts"][reference_part]
    reference = np.asarray(record["T_world_from_part"], dtype=np.float64)
    return reference @ value


def static_candidate_scores(
    candidates: list[np.ndarray],
    evidence_frames: list[int],
    evaluator: Callable[[int, np.ndarray], float],
) -> np.ndarray:
    """Aggregate a shared pose over every available evidence frame."""

    if not candidates or not evidence_frames:
        raise ValueError("static candidate scoring needs candidates and evidence")
    scores = np.zeros(len(candidates), dtype=np.float64)
    for candidate_index, pose in enumerate(candidates):
        values = [float(evaluator(frame, pose)) for frame in evidence_frames]
        scores[candidate_index] = float(np.mean(values))
    return scores
