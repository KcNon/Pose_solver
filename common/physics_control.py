"""Pure helpers for selecting force-control profiles during contact."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


DYNAMIC_COLLISION_APPROXIMATIONS = {
    "convexDecomposition",
    "convexHull",
    "sdf",
    "sphereFill",
}


DEFAULT_PLACE_RELEASE_TRIALS = (
    {
        "name": "centered",
        "xy_offset_m": [0.0, 0.0],
        "tilt_deg": [0.0, 0.0],
        "yaw_deg": 0.0,
    },
    {
        "name": "offset_x_plus_1mm",
        "xy_offset_m": [0.001, 0.0],
        "tilt_deg": [0.0, 0.0],
        "yaw_deg": 0.0,
    },
    {
        "name": "offset_x_minus_1mm",
        "xy_offset_m": [-0.001, 0.0],
        "tilt_deg": [0.0, 0.0],
        "yaw_deg": 0.0,
    },
    {
        "name": "tilt_x_plus_1deg",
        "xy_offset_m": [0.0, 0.0],
        "tilt_deg": [1.0, 0.0],
        "yaw_deg": 0.0,
    },
    {
        "name": "tilt_x_minus_1deg",
        "xy_offset_m": [0.0, 0.0],
        "tilt_deg": [-1.0, 0.0],
        "yaw_deg": 0.0,
    },
)


DEFAULT_PHYSICS_REFINEMENT_CANDIDATES = (
    {"name": "visual_pose", "xy_offset_m": [0.0, 0.0], "tilt_deg": [0.0, 0.0]},
    {"name": "x_plus_2mm", "xy_offset_m": [0.002, 0.0], "tilt_deg": [0.0, 0.0]},
    {"name": "x_minus_2mm", "xy_offset_m": [-0.002, 0.0], "tilt_deg": [0.0, 0.0]},
    {"name": "y_plus_2mm", "xy_offset_m": [0.0, 0.002], "tilt_deg": [0.0, 0.0]},
    {"name": "y_minus_2mm", "xy_offset_m": [0.0, -0.002], "tilt_deg": [0.0, 0.0]},
    {"name": "tilt_x_plus_2deg", "xy_offset_m": [0.0, 0.0], "tilt_deg": [2.0, 0.0]},
    {"name": "tilt_x_minus_2deg", "xy_offset_m": [0.0, 0.0], "tilt_deg": [-2.0, 0.0]},
    {"name": "tilt_y_plus_2deg", "xy_offset_m": [0.0, 0.0], "tilt_deg": [0.0, 2.0]},
    {"name": "tilt_y_minus_2deg", "xy_offset_m": [0.0, 0.0], "tilt_deg": [0.0, -2.0]},
)


def physics_pose_refinement_settings(
    simulation: dict[str, Any],
) -> dict[str, Any]:
    """Resolve bounded, stage-aware physical pose-projection settings."""
    raw = dict(simulation.get("physics_pose_refinement", {}))
    tube = dict(raw.get("tube", {}))
    result: dict[str, Any] = {
        "enabled": bool(raw.get("enabled", False)),
        "settle_seconds": float(raw.get("settle_seconds", 3.0)),
        "initial_height_m": float(raw.get("initial_height_m", 0.002)),
        "sample_seconds": float(raw.get("sample_seconds", 0.1)),
        "apply_frame_range": [
            int(value) for value in raw.get("apply_frame_range", [0, 0])
        ],
        "maximum_visual_translation_m": float(
            raw.get("maximum_visual_translation_m", 0.005)
        ),
        "maximum_visual_tilt_deg": float(
            raw.get("maximum_visual_tilt_deg", 3.0)
        ),
        "maximum_final_linear_speed_mps": float(
            raw.get("maximum_final_linear_speed_mps", 0.01)
        ),
        "maximum_final_angular_speed_radps": float(
            raw.get("maximum_final_angular_speed_radps", 0.1)
        ),
        "maximum_penetration_m": float(
            raw.get("maximum_penetration_m", 0.002)
        ),
        "require_contact": bool(raw.get("require_contact", True)),
        "candidates": list(
            raw.get("candidates", DEFAULT_PHYSICS_REFINEMENT_CANDIDATES)
        ),
        "score_weights": {
            **{
                "visual_translation": 1.0,
                "visual_tilt": 1.0,
                "linear_speed": 1.0,
                "angular_speed": 1.0,
                "penetration": 2.0,
                "missing_contact": 25.0,
                "tube_energy": 0.25,
            },
            **dict(raw.get("score_weights", {})),
        },
        "tube": {
            "enabled": bool(tube.get("enabled", True)),
            "mount_point_part_m": list(
                tube.get("mount_point_part_m", [0.0, 0.0, 0.0])
            ),
            "direction_part": list(
                tube.get("direction_part", [0.0, -1.0, 0.0])
            ),
            "body_axis_origin_body_m": list(
                tube.get("body_axis_origin_body_m", [0.0, 0.0, 0.0])
            ),
            "length_m": float(tube.get("length_m", 0.18)),
            "radius_m": float(tube.get("radius_m", 0.002)),
            "guide_depth_m": float(tube.get("guide_depth_m", 0.025)),
            "lateral_stiffness_n_per_m": float(
                tube.get("lateral_stiffness_n_per_m", 8.0)
            ),
            "lateral_damping_ns_per_m": float(
                tube.get("lateral_damping_ns_per_m", 0.12)
            ),
            "angular_stiffness_nm_per_rad": float(
                tube.get("angular_stiffness_nm_per_rad", 0.06)
            ),
            "angular_damping_nms_per_rad": float(
                tube.get("angular_damping_nms_per_rad", 0.008)
            ),
            "maximum_force_n": float(tube.get("maximum_force_n", 0.5)),
            "maximum_torque_nm": float(tube.get("maximum_torque_nm", 0.08)),
        },
    }
    if len(result["apply_frame_range"]) != 2:
        raise ValueError("physics_pose_refinement.apply_frame_range must be [start, end]")
    if result["apply_frame_range"][1] < result["apply_frame_range"][0]:
        raise ValueError("physics pose refinement frame range is reversed")
    positive = (
        "settle_seconds",
        "sample_seconds",
        "maximum_visual_translation_m",
        "maximum_visual_tilt_deg",
        "maximum_final_linear_speed_mps",
        "maximum_final_angular_speed_radps",
        "maximum_penetration_m",
    )
    if any(result[key] <= 0.0 for key in positive):
        raise ValueError("physics pose refinement limits and times must be positive")
    if result["initial_height_m"] < 0.0:
        raise ValueError("physics pose refinement initial height must be non-negative")
    if not result["candidates"]:
        raise ValueError("physics pose refinement requires at least one candidate")
    for candidate in result["candidates"]:
        xy = np.asarray(candidate.get("xy_offset_m", []), dtype=np.float64)
        tilt = np.asarray(candidate.get("tilt_deg", []), dtype=np.float64)
        if (
            not str(candidate.get("name", ""))
            or xy.shape != (2,)
            or tilt.shape != (2,)
            or not np.isfinite(xy).all()
            or not np.isfinite(tilt).all()
        ):
            raise ValueError("every physics refinement candidate needs finite 2D offsets and tilts")
    for key in (
        "mount_point_part_m",
        "direction_part",
        "body_axis_origin_body_m",
    ):
        value = np.asarray(result["tube"][key], dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"physics pose refinement tube {key} must be a finite 3-vector")
    if np.linalg.norm(np.asarray(result["tube"]["direction_part"])) <= 1e-12:
        raise ValueError("physics pose refinement tube direction must be non-zero")
    tube_positive = (
        "length_m",
        "radius_m",
        "guide_depth_m",
        "lateral_stiffness_n_per_m",
        "lateral_damping_ns_per_m",
        "angular_stiffness_nm_per_rad",
        "angular_damping_nms_per_rad",
        "maximum_force_n",
        "maximum_torque_nm",
    )
    if any(result["tube"][key] <= 0.0 for key in tube_positive):
        raise ValueError("physics pose refinement tube parameters must be positive")
    if any(float(value) < 0.0 for value in result["score_weights"].values()):
        raise ValueError("physics pose refinement score weights must be non-negative")
    return result


def _clip_norm(vector: np.ndarray, maximum: float) -> tuple[np.ndarray, bool]:
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= maximum:
        return vector, False
    return vector * (maximum / max(magnitude, 1e-12)), True


def elastic_tube_wrench(
    *,
    position_world: np.ndarray,
    rotation_world_from_part: np.ndarray,
    linear_velocity_world: np.ndarray,
    angular_velocity_world: np.ndarray,
    body_axis_origin_world: np.ndarray,
    body_axis_world: np.ndarray,
    tube: dict[str, Any],
) -> dict[str, Any]:
    """Approximate flexible dip-tube preload without treating it as a rigid pose.

    The guide point is a short distance below the nozzle mount. Its radial
    displacement from the bottle axis creates a restoring force; tube tangent
    misalignment creates a bending torque. Axial translation and screw-axis
    yaw are deliberately left unconstrained.
    """
    position = np.asarray(position_world, dtype=np.float64)
    rotation = np.asarray(rotation_world_from_part, dtype=np.float64)
    linear = np.asarray(linear_velocity_world, dtype=np.float64)
    angular = np.asarray(angular_velocity_world, dtype=np.float64)
    origin = np.asarray(body_axis_origin_world, dtype=np.float64)
    body_axis = np.asarray(body_axis_world, dtype=np.float64)
    mount_part = np.asarray(tube["mount_point_part_m"], dtype=np.float64)
    direction_part = np.asarray(tube["direction_part"], dtype=np.float64)
    body_axis /= np.linalg.norm(body_axis)
    direction_part /= np.linalg.norm(direction_part)
    direction_world = rotation @ direction_part
    mount_world = position + rotation @ mount_part
    guide_world = mount_world + float(tube["guide_depth_m"]) * direction_world
    radial = guide_world - origin
    radial -= body_axis * float(np.dot(radial, body_axis))
    guide_velocity = linear + np.cross(angular, guide_world - position)
    radial_velocity = guide_velocity - body_axis * float(
        np.dot(guide_velocity, body_axis)
    )
    force = (
        -float(tube["lateral_stiffness_n_per_m"]) * radial
        -float(tube["lateral_damping_ns_per_m"]) * radial_velocity
    )
    force, force_saturated = _clip_norm(force, float(tube["maximum_force_n"]))

    desired_direction = -body_axis
    bend_axis = np.cross(direction_world, desired_direction)
    bend_sine = float(np.linalg.norm(bend_axis))
    bend_cosine = float(np.clip(np.dot(direction_world, desired_direction), -1.0, 1.0))
    bend_angle = math.atan2(bend_sine, bend_cosine)
    bend_vector = (
        bend_axis * (bend_angle / bend_sine)
        if bend_sine > 1e-12
        else np.zeros(3, dtype=np.float64)
    )
    angular_perpendicular = angular - body_axis * float(np.dot(angular, body_axis))
    torque = (
        float(tube["angular_stiffness_nm_per_rad"]) * bend_vector
        - float(tube["angular_damping_nms_per_rad"]) * angular_perpendicular
    )
    torque, torque_saturated = _clip_norm(
        torque, float(tube["maximum_torque_nm"])
    )
    elastic_energy = (
        0.5 * float(tube["lateral_stiffness_n_per_m"]) * float(np.dot(radial, radial))
        + 0.5 * float(tube["angular_stiffness_nm_per_rad"]) * bend_angle**2
    )
    return {
        "force_world_n": force,
        "torque_world_nm": torque,
        "application_point_world_m": guide_world,
        "mount_point_world_m": mount_world,
        "direction_world": direction_world,
        "radial_deflection_m": float(np.linalg.norm(radial)),
        "bend_angle_deg": math.degrees(bend_angle),
        "elastic_energy_j": float(elastic_energy),
        "force_saturated": force_saturated,
        "torque_saturated": torque_saturated,
    }


def score_physics_pose_candidate(
    metrics: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Score a rollout while retaining explicit physical acceptance gates."""
    weights = settings["score_weights"]
    normalized = {
        "visual_translation": float(metrics["translation_error_m"])
        / float(settings["maximum_visual_translation_m"]),
        "visual_tilt": float(metrics["tilt_error_deg"])
        / float(settings["maximum_visual_tilt_deg"]),
        "linear_speed": float(metrics["final_linear_speed_mps"])
        / float(settings["maximum_final_linear_speed_mps"]),
        "angular_speed": float(metrics["final_angular_speed_radps"])
        / float(settings["maximum_final_angular_speed_radps"]),
        "penetration": float(metrics["maximum_contact_penetration_m"])
        / float(settings["maximum_penetration_m"]),
        "missing_contact": 0.0 if bool(metrics["contact_observed"]) else 1.0,
        "tube_energy": float(metrics.get("final_tube_energy_j", 0.0)) / 0.01,
    }
    terms = {
        key: float(weights[key]) * value**2
        for key, value in normalized.items()
    }
    gates = {
        "visual_translation": normalized["visual_translation"] <= 1.0,
        "visual_tilt": normalized["visual_tilt"] <= 1.0,
        "linear_speed": normalized["linear_speed"] <= 1.0,
        "angular_speed": normalized["angular_speed"] <= 1.0,
        "penetration": normalized["penetration"] <= 1.0,
        "contact": bool(metrics["contact_observed"]) or not settings["require_contact"],
    }
    return {
        "score": float(sum(terms.values())),
        "normalized": normalized,
        "terms": terms,
        "gates": gates,
        "accepted": all(gates.values()),
    }


def place_release_settings(simulation: dict[str, Any]) -> dict[str, Any]:
    """Return bounded settings for passive placement stability validation."""
    raw = dict(simulation.get("place_release", {}))
    result: dict[str, Any] = {
        "initial_height_m": float(raw.get("initial_height_m", 0.003)),
        "settle_seconds": float(raw.get("settle_seconds", 5.0)),
        "contact_window_seconds": float(
            raw.get("contact_window_seconds", 1.0)
        ),
        "minimum_contact_fraction": float(
            raw.get("minimum_contact_fraction", 0.1)
        ),
        "maximum_contact_gap_seconds": float(
            raw.get("maximum_contact_gap_seconds", 0.12)
        ),
        "maximum_lateral_error_m": float(
            raw.get("maximum_lateral_error_m", 0.005)
        ),
        "maximum_axial_error_m": float(
            raw.get("maximum_axial_error_m", 0.008)
        ),
        "maximum_tilt_error_deg": float(
            raw.get("maximum_tilt_error_deg", 5.0)
        ),
        "maximum_final_linear_speed_mps": float(
            raw.get("maximum_final_linear_speed_mps", 0.01)
        ),
        "maximum_final_angular_speed_radps": float(
            raw.get("maximum_final_angular_speed_radps", 0.1)
        ),
        "trials": list(raw.get("trials", DEFAULT_PLACE_RELEASE_TRIALS)),
    }
    positive = (
        "initial_height_m",
        "settle_seconds",
        "contact_window_seconds",
        "maximum_lateral_error_m",
        "maximum_axial_error_m",
        "maximum_tilt_error_deg",
        "maximum_final_linear_speed_mps",
        "maximum_final_angular_speed_radps",
    )
    if any(result[key] <= 0.0 for key in positive):
        raise ValueError("place_release distances, times, and limits must be positive")
    if not 0.0 <= result["minimum_contact_fraction"] <= 1.0:
        raise ValueError(
            "place_release.minimum_contact_fraction must be in [0, 1]"
        )
    if result["maximum_contact_gap_seconds"] < 0.0:
        raise ValueError(
            "place_release.maximum_contact_gap_seconds must be non-negative"
        )
    if result["contact_window_seconds"] > result["settle_seconds"]:
        raise ValueError(
            "place_release contact window cannot exceed settle_seconds"
        )
    if not result["trials"]:
        raise ValueError("place_release.trials must not be empty")
    for trial in result["trials"]:
        if not isinstance(trial, dict) or not str(trial.get("name", "")):
            raise ValueError("every place_release trial requires a name")
        xy = np.asarray(trial.get("xy_offset_m", []), dtype=np.float64)
        tilt = np.asarray(trial.get("tilt_deg", []), dtype=np.float64)
        if (
            xy.shape != (2,)
            or tilt.shape != (2,)
            or not np.isfinite(xy).all()
            or not np.isfinite(tilt).all()
        ):
            raise ValueError(
                "place_release trial offsets and tilts must each contain two finite values"
            )
    return result


def assembly_target_translation(
    simulation: dict[str, Any],
    *,
    part: str,
    frame_id: int,
    reference_rotation: np.ndarray,
) -> dict[str, Any]:
    """Resolve a contact-calibrated assembly-target translation.

    Pose trajectories describe the reconstructed visual relationship.  A
    connector may have a different physically seated relationship when the
    reconstructed meshes overlap or omit the true mating surface.  Keep that
    correction explicit in the simulation config and ramp it in before the
    static assembly interval instead of silently editing trajectory poses.
    """
    corrections = simulation.get("assembly_target_corrections", {})
    raw = corrections.get(part)
    if raw is None:
        zero = np.zeros(3, dtype=np.float64)
        return {
            "enabled": False,
            "fraction": 0.0,
            "translation_reference_m": zero,
            "translation_world_m": zero,
            "source": None,
        }
    translation = np.asarray(
        raw.get("translation_reference_m", []), dtype=np.float64
    )
    rotation = np.asarray(reference_rotation, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError(
            "assembly_target_corrections translation_reference_m must "
            "contain three finite values"
        )
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("reference_rotation must be a finite 3x3 matrix")
    ramp = raw.get("ramp_frame_range", [frame_id, frame_id])
    if not isinstance(ramp, list) or len(ramp) != 2:
        raise ValueError(
            "assembly_target_corrections ramp_frame_range must be [start, end]"
        )
    start, end = (int(ramp[0]), int(ramp[1]))
    if end < start:
        raise ValueError(
            "assembly_target_corrections ramp_frame_range end precedes start"
        )
    if frame_id < start:
        fraction = 0.0
    elif end == start or frame_id >= end:
        fraction = 1.0
    else:
        fraction = float(frame_id - start) / float(end - start)
    reference_value = fraction * translation
    return {
        "enabled": True,
        "fraction": fraction,
        "translation_reference_m": reference_value,
        "translation_world_m": rotation @ reference_value,
        "source": str(raw.get("source", "configured")),
    }


def sustained_contact_summary(
    history: list[bool] | tuple[bool, ...],
    *,
    physics_dt: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate contact continuity over the final assembly-hold window."""
    if physics_dt <= 0.0:
        raise ValueError("physics_dt must be positive")
    window_seconds = float(settings.get("contact_window_seconds", 0.5))
    minimum_fraction = float(settings.get("minimum_contact_fraction", 0.5))
    maximum_gap_seconds = float(
        settings.get("maximum_contact_gap_seconds", 0.1)
    )
    if window_seconds <= 0.0:
        raise ValueError("assembly_lock.contact_window_seconds must be positive")
    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError(
            "assembly_lock.minimum_contact_fraction must be in [0, 1]"
        )
    if maximum_gap_seconds < 0.0:
        raise ValueError(
            "assembly_lock.maximum_contact_gap_seconds must be non-negative"
        )
    values = [bool(value) for value in history]
    window_steps = max(1, int(math.ceil(window_seconds / physics_dt)))
    window = values[-window_steps:]
    contact_steps = sum(window)
    fraction = float(contact_steps / len(window)) if window else 0.0
    trailing_gap_steps = 0
    for value in reversed(window):
        if value:
            break
        trailing_gap_steps += 1
    trailing_gap_seconds = trailing_gap_steps * physics_dt
    longest_gap_steps = 0
    current_gap_steps = 0
    for value in window:
        if value:
            current_gap_steps = 0
        else:
            current_gap_steps += 1
            longest_gap_steps = max(longest_gap_steps, current_gap_steps)
    longest_gap_seconds = longest_gap_steps * physics_dt
    observed = any(values)
    sustained = bool(
        window
        and fraction >= minimum_fraction
        and longest_gap_seconds <= maximum_gap_seconds
    )
    return {
        "observed": observed,
        "sustained": sustained,
        "window_seconds": window_seconds,
        "window_steps": len(window),
        "contact_steps": contact_steps,
        "contact_fraction": fraction,
        "minimum_contact_fraction": minimum_fraction,
        "trailing_gap_seconds": trailing_gap_seconds,
        "longest_gap_seconds": longest_gap_seconds,
        "maximum_contact_gap_seconds": maximum_gap_seconds,
    }


def dynamic_collision_approximation(simulation: dict[str, Any]) -> str:
    """Return a PhysX-compatible approximation for a dynamic mesh.

    Dense reconstructed meshes are frequently concave or hollow. SDF keeps
    those details, while the legacy convex-decomposition default can close a
    cavity and report a large false penetration. Explicit legacy configs can
    continue to request convex decomposition.
    """
    approximation = str(
        simulation.get(
            "dynamic_collision_approximation",
            "convexDecomposition",
        )
    )
    if approximation not in DYNAMIC_COLLISION_APPROXIMATIONS:
        raise ValueError(
            "simulation.dynamic_collision_approximation must be one of "
            f"{sorted(DYNAMIC_COLLISION_APPROXIMATIONS)}, got "
            f"{approximation!r}"
        )
    return approximation


def transformed_bounds_minimum_z(
    bounds: list[list[float]] | np.ndarray,
    transform: list[list[float]] | np.ndarray,
) -> float:
    """Return the world minimum Z of a transformed axis-aligned box."""
    bounds_array = np.asarray(bounds, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if bounds_array.shape != (2, 3) or matrix.shape != (4, 4):
        raise ValueError("bounds must be 2x3 and transform must be 4x4")
    corners = np.asarray(
        [
            [x, y, z]
            for x in bounds_array[:, 0]
            for y in bounds_array[:, 1]
            for z in bounds_array[:, 2]
        ],
        dtype=np.float64,
    )
    world = corners @ matrix[:3, :3].T + matrix[:3, 3]
    return float(np.min(world[:, 2]))


def rigid_body_controller_parameters(
    part_info: dict[str, Any],
    override: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Derive bounded-force controller parameters from exported geometry.

    Isaac videos previously carried parameters keyed by the rice-cooker part
    names.  A dataset-level pipeline cannot know those names, so use the
    configured mass and canonical mesh extents to estimate a conservative
    scalar inertia.  Every value remains explicitly overrideable from the
    simulation config for a validated physical asset.
    """

    mass = float(part_info.get("mass_kg", 1.0))
    extents = np.asarray(part_info.get("canonical_extents_m", []), dtype=float)
    if mass <= 0.0:
        raise ValueError("part mass_kg must be positive")
    if extents.shape != (3,) or not np.isfinite(extents).all() or np.any(extents <= 0.0):
        raise ValueError("canonical_extents_m must contain three positive values")
    # Mean principal inertia of an axis-aligned box with these extents.
    inertia = mass * float(np.square(extents).sum()) / 18.0
    defaults = {
        "inertia_scale": max(inertia, 1e-5),
        "force_limit_n": max(5.0, 24.0 * mass),
        "torque_limit_nm": max(0.05, 120.0 * inertia),
    }
    configured = override or {}
    result = {
        key: float(configured.get(key, value))
        for key, value in defaults.items()
    }
    if any(value <= 0.0 for value in result.values()):
        raise ValueError("controller parameters must be positive")
    return {**result, "mass_kg": mass}


def settled_contact_settings(simulation: dict[str, Any]) -> dict[str, Any]:
    """Return validated settings for contact-aware settled-state control."""
    raw = simulation.get("settled_contact_control", {})
    enabled = bool(raw.get("enabled", False))
    states = tuple(str(value) for value in raw.get("states", ["static"]))
    frequency = float(raw.get("frequency_radps", 8.0))
    damping_ratio = float(raw.get("damping_ratio", 2.0))
    maximum_position_error = float(
        raw.get("maximum_position_error_m", 0.01)
    )
    if not states:
        raise ValueError("settled_contact_control.states must not be empty")
    if frequency <= 0.0:
        raise ValueError(
            "settled_contact_control.frequency_radps must be positive"
        )
    if damping_ratio <= 0.0:
        raise ValueError(
            "settled_contact_control.damping_ratio must be positive"
        )
    if maximum_position_error <= 0.0:
        raise ValueError(
            "settled_contact_control.maximum_position_error_m "
            "must be positive"
        )
    return {
        "enabled": enabled,
        "states": states,
        "frequency_radps": frequency,
        "damping_ratio": damping_ratio,
        "maximum_position_error_m": maximum_position_error,
    }


def select_control_profile(
    *,
    state: str,
    contact_latched: bool,
    position_error_m: float,
    tracking_frequency_radps: float,
    settled_settings: dict[str, Any],
) -> dict[str, Any]:
    """Select tracking or compliant settled-contact controller parameters."""
    if (
        settled_settings["enabled"]
        and contact_latched
        and state in settled_settings["states"]
        and position_error_m
        <= float(settled_settings["maximum_position_error_m"])
    ):
        return {
            "mode": "settled_contact",
            "frequency_radps": float(settled_settings["frequency_radps"]),
            "damping_ratio": float(settled_settings["damping_ratio"]),
        }
    return {
        "mode": "tracking",
        "frequency_radps": float(tracking_frequency_radps),
        "damping_ratio": 1.0,
    }
