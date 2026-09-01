"""Engine-independent contract for repeatable assembly physics validation."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


SUPPORTED_INTERFACE_TYPES = {
    "cylindrical_insertion",
    "peg_hole",
    "planar_support",
    "compound",
}
PARAMETER_SOURCES = {"measured", "cad", "mesh_fit", "engineering_estimate"}


def _finite_vector(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain three finite values")
    return result


def validate_assembly_interface(
    interface: dict[str, Any], *, parts: set[str]
) -> dict[str, Any]:
    """Validate functional connector metadata without inferring semantics."""
    if not isinstance(interface, dict):
        raise ValueError("assembly_interface must be an object")
    interface_type = str(interface.get("type", ""))
    if interface_type not in SUPPORTED_INTERFACE_TYPES:
        raise ValueError(
            "assembly_interface.type must be one of "
            f"{sorted(SUPPORTED_INTERFACE_TYPES)}"
        )
    reference = str(interface.get("reference_part", ""))
    moving = str(interface.get("moving_part", ""))
    if reference not in parts or moving not in parts or reference == moving:
        raise ValueError("assembly_interface has invalid reference/moving parts")
    source = str(interface.get("parameter_source", ""))
    if source not in PARAMETER_SOURCES:
        raise ValueError(
            f"assembly_interface.parameter_source must be one of {sorted(PARAMETER_SOURCES)}"
        )
    confidence = str(interface.get("confidence", "unknown"))
    if confidence not in {"low", "medium", "high", "unknown"}:
        raise ValueError("assembly_interface.confidence is invalid")

    result = dict(interface)
    result["type"] = interface_type
    result["reference_part"] = reference
    result["moving_part"] = moving
    result["parameter_source"] = source
    result["confidence"] = confidence
    if interface_type == "cylindrical_insertion":
        reference_axis = _finite_vector(
            interface["reference_axis_part"],
            "assembly_interface.reference_axis_part",
        )
        moving_axis = _finite_vector(
            interface["moving_axis_part"],
            "assembly_interface.moving_axis_part",
        )
        if np.linalg.norm(reference_axis) <= 1e-12 or np.linalg.norm(moving_axis) <= 1e-12:
            raise ValueError("assembly interface axes must be non-zero")
        neck_radius = float(interface["reference_outer_radius_m"])
        sleeve_radius = float(interface["moving_inner_radius_m"])
        if neck_radius <= 0.0 or sleeve_radius <= neck_radius:
            raise ValueError(
                "cylindrical insertion requires moving_inner_radius_m greater "
                "than reference_outer_radius_m"
            )
        result["reference_axis_part"] = (
            reference_axis / np.linalg.norm(reference_axis)
        ).tolist()
        result["moving_axis_part"] = (
            moving_axis / np.linalg.norm(moving_axis)
        ).tolist()
        result["radial_clearance_m"] = sleeve_radius - neck_radius
    return result


def generate_standard_perturbation_trials(
    translation_levels_m: list[float], tilt_levels_deg: list[float]
) -> list[dict[str, Any]]:
    """Generate a deterministic, symmetric perturbation suite."""
    translation = sorted({float(value) for value in translation_levels_m})
    tilt = sorted({float(value) for value in tilt_levels_deg})
    if any(value <= 0.0 for value in translation + tilt):
        raise ValueError("assembly validation perturbation levels must be positive")
    trials: list[dict[str, Any]] = [
        {
            "name": "visual_pose_release",
            "kind": "baseline",
            "xy_offset_m": [0.0, 0.0],
            "tilt_deg": [0.0, 0.0],
            "yaw_deg": 0.0,
        }
    ]
    for value in translation:
        millimeters = f"{value * 1000.0:g}mm"
        for axis, index in (("x", 0), ("y", 1)):
            for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                offset = [0.0, 0.0]
                offset[index] = sign * value
                trials.append(
                    {
                        "name": f"offset_{axis}_{sign_name}_{millimeters}",
                        "kind": "translation_perturbation",
                        "magnitude": value,
                        "axis": axis,
                        "xy_offset_m": offset,
                        "tilt_deg": [0.0, 0.0],
                        "yaw_deg": 0.0,
                    }
                )
    for value in tilt:
        degrees = f"{value:g}deg"
        for axis, index in (("x", 0), ("y", 1)):
            for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                angles = [0.0, 0.0]
                angles[index] = sign * value
                trials.append(
                    {
                        "name": f"tilt_{axis}_{sign_name}_{degrees}",
                        "kind": "tilt_perturbation",
                        "magnitude": value,
                        "axis": axis,
                        "xy_offset_m": [0.0, 0.0],
                        "tilt_deg": angles,
                        "yaw_deg": 0.0,
                    }
                )
    return trials


def assembly_validation_settings(simulation: dict[str, Any]) -> dict[str, Any]:
    """Resolve a fail-closed protocol that never mutates the visual pose."""
    raw = dict(simulation.get("assembly_validation", {}))
    if bool(raw.get("allow_pose_mutation", False)):
        raise ValueError("assembly validation cannot enable pose mutation")
    translation_levels = list(raw.get("translation_levels_m", [0.001, 0.002, 0.005]))
    tilt_levels = list(raw.get("tilt_levels_deg", [1.0, 3.0, 5.0]))
    trials = list(
        raw.get(
            "trials",
            generate_standard_perturbation_trials(translation_levels, tilt_levels),
        )
    )
    result = {
        "protocol_version": 1,
        "target_pose_source": "frozen_visual_assembly_pose",
        "allow_pose_mutation": False,
        "external_forces_enabled": False,
        "controller_enabled": False,
        "preload_enabled": False,
        "fixed_joint_enabled": False,
        "initial_height_m": float(raw.get("release_height_m", 0.0)),
        "settle_seconds": float(raw.get("settle_seconds", 3.0)),
        "contact_window_seconds": float(raw.get("contact_window_seconds", 1.0)),
        "minimum_contact_fraction": float(raw.get("minimum_contact_fraction", 0.1)),
        "maximum_contact_gap_seconds": float(raw.get("maximum_contact_gap_seconds", 0.12)),
        "maximum_lateral_error_m": float(raw.get("maximum_lateral_error_m", 0.005)),
        "maximum_axial_error_m": float(raw.get("maximum_axial_error_m", 0.008)),
        "maximum_tilt_error_deg": float(raw.get("maximum_tilt_error_deg", 5.0)),
        "maximum_final_linear_speed_mps": float(raw.get("maximum_final_linear_speed_mps", 0.01)),
        "maximum_final_angular_speed_radps": float(raw.get("maximum_final_angular_speed_radps", 0.1)),
        "translation_levels_m": translation_levels,
        "tilt_levels_deg": tilt_levels,
        "trials": trials,
    }
    if result["initial_height_m"] < 0.0:
        raise ValueError("assembly_validation.release_height_m must be non-negative")
    positive = (
        "settle_seconds",
        "contact_window_seconds",
        "maximum_lateral_error_m",
        "maximum_axial_error_m",
        "maximum_tilt_error_deg",
        "maximum_final_linear_speed_mps",
        "maximum_final_angular_speed_radps",
    )
    if any(float(result[key]) <= 0.0 for key in positive):
        raise ValueError("assembly validation limits and durations must be positive")
    if result["contact_window_seconds"] > result["settle_seconds"]:
        raise ValueError("assembly validation contact window exceeds settle time")
    if not 0.0 <= result["minimum_contact_fraction"] <= 1.0:
        raise ValueError("assembly validation contact fraction must be in [0, 1]")
    if result["maximum_contact_gap_seconds"] < 0.0:
        raise ValueError("assembly validation contact gap must be non-negative")
    if not trials or trials[0].get("kind") != "baseline":
        raise ValueError("assembly validation trials must start with a baseline")
    for trial in trials:
        xy = np.asarray(trial.get("xy_offset_m", []), dtype=np.float64)
        tilt_value = np.asarray(trial.get("tilt_deg", []), dtype=np.float64)
        if (
            not str(trial.get("name", ""))
            or xy.shape != (2,)
            or tilt_value.shape != (2,)
            or not np.isfinite(xy).all()
            or not np.isfinite(tilt_value).all()
        ):
            raise ValueError("assembly validation trials require finite offsets and tilts")
    return result


def summarize_validation_trials(
    trials: list[dict[str, Any]], settings: dict[str, Any]
) -> dict[str, Any]:
    """Separate frozen-pose validity from perturbation recovery capability."""
    baseline = [row for row in trials if row.get("input", {}).get("kind") == "baseline"]
    if len(baseline) != 1:
        raise ValueError("assembly validation requires exactly one baseline result")
    perturbations = [row for row in trials if row not in baseline]
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in perturbations:
        spec = row.get("input", {})
        key = f"{spec.get('kind')}:{float(spec.get('magnitude', 0.0)):g}"
        groups[key].append(bool(row.get("success", False)))
    return {
        "validation_passed": bool(baseline[0].get("success", False)),
        "baseline": baseline[0],
        "perturbation_trial_count": len(perturbations),
        "perturbation_success_count": sum(bool(row.get("success", False)) for row in perturbations),
        "perturbation_success_rate": (
            sum(bool(row.get("success", False)) for row in perturbations)
            / len(perturbations)
            if perturbations
            else None
        ),
        "recovery_by_level": {
            key: {
                "trial_count": len(values),
                "success_count": sum(values),
                "success_rate": sum(values) / len(values),
            }
            for key, values in sorted(groups.items())
        },
        "pose_mutated": False,
        "target_pose_source": settings["target_pose_source"],
        "claim": (
            "Baseline tests physical feasibility of the frozen visual pose; "
            "perturbation results measure a recovery basin and do not establish "
            "ground-truth pose accuracy."
        ),
    }


def validation_readiness_report(
    interface: dict[str, Any] | None,
    *,
    part_info: dict[str, dict[str, Any]],
    transform_checks: dict[str, dict[str, Any]],
    simulation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Report whether a case is runnable and what it is allowed to claim."""
    failures: list[str] = []
    warnings: list[str] = []
    if interface is None:
        failures.append("assembly_interface_missing")
    for part, check in transform_checks.items():
        if not bool(check.get("passed", False)):
            failures.append(f"transform_consistency_failed:{part}")
    for part, info in part_info.items():
        proxy = info.get("collision_proxy", {})
        if proxy.get("type") == "raw":
            warnings.append(f"raw_collision_proxy:{part}")
        if info.get("mass_source") != "measured":
            warnings.append(f"unmeasured_mass:{part}")
    contact_margin = None
    if interface and interface.get("type") == "cylindrical_insertion":
        contact_offset = float((simulation or {}).get("contact_offset_m", 0.0))
        clearance = float(interface["radial_clearance_m"])
        contact_margin = {
            "radial_clearance_m": clearance,
            "contact_offset_per_collider_m": contact_offset,
            "combined_contact_offset_m": 2.0 * contact_offset,
            "remaining_detection_clearance_m": clearance - 2.0 * contact_offset,
        }
        if 2.0 * contact_offset >= clearance:
            failures.append("combined_contact_offset_closes_radial_clearance")
    metric_claim_allowed = bool(
        interface
        and interface.get("parameter_source") in {"measured", "cad"}
        and interface.get("confidence") == "high"
        and not failures
    )
    if interface and not metric_claim_allowed:
        warnings.append("connector_dimensions_not_metric_ground_truth")
    return {
        "runnable": not failures,
        "metric_physical_accuracy_claim_allowed": metric_claim_allowed,
        "failures": failures,
        "warnings": sorted(set(warnings)),
        "contact_margin": contact_margin,
        "scope": (
            "Physics feasibility under declared proxies and assumptions; pose "
            "accuracy requires independent visual or ground-truth evidence."
        ),
    }
