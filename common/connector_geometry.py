"""Connector-level geometry checks between observed pose and simulation.

Silhouette agreement can validate the global pose of two parts without
proving that a tight mechanical interface is feasible.  This module evaluates
the connector frames themselves and fails closed when manufacturing data such
as radial clearance, thread pitch, or thread phase is unavailable.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


CONNECTOR_TYPES = {"insert", "screw"}
AXIS_INDICES = {"x": 0, "y": 1, "z": 2}


def _unit(value: Iterable[float], label: str) -> np.ndarray:
    vector = np.asarray(list(value), dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain three finite values")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError(f"{label} cannot be zero")
    return vector / norm


def fit_cylindrical_axis_from_slab(
    vertices: np.ndarray,
    *,
    selector_axis: str,
    minimum: float,
    maximum: float,
    direction_sign: float,
    origin_coordinate: float,
    bins: int = 8,
    quantile: float = 0.02,
    minimum_points_per_bin: int = 50,
) -> dict[str, Any]:
    """Fit a connector centerline from robust cross-section centres.

    The caller selects a cylindrical slab using a known mesh coordinate axis.
    Each cross-section centre is estimated from opposing quantiles rather than
    a vertex mean, which is less sensitive to uneven triangle density. A line
    through those centres supplies the physical axis and an origin at an
    explicitly requested mesh coordinate. This keeps connector geometry tied
    to the asset instead of reverse-engineering it from a solved trajectory.
    """

    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if selector_axis not in AXIS_INDICES:
        raise ValueError(f"selector_axis must be one of {sorted(AXIS_INDICES)}")
    minimum = float(minimum)
    maximum = float(maximum)
    if not np.isfinite([minimum, maximum, origin_coordinate]).all():
        raise ValueError("slab bounds and origin_coordinate must be finite")
    if maximum <= minimum:
        raise ValueError("maximum must exceed minimum")
    if float(direction_sign) == 0.0:
        raise ValueError("direction_sign cannot be zero")
    bins = int(bins)
    if bins < 3:
        raise ValueError("bins must be at least three")
    quantile = float(quantile)
    if not 0.0 < quantile < 0.5:
        raise ValueError("quantile must lie between zero and one half")
    axis_index = AXIS_INDICES[selector_axis]
    other_indices = [index for index in range(3) if index != axis_index]
    selected = points[
        (points[:, axis_index] >= minimum)
        & (points[:, axis_index] <= maximum)
    ]
    if len(selected) < bins * int(minimum_points_per_bin):
        raise ValueError("connector slab contains too few vertices")

    edges = np.linspace(minimum, maximum, bins + 1)
    centres: list[np.ndarray] = []
    counts: list[int] = []
    radii: list[float] = []
    for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (
            (selected[:, axis_index] >= lower)
            & (
                (selected[:, axis_index] <= upper)
                if bin_index == bins - 1
                else (selected[:, axis_index] < upper)
            )
        )
        values = selected[in_bin]
        if len(values) < int(minimum_points_per_bin):
            continue
        axial = float(np.median(values[:, axis_index]))
        transverse = values[:, other_indices]
        low = np.quantile(transverse, quantile, axis=0)
        high = np.quantile(transverse, 1.0 - quantile, axis=0)
        centre_transverse = 0.5 * (low + high)
        centre = np.zeros(3, dtype=np.float64)
        centre[axis_index] = axial
        centre[other_indices] = centre_transverse
        centres.append(centre)
        counts.append(int(len(values)))
        radii.extend(
            np.linalg.norm(
                transverse - centre_transverse[None, :], axis=1
            ).tolist()
        )
    if len(centres) < 3:
        raise ValueError("fewer than three populated connector cross-sections")

    centre_array = np.asarray(centres, dtype=np.float64)
    axial_values = centre_array[:, axis_index]
    design = np.column_stack((axial_values, np.ones(len(axial_values))))
    direction = np.zeros(3, dtype=np.float64)
    direction[axis_index] = 1.0
    origin = np.zeros(3, dtype=np.float64)
    origin[axis_index] = float(origin_coordinate)
    predicted = centre_array.copy()
    for other in other_indices:
        slope, intercept = np.linalg.lstsq(
            design, centre_array[:, other], rcond=None
        )[0]
        direction[other] = float(slope)
        origin[other] = float(slope * origin_coordinate + intercept)
        predicted[:, other] = slope * axial_values + intercept
    direction = _unit(direction, "fitted connector axis")
    if float(direction_sign) < 0.0:
        direction = -direction
    centreline_residuals = np.linalg.norm(
        centre_array[:, other_indices] - predicted[:, other_indices], axis=1
    )
    radius_values = np.asarray(radii, dtype=np.float64)
    return {
        "method": "robust_cross_section_centerline",
        "selector_axis": selector_axis,
        "slab": [minimum, maximum],
        "origin_coordinate": float(origin_coordinate),
        "axis_part": direction.tolist(),
        "origin_raw": origin.tolist(),
        "selected_vertices": int(len(selected)),
        "populated_bins": int(len(centres)),
        "bin_point_counts": counts,
        "cross_section_centres": centre_array.tolist(),
        "centreline_rms_residual": float(
            np.sqrt(np.mean(np.square(centreline_residuals)))
        ),
        "radius_quantiles": {
            "q10": float(np.quantile(radius_values, 0.10)),
            "median": float(np.median(radius_values)),
            "q90": float(np.quantile(radius_values, 0.90)),
        },
    }


def validate_connector_config(
    connectors: dict[str, Any] | None,
    *,
    parts: Iterable[str],
    frame_start: int,
    frame_end: int,
) -> dict[str, Any]:
    """Validate the connector asset contract without requiring dimensions.

    Missing manufacturing dimensions are a valid intermediate state: the pose
    pipeline can still produce a diagnostic report, but the report will mark
    the connector as not ready for simulation.
    """

    if connectors is None:
        return {}
    if not isinstance(connectors, dict):
        raise ValueError("connectors must be an object")
    part_names = set(map(str, parts))
    for name, value in connectors.items():
        if not name or not isinstance(value, dict):
            raise ValueError("connector names must be non-empty objects")
        connector_type = str(value.get("type", "insert"))
        if connector_type not in CONNECTOR_TYPES:
            raise ValueError(f"{name}: unsupported connector type")
        reference_part = str(value.get("reference_part", ""))
        moving_part = str(value.get("moving_part", ""))
        if (
            reference_part not in part_names
            or moving_part not in part_names
            or reference_part == moving_part
        ):
            raise ValueError(f"{name}: invalid connector parts")
        _unit(value["reference_axis_part"], f"{name}.reference_axis_part")
        _unit(value["moving_axis_part"], f"{name}.moving_axis_part")
        for role in ("reference", "moving"):
            part_key = f"{role}_origin_part_m"
            raw_key = f"{role}_origin_raw"
            if part_key not in value and raw_key not in value:
                raise ValueError(
                    f"{name}: {role} connector origin is required"
                )
            origin = value.get(part_key, value.get(raw_key))
            if np.asarray(origin, dtype=np.float64).shape != (3,):
                raise ValueError(f"{name}: invalid {role} connector origin")
        values = value.get(
            "validation_frame_range", [frame_start, frame_end]
        )
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(
                f"{name}: validation_frame_range must be [start, end]"
            )
        start, end = map(int, values)
        if start < frame_start or end > frame_end or start > end:
            raise ValueError(f"{name}: invalid validation_frame_range")
        for key in (
            "maximum_axis_angle_deg",
            "maximum_radial_offset_m",
            "insertion_length_m",
            "radial_clearance_m",
        ):
            if value.get(key) is not None and float(value[key]) <= 0.0:
                raise ValueError(f"{name}: {key} must be positive")
        evidence = value.get("geometry_evidence")
        if evidence is not None:
            if not isinstance(evidence, dict):
                raise ValueError(f"{name}: geometry_evidence must be an object")
            maximum_residual = evidence.get("maximum_rms_residual_m")
            if maximum_residual is None or float(maximum_residual) <= 0.0:
                raise ValueError(
                    f"{name}: geometry_evidence.maximum_rms_residual_m "
                    "must be positive"
                )
            for role in ("reference", "moving"):
                role_evidence = evidence.get(role)
                if not isinstance(role_evidence, dict):
                    raise ValueError(
                        f"{name}: geometry_evidence.{role} is required"
                    )
                residual = role_evidence.get("rms_residual_m")
                if residual is None or float(residual) < 0.0:
                    raise ValueError(
                        f"{name}: geometry_evidence.{role}.rms_residual_m "
                        "must be non-negative"
                    )
        if connector_type == "screw":
            thread = value.get("thread", {})
            if not isinstance(thread, dict):
                raise ValueError(f"{name}: thread must be an object")
            if thread.get("pitch_m") is not None and float(
                thread["pitch_m"]
            ) <= 0.0:
                raise ValueError(f"{name}: thread.pitch_m must be positive")
            handedness = thread.get("handedness")
            if handedness is not None and handedness not in {"left", "right"}:
                raise ValueError(
                    f"{name}: thread.handedness must be left or right"
                )
            for role in ("reference", "moving"):
                key = f"{role}_zero_direction_part"
                if thread.get(key) is not None:
                    _unit(thread[key], f"{name}.thread.{key}")
    return connectors


def connector_origin_part_m(
    connector: dict[str, Any],
    role: str,
    part: str,
    trajectory: dict[str, Any],
) -> np.ndarray:
    """Resolve an asset connector origin into the trajectory part frame."""

    part_key = f"{role}_origin_part_m"
    if part_key in connector:
        return np.asarray(connector[part_key], dtype=np.float64)
    raw_key = f"{role}_origin_raw"
    return float(trajectory["scales"][part]) * (
        np.asarray(connector[raw_key], dtype=np.float64)
        - np.asarray(trajectory["raw_mesh_origins"][part], dtype=np.float64)
    )


def _world_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transform[:3, :3] @ point + transform[:3, 3]


def _signed_angle(
    reference: np.ndarray,
    moving: np.ndarray,
    axis: np.ndarray,
) -> float:
    left = reference - axis * float(np.dot(reference, axis))
    right = moving - axis * float(np.dot(moving, axis))
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm < 1e-9 or right_norm < 1e-9:
        raise ValueError("thread zero direction is parallel to connector axis")
    left /= left_norm
    right /= right_norm
    return float(np.arctan2(
        np.dot(axis, np.cross(left, right)), np.dot(left, right)
    ))


def connector_frame_metrics(
    connector: dict[str, Any],
    trajectory: dict[str, Any],
    frame: int,
) -> dict[str, Any]:
    """Measure axis, origin, and optional thread phase at one frame."""

    reference_part = str(connector["reference_part"])
    moving_part = str(connector["moving_part"])
    records = trajectory["frames"][f"{int(frame):06d}"]["parts"]
    reference_world = np.asarray(
        records[reference_part]["T_world_from_part"], dtype=np.float64
    )
    moving_world = np.asarray(
        records[moving_part]["T_world_from_part"], dtype=np.float64
    )
    reference_axis = reference_world[:3, :3] @ _unit(
        connector["reference_axis_part"], "reference_axis_part"
    )
    moving_axis = moving_world[:3, :3] @ _unit(
        connector["moving_axis_part"], "moving_axis_part"
    )
    dot = float(np.clip(np.dot(reference_axis, moving_axis), -1.0, 1.0))
    if bool(connector.get("allow_axis_flip", False)) and dot < 0.0:
        moving_axis = -moving_axis
        dot = -dot
    axis_angle_deg = float(np.degrees(np.arccos(dot)))
    reference_origin = connector_origin_part_m(
        connector, "reference", reference_part, trajectory
    )
    moving_origin = connector_origin_part_m(
        connector, "moving", moving_part, trajectory
    )
    reference_point = _world_point(reference_world, reference_origin)
    moving_point = _world_point(moving_world, moving_origin)
    delta = moving_point - reference_point
    axial_offset_m = float(np.dot(delta, reference_axis))
    radial = delta - reference_axis * axial_offset_m
    radial_offset_m = float(np.linalg.norm(radial))
    insertion_length = connector.get("insertion_length_m")
    tilt_tip_offset_m = (
        None
        if insertion_length is None
        else float(insertion_length) * float(np.sin(np.radians(axis_angle_deg)))
    )
    effective_radial_error_m = (
        None
        if tilt_tip_offset_m is None
        else radial_offset_m + tilt_tip_offset_m
    )
    result: dict[str, Any] = {
        "frame": int(frame),
        "axis_angle_deg": axis_angle_deg,
        "radial_offset_m": radial_offset_m,
        "axial_offset_m": axial_offset_m,
        "tilt_tip_offset_m": tilt_tip_offset_m,
        "effective_radial_error_m": effective_radial_error_m,
    }
    thread = connector.get("thread", {})
    reference_zero = thread.get("reference_zero_direction_part")
    moving_zero = thread.get("moving_zero_direction_part")
    if reference_zero is not None and moving_zero is not None:
        reference_direction = reference_world[:3, :3] @ _unit(
            reference_zero, "reference_zero_direction_part"
        )
        moving_direction = moving_world[:3, :3] @ _unit(
            moving_zero, "moving_zero_direction_part"
        )
        result["thread_phase_rad"] = _signed_angle(
            reference_direction, moving_direction, reference_axis
        )
    else:
        result["thread_phase_rad"] = None
    return result


def evaluate_connector_trajectory(
    name: str,
    connector: dict[str, Any],
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    """Return a fail-closed insertion/screw readiness report."""

    trajectory_frames = {
        int(frame_id) for frame_id in trajectory.get("frames", {})
    }
    if not trajectory_frames:
        raise ValueError("trajectory has no frames")
    start, end = map(
        int,
        connector.get(
            "validation_frame_range",
            [min(trajectory_frames), max(trajectory_frames)],
        ),
    )
    frames = [
        frame for frame in range(start, end + 1)
        if frame in trajectory_frames
    ]
    rows = [connector_frame_metrics(connector, trajectory, frame) for frame in frames]
    diagnostic_specs = dict(connector.get("diagnostic_ranges", {}))
    diagnostic_frames: set[int] = set()
    for label, values in diagnostic_specs.items():
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"{name}: diagnostic range {label!r} is invalid")
        range_start, range_end = map(int, values)
        if range_start > range_end:
            raise ValueError(f"{name}: diagnostic range {label!r} is invalid")
        diagnostic_frames.update(range(range_start, range_end + 1))
    extra_rows = {
        frame: connector_frame_metrics(connector, trajectory, frame)
        for frame in sorted(diagnostic_frames.difference(frames))
        if frame in trajectory_frames
    }
    failures: list[str] = []
    missing_metadata: list[str] = []
    for key in ("insertion_length_m", "radial_clearance_m"):
        if connector.get(key) is None:
            missing_metadata.append(key)
    maximum_axis = connector.get("maximum_axis_angle_deg")
    maximum_radial = connector.get("maximum_radial_offset_m")
    if maximum_axis is None:
        missing_metadata.append("maximum_axis_angle_deg")
    if maximum_radial is None:
        missing_metadata.append("maximum_radial_offset_m")

    maximum_axis_seen = max((row["axis_angle_deg"] for row in rows), default=None)
    maximum_radial_seen = max((row["radial_offset_m"] for row in rows), default=None)
    maximum_effective = max(
        (
            row["effective_radial_error_m"]
            for row in rows
            if row["effective_radial_error_m"] is not None
        ),
        default=None,
    )
    if maximum_axis is not None and maximum_axis_seen is not None:
        if maximum_axis_seen > float(maximum_axis):
            failures.append("axis_angle_exceeds_limit")
    if maximum_radial is not None and maximum_radial_seen is not None:
        if maximum_radial_seen > float(maximum_radial):
            failures.append("radial_offset_exceeds_limit")

    clearance = connector.get("radial_clearance_m")
    clearance_margin = None
    if clearance is not None and maximum_effective is not None:
        clearance_margin = float(clearance) - maximum_effective
        if clearance_margin < float(connector.get("minimum_clearance_margin_m", 0.0)):
            failures.append("insufficient_radial_clearance")

    thread_report = None
    if str(connector.get("type", "insert")) == "screw":
        thread = dict(connector.get("thread", {}))
        required_thread = (
            "pitch_m",
            "handedness",
            "reference_zero_direction_part",
            "moving_zero_direction_part",
            "entry_phase_rad",
        )
        for key in required_thread:
            if thread.get(key) is None:
                missing_metadata.append(f"thread.{key}")
        phases = [row["thread_phase_rad"] for row in rows]
        helical_residuals: list[float] = []
        if (
            thread.get("pitch_m") is not None
            and all(value is not None for value in phases)
            and len(rows) >= 2
        ):
            unwrapped = np.unwrap(np.asarray(phases, dtype=np.float64))
            handedness = 1.0 if thread.get("handedness") == "right" else -1.0
            pitch = float(thread["pitch_m"])
            axial = np.asarray([row["axial_offset_m"] for row in rows])
            expected = handedness * pitch * np.diff(unwrapped) / (2.0 * np.pi)
            helical_residuals = (np.diff(axial) - expected).tolist()
            maximum_residual = float(
                np.max(np.abs(helical_residuals), initial=0.0)
            )
            allowed = thread.get("maximum_helical_residual_m")
            if allowed is None:
                missing_metadata.append("thread.maximum_helical_residual_m")
            elif maximum_residual > float(allowed):
                failures.append("helical_motion_residual_exceeds_limit")
        else:
            maximum_residual = None
        thread_report = {
            "pitch_m": thread.get("pitch_m"),
            "handedness": thread.get("handedness"),
            "maximum_helical_residual_m": maximum_residual,
            "evaluated_step_count": len(helical_residuals),
        }

    if bool(connector.get("require_collision_validation", True)):
        collision = connector.get("collision_validation")
        if not isinstance(collision, dict) or collision.get("passed") is not True:
            missing_metadata.append("collision_validation.passed")
    geometry_evidence = connector.get("geometry_evidence")
    geometry_evidence_report = None
    if isinstance(geometry_evidence, dict):
        maximum_residual = float(
            geometry_evidence["maximum_rms_residual_m"]
        )
        residuals = {
            role: float(geometry_evidence[role]["rms_residual_m"])
            for role in ("reference", "moving")
        }
        geometry_evidence_report = {
            "maximum_rms_residual_m": maximum_residual,
            "residuals_m": residuals,
            "passed": all(
                residual <= maximum_residual
                for residual in residuals.values()
            ),
            "sources": {
                role: geometry_evidence[role].get("source")
                for role in ("reference", "moving")
            },
        }
        if not geometry_evidence_report["passed"]:
            failures.append("connector_geometry_evidence_failed")
    missing_metadata = sorted(set(missing_metadata))
    if missing_metadata:
        failures.append("missing_manufacturing_or_collision_metadata")
    kinematic_alignment_passed = not any(
        value in failures
        for value in (
            "axis_angle_exceeds_limit",
            "radial_offset_exceeds_limit",
        )
    )
    diagnostic_ranges = {}
    readiness_rows = {int(row["frame"]): row for row in rows}
    for label, values in diagnostic_specs.items():
        range_start, range_end = map(int, values)
        selected_rows = [
            readiness_rows.get(frame, extra_rows.get(frame))
            for frame in range(range_start, range_end + 1)
        ]
        selected_rows = [row for row in selected_rows if row is not None]
        axial_values = [row["axial_offset_m"] for row in selected_rows]
        direction = float(connector.get("insertion_direction", 1.0))
        monotonic_violations = sum(
            1
            for first, second in zip(axial_values, axial_values[1:])
            if direction * (second - first) < -1e-7
        )
        diagnostic_ranges[str(label)] = {
            "frame_range": [range_start, range_end],
            "evaluated_frame_count": len(selected_rows),
            "maximum_axis_angle_deg": max(
                (row["axis_angle_deg"] for row in selected_rows),
                default=None,
            ),
            "median_axis_angle_deg": (
                float(np.median([
                    row["axis_angle_deg"] for row in selected_rows
                ]))
                if selected_rows
                else None
            ),
            "maximum_radial_offset_m": max(
                (row["radial_offset_m"] for row in selected_rows),
                default=None,
            ),
            "median_radial_offset_m": (
                float(np.median([
                    row["radial_offset_m"] for row in selected_rows
                ]))
                if selected_rows
                else None
            ),
            "axial_offset_start_m": (
                float(axial_values[0]) if axial_values else None
            ),
            "axial_offset_end_m": (
                float(axial_values[-1]) if axial_values else None
            ),
            "monotonic_axial_violations": int(monotonic_violations),
        }

    simulation_ready = not failures and bool(rows)
    return {
        "name": str(name),
        "type": str(connector.get("type", "insert")),
        "reference_part": str(connector["reference_part"]),
        "moving_part": str(connector["moving_part"]),
        "validation_frame_range": [start, end],
        "evaluated_frame_count": len(rows),
        "kinematic_alignment_passed": kinematic_alignment_passed,
        "manufacturing_metadata_complete": not missing_metadata,
        "simulation_ready": simulation_ready,
        "failures": failures,
        "missing_metadata": missing_metadata,
        "summary": {
            "maximum_axis_angle_deg": maximum_axis_seen,
            "maximum_radial_offset_m": maximum_radial_seen,
            "maximum_effective_radial_error_m": maximum_effective,
            "radial_clearance_m": clearance,
            "minimum_clearance_margin_m": clearance_margin,
        },
        "thread": thread_report,
        "geometry_evidence": geometry_evidence_report,
        "diagnostic_ranges": diagnostic_ranges,
        "frames": rows,
    }


def evaluate_connectors(
    connectors: dict[str, Any],
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    reports = {
        name: evaluate_connector_trajectory(name, value, trajectory)
        for name, value in connectors.items()
        if value.get("enabled", True)
    }
    return {
        "schema_version": 1,
        "connectors": reports,
        "simulation_ready": bool(reports) and all(
            report["simulation_ready"] for report in reports.values()
        ),
        "failures": {
            name: report["failures"]
            for name, report in reports.items()
            if report["failures"]
        },
    }
