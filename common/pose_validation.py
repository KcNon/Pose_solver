"""Trajectory coverage, per-frame motion, and assembly-entry validation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from common.gicp import transform_angle
from common.symmetry import (
    axis_direction_error_deg,
    symmetry_spec_from_state,
)


def validate_world_poses(
    config: dict,
    world_poses: dict[str, dict[int, np.ndarray]],
) -> tuple[dict, list[str]]:
    report = {}
    failures = []
    for part in config["parts"]:
        validation = config["states"][part].get("validation", {})
        frames = sorted(world_poses[part])
        translations = []
        rotations = []
        axes = []
        symmetry = symmetry_spec_from_state(config["states"][part])
        axis = validation.get(
            "axis_raw", symmetry.axis_raw
        )
        axis = None if axis is None else np.asarray(axis, dtype=np.float64)
        for previous_frame, frame in zip(frames[:-1], frames[1:]):
            previous = world_poses[part][previous_frame]
            current = world_poses[part][frame]
            delta = np.linalg.inv(previous) @ current
            translations.append(
                (frame, float(np.linalg.norm(delta[:3, 3])))
            )
            rotations.append((frame, transform_angle(delta)))
            if axis is not None:
                axes.append(
                    (
                        frame,
                        axis_direction_error_deg(previous, current, axis),
                    )
                )
        limits = {
            "translation_step_m": validation.get("max_translation_step_m"),
            "rotation_step_deg": validation.get("max_rotation_step_deg"),
            "axis_step_deg": validation.get("max_axis_step_deg"),
        }
        measurements = {
            "translation_step_m": translations,
            "rotation_step_deg": rotations,
            "axis_step_deg": axes,
        }
        violations = [
            {
                "frame": frame,
                "metric": metric,
                "value": value,
                "limit": float(limit),
            }
            for metric, limit in limits.items()
            if limit is not None
            for frame, value in measurements[metric]
            if value > float(limit)
        ]

        def maximum(rows: list[tuple[int, float]], unit: str):
            if not rows:
                return None
            frame, value = max(rows, key=lambda row: row[1])
            return {"frame": frame, unit: value}

        report[part] = {
            "symmetry": symmetry.as_dict(),
            "limits": limits,
            "max_translation_step": maximum(translations, "value_m"),
            "max_rotation_step": maximum(rotations, "value_deg"),
            "max_axis_step": maximum(axes, "value_deg"),
            "violations": violations,
        }
        if violations and validation.get("fail_on_violation", True):
            failures.append(
                f"{part}: {len(violations)} trajectory-limit violations"
            )
    return report, failures


def _part_vertices(config: dict, trajectory: dict, part: str) -> np.ndarray:
    """Return raw mesh vertices in the centered, metric part frame."""
    mesh = trimesh.load(
        Path(config["mesh_dir"]) / f"{part}.glb", force="mesh"
    )
    scale = float(trajectory["scales"][part])
    origin = np.asarray(
        trajectory["raw_mesh_origins"][part], dtype=np.float64
    )
    return (np.asarray(mesh.vertices, dtype=np.float64) - origin) * scale


def validate_assembly_entries(
    config: dict,
    trajectory: dict,
) -> tuple[list[dict], list[str]]:
    """Validate that an inserted part clears a container rim before centering.

    This check is deliberately relationship-based rather than object-specific.
    Each rule names a container, a moving part, the container's insertion axis,
    and the maximum radial distance regarded as entering its opening.
    """
    reports: list[dict] = []
    failures: list[str] = []
    for rule in config.get("assembly_validation", []):
        container = str(rule["container"])
        moving = str(rule["moving_part"])
        start, end = map(int, rule["frame_range"])
        axis = np.asarray(
            rule.get("container_axis_part", [0.0, 1.0, 0.0]),
            dtype=np.float64,
        )
        axis /= np.linalg.norm(axis)
        radial_limit = float(rule["max_center_radial_m"])
        clearance_tolerance = float(
            rule.get("rim_clearance_tolerance_m", 0.0)
        )

        container_vertices = _part_vertices(
            config, trajectory, container
        )
        moving_vertices = _part_vertices(config, trajectory, moving)
        rim_coordinate = float(np.max(container_vertices @ axis))

        rows = []
        crossings = []
        previous_inside = False
        for frame in range(start, end + 1):
            key = f"{frame:06d}"
            if key not in trajectory["frames"]:
                continue
            records = trajectory["frames"][key]["parts"]
            container_pose = np.asarray(
                records[container]["T_world_from_part"], dtype=np.float64
            )
            moving_pose = np.asarray(
                records[moving]["T_world_from_part"], dtype=np.float64
            )
            relative = np.linalg.inv(container_pose) @ moving_pose
            center = relative[:3, 3]
            axial_center = float(center @ axis)
            radial_vector = center - axial_center * axis
            radial = float(np.linalg.norm(radial_vector))
            transformed = (
                moving_vertices @ relative[:3, :3].T
                + relative[:3, 3]
            )
            lowest = float(np.min(transformed @ axis))
            inside = radial <= radial_limit
            row = {
                "frame": frame,
                "center_radial_m": radial,
                "center_axial_m": axial_center,
                "lowest_moving_point_axial_m": lowest,
                "container_rim_axial_m": rim_coordinate,
                "inside_entry_radius": inside,
            }
            rows.append(row)
            if inside and not previous_inside:
                row["clears_rim"] = (
                    lowest >= rim_coordinate - clearance_tolerance
                )
                crossings.append(row)
            previous_inside = inside

        crossing_violations = [
            row for row in crossings if not row["clears_rim"]
        ]
        first_crossing_frame = (
            crossings[0]["frame"] if crossings else None
        )
        post_entry_rows = (
            [
                row for row in rows
                if first_crossing_frame is not None
                and row["frame"] >= first_crossing_frame
            ]
        )
        radial_reentries = [
            row for row in post_entry_rows
            if not row["inside_entry_radius"]
        ]
        passed = bool(crossings) and not crossing_violations
        if rule.get("require_stay_centered_after_entry", True):
            passed = passed and not radial_reentries
        name = str(rule.get("name", f"{moving}_into_{container}"))
        report = {
            "name": name,
            "container": container,
            "moving_part": moving,
            "frame_range": [start, end],
            "container_axis_part": axis.tolist(),
            "max_center_radial_m": radial_limit,
            "rim_clearance_tolerance_m": clearance_tolerance,
            "container_rim_axial_m": rim_coordinate,
            "entry_crossings": crossings,
            "crossing_violations": crossing_violations,
            "post_entry_radial_violations": radial_reentries,
            "max_post_entry_radial_m": (
                max(row["center_radial_m"] for row in post_entry_rows)
                if post_entry_rows
                else None
            ),
            "passed": passed,
            "frames": rows,
        }
        reports.append(report)
        if not passed and rule.get("fail_on_violation", True):
            failures.append(
                f"{name}: moving part did not clear the rim before entry"
            )
    return reports, failures


def validate_trajectory(
    config: dict,
    trajectory: dict,
) -> tuple[dict, list[str]]:
    """Validate a serialized trajectory after solving or post-processing."""
    world_poses = {
        part: {
            int(key): np.asarray(
                frame["parts"][part]["T_world_from_part"],
                dtype=np.float64,
            )
            for key, frame in trajectory["frames"].items()
            if part in frame["parts"]
        }
        for part in trajectory["parts"]
    }
    motion, motion_failures = validate_world_poses(config, world_poses)
    assembly, assembly_failures = validate_assembly_entries(
        config, trajectory
    )
    failures = motion_failures + assembly_failures
    return {
        "motion": motion,
        "assembly": assembly,
        "passed": not failures,
        "failures": failures,
    }, failures
