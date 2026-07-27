"""Continuous, relationship-based geometric constraints for pose trajectories.

Two complementary backends are provided:

``insert_into``
    An analytic compatibility backend for a moving proxy and a cylindrical
    container.  It understands that the cavity is free space.

``pairwise_contact``
    A generic sampled-surface backend for any two configured rigid-body
    geometries.  It combines continuous near-field non-penetration, persistent
    contact, and optional axis/axis-origin alignment factors.  The solver only
    sees part IDs, meshes, poses, and factor parameters; it contains no
    object-name or application-specific branches.

Both backends inspect interpolated poses, preventing tunnelling between valid
discrete video frames.  Image evidence remains a separate acceptance gate in
the stage runner; these geometric factors never invoke a physics simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


def _unit(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise ValueError("axis must be non-zero")
    return result / norm


def axis_vector(axis: str | list[float] | np.ndarray) -> np.ndarray:
    if isinstance(axis, str):
        values = {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
        }
        if axis not in values:
            raise ValueError(f"unsupported axis {axis!r}")
        axis = values[axis]
    return _unit(np.asarray(axis, dtype=np.float64))


@dataclass(frozen=True)
class CylindricalContainer:
    axis: np.ndarray
    inner_radius_m: float
    outer_radius_m: float
    floor_top_m: float
    rim_top_m: float
    floor_thickness_m: float

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "CylindricalContainer":
        if str(spec.get("type")) != "cylindrical_container":
            raise ValueError(
                "insert_into currently requires a cylindrical_container proxy"
            )
        result = cls(
            axis=axis_vector(spec.get("axis", "y")),
            inner_radius_m=float(spec["inner_radius_m"]),
            outer_radius_m=float(spec["outer_radius_m"]),
            floor_top_m=float(spec["floor_top_m"]),
            rim_top_m=float(spec["rim_top_m"]),
            floor_thickness_m=float(spec["floor_thickness_m"]),
        )
        if not 0.0 < result.inner_radius_m < result.outer_radius_m:
            raise ValueError("invalid cylindrical-container radii")
        if (
            result.floor_thickness_m <= 0.0
            or result.rim_top_m <= result.floor_top_m
        ):
            raise ValueError("invalid cylindrical-container axial dimensions")
        return result


@dataclass
class SampledSurface:
    """A deterministic point/normal representation of an arbitrary surface."""

    points: np.ndarray
    normals: np.ndarray

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64)
        self.normals = np.asarray(self.normals, dtype=np.float64)
        if (
            self.points.ndim != 2
            or self.points.shape[1] != 3
            or self.normals.shape != self.points.shape
            or len(self.points) == 0
        ):
            raise ValueError("sampled surfaces require matching non-empty Nx3 arrays")
        lengths = np.linalg.norm(self.normals, axis=1)
        valid = lengths > 1e-12
        if not np.all(valid):
            raise ValueError("sampled surface normals must be non-zero")
        self.normals = self.normals / lengths[:, None]
        self.tree = cKDTree(self.points)


def interpolate_pose(first: np.ndarray, second: np.ndarray, amount: float) -> np.ndarray:
    """Interpolate a rigid transform with linear translation and rotation SLERP."""
    amount = float(amount)
    result = np.eye(4, dtype=np.float64)
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    result[:3, 3] = (1.0 - amount) * first[:3, 3] + amount * second[:3, 3]
    rotations = Rotation.from_matrix(
        np.stack((first[:3, :3], second[:3, :3]))
    )
    result[:3, :3] = Slerp([0.0, 1.0], rotations)([amount]).as_matrix()[0]
    return result


def pose_delta(reference: np.ndarray, pose: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64)
    rotation = pose[:3, :3] @ reference[:3, :3].T
    return np.concatenate(
        (
            pose[:3, 3] - reference[:3, 3],
            Rotation.from_matrix(rotation).as_rotvec(),
        )
    )


def apply_container_delta(
    pose: np.ndarray,
    translation_m: np.ndarray,
    rotation_rad: np.ndarray,
) -> np.ndarray:
    """Apply a container-frame delta, rotating about the moving-part origin."""
    result = np.asarray(pose, dtype=np.float64).copy()
    result[:3, 3] += np.asarray(translation_m, dtype=np.float64)
    result[:3, :3] = (
        Rotation.from_rotvec(np.asarray(rotation_rad, dtype=np.float64)).as_matrix()
        @ result[:3, :3]
    )
    return result


def _point_coordinates(
    relative_pose: np.ndarray,
    moving_points: np.ndarray,
    container: CylindricalContainer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = (
        np.asarray(moving_points, dtype=np.float64)
        @ np.asarray(relative_pose, dtype=np.float64)[:3, :3].T
        + np.asarray(relative_pose, dtype=np.float64)[:3, 3]
    )
    axial = points @ container.axis
    radial_vectors = points - axial[:, None] * container.axis[None, :]
    radial = np.linalg.norm(radial_vectors, axis=1)
    return points, axial, radial


def insertion_pose_metrics(
    relative_pose: np.ndarray,
    moving_points: np.ndarray,
    container: CylindricalContainer,
    *,
    containment_active: bool,
    contact_tolerance_m: float = 0.0,
) -> dict[str, Any]:
    """Measure wall/floor penetration for one relative moving-part pose.

    ``contact_tolerance_m`` is a numerical allowance, not a requested gap.
    Penetrations shallower than that tolerance are reported as zero.
    """
    _, axial, radial = _point_coordinates(
        relative_pose, moving_points, container
    )
    floor_bottom = container.floor_top_m - container.floor_thickness_m

    wall = (
        (axial >= container.floor_top_m)
        & (axial <= container.rim_top_m)
        & (radial >= container.inner_radius_m)
        & (radial <= container.outer_radius_m)
    )
    wall_depths = np.zeros_like(axial)
    if np.any(wall):
        wall_depths[wall] = np.minimum.reduce(
            (
                axial[wall] - container.floor_top_m,
                container.rim_top_m - axial[wall],
                radial[wall] - container.inner_radius_m,
                container.outer_radius_m - radial[wall],
            )
        )

    floor = (
        (axial >= floor_bottom)
        & (axial <= container.floor_top_m)
        & (radial <= container.outer_radius_m)
    )
    floor_depths = np.zeros_like(axial)
    if np.any(floor):
        floor_depths[floor] = np.minimum.reduce(
            (
                axial[floor] - floor_bottom,
                container.floor_top_m - axial[floor],
                container.outer_radius_m - radial[floor],
            )
        )

    # Once the part has entered through the opening, these envelope terms also
    # catch a pose that skips completely through a thin wall/floor between
    # samples.  A flange may remain outside provided it stays above the rim.
    corridor_depths = np.zeros_like(axial)
    floor_envelope_depth = 0.0
    if containment_active:
        below_rim = axial < container.rim_top_m
        if np.any(below_rim):
            radial_excess = np.maximum(
                radial[below_rim] - container.inner_radius_m, 0.0
            )
            axial_overlap = np.maximum(
                container.rim_top_m - axial[below_rim], 0.0
            )
            corridor_depths[below_rim] = np.minimum(
                radial_excess, axial_overlap
            )
        floor_envelope_depth = max(
            container.floor_top_m - float(np.min(axial)), 0.0
        )

    all_depths = np.maximum.reduce(
        (wall_depths, floor_depths, corridor_depths)
    )
    tolerance = max(float(contact_tolerance_m), 0.0)
    effective = np.maximum(all_depths - tolerance, 0.0)
    floor_effective = max(floor_envelope_depth - tolerance, 0.0)
    maximum = max(float(np.max(effective, initial=0.0)), floor_effective)
    positive = effective[effective > 0.0]
    rms = (
        float(np.sqrt(np.mean(positive**2)))
        if len(positive)
        else 0.0
    )
    rms = max(rms, floor_effective)
    center = np.asarray(relative_pose, dtype=np.float64)[:3, 3]
    center_axial = float(center @ container.axis)
    center_radial = float(
        np.linalg.norm(center - center_axial * container.axis)
    )
    return {
        "max_penetration_m": maximum,
        "rms_penetration_m": rms,
        "penetrating_samples": int(np.count_nonzero(effective))
        + int(floor_effective > 0.0),
        "wall_samples": int(np.count_nonzero(wall)),
        "floor_samples": int(np.count_nonzero(floor)),
        "corridor_samples": int(np.count_nonzero(corridor_depths > tolerance)),
        "floor_envelope_penetration_m": floor_effective,
        "minimum_axial_m": float(np.min(axial)),
        "center_axial_m": center_axial,
        "center_radial_m": center_radial,
        "containment_active": bool(containment_active),
    }


def infer_entry_frame(
    poses: dict[int, np.ndarray],
    moving_points: np.ndarray,
    container: CylindricalContainer,
    entry_center_radius_m: float,
) -> int | None:
    """Find the first centered pose from which the part stays centered."""
    flags = []
    for frame in sorted(poses):
        center = np.asarray(poses[frame], dtype=np.float64)[:3, 3]
        axial = float(center @ container.axis)
        radial = float(np.linalg.norm(center - axial * container.axis))
        flags.append((frame, radial <= float(entry_center_radius_m)))
    for index, (frame, inside) in enumerate(flags):
        if inside and all(value for _, value in flags[index:]):
            return frame
    return None


def evaluate_insertion_trajectory(
    poses: dict[int, np.ndarray],
    moving_points: np.ndarray,
    container: CylindricalContainer,
    *,
    substeps: int,
    entry_center_radius_m: float,
    contact_tolerance_m: float,
    entry_frame: int | None = None,
) -> dict[str, Any]:
    frames = sorted(poses)
    if not frames:
        raise ValueError("cannot evaluate an empty trajectory")
    resolved_entry = (
        infer_entry_frame(
            poses, moving_points, container, entry_center_radius_m
        )
        if entry_frame is None
        else int(entry_frame)
    )
    rows: list[dict[str, Any]] = []
    sample_count = max(1, int(substeps))
    for index, frame in enumerate(frames):
        if index + 1 < len(frames):
            amounts = np.linspace(0.0, 1.0, sample_count + 1)[:-1]
            following = frames[index + 1]
        else:
            amounts = np.asarray([0.0])
            following = frame
        for amount in amounts:
            pose = (
                poses[frame]
                if following == frame
                else interpolate_pose(poses[frame], poses[following], amount)
            )
            sample_frame = float(frame + amount * (following - frame))
            metrics = insertion_pose_metrics(
                pose,
                moving_points,
                container,
                containment_active=(
                    resolved_entry is not None
                    and sample_frame >= float(resolved_entry)
                ),
                contact_tolerance_m=contact_tolerance_m,
            )
            metrics.update(
                {
                    "frame": sample_frame,
                    "segment_start": frame,
                    "interpolation": float(amount),
                }
            )
            rows.append(metrics)
    worst = max(rows, key=lambda item: item["max_penetration_m"])
    violating = [
        row for row in rows if row["max_penetration_m"] > 0.0
    ]
    return {
        "entry_frame": resolved_entry,
        "substeps_per_frame": sample_count,
        "sample_count": len(rows),
        "violating_samples": len(violating),
        "max_penetration_m": float(worst["max_penetration_m"]),
        "rms_penetration_m": float(
            np.sqrt(
                np.mean(
                    np.square(
                        [row["rms_penetration_m"] for row in rows]
                    )
                )
            )
        ),
        "worst_sample": worst,
        "samples": rows,
    }


def _local_geometric_cost(
    frame_index: int,
    frames: list[int],
    poses: dict[int, np.ndarray],
    moving_points: np.ndarray,
    container: CylindricalContainer,
    config: dict[str, Any],
    entry_frame: int | None,
) -> float:
    scale = max(float(config.get("penetration_scale_m", 0.005)), 1e-6)
    substeps = max(1, int(config.get("continuous_substeps", 8)))
    indices = sorted(
        set(
            max(0, min(len(frames) - 1, value))
            for value in (frame_index - 1, frame_index)
        )
    )
    values = []
    for index in indices:
        first_frame = frames[index]
        if index + 1 < len(frames):
            second_frame = frames[index + 1]
            amounts = np.linspace(0.0, 1.0, substeps + 1)
        else:
            second_frame = first_frame
            amounts = [0.0]
        for amount in amounts:
            pose = (
                poses[first_frame]
                if first_frame == second_frame
                else interpolate_pose(
                    poses[first_frame], poses[second_frame], amount
                )
            )
            sample_frame = first_frame + amount * (
                second_frame - first_frame
            )
            item = insertion_pose_metrics(
                pose,
                moving_points,
                container,
                containment_active=(
                    entry_frame is not None
                    and sample_frame >= float(entry_frame)
                ),
                contact_tolerance_m=float(
                    config.get("contact_tolerance_m", 0.001)
                ),
            )
            value = (
                (item["max_penetration_m"] / scale) ** 2
                + 0.25 * (item["rms_penetration_m"] / scale) ** 2
            )
            seat_weight = float(config.get("seat_axis_alignment_weight", 0.0))
            seat_activation = float(
                config.get("seat_activation_height_m", 0.02)
            )
            if (
                seat_weight > 0.0
                and item["containment_active"]
                and item["minimum_axial_m"]
                <= container.floor_top_m + seat_activation
            ):
                moving_axis = axis_vector(
                    config.get("moving_axis_part", [0.0, 1.0, 0.0])
                )
                transformed_axis = pose[:3, :3] @ moving_axis
                angle_deg = float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                abs(float(transformed_axis @ container.axis)),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                tolerance_deg = float(
                    config.get("seat_axis_tolerance_deg", 1.0)
                )
                scale_deg = max(
                    float(config.get("seat_axis_scale_deg", 5.0)), 1e-6
                )
                value += seat_weight * (
                    max(angle_deg - tolerance_deg, 0.0) / scale_deg
                ) ** 2
            values.append(value)
    return float(np.mean(values)) if values else 0.0


def refine_insert_trajectory(
    initial_poses: dict[int, np.ndarray],
    moving_points: np.ndarray,
    container: CylindricalContainer,
    config: dict[str, Any],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Reduce continuous insert/container collisions with bounded SE(3) search."""
    frames = sorted(initial_poses)
    poses = {
        frame: np.asarray(initial_poses[frame], dtype=np.float64).copy()
        for frame in frames
    }
    baseline = {frame: pose.copy() for frame, pose in poses.items()}
    entry_frame = config.get("entry_frame")
    if entry_frame is None:
        entry_frame = infer_entry_frame(
            poses,
            moving_points,
            container,
            float(config["entry_center_radius_m"]),
        )
    maximum_translation = float(
        config.get("maximum_translation_delta_m", 0.02)
    )
    maximum_rotation = float(config.get("maximum_rotation_delta_deg", 10.0))
    prior_weight = float(config.get("prior_weight", 0.01))
    prior_translation_weight = float(
        config.get("prior_translation_weight", prior_weight)
    )
    prior_rotation_weight = float(
        config.get("prior_rotation_weight", prior_weight)
    )
    smoothness_weight = float(config.get("smoothness_weight", 0.02))
    translation_steps = [
        float(value)
        for value in config.get(
            "translation_steps_m", [0.008, 0.004, 0.002, 0.001]
        )
    ]
    rotation_steps = [
        float(value)
        for value in config.get(
            "rotation_steps_deg", [4.0, 2.0, 1.0, 0.5]
        )
    ]
    optimize_start, optimize_end = map(
        int, config.get("optimize_frame_range", [frames[0], frames[-1]])
    )
    optimizable = [
        index
        for index, frame in enumerate(frames)
        if optimize_start <= frame <= optimize_end
    ]
    evaluations = 0

    def cost(index: int, candidate: np.ndarray) -> float:
        nonlocal evaluations
        frame = frames[index]
        delta = pose_delta(baseline[frame], candidate)
        translation_norm = float(np.linalg.norm(delta[:3]))
        rotation_deg = float(np.degrees(np.linalg.norm(delta[3:])))
        if (
            translation_norm > maximum_translation + 1e-9
            or rotation_deg > maximum_rotation + 1e-9
        ):
            return float("inf")
        old = poses[frame]
        poses[frame] = candidate
        geometric = _local_geometric_cost(
            index,
            frames,
            poses,
            moving_points,
            container,
            config,
            entry_frame,
        )
        poses[frame] = old
        regularizer = (
            prior_translation_weight
            * (translation_norm / max(maximum_translation, 1e-9)) ** 2
            + prior_rotation_weight
            * (rotation_deg / max(maximum_rotation, 1e-9)) ** 2
        )
        if 0 < index < len(frames) - 1:
            prior_delta = pose_delta(baseline[frame], candidate)
            left_delta = pose_delta(
                baseline[frames[index - 1]], poses[frames[index - 1]]
            )
            right_delta = pose_delta(
                baseline[frames[index + 1]], poses[frames[index + 1]]
            )
            mean_delta = 0.5 * (left_delta + right_delta)
            regularizer += smoothness_weight * (
                (
                    np.linalg.norm(prior_delta[:3] - mean_delta[:3])
                    / max(maximum_translation, 1e-9)
                )
                ** 2
                + (
                    np.degrees(
                        np.linalg.norm(prior_delta[3:] - mean_delta[3:])
                    )
                    / max(maximum_rotation, 1e-9)
                )
                ** 2
            )
        evaluations += 1
        return float(geometric + regularizer)

    stages = max(len(translation_steps), len(rotation_steps))
    for stage in range(stages):
        translation_step = translation_steps[
            min(stage, len(translation_steps) - 1)
        ]
        rotation_step = rotation_steps[min(stage, len(rotation_steps) - 1)]
        for order in (optimizable, list(reversed(optimizable))):
            for index in order:
                frame = frames[index]
                best = poses[frame]
                best_cost = cost(index, best)
                for direction_index in range(3):
                    direction = np.zeros(3, dtype=np.float64)
                    direction[direction_index] = 1.0
                    for sign in (-1.0, 1.0):
                        candidate = apply_container_delta(
                            best,
                            sign * translation_step * direction,
                            np.zeros(3),
                        )
                        value = cost(index, candidate)
                        if value + 1e-12 < best_cost:
                            best, best_cost = candidate, value
                moving_axis = axis_vector(
                    config.get("moving_axis_part", [0.0, 1.0, 0.0])
                )
                axis_in_container = best[:3, :3] @ moving_axis
                rotation_directions = [
                    direction
                    for direction in np.eye(3)
                    if abs(float(direction @ axis_in_container)) < 0.95
                ]
                for direction in rotation_directions:
                    for sign in (-1.0, 1.0):
                        candidate = apply_container_delta(
                            best,
                            np.zeros(3),
                            np.deg2rad(sign * rotation_step) * direction,
                        )
                        value = cost(index, candidate)
                        if value + 1e-12 < best_cost:
                            best, best_cost = candidate, value
                poses[frame] = best

    before = evaluate_insertion_trajectory(
        baseline,
        moving_points,
        container,
        substeps=int(config.get("continuous_substeps", 8)),
        entry_center_radius_m=float(config["entry_center_radius_m"]),
        contact_tolerance_m=float(config.get("contact_tolerance_m", 0.001)),
        entry_frame=entry_frame,
    )
    after = evaluate_insertion_trajectory(
        poses,
        moving_points,
        container,
        substeps=int(config.get("continuous_substeps", 8)),
        entry_center_radius_m=float(config["entry_center_radius_m"]),
        contact_tolerance_m=float(config.get("contact_tolerance_m", 0.001)),
        entry_frame=entry_frame,
    )
    corrections = {
        f"{frame:06d}": {
            "translation_delta_m": pose_delta(
                baseline[frame], poses[frame]
            )[:3].tolist(),
            "translation_delta_norm_m": float(
                np.linalg.norm(pose_delta(baseline[frame], poses[frame])[:3])
            ),
            "rotation_delta_deg": float(
                np.degrees(
                    np.linalg.norm(
                        pose_delta(baseline[frame], poses[frame])[3:]
                    )
                )
            ),
        }
        for frame in frames
    }
    return poses, {
        "entry_frame": entry_frame,
        "evaluations": evaluations,
        "before": before,
        "proposed_after": after,
        "corrections": corrections,
    }


def _directed_surface_metrics(
    source: SampledSurface,
    target: SampledSurface,
    transform_target_from_source: np.ndarray,
    *,
    contact_tolerance_m: float,
    near_field_m: float,
    maximum_normal_dot: float,
) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(transform_target_from_source, dtype=np.float64)
    points = source.points @ transform[:3, :3].T + transform[:3, 3]
    distances, indices = target.tree.query(points, k=1, workers=1)
    offsets = points - target.points[indices]
    source_normals = source.normals @ transform[:3, :3].T
    normal_dot = np.einsum(
        "ij,ij->i", source_normals, target.normals[indices]
    )
    signed_plane_distance = np.einsum(
        "ij,ij->i", offsets, target.normals[indices]
    )
    penetration = np.maximum(
        -signed_plane_distance - max(float(contact_tolerance_m), 0.0),
        0.0,
    )
    penetration[distances > max(float(near_field_m), 1e-6)] = 0.0
    # Surface crossings have opposing local normals.  Rejecting parallel or
    # orthogonal nearest patches avoids classifying two nearby but separated
    # finite surfaces as one lying "behind" the other's infinite tangent plane.
    penetration[normal_dot > float(maximum_normal_dot)] = 0.0
    return np.asarray(distances, dtype=np.float64), penetration


def surface_pair_pose_metrics(
    relative_pose: np.ndarray,
    moving_surface: SampledSurface,
    reference_surface: SampledSurface,
    *,
    contact_tolerance_m: float = 0.001,
    near_field_m: float = 0.03,
    penetration_quantile: float = 0.99,
    contact_quantile: float = 0.005,
    minimum_penetrating_samples: int = 5,
    maximum_normal_dot: float = -0.1,
) -> dict[str, Any]:
    """Approximate symmetric penetration/contact for arbitrary open surfaces.

    Reconstructions are frequently non-watertight, so a global inside/outside
    query is unreliable.  This metric uses local oriented tangent planes in a
    bounded near field.  Sampling in both directions makes it sensitive to
    either surface crossing the other, while the robust quantile prevents one
    noisy triangle from controlling a trajectory.
    """
    relative = np.asarray(relative_pose, dtype=np.float64)
    moving_distances, moving_depths = _directed_surface_metrics(
        moving_surface,
        reference_surface,
        relative,
        contact_tolerance_m=contact_tolerance_m,
        near_field_m=near_field_m,
        maximum_normal_dot=maximum_normal_dot,
    )
    reference_distances, reference_depths = _directed_surface_metrics(
        reference_surface,
        moving_surface,
        np.linalg.inv(relative),
        contact_tolerance_m=contact_tolerance_m,
        near_field_m=near_field_m,
        maximum_normal_dot=maximum_normal_dot,
    )
    depths = np.concatenate((moving_depths, reference_depths))
    positive = depths[depths > 0.0]
    minimum_support = max(1, int(minimum_penetrating_samples))
    quantile = float(np.clip(penetration_quantile, 0.5, 1.0))
    if len(positive) >= minimum_support:
        robust_maximum = float(np.quantile(positive, quantile))
        rms = float(np.sqrt(np.mean(np.square(positive))))
    else:
        robust_maximum = 0.0
        rms = 0.0
    distances = np.concatenate((moving_distances, reference_distances))
    contact_q = float(np.clip(contact_quantile, 0.0, 0.5))
    contact_distance = float(np.quantile(distances, contact_q))
    return {
        "max_penetration_m": robust_maximum,
        "raw_max_penetration_m": float(np.max(depths, initial=0.0)),
        "rms_penetration_m": rms,
        "penetrating_samples": int(len(positive)),
        "moving_penetrating_samples": int(np.count_nonzero(moving_depths)),
        "reference_penetrating_samples": int(
            np.count_nonzero(reference_depths)
        ),
        "contact_distance_m": contact_distance,
        "contact_support_fraction": float(
            np.mean(distances <= max(float(near_field_m), 1e-6))
        ),
    }


def _aligned_axis_target(
    current: np.ndarray,
    reference: np.ndarray,
    allow_flip: bool,
) -> np.ndarray:
    target = _unit(reference)
    if allow_flip and float(np.dot(current, target)) < 0.0:
        target = -target
    return target


def pairwise_alignment_metrics(
    relative_pose: np.ndarray,
    *,
    reference_axis: np.ndarray | list[float] | str | None,
    moving_axis: np.ndarray | list[float] | str | None,
    allow_axis_flip: bool = False,
) -> dict[str, float | None]:
    """Return generic axis-angle and axis-origin offset measurements."""
    if reference_axis is None or moving_axis is None:
        return {"axis_angle_deg": None, "axis_offset_m": None}
    fixed = axis_vector(reference_axis)
    moving = (
        np.asarray(relative_pose, dtype=np.float64)[:3, :3]
        @ axis_vector(moving_axis)
    )
    target = _aligned_axis_target(moving, fixed, allow_axis_flip)
    angle = float(
        np.degrees(
            np.arccos(np.clip(float(np.dot(moving, target)), -1.0, 1.0))
        )
    )
    translation = np.asarray(relative_pose, dtype=np.float64)[:3, 3]
    radial = translation - float(np.dot(translation, fixed)) * fixed
    return {
        "axis_angle_deg": angle,
        "axis_offset_m": float(np.linalg.norm(radial)),
    }


def project_pairwise_alignment(
    relative_pose: np.ndarray,
    *,
    reference_axis: np.ndarray | list[float] | str | None,
    moving_axis: np.ndarray | list[float] | str | None,
    allow_axis_flip: bool,
    target_axis_offset_m: float,
) -> np.ndarray:
    """Project a pose onto generic coaxial bounds while retaining axial/yaw DOFs."""
    result = np.asarray(relative_pose, dtype=np.float64).copy()
    if reference_axis is None or moving_axis is None:
        return result
    fixed = axis_vector(reference_axis)
    moving = result[:3, :3] @ axis_vector(moving_axis)
    target = _aligned_axis_target(moving, fixed, allow_axis_flip)
    dot = float(np.clip(np.dot(moving, target), -1.0, 1.0))
    cross = np.cross(moving, target)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm > 1e-10:
        increment = Rotation.from_rotvec(
            cross / cross_norm * float(np.arccos(dot))
        ).as_matrix()
        result[:3, :3] = increment @ result[:3, :3]
    elif dot < 0.0:
        seed = np.asarray([1.0, 0.0, 0.0])
        if abs(float(np.dot(seed, moving))) > 0.85:
            seed = np.asarray([0.0, 1.0, 0.0])
        perpendicular = _unit(np.cross(moving, seed))
        result[:3, :3] = (
            Rotation.from_rotvec(np.pi * perpendicular).as_matrix()
            @ result[:3, :3]
        )
    translation = result[:3, 3]
    axial = float(np.dot(translation, fixed)) * fixed
    radial = translation - axial
    radial_norm = float(np.linalg.norm(radial))
    target_offset = max(float(target_axis_offset_m), 0.0)
    if radial_norm > target_offset:
        radial *= target_offset / radial_norm
        result[:3, 3] = axial + radial
    return result


def evaluate_surface_contact_trajectory(
    poses: dict[int, np.ndarray],
    moving_surface: SampledSurface,
    reference_surface: SampledSurface,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate arbitrary pairwise surface/contact factors continuously."""
    frames = sorted(poses)
    if not frames:
        raise ValueError("cannot evaluate an empty trajectory")
    substeps = max(1, int(config.get("continuous_substeps", 4)))
    contact_start = int(config.get("contact_start_frame", frames[0]))
    maximum_contact_gap = float(config.get("maximum_contact_gap_m", 0.004))
    axis_tolerance = float(config.get("axis_tolerance_deg", 180.0))
    offset_tolerance = float(config.get("maximum_axis_offset_m", float("inf")))
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        if index + 1 < len(frames):
            following = frames[index + 1]
            amounts = np.linspace(0.0, 1.0, substeps + 1)[:-1]
        else:
            following = frame
            amounts = np.asarray([0.0])
        for amount in amounts:
            pose = (
                poses[frame]
                if following == frame
                else interpolate_pose(poses[frame], poses[following], amount)
            )
            sample_frame = float(frame + amount * (following - frame))
            surface = surface_pair_pose_metrics(
                pose,
                moving_surface,
                reference_surface,
                contact_tolerance_m=float(
                    config.get("contact_tolerance_m", 0.001)
                ),
                near_field_m=float(config.get("near_field_m", 0.03)),
                penetration_quantile=float(
                    config.get("penetration_quantile", 0.99)
                ),
                contact_quantile=float(config.get("contact_quantile", 0.005)),
                minimum_penetrating_samples=int(
                    config.get("minimum_penetrating_samples", 5)
                ),
                maximum_normal_dot=float(
                    config.get("maximum_normal_dot", -0.1)
                ),
            )
            alignment = pairwise_alignment_metrics(
                pose,
                reference_axis=config.get("reference_axis_part"),
                moving_axis=config.get("moving_axis_part"),
                allow_axis_flip=bool(config.get("allow_axis_flip", False)),
            )
            contact_active = sample_frame >= float(contact_start)
            surface.update(alignment)
            surface.update(
                {
                    "frame": sample_frame,
                    "segment_start": frame,
                    "interpolation": float(amount),
                    "contact_active": contact_active,
                    "contact_gap_violation_m": (
                        max(
                            surface["contact_distance_m"]
                            - maximum_contact_gap,
                            0.0,
                        )
                        if contact_active
                        else 0.0
                    ),
                    "axis_angle_violation_deg": (
                        max(
                            float(alignment["axis_angle_deg"])
                            - axis_tolerance,
                            0.0,
                        )
                        if contact_active
                        and alignment["axis_angle_deg"] is not None
                        else 0.0
                    ),
                    "axis_offset_violation_m": (
                        max(
                            float(alignment["axis_offset_m"])
                            - offset_tolerance,
                            0.0,
                        )
                        if contact_active
                        and alignment["axis_offset_m"] is not None
                        else 0.0
                    ),
                }
            )
            rows.append(surface)
    maximum_penetration = max(row["max_penetration_m"] for row in rows)
    violating = [
        row
        for row in rows
        if (
            row["max_penetration_m"] > 0.0
            or row["contact_gap_violation_m"] > 0.0
            or row["axis_angle_violation_deg"] > 0.0
            or row["axis_offset_violation_m"] > 0.0
        )
    ]
    return {
        "substeps_per_frame": substeps,
        "sample_count": len(rows),
        "violating_samples": len(violating),
        "max_penetration_m": float(maximum_penetration),
        "max_contact_gap_violation_m": float(
            max(row["contact_gap_violation_m"] for row in rows)
        ),
        "max_axis_angle_violation_deg": float(
            max(row["axis_angle_violation_deg"] for row in rows)
        ),
        "max_axis_offset_violation_m": float(
            max(row["axis_offset_violation_m"] for row in rows)
        ),
        "max_axis_angle_deg": float(
            max(
                (
                    row["axis_angle_deg"]
                    for row in rows
                    if row["axis_angle_deg"] is not None
                ),
                default=0.0,
            )
        ),
        "max_axis_offset_m": float(
            max(
                (
                    row["axis_offset_m"]
                    for row in rows
                    if row["axis_offset_m"] is not None
                ),
                default=0.0,
            )
        ),
        "worst_penetration_sample": max(
            rows, key=lambda item: item["max_penetration_m"]
        ),
        "samples": rows,
    }


def _surface_contact_local_cost(
    frame_index: int,
    frames: list[int],
    poses: dict[int, np.ndarray],
    moving_surface: SampledSurface,
    reference_surface: SampledSurface,
    config: dict[str, Any],
) -> float:
    indices = sorted(
        set(
            max(0, min(len(frames) - 1, value))
            for value in (frame_index - 1, frame_index)
        )
    )
    local = {
        frames[index]: poses[frames[index]]
        for index in indices
    }
    for index in indices:
        if index + 1 < len(frames):
            local[frames[index + 1]] = poses[frames[index + 1]]
    metrics = evaluate_surface_contact_trajectory(
        local, moving_surface, reference_surface, config
    )
    penetration_scale = max(
        float(config.get("penetration_scale_m", 0.005)), 1e-6
    )
    contact_scale = max(float(config.get("contact_scale_m", 0.004)), 1e-6)
    axis_scale = max(float(config.get("axis_scale_deg", 5.0)), 1e-6)
    offset_scale = max(float(config.get("axis_offset_scale_m", 0.02)), 1e-6)
    rows = metrics["samples"]
    return float(
        np.mean(
            [
                (row["max_penetration_m"] / penetration_scale) ** 2
                + 0.25
                * (row["rms_penetration_m"] / penetration_scale) ** 2
                + float(config.get("contact_weight", 0.25))
                * (row["contact_gap_violation_m"] / contact_scale) ** 2
                + float(config.get("axis_alignment_weight", 0.0))
                * (row["axis_angle_violation_deg"] / axis_scale) ** 2
                + float(config.get("axis_offset_weight", 0.0))
                * (row["axis_offset_violation_m"] / offset_scale) ** 2
                for row in rows
            ]
        )
    )


def refine_surface_contact_trajectory(
    initial_poses: dict[int, np.ndarray],
    moving_surface: SampledSurface,
    reference_surface: SampledSurface,
    config: dict[str, Any],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Bounded SE(3) refinement for a generic persistent rigid-body contact."""
    frames = sorted(initial_poses)
    poses = {
        frame: np.asarray(initial_poses[frame], dtype=np.float64).copy()
        for frame in frames
    }
    baseline = {frame: value.copy() for frame, value in poses.items()}
    maximum_translation = float(
        config.get("maximum_translation_delta_m", 0.04)
    )
    maximum_rotation = float(config.get("maximum_rotation_delta_deg", 10.0))
    prior_weight = float(config.get("prior_weight", 0.01))
    smoothness_weight = float(config.get("smoothness_weight", 0.02))
    translation_steps = [
        float(value)
        for value in config.get(
            "translation_steps_m", [0.01, 0.005, 0.002, 0.001]
        )
    ]
    rotation_steps = [
        float(value)
        for value in config.get(
            "rotation_steps_deg", [4.0, 2.0, 1.0, 0.5]
        )
    ]
    optimize_start, optimize_end = map(
        int, config.get("optimize_frame_range", [frames[0], frames[-1]])
    )
    optimizable = [
        index
        for index, frame in enumerate(frames)
        if optimize_start <= frame <= optimize_end
    ]
    evaluations = 0

    def cost(index: int, candidate: np.ndarray) -> float:
        nonlocal evaluations
        frame = frames[index]
        delta = pose_delta(baseline[frame], candidate)
        translation_norm = float(np.linalg.norm(delta[:3]))
        rotation_deg = float(np.degrees(np.linalg.norm(delta[3:])))
        if (
            translation_norm > maximum_translation + 1e-9
            or rotation_deg > maximum_rotation + 1e-9
        ):
            return float("inf")
        old = poses[frame]
        poses[frame] = candidate
        geometric = _surface_contact_local_cost(
            index,
            frames,
            poses,
            moving_surface,
            reference_surface,
            config,
        )
        poses[frame] = old
        regularizer = prior_weight * (
            (translation_norm / max(maximum_translation, 1e-9)) ** 2
            + (rotation_deg / max(maximum_rotation, 1e-9)) ** 2
        )
        if 0 < index < len(frames) - 1:
            current_delta = pose_delta(baseline[frame], candidate)
            neighbour_delta = 0.5 * (
                pose_delta(
                    baseline[frames[index - 1]], poses[frames[index - 1]]
                )
                + pose_delta(
                    baseline[frames[index + 1]], poses[frames[index + 1]]
                )
            )
            regularizer += smoothness_weight * (
                (
                    np.linalg.norm(current_delta[:3] - neighbour_delta[:3])
                    / max(maximum_translation, 1e-9)
                )
                ** 2
                + (
                    np.degrees(
                        np.linalg.norm(
                            current_delta[3:] - neighbour_delta[3:]
                        )
                    )
                    / max(maximum_rotation, 1e-9)
                )
                ** 2
            )
        evaluations += 1
        return float(geometric + regularizer)

    # A generic axis projection gives coordinate search a useful candidate for
    # coaxial contacts without encoding any object category.
    for index in optimizable:
        frame = frames[index]
        projected = project_pairwise_alignment(
            poses[frame],
            reference_axis=config.get("reference_axis_part"),
            moving_axis=config.get("moving_axis_part"),
            allow_axis_flip=bool(config.get("allow_axis_flip", False)),
            target_axis_offset_m=float(
                config.get(
                    "target_axis_offset_m",
                    config.get("maximum_axis_offset_m", 0.0),
                )
            ),
        )
        if cost(index, projected) < cost(index, poses[frame]):
            poses[frame] = projected

    stages = max(len(translation_steps), len(rotation_steps))
    for stage in range(stages):
        translation_step = translation_steps[
            min(stage, len(translation_steps) - 1)
        ]
        rotation_step = rotation_steps[min(stage, len(rotation_steps) - 1)]
        for order in (optimizable, list(reversed(optimizable))):
            for index in order:
                frame = frames[index]
                best = poses[frame]
                best_cost = cost(index, best)
                for direction_index in range(3):
                    direction = np.zeros(3, dtype=np.float64)
                    direction[direction_index] = 1.0
                    for sign in (-1.0, 1.0):
                        candidate = apply_container_delta(
                            best,
                            sign * translation_step * direction,
                            np.zeros(3),
                        )
                        value = cost(index, candidate)
                        if value + 1e-12 < best_cost:
                            best, best_cost = candidate, value
                for direction_index in range(3):
                    direction = np.zeros(3, dtype=np.float64)
                    direction[direction_index] = 1.0
                    for sign in (-1.0, 1.0):
                        candidate = apply_container_delta(
                            best,
                            np.zeros(3),
                            np.deg2rad(sign * rotation_step) * direction,
                        )
                        value = cost(index, candidate)
                        if value + 1e-12 < best_cost:
                            best, best_cost = candidate, value
                poses[frame] = best

    before = evaluate_surface_contact_trajectory(
        baseline, moving_surface, reference_surface, config
    )
    after = evaluate_surface_contact_trajectory(
        poses, moving_surface, reference_surface, config
    )
    corrections = {}
    for frame in frames:
        delta = pose_delta(baseline[frame], poses[frame])
        corrections[f"{frame:06d}"] = {
            "translation_delta_m": delta[:3].tolist(),
            "translation_delta_norm_m": float(np.linalg.norm(delta[:3])),
            "rotation_delta_deg": float(
                np.degrees(np.linalg.norm(delta[3:]))
            ),
        }
    return poses, {
        "evaluations": evaluations,
        "before": before,
        "proposed_after": after,
        "corrections": corrections,
    }
