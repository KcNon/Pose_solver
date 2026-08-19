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
    reference_axis_origin_m: np.ndarray | list[float] | None = None,
    moving_axis_origin_m: np.ndarray | list[float] | None = None,
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
    if reference_axis_origin_m is not None or moving_axis_origin_m is not None:
        _, radial = _axis_origin_coordinates(
            np.asarray(relative_pose, dtype=np.float64),
            fixed,
            np.asarray(
                reference_axis_origin_m
                if reference_axis_origin_m is not None
                else [0.0, 0.0, 0.0],
                dtype=np.float64,
            ),
            np.asarray(
                moving_axis_origin_m
                if moving_axis_origin_m is not None
                else [0.0, 0.0, 0.0],
                dtype=np.float64,
            ),
        )
    else:
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


def project_coaxial_pose(
    relative_pose: np.ndarray,
    *,
    reference_axis: np.ndarray | list[float] | str,
    moving_axis: np.ndarray | list[float] | str,
    reference_axis_origin_m: np.ndarray | list[float] | None = None,
    moving_axis_origin_m: np.ndarray | list[float] | None = None,
    allow_axis_flip: bool = False,
    target_axis_offset_m: float = 0.0,
    twist_rad: float = 0.0,
    axial_delta_m: float = 0.0,
) -> np.ndarray:
    """Project a relative pose onto two physical axis lines.

    Unlike :func:`project_pairwise_alignment`, this uses configured points on
    the two axes instead of assuming that both part origins lie on them.  That
    distinction is essential for handles, spray heads, and other asymmetric
    parts whose canonical origin is far from the insertion shaft.
    """

    original = np.asarray(relative_pose, dtype=np.float64)
    fixed = axis_vector(reference_axis)
    moving = axis_vector(moving_axis)
    fixed_origin = np.asarray(
        reference_axis_origin_m
        if reference_axis_origin_m is not None
        else [0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    moving_origin = np.asarray(
        moving_axis_origin_m
        if moving_axis_origin_m is not None
        else [0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    axial, radial = _axis_origin_coordinates(
        original, fixed, fixed_origin, moving_origin
    )
    result = original.copy()
    result[:3, :3] = _align_rotation_to_axis(
        result[:3, :3], fixed, moving, allow_axis_flip=allow_axis_flip
    )
    if abs(float(twist_rad)) > 1e-12:
        result[:3, :3] = (
            Rotation.from_rotvec(float(twist_rad) * fixed).as_matrix()
            @ result[:3, :3]
        )
    target_offset = max(float(target_axis_offset_m), 0.0)
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm > 1e-10 and target_offset > 0.0:
        radial = radial * (target_offset / radial_norm)
    else:
        radial = np.zeros(3, dtype=np.float64)
    return _set_axis_origin_coordinates(
        result,
        fixed,
        fixed_origin,
        moving_origin,
        axial + float(axial_delta_m),
        radial,
    )


def _align_rotation_to_axis(
    rotation: np.ndarray,
    reference_axis: np.ndarray,
    moving_axis: np.ndarray,
    *,
    allow_axis_flip: bool,
) -> np.ndarray:
    """Return the nearest rotation whose moving axis matches the reference."""

    result = np.asarray(rotation, dtype=np.float64).copy()
    current = result @ moving_axis
    target = _aligned_axis_target(current, reference_axis, allow_axis_flip)
    dot = float(np.clip(np.dot(current, target), -1.0, 1.0))
    cross = np.cross(current, target)
    norm = float(np.linalg.norm(cross))
    if norm > 1e-10:
        correction = Rotation.from_rotvec(
            cross / norm * float(np.arctan2(norm, dot))
        ).as_matrix()
        result = correction @ result
    elif dot < 0.0:
        seed = np.asarray([1.0, 0.0, 0.0])
        if abs(float(np.dot(seed, current))) > 0.85:
            seed = np.asarray([0.0, 1.0, 0.0])
        perpendicular = _unit(np.cross(current, seed))
        result = Rotation.from_rotvec(np.pi * perpendicular).as_matrix() @ result
    return result


def _axis_origin_coordinates(
    pose: np.ndarray,
    reference_axis: np.ndarray,
    reference_origin: np.ndarray,
    moving_origin: np.ndarray,
) -> tuple[float, np.ndarray]:
    transformed = (
        np.asarray(pose, dtype=np.float64)[:3, :3] @ moving_origin
        + np.asarray(pose, dtype=np.float64)[:3, 3]
    )
    offset = transformed - reference_origin
    axial = float(np.dot(offset, reference_axis))
    return axial, offset - axial * reference_axis


def _set_axis_origin_coordinates(
    pose: np.ndarray,
    reference_axis: np.ndarray,
    reference_origin: np.ndarray,
    moving_origin: np.ndarray,
    axial: float,
    radial: np.ndarray,
) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float64).copy()
    target = reference_origin + float(axial) * reference_axis + radial
    result[:3, 3] = target - result[:3, :3] @ moving_origin
    return result


def _signed_twist(
    first_rotation: np.ndarray,
    second_rotation: np.ndarray,
    reference_axis: np.ndarray,
    moving_axis: np.ndarray,
) -> float:
    """Measure the modulo twist between two already axis-aligned rotations."""

    seed = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, moving_axis))) > 0.85:
        seed = np.asarray([0.0, 1.0, 0.0])
    moving_reference = _unit(
        seed - float(np.dot(seed, moving_axis)) * moving_axis
    )
    first = np.asarray(first_rotation, dtype=np.float64) @ moving_reference
    second = np.asarray(second_rotation, dtype=np.float64) @ moving_reference
    first = _unit(first - float(np.dot(first, reference_axis)) * reference_axis)
    second = _unit(
        second - float(np.dot(second, reference_axis)) * reference_axis
    )
    return float(
        np.arctan2(
            float(np.dot(reference_axis, np.cross(first, second))),
            float(np.clip(np.dot(first, second), -1.0, 1.0)),
        )
    )


def screw_pose_metrics(
    relative_pose: np.ndarray,
    *,
    reference_axis: np.ndarray | list[float] | str,
    moving_axis: np.ndarray | list[float] | str,
    reference_axis_origin_m: np.ndarray | list[float] | None = None,
    moving_axis_origin_m: np.ndarray | list[float] | None = None,
    allow_axis_flip: bool = False,
) -> dict[str, float]:
    """Measure coaxial error using explicit axis origins, not mesh centroids."""

    fixed = axis_vector(reference_axis)
    moving = axis_vector(moving_axis)
    fixed_origin = np.asarray(
        reference_axis_origin_m if reference_axis_origin_m is not None else [0, 0, 0],
        dtype=np.float64,
    )
    moving_origin = np.asarray(
        moving_axis_origin_m if moving_axis_origin_m is not None else [0, 0, 0],
        dtype=np.float64,
    )
    pose = np.asarray(relative_pose, dtype=np.float64)
    transformed_axis = pose[:3, :3] @ moving
    target = _aligned_axis_target(transformed_axis, fixed, allow_axis_flip)
    angle = float(
        np.degrees(
            np.arccos(
                np.clip(float(np.dot(transformed_axis, target)), -1.0, 1.0)
            )
        )
    )
    axial, radial = _axis_origin_coordinates(
        pose, fixed, fixed_origin, moving_origin
    )
    return {
        "axis_angle_deg": angle,
        "axis_offset_m": float(np.linalg.norm(radial)),
        "axis_origin_axial_m": axial,
    }


def evaluate_screw_trajectory(
    poses: dict[int, np.ndarray],
    config: dict[str, Any],
    moving_surface: SampledSurface | None = None,
    reference_surface: SampledSurface | None = None,
) -> dict[str, Any]:
    """Evaluate screw kinematics and optional continuous non-penetration."""

    frames = sorted(poses)
    if not frames:
        raise ValueError("cannot evaluate an empty screw trajectory")
    contact = int(config.get("contact_frame", frames[0]))
    fixed = axis_vector(config.get("reference_axis_part", "z"))
    moving = axis_vector(config.get("moving_axis_part", "z"))
    fixed_origin = np.asarray(
        config.get("reference_axis_origin_part_m", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    moving_origin = np.asarray(
        config.get("moving_axis_origin_part_m", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    rows = []
    for frame in frames:
        item = screw_pose_metrics(
            poses[frame],
            reference_axis=fixed,
            moving_axis=moving,
            reference_axis_origin_m=fixed_origin,
            moving_axis_origin_m=moving_origin,
            allow_axis_flip=bool(config.get("allow_axis_flip", False)),
        )
        item["frame"] = int(frame)
        rows.append(item)

    contact_frames = [frame for frame in frames if frame >= contact]
    cumulative = 0.0
    helix_rows = []
    if contact_frames:
        first = contact_frames[0]
        first_axial = next(
            row["axis_origin_axial_m"] for row in rows if row["frame"] == first
        )
        previous = np.asarray(poses[first], dtype=np.float64)[:3, :3]
        for frame in contact_frames:
            rotation = np.asarray(poses[frame], dtype=np.float64)[:3, :3]
            if frame != first:
                increment = Rotation.from_matrix(rotation @ previous.T).as_rotvec()
                cumulative += float(np.dot(increment, fixed))
            axial = next(
                row["axis_origin_axial_m"]
                for row in rows
                if row["frame"] == frame
            )
            expected = first_axial + float(
                config.get("pitch_m_per_turn", 0.0)
            ) * abs(cumulative) / (2.0 * np.pi)
            helix_rows.append(
                {
                    "frame": int(frame),
                    "cumulative_rotation_rad": cumulative,
                    "axis_origin_axial_m": axial,
                    "expected_axial_m": expected,
                    "helix_residual_m": float(abs(axial - expected)),
                }
            )
            previous = rotation
    axial_values = [row["axis_origin_axial_m"] for row in rows]
    direction = float(config.get("insertion_direction", 1.0))
    monotonic_violations = sum(
        1
        for first, second in zip(axial_values, axial_values[1:])
        if direction * (second - first) < -1e-7
    )
    contact_rows = [row for row in rows if row["frame"] >= contact]
    result = {
        "max_penetration_m": 0.0,
        "rms_penetration_m": 0.0,
        "violating_samples": 0,
        "collision_evaluated": False,
        "max_axis_angle_deg": max(
            (row["axis_angle_deg"] for row in contact_rows), default=0.0
        ),
        "max_axis_offset_m": max(
            (row["axis_offset_m"] for row in contact_rows), default=0.0
        ),
        "max_helix_residual_m": max(
            (row["helix_residual_m"] for row in helix_rows), default=0.0
        ),
        "monotonic_axial_violations": int(monotonic_violations),
        "samples": rows,
        "helix_samples": helix_rows,
    }
    if (moving_surface is None) != (reference_surface is None):
        raise ValueError(
            "moving_surface and reference_surface must be provided together"
        )
    if moving_surface is not None and reference_surface is not None:
        collision_config = dict(config)
        collision_config["continuous_substeps"] = int(
            config.get("collision_continuous_substeps", 2)
        )
        # Contact distance is not a screw-trajectory acceptance condition;
        # surfaces may approach or separate before seating.  This pass exists
        # specifically to reject material interpenetration.
        collision_config["contact_start_frame"] = max(frames) + 1
        collision = evaluate_surface_contact_trajectory(
            poses,
            moving_surface,
            reference_surface,
            collision_config,
        )
        penetration_rows = [
            row
            for row in collision["samples"]
            if float(row.get("max_penetration_m", 0.0)) > 0.0
        ]
        worst = collision["worst_penetration_sample"]
        result.update({
            "max_penetration_m": float(collision["max_penetration_m"]),
            "rms_penetration_m": float(
                worst.get("rms_penetration_m", 0.0)
            ),
            "violating_samples": int(len(penetration_rows)),
            "collision_evaluated": True,
            "collision_substeps_per_frame": int(
                collision["substeps_per_frame"]
            ),
            "collision_sample_count": int(collision["sample_count"]),
            "worst_penetration_sample": worst,
        })
    return result


def refine_screw_trajectory(
    initial_poses: dict[int, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Project a visual trajectory onto insertion then screw-motion DOFs.

    Before contact the axis converges smoothly toward the opening.  From
    contact to the seated frame the only remaining DOFs are rotation about
    the reference axis and pitch-coupled axial translation.  A terminal stable
    anchor supplies the modulo yaw; ``turns`` resolves the otherwise invisible
    full revolutions.
    """

    frames = sorted(initial_poses)
    if not frames:
        raise ValueError("cannot refine an empty screw trajectory")
    start = int(config.get("insertion_start_frame", frames[0]))
    contact = int(config["contact_frame"])
    seat = int(config.get("seat_frame", frames[-1]))
    terminal = int(config.get("terminal_anchor_frame", seat))
    if not (frames[0] <= start <= contact < seat <= frames[-1]):
        raise ValueError("screw frames must satisfy start <= contact < seat")
    if terminal not in initial_poses:
        raise ValueError("terminal_anchor_frame is outside the relation frames")

    fixed = axis_vector(config.get("reference_axis_part", "z"))
    moving = axis_vector(config.get("moving_axis_part", "z"))
    fixed_origin = np.asarray(
        config.get("reference_axis_origin_part_m", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    moving_origin = np.asarray(
        config.get("moving_axis_origin_part_m", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    allow_flip = bool(config.get("allow_axis_flip", False))
    target_offset = max(float(config.get("target_axis_offset_m", 0.0)), 0.0)

    aligned: dict[int, np.ndarray] = {}
    for frame, pose in initial_poses.items():
        value = np.asarray(pose, dtype=np.float64).copy()
        value[:3, :3] = _align_rotation_to_axis(
            value[:3, :3], fixed, moving, allow_axis_flip=allow_flip
        )
        aligned[int(frame)] = value

    start_axial, start_radial = _axis_origin_coordinates(
        initial_poses[start], fixed, fixed_origin, moving_origin
    )
    contact_axial, contact_radial = _axis_origin_coordinates(
        aligned[contact], fixed, fixed_origin, moving_origin
    )
    radial_direction = (
        start_radial / np.linalg.norm(start_radial)
        if np.linalg.norm(start_radial) > 1e-10
        else np.zeros(3, dtype=np.float64)
    )
    contact_radial_target = target_offset * radial_direction
    contact_pose = _set_axis_origin_coordinates(
        aligned[contact],
        fixed,
        fixed_origin,
        moving_origin,
        contact_axial,
        contact_radial_target,
    )

    terminal_rotation = aligned[terminal][:3, :3]
    modulo_terminal = _signed_twist(
        contact_pose[:3, :3], terminal_rotation, fixed, moving
    )
    handedness = 1.0 if float(config.get("handedness", 1.0)) >= 0.0 else -1.0
    turns = max(float(config.get("turns", 1.0)), 0.0)
    total_phase = handedness * 2.0 * np.pi * turns + modulo_terminal
    if total_phase * handedness < 0.0:
        total_phase += handedness * 2.0 * np.pi
    pitch = float(config["pitch_m_per_turn"])
    observation_weight = float(
        np.clip(config.get("phase_observation_weight", 0.5), 0.0, 1.0)
    )

    screw_frames = [frame for frame in frames if contact <= frame <= seat]
    observed_phase = []
    expected_phase = []
    for frame in screw_frames:
        progress = (frame - contact) / max(seat - contact, 1)
        expected = total_phase * progress
        modulo = _signed_twist(
            contact_pose[:3, :3], aligned[frame][:3, :3], fixed, moving
        )
        candidates = [modulo + 2.0 * np.pi * value for value in range(-6, 7)]
        selected = min(candidates, key=lambda value: abs(value - expected))
        expected_phase.append(expected)
        observed_phase.append(selected)
    phase = (
        (1.0 - observation_weight) * np.asarray(expected_phase)
        + observation_weight * np.asarray(observed_phase)
    )
    signed_progress = handedness * phase
    signed_total = handedness * total_phase
    signed_progress = np.clip(signed_progress, 0.0, signed_total)
    signed_progress = np.maximum.accumulate(signed_progress)
    if len(signed_progress):
        maximum_step = np.deg2rad(
            float(config.get("maximum_phase_step_deg", 20.0))
        )
        minimum_required_step = signed_total / max(
            len(signed_progress) - 1, 1
        )
        maximum_step = max(maximum_step, minimum_required_step)
        bounded = np.zeros_like(signed_progress)
        for index in range(1, len(bounded)):
            remaining = len(bounded) - 1 - index
            lower = max(
                bounded[index - 1],
                signed_total - maximum_step * remaining,
            )
            upper = min(
                signed_total, bounded[index - 1] + maximum_step
            )
            bounded[index] = float(
                np.clip(signed_progress[index], lower, upper)
            )
        bounded[-1] = signed_total
        signed_progress = bounded
    phase = handedness * signed_progress
    phase_by_frame = dict(zip(screw_frames, phase.tolist()))

    result = {
        frame: np.asarray(pose, dtype=np.float64).copy()
        for frame, pose in initial_poses.items()
    }
    for frame in frames:
        if frame < start:
            continue
        if frame <= contact:
            amount = (frame - start) / max(contact - start, 1)
            amount = float(np.clip(amount, 0.0, 1.0))
            amount = amount * amount * (3.0 - 2.0 * amount)
            interpolated = interpolate_pose(
                np.asarray(initial_poses[start], dtype=np.float64),
                contact_pose,
                amount,
            )
            axial = (1.0 - amount) * start_axial + amount * contact_axial
            radial = (1.0 - amount) * start_radial + amount * contact_radial_target
            result[frame] = _set_axis_origin_coordinates(
                interpolated,
                fixed,
                fixed_origin,
                moving_origin,
                axial,
                radial,
            )
            continue
        current_phase = phase_by_frame.get(frame, total_phase)
        rotation = (
            Rotation.from_rotvec(current_phase * fixed).as_matrix()
            @ contact_pose[:3, :3]
        )
        value = np.eye(4, dtype=np.float64)
        value[:3, :3] = rotation
        axial = contact_axial + pitch * abs(current_phase) / (2.0 * np.pi)
        result[frame] = _set_axis_origin_coordinates(
            value,
            fixed,
            fixed_origin,
            moving_origin,
            axial,
            contact_radial_target,
        )

    before = evaluate_screw_trajectory(initial_poses, config)
    after = evaluate_screw_trajectory(result, config)
    return result, {
        "evaluations": 0,
        "before": before,
        "proposed_after": after,
        "contact_frame": contact,
        "seat_frame": seat,
        "terminal_anchor_frame": terminal,
        "contact_axial_m": contact_axial,
        "total_phase_rad": total_phase,
        "effective_turns": abs(total_phase) / (2.0 * np.pi),
        "phase_by_frame_rad": {
            f"{frame:06d}": float(value)
            for frame, value in phase_by_frame.items()
        },
    }


def solve_monotonic_axial_path(
    unary_costs: np.ndarray,
    axial_grid_m: np.ndarray,
    *,
    direction: float,
    maximum_step_m: float,
    maximum_backtrack_m: float = 0.0,
    temporal_weight: float = 0.05,
    terminal_index: int | None = None,
) -> tuple[np.ndarray, float]:
    """Solve a visibility-weighted 1-DoF insertion trajectory.

    ``unary_costs`` contains one multi-view render cost per frame and axial
    candidate.  The dynamic program keeps the path physically directed while
    still allowing a small configured backtrack for hand re-positioning.  A
    terminal index can pin the final dynamic frame to a separately validated
    stable assembly anchor.

    This is deliberately independent of any object category: the caller
    supplies the physical axis grid and insertion direction.
    """

    costs = np.asarray(unary_costs, dtype=np.float64)
    grid = np.asarray(axial_grid_m, dtype=np.float64).reshape(-1)
    if costs.ndim != 2 or costs.shape[1] != len(grid):
        raise ValueError(
            "unary_costs must have shape (frames, len(axial_grid_m))"
        )
    if costs.shape[0] == 0 or len(grid) == 0:
        raise ValueError("cannot solve an empty axial trajectory")
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("axial_grid_m must be finite and strictly increasing")
    if maximum_step_m <= 0.0:
        raise ValueError("maximum_step_m must be positive")
    if maximum_backtrack_m < 0.0:
        raise ValueError("maximum_backtrack_m cannot be negative")
    sign = 1.0 if float(direction) >= 0.0 else -1.0
    frame_count, candidate_count = costs.shape
    accumulated = np.full_like(costs, np.inf)
    parents = np.full((frame_count, candidate_count), -1, dtype=np.int64)
    accumulated[0] = costs[0]
    step_scale = max(float(maximum_step_m), 1e-9)
    for frame in range(1, frame_count):
        for current in range(candidate_count):
            delta = grid[current] - grid
            directed = sign * delta
            valid = (
                (np.abs(delta) <= float(maximum_step_m) + 1e-12)
                & (directed >= -float(maximum_backtrack_m) - 1e-12)
                & np.isfinite(accumulated[frame - 1])
            )
            if not np.any(valid):
                continue
            transition = float(temporal_weight) * (delta / step_scale) ** 2
            values = accumulated[frame - 1] + transition
            values[~valid] = np.inf
            parent = int(np.argmin(values))
            accumulated[frame, current] = (
                float(values[parent]) + float(costs[frame, current])
            )
            parents[frame, current] = parent
    if terminal_index is None:
        current = int(np.argmin(accumulated[-1]))
    else:
        current = int(terminal_index)
        if not 0 <= current < candidate_count:
            raise ValueError("terminal_index is outside axial_grid_m")
    if not np.isfinite(accumulated[-1, current]):
        raise ValueError(
            "no feasible axial path reaches the requested terminal candidate"
        )
    path = np.empty(frame_count, dtype=np.int64)
    path[-1] = current
    for frame in range(frame_count - 1, 0, -1):
        current = int(parents[frame, current])
        if current < 0:
            raise RuntimeError("axial path backtracking encountered no parent")
        path[frame - 1] = current
    return path, float(accumulated[-1, path[-1]])


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
