"""Fast multi-view render losses and bounded SE(3) pose refinement.

The optimization renderer samples the same visual mesh used by the final EGL
renderer, projects those surface samples into calibrated cameras, and
rasterizes a low-resolution silhouette/depth buffer.  This makes it practical
to evaluate many bounded pose candidates without changing the mesh scale or
the project's camera convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from common.geom import project_points
from common.pose_refinement import silhouette_metrics
from common.pose_transforms import transform_points


@dataclass(frozen=True)
class RenderObservation:
    view: str
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    target_mask: np.ndarray
    observed_depth: np.ndarray | None = None
    # Depth support and silhouette support are deliberately independent.  A
    # monocular-depth failure must not discard an otherwise accurate mask, but
    # the same depth image can still be useful for identifying foreground
    # occluders.  ``depth_loss_enabled`` controls only the metric term.
    depth_loss_enabled: bool = True
    # Pixels occupied by another labelled rigid part are known occluders for
    # this part.  They are unknown, not silhouette background.
    known_occluder_mask: np.ndarray | None = None


def apply_world_pose_delta(
    pose: np.ndarray,
    translation_m: Iterable[float],
    rotation_world_rad: Iterable[float],
) -> np.ndarray:
    """Apply a world-frame translation and rotation about the part origin."""
    result = np.asarray(pose, dtype=np.float64).copy()
    result[:3, 3] += np.asarray(translation_m, dtype=np.float64)
    increment = Rotation.from_rotvec(
        np.asarray(rotation_world_rad, dtype=np.float64)
    ).as_matrix()
    result[:3, :3] = increment @ result[:3, :3]
    return result


def world_pose_delta_vector(reference: np.ndarray, pose: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64)
    rotation = pose[:3, :3] @ reference[:3, :3].T
    return np.concatenate(
        (
            pose[:3, 3] - reference[:3, 3],
            Rotation.from_matrix(rotation).as_rotvec(),
        )
    )


def clamp_pose_step(
    reference: np.ndarray,
    pose: np.ndarray,
    *,
    maximum_translation_m: float,
    maximum_rotation_deg: float,
) -> np.ndarray:
    """Move from ``reference`` toward ``pose`` without exceeding SE(3) limits."""

    reference = np.asarray(reference, dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64)
    delta = world_pose_delta_vector(reference, pose)
    translation = delta[:3]
    translation_norm = float(np.linalg.norm(translation))
    safe_translation = max(0.0, maximum_translation_m - 1e-9)
    if translation_norm > safe_translation:
        translation *= safe_translation / translation_norm
    rotation = delta[3:]
    rotation_deg = float(np.degrees(np.linalg.norm(rotation)))
    safe_rotation_deg = max(0.0, maximum_rotation_deg - 1e-7)
    if rotation_deg > safe_rotation_deg:
        rotation *= safe_rotation_deg / rotation_deg
    result = np.eye(4)
    result[:3, :3] = (
        Rotation.from_rotvec(rotation).as_matrix() @ reference[:3, :3]
    )
    result[:3, 3] = reference[:3, 3] + translation
    return result


def symmetry_aware_rotation_directions(
    pose: np.ndarray,
    symmetry_axis_part: np.ndarray | None,
) -> list[np.ndarray]:
    """Return observable world rotation axes, excluding continuous axial yaw."""
    if symmetry_axis_part is None:
        return [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
    axis = np.asarray(pose, dtype=np.float64)[:3, :3] @ np.asarray(
        symmetry_axis_part, dtype=np.float64
    )
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, axis))) > 0.85:
        seed = np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, seed)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = np.cross(axis, first)
    second /= max(float(np.linalg.norm(second)), 1e-12)
    return [first, second]


def coarse_reacquire_pose(
    objective: "MultiViewRenderObjective",
    initial_pose: np.ndarray,
    *,
    views: list[str],
    translation_radii_m: list[float],
    rotation_angles_deg: list[float],
    symmetry_axis_part: np.ndarray | None,
    optimize_rotation: bool = True,
    alternating_passes: int = 2,
    rotation_reference_pose: np.ndarray | None = None,
    rotation_prior_weight: float = 0.0,
    rotation_prior_scale_deg: float = 90.0,
    translation_reference_pose: np.ndarray | None = None,
    translation_prior_weight: float = 0.0,
    translation_prior_scale_m: float = 0.04,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministically escape a bad local pose basin using mask evidence.

    The regular frame optimizer is deliberately local.  When tracking leaves
    the image and later re-enters, however, its initial pose can be outside
    that basin.  This coarse search alternates a 3-D translation lattice and
    broad world-axis rotations, scoring every candidate in all supplied mask
    views.  A bounded local refinement still runs afterwards.
    """

    best = np.asarray(initial_pose, dtype=np.float64).copy()
    best_data = objective.evaluate(best, views)
    initial_data = best_data
    evaluations = 1
    reference = (
        None
        if rotation_reference_pose is None
        else np.asarray(rotation_reference_pose, dtype=np.float64)
    )
    prior_weight = max(0.0, float(rotation_prior_weight))
    prior_scale = max(1e-6, float(rotation_prior_scale_deg))
    translation_reference = (
        None
        if translation_reference_pose is None
        else np.asarray(translation_reference_pose, dtype=np.float64)
    )
    translation_weight = max(0.0, float(translation_prior_weight))
    translation_scale = max(1e-9, float(translation_prior_scale_m))

    def rotation_from_reference_deg(pose: np.ndarray) -> float:
        if reference is None:
            return 0.0
        delta = Rotation.from_matrix(reference[:3, :3]).inv() * Rotation.from_matrix(
            np.asarray(pose, dtype=np.float64)[:3, :3]
        )
        return float(np.degrees(delta.magnitude()))

    def regularized_score(data: dict[str, Any], pose: np.ndarray) -> float:
        angle = rotation_from_reference_deg(pose)
        translation = (
            0.0
            if translation_reference is None
            else float(np.linalg.norm(
                np.asarray(pose, dtype=np.float64)[:3, 3]
                - translation_reference[:3, 3]
            ))
        )
        return (
            float(data["loss"])
            + prior_weight * (angle / prior_scale) ** 2
            + translation_weight * (translation / translation_scale) ** 2
        )

    best_score = regularized_score(best_data, best)
    directions = (
        symmetry_aware_rotation_directions(best, symmetry_axis_part)
        if optimize_rotation
        else []
    )
    for _ in range(max(1, int(alternating_passes))):
        for radius in translation_radii_m:
            radius = abs(float(radius))
            if radius <= 0.0:
                continue
            center = best.copy()
            for dx in (-radius, 0.0, radius):
                for dy in (-radius, 0.0, radius):
                    for dz in (-radius, 0.0, radius):
                        if dx == dy == dz == 0.0:
                            continue
                        candidate = center.copy()
                        candidate[:3, 3] += [dx, dy, dz]
                        data = objective.evaluate(candidate, views)
                        evaluations += 1
                        score = regularized_score(data, candidate)
                        if score < best_score:
                            best, best_data, best_score = candidate, data, score
        if optimize_rotation:
            # Recompute observable axes after translation/rotation updates.
            directions = symmetry_aware_rotation_directions(
                best, symmetry_axis_part
            )
            center = best.copy()
            for angle_deg in rotation_angles_deg:
                angle = np.deg2rad(abs(float(angle_deg)))
                if angle <= 0.0:
                    continue
                for direction in directions:
                    for sign in (-1.0, 1.0):
                        candidate = center.copy()
                        increment = Rotation.from_rotvec(
                            sign * angle * direction
                        ).as_matrix()
                        candidate[:3, :3] = increment @ candidate[:3, :3]
                        data = objective.evaluate(candidate, views)
                        evaluations += 1
                        score = regularized_score(data, candidate)
                        if score < best_score:
                            best, best_data, best_score = candidate, data, score
    delta = world_pose_delta_vector(initial_pose, best)
    return best, {
        "evaluations": evaluations,
        "baseline": initial_data,
        "selected": best_data,
        "loss_improvement": float(
            initial_data["loss"] - best_data["loss"]
        ),
        "translation_delta_norm_m": float(np.linalg.norm(delta[:3])),
        "rotation_delta_deg": float(
            np.degrees(np.linalg.norm(delta[3:]))
        ),
        "rotation_prior": {
            "enabled": bool(reference is not None and prior_weight > 0.0),
            "weight": prior_weight,
            "scale_deg": prior_scale,
            "selected_rotation_from_reference_deg": (
                rotation_from_reference_deg(best) if reference is not None else None
            ),
            "selected_regularized_score": best_score,
        },
        "translation_prior": {
            "enabled": bool(
                translation_reference is not None and translation_weight > 0.0
            ),
            "weight": translation_weight,
            "scale_m": translation_scale,
            "selected_translation_from_reference_m": (
                float(np.linalg.norm(best[:3, 3] - translation_reference[:3, 3]))
                if translation_reference is not None
                else None
            ),
        },
    }


def rasterize_surface_points(
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image_hw: tuple[int, int],
    *,
    dilation_pixels: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a sampled mesh surface into a mask and front-depth buffer."""
    height, width = map(int, image_hw)
    uv, depth = project_points(
        points_world,
        np.asarray(intrinsics, dtype=np.float64),
        np.asarray(extrinsics, dtype=np.float64),
    )
    columns = np.rint(uv[:, 0]).astype(np.int64)
    rows = np.rint(uv[:, 1]).astype(np.int64)
    valid = (
        (depth > 1e-4)
        & (columns >= 0)
        & (columns < width)
        & (rows >= 0)
        & (rows < height)
    )
    columns = columns[valid]
    rows = rows[valid]
    values = depth[valid]

    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    if len(columns):
        np.minimum.at(depth_buffer, (rows, columns), values.astype(np.float32))
    center_mask = np.isfinite(depth_buffer)
    dilation_pixels = max(0, int(dilation_pixels))
    if dilation_pixels:
        kernel_size = 2 * dilation_pixels + 1
        mask = cv2.dilate(
            center_mask.astype(np.uint8),
            np.ones((kernel_size, kernel_size), np.uint8),
            iterations=1,
        ).astype(bool)
    else:
        mask = center_mask
    return mask, depth_buffer


def foreground_occlusion_mask(
    predicted_depth: np.ndarray,
    observed_depth: np.ndarray,
    *,
    margin_m: float = 0.015,
    dilation_pixels: int = 1,
) -> np.ndarray:
    """Pixels where observed scene geometry is in front of the rendered part.

    SAM supplies a modal (visible-only) mask while mesh rendering is amodal.
    Treating a hand-covered part pixel as a false positive biases scale toward
    smaller meshes and can pull pose toward the exposed fragment.  This mask
    marks those pixels as unknown rather than background.
    """

    predicted = np.asarray(predicted_depth, dtype=np.float32)
    observed = np.asarray(observed_depth, dtype=np.float32)
    occluded = (
        np.isfinite(predicted)
        & np.isfinite(observed)
        & (observed > 1e-4)
        & (observed + float(margin_m) < predicted)
    )
    dilation_pixels = max(0, int(dilation_pixels))
    if dilation_pixels and occluded.any():
        size = 2 * dilation_pixels + 1
        occluded = cv2.dilate(
            occluded.astype(np.uint8),
            np.ones((size, size), np.uint8),
            iterations=1,
        ).astype(bool)
    return occluded


class MultiViewRenderObjective:
    def __init__(
        self,
        canonical_surface_points: np.ndarray,
        observations: list[RenderObservation],
        config: dict[str, Any],
    ):
        self.points = np.asarray(canonical_surface_points, dtype=np.float64)
        self.observations = list(observations)
        self.config = dict(config)
        self.by_view = {item.view: item for item in observations}

    def evaluate(
        self,
        pose: np.ndarray,
        views: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        selected = (
            self.observations
            if views is None
            else [self.by_view[name] for name in views if name in self.by_view]
        )
        points_world = transform_points(self.points, pose)
        weights = self.config.get("weights", {})
        iou_weight = float(weights.get("iou", 1.0))
        contour_weight = float(weights.get("contour", 0.25))
        coverage_weight = float(weights.get("target_coverage", 0.15))
        depth_weight = float(weights.get("depth", 0.10))
        edge_cap = float(self.config.get("edge_cap_pixels", 15.0))
        depth_cap = float(self.config.get("depth_residual_cap_m", 0.05))
        dilation = int(self.config.get("dilation_pixels", 1))
        rows: list[dict[str, Any]] = []
        for observation in selected:
            full_predicted, predicted_depth = rasterize_surface_points(
                points_world,
                observation.intrinsics,
                observation.extrinsics,
                observation.target_mask.shape,
                dilation_pixels=dilation,
            )
            target = np.asarray(observation.target_mask, dtype=bool)
            occluded = np.zeros_like(target, dtype=bool)
            if observation.known_occluder_mask is not None:
                known = np.asarray(
                    observation.known_occluder_mask, dtype=bool
                )
                if known.shape != target.shape:
                    raise ValueError(
                        "known_occluder_mask must match target_mask shape"
                    )
                known_dilation = max(
                    0,
                    int(
                        self.config.get(
                            "known_occluder_dilation_pixels", 0
                        )
                    ),
                )
                if known_dilation and known.any():
                    size = 2 * known_dilation + 1
                    known = cv2.dilate(
                        known.astype(np.uint8),
                        np.ones((size, size), dtype=np.uint8),
                        iterations=1,
                    ).astype(bool)
                occluded |= known & full_predicted
            if (
                self.config.get("occlusion_aware", False)
                and observation.observed_depth is not None
            ):
                occluded |= foreground_occlusion_mask(
                    predicted_depth,
                    observation.observed_depth,
                    margin_m=float(
                        self.config.get("occlusion_depth_margin_m", 0.015)
                    ),
                    dilation_pixels=dilation,
                )
            # Never remove an observed target pixel: depth is least reliable
            # on thin/transparent object surfaces, and the segmentation is the
            # direct evidence that the part is visible there.
            occluded &= ~target
            predicted = full_predicted & ~occluded
            iou, contour, _ = silhouette_metrics(predicted, target)
            intersection = int(np.logical_and(predicted, target).sum())
            coverage = float(intersection / max(int(target.sum()), 1))
            precision = float(intersection / max(int(predicted.sum()), 1))
            depth_loss = None
            depth_pixels = 0
            if (
                observation.depth_loss_enabled
                and observation.observed_depth is not None
            ):
                observed_depth = np.asarray(
                    observation.observed_depth, dtype=np.float32
                )
                overlap = (
                    target
                    & np.isfinite(predicted_depth)
                    & np.isfinite(observed_depth)
                    & (observed_depth > 1e-4)
                )
                depth_pixels = int(overlap.sum())
                if depth_pixels >= int(self.config.get("min_depth_pixels", 20)):
                    residual = np.abs(
                        predicted_depth[overlap] - observed_depth[overlap]
                    )
                    depth_loss = float(
                        np.mean(np.minimum(residual, depth_cap)) / depth_cap
                    )
            value = (
                iou_weight * (1.0 - iou)
                + contour_weight * min(contour, edge_cap) / edge_cap
                + coverage_weight * (1.0 - coverage)
            )
            if depth_loss is not None:
                value += depth_weight * depth_loss
            rows.append(
                {
                    "view": observation.view,
                    "loss": float(value),
                    "iou": float(iou),
                    "contour_chamfer_px": float(contour),
                    "target_coverage": coverage,
                    "precision": precision,
                    "depth_loss": depth_loss,
                    "depth_pixels": depth_pixels,
                    "target_pixels": int(target.sum()),
                    "rendered_pixels": int(predicted.sum()),
                    "full_rendered_pixels": int(full_predicted.sum()),
                    "ignored_occluded_pixels": int(
                        np.logical_and(full_predicted, occluded).sum()
                    ),
                }
            )
        if not rows:
            return {
                "loss": float("inf"),
                "mean_iou": 0.0,
                "mean_contour_chamfer_px": edge_cap,
                "views": [],
            }
        losses = np.asarray([row["loss"] for row in rows], dtype=np.float64)
        trim_worst = int(self.config.get("trim_worst_views", 0))
        if trim_worst > 0 and len(losses) > trim_worst + 1:
            keep = np.argsort(losses)[: len(losses) - trim_worst]
        else:
            keep = np.arange(len(losses))
        return {
            "loss": float(np.mean(losses[keep])),
            "worst_view_loss": float(np.max(losses)),
            "mean_iou": float(np.mean([rows[index]["iou"] for index in keep])),
            "worst_view_iou": float(min(row["iou"] for row in rows)),
            "mean_contour_chamfer_px": float(
                np.mean([rows[index]["contour_chamfer_px"] for index in keep])
            ),
            "mean_target_coverage": float(
                np.mean([rows[index]["target_coverage"] for index in keep])
            ),
            "views": rows,
            "aggregated_view_indices": keep.tolist(),
        }


def refine_pose_coordinate_search(
    objective: MultiViewRenderObjective,
    initial_pose: np.ndarray,
    *,
    optimize_views: list[str],
    holdout_views: list[str],
    translation_steps_m: list[float],
    rotation_steps_deg: list[float],
    symmetry_axis_part: np.ndarray | None,
    optimize_rotation: bool,
    maximum_translation_delta_m: float,
    maximum_rotation_delta_deg: float,
    minimum_improvement: float,
    maximum_holdout_degradation: float,
    minimum_refined_iou: float = 0.0,
    minimum_refined_target_coverage: float = 0.0,
    minimum_holdout_iou: float = 0.0,
    minimum_per_view_iou: float = 0.0,
    maximum_worst_view_loss: float = 1.0e9,
    require_independent_holdout: bool = False,
    previous_pose: np.ndarray | None = None,
    next_pose: np.ndarray | None = None,
    maximum_step_translation_m: float | None = None,
    maximum_step_rotation_deg: float | None = None,
    prior_weight: float = 0.03,
    temporal_delta_reference: np.ndarray | None = None,
    temporal_weight: float = 0.02,
    orientation_constraints: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bounded derivative-free refinement with an independent holdout gate."""
    initial = np.asarray(initial_pose, dtype=np.float64)
    effective_optimize = [
        name for name in optimize_views if name in objective.by_view
    ]
    effective_holdout = [
        name for name in holdout_views if name in objective.by_view
    ]
    baseline_opt = objective.evaluate(initial, effective_optimize)
    baseline_holdout = (
        objective.evaluate(initial, effective_holdout)
        if effective_holdout
        else {"loss": 0.0, "views": []}
    )
    best_pose = initial.copy()
    best_data = baseline_opt
    boundary_poses = [
        np.asarray(value, dtype=np.float64)
        for value in (previous_pose, next_pose)
        if value is not None
    ]
    normalized_orientation_constraints = []
    for raw_constraint in orientation_constraints or []:
        axis_part = np.asarray(
            raw_constraint["axis_part"], dtype=np.float64
        )
        target_world = np.asarray(
            raw_constraint["target_world"], dtype=np.float64
        )
        axis_norm = float(np.linalg.norm(axis_part))
        target_norm = float(np.linalg.norm(target_world))
        if (
            axis_part.shape != (3,)
            or target_world.shape != (3,)
            or axis_norm <= 1e-12
            or target_norm <= 1e-12
        ):
            raise ValueError(
                "orientation constraint axes must be non-zero 3-vectors"
            )
        minimum = float(raw_constraint.get("minimum_alignment", -1.0))
        maximum = float(raw_constraint.get("maximum_alignment", 1.0))
        if not -1.0 <= minimum <= maximum <= 1.0:
            raise ValueError(
                "orientation constraint must satisfy "
                "-1 <= minimum_alignment <= maximum_alignment <= 1"
            )
        normalized_orientation_constraints.append({
            "label": str(raw_constraint.get("label", "direction")),
            "axis_part": axis_part / axis_norm,
            "target_world": target_world / target_norm,
            "minimum_alignment": minimum,
            "maximum_alignment": maximum,
        })

    def orientation_alignments(pose: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "label": constraint["label"],
                "alignment": float(np.dot(
                    pose[:3, :3] @ constraint["axis_part"],
                    constraint["target_world"],
                )),
                "minimum_alignment": constraint["minimum_alignment"],
                "maximum_alignment": constraint["maximum_alignment"],
            }
            for constraint in normalized_orientation_constraints
        ]

    def violates_orientation(pose: np.ndarray) -> bool:
        return any(
            row["alignment"] < row["minimum_alignment"] - 1e-9
            or row["alignment"] > row["maximum_alignment"] + 1e-9
            for row in orientation_alignments(pose)
        )

    def violates_boundary(pose: np.ndarray) -> bool:
        for boundary in boundary_poses:
            step = world_pose_delta_vector(boundary, pose)
            if (
                maximum_step_translation_m is not None
                and float(np.linalg.norm(step[:3]))
                > maximum_step_translation_m + 1e-9
            ):
                return True
            if (
                maximum_step_rotation_deg is not None
                and float(np.degrees(np.linalg.norm(step[3:])))
                > maximum_step_rotation_deg + 1e-9
            ):
                return True
        return False

    best_total = (
        float("inf")
        if violates_boundary(initial) or violates_orientation(initial)
        else float(baseline_opt["loss"])
    )
    evaluations = 1
    temporal_reference = (
        np.zeros(6, dtype=np.float64)
        if temporal_delta_reference is None
        else np.asarray(temporal_delta_reference, dtype=np.float64)
    )

    def total(pose: np.ndarray) -> tuple[float, dict[str, Any]]:
        nonlocal evaluations
        if violates_boundary(pose) or violates_orientation(pose):
            return float("inf"), {}
        delta = world_pose_delta_vector(initial, pose)
        translation_norm = float(np.linalg.norm(delta[:3]))
        rotation_deg = float(np.degrees(np.linalg.norm(delta[3:])))
        if (
            translation_norm > maximum_translation_delta_m + 1e-9
            or rotation_deg > maximum_rotation_delta_deg + 1e-9
        ):
            return float("inf"), {}
        data = objective.evaluate(pose, effective_optimize)
        evaluations += 1
        regularizer = prior_weight * (
            (translation_norm / max(maximum_translation_delta_m, 1e-9)) ** 2
            + (rotation_deg / max(maximum_rotation_delta_deg, 1e-9)) ** 2
        )
        temporal_translation = np.linalg.norm(delta[:3] - temporal_reference[:3])
        temporal_rotation = np.degrees(
            np.linalg.norm(delta[3:] - temporal_reference[3:])
        )
        regularizer += temporal_weight * (
            (temporal_translation / max(maximum_translation_delta_m, 1e-9)) ** 2
            + (temporal_rotation / max(maximum_rotation_delta_deg, 1e-9)) ** 2
        )
        return float(data["loss"] + regularizer), data

    translation_directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    rotation_directions = (
        symmetry_aware_rotation_directions(initial, symmetry_axis_part)
        if optimize_rotation
        else []
    )
    stages = max(len(translation_steps_m), len(rotation_steps_deg))
    for stage in range(stages):
        translation_step = float(
            translation_steps_m[min(stage, len(translation_steps_m) - 1)]
        )
        rotation_step = float(
            rotation_steps_deg[min(stage, len(rotation_steps_deg) - 1)]
        )
        for direction in translation_directions:
            for sign in (-1.0, 1.0):
                candidate = best_pose.copy()
                candidate[:3, 3] += sign * translation_step * direction
                value, data = total(candidate)
                if value < best_total:
                    best_pose, best_total, best_data = candidate, value, data
        for direction in rotation_directions:
            for sign in (-1.0, 1.0):
                candidate = best_pose.copy()
                increment = Rotation.from_rotvec(
                    np.deg2rad(sign * rotation_step) * direction
                ).as_matrix()
                candidate[:3, :3] = increment @ candidate[:3, :3]
                value, data = total(candidate)
                if value < best_total:
                    best_pose, best_total, best_data = candidate, value, data

    refined_holdout = (
        objective.evaluate(best_pose, effective_holdout)
        if effective_holdout
        else {"loss": 0.0, "views": []}
    )
    data_improvement = float(baseline_opt["loss"] - best_data["loss"])
    holdout_degradation = float(
        refined_holdout["loss"] - baseline_holdout["loss"]
    )
    absolute_gate_failures = []
    if require_independent_holdout and not effective_holdout:
        absolute_gate_failures.append("independent_holdout_missing")
    if float(best_data.get("mean_iou", 0.0)) < minimum_refined_iou:
        absolute_gate_failures.append("optimize_iou_below_minimum")
    if (
        float(best_data.get("mean_target_coverage", 0.0))
        < minimum_refined_target_coverage
    ):
        absolute_gate_failures.append("optimize_coverage_below_minimum")
    if (
        effective_holdout
        and float(refined_holdout.get("mean_iou", 0.0))
        < minimum_holdout_iou
    ):
        absolute_gate_failures.append("holdout_iou_below_minimum")
    selected_view_rows = list(best_data.get("views", [])) + list(
        refined_holdout.get("views", [])
    )
    if selected_view_rows and min(
        float(row.get("iou", 0.0)) for row in selected_view_rows
    ) < float(minimum_per_view_iou):
        absolute_gate_failures.append("worst_view_iou_below_minimum")
    if selected_view_rows and max(
        float(row.get("loss", float("inf"))) for row in selected_view_rows
    ) > float(maximum_worst_view_loss):
        absolute_gate_failures.append("worst_view_loss_above_maximum")
    accepted = bool(
        data_improvement >= minimum_improvement
        and (
            not effective_holdout
            or holdout_degradation <= maximum_holdout_degradation
        )
        and not absolute_gate_failures
    )
    selected = best_pose if accepted else initial.copy()
    delta = world_pose_delta_vector(initial, selected)
    return selected, {
        "accepted": accepted,
        "evaluations": evaluations,
        "baseline_optimize": baseline_opt,
        "refined_optimize": best_data,
        "baseline_holdout": baseline_holdout,
        "refined_holdout": refined_holdout,
        "optimize_loss_improvement": data_improvement,
        "holdout_loss_degradation": holdout_degradation,
        "absolute_gate_failures": absolute_gate_failures,
        "absolute_gates": {
            "minimum_refined_iou": float(minimum_refined_iou),
            "minimum_refined_target_coverage": float(
                minimum_refined_target_coverage
            ),
            "minimum_holdout_iou": float(minimum_holdout_iou),
            "minimum_per_view_iou": float(minimum_per_view_iou),
            "maximum_worst_view_loss": float(maximum_worst_view_loss),
            "require_independent_holdout": bool(
                require_independent_holdout
            ),
        },
        "trajectory_boundary_gate": {
            "previous_pose": previous_pose is not None,
            "next_pose": next_pose is not None,
            "maximum_step_translation_m": maximum_step_translation_m,
            "maximum_step_rotation_deg": maximum_step_rotation_deg,
        },
        "orientation_constraints": orientation_alignments(selected),
        "best_pose_before_gates": best_pose.tolist(),
        "translation_delta_m": delta[:3].tolist(),
        "translation_delta_norm_m": float(np.linalg.norm(delta[:3])),
        "rotation_delta_world_rad": delta[3:].tolist(),
        "rotation_delta_deg": float(np.degrees(np.linalg.norm(delta[3:]))),
    }
