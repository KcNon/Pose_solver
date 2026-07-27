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
            predicted, predicted_depth = rasterize_surface_points(
                points_world,
                observation.intrinsics,
                observation.extrinsics,
                observation.target_mask.shape,
                dilation_pixels=dilation,
            )
            target = np.asarray(observation.target_mask, dtype=bool)
            iou, contour, _ = silhouette_metrics(predicted, target)
            intersection = int(np.logical_and(predicted, target).sum())
            coverage = float(intersection / max(int(target.sum()), 1))
            precision = float(intersection / max(int(predicted.sum()), 1))
            depth_loss = None
            depth_pixels = 0
            if observation.observed_depth is not None:
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
            "mean_iou": float(np.mean([rows[index]["iou"] for index in keep])),
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
    prior_weight: float = 0.03,
    temporal_delta_reference: np.ndarray | None = None,
    temporal_weight: float = 0.02,
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
    best_total = float(baseline_opt["loss"])
    evaluations = 1
    temporal_reference = (
        np.zeros(6, dtype=np.float64)
        if temporal_delta_reference is None
        else np.asarray(temporal_delta_reference, dtype=np.float64)
    )

    def total(pose: np.ndarray) -> tuple[float, dict[str, Any]]:
        nonlocal evaluations
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
    accepted = bool(
        data_improvement >= minimum_improvement
        and (
            not effective_holdout
            or holdout_degradation <= maximum_holdout_degradation
        )
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
        "translation_delta_m": delta[:3].tolist(),
        "translation_delta_norm_m": float(np.linalg.norm(delta[:3])),
        "rotation_delta_world_rad": delta[3:].tolist(),
        "rotation_delta_deg": float(np.degrees(np.linalg.norm(delta[3:]))),
    }
