"""Exact triangle-rasterized multi-view pose objective.

The fast pose search uses sampled mesh points.  This module supplies the same
``evaluate`` interface backed by the final EGL triangle renderer, so a cheap
search proposal can be polished and accepted only when it improves the actual
mesh silhouette used by QA.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import trimesh

from common.mesh_render import SceneRenderer
from common.pose_refinement import silhouette_metrics
from common.pose_transforms import similarity_from_rigid
from common.render_loss_refinement import (
    RenderObservation,
    foreground_occlusion_mask,
)


class ExactMultiViewRenderObjective:
    def __init__(
        self,
        raw_mesh: trimesh.Trimesh,
        scale: float,
        raw_origin: np.ndarray,
        observations: list[RenderObservation],
        config: dict[str, Any],
        renderer: SceneRenderer,
    ) -> None:
        self.mesh = raw_mesh
        self.scale = float(scale)
        self.raw_origin = np.asarray(raw_origin, dtype=np.float64)
        self.observations = list(observations)
        self.config = dict(config)
        self.renderer = renderer
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
        transform = similarity_from_rigid(
            np.asarray(pose, dtype=np.float64),
            self.scale,
            self.raw_origin,
        )
        weights = self.config.get("weights", {})
        iou_weight = float(weights.get("iou", 1.0))
        contour_weight = float(weights.get("contour", 0.25))
        coverage_weight = float(weights.get("target_coverage", 0.15))
        depth_weight = float(weights.get("depth", 0.10))
        edge_cap = float(self.config.get("edge_cap_pixels", 15.0))
        depth_cap = float(self.config.get("depth_residual_cap_m", 0.05))
        rows = []
        for observation in selected:
            _, rendered_depth = self.renderer.render(
                [(self.mesh, transform)],
                observation.intrinsics,
                observation.extrinsics,
            )
            full_predicted = rendered_depth > 0.0
            target = np.asarray(observation.target_mask, dtype=bool)
            occluded = np.zeros_like(target, dtype=bool)
            if observation.known_occluder_mask is not None:
                occluded |= (
                    np.asarray(observation.known_occluder_mask, dtype=bool)
                    & full_predicted
                )
            if (
                self.config.get("occlusion_aware", False)
                and observation.observed_depth is not None
            ):
                occluded |= foreground_occlusion_mask(
                    np.where(full_predicted, rendered_depth, np.inf),
                    observation.observed_depth,
                    margin_m=float(
                        self.config.get("occlusion_depth_margin_m", 0.015)
                    ),
                    dilation_pixels=int(
                        self.config.get("occlusion_dilation_pixels", 1)
                    ),
                )
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
                    & full_predicted
                    & np.isfinite(observed_depth)
                    & (observed_depth > 1e-4)
                )
                depth_pixels = int(overlap.sum())
                if depth_pixels >= int(
                    self.config.get("min_depth_pixels", 20)
                ):
                    residual = np.abs(
                        rendered_depth[overlap] - observed_depth[overlap]
                    )
                    depth_loss = float(
                        np.mean(np.minimum(residual, depth_cap)) / depth_cap
                    )
            loss = (
                iou_weight * (1.0 - iou)
                + contour_weight * min(contour, edge_cap) / edge_cap
                + coverage_weight * (1.0 - coverage)
            )
            if depth_loss is not None:
                loss += depth_weight * depth_loss
            rows.append({
                "view": observation.view,
                "loss": float(loss),
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
            })
        if not rows:
            return {
                "loss": float("inf"),
                "mean_iou": 0.0,
                "mean_contour_chamfer_px": edge_cap,
                "views": [],
            }
        losses = np.asarray([row["loss"] for row in rows], dtype=np.float64)
        trim_worst = int(self.config.get("trim_worst_views", 0))
        if trim_worst > 0 and len(rows) > trim_worst + 1:
            keep = np.argsort(losses)[: len(rows) - trim_worst]
        else:
            keep = np.arange(len(rows))
        return {
            "loss": float(np.mean(losses[keep])),
            "worst_view_loss": float(np.max(losses)),
            "mean_iou": float(np.mean([rows[index]["iou"] for index in keep])),
            "worst_view_iou": float(min(row["iou"] for row in rows)),
            "mean_contour_chamfer_px": float(np.mean([
                rows[index]["contour_chamfer_px"] for index in keep
            ])),
            "mean_target_coverage": float(np.mean([
                rows[index]["target_coverage"] for index in keep
            ])),
            "views": rows,
            "aggregated_view_indices": keep.tolist(),
        }
