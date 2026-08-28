"""Cross-validated multi-view silhouette calibration for generated meshes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image

from common.io_utils import load_json
from common.multiview_quality import (
    cloud_supported_view_quality,
    mask_area_quality,
)
from common.mesh_render import SceneRenderer
from common.normalized_recon import load_recon
from common.pose_refinement import sample_canonical, silhouette_metrics
from common.pose_transforms import rigid_from_similarity, similarity_from_rigid
from common.pose_visualization import camera_from_recon
from common.render_loss_refinement import (
    MultiViewRenderObjective,
    RenderObservation,
    foreground_occlusion_mask,
)


def _evaluate_exact_mesh(
    mesh: trimesh.Trimesh,
    transform: np.ndarray,
    observations: list[RenderObservation],
    settings: dict[str, Any],
    renderer: SceneRenderer,
    views: list[str],
) -> dict[str, Any]:
    """Evaluate a candidate with the same triangle rasterizer as final QA.

    Point-sampled silhouettes are useful for fast SE(3) search, but their
    dilation makes projected area depend on resolution and sampling density.
    That bias is unacceptable for physical scale, so scale calibration can use
    the exact mesh rasterizer while keeping the already solved rigid pose.
    """

    selected = [row for row in observations if row.view in set(views)]
    weights = settings.get("weights", {})
    iou_weight = float(weights.get("iou", 1.0))
    contour_weight = float(weights.get("contour", 0.25))
    coverage_weight = float(weights.get("target_coverage", 0.15))
    depth_weight = float(weights.get("depth", 0.10))
    edge_cap = float(settings.get("edge_cap_pixels", 15.0))
    depth_cap = float(settings.get("depth_residual_cap_m", 0.05))
    rows = []
    for observation in selected:
        _, rendered_depth = renderer.render(
            [(mesh, transform)],
            observation.intrinsics,
            observation.extrinsics,
        )
        full_predicted = rendered_depth > 0.0
        target = np.asarray(observation.target_mask, dtype=bool)
        occluded = np.zeros_like(target, dtype=bool)
        if (
            settings.get("occlusion_aware", False)
            and observation.observed_depth is not None
        ):
            occluded = foreground_occlusion_mask(
                np.where(full_predicted, rendered_depth, np.inf),
                observation.observed_depth,
                margin_m=float(
                    settings.get("occlusion_depth_margin_m", 0.015)
                ),
                dilation_pixels=int(
                    settings.get("occlusion_dilation_pixels", 1)
                ),
            )
            occluded &= ~target
        predicted = full_predicted & ~occluded
        iou, contour, _ = silhouette_metrics(predicted, target)
        intersection = int(np.logical_and(predicted, target).sum())
        coverage = float(intersection / max(int(target.sum()), 1))
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
            if depth_pixels >= int(settings.get("min_depth_pixels", 20)):
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
    trim_worst = int(settings.get("trim_worst_views", 0))
    if trim_worst > 0 and len(losses) > trim_worst + 1:
        keep = np.argsort(losses)[:len(losses) - trim_worst]
    else:
        keep = np.arange(len(losses))
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


def _configured_prior_cross_frame_gate(
    baseline: dict[str, Any],
    proposed: dict[str, Any],
    *,
    maximum_iou_degradation: float,
    minimum_improved_fraction: float,
) -> tuple[bool, dict[str, Any]]:
    """Reject a local scale win that damages another stable anchor."""

    deltas = []
    holdout_deltas = []
    for frame, baseline_row in baseline.get("frames", {}).items():
        proposed_row = proposed.get("frames", {}).get(frame)
        if proposed_row is None:
            continue
        deltas.append(
            float(proposed_row["optimize_iou"])
            - float(baseline_row["optimize_iou"])
        )
        if (
            proposed_row.get("holdout_iou") is not None
            and baseline_row.get("holdout_iou") is not None
        ):
            holdout_deltas.append(
                float(proposed_row["holdout_iou"])
                - float(baseline_row["holdout_iou"])
            )
    improved_fraction = (
        float(np.mean(np.asarray(deltas) > 0.0)) if deltas else 0.0
    )
    worst_delta = min(deltas) if deltas else -float("inf")
    worst_holdout_delta = (
        min(holdout_deltas) if holdout_deltas else None
    )
    passed = bool(
        deltas
        and worst_delta >= -maximum_iou_degradation
        and improved_fraction >= minimum_improved_fraction
        and (
            worst_holdout_delta is None
            or worst_holdout_delta >= -maximum_iou_degradation
        )
    )
    return passed, {
        "passed": passed,
        "optimize_iou_deltas": deltas,
        "holdout_iou_deltas": holdout_deltas,
        "worst_optimize_iou_delta": worst_delta,
        "worst_holdout_iou_delta": worst_holdout_delta,
        "improved_frame_fraction": improved_fraction,
        "maximum_iou_degradation": maximum_iou_degradation,
        "minimum_improved_fraction": minimum_improved_fraction,
    }


def _rendered_to_target_ratios(
    evaluations: list[dict[str, Any]],
) -> list[float]:
    ratios = []
    for evaluation in evaluations:
        for row in evaluation.get("views", []):
            target = int(row.get("target_pixels", 0))
            rendered = int(
                row.get(
                    "full_rendered_pixels",
                    row.get("rendered_pixels", 0),
                )
            )
            if target > 0 and rendered > 0:
                ratios.append(float(rendered / target))
    return ratios


def _scale_area_gate(
    baseline_ratio: float | None,
    proposed_ratio: float | None,
    *,
    maximum_log_degradation: float,
) -> tuple[bool, float | None, float | None]:
    """Require a scale proposal not to worsen exact projected-area error."""

    if baseline_ratio is None or proposed_ratio is None:
        return True, None, None
    baseline_error = abs(float(np.log(baseline_ratio)))
    proposed_error = abs(float(np.log(proposed_ratio)))
    return (
        proposed_error <= baseline_error + maximum_log_degradation,
        baseline_error,
        proposed_error,
    )


def _mesh_bottom_gap(
    mesh: trimesh.Trimesh,
    transform: np.ndarray,
    plane: dict[str, Any],
) -> float:
    """Return the robust signed distance of a transformed mesh to a plane."""

    normal = np.asarray(plane["normal_world"], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    point = np.asarray(plane["point_world"], dtype=np.float64)
    vertices = (
        np.asarray(mesh.vertices, dtype=np.float64)
        @ np.asarray(transform[:3, :3], dtype=np.float64).T
        + np.asarray(transform[:3, 3], dtype=np.float64)
    )
    return float(np.quantile((vertices - point) @ normal, 0.005))


def _preserve_support_contact(
    mesh: trimesh.Trimesh,
    transform: np.ndarray,
    plane: dict[str, Any],
    target_bottom_gap: float,
) -> np.ndarray:
    """Shift a scale candidate so its existing support contact is preserved."""

    adjusted = np.asarray(transform, dtype=np.float64).copy()
    normal = np.asarray(plane["normal_world"], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    gap = _mesh_bottom_gap(mesh, adjusted, plane)
    adjusted[:3, 3] += (float(target_bottom_gap) - gap) * normal
    return adjusted


def select_scale_candidate_index(
    rows: list[dict[str, Any]],
    *,
    visual_loss_tie_tolerance: float,
) -> tuple[int, dict[str, Any]]:
    """Choose physical projected size among visually equivalent candidates.

    Silhouette IoU is often nearly flat around its optimum: a slightly larger
    render can cover more modal-mask pixels and win by a tiny loss margin even
    when its projected area is already too large.  Once candidates are within
    a small loss tolerance of the visual optimum, exact rendered/target area
    is the stronger scale observable.  Pose remains frozen throughout.
    """

    if not rows:
        raise ValueError("scale candidate rows must not be empty")
    tolerance = float(visual_loss_tie_tolerance)
    if tolerance < 0.0:
        raise ValueError("visual_loss_tie_tolerance must be non-negative")
    finite = [
        index
        for index, row in enumerate(rows)
        if np.isfinite(float(row.get("optimize_loss", float("inf"))))
    ]
    if not finite:
        raise ValueError("scale candidates contain no finite optimize loss")
    best_loss = min(float(rows[index]["optimize_loss"]) for index in finite)
    visually_equivalent = [
        index
        for index in finite
        if float(rows[index]["optimize_loss"]) <= best_loss + tolerance + 1e-12
    ]

    def area_error(index: int) -> float:
        ratio = rows[index].get("optimize_rendered_to_target_median")
        if ratio is None or not np.isfinite(float(ratio)) or float(ratio) <= 0.0:
            return float("inf")
        return abs(float(np.log(float(ratio))))

    selected = min(
        visually_equivalent,
        key=lambda index: (
            area_error(index),
            float(rows[index]["optimize_loss"]),
            abs(float(rows[index]["scale_factor"]) - 1.0),
        ),
    )
    return selected, {
        "policy": (
            "minimum exact projected-area error inside visual-loss tie band"
        ),
        "best_visual_loss": best_loss,
        "visual_loss_tie_tolerance": tolerance,
        "visually_equivalent_indices": visually_equivalent,
        "selected_index": selected,
        "selected_area_log_error": area_error(selected),
    }


def build_render_observations(
    cfg: dict[str, Any],
    part: str,
    frame: int,
    settings: dict[str, Any],
) -> list[RenderObservation]:
    timestamp = f"{frame:06d}"
    width, height = [
        int(value) for value in settings.get("resolution", [160, 90])
    ]
    part_id = int(cfg["part_ids"][part])
    labels_by_view = {
        view: np.asarray(
            Image.open(Path(cfg["masks_dir"]) / timestamp / f"{view}.png")
        )
        for view in cfg["views"]
        if (Path(cfg["masks_dir"]) / timestamp / f"{view}.png").exists()
    }
    quality = mask_area_quality(
        labels_by_view,
        part_id,
        minimum_pixels=int(settings.get("minimum_full_mask_pixels", 800)),
        maximum_area_ratio=float(settings.get("maximum_mask_area_ratio", 4.0)),
        minimum_area_ratio=float(settings.get("minimum_mask_area_ratio", 0.0)),
    )
    trusted_views = set(cfg["views"])
    if settings.get("use_cloud_supported_view_gate", False):
        summary_path = Path(
            settings.get(
                "quality_cloud_summary",
                Path(cfg["point_cloud_root"]) / "quality_cloud_summary.json",
            )
        )
        cache = getattr(build_render_observations, "_quality_cloud_cache", {})
        cache_key = str(summary_path.resolve())
        if cache_key not in cache:
            cache[cache_key] = load_json(summary_path).get("frames", {})
            build_render_observations._quality_cloud_cache = cache
        frame_cloud = cache[cache_key].get(timestamp, {}).get(part, {})
        cloud_gate = cloud_supported_view_quality(
            frame_cloud,
            list(cfg["views"]),
            minimum_supported_points=int(
                settings.get("minimum_supported_points_per_view", 30)
            ),
            minimum_support_fraction=float(
                settings.get("minimum_view_support_fraction", 0.25)
            ),
        )
        trusted_views = {
            view
            for view, row in cloud_gate["views"].items()
            if row["valid"]
        }
    recon = load_recon(cfg, timestamp, backend=cfg["recon_backend"])
    result = []
    for view_index, view in enumerate(cfg["views"]):
        if not quality["views"].get(view, {}).get("valid", False):
            continue
        if view not in trusted_views:
            continue
        target = cv2.resize(
            (labels_by_view[view] == part_id).astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if int(target.sum()) < int(settings.get("min_mask_pixels", 30)):
            continue
        intrinsics, extrinsics = camera_from_recon(
            recon, view_index, (height, width)
        )
        result.append(
            RenderObservation(
                view=view,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                target_mask=target,
                observed_depth=cv2.resize(
                    recon["depth"][view_index].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                ),
            )
        )
    return result


def select_scale_anchor_frames(
    anchor_frames: list[int],
    static_ranges: list[list[int]] | list[tuple[int, int]],
    *,
    first_static_interval_only: bool,
) -> tuple[list[int], dict[str, Any]]:
    """Select geometrically comparable frames for one physical scale.

    Later static intervals are often only *kinematically* static: the object
    can be partly outside the rig, held by a hand, or assembled with another
    part.  Such masks are useful for tracking but are not independent metric
    scale evidence.  Prefer the earliest settled interval that contains a
    scale anchor and keep the decision auditable in the calibration report.
    """

    available = sorted({int(frame) for frame in anchor_frames})
    report: dict[str, Any] = {
        "policy": (
            "earliest_static_interval"
            if first_static_interval_only
            else "all_available_anchors"
        ),
        "available_anchor_frames": available,
        "selected_static_range": None,
        "excluded_anchor_frames": [],
    }
    if not first_static_interval_only or not available:
        report["selected_anchor_frames"] = available
        return available, report

    for start, end in sorted(
        (int(pair[0]), int(pair[1])) for pair in static_ranges
    ):
        selected = [
            frame for frame in available if start <= frame <= end
        ]
        if selected:
            report.update({
                "selected_static_range": [start, end],
                "selected_anchor_frames": selected,
                "excluded_anchor_frames": sorted(set(available) - set(selected)),
            })
            return selected, report

    # Explicit/manual anchors can sit one frame outside a detected interval.
    # Falling back is safer than silently disabling scale calibration.
    report.update({
        "policy": "all_available_anchors_fallback",
        "reason": "no_anchor_inside_static_ranges",
        "selected_anchor_frames": available,
    })
    return available, report


def calibrate_part_scale(
    *,
    cfg: dict[str, Any],
    part: str,
    mesh: trimesh.Trimesh,
    raw_origin: np.ndarray,
    base_scale: float,
    anchors: dict[int, np.ndarray],
    settings: dict[str, Any],
    seed: int,
) -> tuple[float, dict[int, np.ndarray], dict[str, Any]]:
    """Select scale on optimize cameras and independently gate holdout cameras.

    Candidate geometry is scaled about the canonical part origin, so the
    already estimated SE(3) pose is frozen.  Updating the raw-mesh similarity
    afterward preserves that exact rigid pose.
    """
    factors = sorted(
        {float(value) for value in settings.get("scale_factors", [1.0])}
    )
    if not factors or any(value <= 0.0 for value in factors):
        raise ValueError("silhouette scale_factors must be positive")
    maximum_frames = int(settings.get("maximum_anchor_frames", 3))
    frame_ids = sorted(anchors)
    requested_frames = settings.get("anchor_frames")
    if requested_frames is not None:
        requested = {int(value) for value in requested_frames}
        frame_ids = [frame for frame in frame_ids if frame in requested]
    frame_ids, anchor_selection = select_scale_anchor_frames(
        frame_ids,
        cfg.get("states", {}).get(part, {}).get("static_ranges", []),
        first_static_interval_only=bool(
            settings.get("first_static_interval_only", True)
        ),
    )
    if len(frame_ids) > maximum_frames:
        indices = np.linspace(0, len(frame_ids) - 1, maximum_frames)
        frame_ids = sorted({frame_ids[int(round(value))] for value in indices})
    anchor_selection["sampled_anchor_frames"] = frame_ids
    observations = {
        frame: build_render_observations(cfg, part, frame, settings)
        for frame in frame_ids
    }
    support_plane = settings.get("support_plane")
    support_contact_frames = {
        int(value) for value in settings.get("support_contact_frames", [])
    }
    support_bottom_gaps = {
        frame: _mesh_bottom_gap(mesh, anchors[frame], support_plane)
        for frame in frame_ids
        if support_plane is not None and frame in support_contact_frames
    }
    optimize_views = [
        str(value) for value in settings.get("optimize_views", cfg["views"])
    ]
    holdout_views = [str(value) for value in settings.get("holdout_views", [])]
    minimum_views = int(settings.get("minimum_optimize_views", 3))
    usable = [
        frame
        for frame in frame_ids
        if len({item.view for item in observations[frame]}.intersection(optimize_views))
        >= minimum_views
    ]
    exact_mesh_render = bool(settings.get("exact_mesh_render", False))
    report: dict[str, Any] = {
        "method": (
            "frozen_pose_exact_mesh_optimize_holdout_multiview_scale"
            if exact_mesh_render
            else "frozen_pose_optimize_holdout_multiview_silhouette_scale"
        ),
        "base_scale": float(base_scale),
        "anchor_frames": frame_ids,
        "anchor_selection": anchor_selection,
        "usable_anchor_frames": usable,
        "scale_factors": factors,
        "optimize_views": optimize_views,
        "holdout_views": holdout_views,
        "support_contact_frames": sorted(support_bottom_gaps),
        "candidates": [],
    }
    if not usable:
        report.update({"accepted": False, "reason": "insufficient_views"})
        return float(base_scale), anchors, report

    base_points = None
    renderer = None
    if exact_mesh_render:
        width, height = [
            int(value) for value in settings.get("resolution", [160, 90])
        ]
        renderer = SceneRenderer(
            width, height, cache_mesh_resources=True
        )
    else:
        base_points = sample_canonical(
            mesh,
            float(base_scale),
            np.asarray(raw_origin, dtype=np.float64),
            count=int(settings.get("surface_points", 30000)),
            seed=int(seed),
        )
    try:
        for factor in factors:
            optimize_rows = []
            holdout_rows = []
            frame_rows = {}
            for frame in usable:
                pose = rigid_from_similarity(anchors[frame], raw_origin)
                transform = similarity_from_rigid(
                    pose,
                    float(base_scale) * factor,
                    raw_origin,
                )
                if frame in support_bottom_gaps:
                    transform = _preserve_support_contact(
                        mesh,
                        transform,
                        support_plane,
                        support_bottom_gaps[frame],
                    )
                    pose = rigid_from_similarity(transform, raw_origin)
                if exact_mesh_render:
                    optimize = _evaluate_exact_mesh(
                        mesh,
                        transform,
                        observations[frame],
                        settings,
                        renderer,
                        optimize_views,
                    )
                    holdout = _evaluate_exact_mesh(
                        mesh,
                        transform,
                        observations[frame],
                        settings,
                        renderer,
                        holdout_views,
                    )
                else:
                    objective = MultiViewRenderObjective(
                        base_points * factor, observations[frame], settings
                    )
                    optimize = objective.evaluate(pose, optimize_views)
                    holdout = objective.evaluate(pose, holdout_views)
                optimize_rows.append(optimize)
                if holdout["views"]:
                    holdout_rows.append(holdout)
                frame_area_ratios = _rendered_to_target_ratios([optimize])
                frame_rows[f"{frame:06d}"] = {
                    "optimize_loss": float(optimize["loss"]),
                    "optimize_iou": float(optimize["mean_iou"]),
                    "holdout_loss": float(holdout["loss"]),
                    "holdout_iou": float(holdout["mean_iou"]),
                    "optimize_rendered_to_target_median": (
                        float(np.median(frame_area_ratios))
                        if frame_area_ratios
                        else None
                    ),
                }
            optimize_area_ratios = _rendered_to_target_ratios(optimize_rows)
            holdout_area_ratios = _rendered_to_target_ratios(holdout_rows)
            row = {
                "scale_factor": factor,
                "absolute_scale": float(base_scale * factor),
                "optimize_loss": float(np.mean([
                    item["loss"] for item in optimize_rows
                ])),
                "optimize_iou": float(np.mean([
                    item["mean_iou"] for item in optimize_rows
                ])),
                "holdout_loss": (
                    float(np.mean([item["loss"] for item in holdout_rows]))
                    if holdout_rows
                    else None
                ),
                "holdout_iou": (
                    float(np.mean([
                        item["mean_iou"] for item in holdout_rows
                    ]))
                    if holdout_rows
                    else None
                ),
                "optimize_rendered_to_target_median": (
                    float(np.median(optimize_area_ratios))
                    if optimize_area_ratios
                    else None
                ),
                "holdout_rendered_to_target_median": (
                    float(np.median(holdout_area_ratios))
                    if holdout_area_ratios
                    else None
                ),
                "frames": frame_rows,
            }
            report["candidates"].append(row)
    finally:
        if renderer is not None:
            renderer.close()

    rows = report["candidates"]
    baseline_index = min(
        range(len(rows)), key=lambda index: abs(rows[index]["scale_factor"] - 1.0)
    )
    proposed_index, candidate_selection = select_scale_candidate_index(
        rows,
        visual_loss_tie_tolerance=float(
            settings.get("visual_loss_tie_tolerance", 0.01)
        ),
    )
    baseline = rows[baseline_index]
    proposed = rows[proposed_index]
    minimum_improvement = float(settings.get("minimum_improvement", 0.01))
    maximum_holdout_degradation = float(
        settings.get("maximum_holdout_degradation", 0.02)
    )
    holdout_passed = bool(
        proposed["holdout_loss"] is None
        or baseline["holdout_loss"] is None
        or proposed["holdout_loss"]
        <= baseline["holdout_loss"] + maximum_holdout_degradation
    )
    improvement = float(baseline["optimize_loss"] - proposed["optimize_loss"])
    baseline_area_ratio = baseline.get("optimize_rendered_to_target_median")
    proposed_area_ratio = proposed.get("optimize_rendered_to_target_median")
    maximum_area_log_degradation = float(
        settings.get("maximum_area_log_degradation", 0.03)
    )
    (
        area_gate_passed,
        baseline_area_log_error,
        proposed_area_log_error,
    ) = _scale_area_gate(
        baseline_area_ratio,
        proposed_area_ratio,
        maximum_log_degradation=maximum_area_log_degradation,
    )
    configured_prior = bool(settings.get("configured_scale_prior", False))
    prior_trust_iou = float(
        settings.get(
            "minimum_iou_to_trust_configured_scale_prior", 0.62
        )
    )
    prior_trusted = bool(
        configured_prior
        and float(baseline["optimize_iou"]) >= prior_trust_iou
    )
    prior_cross_frame_passed = True
    prior_cross_frame_report = None
    if configured_prior and proposed_index != baseline_index:
        prior_cross_frame_passed, prior_cross_frame_report = (
            _configured_prior_cross_frame_gate(
                baseline,
                proposed,
                maximum_iou_degradation=float(settings.get(
                    "maximum_configured_prior_frame_iou_degradation", 0.01
                )),
                minimum_improved_fraction=float(settings.get(
                    "minimum_configured_prior_improved_frame_fraction", 0.5
                )),
            )
        )
    accepted = bool(
        not prior_trusted
        and improvement >= minimum_improvement
        and holdout_passed
        and area_gate_passed
        and prior_cross_frame_passed
    )
    selected_index = proposed_index if accepted else baseline_index
    selected = rows[selected_index]
    minimum_optimize_iou = float(settings.get("minimum_selected_iou", 0.0))
    minimum_holdout_iou = float(settings.get("minimum_selected_holdout_iou", 0.0))
    quality_passed = bool(
        float(selected["optimize_iou"]) >= minimum_optimize_iou
        and (
            selected["holdout_iou"] is None
            or float(selected["holdout_iou"]) >= minimum_holdout_iou
        )
    )
    selected_scale = float(selected["absolute_scale"])
    updated_anchors = {}
    for frame, similarity in anchors.items():
        rigid = rigid_from_similarity(similarity, raw_origin)
        updated = similarity_from_rigid(
            rigid, selected_scale, raw_origin
        )
        if frame in support_bottom_gaps:
            updated = _preserve_support_contact(
                mesh,
                updated,
                support_plane,
                support_bottom_gaps[frame],
            )
        updated_anchors[frame] = updated
    report.update(
        {
            "baseline_index": baseline_index,
            "proposed_index": proposed_index,
            "candidate_selection": candidate_selection,
            "selected_index": selected_index,
            "selected_scale_factor": float(selected["scale_factor"]),
            "selected_absolute_scale": selected_scale,
            "optimize_loss_improvement": improvement,
            "holdout_passed": holdout_passed,
            "area_gate_passed": area_gate_passed,
            "baseline_area_log_error": baseline_area_log_error,
            "proposed_area_log_error": proposed_area_log_error,
            "maximum_area_log_degradation": maximum_area_log_degradation,
            "accepted": accepted,
            "configured_scale_prior": configured_prior,
            "configured_scale_prior_trusted": prior_trusted,
            "configured_scale_prior_cross_frame_gate": (
                prior_cross_frame_report
            ),
            "minimum_iou_to_trust_configured_scale_prior": prior_trust_iou,
            "quality_passed": quality_passed,
            "minimum_selected_iou": minimum_optimize_iou,
            "minimum_selected_holdout_iou": minimum_holdout_iou,
        }
    )
    return selected_scale, updated_anchors, report
