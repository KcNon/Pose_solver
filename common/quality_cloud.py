"""Quality-aware multi-view point-cloud construction.

The original backprojection path concatenates every accepted pixel from all
views.  This module keeps the views separate long enough to reject mask/depth
boundaries and to measure whether a 3D sample is supported by another camera.
The support test is deliberately diagnostic: unsupported points may be valid
because of visibility, so callers choose whether to retain or reject them.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from common.geom import backproject_view


@dataclass
class ViewCloud:
    points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    support: np.ndarray
    candidate_pixels: int


def filter_centroid_consistent_views(
    clouds: list[ViewCloud],
    *,
    radius_m: float,
    minimum_points: int = 30,
) -> tuple[list[ViewCloud], dict]:
    """Keep the largest group of view clouds with nearby robust centroids."""
    active = [
        index for index, cloud in enumerate(clouds)
        if len(cloud.points) >= int(minimum_points)
    ]
    centers = {
        index: np.median(clouds[index].points, axis=0)
        for index in active
    }
    if len(active) <= 1:
        selected = list(active)
    else:
        candidates = []
        for index in active:
            members = [
                other for other in active
                if np.linalg.norm(centers[other] - centers[index])
                <= float(radius_m)
            ]
            distances = [
                float(np.linalg.norm(centers[other] - centers[index]))
                for other in members
            ]
            candidates.append((
                len(members),
                -float(np.mean(distances)) if distances else 0.0,
                -index,
                members,
            ))
        best = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        selected = list(best[3])
    selected_set = set(selected)
    filtered = []
    for index, cloud in enumerate(clouds):
        if index in selected_set:
            filtered.append(cloud)
            continue
        filtered.append(ViewCloud(
            points=np.empty((0, 3), dtype=np.float64),
            colors=np.empty((0, 3), dtype=np.uint8),
            confidence=np.empty((0,), dtype=np.float32),
            support=np.empty((0,), dtype=np.int16),
            candidate_pixels=cloud.candidate_pixels,
        ))
    return filtered, {
        "enabled": True,
        "radius_m": float(radius_m),
        "minimum_points": int(minimum_points),
        "active_view_indices": active,
        "selected_view_indices": selected,
        "rejected_view_indices": [
            index for index in active if index not in selected_set
        ],
        "centroids_world": {
            str(index): centers[index].tolist() for index in active
        },
    }


def eroded_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    value = mask.astype(np.uint8)
    if iterations > 0:
        value = cv2.erode(value, np.ones((3, 3), np.uint8), iterations=iterations)
    return value.astype(bool)


def smooth_depth_mask(depth: np.ndarray, threshold_m: float) -> np.ndarray:
    """Return pixels whose local depth range is below ``threshold_m``.

    Invalid neighbours are replaced with the centre value.  This makes the
    mask reject actual object/background discontinuities without deleting a
    band around missing depth pixels merely because they are encoded as zero.
    """
    valid = np.isfinite(depth) & (depth > 1e-3)
    centre = np.where(valid, depth, 0.0).astype(np.float32)
    local_min = centre.copy()
    local_max = centre.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.roll(centre, (dy, dx), axis=(0, 1))
        shifted_valid = np.roll(valid, (dy, dx), axis=(0, 1))
        neighbour = np.where(shifted_valid, shifted, centre)
        local_min = np.minimum(local_min, neighbour)
        local_max = np.maximum(local_max, neighbour)
    smooth = valid & ((local_max - local_min) <= float(threshold_m))
    smooth[[0, -1], :] = False
    smooth[:, [0, -1]] = False
    return smooth


def prepare_view_clouds(
    depth: np.ndarray,
    colors: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    confidence: np.ndarray,
    masks: list[np.ndarray],
    *,
    conf_quantile: float = 0.5,
    mask_erode: int = 2,
    depth_edge_m: float = 0.008,
    stride: int = 2,
) -> list[ViewCloud]:
    """Backproject independently filtered clouds for every view."""
    if int(stride) < 1:
        raise ValueError("point-cloud stride must be at least 1")
    result: list[ViewCloud] = []
    for view in range(depth.shape[0]):
        mask = eroded_mask(masks[view], mask_erode)
        valid = mask & np.isfinite(depth[view]) & (depth[view] > 1e-3)
        valid &= np.isfinite(confidence[view]) & (confidence[view] > 0)
        if depth_edge_m > 0:
            valid &= smooth_depth_mask(depth[view], depth_edge_m)
        values = confidence[view][valid]
        if len(values):
            valid &= confidence[view] >= float(np.quantile(values, conf_quantile))
        sample = np.zeros_like(valid)
        sample[::stride, ::stride] = True
        valid &= sample
        points, point_colors = backproject_view(
            depth[view], intrinsics[view], extrinsics[view],
            mask=valid, color=colors[view])
        result.append(ViewCloud(
            points=np.asarray(points, np.float64),
            colors=np.asarray(point_colors, np.uint8),
            confidence=np.asarray(confidence[view][valid], np.float32),
            support=np.zeros(len(points), np.int16),
            candidate_pixels=int(valid.sum()),
        ))
    return result


def assign_cross_view_support(
    clouds: list[ViewCloud],
    radius_m: float,
    *,
    workers: int = 4,
) -> None:
    """Count other views with a point within ``radius_m`` of each sample."""
    if not 1 <= int(workers) <= 32:
        raise ValueError("nearest-neighbour workers must be in [1, 32]")
    trees = [cKDTree(cloud.points) if len(cloud.points) else None for cloud in clouds]
    for source_index, source in enumerate(clouds):
        support = np.zeros(len(source.points), np.int16)
        for target_index, tree in enumerate(trees):
            if target_index == source_index or tree is None or not len(source.points):
                continue
            distance, _ = tree.query(
                source.points, k=1, workers=int(workers)
            )
            support += distance <= float(radius_m)
        source.support = support


def fuse_supported_clouds(
    clouds: list[ViewCloud],
    *,
    min_support: int = 1,
    retain_unsupported_fraction: float = 0.0,
    voxel_size_m: float = 0.0,
    max_points: int = 80000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fuse supported samples, optionally retaining top-confidence singletons."""
    if float(voxel_size_m) < 0:
        raise ValueError("voxel_size_m cannot be negative")
    points, colors, confidences, supports = [], [], [], []
    stats = {"views": [], "support_histogram": {}}
    rng = np.random.default_rng(seed)
    for cloud in clouds:
        keep = cloud.support >= int(min_support)
        unsupported = np.flatnonzero(~keep)
        if retain_unsupported_fraction > 0 and len(unsupported):
            count = int(round(len(unsupported) * retain_unsupported_fraction))
            if count > 0:
                ranked = unsupported[np.argsort(cloud.confidence[unsupported])[-count:]]
                keep[ranked] = True
        points.append(cloud.points[keep])
        colors.append(cloud.colors[keep])
        confidences.append(cloud.confidence[keep])
        supports.append(cloud.support[keep])
        stats["views"].append({
            "candidate_points": int(len(cloud.points)),
            "supported_points": int((cloud.support >= min_support).sum()),
            "retained_points": int(keep.sum()),
        })
        values, counts = np.unique(cloud.support, return_counts=True)
        for value, count in zip(values, counts):
            key = str(int(value))
            stats["support_histogram"][key] = stats["support_histogram"].get(key, 0) + int(count)
    fused_points = np.concatenate(points) if points else np.empty((0, 3), np.float64)
    fused_colors = np.concatenate(colors) if colors else np.empty((0, 3), np.uint8)
    fused_confidence = (
        np.concatenate(confidences) if confidences else np.empty((0,), np.float32)
    )
    fused_support = (
        np.concatenate(supports) if supports else np.empty((0,), np.int16)
    )
    stats["pre_voxel_points"] = int(len(fused_points))
    if voxel_size_m > 0 and len(fused_points):
        cells = np.floor(fused_points / float(voxel_size_m)).astype(np.int64)
        _, inverse = np.unique(cells, axis=0, return_inverse=True)
        count = int(inverse.max()) + 1
        confidence_cap = float(
            np.quantile(fused_confidence, 0.99)
            if len(fused_confidence) >= 100
            else np.max(fused_confidence)
        )
        weights = np.minimum(fused_confidence, confidence_cap).astype(np.float64)
        weights = np.maximum(weights, 1e-6) * (1.0 + fused_support)
        weight_sum = np.bincount(inverse, weights=weights, minlength=count)
        averaged_points = np.column_stack([
            np.bincount(
                inverse,
                weights=weights * fused_points[:, axis],
                minlength=count,
            ) / weight_sum
            for axis in range(3)
        ])
        averaged_colors = np.column_stack([
            np.bincount(
                inverse,
                weights=weights * fused_colors[:, channel],
                minlength=count,
            ) / weight_sum
            for channel in range(3)
        ])
        fused_points = averaged_points
        fused_colors = np.clip(np.rint(averaged_colors), 0, 255).astype(np.uint8)
    if len(fused_points) > max_points:
        index = rng.choice(len(fused_points), max_points, replace=False)
        fused_points, fused_colors = fused_points[index], fused_colors[index]
    stats["fused_points"] = int(len(fused_points))
    stats["fusion_voxel_m"] = float(voxel_size_m)
    return fused_points, fused_colors, stats


def supported_view_clouds(
    clouds: list[ViewCloud],
    *,
    min_support: int = 1,
) -> list[ViewCloud]:
    """Return per-view clouds containing only cross-view-supported samples.

    Quality metrics must be computed on the points that will actually reach
    registration.  Measuring the unfiltered candidates lets a single bad
    camera dominate the frame report even when the supported subset is sound;
    conversely, retaining unsupported points can make a non-empty cloud look
    usable when no camera agrees with any other camera.
    """

    result = []
    for cloud in clouds:
        keep = cloud.support >= int(min_support)
        result.append(ViewCloud(
            points=cloud.points[keep],
            colors=cloud.colors[keep],
            confidence=cloud.confidence[keep],
            support=cloud.support[keep],
            candidate_pixels=cloud.candidate_pixels,
        ))
    return result


def quality_gate(
    stats: dict,
    cross_view: dict,
    reprojection_depth: dict,
    *,
    minimum_fused_points: int = 300,
    minimum_supported_views: int = 3,
    minimum_supported_points_per_view: int = 30,
    minimum_support_fraction: float = 0.02,
    maximum_cross_view_median_m: float = 0.04,
    minimum_cross_view_overlap_ratio: float = 0.10,
    maximum_reprojection_median_m: float = 0.04,
    minimum_reprojection_inlier_ratio: float = 0.10,
    allow_single_view: bool = False,
    allow_reprojection_override: bool = False,
) -> dict:
    """Fail-closed quality decision for a fused multi-view cloud."""

    view_rows = list(stats.get("views", []))
    candidates = int(sum(row.get("candidate_points", 0) for row in view_rows))
    supported = int(sum(row.get("supported_points", 0) for row in view_rows))
    supported_views = int(sum(
        int(row.get("supported_points", 0))
        >= int(minimum_supported_points_per_view)
        for row in view_rows
    ))
    support_fraction = float(supported / max(candidates, 1))
    single_view_fallback_used = bool(
        allow_single_view and supported_views == 1
    )
    reprojection_median = reprojection_depth.get("median_m")
    reprojection_inlier = reprojection_depth.get("inlier_ratio")
    reprojection_override_used = bool(
        allow_reprojection_override
        and supported_views >= 2
        and reprojection_median is not None
        and reprojection_inlier is not None
        and float(reprojection_median) <= float(maximum_reprojection_median_m)
        and float(reprojection_inlier) >= float(minimum_reprojection_inlier_ratio)
    )
    reasons = []
    if int(stats.get("fused_points", 0)) < int(minimum_fused_points):
        reasons.append("too_few_fused_points")
    if supported_views < int(minimum_supported_views):
        reasons.append("too_few_supported_views")
    if support_fraction < float(minimum_support_fraction):
        reasons.append("support_fraction_below_minimum")

    cross_median = cross_view.get("median_m")
    cross_overlap = cross_view.get("overlap_ratio")
    if cross_median is None:
        if not single_view_fallback_used and not reprojection_override_used:
            reasons.append("missing_cross_view_consistency")
    elif (
        not reprojection_override_used
        and float(cross_median) > float(maximum_cross_view_median_m)
    ):
        reasons.append("cross_view_median_above_maximum")
    if cross_overlap is None:
        if not single_view_fallback_used and not reprojection_override_used:
            reasons.append("missing_cross_view_overlap")
    elif (
        not reprojection_override_used
        and float(cross_overlap) < float(minimum_cross_view_overlap_ratio)
    ):
        reasons.append("cross_view_overlap_below_minimum")

    if reprojection_median is None:
        if not single_view_fallback_used:
            reasons.append("missing_reprojection_consistency")
    elif float(reprojection_median) > float(maximum_reprojection_median_m):
        reasons.append("reprojection_median_above_maximum")
    if reprojection_inlier is None:
        if not single_view_fallback_used:
            reasons.append("missing_reprojection_inliers")
    elif float(reprojection_inlier) < float(minimum_reprojection_inlier_ratio):
        reasons.append("reprojection_inlier_ratio_below_minimum")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "candidate_points": candidates,
        "supported_points": supported,
        "supported_views": supported_views,
        "support_fraction": support_fraction,
        "single_view_fallback_used": single_view_fallback_used,
        "reprojection_override_used": reprojection_override_used,
        "thresholds": {
            "minimum_fused_points": int(minimum_fused_points),
            "minimum_supported_views": int(minimum_supported_views),
            "minimum_supported_points_per_view": int(
                minimum_supported_points_per_view
            ),
            "minimum_support_fraction": float(minimum_support_fraction),
            "maximum_cross_view_median_m": float(
                maximum_cross_view_median_m
            ),
            "minimum_cross_view_overlap_ratio": float(
                minimum_cross_view_overlap_ratio
            ),
            "maximum_reprojection_median_m": float(
                maximum_reprojection_median_m
            ),
            "minimum_reprojection_inlier_ratio": float(
                minimum_reprojection_inlier_ratio
            ),
        },
    }


def cross_view_consistency(
    clouds: list[ViewCloud],
    radius_m: float,
    *,
    workers: int = 4,
) -> dict:
    """Symmetric nearest-neighbour consistency over overlapping view pairs."""
    if not 1 <= int(workers) <= 32:
        raise ValueError("nearest-neighbour workers must be in [1, 32]")
    pair_reports = []
    for first in range(len(clouds)):
        if not len(clouds[first].points):
            continue
        for second in range(first + 1, len(clouds)):
            if not len(clouds[second].points):
                continue
            a, b = clouds[first].points, clouds[second].points
            d_ab, _ = cKDTree(b).query(a, k=1, workers=int(workers))
            d_ba, _ = cKDTree(a).query(b, k=1, workers=int(workers))
            distances = np.concatenate((d_ab, d_ba))
            pair_reports.append({
                "views": [first, second],
                "median_m": float(np.median(distances)),
                "p90_m": float(np.quantile(distances, 0.9)),
                "overlap_ratio": float(np.mean(distances <= radius_m)),
                "samples": int(len(distances)),
            })
    if not pair_reports:
        return {"pairs": [], "median_m": None, "p90_m": None,
                "overlap_ratio": None, "samples": 0}
    weights = np.asarray([item["samples"] for item in pair_reports], float)
    return {
        "pairs": pair_reports,
        "median_m": float(np.average([item["median_m"] for item in pair_reports], weights=weights)),
        "p90_m": float(np.average([item["p90_m"] for item in pair_reports], weights=weights)),
        "overlap_ratio": float(np.average([item["overlap_ratio"] for item in pair_reports], weights=weights)),
        "samples": int(weights.sum()),
    }


def reprojection_depth_consistency(
    clouds: list[ViewCloud],
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    masks: list[np.ndarray],
    *,
    threshold_m: float = 0.008,
    target_mask_erode: int = 1,
) -> dict:
    """Compare each world point against another view's depth at its projection.

    Unlike point-cloud nearest neighbours, this only compares samples with a
    defined correspondence: the same target-camera ray inside the target part
    mask.  Different visible sides of an object therefore do not create an
    artificial error merely because their Euclidean point sets do not overlap.
    """
    pair_reports = []
    for source_index, source in enumerate(clouds):
        if not len(source.points):
            continue
        for target_index in range(len(clouds)):
            if target_index == source_index:
                continue
            E = np.asarray(extrinsics[target_index], np.float64)
            camera = source.points @ E[:3, :3].T + E[:3, 3]
            positive = camera[:, 2] > 1e-4
            projected = camera @ np.asarray(intrinsics[target_index], np.float64).T
            u = np.rint(projected[:, 0] / projected[:, 2]).astype(np.int64)
            v = np.rint(projected[:, 1] / projected[:, 2]).astype(np.int64)
            height, width = depth[target_index].shape
            inside = positive & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            index = np.flatnonzero(inside)
            if not len(index):
                continue
            target_mask = eroded_mask(masks[target_index], target_mask_erode)
            valid = target_mask[v[index], u[index]]
            target_depth = depth[target_index][v[index], u[index]]
            valid &= np.isfinite(target_depth) & (target_depth > 1e-3)
            index = index[valid]
            target_depth = target_depth[valid]
            if not len(index):
                continue
            residual = np.abs(camera[index, 2] - target_depth)
            pair_reports.append({
                "views": [source_index, target_index],
                "median_m": float(np.median(residual)),
                "p90_m": float(np.quantile(residual, 0.9)),
                "inlier_ratio": float(np.mean(residual <= threshold_m)),
                "samples": int(len(residual)),
            })
    if not pair_reports:
        return {"pairs": [], "median_m": None, "p90_m": None,
                "inlier_ratio": None, "samples": 0}
    weights = np.asarray([item["samples"] for item in pair_reports], float)
    return {
        "pairs": pair_reports,
        "median_m": float(np.average([item["median_m"] for item in pair_reports], weights=weights)),
        "p90_m": float(np.average([item["p90_m"] for item in pair_reports], weights=weights)),
        "inlier_ratio": float(np.average([item["inlier_ratio"] for item in pair_reports], weights=weights)),
        "samples": int(weights.sum()),
    }
