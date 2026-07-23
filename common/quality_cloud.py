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


def assign_cross_view_support(clouds: list[ViewCloud], radius_m: float) -> None:
    """Count other views with a point within ``radius_m`` of each sample."""
    trees = [cKDTree(cloud.points) if len(cloud.points) else None for cloud in clouds]
    for source_index, source in enumerate(clouds):
        support = np.zeros(len(source.points), np.int16)
        for target_index, tree in enumerate(trees):
            if target_index == source_index or tree is None or not len(source.points):
                continue
            distance, _ = tree.query(source.points, k=1, workers=-1)
            support += distance <= float(radius_m)
        source.support = support


def fuse_supported_clouds(
    clouds: list[ViewCloud],
    *,
    min_support: int = 1,
    retain_unsupported_fraction: float = 0.0,
    max_points: int = 80000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fuse supported samples, optionally retaining top-confidence singletons."""
    points, colors = [], []
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
    if len(fused_points) > max_points:
        index = rng.choice(len(fused_points), max_points, replace=False)
        fused_points, fused_colors = fused_points[index], fused_colors[index]
    stats["fused_points"] = int(len(fused_points))
    return fused_points, fused_colors, stats


def cross_view_consistency(clouds: list[ViewCloud], radius_m: float) -> dict:
    """Symmetric nearest-neighbour consistency over overlapping view pairs."""
    pair_reports = []
    for first in range(len(clouds)):
        if not len(clouds[first].points):
            continue
        for second in range(first + 1, len(clouds)):
            if not len(clouds[second].points):
                continue
            a, b = clouds[first].points, clouds[second].points
            d_ab, _ = cKDTree(b).query(a, k=1, workers=-1)
            d_ba, _ = cKDTree(a).query(b, k=1, workers=-1)
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
