"""Depth-aware mask transfer between calibrated camera views."""
from __future__ import annotations

import numpy as np

from common.geom import backproject_view, project_points, rasterize_points


def project_mask_to_view(
    source_mask: np.ndarray,
    source_depth: np.ndarray,
    source_intrinsic: np.ndarray,
    source_extrinsic: np.ndarray,
    target_depth: np.ndarray,
    target_intrinsic: np.ndarray,
    target_extrinsic: np.ndarray,
    *,
    depth_tolerance: float = 0.03,
    stride: int = 1,
    dilate: int = 2,
    close_kernel: int = 5,
) -> np.ndarray:
    """Project a reliable source-view mask into a target view.

    Target depth is used as an occlusion test, so points hidden behind another
    surface do not become target foreground.  The result is a geometric prior:
    it may miss surfaces not visible in any source view and should normally be
    fed back to SAM rather than blindly replacing a valid target mask.
    """
    import cv2

    mask = np.asarray(source_mask, dtype=bool)
    if stride > 1:
        sample = np.zeros_like(mask)
        sample[::stride, ::stride] = True
        mask &= sample
    points, _ = backproject_view(
        np.asarray(source_depth, dtype=np.float32),
        np.asarray(source_intrinsic),
        np.asarray(source_extrinsic),
        mask=mask,
    )
    if not len(points):
        return np.zeros(target_depth.shape, dtype=bool)
    uv, z = project_points(points, target_intrinsic, target_extrinsic)
    projected = rasterize_points(
        uv,
        z,
        target_depth.shape,
        depth_map=target_depth,
        depth_tol=depth_tolerance,
        dilate=dilate,
    )
    if close_kernel > 1 and projected.any():
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
        )
        projected = cv2.morphologyEx(
            projected.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
    return projected


def multiview_geometric_prior(
    masks: list[np.ndarray],
    reliable: list[bool],
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    target_index: int,
    *,
    minimum_source_views: int = 1,
    depth_tolerance: float = 0.03,
    minimum_pixels: int = 100,
) -> tuple[np.ndarray, dict]:
    """Fuse projections from reliable source views into one target prior."""
    target_shape = tuple(depth[target_index].shape)
    support = np.zeros(target_shape, dtype=np.uint16)
    used_sources = []
    for source_index, is_reliable in enumerate(reliable):
        if source_index == target_index or not is_reliable:
            continue
        projected = project_mask_to_view(
            masks[source_index],
            depth[source_index],
            intrinsics[source_index],
            extrinsics[source_index],
            depth[target_index],
            intrinsics[target_index],
            extrinsics[target_index],
            depth_tolerance=depth_tolerance,
        )
        if int(projected.sum()) < minimum_pixels:
            continue
        support += projected.astype(np.uint16)
        used_sources.append(source_index)
    prior = support >= max(1, int(minimum_source_views))
    return prior, {
        "target_index": int(target_index),
        "source_indices": used_sources,
        "minimum_source_views": int(minimum_source_views),
        "pixels": int(prior.sum()),
        "max_support": int(support.max()) if support.size else 0,
    }
