"""Shared mesh styling, camera scaling, and silhouette review metrics."""
from __future__ import annotations

import colorsys

import cv2
import numpy as np
import trimesh

from common.normalized_recon import scale_intrinsics


def tile_image_panels(
    panels: list[np.ndarray],
    *,
    columns: int = 4,
    fill_value: int = 0,
) -> np.ndarray:
    """Tile equally sized images, padding the last row when it is incomplete."""
    if not panels:
        raise ValueError("at least one image panel is required")
    if columns <= 0:
        raise ValueError("columns must be positive")
    shape = panels[0].shape
    if any(panel.shape != shape for panel in panels):
        raise ValueError("all image panels must have the same shape")
    padded = list(panels)
    missing = (-len(padded)) % columns
    padded.extend([
        np.full_like(panels[0], fill_value) for _ in range(missing)
    ])
    return np.vstack([
        np.hstack(padded[index : index + columns])
        for index in range(0, len(padded), columns)
    ])


def part_color(index: int) -> tuple[int, int, int]:
    hue = (0.31 + 0.38196601125 * index) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.68, 0.92)
    return tuple(int(round(255 * value)) for value in rgb)


def solid_mesh(
    mesh: trimesh.Trimesh,
    rgb: tuple[int, int, int],
) -> trimesh.Trimesh:
    result = mesh.copy()
    rgba = np.asarray([*rgb, 255], dtype=np.uint8)
    result.visual = trimesh.visual.ColorVisuals(
        mesh=result,
        face_colors=np.tile(rgba, (len(result.faces), 1)),
    )
    return result


def camera_from_recon(
    recon: dict,
    view_index: int,
    output_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = scale_intrinsics(
        recon["intrinsics"][view_index],
        recon["depth_hw"],
        output_hw,
    )
    return intrinsics, recon["extrinsics"][view_index]


def silhouette_review_metrics(
    rendered: np.ndarray,
    target: np.ndarray,
    edge_cap_px: float = 20.0,
) -> tuple[float, float]:
    union = np.logical_or(rendered, target).sum()
    iou = (
        float(np.logical_and(rendered, target).sum() / union)
        if union
        else 1.0
    )
    kernel = np.ones((3, 3), np.uint8)
    rendered_edge = cv2.morphologyEx(
        rendered.astype(np.uint8), cv2.MORPH_GRADIENT, kernel
    ).astype(bool)
    target_edge = cv2.morphologyEx(
        target.astype(np.uint8), cv2.MORPH_GRADIENT, kernel
    ).astype(bool)
    if not rendered_edge.any() or not target_edge.any():
        return iou, edge_cap_px
    target_distance = cv2.distanceTransform(
        (~target_edge).astype(np.uint8), cv2.DIST_L2, 3
    )
    rendered_distance = cv2.distanceTransform(
        (~rendered_edge).astype(np.uint8), cv2.DIST_L2, 3
    )
    chamfer = 0.5 * (
        float(np.minimum(target_distance[rendered_edge], edge_cap_px).mean())
        + float(np.minimum(rendered_distance[target_edge], edge_cap_px).mean())
    )
    return iou, chamfer
