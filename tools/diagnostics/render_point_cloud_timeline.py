#!/usr/bin/env python
"""Render a fixed-scale timeline from per-view clouds to the final pose.

The quality-cloud stage deliberately preserves one PLY per source camera.  This
tool visualizes those prepared candidates, recomputes the cross-view-supported
subset, compares it with the final fused PLY used by registration, and overlays
the trajectory mesh.  Orthographic panels never auto-rescale per frame.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.cloud_io import read_ply_xyz
from common.io_utils import load_json, write_json
from common.mask_io import frame_path
from common.normalized_recon import load_recon, scale_intrinsics
from common.quality_cloud import ViewCloud, assign_cross_view_support


VIEW_COLORS_RGB = [
    (66, 165, 245),
    (255, 167, 38),
    (102, 187, 106),
    (239, 83, 80),
    (171, 71, 188),
    (38, 198, 218),
    (255, 238, 88),
    (236, 64, 122),
]
BACKGROUND = (11, 16, 24)
PANEL_BACKGROUND = (15, 22, 32)
GRID_COLOR = (45, 57, 72)
TEXT_COLOR = (239, 244, 250)
MUTED_TEXT = (177, 191, 208)


@dataclass(frozen=True)
class ProjectionBounds:
    """Fixed metric bounds in the coordinates produced by ``basis``."""

    basis: np.ndarray
    center_xy: np.ndarray
    span_m: float


@dataclass
class FrameClouds:
    names: list[str]
    candidates: list[np.ndarray]
    support: list[np.ndarray]
    fused: np.ndarray


def _sample(points: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) <= int(maximum):
        return points
    index = np.random.default_rng(seed).choice(
        len(points), int(maximum), replace=False
    )
    return points[index]


def isometric_basis() -> np.ndarray:
    yaw = np.radians(-45.0)
    elevation = np.radians(28.0)
    camera = np.asarray([
        np.cos(elevation) * np.cos(yaw),
        np.cos(elevation) * np.sin(yaw),
        np.sin(elevation),
    ])
    camera /= np.linalg.norm(camera)
    right = np.cross(np.asarray([0.0, 0.0, 1.0]), camera)
    right /= np.linalg.norm(right)
    up = np.cross(camera, right)
    up /= np.linalg.norm(up)
    return np.stack((right, up, camera))


def top_basis() -> np.ndarray:
    return np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)


def front_basis() -> np.ndarray:
    return np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)


def project_world_points(
    points: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into one camera; return UV and camera depth."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return np.empty((0, 2), float), np.empty((0,), float)
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    camera = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    projected = camera @ np.asarray(intrinsic, dtype=np.float64).T
    denominator = projected[:, 2]
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    uv[valid] = projected[valid, :2] / denominator[valid, None]
    return uv, camera[:, 2]


def robust_pca_extents_mm(points: np.ndarray) -> list[float] | None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 6:
        return None
    center = np.median(points, axis=0)
    covariance = np.cov((points - center).T)
    _, vectors = np.linalg.eigh(covariance)
    projected = (points - center) @ vectors
    low, high = np.quantile(projected, [0.01, 0.99], axis=0)
    return sorted((1000.0 * (high - low)).tolist(), reverse=True)


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = TEXT_COLOR,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _panel(width: int, height: int, title: str) -> np.ndarray:
    image = np.full((height, width, 3), PANEL_BACKGROUND, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (width - 1, height - 1), (74, 91, 112), 2)
    _draw_text(image, title, (14, 27), scale=0.65, thickness=2)
    return image


def _letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, int, int]:
    width, height = size
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    output = np.full((height, width, 3), PANEL_BACKGROUND, dtype=np.uint8)
    x0 = (width - resized_w) // 2
    y0 = (height - resized_h) // 2
    output[y0:y0 + resized_h, x0:x0 + resized_w] = resized
    return output, scale, x0, y0


def _draw_projected_clouds(
    image: np.ndarray,
    clouds: Iterable[np.ndarray],
    colors: Iterable[tuple[int, int, int]],
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    *,
    radius: int = 2,
    alpha: float = 0.78,
) -> None:
    entries: list[tuple[float, int, int, tuple[int, int, int]]] = []
    height, width = image.shape[:2]
    for cloud, color in zip(clouds, colors):
        uv, depth = project_world_points(cloud, intrinsic, extrinsic)
        if not len(uv):
            continue
        finite = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 1e-4)
        pixels = np.zeros((len(uv), 2), dtype=np.int64)
        pixels[finite] = np.rint(uv[finite]).astype(np.int64)
        valid = (
            finite
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        entries.extend(
            (float(z), int(x), int(y), color)
            for (x, y), z in zip(pixels[valid], depth[valid])
        )
    if not entries:
        return
    overlay = image.copy()
    for _, x, y, color in sorted(entries, reverse=True):
        cv2.circle(overlay, (x, y), radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def render_camera_panel(
    rgb: np.ndarray,
    mask: np.ndarray,
    clouds: list[np.ndarray],
    colors: list[tuple[int, int, int]],
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    *,
    title: str,
    size: tuple[int, int],
    mesh_points: np.ndarray | None = None,
    pose_valid: bool = True,
) -> np.ndarray:
    source = np.asarray(rgb, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(source, contours, -1, (80, 255, 120), 3, cv2.LINE_AA)
    _draw_projected_clouds(
        source, clouds, colors, intrinsic, extrinsic, radius=3, alpha=0.70
    )
    if mesh_points is not None and len(mesh_points):
        mesh_color = (255, 255, 255) if pose_valid else (255, 65, 65)
        _draw_projected_clouds(
            source,
            [mesh_points],
            [mesh_color],
            intrinsic,
            extrinsic,
            radius=1,
            alpha=0.88,
        )
    panel, _, _, _ = _letterbox(source, size)
    cv2.rectangle(panel, (0, 0), (size[0] - 1, size[1] - 1), (74, 91, 112), 2)
    cv2.rectangle(panel, (0, 0), (size[0] - 1, 39), (8, 13, 20), -1)
    _draw_text(panel, title, (14, 27), scale=0.62, thickness=2)
    return panel


def projection_bounds(
    point_sets: Iterable[np.ndarray],
    basis: np.ndarray,
    *,
    padding: float = 1.12,
    minimum_span_m: float = 0.15,
) -> ProjectionBounds:
    nonempty = [
        np.asarray(points, dtype=np.float64).reshape(-1, 3)
        for points in point_sets
        if len(points)
    ]
    if not nonempty:
        return ProjectionBounds(
            basis=np.asarray(basis, float),
            center_xy=np.zeros(2, float),
            span_m=float(minimum_span_m),
        )
    points = np.concatenate(nonempty, axis=0)
    projected = points @ np.asarray(basis, dtype=np.float64).T
    low, high = np.quantile(projected[:, :2], [0.005, 0.995], axis=0)
    center = 0.5 * (low + high)
    span = max(float(np.max(high - low)) * float(padding), float(minimum_span_m))
    return ProjectionBounds(
        basis=np.asarray(basis, float), center_xy=center, span_m=span
    )


def _metric_grid(
    image: np.ndarray,
    *,
    center_xy: np.ndarray,
    span_m: float,
    padding: int,
    step_m: float = 0.05,
) -> None:
    height, width = image.shape[:2]
    pixels_per_m = min(
        (width - 2 * padding) / span_m,
        (height - 2 * padding) / span_m,
    )
    for axis in range(2):
        low = center_xy[axis] - span_m / 2
        high = center_xy[axis] + span_m / 2
        first = np.ceil(low / step_m) * step_m
        for value in np.arange(first, high + step_m * 0.5, step_m):
            if axis == 0:
                x = int(round(width / 2 + (value - center_xy[0]) * pixels_per_m))
                cv2.line(image, (x, padding), (x, height - padding), GRID_COLOR, 1)
            else:
                y = int(round(height / 2 - (value - center_xy[1]) * pixels_per_m))
                cv2.line(image, (padding, y), (width - padding, y), GRID_COLOR, 1)


def render_orthographic_panel(
    clouds: list[np.ndarray],
    colors: list[tuple[int, int, int]],
    bounds: ProjectionBounds,
    *,
    size: tuple[int, int],
    title: str,
    mesh_points: np.ndarray | None = None,
    pose_valid: bool = True,
    center_world: np.ndarray | None = None,
) -> np.ndarray:
    width, height = size
    image = _panel(width, height, title)
    padding = 34
    basis = bounds.basis
    center_xy = bounds.center_xy.copy()
    if center_world is not None:
        center_xy = np.asarray(center_world, float) @ basis.T
        center_xy = center_xy[:2]
    _metric_grid(
        image,
        center_xy=center_xy,
        span_m=bounds.span_m,
        padding=padding,
    )
    pixels_per_m = min(
        (width - 2 * padding) / bounds.span_m,
        (height - 2 * padding) / bounds.span_m,
    )
    draw_clouds = list(clouds)
    draw_colors = list(colors)
    if mesh_points is not None and len(mesh_points):
        draw_clouds.append(mesh_points)
        draw_colors.append((255, 255, 255) if pose_valid else (255, 65, 65))
    entries: list[tuple[float, int, int, tuple[int, int, int], int]] = []
    for index, (cloud, color) in enumerate(zip(draw_clouds, draw_colors)):
        cloud = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
        if not len(cloud):
            continue
        projected = cloud @ basis.T
        x = np.rint(width / 2 + (projected[:, 0] - center_xy[0]) * pixels_per_m).astype(int)
        y = np.rint(height / 2 - (projected[:, 1] - center_xy[1]) * pixels_per_m).astype(int)
        valid = (x >= 0) & (x < width) & (y >= 36) & (y < height)
        radius = 1 if index == len(draw_clouds) - 1 and mesh_points is not None else 2
        entries.extend(
            (float(z), int(px), int(py), color, radius)
            for z, px, py in zip(projected[valid, 2], x[valid], y[valid])
        )
    for _, x, y, color, radius in sorted(entries):
        cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)
    bar_m = 0.10 if bounds.span_m >= 0.30 else 0.05
    bar_px = int(round(bar_m * pixels_per_m))
    y = height - 22
    cv2.line(image, (width - 30 - bar_px, y), (width - 30, y), TEXT_COLOR, 3)
    _draw_text(
        image,
        f"{int(round(bar_m * 1000))} mm",
        (width - 36 - bar_px, y - 8),
        scale=0.42,
        color=MUTED_TEXT,
    )
    return image


def _view_clouds(
    quality_root: Path,
    timestamp: str,
    part: str,
    views: list[str],
    *,
    support_radius_m: float,
    min_support: int,
    maximum_per_view: int,
) -> FrameClouds:
    candidates = []
    for view_index, view in enumerate(views):
        path = quality_root / timestamp / "views" / part / f"{view}.ply"
        points = read_ply_xyz(path) if path.exists() else np.empty((0, 3), float)
        candidates.append(_sample(points, maximum_per_view, 7919 + view_index))
    wrapped = [
        ViewCloud(
            points=points,
            colors=np.zeros((len(points), 3), np.uint8),
            confidence=np.ones(len(points), np.float32),
            support=np.zeros(len(points), np.int16),
            candidate_pixels=len(points),
        )
        for points in candidates
    ]
    assign_cross_view_support(wrapped, float(support_radius_m))
    supported = [
        cloud.points[cloud.support >= int(min_support)] for cloud in wrapped
    ]
    fused_path = quality_root / timestamp / f"{part}.ply"
    fused = read_ply_xyz(fused_path) if fused_path.exists() else np.empty((0, 3), float)
    return FrameClouds(
        names=list(views),
        candidates=candidates,
        support=supported,
        fused=fused,
    )


def _load_mask(config: dict, timestamp: str, view: str, part: str) -> np.ndarray:
    path = Path(config["masks_dir"]) / timestamp / f"{view}.png"
    value = np.asarray(Image.open(path))
    part_id = int(config.get("part_ids", {}).get(
        part, list(config["parts"]).index(part) + 1
    ))
    if value.ndim == 2:
        return (value == part_id).astype(np.uint8) * 255
    raise ValueError(
        f"timeline diagnostic requires palette masks, got RGB mask {path}"
    )


def _load_rgb(config: dict, timestamp: str, view: str) -> np.ndarray:
    path = frame_path(
        config["frames_dir"],
        config.get("frames_layout", "normalized"),
        timestamp,
        view,
    )
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _mesh_vertices(mesh: trimesh.Trimesh, maximum: int, seed: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return _sample(vertices, maximum, seed)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _mesh_cloud_distance_mm(
    mesh_points: np.ndarray, cloud: np.ndarray
) -> tuple[float | None, float | None]:
    if len(mesh_points) < 3 or len(cloud) < 3:
        return None, None
    distance, _ = cKDTree(mesh_points).query(cloud, k=1, workers=-1)
    return float(np.median(distance) * 1000.0), float(np.quantile(distance, 0.90) * 1000.0)


def _view_centroid_spread_mm(clouds: list[np.ndarray]) -> float | None:
    centers = [np.median(cloud, axis=0) for cloud in clouds if len(cloud) >= 10]
    if len(centers) < 2:
        return None
    values = []
    for first in range(len(centers)):
        for second in range(first + 1, len(centers)):
            values.append(np.linalg.norm(centers[first] - centers[second]))
    return float(np.median(values) * 1000.0)


def _quality_metrics(row: dict) -> dict:
    views = list(row.get("views", []))
    gate = row.get("quality_gate", {}) or {}
    cross = row.get("cross_view", {}) or {}
    reprojection = row.get("reprojection_depth", {}) or {}
    candidate = int(sum(item.get("candidate_points", 0) for item in views))
    supported = int(sum(item.get("supported_points", 0) for item in views))
    retained = int(sum(item.get("retained_points", 0) for item in views))
    return {
        "cloud_status": row.get("status"),
        "candidate_points": candidate,
        "supported_points": supported,
        "retained_points_before_voxel": retained,
        "support_fraction": float(supported / max(candidate, 1)),
        "supported_views": gate.get(
            "supported_views",
            int(sum(int(item.get("supported_points", 0)) >= 30 for item in views)),
        ),
        "quality_passed": (
            bool(gate.get("passed")) if "passed" in gate else None
        ),
        "quality_reasons": list(gate.get("reasons", [])),
        "cross_view_median_mm": (
            None if cross.get("median_m") is None
            else 1000.0 * float(cross["median_m"])
        ),
        "cross_view_overlap_ratio": cross.get("overlap_ratio"),
        "reprojection_median_mm": (
            None if reprojection.get("median_m") is None
            else 1000.0 * float(reprojection["median_m"])
        ),
        "reprojection_inlier_ratio": reprojection.get("inlier_ratio"),
    }


def _registration_metrics(report: dict) -> dict:
    quality = report.get("quality", {}) or {}
    return {
        "registration_status": report.get("status"),
        "tracklet_id": report.get("tracklet_id"),
        "tracklet_entry": bool(report.get("tracklet_entry", False)),
        "registration_fitness_8mm": quality.get("fitness_8mm"),
        "registration_median_nn_mm": (
            None if quality.get("median_nn_m") is None
            else 1000.0 * float(quality["median_nn_m"])
        ),
    }


def _fmt(value: object, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.{digits}f}"
    return str(value)


def _compose_frame(
    *,
    timestamp: str,
    part: str,
    primary_view: str,
    rgb: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    clouds: FrameClouds,
    mesh_world: np.ndarray,
    pose_valid: bool,
    metrics: dict,
    fixed_world_bounds: ProjectionBounds,
    follow_span_m: float,
    follow_center: np.ndarray,
    views: list[str],
) -> np.ndarray:
    width, height = 1920, 1080
    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    _draw_text(
        canvas,
        f"Point-cloud timeline | {part} | frame {timestamp} | primary {primary_view}",
        (24, 39),
        scale=0.82,
        thickness=2,
    )
    state = metrics.get("state", "unknown")
    pose_label = "VALID" if pose_valid else "INVALID / hidden in final render"
    pose_color = (93, 225, 132) if pose_valid else (255, 86, 86)
    _draw_text(canvas, f"state {state} | pose {pose_label}", (24, 65), scale=0.56, color=pose_color, thickness=2)

    panel_w, panel_h = 930, 425
    top_y, bottom_y = 78, 523
    candidate_colors = [VIEW_COLORS_RGB[index % len(VIEW_COLORS_RGB)] for index in range(len(views))]
    candidate_panel = render_camera_panel(
        rgb,
        mask,
        clouds.candidates,
        candidate_colors,
        intrinsic,
        extrinsic,
        title=f"A  Prepared per-view candidates | {sum(map(len, clouds.candidates)):,} points",
        size=(panel_w, panel_h),
    )
    fused_panel = render_camera_panel(
        rgb,
        mask,
        [clouds.fused],
        [(255, 214, 64)],
        intrinsic,
        extrinsic,
        title=f"B  Final fused input + trajectory mesh | {len(clouds.fused):,} points",
        size=(panel_w, panel_h),
        mesh_points=mesh_world,
        pose_valid=pose_valid,
    )
    support_panel = render_orthographic_panel(
        clouds.support,
        candidate_colors,
        fixed_world_bounds,
        size=(panel_w, panel_h),
        title=(
            "C  Cross-view-supported subset + mesh | FIXED world scale | "
            f"{sum(map(len, clouds.support)):,} points"
        ),
        mesh_points=mesh_world,
        pose_valid=pose_valid,
    )
    half_w = panel_w // 2
    top_bounds = ProjectionBounds(top_basis(), np.zeros(2), follow_span_m)
    front_bounds = ProjectionBounds(front_basis(), np.zeros(2), follow_span_m)
    follow_top = render_orthographic_panel(
        [clouds.fused],
        [(255, 214, 64)],
        top_bounds,
        size=(half_w, panel_h),
        title="D1  Top | fixed metric scale",
        mesh_points=mesh_world,
        pose_valid=pose_valid,
        center_world=follow_center,
    )
    follow_front = render_orthographic_panel(
        [clouds.fused],
        [(255, 214, 64)],
        front_bounds,
        size=(panel_w - half_w, panel_h),
        title="D2  Front | same scale",
        mesh_points=mesh_world,
        pose_valid=pose_valid,
        center_world=follow_center,
    )
    canvas[top_y:top_y + panel_h, 20:20 + panel_w] = candidate_panel
    canvas[top_y:top_y + panel_h, 970:970 + panel_w] = fused_panel
    canvas[bottom_y:bottom_y + panel_h, 20:20 + panel_w] = support_panel
    canvas[bottom_y:bottom_y + panel_h, 970:970 + half_w] = follow_top
    canvas[bottom_y:bottom_y + panel_h, 970 + half_w:1900] = follow_front

    metric_y = 976
    quality_passed = metrics.get("quality_passed")
    quality_label = (
        "PASS" if quality_passed is True
        else "FAIL" if quality_passed is False
        else "n/a (legacy cloud)"
    )
    first = (
        f"candidate {metrics.get('candidate_points', 0):,} | "
        f"supported {metrics.get('supported_points', 0):,} "
        f"({_fmt(100.0 * metrics.get('support_fraction', 0.0))}%) | "
        f"retained {metrics.get('retained_points_before_voxel', 0):,} | "
        f"reliable views {_fmt(metrics.get('observing_views'), 0)}/8 | "
        f"gate {quality_label}"
    )
    second = (
        f"cross-view median {_fmt(metrics.get('cross_view_median_mm'), 2)} mm | "
        f"reprojection {_fmt(metrics.get('reprojection_median_mm'), 2)} mm | "
        f"view-centroid spread {_fmt(metrics.get('view_centroid_spread_mm'), 1)} mm | "
        f"cloud-mesh median {_fmt(metrics.get('cloud_mesh_median_mm'), 1)} mm | "
        f"ICP fitness {_fmt(metrics.get('registration_fitness_8mm'), 3)}"
    )
    _draw_text(canvas, first, (24, metric_y), scale=0.55, color=TEXT_COLOR, thickness=2)
    _draw_text(canvas, second, (24, metric_y + 27), scale=0.52, color=MUTED_TEXT)
    legend_y = metric_y + 58
    x = 24
    for index, view in enumerate(views):
        color = candidate_colors[index]
        cv2.rectangle(canvas, (x, legend_y - 13), (x + 14, legend_y + 1), color, -1)
        _draw_text(canvas, view, (x + 19, legend_y), scale=0.40, color=MUTED_TEXT)
        x += 112
    _draw_text(canvas, "mask", (x + 4, legend_y), scale=0.40, color=(80, 255, 120))
    _draw_text(canvas, "fused", (x + 70, legend_y), scale=0.40, color=(255, 214, 64))
    _draw_text(
        canvas,
        "mesh white=valid red=invalid",
        (x + 146, legend_y),
        scale=0.40,
        color=MUTED_TEXT,
    )
    return canvas


def _aggregate(rows: list[dict]) -> dict:
    result = {"frames": len(rows), "pose_valid_frames": int(sum(bool(row.get("pose_valid")) for row in rows))}
    fields = [
        "support_fraction",
        "supported_views",
        "cross_view_median_mm",
        "reprojection_median_mm",
        "view_centroid_spread_mm",
        "cloud_mesh_median_mm",
        "cloud_mesh_p90_mm",
        "cloud_center_step_mm",
        "translation_step_mm",
        "rotation_step_deg",
        "registration_fitness_8mm",
        "registration_median_nn_mm",
    ]
    for field in fields:
        values = np.asarray([
            float(row[field]) for row in rows
            if row.get(field) is not None and np.isfinite(float(row[field]))
        ])
        result[field] = None if not len(values) else {
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p90": float(np.quantile(values, 0.90)),
            "maximum": float(np.max(values)),
            "samples": int(len(values)),
        }
    return result


def _parse_segments(values: list[str], start: int, end: int) -> dict[str, tuple[int, int]]:
    if not values:
        return {"all": (start, end)}
    result = {}
    for value in values:
        if ":" not in value or "-" not in value:
            raise ValueError(f"segment must be NAME:START-END, got {value!r}")
        name, limits = value.split(":", 1)
        first, last = limits.split("-", 1)
        result[name] = (int(first), int(last))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parts", nargs="+")
    parser.add_argument("--primary-view")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-points-per-view", type=int, default=2500)
    parser.add_argument("--max-mesh-points", type=int, default=12000)
    parser.add_argument("--bounds-stride", type=int, default=5)
    parser.add_argument("--keyframes", nargs="*", type=int, default=[])
    parser.add_argument("--segment", action="append", default=[])
    args = parser.parse_args()

    config = load_json(args.pipeline.resolve())
    trajectory = load_json(args.trajectory.resolve())
    quality_root = (
        args.quality_root.resolve()
        if args.quality_root is not None
        else Path(config["point_cloud_root"]).resolve()
    )
    summary = load_json(quality_root / "quality_cloud_summary.json")
    registration_path = args.trajectory.resolve().parent / "pair_registrations.json"
    registration = load_json(registration_path) if registration_path.exists() else {}
    views = [str(value) for value in config["views"]]
    parts = [str(value) for value in (args.parts or config["parts"])]
    unknown = sorted(set(parts).difference(config["parts"]))
    if unknown:
        raise ValueError(f"unknown parts: {unknown}")
    primary_view = str(
        args.primary_view or config.get("render", {}).get("primary_view", views[0])
    )
    if primary_view not in views:
        raise ValueError(f"primary view {primary_view!r} is not in {views}")
    primary_index = views.index(primary_view)
    configured_frames = config.get("frames", {})
    start = int(configured_frames.get("start", 0) if args.start is None else args.start)
    end = int(configured_frames.get("end", start) if args.end is None else args.end)
    frame_numbers = list(range(start, end + 1))
    segments = _parse_segments(args.segment, start, end)
    parameters = summary.get("parameters", {})
    support_radius_m = float(parameters.get("support_radius_m", 0.01))
    min_support_default = int(parameters.get("min_support", 1))
    min_support_by_part = parameters.get("min_support_by_part", {})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keyframe_dir = args.output_dir / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": 1,
        "pipeline": str(args.pipeline.resolve()),
        "trajectory": str(args.trajectory.resolve()),
        "quality_root": str(quality_root),
        "primary_view": primary_view,
        "frame_range": [start, end],
        "stages": {
            "A": "per-view prepared candidates after mask/confidence/depth-edge filters",
            "B": "final fused PLY actually supplied to registration plus trajectory mesh",
            "C": "recomputed cross-view-supported subset before unsupported retention",
            "D": "final fused PLY plus mesh in object-following views with fixed metric scale",
        },
        "parts": {},
    }
    for part_index, part in enumerate(parts):
        mesh = trimesh.load(
            Path(config["mesh_dir"]) / f"{part}.glb", force="mesh"
        )
        raw_mesh_points = _mesh_vertices(
            mesh, int(args.max_mesh_points), 1907 + part_index
        )
        min_support = int(min_support_by_part.get(part, min_support_default))
        bounds_samples = []
        follow_radius_samples = []
        for frame in frame_numbers[::max(1, int(args.bounds_stride))]:
            timestamp = f"{frame:06d}"
            fused_path = quality_root / timestamp / f"{part}.ply"
            if fused_path.exists():
                fused = _sample(read_ply_xyz(fused_path), 1200, 12011 + frame)
                if len(fused):
                    bounds_samples.append(fused)
                    center = np.median(fused, axis=0)
                    follow_radius_samples.append(np.linalg.norm(fused - center, axis=1))
        fixed_world_bounds = projection_bounds(
            bounds_samples, isometric_basis(), padding=1.18, minimum_span_m=0.25
        )
        radii = np.concatenate(follow_radius_samples) if follow_radius_samples else np.asarray([0.15])
        follow_span_m = max(0.20, 2.30 * float(np.quantile(radii, 0.995)))
        output_path = args.output_dir / f"{part}_point_cloud_{primary_view}.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.fps),
            (1920, 1080),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {output_path}")
        part_rows: dict[str, dict] = {}
        previous_center = None
        try:
            for position, frame in enumerate(frame_numbers, start=1):
                timestamp = f"{frame:06d}"
                frame_clouds = _view_clouds(
                    quality_root,
                    timestamp,
                    part,
                    views,
                    support_radius_m=support_radius_m,
                    min_support=min_support,
                    maximum_per_view=int(args.max_points_per_view),
                )
                trajectory_row = trajectory["frames"][timestamp]["parts"][part]
                pose_valid = bool(trajectory_row.get("pose_valid", True))
                transform = np.asarray(
                    trajectory_row["S_world_from_raw_mesh"], dtype=np.float64
                )
                mesh_world = _transform_points(raw_mesh_points, transform)
                rgb = _load_rgb(config, timestamp, primary_view)
                mask = _load_mask(config, timestamp, primary_view, part)
                recon = load_recon(config, timestamp, backend=config["recon_backend"])
                intrinsic = scale_intrinsics(
                    recon["intrinsics"][primary_index],
                    recon["depth_hw"],
                    rgb.shape[:2],
                )
                extrinsic = recon["extrinsics"][primary_index]
                row = _quality_metrics(
                    summary.get("frames", {}).get(timestamp, {}).get(part, {})
                )
                row.update(_registration_metrics(
                    registration.get(part, {}).get(timestamp, {})
                ))
                center = (
                    np.median(frame_clouds.fused, axis=0)
                    if len(frame_clouds.fused)
                    else np.median(mesh_world, axis=0)
                )
                cloud_step = (
                    None if previous_center is None
                    else 1000.0 * float(np.linalg.norm(center - previous_center))
                )
                previous_center = center
                cloud_mesh_median, cloud_mesh_p90 = _mesh_cloud_distance_mm(
                    mesh_world, frame_clouds.fused
                )
                row.update({
                    "frame": frame,
                    "part": part,
                    "state": trajectory_row.get("state"),
                    "source": trajectory_row.get("source"),
                    "pose_valid": pose_valid,
                    "observing_views": int(trajectory_row.get("observing_views", 0)),
                    "visible_views": list(trajectory_row.get("visible_views", [])),
                    "final_fused_points": int(len(frame_clouds.fused)),
                    "recomputed_supported_points": int(sum(map(len, frame_clouds.support))),
                    "view_centroid_spread_mm": _view_centroid_spread_mm(frame_clouds.candidates),
                    "cloud_mesh_median_mm": cloud_mesh_median,
                    "cloud_mesh_p90_mm": cloud_mesh_p90,
                    "cloud_center_world": center.tolist(),
                    "cloud_center_step_mm": cloud_step,
                    "fused_pca_extents_mm": robust_pca_extents_mm(frame_clouds.fused),
                    "translation_step_mm": (
                        None if trajectory_row.get("translation_step_m") is None
                        else 1000.0 * float(trajectory_row["translation_step_m"])
                    ),
                    "rotation_step_deg": trajectory_row.get("rotation_step_deg"),
                })
                part_rows[timestamp] = row
                video_frame = _compose_frame(
                    timestamp=timestamp,
                    part=part,
                    primary_view=primary_view,
                    rgb=rgb,
                    mask=mask,
                    intrinsic=intrinsic,
                    extrinsic=extrinsic,
                    clouds=frame_clouds,
                    mesh_world=mesh_world,
                    pose_valid=pose_valid,
                    metrics=row,
                    fixed_world_bounds=fixed_world_bounds,
                    follow_span_m=follow_span_m,
                    follow_center=center,
                    views=views,
                )
                writer.write(cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR))
                if frame in set(args.keyframes):
                    cv2.imwrite(
                        str(keyframe_dir / f"{part}_{timestamp}.jpg"),
                        cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR),
                    )
                if position % 10 == 0 or position == len(frame_numbers):
                    print(
                        f"{part} timeline {position}/{len(frame_numbers)} frame {timestamp}",
                        flush=True,
                    )
        finally:
            writer.release()
        rows = list(part_rows.values())
        report["parts"][part] = {
            "video": str(output_path.resolve()),
            "fixed_world_span_m": fixed_world_bounds.span_m,
            "follow_span_m": follow_span_m,
            "frames": part_rows,
            "summary": _aggregate(rows),
            "segments": {
                name: {
                    "frame_range": [first, last],
                    **_aggregate([
                        row for row in rows
                        if first <= int(row["frame"]) <= last
                    ]),
                }
                for name, (first, last) in segments.items()
            },
        }
        print(f"wrote {output_path}", flush=True)
    metrics_path = args.output_dir / "point_cloud_metrics.json"
    write_json(metrics_path, report)
    print(f"wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
