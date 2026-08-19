#!/usr/bin/env python
"""Visualize raw mask backprojections and quality-filtered point clouds."""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.backproject_utils import load_palette_masks, load_recon_colors
from common.cloud_io import read_ply_xyz, write_ply
from common.io_utils import load_json, write_json
from common.mask_io import frame_path
from common.normalized_recon import load_recon
from common.quality_cloud import prepare_view_clouds


VIEW_COLORS_RGB = np.asarray(
    [
        [230, 57, 70],
        [255, 170, 35],
        [72, 190, 105],
        [45, 165, 225],
        [105, 95, 220],
        [220, 80, 190],
        [75, 210, 205],
        [210, 210, 75],
    ],
    dtype=np.uint8,
)
PART_COLORS_RGB = np.asarray(
    [[255, 145, 35], [70, 220, 120], [100, 160, 255]], dtype=np.uint8
)


def _cloud_stats(clouds: list[np.ndarray], views: list[str]) -> dict[str, Any]:
    counts = {view: int(len(points)) for view, points in zip(views, clouds)}
    centers = {
        view: np.median(points, axis=0).tolist()
        for view, points in zip(views, clouds)
        if len(points)
    }
    distances = []
    for first, second in combinations(centers, 2):
        distances.append({
            "views": [first, second],
            "distance_m": float(
                np.linalg.norm(
                    np.asarray(centers[first]) - np.asarray(centers[second])
                )
            ),
        })
    all_points = [points for points in clouds if len(points)]
    fused = (
        np.concatenate(all_points, axis=0)
        if all_points
        else np.empty((0, 3), dtype=np.float64)
    )
    return {
        "point_count": int(len(fused)),
        "per_view_points": counts,
        "centroids_world": centers,
        "pairwise_centroid_distances": distances,
        "centroid_distance_median_m": (
            float(np.median([row["distance_m"] for row in distances]))
            if distances
            else None
        ),
        "centroid_distance_max_m": (
            max(row["distance_m"] for row in distances)
            if distances
            else None
        ),
        "full_extent_m": (
            (np.max(fused, axis=0) - np.min(fused, axis=0)).tolist()
            if len(fused)
            else None
        ),
        "robust_extent_m": (
            (
                np.quantile(fused, 0.99, axis=0)
                - np.quantile(fused, 0.01, axis=0)
            ).tolist()
            if len(fused)
            else None
        ),
    }


def _colored_cloud(
    clouds: list[np.ndarray],
    colors_rgb: np.ndarray = VIEW_COLORS_RGB,
) -> tuple[np.ndarray, np.ndarray]:
    points = []
    colors = []
    for index, cloud in enumerate(clouds):
        if not len(cloud):
            continue
        points.append(np.asarray(cloud, dtype=np.float64))
        colors.append(np.repeat(
            colors_rgb[index % len(colors_rgb)][None, :],
            len(cloud),
            axis=0,
        ))
    if not points:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint8),
        )
    return np.concatenate(points), np.concatenate(colors)


def _projection_bases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    return [
        (
            "isometric",
            np.asarray([1.0, -1.0, 0.0]) / np.sqrt(2.0),
            np.asarray([-1.0, -1.0, 2.0]) / np.sqrt(6.0),
        ),
        ("world XY", np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0])),
        ("world XZ", np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 1.0])),
    ]


def _projection_limits(
    clouds: list[np.ndarray],
    u_axis: np.ndarray,
    v_axis: np.ndarray,
) -> tuple[float, float, float, float]:
    populated = [cloud for cloud in clouds if len(cloud)]
    if not populated:
        return -1.0, 1.0, -1.0, 1.0
    points = np.concatenate(populated)
    u = points @ u_axis
    v = points @ v_axis
    low_u, high_u = np.quantile(u, [0.002, 0.998])
    low_v, high_v = np.quantile(v, [0.002, 0.998])
    span = max(float(high_u - low_u), float(high_v - low_v), 1e-4)
    center_u = 0.5 * float(high_u + low_u)
    center_v = 0.5 * float(high_v + low_v)
    half = 0.58 * span
    return center_u - half, center_u + half, center_v - half, center_v + half


def _render_projection(
    clouds: list[np.ndarray],
    labels: list[str],
    *,
    title: str,
    axis_title: str,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    limits: tuple[float, float, float, float],
    colors_rgb: np.ndarray = VIEW_COLORS_RGB,
    width: int = 640,
    height: int = 500,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    left, right, top, bottom = 34, width - 24, 58, height - 82
    low_u, high_u, low_v, high_v = limits
    visible_counts = []
    for index, points in enumerate(clouds):
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(points):
            u = np.asarray(points) @ u_axis
            v = np.asarray(points) @ v_axis
            x = np.rint(
                left + (u - low_u) / max(high_u - low_u, 1e-9) * (right - left)
            ).astype(np.int32)
            y = np.rint(
                bottom - (v - low_v) / max(high_v - low_v, 1e-9) * (bottom - top)
            ).astype(np.int32)
            valid = (x >= left) & (x <= right) & (y >= top) & (y <= bottom)
            mask[y[valid], x[valid]] = 255
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
            color = colors_rgb[index % len(colors_rgb)][::-1]
            canvas[mask > 0] = color
            visible_counts.append(int(valid.sum()))
        else:
            visible_counts.append(0)
    cv2.rectangle(canvas, (left, top), (right, bottom), (85, 85, 85), 1)
    cv2.putText(
        canvas,
        title,
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        axis_title,
        (18, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )
    for index, (label, count) in enumerate(zip(labels, visible_counts)):
        column = index % 3
        row = index // 3
        x = 18 + column * 205
        y = height - 52 + row * 24
        color = tuple(
            int(value)
            for value in colors_rgb[index % len(colors_rgb)][::-1]
        )
        cv2.circle(canvas, (x + 5, y - 5), 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{label} {count}",
            (x + 16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _render_cloud_review(
    raw_clouds: list[np.ndarray],
    quality_clouds: list[np.ndarray],
    labels: list[str],
    *,
    frame: int,
    part: str,
    colors_rgb: np.ndarray = VIEW_COLORS_RGB,
) -> np.ndarray:
    panels = []
    combined = raw_clouds + quality_clouds
    for row_name, clouds in (
        ("RAW mask backprojection", raw_clouds),
        ("QUALITY selected views", quality_clouds),
    ):
        row = []
        for axis_title, u_axis, v_axis in _projection_bases():
            row.append(_render_projection(
                clouds,
                labels,
                title=f"{frame:06d} {part} | {row_name}",
                axis_title=axis_title,
                u_axis=u_axis,
                v_axis=v_axis,
                limits=_projection_limits(combined, u_axis, v_axis),
                colors_rgb=colors_rgb,
            ))
        panels.append(np.concatenate(row, axis=1))
    return np.concatenate(panels, axis=0)


def _render_mask_mosaic(
    config: dict[str, Any],
    labels_by_view: dict[str, np.ndarray],
    views: list[str],
    frame: int,
) -> np.ndarray:
    tile_width, tile_height = 640, 360
    timestamp = f"{frame:06d}"
    tiles = []
    for view in views:
        path = frame_path(
            config["frames_dir"],
            config.get("frames_layout", "normalized"),
            timestamp,
            view,
        )
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (tile_width, tile_height))
        labels = cv2.resize(
            labels_by_view[view],
            (tile_width, tile_height),
            interpolation=cv2.INTER_NEAREST,
        )
        counts = []
        for part_index, part in enumerate(config["parts"]):
            part_id = int(config["part_ids"][part])
            selected = labels == part_id
            color = PART_COLORS_RGB[part_index % len(PART_COLORS_RGB)][::-1]
            layer = np.zeros_like(image)
            layer[:] = color
            image[selected] = (
                0.55 * image[selected] + 0.45 * layer[selected]
            ).astype(np.uint8)
            contours, _ = cv2.findContours(
                selected.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(image, contours, -1, tuple(int(v) for v in color), 2)
            counts.append(f"{part}:{int(selected.sum())}")
        cv2.putText(
            image,
            f"{timestamp} {view}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            " ".join(counts),
            (12, tile_height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(image)
    columns = 3
    rows = [
        np.concatenate(tiles[index:index + columns], axis=1)
        for index in range(0, len(tiles), columns)
    ]
    if len(rows[-1]) < columns * tile_width:
        padding = np.zeros(
            (tile_height, columns * tile_width - rows[-1].shape[1], 3),
            dtype=np.uint8,
        )
        rows[-1] = np.concatenate((rows[-1], padding), axis=1)
    return np.concatenate(rows, axis=0)


def _quality_clouds(
    quality_root: Path,
    timestamp: str,
    part: str,
    views: list[str],
) -> list[np.ndarray]:
    root = quality_root / timestamp / "views" / part
    return [
        read_ply_xyz(root / f"{view}.ply")
        if (root / f"{view}.ply").exists()
        else np.empty((0, 3), dtype=np.float64)
        for view in views
    ]


def _contact_sheet(paths: list[Path], output: Path) -> None:
    thumbnails = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        thumbnails.append(cv2.resize(image, (800, 420), interpolation=cv2.INTER_AREA))
    rows = [
        np.concatenate(thumbnails[index:index + 2], axis=1)
        for index in range(0, len(thumbnails), 2)
    ]
    if rows[-1].shape[1] < 1600:
        rows[-1] = np.concatenate((
            rows[-1],
            np.zeros((420, 1600 - rows[-1].shape[1], 3), dtype=np.uint8),
        ), axis=1)
    cv2.imwrite(str(output), np.concatenate(rows, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--parts", nargs="+")
    parser.add_argument("--quality-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conf-quantile", type=float)
    parser.add_argument("--stride", type=int)
    args = parser.parse_args()

    config = load_json(args.config.resolve())
    parts = list(args.parts or config["parts"])
    unknown = sorted(set(parts).difference(config["parts"]))
    if unknown:
        raise ValueError(f"unknown parts: {unknown}")
    views = [str(value) for value in config["views"]]
    quality_settings = config.get("quality_cloud", {})
    conf_quantile = float(
        quality_settings.get("conf_quantile", 0.5)
        if args.conf_quantile is None
        else args.conf_quantile
    )
    stride = int(
        quality_settings.get("stride", 2) if args.stride is None else args.stride
    )
    quality_root = Path(
        args.quality_root or config["point_cloud_root"]
    ).resolve()
    summary_path = quality_root / "quality_cloud_summary.json"
    quality_summary = load_json(summary_path) if summary_path.exists() else None
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "direct_mask_depth_backprojection_vs_quality_cloud",
        "config": str(args.config.resolve()),
        "quality_root": str(quality_root),
        "conf_quantile": conf_quantile,
        "stride": stride,
        "views": views,
        "frames": {},
    }
    review_paths = []
    all_parts_review_paths = []
    for frame in args.frames:
        timestamp = f"{frame:06d}"
        recon = load_recon(config, timestamp, backend=config["recon_backend"])
        colors = load_recon_colors(recon, config, timestamp)
        masks = load_palette_masks(
            config["masks_dir"],
            timestamp,
            config["parts"],
            recon["depth_hw"],
            views=views,
            part_ids=config.get("part_ids"),
        )
        labels_by_view = {
            view: np.asarray(
                Image.open(Path(config["masks_dir"]) / timestamp / f"{view}.png")
            )
            for view in views
        }
        mask_path = args.output_root / f"{timestamp}_masks.jpg"
        cv2.imwrite(
            str(mask_path),
            _render_mask_mosaic(config, labels_by_view, views, frame),
        )
        frame_report = {"mask_mosaic": str(mask_path.resolve()), "parts": {}}
        raw_part_clouds = []
        quality_fused_part_clouds = []
        for part in parts:
            raw = prepare_view_clouds(
                np.asarray(recon["depth"], dtype=np.float32),
                colors,
                recon["intrinsics"],
                recon["extrinsics"],
                recon["conf"],
                masks[part],
                conf_quantile=conf_quantile,
                mask_erode=0,
                depth_edge_m=0.0,
                stride=stride,
            )
            raw_clouds = [cloud.points for cloud in raw]
            raw_part_clouds.append(
                np.concatenate([cloud for cloud in raw_clouds if len(cloud)])
                if any(len(cloud) for cloud in raw_clouds)
                else np.empty((0, 3), dtype=np.float64)
            )
            selected_clouds = _quality_clouds(
                quality_root, timestamp, part, views
            )
            raw_points, raw_colors = _colored_cloud(raw_clouds)
            quality_points, quality_colors = _colored_cloud(selected_clouds)
            raw_path = args.output_root / f"{timestamp}_{part}_raw_by_view.ply"
            quality_path = (
                args.output_root / f"{timestamp}_{part}_quality_by_view.ply"
            )
            write_ply(raw_path, raw_points, raw_colors)
            write_ply(quality_path, quality_points, quality_colors)
            review = _render_cloud_review(
                raw_clouds,
                selected_clouds,
                views,
                frame=frame,
                part=part,
            )
            review_path = args.output_root / f"{timestamp}_{part}.jpg"
            cv2.imwrite(str(review_path), review)
            review_paths.append(review_path)
            existing_fused = quality_root / timestamp / f"{part}.ply"
            quality_fused_part_clouds.append(
                read_ply_xyz(existing_fused)
                if existing_fused.exists()
                else np.empty((0, 3), dtype=np.float64)
            )
            part_report = {
                "raw": _cloud_stats(raw_clouds, views),
                "quality_selected": _cloud_stats(selected_clouds, views),
                "raw_view_colored_ply": str(raw_path.resolve()),
                "quality_view_colored_ply": str(quality_path.resolve()),
                "quality_fused_ply": (
                    str(existing_fused.resolve()) if existing_fused.exists() else None
                ),
                "review_image": str(review_path.resolve()),
            }
            if quality_summary is not None:
                part_report["quality_report"] = quality_summary.get(
                    "frames", {}
                ).get(timestamp, {}).get(part)
            frame_report["parts"][part] = part_report
            print(
                f"{timestamp} {part}: raw={len(raw_points)} "
                f"quality_views={len(quality_points)}",
                flush=True,
            )
        raw_all_points, raw_all_colors = _colored_cloud(
            raw_part_clouds, PART_COLORS_RGB
        )
        quality_all_points, quality_all_colors = _colored_cloud(
            quality_fused_part_clouds, PART_COLORS_RGB
        )
        raw_all_path = args.output_root / f"{timestamp}_all_parts_raw.ply"
        quality_all_path = args.output_root / f"{timestamp}_all_parts_quality.ply"
        write_ply(raw_all_path, raw_all_points, raw_all_colors)
        write_ply(quality_all_path, quality_all_points, quality_all_colors)
        all_parts_image = _render_cloud_review(
            raw_part_clouds,
            quality_fused_part_clouds,
            parts,
            frame=frame,
            part="all parts",
            colors_rgb=PART_COLORS_RGB,
        )
        all_parts_path = args.output_root / f"{timestamp}_all_parts.jpg"
        cv2.imwrite(str(all_parts_path), all_parts_image)
        all_parts_review_paths.append(all_parts_path)
        frame_report["all_parts"] = {
            "raw_part_colored_ply": str(raw_all_path.resolve()),
            "quality_part_colored_ply": str(quality_all_path.resolve()),
            "review_image": str(all_parts_path.resolve()),
        }
        report["frames"][timestamp] = frame_report
    contact_sheet = args.output_root / "point_cloud_contact_sheet.jpg"
    _contact_sheet(review_paths, contact_sheet)
    report["contact_sheet"] = str(contact_sheet.resolve())
    all_parts_contact_sheet = args.output_root / "all_parts_contact_sheet.jpg"
    _contact_sheet(all_parts_review_paths, all_parts_contact_sheet)
    report["all_parts_contact_sheet"] = str(all_parts_contact_sheet.resolve())
    report_path = args.output_root / "mask_backprojection_review.json"
    write_json(report_path, report)
    print(f"report -> {report_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
