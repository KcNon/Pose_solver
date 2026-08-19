#!/usr/bin/env python
"""Render per-camera point clouds as a color-coded multi-panel PNG/GIF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.cloud_io import read_ply_xyz
from common.io_utils import write_json


VIEW_COLORS = [
    (66, 165, 245),
    (255, 167, 38),
    (102, 187, 106),
    (239, 83, 80),
    (171, 71, 188),
    (38, 198, 218),
    (255, 238, 88),
    (236, 64, 122),
]


def _font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _rotation_yaw_elevation(yaw_deg: float, elevation_deg: float) -> np.ndarray:
    yaw = np.radians(yaw_deg)
    elevation = np.radians(elevation_deg)
    camera = np.asarray([
        np.cos(elevation) * np.cos(yaw),
        np.cos(elevation) * np.sin(yaw),
        np.sin(elevation),
    ])
    camera /= np.linalg.norm(camera)
    up_hint = np.asarray([0.0, 0.0, 1.0])
    right = np.cross(up_hint, camera)
    if np.linalg.norm(right) < 1e-8:
        right = np.asarray([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(camera, right)
    up /= np.linalg.norm(up)
    return np.stack([right, up, camera])


def projection_bases() -> list[tuple[str, np.ndarray]]:
    return [
        ("Isometric", _rotation_yaw_elevation(-45.0, 28.0)),
        ("Top  (X / Y)", np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)),
        ("Front  (X / Z)", np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)),
        ("Side  (Y / Z)", np.asarray([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)),
    ]


def _sample_cloud(points: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    index = np.random.default_rng(seed).choice(len(points), maximum, replace=False)
    return points[index]


def render_projection(
    clouds: list[np.ndarray],
    colors: list[tuple[int, int, int]],
    basis: np.ndarray,
    *,
    size: tuple[int, int],
    point_radius: int = 1,
    title: str = "",
) -> Image.Image:
    width, height = size
    canvas = np.full((height, width, 3), (13, 18, 27), dtype=np.uint8)
    nonempty = [cloud for cloud in clouds if len(cloud)]
    if not nonempty:
        return Image.fromarray(canvas)
    all_points = np.concatenate(nonempty)
    center = np.median(all_points, axis=0)
    transformed = [(cloud - center) @ basis.T for cloud in clouds]
    projected_all = np.concatenate([value for value in transformed if len(value)])
    low = np.quantile(projected_all[:, :2], 0.005, axis=0)
    high = np.quantile(projected_all[:, :2], 0.995, axis=0)
    span = np.maximum(high - low, 1e-6)
    padding = 34
    scale = min((width - 2 * padding) / span[0], (height - 2 * padding) / span[1])
    midpoint = 0.5 * (low + high)
    depth_low, depth_high = np.quantile(projected_all[:, 2], [0.02, 0.98])
    depth_span = max(float(depth_high - depth_low), 1e-8)

    # A metric grid makes changes in density and object extent visible.
    grid_color = np.asarray((31, 42, 57), dtype=np.uint8)
    for fraction in np.linspace(0.1, 0.9, 9):
        x = int(round(fraction * (width - 1)))
        y = int(round(fraction * (height - 1)))
        canvas[:, x:x + 1] = grid_color
        canvas[y:y + 1, :] = grid_color

    entries = []
    for cloud_index, (value, color) in enumerate(zip(transformed, colors)):
        if not len(value):
            continue
        x = np.rint((value[:, 0] - midpoint[0]) * scale + width / 2).astype(int)
        y = np.rint(height / 2 - (value[:, 1] - midpoint[1]) * scale).astype(int)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not valid.any():
            continue
        depth = value[valid, 2]
        brightness = 0.62 + 0.38 * np.clip(
            (depth - depth_low) / depth_span, 0.0, 1.0
        )
        rgb = np.clip(
            np.asarray(color, float)[None] * brightness[:, None], 0, 255
        ).astype(np.uint8)
        entries.append((depth, x[valid], y[valid], rgb, cloud_index))
    if entries:
        depth = np.concatenate([entry[0] for entry in entries])
        x = np.concatenate([entry[1] for entry in entries])
        y = np.concatenate([entry[2] for entry in entries])
        rgb = np.concatenate([entry[3] for entry in entries])
        order = np.argsort(depth)
        x, y, rgb = x[order], y[order], rgb[order]
        for dy in range(-point_radius, point_radius + 1):
            for dx in range(-point_radius, point_radius + 1):
                if dx * dx + dy * dy > point_radius * point_radius + 1:
                    continue
                xx = np.clip(x + dx, 0, width - 1)
                yy = np.clip(y + dy, 0, height - 1)
                canvas[yy, xx] = rgb
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(73, 91, 112), width=2)
    draw.text((16, 12), title, font=_font(22, bold=True), fill=(235, 241, 248))
    extent_mm = np.ptp(all_points, axis=0) * 1000.0
    extent_text = "extent XYZ: " + " / ".join(f"{value:.1f}" for value in extent_mm) + " mm"
    draw.text((16, height - 30), extent_text, font=_font(15), fill=(178, 190, 205))
    return image


def load_clouds(view_dir: Path, max_points: int) -> tuple[list[str], list[np.ndarray]]:
    paths = sorted(view_dir.glob("*.ply"))
    names, clouds = [], []
    for index, path in enumerate(paths):
        points = read_ply_xyz(path)
        names.append(path.stem)
        clouds.append(_sample_cloud(points, max_points, 1701 + index))
    return names, clouds


def render_summary(
    names: list[str],
    clouds: list[np.ndarray],
    *,
    timestamp: str,
    part: str,
    fused_points: int | None,
    cross_view_mm: float | None,
    reprojection_mm: float | None,
) -> Image.Image:
    width, height = 1580, 1120
    output = Image.new("RGB", (width, height), (8, 12, 19))
    draw = ImageDraw.Draw(output)
    draw.text(
        (34, 24),
        f"Multi-view point cloud  |  frame {timestamp}  |  part {part}",
        font=_font(32, bold=True),
        fill=(245, 248, 252),
    )
    metric_text = [f"views {len(names)}", f"candidates {sum(len(cloud) for cloud in clouds):,}"]
    if fused_points is not None:
        metric_text.append(f"fused {fused_points:,}")
    if cross_view_mm is not None:
        metric_text.append(f"cross-view median {cross_view_mm:.2f} mm")
    if reprojection_mm is not None:
        metric_text.append(f"reprojection median {reprojection_mm:.2f} mm")
    draw.text((36, 70), "   |   ".join(metric_text), font=_font(18), fill=(177, 192, 210))

    panel_size = (750, 445)
    positions = [(25, 112), (805, 112), (25, 577), (805, 577)]
    colors = [VIEW_COLORS[index % len(VIEW_COLORS)] for index in range(len(names))]
    for (title, basis), position in zip(projection_bases(), positions):
        panel = render_projection(
            clouds,
            colors,
            basis,
            size=panel_size,
            point_radius=1,
            title=title,
        )
        output.paste(panel, position)

    draw = ImageDraw.Draw(output)
    legend_y = 1044
    cursor_x = 30
    for index, (name, cloud) in enumerate(zip(names, clouds)):
        color = colors[index]
        draw.rounded_rectangle(
            (cursor_x, legend_y, cursor_x + 18, legend_y + 18),
            radius=4,
            fill=color,
        )
        label = f"{name}  {len(cloud):,}"
        draw.text((cursor_x + 25, legend_y - 2), label, font=_font(15), fill=(223, 230, 239))
        cursor_x += 25 + int(draw.textlength(label, font=_font(15))) + 28
        if cursor_x > width - 210:
            legend_y += 30
            cursor_x = 30
    return output


def render_turntable(
    names: list[str],
    clouds: list[np.ndarray],
    output: Path,
    *,
    frames: int,
) -> None:
    colors = [VIEW_COLORS[index % len(VIEW_COLORS)] for index in range(len(names))]
    images = []
    for index in range(frames):
        yaw = -180.0 + 360.0 * index / frames
        images.append(render_projection(
            clouds,
            colors,
            _rotation_yaw_elevation(yaw, 25.0),
            size=(900, 680),
            point_radius=1,
            title=f"Turntable  {yaw:+.0f} deg",
        ))
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=90,
        loop=0,
        optimize=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--part", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--gif-frames", type=int, default=32)
    parser.add_argument("--max-points-per-view", type=int, default=25000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    root = args.quality_root.resolve()
    timestamp = f"{int(args.timestamp):06d}"
    view_dir = root / timestamp / "views" / args.part
    names, clouds = load_clouds(view_dir, int(args.max_points_per_view))
    if not names:
        raise FileNotFoundError(f"no per-view PLY files under {view_dir}")
    summary_path = root / "quality_cloud_summary.json"
    row = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = summary.get("frames", {}).get(timestamp, {}).get(args.part, {})
    cross = (row.get("cross_view") or {}).get("median_m")
    reprojection = (row.get("reprojection_depth") or {}).get("median_m")
    fused_path = root / timestamp / f"{args.part}.ply"
    fused_points = len(read_ply_xyz(fused_path)) if fused_path.exists() else None
    image = render_summary(
        names,
        clouds,
        timestamp=timestamp,
        part=args.part,
        fused_points=fused_points,
        cross_view_mm=None if cross is None else 1000.0 * float(cross),
        reprojection_mm=(
            None if reprojection is None else 1000.0 * float(reprojection)
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    if args.gif is not None:
        render_turntable(
            names, clouds, args.gif, frames=max(4, int(args.gif_frames))
        )
    report = {
        "quality_root": str(root),
        "timestamp": timestamp,
        "part": args.part,
        "views": {
            name: {"displayed_points": int(len(cloud))}
            for name, cloud in zip(names, clouds)
        },
        "fused_points": fused_points,
        "cross_view_median_mm": None if cross is None else 1000.0 * float(cross),
        "reprojection_median_mm": (
            None if reprojection is None else 1000.0 * float(reprojection)
        ),
        "png": str(args.output.resolve()),
        "gif": str(args.gif.resolve()) if args.gif is not None else None,
    }
    report_path = args.report or args.output.with_suffix(".json")
    write_json(report_path, report)
    print(f"wrote {args.output}")
    if args.gif is not None:
        print(f"wrote {args.gif}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
