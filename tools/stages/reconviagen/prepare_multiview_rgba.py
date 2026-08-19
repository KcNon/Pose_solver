#!/usr/bin/env python
"""Prepare synchronized, mask-cropped RGBA inputs for part reconstruction.

This stage is intended for assembly recordings that do not have a dedicated
scan video for every part.  It selects one clean synchronized timestamp from
the user-supplied appearance interval, rejects weak camera views, and writes
one RGBA crop per retained view.  Selection is deterministic and recorded in
``input_manifest.json`` so no manual keyframe or camera exclusion is hidden in
the reconstruction step.
"""
from __future__ import annotations

import argparse
import json
from math import log1p
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mask_path(root: Path, timestamp: str, view: str) -> Path:
    timestamp_first = root / timestamp / f"{view}.png"
    if timestamp_first.exists():
        return timestamp_first
    return root / view / f"{timestamp}.png"


def _frame_path(root: Path, timestamp: str, view: str) -> Path:
    view_first = root / view / f"{timestamp}.jpg"
    if view_first.exists():
        return view_first
    timestamp_first = root / timestamp / f"{view}.jpg"
    if timestamp_first.exists():
        return timestamp_first
    png = root / view / f"{timestamp}.png"
    return png


def _quality(mask: np.ndarray, *, min_pixels: int) -> dict[str, float] | None:
    ys, xs = np.nonzero(mask)
    pixels = int(len(xs))
    if pixels < min_pixels:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    component_fraction = float(largest / max(pixels, 1))
    height, width = mask.shape
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    border_margin = float(min(x1, y1, width - x2, height - y2))
    bbox_area = max((x2 - x1) * (y2 - y1), 1)
    fill_fraction = float(pixels / bbox_area)
    score = (
        log1p(pixels)
        + 2.0 * component_fraction
        + 0.03 * min(border_margin, 30.0)
        + 0.25 * min(fill_fraction, 1.0)
    )
    return {
        "pixels": float(pixels),
        "component_fraction": component_fraction,
        "border_margin_pixels": border_margin,
        "bbox_fill_fraction": fill_fraction,
        "score": float(score),
    }


def _crop_rgba(rgb: np.ndarray, mask: np.ndarray, padding: float) -> Image.Image:
    ys, xs = np.nonzero(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    side = max(32, int(np.ceil(max(x2 - x1, y2 - y1) * (1 + 2 * padding))))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left, top = int(np.floor(cx - side / 2)), int(np.floor(cy - side / 2))
    rgba = np.zeros((side, side, 4), dtype=np.uint8)
    sx1, sy1 = max(left, 0), max(top, 0)
    sx2, sy2 = min(left + side, rgb.shape[1]), min(top + side, rgb.shape[0])
    dx1, dy1 = sx1 - left, sy1 - top
    dx2, dy2 = dx1 + sx2 - sx1, dy1 + sy2 - sy1
    rgba[dy1:dy2, dx1:dx2, :3] = rgb[sy1:sy2, sx1:sx2]
    rgba[dy1:dy2, dx1:dx2, 3] = mask[sy1:sy2, sx1:sx2] * 255
    return Image.fromarray(rgba, "RGBA")


def _contact_sheet(paths: list[Path], destination: Path) -> None:
    tile_size = 300
    columns = 4
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + 28)), (20, 20, 20))
    for index, path in enumerate(paths):
        rgba = Image.open(path).convert("RGBA")
        checker = Image.new("RGBA", rgba.size, (215, 215, 215, 255))
        tile = Image.alpha_composite(checker, rgba).convert("RGB")
        tile.thumbnail((tile_size, tile_size))
        x = (index % columns) * tile_size
        y = (index // columns) * (tile_size + 28)
        sheet.paste(tile, (x + (tile_size - tile.width) // 2, y))
        ImageDraw.Draw(sheet).text((x + 6, y + tile_size + 5), path.stem, fill=(255, 255, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=95)


def _candidate_range(
    part: str,
    parts: dict[str, dict[str, Any]],
    *,
    minimum_delay: int,
    window: int,
    maximum_frame: int,
) -> range:
    start = int(parts[part]["start_frame"]) + minimum_delay
    later_starts = sorted(
        int(spec["start_frame"])
        for name, spec in parts.items()
        if name != part and int(spec["start_frame"]) > int(parts[part]["start_frame"])
    )
    end = min(start + window, maximum_frame)
    if later_starts:
        end = min(end, later_starts[0] - 1)
    if end < start:
        end = start
    return range(start, end + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--parts", nargs="+")
    parser.add_argument("--minimum-delay", type=int, default=5)
    parser.add_argument("--candidate-window", type=int, default=90)
    parser.add_argument("--minimum-views", type=int, default=4)
    parser.add_argument("--maximum-views", type=int, default=8)
    parser.add_argument("--minimum-mask-pixels", type=int, default=500)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument(
        "--frame-start",
        type=int,
        help="Optional shared first candidate frame for every requested part.",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        help="Optional shared last candidate frame for every requested part.",
    )
    args = parser.parse_args()

    if (args.frame_start is None) != (args.frame_end is None):
        parser.error("--frame-start and --frame-end must be provided together")
    if args.frame_start is not None and args.frame_end < args.frame_start:
        parser.error("--frame-end must be greater than or equal to --frame-start")

    mask_config_path = Path(args.mask_config).resolve()
    config = _load(mask_config_path)
    configured_parts = config["parts"]
    parts = args.parts or list(configured_parts)
    unknown = set(parts).difference(configured_parts)
    if unknown:
        raise ValueError(f"unknown parts: {sorted(unknown)}")
    views = list(config["views"])
    frames_root = Path(config["frames_dir"]).resolve()
    masks_root = Path(config["output_root"]).resolve() / "masks"
    output_root = Path(args.output_root).resolve()
    rgba_root = output_root / "rgba"
    mesh_root = output_root / "meshes"
    maximum_frame = max(
        int(path.stem)
        for path in (frames_root / views[0]).iterdir()
        if path.is_file() and path.stem.isdigit()
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "mask_config": str(mask_config_path),
        "selection": {
            "minimum_delay": args.minimum_delay,
            "candidate_window": args.candidate_window,
            "minimum_views": args.minimum_views,
            "maximum_views": args.maximum_views,
            "minimum_mask_pixels": args.minimum_mask_pixels,
            "padding": args.padding,
            "frame_start": args.frame_start,
            "frame_end": args.frame_end,
        },
        "parts": {},
    }
    for part in parts:
        label = int(configured_parts[part]["id"])
        candidates = []
        if args.frame_start is None:
            candidate_frames = _candidate_range(
                part,
                configured_parts,
                minimum_delay=args.minimum_delay,
                window=args.candidate_window,
                maximum_frame=maximum_frame,
            )
        else:
            candidate_frames = range(
                max(0, args.frame_start),
                min(maximum_frame, args.frame_end) + 1,
            )
        for frame in candidate_frames:
            timestamp = f"{frame:06d}"
            per_view = {}
            for view in views:
                path = _mask_path(masks_root, timestamp, view)
                if not path.exists():
                    continue
                labels = np.asarray(Image.open(path))
                metrics = _quality(labels == label, min_pixels=args.minimum_mask_pixels)
                if metrics is not None:
                    per_view[view] = metrics
            if len(per_view) < args.minimum_views:
                continue
            areas = np.asarray([item["pixels"] for item in per_view.values()])
            median_area = float(np.median(areas))
            retained = {
                view: metrics for view, metrics in per_view.items()
                if 0.35 * median_area <= metrics["pixels"] <= 2.85 * median_area
                and metrics["component_fraction"] >= 0.70
                and metrics["border_margin_pixels"] >= 2.0
            }
            if len(retained) < args.minimum_views:
                continue
            scores = np.asarray([item["score"] for item in retained.values()])
            candidates.append({
                "timestamp": timestamp,
                "views": retained,
                "view_count": len(retained),
                "score": float(np.median(scores) + 0.15 * len(retained)),
            })
        if not candidates:
            raise RuntimeError(f"{part}: no reconstruction frame passed automatic quality gates")
        selected = max(candidates, key=lambda item: (item["score"], item["view_count"]))
        timestamp = selected["timestamp"]
        ranked_views = sorted(
            selected["views"].items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )[: args.maximum_views]
        destination_dir = rgba_root / part
        destination_dir.mkdir(parents=True, exist_ok=True)
        for stale in destination_dir.glob("*.png"):
            stale.unlink()
        written = []
        records = []
        for view, metrics in ranked_views:
            frame_path = _frame_path(frames_root, timestamp, view)
            mask_path = _mask_path(masks_root, timestamp, view)
            rgb = np.asarray(Image.open(frame_path).convert("RGB"))
            labels = np.asarray(Image.open(mask_path))
            rgba = _crop_rgba(rgb, (labels == label).astype(np.uint8), args.padding)
            destination = destination_dir / f"{view}.png"
            rgba.save(destination)
            written.append(destination)
            records.append({
                "view": view,
                "frame": timestamp,
                "rgb": str(frame_path),
                "mask": str(mask_path),
                "rgba": str(destination),
                **metrics,
            })
        _contact_sheet(written, output_root / "reviews" / f"{part}.jpg")
        manifest["parts"][part] = {
            "label": label,
            "selected_frame": timestamp,
            "selection_score": selected["score"],
            "candidate_count": len(candidates),
            "automatically_excluded_views": sorted(
                set(selected["views"]).difference(view for view, _ in ranked_views)
            ),
            "views": records,
        }
        print(f"{part}: frame {timestamp}, {len(records)} views")

    mesh_root.mkdir(parents=True, exist_ok=True)
    runtime_config = {
        "output_root": str(output_root),
        "rgba_root": str(rgba_root),
        "mesh_root": str(mesh_root),
        "recon_python": "/data_ft_9_10/wentai/projects/ReconViaGen/.venv/bin/python",
        "parts": {part: {} for part in parts},
        "reconstruction": {
            "strategy": "adaptive_guidance_weight",
            "pipeline_type": "1024_cascade",
            "ss_source": "mesh",
            "seed": 0,
            "decimation_target": 500000,
            "texture_size": 2048,
        },
    }
    (output_root / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_root / "reconstruction.json").write_text(
        json.dumps(runtime_config, indent=2), encoding="utf-8"
    )
    print(f"manifest -> {output_root / 'input_manifest.json'}")
    print(f"reconstruction config -> {output_root / 'reconstruction.json'}")


if __name__ == "__main__":
    main()
