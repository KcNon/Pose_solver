#!/usr/bin/env python
"""Fuse temporal and body branches for all normalized camera views."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.mask_io import (
    DEFAULT_PARTS,
    PART_COLORS,
    VIEW_NAMES,
    masks_to_label_map,
    save_palette_png,
)


def timestamps(mask_root: Path) -> list[str]:
    return sorted(p.name for p in mask_root.iterdir() if p.is_dir() and p.name.isdigit())


def read_label(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    label = np.asarray(Image.open(path))
    if label.ndim != 2:
        raise ValueError(f"expected indexed label image: {path}")
    return label


def blend(image: np.ndarray, mask: np.ndarray, rgb: list[int], alpha: float = 0.48) -> None:
    if not mask.any():
        return
    color = np.asarray(rgb[::-1], dtype=np.float32)
    image[mask] = (
        image[mask].astype(np.float32) * (1.0 - alpha) + color * alpha
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(
        image, contours, -1, tuple(int(v) for v in color), 2, cv2.LINE_AA
    )


def write_review(
    frames_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    frame_ids: list[str],
    view: str,
    fps: float,
) -> None:
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    video_path = preview_dir / f"{view}_overlay.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to create {video_path}")

    cells: list[np.ndarray] = []
    for index, timestamp in enumerate(frame_ids):
        image_path = frames_dir / view / f"{timestamp}.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read {image_path}")
        label = read_label(mask_dir / timestamp / f"{view}.png")
        overlay = image.copy()
        for part, part_id in (("body", 2), ("inner_pot", 3), ("lid", 1)):
            blend(overlay, label == part_id, PART_COLORS[part])
        cv2.putText(
            overlay,
            f"{view}  {timestamp}  {index + 1}/{len(frame_ids)}",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        writer.write(cv2.resize(overlay, (1280, 720), interpolation=cv2.INTER_AREA))
        cell = cv2.resize(overlay, (240, 135), interpolation=cv2.INTER_AREA)
        cv2.putText(
            cell, str(index), (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, (0, 255, 255), 1, cv2.LINE_AA,
        )
        cells.append(cell)
    writer.release()

    columns = 10
    blank = np.zeros_like(cells[0])
    cells.extend([blank] * ((columns - len(cells) % columns) % columns))
    sheet = np.vstack([
        np.hstack(cells[i:i + columns]) for i in range(0, len(cells), columns)
    ])
    cv2.imwrite(
        str(preview_dir / f"{view}_contact_sheet.jpg"),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )


def positive_stats(values: list[int]) -> dict[str, int]:
    array = np.asarray(values, dtype=np.int64)
    positive = array[array > 0]
    return {
        "nonempty_frames": int((array > 0).sum()),
        "empty_frames": int((array == 0).sum()),
        "min_positive_pixels": int(positive.min()) if len(positive) else 0,
        "median_positive_pixels": int(np.median(positive)) if len(positive) else 0,
        "max_pixels": int(array.max()) if len(array) else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-dir", required=True)
    parser.add_argument("--body-dir", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", nargs="+", choices=VIEW_NAMES, default=VIEW_NAMES)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--skip-review", action="store_true")
    args = parser.parse_args()

    temporal_dir = Path(args.temporal_dir)
    body_dir = Path(args.body_dir)
    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = timestamps(temporal_dir)
    if frame_ids != timestamps(body_dir):
        raise RuntimeError("temporal and body branches have different timestamp directories")
    if not frame_ids:
        raise RuntimeError("no temporal masks found")

    all_stats: dict[str, dict[str, dict[str, int]]] = {}
    for view in args.views:
        areas = {part: [] for part in DEFAULT_PARTS}
        for timestamp in frame_ids:
            temporal_path = temporal_dir / timestamp / f"{view}.png"
            body_path = body_dir / timestamp / f"{view}.png"
            temporal = read_label(temporal_path)
            body_label = read_label(body_path)
            if temporal.shape != body_label.shape:
                raise ValueError(f"shape mismatch for {view} at {timestamp}")

            lid = temporal == 1
            inner_pot = (temporal == 3) & ~lid
            body = (body_label == 2) & ~inner_pot & ~lid
            masks = {"lid": lid, "body": body, "inner_pot": inner_pot}
            save_palette_png(
                masks_to_label_map(masks, DEFAULT_PARTS),
                str(mask_dir / timestamp / f"{view}.png"),
                DEFAULT_PARTS,
            )
            for part, mask in masks.items():
                areas[part].append(int(mask.sum()))
        all_stats[view] = {part: positive_stats(values) for part, values in areas.items()}
        print(f"{view}: fused {len(frame_ids)} frames", flush=True)

    bbox_source = temporal_dir / "bbox.json"
    if bbox_source.exists():
        with open(bbox_source, encoding="utf-8") as file:
            bbox = json.load(file)
        bbox["frames"] = {
            timestamp: {view: view_data for view, view_data in frame.items() if view in args.views}
            for timestamp, frame in bbox.get("frames", {}).items()
        }
        with open(mask_dir / "bbox.json", "w", encoding="utf-8") as file:
            json.dump(bbox, file, ensure_ascii=False, indent=2)

    summary = {
        "method": "qwen_first_frame_bbox_plus_sam3_temporal_hybrid",
        "views": args.views,
        "qwen_bbox_frames": [frame_ids[0]],
        "frames_per_view": len(frame_ids),
        "total_mask_pngs": len(frame_ids) * len(args.views),
        "sources": {"lid_and_inner_pot": str(temporal_dir), "body": str(body_dir)},
        "visibility_order_front_to_back": ["lid", "inner_pot", "body"],
        "stats": all_stats,
    }
    with open(output_dir / "temporal_masks.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    if not args.skip_review:
        for view in args.views:
            write_review(frames_dir, mask_dir, output_dir, frame_ids, view, args.fps)
            print(f"{view}: review artifacts written", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"done -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
