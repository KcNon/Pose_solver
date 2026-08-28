#!/usr/bin/env python3
"""Bounded analysis of labelled hand occlusion in multi-view masks."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import write_json


MAX_FRAMES = 2_000
MAX_VIEWS = 16


def mask_path(
    root: Path,
    layout: str,
    view: str,
    timestamp: str,
) -> Path:
    if layout == "view_first":
        return root / view / f"{timestamp}.png"
    return root / timestamp / f"{view}.png"


def boundary_touch_ratio(target: np.ndarray, occluder: np.ndarray) -> float:
    boundary = cv2.morphologyEx(
        target.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    count = int(boundary.sum())
    if count == 0:
        return 0.0
    nearby = cv2.dilate(
        occluder.astype(np.uint8),
        np.ones((5, 5), np.uint8),
        iterations=1,
    ).astype(bool)
    return float(np.logical_and(boundary, nearby).sum() / count)


def evenly_spaced(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[index] for index in sorted(set(indices.tolist()))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks-root", required=True, type=Path)
    parser.add_argument(
        "--layout", choices=("frame_first", "view_first"), required=True
    )
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--frame-start", required=True, type=int)
    parser.add_argument("--frame-end", required=True, type=int)
    parser.add_argument("--target-labels", nargs="+", required=True, type=int)
    parser.add_argument("--occluder-label", required=True, type=int)
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frames = list(range(args.frame_start, args.frame_end + 1))
    views = [str(view) for view in args.views]
    if not frames or len(frames) > MAX_FRAMES:
        raise ValueError(f"frame count must be in [1, {MAX_FRAMES}]")
    if not views or len(views) > MAX_VIEWS or len(set(views)) != len(views):
        raise ValueError(f"views must be unique with at most {MAX_VIEWS} entries")
    if not 1 <= args.sample_count <= len(frames):
        raise ValueError("sample-count must be within the requested frame count")

    rows = []
    observed_labels: set[int] = set()
    for index, frame in enumerate(frames):
        timestamp = f"{frame:06d}"
        hand_pixels = 0
        hand_views = 0
        target_stats = {
            str(label): {
                "pixels": 0,
                "visible_views": 0,
                "contact_views": 0,
                "maximum_boundary_touch_ratio": 0.0,
            }
            for label in args.target_labels
        }
        for view in views:
            path = mask_path(args.masks_root, args.layout, view, timestamp)
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                labels = np.asarray(image)
            if labels.ndim != 2:
                raise ValueError(f"expected indexed label mask: {path}")
            observed_labels.update(int(value) for value in np.unique(labels))
            hand = labels == int(args.occluder_label)
            current_hand_pixels = int(hand.sum())
            hand_pixels += current_hand_pixels
            hand_views += int(current_hand_pixels > 0)
            for label in args.target_labels:
                target = labels == int(label)
                pixels = int(target.sum())
                touch = boundary_touch_ratio(target, hand)
                stats = target_stats[str(label)]
                stats["pixels"] += pixels
                stats["visible_views"] += int(pixels > 0)
                stats["contact_views"] += int(touch > 0.0)
                stats["maximum_boundary_touch_ratio"] = max(
                    float(stats["maximum_boundary_touch_ratio"]), touch
                )
        maximum_touch = max(
            float(value["maximum_boundary_touch_ratio"])
            for value in target_stats.values()
        )
        rows.append({
            "frame": frame,
            "hand_pixels": hand_pixels,
            "hand_views": hand_views,
            "maximum_boundary_touch_ratio": maximum_touch,
            "targets": target_stats,
        })
        if (index + 1) % 50 == 0 or index + 1 == len(frames):
            print(f"masks {index + 1}/{len(frames)}", flush=True)

    clean = [row["frame"] for row in rows if row["hand_views"] == 0]
    touching = [
        row for row in rows if row["maximum_boundary_touch_ratio"] > 0.0
    ]
    touching.sort(
        key=lambda row: (
            float(row["maximum_boundary_touch_ratio"]),
            int(row["hand_views"]),
            int(row["hand_pixels"]),
        ),
        reverse=True,
    )
    sample_count = int(args.sample_count)
    clean_count = min(len(clean), max(4, sample_count // 4))
    occluded_count = min(len(touching), sample_count - clean_count)
    selected = evenly_spaced(clean, clean_count)
    selected.extend(int(row["frame"]) for row in touching[:occluded_count])

    hand_frames = [row["frame"] for row in rows if row["hand_views"] > 0]
    reentry = []
    for previous, current in zip(rows[:-1], rows[1:]):
        if previous["hand_views"] > 0 and current["hand_views"] == 0:
            reentry.append(int(current["frame"]))
    selected.extend(reentry)
    selected = sorted(set(selected))[:sample_count]

    write_json(args.output, {
        "masks_root": str(args.masks_root.resolve()),
        "layout": args.layout,
        "frame_range": [args.frame_start, args.frame_end],
        "views": views,
        "target_labels": args.target_labels,
        "occluder_label": args.occluder_label,
        "observed_labels": sorted(observed_labels),
        "summary": {
            "frames": len(rows),
            "clean_frames": len(clean),
            "hand_frames": len(hand_frames),
            "target_touch_frames": len(touching),
            "first_hand_frame": min(hand_frames) if hand_frames else None,
            "last_hand_frame": max(hand_frames) if hand_frames else None,
            "reentry_frames": reentry,
            "recommended_frames": selected,
        },
        "frames": {f"{row['frame']:06d}": row for row in rows},
    })
    print(f"report -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
