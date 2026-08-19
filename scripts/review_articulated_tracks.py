#!/usr/bin/env python
"""Measure and visualize articulated part tracks against whole-object masks."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.masking.io import frame_path, load_label_mask, write_json
from common.masking.schema import load_mask_pipeline_config


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def _metrics(parts: dict[str, np.ndarray], whole: np.ndarray) -> dict:
    names = list(parts)
    union = np.logical_or.reduce(list(parts.values()))
    intersection = np.logical_and.reduce(list(parts.values()))
    union_whole = np.logical_or(union, whole)
    inside = np.logical_and(union, whole)
    result = {
        "whole_pixels": int(whole.sum()),
        "part_pixels": {name: int(parts[name].sum()) for name in names},
        "part_outside_whole_fraction": {
            name: _ratio(
                int(np.logical_and(parts[name], ~whole).sum()),
                int(parts[name].sum()),
            )
            for name in names
        },
        "part_union_whole_iou": _ratio(int(inside.sum()), int(union_whole.sum())),
        "whole_coverage": _ratio(int(inside.sum()), int(whole.sum())),
        "union_outside_whole_fraction": _ratio(
            int(np.logical_and(union, ~whole).sum()), int(union.sum())
        ),
        "cross_part_iou": _ratio(
            int(intersection.sum()),
            int(np.logical_or.reduce(list(parts.values())).sum()),
        ),
        "cross_part_overlap_pixels": int(intersection.sum()),
    }
    return result


def _overlay(
    image: np.ndarray,
    parts: dict[str, np.ndarray],
    whole: np.ndarray,
    colors: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    output = image.copy()
    for name, selected in parts.items():
        color = np.asarray(colors[name], dtype=np.float32)
        output[selected] = (
            0.40 * output[selected].astype(np.float32) + 0.60 * color
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, tuple(int(v) for v in color), 2)
    union = np.logical_or.reduce(list(parts.values()))
    missed = np.logical_and(whole, ~union)
    excess = np.logical_and(union, ~whole)
    output[missed] = (255, 0, 255)
    output[excess] = (0, 255, 255)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-config", type=Path, required=True)
    parser.add_argument("--whole-mask-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--views", nargs="+")
    parser.add_argument("--timestamps", nargs="+")
    parser.add_argument("--range-start")
    parser.add_argument("--range-end")
    parser.add_argument("--preview-timestamps", nargs="+")
    args = parser.parse_args()

    config = load_mask_pipeline_config(args.mask_config)
    views = list(args.views or config.views)
    tracks_root = Path(config.work_root) / "tracks"
    available = sorted(
        {
            path.parent.name
            for part in config.parts
            for path in (tracks_root / part.name).glob("*/*.png")
        },
        key=int,
    )
    timestamps = list(args.timestamps or available)
    if args.range_start is not None:
        timestamps = [value for value in timestamps if int(value) >= int(args.range_start)]
    if args.range_end is not None:
        timestamps = [value for value in timestamps if int(value) <= int(args.range_end)]
    if not timestamps:
        raise RuntimeError("no tracked timestamps selected")

    colors = {part.name: tuple(part.color[::-1]) for part in config.parts}
    report: dict = {
        "mask_config": str(args.mask_config.resolve()),
        "whole_mask_root": str(args.whole_mask_root.resolve()),
        "parts": list(config.part_names),
        "frames": {},
        "missing": [],
    }
    rows: list[dict] = []
    preview_set = set(args.preview_timestamps or [])
    previews: dict[str, list[np.ndarray]] = {}

    for timestamp in timestamps:
        frame_report = {}
        for view in views:
            paths = {
                part.name: tracks_root / part.name / timestamp / f"{view}.png"
                for part in config.parts
            }
            whole_path = args.whole_mask_root / timestamp / f"{view}.png"
            missing = [str(path) for path in [*paths.values(), whole_path] if not path.exists()]
            if missing:
                report["missing"].append({"timestamp": timestamp, "view": view, "paths": missing})
                continue
            parts = {name: load_label_mask(path) > 0 for name, path in paths.items()}
            whole = load_label_mask(whole_path) > 0
            values = _metrics(parts, whole)
            frame_report[view] = values
            rows.append(values)

            if timestamp in preview_set:
                source = cv2.imread(str(frame_path(config.frames_dir, view, timestamp)))
                if source is None:
                    raise RuntimeError(f"failed to read source image for {timestamp}/{view}")
                preview = _overlay(source, parts, whole, colors)
                cv2.putText(
                    preview,
                    f"{timestamp} {view} union IoU {values['part_union_whole_iou']:.3f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                previews.setdefault(timestamp, []).append(
                    cv2.resize(preview, (480, 270), interpolation=cv2.INTER_AREA)
                )
        report["frames"][timestamp] = frame_report

    if not rows:
        raise RuntimeError("no complete part/whole mask tuples found")
    keys = [
        "part_union_whole_iou",
        "whole_coverage",
        "union_outside_whole_fraction",
        "cross_part_iou",
        "cross_part_overlap_pixels",
    ]
    report["summary"] = {
        "evaluated_view_frames": len(rows),
        "means": {key: float(np.mean([row[key] for row in rows])) for key in keys},
        "minima": {key: float(np.min([row[key] for row in rows])) for key in keys},
        "maxima": {key: float(np.max([row[key] for row in rows])) for key in keys},
        "mean_part_outside_whole_fraction": {
            name: float(np.mean([row["part_outside_whole_fraction"][name] for row in rows]))
            for name in config.part_names
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    for timestamp, images in previews.items():
        columns = min(4, len(images))
        row_count = math.ceil(len(images) / columns)
        images.extend([np.zeros_like(images[0]) for _ in range(row_count * columns - len(images))])
        sheet = np.vstack([
            np.hstack(images[offset : offset + columns])
            for offset in range(0, len(images), columns)
        ])
        cv2.imwrite(str(args.output_root / f"{timestamp}_all_views.jpg"), sheet)
    write_json(args.output_root / "metrics.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"review -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
