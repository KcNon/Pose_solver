#!/usr/bin/env python
"""Merge disjoint multiview metric shards and recompute their summary."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json


def merge_reports(
    paths: list[Path], *, trajectory_override: str | None = None
) -> dict:
    if not paths:
        raise ValueError("at least one metrics report is required")
    reports = [load_json(path) for path in paths]
    merged = dict(reports[0])
    merged["frames"] = {}
    for path, report in zip(paths, reports):
        for key in ("config", "resolution"):
            if report.get(key) != reports[0].get(key):
                raise ValueError(f"{path}: incompatible {key}")
        if (
            trajectory_override is None
            and report.get("trajectory") != reports[0].get("trajectory")
        ):
            raise ValueError(f"{path}: incompatible trajectory")
        overlap = set(merged["frames"]).intersection(report.get("frames", {}))
        if overlap:
            raise ValueError(f"{path}: overlapping frames {sorted(overlap)[:3]}")
        merged["frames"].update(report.get("frames", {}))
    merged["frames"] = {
        frame: merged["frames"][frame] for frame in sorted(merged["frames"])
    }
    parts = list(reports[0].get("summary", {}))
    views = list(next(iter(merged["frames"].values()), {}))
    merged["summary"] = {}
    for part in parts:
        per_view = {}
        all_iou, all_chamfer = [], []
        for view in views:
            rows = [
                frame[view][part]
                for frame in merged["frames"].values()
                if view in frame
                and int(frame[view][part].get("mask_pixels", 0)) > 0
                and int(frame[view][part].get("rendered_pixels", 0)) > 0
            ]
            ious = [float(row["silhouette_iou"]) for row in rows]
            chamfers = [float(row["contour_chamfer_px"]) for row in rows]
            per_view[view] = {
                "visible_frames": len(rows),
                "mean_iou": float(np.mean(ious)) if rows else None,
                "mean_contour_chamfer_px": (
                    float(np.mean(chamfers)) if rows else None
                ),
            }
            all_iou.extend(ious)
            all_chamfer.extend(chamfers)
        merged["summary"][part] = {
            "per_view": per_view,
            "all_views": {
                "visible_observations": len(all_iou),
                "mean_iou": float(np.mean(all_iou)) if all_iou else None,
                "mean_contour_chamfer_px": (
                    float(np.mean(all_chamfer)) if all_chamfer else None
                ),
            },
        }
    merged["shards"] = [str(path.resolve()) for path in paths]
    merged["shard_trajectories"] = [
        report.get("trajectory") for report in reports
    ]
    if trajectory_override is not None:
        merged["trajectory"] = str(Path(trajectory_override).resolve())
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--trajectory",
        default=None,
        help="audited trajectory path when unchanged frame shards were reused",
    )
    args = parser.parse_args()
    merged = merge_reports(
        [Path(value) for value in args.reports],
        trajectory_override=args.trajectory,
    )
    write_json(Path(args.output), merged)
    print(
        f"merged {len(args.reports)} shards / {len(merged['frames'])} frames "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
