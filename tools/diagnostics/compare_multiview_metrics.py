#!/usr/bin/env python
"""Compare per-part multi-view pose metrics against a frozen baseline."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json


def compact(summary: dict, part: str) -> dict:
    values = summary.get(part, {}).get("all_views", {})
    return {
        "visible_observations": int(values.get("visible_observations", 0)),
        "mean_iou": values.get("mean_iou"),
        "mean_contour_chamfer_px": values.get("mean_contour_chamfer_px"),
    }


def compact_per_frame(report: dict, part: str) -> dict[str, dict]:
    result = {}
    for frame, views in sorted(report.get("frames", {}).items()):
        rows = [
            values[part]
            for values in views.values()
            if part in values
        ]
        ious = [
            float(row["silhouette_iou"])
            for row in rows
            if row.get("silhouette_iou") is not None
        ]
        chamfers = [
            float(row["contour_chamfer_px"])
            for row in rows
            if row.get("contour_chamfer_px") is not None
        ]
        result[frame] = {
            "visible_observations": len(rows),
            "mean_iou": float(np.mean(ious)) if ious else None,
            "mean_contour_chamfer_px": (
                float(np.mean(chamfers)) if chamfers else None
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    candidate = load_json(args.candidate)
    baseline = load_json(args.baseline)
    parts = sorted(set(candidate.get("summary", {})) | set(
        baseline.get("summary", {})
    ))
    reports = {}
    iou_deltas = []
    for part in parts:
        new = compact(candidate.get("summary", {}), part)
        old = compact(baseline.get("summary", {}), part)
        old_iou = old["mean_iou"]
        new_iou = new["mean_iou"]
        delta = (
            None if old_iou is None or new_iou is None
            else float(new_iou) - float(old_iou)
        )
        relative = (
            None if delta is None or abs(float(old_iou)) < 1e-12
            else delta / float(old_iou)
        )
        old_chamfer = old["mean_contour_chamfer_px"]
        new_chamfer = new["mean_contour_chamfer_px"]
        chamfer_delta = (
            None if old_chamfer is None or new_chamfer is None
            else float(new_chamfer) - float(old_chamfer)
        )
        if delta is not None:
            iou_deltas.append(delta)
        candidate_frames = compact_per_frame(candidate, part)
        baseline_frames = compact_per_frame(baseline, part)
        per_frame = {}
        for frame in sorted(set(candidate_frames) | set(baseline_frames)):
            candidate_frame = candidate_frames.get(frame, {})
            baseline_frame = baseline_frames.get(frame, {})
            candidate_iou = candidate_frame.get("mean_iou")
            baseline_iou = baseline_frame.get("mean_iou")
            per_frame[frame] = {
                "baseline": baseline_frame,
                "candidate": candidate_frame,
                "mean_iou_delta": (
                    None
                    if candidate_iou is None or baseline_iou is None
                    else float(candidate_iou) - float(baseline_iou)
                ),
            }
        reports[part] = {
            "baseline": old,
            "candidate": new,
            "mean_iou_delta": delta,
            "mean_iou_relative_change": relative,
            "mean_contour_chamfer_delta_px": chamfer_delta,
            "iou_improved": None if delta is None else delta > 0.0,
            "per_frame": per_frame,
        }
    report = {
        "schema_version": 1,
        "candidate": str(args.candidate.resolve()),
        "baseline": str(args.baseline.resolve()),
        "parts": reports,
        "summary": {
            "compared_parts": len(reports),
            "improved_parts": sum(
                value["iou_improved"] is True for value in reports.values()
            ),
            "mean_part_iou_delta": (
                float(np.mean(iou_deltas)) if iou_deltas else None
            ),
        },
    }
    write_json(args.output, report)
    print(f"metric comparison -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
