#!/usr/bin/env python
"""Compare raw and gauge-corrected depth/cloud stability with one report."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, write_json
from common.mesh_align import read_ply_xyz


def cloud_stability(root: Path, part: str, start: int, end: int,
                    max_points: int = 4000) -> dict:
    rng = np.random.default_rng(17)
    clouds = []
    counts = []
    for frame in range(start, end + 1):
        path = root / f"{frame:06d}" / f"{part}.ply"
        if not path.exists():
            clouds.append(None)
            counts.append(0)
            continue
        points = read_ply_xyz(str(path))
        counts.append(int(len(points)))
        if len(points) > max_points:
            points = points[rng.choice(len(points), max_points, replace=False)]
        clouds.append(points)

    centroids = np.stack([cloud.mean(axis=0) for cloud in clouds if cloud is not None])
    reference = np.median(centroids, axis=0)
    spread = np.linalg.norm(centroids - reference, axis=1) * 1000.0
    centroid_steps = np.linalg.norm(np.diff(centroids, axis=0), axis=1) * 1000.0
    surface_steps = []
    for previous, current in zip(clouds[:-1], clouds[1:]):
        if previous is None or current is None:
            continue
        distances, _ = cKDTree(previous).query(current, k=1)
        surface_steps.append(float(np.median(distances) * 1000.0))

    def summary(values: np.ndarray | list[float]) -> dict:
        values = np.asarray(values, float)
        return {
            "median_mm": float(np.median(values)),
            "p95_mm": float(np.quantile(values, 0.95)),
            "max_mm": float(np.max(values)),
        }

    return {
        "root": str(root),
        "part": part,
        "n_frames": int(sum(cloud is not None for cloud in clouds)),
        "point_count_median": float(np.median([n for n in counts if n > 0])),
        # Centroid and unaligned NN metrics are deliberately marked as
        # visibility-sensitive; they are supporting diagnostics, not a depth
        # accuracy measurement.
        "centroid_spread_visibility_sensitive": summary(spread),
        "centroid_step_visibility_sensitive": summary(centroid_steps),
        "surface_nn_step_visibility_sensitive": summary(surface_steps),
    }


def reduction(before: float, after: float) -> dict:
    return {
        "raw": before,
        "gauge": after,
        "absolute_change": after - before,
        "relative_change_pct": 100.0 * (after - before) / before if before else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    base = ROOT / "experiments" / "three_part_multiview_111f" / "outputs"
    parser.add_argument("--raw-report", default=str(base / "diagnostics" / "depth_stability_support70_raw.json"))
    parser.add_argument("--gauge-report", default=str(base / "diagnostics" / "depth_stability_support70_gauge.json"))
    parser.add_argument("--raw-cloud-root", default=str(base / "parts_ply" / "da3_self_cond"))
    parser.add_argument("--gauge-cloud-root", default=str(base / "parts_ply" / "da3_self_cond_gauge"))
    parser.add_argument("--part", default="body")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=110)
    parser.add_argument("--out", default=str(base / "diagnostics" / "raw_vs_gauge.json"))
    args = parser.parse_args()

    raw = load_json(Path(args.raw_report))
    gauge = load_json(Path(args.gauge_report))
    raw_summary, gauge_summary = raw["summary"], gauge["summary"]
    report = {
        "raw_depth_report": args.raw_report,
        "gauge_depth_report": args.gauge_report,
        "part": args.part,
        "frames": [args.start, args.end],
        "depth_temporal": {
            "per_pixel_raw_std_mm": reduction(
                raw_summary["raw_std_mm"], gauge_summary["raw_std_mm"]),
            "per_frame_shift_std_mm": reduction(
                raw_summary["frame_shift_std_mm"], gauge_summary["frame_shift_std_mm"]),
            "shift_removed_residual_std_mm": reduction(
                raw_summary["residual_std_mm"], gauge_summary["residual_std_mm"]),
        },
        "per_view": {
            view: {
                "raw_std_mm": raw["views"][view]["per_pixel_temporal_std_mm"]["raw"],
                "gauge_std_mm": gauge["views"][view]["per_pixel_temporal_std_mm"]["raw"],
                "raw_shift_std_mm": raw["views"][view]["per_frame_shift_mm"]["std"],
                "gauge_shift_std_mm": gauge["views"][view]["per_frame_shift_mm"]["std"],
            }
            for view in raw["views"] if view in gauge["views"]
        },
        "raw_cloud": cloud_stability(
            Path(args.raw_cloud_root), args.part, args.start, args.end),
        "gauge_cloud": cloud_stability(
            Path(args.gauge_cloud_root), args.part, args.start, args.end),
        "interpretation": {
            "depth_metrics": "fixed-image-region temporal stability; primary gauge acceptance signal",
            "cloud_metrics": "affected by mask/visibility changes; diagnostic only",
        },
    }
    out = Path(args.out)
    write_json(out, report)

    for name, values in report["depth_temporal"].items():
        print(f"{name}: {values['raw']:.3f} -> {values['gauge']:.3f} mm "
              f"({values['relative_change_pct']:+.1f}%)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
