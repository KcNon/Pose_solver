#!/usr/bin/env python
"""Build per-view and cross-view-supported point clouds for pose solving."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.backproject_utils import load_palette_masks, load_recon_colors
from common.depth_gauge import apply_depth_gauge, load_depth_gauge
from common.icp import write_ply
from common.io_utils import load_json, write_json
from common.normalized_recon import load_recon
from common.quality_cloud import (
    assign_cross_view_support,
    cross_view_consistency,
    fuse_supported_clouds,
    prepare_view_clouds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pose_multiview_111_v4.json"))
    parser.add_argument("--output-root", default=str(
        ROOT / "experiments/three_part_multiview_111f/outputs_v5_pose_precision/parts_ply/da3_quality"))
    parser.add_argument("--depth-gauge", default=str(
        ROOT / "experiments/three_part_multiview_111f/outputs_v5_pose_precision/diagnostics/depth_gauge_cross_view.json"))
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--conf-quantile", type=float, default=0.5)
    parser.add_argument("--mask-erode", type=int, default=2)
    parser.add_argument("--depth-edge-mm", type=float, default=8.0)
    parser.add_argument("--consensus-mm", type=float, default=8.0)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--retain-unsupported", type=float, default=0.15)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=80000)
    parser.add_argument("--no-per-view", action="store_true")
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    gauge = load_depth_gauge(args.depth_gauge)
    start = int(cfg["frames"]["start"] if args.start is None else args.start)
    end = int(cfg["frames"]["end"] if args.end is None else args.end)
    summary = {
        "schema_version": 1,
        "config": str(Path(args.config).resolve()),
        "depth_gauge": str(Path(args.depth_gauge).resolve()),
        "parameters": {
            "conf_quantile": args.conf_quantile, "mask_erode": args.mask_erode,
            "depth_edge_mm": args.depth_edge_mm, "consensus_mm": args.consensus_mm,
            "min_support": args.min_support,
            "retain_unsupported_fraction": args.retain_unsupported,
            "stride": args.stride, "max_points": args.max_points,
        },
        "frames": {},
    }
    for frame in range(start, end + 1):
        timestamp = f"{frame:06d}"
        recon = load_recon(cfg, timestamp, backend=cfg["recon_backend"])
        depth = apply_depth_gauge(recon["depth"], gauge, timestamp)
        colors = load_recon_colors(recon, cfg, timestamp)
        masks = load_palette_masks(
            cfg["masks_dir"], timestamp, cfg["parts"], recon["depth_hw"],
            views=cfg.get("views"),
        )
        frame_dir = output / timestamp
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_report = {}
        for part_index, part in enumerate(cfg["parts"]):
            clouds = prepare_view_clouds(
                depth, colors, recon["intrinsics"], recon["extrinsics"], recon["conf"], masks[part],
                conf_quantile=args.conf_quantile, mask_erode=args.mask_erode,
                depth_edge_m=args.depth_edge_mm / 1000.0, stride=args.stride)
            assign_cross_view_support(clouds, args.consensus_mm / 1000.0)
            points, point_colors, stats = fuse_supported_clouds(
                clouds, min_support=args.min_support,
                retain_unsupported_fraction=args.retain_unsupported,
                max_points=args.max_points, seed=frame * 7 + part_index)
            if len(points):
                write_ply(str(frame_dir / f"{part}.ply"), points, point_colors)
            if not args.no_per_view:
                for view_name, cloud in zip(cfg["views"], clouds):
                    if len(cloud.points):
                        view_dir = frame_dir / "views" / part
                        view_dir.mkdir(parents=True, exist_ok=True)
                        write_ply(str(view_dir / f"{view_name}.ply"),
                                  cloud.points, cloud.colors)
            frame_report[part] = {
                **stats,
                "cross_view": cross_view_consistency(clouds, args.consensus_mm / 1000.0),
            }
        summary["frames"][timestamp] = frame_report
        print(f"[{frame - start + 1:03d}/{end - start + 1:03d}] {timestamp} "
              + " ".join(f"{p}={frame_report[p]['fused_points']}" for p in cfg["parts"]),
              flush=True)

    write_json(output / "quality_cloud_summary.json", summary)
    write_json(output / "complete.json", {"status": "complete", "frames": [start, end]})
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
