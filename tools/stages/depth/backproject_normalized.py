#!/usr/bin/env python
"""Backproject normalized palette masks + depth recon into per-part PLY clouds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, ROOT)

from common.backproject_utils import fuse_part_cloud, load_palette_masks, load_recon_colors
from common.depth_gauge import apply_depth_gauge, load_depth_gauge
from common.cloud_io import write_ply
from common.normalized_recon import (
    all_timestamps,
    load_pipeline,
    load_recon,
    masks_dir,
    output_root,
    resolve_backend,
    sample_timestamps,
)
from common.mask_io import view_names


def parts_ply_root(cfg: dict, backend: str, tag: str | None) -> str:
    name = backend if not tag else f"{backend}_{tag}"
    artifact_root = cfg.get("point_cloud_output_root", output_root(cfg))
    return os.path.join(artifact_root, "parts_ply", name)


def backproject_timestamp(
    cfg: dict,
    backend: str,
    timestamp: str,
    parts: list[str],
    *,
    conf_mode: str,
    conf_quantile: float,
    stride: int,
    max_pts: int,
    tag: str | None,
    depth_gauge: dict | None = None,
) -> dict:
    recon = load_recon(cfg, timestamp, backend=backend)
    depth = recon["depth"]
    if depth_gauge is not None:
        depth = apply_depth_gauge(depth, depth_gauge, timestamp)
    conf = recon["conf"]
    K = recon["intrinsics"]
    E = recon["extrinsics"]
    views = view_names(cfg)
    if recon["n_views"] != len(views):
        raise ValueError(
            f"reconstruction has {recon['n_views']} views but config declares "
            f"{len(views)}: {views}"
        )
    global_thr = float(np.median(conf))
    img = load_recon_colors(recon, cfg, timestamp)
    view_masks = load_palette_masks(
        masks_dir(cfg), timestamp, parts, recon["depth_hw"], views=views
    )

    out_dir = os.path.join(parts_ply_root(cfg, backend, tag), timestamp)
    os.makedirs(out_dir, exist_ok=True)
    summary: dict = {"global_conf_thr": global_thr, "conf_mode": conf_mode, "parts": {}}

    for pi, part in enumerate(parts):
        pts, cols, stats = fuse_part_cloud(
            depth, img, K, E, conf, view_masks[part],
            conf_mode=conf_mode,
            conf_quantile=conf_quantile,
            global_conf_thr=global_thr,
            stride=stride,
            max_pts=max_pts,
            seed=pi + 1,
            views=views,
        )
        ply_path = os.path.join(out_dir, f"{part}.ply")
        if len(pts) == 0:
            summary["parts"][part] = {"n_pts": 0, "status": "empty", "per_view_px": stats}
            continue
        write_ply(ply_path, pts, cols)
        summary["parts"][part] = {"n_pts": int(len(pts)), "status": "ok", "per_view_px": stats}
        print(f"  {part}: {len(pts)} pts  per_view={stats}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--timestamp")
    g.add_argument("--timestamps", nargs="+")
    g.add_argument("--all", action="store_true")
    g.add_argument("--sample", action="store_true")
    ap.add_argument("--pipeline", default=os.path.join(
        ROOT, "configs", "pipeline_data_1_8view.json"
    ))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--conf-mode", choices=["global", "adaptive"], default="adaptive",
                    help="global=median(conf); adaptive=per-view per-part quantile in mask")
    ap.add_argument("--conf-quantile", type=float, default=0.25,
                    help="adaptive mode: keep conf >= this quantile within part mask")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-pts", type=int, default=80000)
    ap.add_argument("--tag", default=None,
                    help="output subdir parts_ply/{backend}_{tag}/ (e.g. adaptive)")
    ap.add_argument("--depth-gauge", default=None,
                    help="depth_gauge.json from calibrate_depth_gauge.py; removes "
                         "per-frame global depth drift before backprojection")
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    backend = resolve_backend(cfg, args.backend)
    parts = cfg.get("parts", ["lid", "body", "inner_pot"])

    if args.timestamp:
        timestamps = [args.timestamp]
    elif args.timestamps:
        timestamps = args.timestamps
    elif args.sample:
        timestamps = sample_timestamps()
    else:
        timestamps = all_timestamps(cfg)
        frame_range = cfg.get("frames")
        if isinstance(frame_range, dict):
            start, end = int(frame_range["start"]), int(frame_range["end"])
            timestamps = [
                timestamp for timestamp in timestamps
                if start <= int(timestamp) <= end
            ]

    depth_gauge_path = args.depth_gauge or cfg.get("depth_gauge_path")
    depth_gauge = load_depth_gauge(depth_gauge_path) if depth_gauge_path else None
    tag = args.tag if args.tag is not None else cfg.get("point_cloud_tag")

    root = parts_ply_root(cfg, backend, tag)
    os.makedirs(root, exist_ok=True)
    full_summary = {
        "backend": backend,
        "tag": tag,
        "conf_mode": args.conf_mode,
        "conf_quantile": args.conf_quantile,
        "depth_gauge": depth_gauge_path,
        "timestamps": {},
    }

    for i, ts in enumerate(timestamps):
        print(f"[{i + 1}/{len(timestamps)}] {ts}")
        full_summary["timestamps"][ts] = backproject_timestamp(
            cfg, backend, ts, parts,
            conf_mode=args.conf_mode,
            conf_quantile=args.conf_quantile,
            stride=args.stride,
            max_pts=args.max_pts,
            tag=tag,
            depth_gauge=depth_gauge,
        )

    summary_path = os.path.join(root, "backproject_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, ensure_ascii=False, indent=2)
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
