#!/usr/bin/env python
"""Backproject normalized palette masks + depth recon into per-part PLY clouds."""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.geom import backproject_view
from common.icp import write_ply
from common.normalized_recon import (
    load_palette_masks,
    load_pipeline,
    load_recon,
    load_recon_colors,
    masks_dir as cfg_masks_dir,
    parts_ply_dir,
    resolve_backend,
    sample_timestamps,
)


def fuse_part_cloud(depth, img, K, E, conf, part_masks, conf_thr, stride, max_pts):
    all_pts, all_cols = [], []
    for v in range(depth.shape[0]):
        m = part_masks[v].astype(bool)
        m &= np.isfinite(depth[v]) & (depth[v] > 1e-3)
        m &= conf[v] > conf_thr
        sub = np.zeros_like(m)
        sub[::stride, ::stride] = True
        m &= sub
        if m.sum() == 0:
            continue
        pts, cols = backproject_view(depth[v], K[v], E[v], mask=m, color=img[v])
        all_pts.append(pts)
        all_cols.append(cols)
    if not all_pts:
        return np.empty((0, 3), np.float32), None
    pts = np.concatenate(all_pts, 0)
    cols = np.concatenate(all_cols, 0)
    if len(pts) > max_pts:
        idx = np.random.default_rng(0).choice(len(pts), max_pts, replace=False)
        pts, cols = pts[idx], cols[idx]
    return pts, cols


def resize_masks_to_depth(masks_by_view: list[np.ndarray], depth_hw: tuple[int, int]) -> list[np.ndarray]:
    h, w = depth_hw
    out = []
    for m in masks_by_view:
        small = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        out.append(small.astype(bool))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline_normalized.json"))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--timestamps", nargs="+", default=None)
    ap.add_argument("--sample", action="store_true", help="use ICP sample frames only")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-pts", type=int, default=80000)
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    backend = resolve_backend(cfg, args.backend)
    parts = cfg.get("parts", ["lid", "body", "inner_pot"])
    mdir = cfg_masks_dir(cfg)
    ply_root = parts_ply_dir(cfg, backend)

    if args.sample:
        timestamps = sample_timestamps()
    elif args.timestamps:
        timestamps = args.timestamps
    else:
        raise SystemExit("provide --timestamps or --sample")

    summary = {"backend": backend, "timestamps": {}}
    for ts in timestamps:
        print(f"\n===== {ts} =====")
        recon = load_recon(cfg, ts, backend=backend)
        depth = recon["depth"]
        conf = recon["conf"]
        K = recon["intrinsics"]
        E = recon["extrinsics"]
        conf_thr = float(np.median(conf))
        img = load_recon_colors(cfg, ts, recon)
        palette = load_palette_masks(mdir, ts, parts)

        odir = os.path.join(ply_root, ts)
        os.makedirs(odir, exist_ok=True)
        summary["timestamps"][ts] = {}

        for part in parts:
            view_masks = resize_masks_to_depth(palette[part], recon["depth_hw"])
            pts, cols = fuse_part_cloud(
                depth, img, K, E, conf, view_masks, conf_thr, args.stride, args.max_pts,
            )
            if len(pts) == 0:
                summary["timestamps"][ts][part] = {"n_pts": 0, "status": "empty"}
                print(f"  {part}: empty")
                continue
            write_ply(os.path.join(odir, f"{part}.ply"), pts, cols)
            summary["timestamps"][ts][part] = {"n_pts": len(pts), "status": "ok"}
            print(f"  {part}: {len(pts)} pts")

    summary_path = os.path.join(ply_root, "backproject_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
