#!/usr/bin/env python
"""Project raw part point clouds (no ICP) onto camera views for sanity check."""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.geom import project_points
from common.mask_io import VIEW_NAMES
from common.normalized_recon import (
    load_pipeline,
    load_view_bundle,
    parts_ply_dir,
    proj_vis_dir,
    read_ply,
    resolve_backend,
    sample_timestamps,
)

COLORS = {"lid": (0, 0, 255), "body": (255, 0, 0), "inner_pot": (0, 255, 0)}


def draw(img, uv, z, color, depth_map=None, depth_tol=0.05, radius=2):
    h, w = img.shape[:2]
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    ok = (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, zz = u[ok], v[ok], z[ok]
    if depth_map is not None and len(u):
        dm = depth_map[v, u]
        vis = ~np.isfinite(dm) | (zz <= dm + depth_tol)
        u, v = u[vis], v[vis]
    out = img.copy()
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            out[np.clip(v + dv, 0, h - 1), np.clip(u + du, 0, w - 1)] = color
    return out


def visualize_timestamp(cfg, backend: str, ts: str, depth_tol: float) -> None:
    ply_root = parts_ply_dir(cfg, backend)
    bundle = load_view_bundle(cfg, ts, backend=backend)
    images, depth, K, E = bundle["images"], bundle["depth"], bundle["intrinsics"], bundle["extrinsics"]
    parts = cfg.get("parts", ["lid", "body", "inner_pot"])

    clouds = {}
    for part in parts:
        path = os.path.join(ply_root, ts, f"{part}.ply")
        if os.path.exists(path):
            clouds[part], _ = read_ply(path)

    out_dir = os.path.join(proj_vis_dir(cfg, backend), "raw", ts)
    os.makedirs(out_dir, exist_ok=True)
    panels = []
    for v in range(images.shape[0]):
        base = cv2.cvtColor(images[v], cv2.COLOR_RGB2BGR)
        overlay = base.copy()
        for part, pts in clouds.items():
            if len(pts) == 0:
                continue
            uv, z = project_points(pts, K[v], E[v])
            overlay = draw(overlay, uv, z, COLORS.get(part, (255, 255, 255)),
                           depth_map=depth[v], depth_tol=depth_tol)
        pair = np.hstack([base, np.full((base.shape[0], 4, 3), 255, np.uint8), overlay])
        cv2.putText(pair, f"raw {ts} {VIEW_NAMES[v]}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"view{v}_{VIEW_NAMES[v]}.jpg"), pair,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        panels.append(pair)

    montage = np.vstack([cv2.resize(p, (p.shape[1] // 2, p.shape[0] // 2)) for p in panels])
    cv2.imwrite(os.path.join(out_dir, "montage.jpg"), montage, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline_normalized.json"))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--timestamp", default=None)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--depth-tol", type=float, default=0.05)
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    backend = resolve_backend(cfg, args.backend)
    if args.sample:
        timestamps = sample_timestamps()
    elif args.timestamp:
        timestamps = [args.timestamp]
    else:
        raise SystemExit("provide --timestamp or --sample")

    for ts in timestamps:
        visualize_timestamp(cfg, backend, ts, args.depth_tol)


if __name__ == "__main__":
    main()
