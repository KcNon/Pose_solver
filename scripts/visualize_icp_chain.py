#!/usr/bin/env python
"""Projection visualization for normalized chain ICP poses."""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.geom import project_points
from common.icp import apply_transform
from common.mask_io import VIEW_NAMES
from common.normalized_recon import (
    icp_out_dir,
    load_pipeline,
    load_view_bundle,
    parts_ply_dir,
    proj_vis_dir,
    read_ply,
    resolve_backend,
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


def visualize_pair(cfg, backend: str, ref: str, src: str, depth_tol: float) -> None:
    icp_json = os.path.join(icp_out_dir(cfg, backend), f"pose_{src}_to_{ref}.json")
    with open(icp_json, encoding="utf-8") as f:
        pose = json.load(f)

    ply_root = parts_ply_dir(cfg, backend)
    src_bundle = load_view_bundle(cfg, src, backend=backend)
    ref_bundle = load_view_bundle(cfg, ref, backend=backend)
    img_s, depth_s, K_s, E_s = (
        src_bundle["images"], src_bundle["depth"],
        src_bundle["intrinsics"], src_bundle["extrinsics"],
    )
    img_r, depth_r, K_r, E_r = (
        ref_bundle["images"], ref_bundle["depth"],
        ref_bundle["intrinsics"], ref_bundle["extrinsics"],
    )

    ref_clouds, src_clouds, T_map, inv_T = {}, {}, {}, {}
    for pname, pinfo in pose["parts"].items():
        rp = os.path.join(ply_root, ref, f"{pname}.ply")
        sp = os.path.join(ply_root, src, f"{pname}.ply")
        if os.path.exists(rp):
            ref_clouds[pname], _ = read_ply(rp)
        if os.path.exists(sp):
            src_clouds[pname], _ = read_ply(sp)
        T = np.array(pinfo["transform_src_to_ref"], dtype=np.float64)
        T_map[pname] = T
        inv_T[pname] = np.linalg.inv(T)

    out_dir = os.path.join(proj_vis_dir(cfg, backend), f"{ref}_to_{src}")
    os.makedirs(out_dir, exist_ok=True)
    panels = []

    for v in range(img_s.shape[0]):
        backward = cv2.cvtColor(img_s[v], cv2.COLOR_RGB2BGR)
        forward = cv2.cvtColor(img_r[v], cv2.COLOR_RGB2BGR)
        for pname in pose["parts"]:
            color = COLORS.get(pname, (255, 255, 255))
            if pname in ref_clouds:
                uv, z = project_points(apply_transform(ref_clouds[pname], inv_T[pname]), K_s[v], E_s[v])
                backward = draw(backward, uv, z, color, depth_map=depth_s[v], depth_tol=depth_tol)
            if pname in src_clouds:
                uv, z = project_points(apply_transform(src_clouds[pname], T_map[pname]), K_r[v], E_r[v])
                forward = draw(forward, uv, z, color, depth_map=depth_r[v], depth_tol=depth_tol)

        cv2.putText(backward, f"backward ref@inv(T) on {src} ({VIEW_NAMES[v]})",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(forward, f"forward src@T on {ref} ({VIEW_NAMES[v]})",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        pair = np.hstack([backward, np.full((backward.shape[0], 4, 3), 255, np.uint8), forward])
        cv2.imwrite(os.path.join(out_dir, f"view{v}_{VIEW_NAMES[v]}.jpg"), pair,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        panels.append(pair)

    montage = np.vstack([cv2.resize(p, (p.shape[1] // 2, p.shape[0] // 2)) for p in panels])
    cv2.imwrite(os.path.join(out_dir, "montage.jpg"), montage, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None)
    ap.add_argument("--src", default=None)
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline_normalized.json"))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--depth-tol", type=float, default=0.05)
    ap.add_argument("--from-summary", action="store_true")
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    backend = resolve_backend(cfg, args.backend)

    if args.from_summary:
        summary_path = os.path.join(icp_out_dir(cfg, backend), "chain_summary.json")
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        for seg in summary["segments"]:
            for pair_key in seg["pairs"]:
                src, ref = pair_key.split("_to_")
                visualize_pair(cfg, backend, ref, src, args.depth_tol)
    else:
        if not args.ref or not args.src:
            raise SystemExit("provide --ref/--src or use --from-summary")
        visualize_pair(cfg, backend, args.ref, args.src, args.depth_tol)


if __name__ == "__main__":
    main()
