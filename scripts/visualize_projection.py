"""Validate ICP poses by projecting point clouds onto original frames.

Uses the same recon backend (vggt | da3) as seg_backproject_parts for cameras.
"""
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
from common.recon_loader import load_view_bundle, output_paths, resolve_backend

VIEW_NAMES = ["2-1", "2-2", "2-3", "2-4", "2-5", "2-6"]
COLORS = {"lid": (0, 0, 255), "body": (255, 0, 0), "inner_pot": (0, 255, 0)}


def load_pipeline(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_ply(path):
    pts = []
    with open(path, encoding="ascii") as f:
        hdr = False
        for line in f:
            if line.strip() == "end_header":
                hdr = True
                continue
            if not hdr:
                continue
            p = line.split()
            if len(p) >= 3:
                pts.append([float(p[0]), float(p[1]), float(p[2])])
    return np.asarray(pts, dtype=np.float64)


def draw(img, uv, z, color, depth_map=None, depth_tol=0.05, radius=2):
    H, W = img.shape[:2]
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, zz = u[ok], v[ok], z[ok]
    if depth_map is not None and len(u):
        dm = depth_map[v, u]
        vis = ~np.isfinite(dm) | (zz <= dm + depth_tol)
        u, v = u[vis], v[vis]
    out = img.copy()
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            out[np.clip(v + dv, 0, H - 1), np.clip(u + du, 0, W - 1)] = color
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="000019")
    ap.add_argument("--src", default="000029")
    ap.add_argument("--icp-json", default=None)
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline.json"))
    ap.add_argument("--recon-backend", choices=["vggt", "da3"], default=None)
    ap.add_argument("--depth-tol", type=float, default=0.05)
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    backend = resolve_backend(cfg, args.recon_backend)
    icp_json = args.icp_json or os.path.join(
        ROOT, "outputs", "icp", backend, f"pose_{args.src}_to_{args.ref}.json")
    with open(icp_json, encoding="utf-8") as f:
        pose = json.load(f)

    reg_method = pose.get("reg_method", "icp")
    vis_backend = backend if reg_method == "icp" else f"{backend}_{reg_method}"

    src_bundle = load_view_bundle(cfg, args.src, VIEW_NAMES, backend=backend)
    ref_bundle = load_view_bundle(cfg, args.ref, VIEW_NAMES, backend=backend)
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
        rp = os.path.join(output_paths(ROOT, backend, args.ref)["parts_ply"], f"{pname}.ply")
        sp = os.path.join(output_paths(ROOT, backend, args.src)["parts_ply"], f"{pname}.ply")
        if os.path.exists(rp):
            ref_clouds[pname] = read_ply(rp)
        if os.path.exists(sp):
            src_clouds[pname] = read_ply(sp)
        T = np.array(pinfo["transform_src_to_ref"], dtype=np.float64)
        T_map[pname] = T
        inv_T[pname] = np.linalg.inv(T)

    out_dir = os.path.join(ROOT, "outputs", "proj_vis", vis_backend, f"{args.ref}_to_{args.src}")
    os.makedirs(out_dir, exist_ok=True)
    panels = []

    for v in range(img_s.shape[0]):
        backward = cv2.cvtColor(img_s[v], cv2.COLOR_RGB2BGR)
        forward = cv2.cvtColor(img_r[v], cv2.COLOR_RGB2BGR)

        for pname in pose["parts"]:
            color = COLORS.get(pname, (255, 255, 255))
            if pname in ref_clouds:
                pts_b = apply_transform(ref_clouds[pname], inv_T[pname])
                uv, z = project_points(pts_b, K_s[v], E_s[v])
                backward = draw(backward, uv, z, color, depth_map=depth_s[v], depth_tol=args.depth_tol)
            if pname in src_clouds:
                pts_f = apply_transform(src_clouds[pname], T_map[pname])
                uv, z = project_points(pts_f, K_r[v], E_r[v])
                forward = draw(forward, uv, z, color, depth_map=depth_r[v], depth_tol=args.depth_tol)

        cv2.putText(backward, f"[{vis_backend}] backward ref@inv(T) on {args.src} ({VIEW_NAMES[v]})",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(forward, f"[{vis_backend}] forward src@T on {args.ref} ({VIEW_NAMES[v]})",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        pair = np.hstack([backward, np.full((backward.shape[0], 4, 3), 255, np.uint8), forward])
        cv2.imwrite(os.path.join(out_dir, f"view{v}_{VIEW_NAMES[v]}.jpg"), pair,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        panels.append(pair)

    montage = np.vstack([cv2.resize(p, (p.shape[1] // 2, p.shape[0] // 2)) for p in panels])
    cv2.imwrite(os.path.join(out_dir, "montage.jpg"), montage, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
